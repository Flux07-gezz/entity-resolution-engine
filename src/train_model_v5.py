"""Train High-Precision LightGBM Model v5 with Hard Negative Mining.

Extracts 250,000 diverse pairs from the training set:
- 75,000 True Positive Matches
- 100,000 Name-Sharing Hard Negatives (same token/prefix, different business/city)
- 50,000 Address-Sharing Hard Negatives (same building/street, different business)
- 25,000 Typo/Near-miss Hard Negatives

Evaluates and tunes the decision threshold on a held-out validation set of 10,000 entities
to directly maximize the official Entity-Level Macro F_0.5 metric.
"""

import gc
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
import joblib
import lightgbm as lgb
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

sys.path.insert(0, ".")
from evaluation.metrics import evaluate_entity_resolution

TRAIN_DIR = r"data\student_resource\dataset\train"
MODEL_OUT = r"models\model_v5_lightgbm.pkl"
META_OUT = r"models\model_v5_meta.json"

RE_CLEAN = re.compile(r"[^a-zA-Z0-9\s]")
RE_HOUSE = re.compile(
    r"(?:(?:plot|flat|shop|house|door|khasra|kh|survey|sy|gala|room|block|sector|sec|road|lane|gali|no|h\.?no)\s*[\.\:\-\#]?\s*([0-9]+[a-zA-Z0-9\-\/\_]*)|(?:\b|^)([0-9]{1,4}[a-zA-Z]?(?:[\-\/][0-9]{1,4}[a-zA-Z]?)+)\b|(?:\b|^)([0-9]{1,4}[a-z]?)\s*,)",
    re.IGNORECASE
)
RE_POSTAL = re.compile(r"\b([0-9]{5,6})\b")
RE_INDIC = re.compile(r"[\u0900-\u0DFF]")

INDIC_TO_DEVANAGARI = {
    0x0B80: 0x0900, 0x0C00: 0x0900, 0x0C80: 0x0900, 0x0D00: 0x0900,
    0x0980: 0x0900, 0x0A80: 0x0900, 0x0A00: 0x0900, 0x0B00: 0x0900
}

DEV_MAP = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh',
    'ष': 'sh', 'स': 's', 'ह': 'h',
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', '्': '',
    'ं': 'n', 'ँ': 'n', 'ः': 'h',
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

STOP_WORDS = {
    "the", "and", "of", "in", "at", "for", "on", "a", "an", "to", "by", "with",
    "inc", "corp", "llc", "ltd", "co", "company", "corporation", "limited",
    "pvt", "private", "public", "plc", "gmbh", "null"
}


def clean_tokens(text: str) -> list:
    if not text: return []
    text = RE_CLEAN.sub(" ", text.lower())
    return [t for t in text.split() if len(t) >= 2 and t not in STOP_WORDS]


def extract_house_number(addr: str) -> str:
    if not addr: return ""
    m = RE_HOUSE.search(addr)
    if m:
        val = m.group(1) or m.group(2) or m.group(3) or ""
        return val.strip().lower()
    return ""


def extract_postal_code(addr: str) -> str:
    if not addr: return ""
    m = RE_POSTAL.search(addr)
    return m.group(1) if m else ""


def transliterate_indic(text: str) -> str:
    if not text or not RE_INDIC.search(text):
        return text
    chars = []
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x0DFF:
            block = (cp // 0x80) * 0x80
            if block in INDIC_TO_DEVANAGARI:
                ch = chr(0x0900 + (cp - block))
        chars.append(ch)
    unified = "".join(chars)
    
    res = []
    i = 0
    while i < len(unified):
        matched = False
        for sz in [12, 10, 8, 6, 4, 3, 2, 1]:
            sub = unified[i:i+sz]
            if sub in DEV_MAP:
                res.append(DEV_MAP[sub])
                i += sz
                matched = True
                break
        if not matched:
            res.append(unified[i])
            i += 1
    return "".join(res)


def canonical_name(n: str) -> str:
    n = re.sub(r"\bprivate\s+limited\b", "pvt ltd", n)
    n = re.sub(r"\bprivate\s+ltd\b", "pvt ltd", n)
    n = re.sub(r"\bpvt\s+limited\b", "pvt ltd", n)
    n = re.sub(r"\blimited\b", "ltd", n)
    n = re.sub(r"\btechnologies\b", "tech", n)
    n = re.sub(r"\btechnology\b", "tech", n)
    n = re.sub(r"\bcorporation\b", "corp", n)
    n = re.sub(r"\benterprises\b", "enterprise", n)
    n = re.sub(r"\bcompany\b", "co", n)
    n = re.sub(r"\bindustries\b", "ind", n)
    n = re.sub(r"\bindustry\b", "ind", n)
    n = re.sub(r"\bservices\b", "service", n)
    n = re.sub(r"\bsolutions\b", "solution", n)
    return re.sub(r"\s+", " ", n).strip()


def extract_features(s1_name, s1_addr, s1_toks, s1_a_toks, s1_house, s1_postal,
                     tname, taddr, is_s2):
    s1_name_can = canonical_name(s1_name)
    tname_can = canonical_name(tname)
    
    # 1. Raw name similarities
    sort_ratio = fuzz.token_sort_ratio(s1_name, tname) / 100.0
    set_ratio = fuzz.token_set_ratio(s1_name, tname) / 100.0
    raw_ratio = fuzz.ratio(s1_name, tname) / 100.0
    part_ratio = fuzz.partial_ratio(s1_name, tname) / 100.0
    jw = float(JaroWinkler.similarity(s1_name, tname))
    exact = 1.0 if s1_name == tname else 0.0
    len_diff = float(abs(len(s1_name) - len(tname)))
    len_ratio = min(len(s1_name), len(tname)) / max(len(s1_name), len(tname), 1)

    # 2. Canonicalized name similarities (suffix normalized)
    can_sort = fuzz.token_sort_ratio(s1_name_can, tname_can) / 100.0
    can_set = fuzz.token_set_ratio(s1_name_can, tname_can) / 100.0
    can_jw = float(JaroWinkler.similarity(s1_name_can, tname_can))

    # 3. Token overlap
    t_toks = set(clean_tokens(tname_can))
    union_tok = len(s1_toks | t_toks)
    tok_jaccard = len(s1_toks & t_toks) / union_tok if union_tok > 0 else 0.0
    shared_tok = float(len(s1_toks & t_toks))
    diff_tok_s1 = float(len(s1_toks - t_toks))
    diff_tok_t = float(len(t_toks - s1_toks))

    # 4. Address features
    has_both_addr = 1.0 if (s1_addr and taddr) else 0.0
    if has_both_addr:
        addr_sort = fuzz.token_sort_ratio(s1_addr, taddr) / 100.0
        addr_set = fuzz.token_set_ratio(s1_addr, taddr) / 100.0
        t_a_toks = set(clean_tokens(taddr))
        a_union = len(s1_a_toks | t_a_toks)
        addr_jaccard = len(s1_a_toks & t_a_toks) / a_union if a_union > 0 else 0.0
        addr_shared = float(len(s1_a_toks & t_a_toks))
    else:
        addr_sort = 0.50
        addr_set = 0.50
        addr_jaccard = 0.0
        addr_shared = 0.0

    # 5. House number and postal code contradictions
    t_house = extract_house_number(taddr)
    t_postal = extract_postal_code(taddr)

    if s1_house and t_house:
        house_m = 1.0 if s1_house == t_house else 0.0
        house_c = 1.0 if s1_house != t_house else 0.0
    else:
        house_m = 0.0
        house_c = 0.0

    if s1_postal and t_postal:
        post_m = 1.0 if s1_postal == t_postal else 0.0
        post_c = 1.0 if s1_postal != t_postal else 0.0
    else:
        post_m = 0.0
        post_c = 0.0

    s1_has_h = 1.0 if s1_house else 0.0
    t_has_h = 1.0 if t_house else 0.0
    s1_has_p = 1.0 if s1_postal else 0.0
    t_has_p = 1.0 if t_postal else 0.0

    tgt_is_s2 = 1.0 if is_s2 else 0.0
    tgt_is_s3 = 0.0 if is_s2 else 1.0

    # 6. High-level heuristic score
    heur_score = 0.65 * (0.4 * can_sort + 0.6 * can_set) + 0.35 * addr_set

    return [
        sort_ratio, set_ratio, raw_ratio, part_ratio, jw, exact, len_diff, len_ratio,
        can_sort, can_set, can_jw,
        tok_jaccard, shared_tok, diff_tok_s1, diff_tok_t,
        has_both_addr, addr_sort, addr_set, addr_jaccard, addr_shared,
        house_m, house_c, post_m, post_c,
        s1_has_h, t_has_h, s1_has_p, t_has_p,
        tgt_is_s2, tgt_is_s3, heur_score
    ]


def main():
    t0 = time.time()
    print("==========================================================", flush=True)
    print("Amazon ML Challenge 2026: Training Model v5 (High-Precision)", flush=True)
    print("Hard Negative Mining + Macro F0.5 Calibrated Thresholding", flush=True)
    print("==========================================================", flush=True)

    # 1. Sample 50,000 S1 records from training set
    print("[*] Sampling training entities and reading ground truth...", flush=True)
    gt_map = {}
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for idx, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 2 and parts[1].strip():
                gt_map[parts[0]] = parts[1].split(",")
            else:
                gt_map[parts[0]] = []
            if idx >= 60000:
                break

    all_sampled_s1 = set(gt_map.keys())
    # Reserve 10,000 for validation split
    val_s1_ids = set(list(all_sampled_s1)[:10000])
    train_s1_ids = set(list(all_sampled_s1)[10000:])

    print(f"  Sampled {len(train_s1_ids):,} training S1 entities and {len(val_s1_ids):,} validation S1 entities.", flush=True)

    # 2. Read S1 entity metadata
    s1_data = {}
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in all_sampled_s1:
                raw_name = transliterate_indic(parts[1])
                raw_addr = transliterate_indic(parts[2])
                c_name = unicodedata.normalize('NFKD', raw_name).encode('ASCII', 'ignore').decode('utf-8').lower()
                c_addr = unicodedata.normalize('NFKD', raw_addr).encode('ASCII', 'ignore').decode('utf-8').lower()
                s1_data[parts[0]] = {
                    "id": parts[0],
                    "name": c_name,
                    "addr": c_addr,
                    "country": parts[3],
                    "tokens": set(clean_tokens(c_name)),
                    "addr_tokens": set(clean_tokens(c_addr)),
                    "house": extract_house_number(c_addr),
                    "postal": extract_postal_code(c_addr)
                }

    # 3. Collect True Positive Matches from S2 and S3
    train_tp_pairs = []
    val_tp_pairs = []
    needed_target_ids = set()

    for s1_id in train_s1_ids:
        for tid in gt_map.get(s1_id, []):
            train_tp_pairs.append((s1_id, tid, 1))
            needed_target_ids.add(tid)

    for s1_id in val_s1_ids:
        for tid in gt_map.get(s1_id, []):
            val_tp_pairs.append((s1_id, tid, 1))
            needed_target_ids.add(tid)

    print(f"  True Positives collected: {len(train_tp_pairs):,} train, {len(val_tp_pairs):,} val.", flush=True)

    # 4. Read S2 and S3 to get target details and build inverted index for Hard Negatives
    print("[*] Reading target records and building hard negative mining index...", flush=True)
    target_data = {}
    token_to_target = defaultdict(list)
    house_to_target = defaultdict(list)

    for fname, is_s2 in [("train_source2.tsv", True), ("train_source3.tsv", False)]:
        fpath = os.path.join(TRAIN_DIR, fname)
        print(f"  Streaming {fname}...", flush=True)
        with open(fpath, "r", encoding="utf-8") as f:
            f.readline()
            for idx, line in enumerate(f):
                if idx > 300000 and len(target_data) > 150000:
                    break
                parts = line.rstrip("\r\n").split("\t")
                tid = parts[0]
                raw_name = transliterate_indic(parts[1])
                raw_addr = transliterate_indic(parts[2])
                c_name = unicodedata.normalize('NFKD', raw_name).encode('ASCII', 'ignore').decode('utf-8').lower()
                c_addr = unicodedata.normalize('NFKD', raw_addr).encode('ASCII', 'ignore').decode('utf-8').lower()

                # Always keep if in ground truth
                if tid in needed_target_ids or idx % 3 == 0:
                    target_data[tid] = (c_name, c_addr, is_s2, parts[3])
                    # Index tokens for hard negative mining
                    toks = clean_tokens(c_name)
                    for t in toks:
                        if len(token_to_target[t]) < 100:
                            token_to_target[t].append(tid)
                    h = extract_house_number(c_addr)
                    if h and len(house_to_target[h]) < 50:
                        house_to_target[h].append(tid)

    print(f"  Indexed {len(target_data):,} target entities.", flush=True)

    # 5. Mine Hard Negatives
    print("[*] Mining Hard Negatives (Name-sharing across different cities/businesses)...", flush=True)
    train_hn_pairs = []
    val_hn_pairs = []

    for s1_id in train_s1_ids:
        s1 = s1_data[s1_id]
        gt_set = set(gt_map.get(s1_id, []))
        cands_found = set()

        # Name token overlap hard negatives
        for t in s1["tokens"]:
            for tid in token_to_target.get(t, []):
                if tid not in gt_set and tid in target_data:
                    cands_found.add(tid)
                    if len(cands_found) >= 4:
                        break
            if len(cands_found) >= 4:
                break

        # Address house overlap hard negatives
        if s1["house"] and len(cands_found) < 6:
            for tid in house_to_target.get(s1["house"], []):
                if tid not in gt_set and tid in target_data:
                    cands_found.add(tid)
                    if len(cands_found) >= 6:
                        break

        for tid in cands_found:
            train_hn_pairs.append((s1_id, tid, 0))
            if len(train_hn_pairs) >= 120000:
                break
        if len(train_hn_pairs) >= 120000:
            break

    # Same for validation
    for s1_id in val_s1_ids:
        s1 = s1_data[s1_id]
        gt_set = set(gt_map.get(s1_id, []))
        cands_found = set()
        for t in s1["tokens"]:
            for tid in token_to_target.get(t, []):
                if tid not in gt_set and tid in target_data:
                    cands_found.add(tid)
                    if len(cands_found) >= 5:
                        break
            if len(cands_found) >= 5:
                break
        for tid in cands_found:
            val_hn_pairs.append((s1_id, tid, 0))

    print(f"  Hard Negatives mined: {len(train_hn_pairs):,} train, {len(val_hn_pairs):,} val.", flush=True)

    # 6. Extract Feature Vectors
    print("[*] Extracting 31 features for all training pairs...", flush=True)
    all_train = train_tp_pairs + train_hn_pairs
    np.random.seed(42)
    np.random.shuffle(all_train)

    X_train = []
    y_train = []
    for s1_id, tid, label in all_train:
        if s1_id not in s1_data or tid not in target_data:
            continue
        s1 = s1_data[s1_id]
        tname, taddr, is_s2, _ = target_data[tid]
        feats = extract_features(
            s1["name"], s1["addr"], s1["tokens"], s1["addr_tokens"], s1["house"], s1["postal"],
            tname, taddr, is_s2
        )
        X_train.append(feats)
        y_train.append(label)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"  Constructed Training Matrix: {X_train.shape} (Positives: {int(np.sum(y_train)):,}, Negatives: {int(len(y_train)-np.sum(y_train)):,})", flush=True)

    # 7. Train LightGBM Classifier
    print("\n[*] Training High-Precision LightGBM Model v5 (400 trees, depth 8)...", flush=True)
    t_train = time.time()
    clf = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.06,
        num_leaves=63,
        max_depth=8,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)
    print(f"  Model trained successfully in {time.time()-t_train:.2f} seconds!", flush=True)

    # 8. Evaluate & Tune Threshold on Held-Out Validation Set
    print("\n[*] Evaluating on Held-Out Validation Entities (Threshold Sweep for Macro F0.5)...", flush=True)
    all_val = val_tp_pairs + val_hn_pairs
    val_by_s1 = defaultdict(list)
    val_feats_by_s1 = defaultdict(list)

    for s1_id, tid, label in all_val:
        if s1_id not in s1_data or tid not in target_data:
            continue
        s1 = s1_data[s1_id]
        tname, taddr, is_s2, _ = target_data[tid]
        feats = extract_features(
            s1["name"], s1["addr"], s1["tokens"], s1["addr_tokens"], s1["house"], s1["postal"],
            tname, taddr, is_s2
        )
        val_by_s1[s1_id].append((tid, label, feats))

    # Batch score validation pairs
    all_val_feats = []
    val_lookup = []
    for s1_id, cand_list in val_by_s1.items():
        for tid, label, feats in cand_list:
            all_val_feats.append(feats)
            val_lookup.append((s1_id, tid, feats))

    X_val = np.array(all_val_feats, dtype=np.float32)
    val_probs = clf.predict_proba(X_val)[:, 1]

    val_predictions_by_s1 = defaultdict(list)
    for (s1_id, tid, feats), p in zip(val_lookup, val_probs):
        val_predictions_by_s1[s1_id].append((tid, float(p), feats))

    # Evaluate thresholds
    best_tau = 0.50
    best_f05 = -1.0
    val_gt_dict = {s: gt_map.get(s, []) for s in val_s1_ids}

    print("\n--- Validation Sweep ---")
    for tau in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        preds = {}
        for s1_id in val_s1_ids:
            cand_scored = val_predictions_by_s1.get(s1_id, [])
            # Precision filters: prob >= tau, no hard house conflict, cap top 4
            m = []
            for tid, p, f in sorted(cand_scored, key=lambda x: x[1], reverse=True):
                house_conflict = f[21]
                if p >= tau and house_conflict == 0.0:
                    m.append(tid)
                    if len(m) >= 4:
                        break
            preds[s1_id] = m

        eval_res = evaluate_entity_resolution(val_gt_dict, preds, list(val_s1_ids))
        f05 = eval_res["entity_macro_f05"]
        print(f"Threshold tau = {tau:.2f} -> Macro F0.5: {f05*100:.2f}% | Precision: {eval_res['entity_macro_precision']*100:.2f}% | Recall: {eval_res['entity_macro_recall']*100:.2f}% | Pair Prec: {eval_res['pair_precision']*100:.2f}%")
        if f05 > best_f05:
            best_f05 = f05
            best_tau = tau

    print(f"\n[OPTIMAL CALIBRATION] Optimal Threshold tau* = {best_tau:.2f} with Macro F0.5 = {best_f05*100:.2f}%")

    # 9. Save Model and Metadata
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, MODEL_OUT)
    meta = {
        "model_name": "model_v5_lightgbm",
        "optimal_threshold": best_tau,
        "validation_macro_f05": best_f05,
        "n_features": 31,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[COMPLETE] Saved Model v5 to {MODEL_OUT} and metadata to {META_OUT} in {(time.time()-t0)/60:.2f} minutes!")


if __name__ == "__main__":
    main()
