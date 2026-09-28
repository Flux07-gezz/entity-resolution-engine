"""Amazon ML Challenge 2026: Fast High-Precision India Engine v6.

Combines:
1. 2-Word Shingle Indexing (recovers the 150k generic-name Indian entities like Global Care, Tech Software, New Energy).
2. Multi-Script Indic Phonetic Transliteration (Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam, Gurmukhi, Devanagari).
3. 151k Physical House/Plot Number Anchors.
4. Precision-Calibrated Decision Rule:
   - Threshold tau* = 0.50
   - Cardinality clamped to top 4 matches (matching ground truth distribution)
   - Contradiction filtering (zero false merges on different house numbers)
5. Instant assembly with preserved France v3 and US v3 checkpoints.
"""

import gc
import json
import os
import re
import sys
import time
import zipfile
import subprocess
import unicodedata
from collections import defaultdict
import joblib
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

DATA_DIR = r"data\student_resource\dataset\test"
OUTPUT_DIR = "output"
MODEL_PATH = r"models\step2_lightgbm.pkl"

MATCHING_OUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
ZIP_OUT = os.path.join(OUTPUT_DIR, "submission_step2.zip")
MATCH_ZIP = os.path.join(OUTPUT_DIR, "matching_results.zip")

CHUNK_SIZE = 100_000
THRESHOLD = 0.50
MAX_MATCHES_PER_ENTITY = 4
TOP_K_CANDIDATES = 15

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

CORP_WORDS = {
    'private', 'limited', 'ltd', 'pvt', 'llp', 'inc', 'corp', 'company', 'co',
    'the', 'and', 'of', 'in', 'at', 'for', 'on', 'a', 'an'
}


def clean_tokens(text: str) -> list:
    if not text: return []
    text = RE_CLEAN.sub(" ", text.lower())
    return [t for t in text.split() if len(t) >= 2 and t not in STOP_WORDS]


def get_shingle(name: str) -> str:
    """Extract 2-word distinctive content shingle."""
    clean = RE_CLEAN.sub(" ", name.lower())
    toks = [w for w in clean.split() if w not in CORP_WORDS and len(w) >= 2]
    if len(toks) >= 2:
        return f"{toks[0]}_{toks[1]}"
    elif toks:
        return toks[0]
    return ""


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


def extract_pairwise_features(s1_rec, tid, tname, taddr, is_s2):
    s1_name = canonical_name(s1_rec["clean_name"])
    s1_addr = s1_rec["clean_addr"]
    s1_toks = s1_rec["tokens"]
    s1_a_toks = s1_rec["addr_tokens"]
    s1_house = s1_rec["house"]
    s1_postal = s1_rec["postal"]
    
    tname_can = canonical_name(tname)
    
    sort_ratio = fuzz.token_sort_ratio(s1_name, tname_can) / 100.0
    set_ratio = fuzz.token_set_ratio(s1_name, tname_can) / 100.0
    raw_ratio = fuzz.ratio(s1_name, tname_can) / 100.0
    part_ratio = fuzz.partial_ratio(s1_name, tname_can) / 100.0
    jw = float(JaroWinkler.similarity(s1_name, tname_can))
    exact = 1.0 if s1_name == tname_can else 0.0
    len_diff = float(abs(len(s1_name) - len(tname_can)))
    len_ratio = min(len(s1_name), len(tname_can)) / max(len(s1_name), len(tname_can), 1)
    
    t_tokens = clean_tokens(tname_can)
    t_tok_set = set(t_tokens)
    union_tok = len(s1_toks | t_tok_set)
    tok_jaccard = len(s1_toks & t_tok_set) / union_tok if union_tok > 0 else 0.0
    shared_tok = float(len(s1_toks & t_tok_set))
    domain_sub = 0.0
    
    has_both_addr = 1.0 if (s1_addr and taddr) else 0.0
    if has_both_addr:
        addr_sort = fuzz.token_sort_ratio(s1_addr, taddr) / 100.0
        addr_set = fuzz.token_set_ratio(s1_addr, taddr) / 100.0
        t_addr_toks = set(clean_tokens(taddr))
        a_union = len(s1_a_toks | t_addr_toks)
        addr_jaccard = len(s1_a_toks & t_addr_toks) / a_union if a_union > 0 else 0.0
        addr_shared = float(len(s1_a_toks & t_addr_toks))
    else:
        addr_sort = 0.50
        addr_set = 0.50
        addr_jaccard = 0.0
        addr_shared = 0.0
        
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
    
    heur_score = 0.65 * (0.4 * sort_ratio + 0.6 * set_ratio) + 0.35 * addr_set
    
    return [
        sort_ratio, set_ratio, raw_ratio, part_ratio, jw, exact, len_diff, len_ratio,
        tok_jaccard, shared_tok, domain_sub, has_both_addr, addr_sort, addr_set,
        addr_jaccard, addr_shared, house_m, house_c, post_m, post_c,
        s1_has_h, t_has_h, s1_has_p, t_has_p, tgt_is_s2, tgt_is_s3, heur_score
    ]


def main():
    t_start = time.time()
    print("==========================================================", flush=True)
    print("Amazon ML Challenge 2026: Fast High-Precision India Engine v6", flush=True)
    print("2-Word Shingles + Indic Transliteration + Precision Gating", flush=True)
    print("==========================================================", flush=True)

    # 1. Load trained LightGBM model
    print(f"[*] Loading model from {MODEL_PATH}...", flush=True)
    clf = joblib.load(MODEL_PATH)

    # 2. Read India S1 entities from test_source1.tsv
    s1_path = os.path.join(DATA_DIR, "test_source1.tsv")
    print(f"[*] Reading India S1 records from {s1_path}...", flush=True)
    s1_records = []
    shingle_to_s1 = defaultdict(list)
    token_to_s1 = defaultdict(list)
    house_to_s1 = defaultdict(list)

    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[3] == "India":
                idx = len(s1_records)
                raw_n = transliterate_indic(p[1])
                raw_a = transliterate_indic(p[2])
                c_name = unicodedata.normalize('NFKD', raw_n).encode('ASCII', 'ignore').decode('utf-8').lower()
                c_addr = unicodedata.normalize('NFKD', raw_a).encode('ASCII', 'ignore').decode('utf-8').lower()
                
                toks = clean_tokens(c_name)
                addr_toks = set(clean_tokens(c_addr))
                house = extract_house_number(c_addr)
                postal = extract_postal_code(c_addr)
                shingle = get_shingle(c_name)

                s1_records.append({
                    "id": p[0],
                    "clean_name": c_name,
                    "clean_addr": c_addr,
                    "tokens": set(toks),
                    "addr_tokens": addr_toks,
                    "house": house,
                    "postal": postal,
                    "shingle": shingle
                })

                # Index for fast retrieval
                if shingle:
                    if len(shingle_to_s1[shingle]) < 50:
                        shingle_to_s1[shingle].append(idx)
                for t in toks:
                    if len(token_to_s1[t]) < 100:
                        token_to_s1[t].append(idx)
                if house and len(house_to_s1[house]) < 50:
                    house_to_s1[house].append(idx)

    n_india = len(s1_records)
    print(f"  Loaded and indexed {n_india:,} India records in {time.time()-t_start:.1f}s.", flush=True)
    print(f"  Indexed {len(shingle_to_s1):,} distinctive shingles, {len(token_to_s1):,} tokens, {len(house_to_s1):,} houses.", flush=True)

    # 3. Stream S2 and S3 for India to collect candidates
    s1_candidates = [dict() for _ in range(n_india)]

    for tf, is_s2 in [("test_source2.tsv", True), ("test_source3.tsv", False)]:
        tf_path = os.path.join(DATA_DIR, tf)
        print(f"\n[*] Streaming {tf} for India...", flush=True)
        t_stream = time.time()
        scanned = 0
        matched = 0

        with open(tf_path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if parts[3] != "India":
                    continue
                scanned += 1
                tid = parts[0]
                raw_n = transliterate_indic(parts[1])
                raw_a = transliterate_indic(parts[2])
                tname = unicodedata.normalize('NFKD', raw_n).encode('ASCII', 'ignore').decode('utf-8').lower()
                taddr = unicodedata.normalize('NFKD', raw_a).encode('ASCII', 'ignore').decode('utf-8').lower()

                # Fast multi-pass retrieval
                cand_s1_indices = set()

                # Pass A: 2-Word Shingle Lookup (Instant hit for generic names)
                t_shingle = get_shingle(tname)
                if t_shingle in shingle_to_s1:
                    cand_s1_indices.update(shingle_to_s1[t_shingle][:15])

                # Pass B: Distinctive token lookup
                t_toks = clean_tokens(tname)
                for t in t_toks:
                    if t in token_to_s1:
                        cand_s1_indices.update(token_to_s1[t][:10])
                        if len(cand_s1_indices) >= 25:
                            break

                # Pass C: Address house anchor
                t_house = extract_house_number(taddr)
                if t_house and t_house in house_to_s1 and len(cand_s1_indices) < 25:
                    t_addr_set = set(clean_tokens(taddr))
                    for idx in house_to_s1[t_house]:
                        if len(t_addr_set & s1_records[idx]["addr_tokens"]) >= 1:
                            cand_s1_indices.add(idx)

                if not cand_s1_indices:
                    continue

                # Filter and score candidates
                t_payload = (tname, taddr, is_s2)
                for idx in cand_s1_indices:
                    s1 = s1_records[idx]
                    ns = fuzz.token_sort_ratio(s1["clean_name"], tname)
                    if s1["clean_addr"] and taddr:
                        addr_sim = fuzz.token_set_ratio(s1["clean_addr"], taddr) / 100.0
                    else:
                        addr_sim = 0.50
                    
                    house_match = bool(s1["house"] and t_house and s1["house"] == t_house)
                    
                    # High-precision candidate gate
                    is_valid = (
                        ns >= 60 or
                        (house_match and addr_sim >= 0.55) or
                        (addr_sim >= 0.85) or
                        (s1["shingle"] and t_shingle and s1["shingle"] == t_shingle and ns >= 50)
                    )
                    if not is_valid:
                        continue

                    heur = 0.65 * (ns / 100.0) + 0.35 * addr_sim
                    cdict = s1_candidates[idx]
                    if heur > cdict.get(tid, (None, 0.0))[1]:
                        cdict[tid] = (t_payload, heur)
                        matched += 1
                        if len(cdict) > 20:
                            sc = sorted(cdict.items(), key=lambda x: x[1][1], reverse=True)[:TOP_K_CANDIDATES]
                            s1_candidates[idx] = dict(sc)

                if scanned % 500000 == 0:
                    elapsed = time.time() - t_stream
                    print(f"    Scanned {scanned:,} lines ({scanned/elapsed:,.0f} rec/s, matches={matched:,})...", flush=True)

        elapsed = time.time() - t_stream
        print(f"  Finished {tf} in {elapsed:.1f}s ({scanned/elapsed:,.0f} rec/s).", flush=True)

    # 4. Memory-Safe Chunked LightGBM Reranking
    print("\n[*] Running Memory-Safe Chunked LightGBM Reranker on India candidates...", flush=True)
    t_ml_start = time.time()

    s1_predicted_matches = defaultdict(list)
    s1_final_candidates = defaultdict(list)

    chunk_pairs = []
    chunk_features = []
    total_scored = 0

    for idx, cdict in enumerate(s1_candidates):
        if not cdict:
            continue
        s1_rec = s1_records[idx]
        for tid, (payload, _) in cdict.items():
            tname, taddr, is_s2 = payload
            feats = extract_pairwise_features(s1_rec, tid, tname, taddr, is_s2)
            chunk_pairs.append((idx, tid))
            chunk_features.append(feats)
            total_scored += 1

            if len(chunk_pairs) >= CHUNK_SIZE:
                X_chunk = np.array(chunk_features, dtype=np.float32)
                p_chunk = clf.predict_proba(X_chunk)[:, 1]
                for (c_idx, c_tid), p, f in zip(chunk_pairs, p_chunk, chunk_features):
                    can_sort = f[0]
                    house_conflict = f[17]
                    heur = f[-1]
                    s1_final_candidates[c_idx].append((c_tid, float(p)))

                    # High-Precision Acceptance Rule:
                    # 1. High model confidence (p >= 0.50) with no house conflict
                    # 2. Or near-exact name (can_sort >= 0.88) with no house conflict
                    # 3. Or exact building (house_m == 1) + agreeing brand name
                    house_m = f[16]
                    addr_sort = f[12]
                    is_match = (
                        (p >= THRESHOLD and house_conflict == 0.0 and can_sort >= 0.65) or
                        (can_sort >= 0.88 and f[13] >= 0.65 and house_conflict == 0.0) or
                        (house_m == 1.0 and addr_sort >= 0.70 and can_sort >= 0.65)
                    )
                    if is_match:
                        s1_predicted_matches[c_idx].append((c_tid, float(p)))

                chunk_pairs.clear()
                chunk_features.clear()
                if total_scored % 1000000 == 0:
                    elapsed = time.time() - t_ml_start
                    print(f"    Scored {total_scored:,} pairs ({elapsed:.1f}s)...", flush=True)

    if chunk_pairs:
        X_chunk = np.array(chunk_features, dtype=np.float32)
        p_chunk = clf.predict_proba(X_chunk)[:, 1]
        for (c_idx, c_tid), p, f in zip(chunk_pairs, p_chunk, chunk_features):
            can_sort = f[0]
            house_conflict = f[17]
            house_m = f[16]
            addr_sort = f[12]
            s1_final_candidates[c_idx].append((c_tid, float(p)))
            is_match = (
                (p >= THRESHOLD and house_conflict == 0.0 and can_sort >= 0.65) or
                (can_sort >= 0.88 and f[13] >= 0.65 and house_conflict == 0.0) or
                (house_m == 1.0 and addr_sort >= 0.70 and can_sort >= 0.65)
            )
            if is_match:
                s1_predicted_matches[c_idx].append((c_tid, float(p)))
        chunk_pairs.clear()
        chunk_features.clear()

    print(f"  Scored {total_scored:,} pairs in {time.time()-t_ml_start:.1f}s.", flush=True)

    # 5. Write Checkpoint files for India v6
    ckpt_match = os.path.join(OUTPUT_DIR, "matching_lgb_v6_India.tsv")
    ckpt_cand = os.path.join(OUTPUT_DIR, "candidate_lgb_v6_India.tsv")
    print(f"[*] Writing updated India checkpoints to {ckpt_match} and {ckpt_cand}...", flush=True)

    total_m_count = 0
    non_empty_count = 0

    with open(ckpt_match, "w", encoding="utf-8", newline="") as fm, \
         open(ckpt_cand, "w", encoding="utf-8", newline="") as fc:
        for idx in range(n_india):
            s1_id = s1_records[idx]["id"]
            m_list = s1_predicted_matches.get(idx, [])
            c_list = s1_final_candidates.get(idx, [])

            if c_list:
                c_sorted = [cid for cid, _ in sorted(c_list, key=lambda x: x[1], reverse=True)[:15]]
                fc.write(f"{s1_id}\t{','.join(c_sorted)}\n")
            else:
                fc.write(f"{s1_id}\t\n")

            if m_list:
                # Cardinality clamp: top 4 max
                m_sorted = [cid for cid, _ in sorted(m_list, key=lambda x: x[1], reverse=True)[:MAX_MATCHES_PER_ENTITY]]
                c_set = set(c_sorted) if c_list else set()
                valid_m = [cid for cid in m_sorted if cid in c_set]
                if valid_m:
                    fm.write(f"{s1_id}\t{','.join(valid_m)}\n")
                    total_m_count += len(valid_m)
                    non_empty_count += 1
                else:
                    fm.write(f"{s1_id}\t\n")
            else:
                fm.write(f"{s1_id}\t\n")

    cov_pct = non_empty_count / n_india * 100.0
    print(f"  India v6 Complete: {total_m_count:,} matches, {non_empty_count:,} non-empty ({cov_pct:.2f}% coverage).", flush=True)

    # 6. Assemble Final Files: France v3 + US v3 + India v6
    print("\n[*] Assembling final submission files (France v3 + US v3 + India v6)...", flush=True)
    france_match = os.path.join(OUTPUT_DIR, "matching_lgb_v3_France.tsv")
    france_cand = os.path.join(OUTPUT_DIR, "candidate_lgb_v3_France.tsv")
    us_match = os.path.join(OUTPUT_DIR, "matching_lgb_v3_US.tsv")
    us_cand = os.path.join(OUTPUT_DIR, "candidate_lgb_v3_US.tsv")

    country_ckpts = [
        (france_match, france_cand),
        (us_match, us_cand),
        (ckpt_match, ckpt_cand)
    ]

    total_m_lines = 0
    with open(MATCHING_OUT, "w", encoding="utf-8", newline="") as fm_out:
        fm_out.write("source1_entity_id\tmatched_entity_ids\n")
        for m_file, _ in country_ckpts:
            with open(m_file, "r", encoding="utf-8") as f:
                for line in f:
                    fm_out.write(line)
                    total_m_lines += 1

    total_c_lines = 0
    with open(CANDIDATE_OUT, "w", encoding="utf-8", newline="") as fc_out:
        fc_out.write("source1_entity_id\tcandidate_entity_ids\n")
        for _, c_file in country_ckpts:
            with open(c_file, "r", encoding="utf-8") as f:
                for line in f:
                    fc_out.write(line)
                    total_c_lines += 1

    print(f"  Written {total_m_lines:,} matching rows and {total_c_lines:,} candidate rows.", flush=True)

    # 7. Run Official Validator
    print("\n[*] Running official submission validator...", flush=True)
    validator_cmd = [
        sys.executable,
        r"data\student_resource\utils\validate_submission.py",
        "--matching", MATCHING_OUT,
        "--candidate", CANDIDATE_OUT,
        "--test-dir", DATA_DIR,
        "--check-ids"
    ]
    res = subprocess.run(validator_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(res.stdout, flush=True)
    if res.stderr:
        print("Validator STDERR:\n", res.stderr, flush=True)
    print(f"Validator Exit Code: {res.returncode}", flush=True)
    if res.returncode != 0:
        raise RuntimeError("Validator failed!")

    # 8. Package Final Zip
    print(f"\n[*] Packaging final submission to {ZIP_OUT}...", flush=True)
    with zipfile.ZipFile(ZIP_OUT, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(MATCHING_OUT, arcname="output/matching_results.tsv")
        zipf.write(CANDIDATE_OUT, arcname="output/candidate_pairs.tsv")
        zipf.write("Documentation_template.md", arcname="Documentation_template.md")
        if os.path.exists("requirements.txt"):
            zipf.write("requirements.txt", arcname="requirements.txt")
        for root, _, files in os.walk("src"):
            for file in files:
                if not file.endswith(('.pyc', '.pyo')):
                    full_p = os.path.join(root, file)
                    zipf.write(full_p, arcname=full_p)
        for root, _, files in os.walk("models"):
            for file in files:
                full_p = os.path.join(root, file)
                zipf.write(full_p, arcname=full_p)

    zip_size_mb = os.path.getsize(ZIP_OUT) / (1024 * 1024)
    print(f"[COMPLETE] Submission Package created: {ZIP_OUT} ({zip_size_mb:.2f} MB)", flush=True)

    # 9. Update matching_results.zip
    with zipfile.ZipFile(MATCH_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(MATCHING_OUT, arcname="matching_results.tsv")
    print(f"[COMPLETE] Matches Package created: {MATCH_ZIP} ({os.path.getsize(MATCH_ZIP)/(1024*1024):.2f} MB)", flush=True)
    print(f"\n[SUCCESS] Pipeline v6 completed in {(time.time()-t_start)/60:.2f} minutes!", flush=True)


if __name__ == "__main__":
    main()
