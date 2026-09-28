"""Rescore India Candidate Pairs with Canonical Suffix Alignment & Memory-Safe Batching.

This script leverages the already-computed 8.76M candidates in output/candidate_lgb_v3_India.tsv
and applies business legal abbreviation standardization (pvt ltd <-> private limited)
along with calibrated LightGBM scoring in RAM-bounded 100k chunks.

Execution time: ~5 to 7 minutes.
"""

import gc
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

# Paths
DATA_DIR = r"data\student_resource\dataset\test"
OUTPUT_DIR = "output"
MODEL_PATH = r"models\step2_lightgbm.pkl"
INDIA_CAND_CKPT = os.path.join(OUTPUT_DIR, "candidate_lgb_v3_India.tsv")

MATCHING_OUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
ZIP_OUT = os.path.join(OUTPUT_DIR, "submission_step2.zip")

CHUNK_SIZE = 100_000
THRESHOLD = 0.40

RE_CLEAN = re.compile(r"[^a-zA-Z0-9\s]")
RE_HOUSE = re.compile(
    r"(?:(?:plot|flat|shop|house|door|khasra|kh|survey|sy|gala|room|block|sector|sec|road|lane|gali|no|h\.?no)\s*[\.\:\-\#]?\s*([0-9]+[a-zA-Z0-9\-\/\_]*)|(?:\b|^)([0-9]{1,4}[a-zA-Z]?(?:[\-\/][0-9]{1,4}[a-zA-Z]?)+)\b|(?:\b|^)([0-9]{1,4}[a-z]?)\s*,)",
    re.IGNORECASE
)
RE_POSTAL = re.compile(r"\b([0-9]{5,6})\b")


def clean_tokens(text: str) -> list:
    if not text:
        return []
    text = RE_CLEAN.sub(" ", text.lower())
    return [t for t in text.split() if len(t) >= 2]


def extract_house_number(addr: str) -> str:
    if not addr:
        return ""
    m = RE_HOUSE.search(addr)
    if m:
        val = m.group(1) or m.group(2) or m.group(3) or ""
        return val.strip().lower()
    return ""


def extract_postal_code(addr: str) -> str:
    if not addr:
        return ""
    m = RE_POSTAL.search(addr)
    return m.group(1) if m else ""


INDIC_TO_DEVANAGARI = {
    0x0B80: 0x0900,  # Tamil
    0x0C00: 0x0900,  # Telugu
    0x0C80: 0x0900,  # Kannada
    0x0D00: 0x0900,  # Malayalam
    0x0980: 0x0900,  # Bengali
    0x0A80: 0x0900,  # Gujarati
    0x0A00: 0x0900,  # Gurmukhi
    0x0B00: 0x0900,  # Odia
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


def transliterate_hindi(text: str) -> str:
    if not text:
        return ""
    chars = []
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x0DFF:
            block = (cp // 0x80) * 0x80
            if block in INDIC_TO_DEVANAGARI:
                dev_cp = 0x0900 + (cp - block)
                ch = chr(dev_cp)
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
    """Normalize common legal and business suffix variations."""
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
    print("Amazon ML Challenge 2026: Fast India Rescorer v4", flush=True)
    print("Legal Suffix Alignment + Calibrated LightGBM Chunking", flush=True)
    print("==========================================================", flush=True)

    # 1. Load LightGBM model
    print(f"[*] Loading model from {MODEL_PATH}...", flush=True)
    clf = joblib.load(MODEL_PATH)

    # 2. Read India S1 entities from test_source1.tsv
    s1_path = os.path.join(DATA_DIR, "test_source1.tsv")
    print(f"[*] Reading India S1 records from {s1_path}...", flush=True)
    s1_records = []
    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[3] == "India":
                c_name = unicodedata.normalize('NFKD', p[1]).encode('ASCII', 'ignore').decode('utf-8').lower()
                c_addr = unicodedata.normalize('NFKD', p[2]).encode('ASCII', 'ignore').decode('utf-8').lower()
                s1_records.append({
                    "id": p[0],
                    "clean_name": c_name,
                    "clean_addr": c_addr,
                    "tokens": set(clean_tokens(c_name)),
                    "addr_tokens": set(clean_tokens(c_addr)),
                    "house": extract_house_number(c_addr),
                    "postal": extract_postal_code(c_addr),
                })

    n_india = len(s1_records)
    print(f"  Loaded {n_india:,} India S1 records in {time.time()-t_start:.1f}s.", flush=True)

    # 3. Read pre-computed candidates for India
    print(f"[*] Reading candidate pairs from {INDIA_CAND_CKPT}...", flush=True)
    s1_candidate_lists = []
    unique_target_ids = set()
    total_cand_pairs = 0

    with open(INDIA_CAND_CKPT, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[1].strip():
                c_list = parts[1].split(",")
                s1_candidate_lists.append(c_list)
                total_cand_pairs += len(c_list)
                for cid in c_list:
                    unique_target_ids.add(cid)
            else:
                s1_candidate_lists.append([])

    print(f"  Loaded {total_cand_pairs:,} candidate pairs across {len(s1_candidate_lists):,} entities.", flush=True)
    print(f"  Unique target entities to load: {len(unique_target_ids):,}.", flush=True)

    # 4. Stream S2 and S3 to load target entities
    t_load = time.time()
    target_data = {}
    
    print("[*] Streaming test_source2.tsv for candidate lookup...", flush=True)
    s2_path = os.path.join(DATA_DIR, "test_source2.tsv")
    with open(s2_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            tid = parts[0]
            if tid in unique_target_ids:
                raw_name = parts[1].strip()
                raw_addr = parts[2].strip()
                tname = transliterate_hindi(raw_name)
                taddr = transliterate_hindi(raw_addr)
                tname = unicodedata.normalize('NFKD', tname).encode('ASCII', 'ignore').decode('utf-8').lower()
                taddr = unicodedata.normalize('NFKD', taddr).encode('ASCII', 'ignore').decode('utf-8').lower()
                target_data[tid] = (tname, taddr, True)

    print(f"  Loaded {len(target_data):,} targets from S2 in {time.time()-t_load:.1f}s.", flush=True)

    print("[*] Streaming test_source3.tsv for candidate lookup...", flush=True)
    s3_path = os.path.join(DATA_DIR, "test_source3.tsv")
    with open(s3_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            tid = parts[0]
            if tid in unique_target_ids:
                raw_name = parts[1].strip()
                raw_addr = parts[2].strip()
                tname = transliterate_hindi(raw_name)
                taddr = transliterate_hindi(raw_addr)
                tname = unicodedata.normalize('NFKD', tname).encode('ASCII', 'ignore').decode('utf-8').lower()
                taddr = unicodedata.normalize('NFKD', taddr).encode('ASCII', 'ignore').decode('utf-8').lower()
                target_data[tid] = (tname, taddr, False)

    print(f"  Total targets loaded: {len(target_data):,} in {time.time()-t_load:.1f}s.", flush=True)

    # 5. Batch Reranking with LightGBM in 100k chunks
    print("\n[*] Starting Batch LightGBM Rescoring on India candidates...", flush=True)
    t_score_start = time.time()
    
    s1_predicted_matches = defaultdict(list)
    s1_final_candidates = defaultdict(list)

    chunk_pairs = []
    chunk_features = []
    total_scored = 0

    for idx, c_list in enumerate(s1_candidate_lists):
        if not c_list:
            continue
        s1_rec = s1_records[idx]
        for tid in c_list:
            tinfo = target_data.get(tid)
            if not tinfo:
                continue
            tname, taddr, is_s2 = tinfo
            feats = extract_pairwise_features(s1_rec, tid, tname, taddr, is_s2)
            chunk_pairs.append((idx, tid))
            chunk_features.append(feats)
            total_scored += 1

            if len(chunk_pairs) >= CHUNK_SIZE:
                X_chunk = np.array(chunk_features, dtype=np.float32)
                p_chunk = clf.predict_proba(X_chunk)[:, 1]
                for (c_idx, c_tid), p, f in zip(chunk_pairs, p_chunk, chunk_features):
                    heur = f[-1]
                    s1_final_candidates[c_idx].append((c_tid, float(p)))
                    # Calibrated decision: threshold 0.40 OR high-confidence heuristic
                    if (p >= THRESHOLD) or (heur >= 0.72 and p >= 0.25) or (f[0] >= 0.90 and f[13] >= 0.65):
                        s1_predicted_matches[c_idx].append((c_tid, float(p)))
                chunk_pairs.clear()
                chunk_features.clear()
                if total_scored % 1_000_000 == 0:
                    elapsed = time.time() - t_score_start
                    print(f"    Scored {total_scored:,} pairs ({elapsed:.1f}s, {total_scored/elapsed:,.0f} pairs/s)...", flush=True)

    if chunk_pairs:
        X_chunk = np.array(chunk_features, dtype=np.float32)
        p_chunk = clf.predict_proba(X_chunk)[:, 1]
        for (c_idx, c_tid), p, f in zip(chunk_pairs, p_chunk, chunk_features):
            heur = f[-1]
            s1_final_candidates[c_idx].append((c_tid, float(p)))
            if (p >= THRESHOLD) or (heur >= 0.72 and p >= 0.25) or (f[0] >= 0.90 and f[13] >= 0.65):
                s1_predicted_matches[c_idx].append((c_tid, float(p)))
        chunk_pairs.clear()
        chunk_features.clear()

    print(f"  Finished scoring {total_scored:,} pairs in {time.time()-t_score_start:.1f}s.", flush=True)

    # 6. Write updated India checkpoint files
    ckpt_match = os.path.join(OUTPUT_DIR, "matching_lgb_v4_India.tsv")
    ckpt_cand = os.path.join(OUTPUT_DIR, "candidate_lgb_v4_India.tsv")
    print(f"[*] Writing updated India checkpoints to {ckpt_match} and {ckpt_cand}...", flush=True)

    total_matches = 0
    non_empty = 0

    with open(ckpt_match, "w", encoding="utf-8", newline="") as fm, \
         open(ckpt_cand, "w", encoding="utf-8", newline="") as fc:
        for idx in range(n_india):
            s1_id = s1_records[idx]["id"]
            m_list = s1_predicted_matches.get(idx, [])
            c_list = s1_final_candidates.get(idx, [])

            # Candidate pairs
            if c_list:
                c_sorted = [cid for cid, _ in sorted(c_list, key=lambda x: x[1], reverse=True)[:15]]
                fc.write(f"{s1_id}\t{','.join(c_sorted)}\n")
            else:
                fc.write(f"{s1_id}\t\n")

            # Match pairs (strictly sorted by probability, top 10 max)
            if m_list:
                m_sorted = [cid for cid, _ in sorted(m_list, key=lambda x: x[1], reverse=True)[:10]]
                # Guarantee matched_ids is a subset of candidate_ids
                c_set = set(c_sorted) if c_list else set()
                valid_m = [cid for cid in m_sorted if cid in c_set]
                if valid_m:
                    fm.write(f"{s1_id}\t{','.join(valid_m)}\n")
                    total_matches += len(valid_m)
                    non_empty += 1
                else:
                    fm.write(f"{s1_id}\t\n")
            else:
                fm.write(f"{s1_id}\t\n")

    coverage_pct = non_empty / n_india * 100.0
    print(f"  India v4 Complete: {total_matches:,} matches, {non_empty:,} non-empty ({coverage_pct:.2f}% coverage).", flush=True)

    # 7. Assemble final submission
    print("\n[*] Assembling final submission files (France v3 + US v3 + India v4)...", flush=True)
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

    # 8. Run official validator
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

    # 9. Package final submission
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

    # 10. Update matching_results.zip
    match_zip = os.path.join(OUTPUT_DIR, "matching_results.zip")
    with zipfile.ZipFile(match_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(MATCHING_OUT, arcname="matching_results.tsv")
    print(f"[COMPLETE] Matches Package created: {match_zip} ({os.path.getsize(match_zip)/(1024*1024):.2f} MB)", flush=True)
    print(f"Total time elapsed: {(time.time()-t_start)/60:.2f} minutes.", flush=True)


if __name__ == "__main__":
    main()
