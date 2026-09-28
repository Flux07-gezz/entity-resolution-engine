"""Run 8: Graph-Pruned Precision Submission Engine.

Takes Run 7 (which has 81k rescued entities + Run 3 peak base) and applies:
1. Physical House Number Contradiction Pruning (eliminates 231k false merges across 158k entities).
2. Guarantees matched_ids is a strict subset of candidate_ids.
3. Official submission validator check.
4. Generates submission_step2.zip and matching_results.zip.
"""

import os
import re
import sys
import time
import zipfile
import subprocess

OUTPUT_DIR = "output"
DATA_DIR = r"data\student_resource\dataset\test"

MATCHING_OUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
ZIP_OUT = os.path.join(OUTPUT_DIR, "submission_step2.zip")
MATCH_ZIP = os.path.join(OUTPUT_DIR, "matching_results.zip")

RE_HOUSE = re.compile(
    r"(?:(?:plot|flat|shop|house|door|khasra|kh|survey|sy|gala|room|block|sector|sec|road|lane|gali|no|h\.?no)\s*[\.\:\-\#]?\s*([0-9]+[a-zA-Z0-9\-\/\_]*)|(?:\b|^)([0-9]{1,4}[a-zA-Z]?(?:[\-\/][0-9]{1,4}[a-zA-Z]?)+)\b|(?:\b|^)([0-9]{1,4}[a-z]?)\s*,)",
    re.IGNORECASE
)

def extract_house(addr: str) -> str:
    if not addr: return ""
    m = RE_HOUSE.search(addr)
    if m:
        val = m.group(1) or m.group(2) or m.group(3) or ""
        return val.strip().lower()
    return ""

t0 = time.time()
print("==========================================================", flush=True)
print("Amazon ML Challenge 2026: Graph-Pruned Precision Engine v8", flush=True)
print("158k Contradiction Prunings + 81k Rescued Entities + Run 3 Base", flush=True)
print("==========================================================", flush=True)

# 1. Load S1 house numbers
s1_houses = {}
with open(os.path.join(DATA_DIR, "test_source1.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        h = extract_house(p[2])
        if h:
            s1_houses[p[0]] = h

print(f"[*] Loaded {len(s1_houses):,} S1 house anchors in {time.time()-t0:.1f}s.", flush=True)

# 2. Load current Run 7 matching and candidate predictions
preds = {}
cands = {}
all_matched_tids = set()

with open(MATCHING_OUT, "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        eid = p[0]
        if len(p) > 1 and p[1].strip():
            m = p[1].split(",")
            preds[eid] = m
            all_matched_tids.update(m)
        else:
            preds[eid] = []

with open(CANDIDATE_OUT, "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        eid = p[0]
        if len(p) > 1 and p[1].strip():
            cands[eid] = p[1].split(",")
        else:
            cands[eid] = []

print(f"[*] Loaded {len(preds):,} S1 predictions with {len(all_matched_tids):,} unique matched target IDs.", flush=True)

# 3. Stream S2 and S3 for matched target houses
target_houses = {}
for src in ["test_source2.tsv", "test_source3.tsv"]:
    t_stream = time.time()
    print(f"[*] Streaming {src} for target house numbers...", flush=True)
    with open(os.path.join(DATA_DIR, src), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in all_matched_tids:
                h = extract_house(p[2])
                if h:
                    target_houses[p[0]] = h
    print(f"    Done {src} in {time.time()-t_stream:.1f}s.", flush=True)

print(f"[*] Captured house numbers for {len(target_houses):,} matched targets.", flush=True)

# 4. Prune Contradictions & Write Final Cleaned TSVs
print("[*] Applying Graph Contradiction Pruner to generate Run 8 files...", flush=True)
pruned_matches = 0
pruned_entities = 0
total_final_matches = 0
total_final_cands = 0
non_empty_entities = 0

with open(MATCHING_OUT, "w", encoding="utf-8", newline="") as fm, \
     open(CANDIDATE_OUT, "w", encoding="utf-8", newline="") as fc:
    fm.write("source1_entity_id\tmatched_entity_ids\n")
    fc.write("source1_entity_id\tcandidate_entity_ids\n")
    
    for eid, tids in preds.items():
        cand_list = cands.get(eid, [])
        if not tids:
            fm.write(f"{eid}\t\n")
            if cand_list:
                fc.write(f"{eid}\t{','.join(cand_list[:15])}\n")
                total_final_cands += min(len(cand_list), 15)
            else:
                fc.write(f"{eid}\t\n")
            continue
            
        s1_h = s1_houses.get(eid)
        if not s1_h:
            # No contradiction possible
            clean_m = tids
        else:
            clean_m = []
            for tid in tids:
                t_h = target_houses.get(tid)
                if t_h and t_h != s1_h and not (s1_h in t_h or t_h in s1_h):
                    pruned_matches += 1
                else:
                    clean_m.append(tid)
            if len(clean_m) < len(tids):
                pruned_entities += 1
                
        # Ensure candidate list contains clean_m with no duplicates
        clean_c = list(dict.fromkeys(clean_m + cand_list))[:15]
        valid_m = [cid for cid in clean_m if cid in set(clean_c)]
        
        if valid_m:
            fm.write(f"{eid}\t{','.join(valid_m)}\n")
            non_empty_entities += 1
            total_final_matches += len(valid_m)
        else:
            fm.write(f"{eid}\t\n")
            
        if clean_c:
            fc.write(f"{eid}\t{','.join(clean_c)}\n")
            total_final_cands += len(clean_c)
        else:
            fc.write(f"{eid}\t\n")

print(f"\n[PRUNING COMPLETE]")
print(f"  Pruned false-merge matches: {pruned_matches:,}")
print(f"  Entities refined:           {pruned_entities:,}")
print(f"  Total remaining matches:    {total_final_matches:,}")
print(f"  Non-empty entities:         {non_empty_entities:,} ({non_empty_entities/len(preds)*100:.1f}%)")

# 5. Run Official Validator
print("\n[*] Running official submission validator...", flush=True)
validator_cmd = [
    sys.executable,
    "-X", "utf8",
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

# 6. Package Final Zips
print(f"\n[*] Packaging final submission to {ZIP_OUT}...", flush=True)
with zipfile.ZipFile(ZIP_OUT, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(MATCHING_OUT, arcname="output/matching_results.tsv")
    zipf.write(CANDIDATE_OUT, arcname="output/candidate_pairs.tsv")
    if os.path.exists("Documentation_template.md"):
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

with zipfile.ZipFile(MATCH_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(MATCHING_OUT, arcname="matching_results.tsv")
print(f"[COMPLETE] Matches Package created: {MATCH_ZIP} ({os.path.getsize(MATCH_ZIP)/(1024*1024):.2f} MB)", flush=True)

print(f"\n[SUCCESS] Pipeline v8 completed in {time.time()-t0:.1f}s!", flush=True)
