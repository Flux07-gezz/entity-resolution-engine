"""Build High-Precision Union Submission (Run 7: Peak 0.691 Baseline + 81k Rescued India Entities).

Key Principle:
1. Base is 100% Run 3 (0.691 peak): France v3, US v3, and all 555,857 non-empty India v3 entities.
2. For the 254k previously empty India entities, inject the 81,123 high-confidence entities rescued by 2-word shingles.
3. Guarantee matched_ids is a strict subset of candidate_ids.
4. Validate with official submission validator.
"""

import os
import sys
import time
import zipfile
import subprocess

OUTPUT_DIR = "output"
MATCHING_OUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
ZIP_OUT = os.path.join(OUTPUT_DIR, "submission_step2.zip")
MATCH_ZIP = os.path.join(OUTPUT_DIR, "matching_results.zip")

DATA_DIR = r"data\student_resource\dataset\test"

t0 = time.time()
print("==========================================================", flush=True)
print("Amazon ML Challenge 2026: Precision Union Submission Engine v7", flush=True)
print("Run 3 Peak Base (0.691) + Run 6 2-Word Shingle Rescued Entities", flush=True)
print("==========================================================", flush=True)

# 1. Load India v3 (Base)
print("[*] Loading India v3 base matches and candidates...", flush=True)
v3_matches = {}
with open(os.path.join(OUTPUT_DIR, "matching_lgb_v3_India.tsv"), "r", encoding="utf-8") as f:
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        v3_matches[p[0]] = p[1].split(",") if len(p) > 1 and p[1].strip() else []

v3_cands = {}
with open(os.path.join(OUTPUT_DIR, "candidate_lgb_v3_India.tsv"), "r", encoding="utf-8") as f:
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        v3_cands[p[0]] = p[1].split(",") if len(p) > 1 and p[1].strip() else []

# 2. Load India v6 (Rescued)
print("[*] Loading India v6 rescued matches and candidates...", flush=True)
v6_matches = {}
with open(os.path.join(OUTPUT_DIR, "matching_lgb_v6_India.tsv"), "r", encoding="utf-8") as f:
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        v6_matches[p[0]] = p[1].split(",") if len(p) > 1 and p[1].strip() else []

v6_cands = {}
with open(os.path.join(OUTPUT_DIR, "candidate_lgb_v6_India.tsv"), "r", encoding="utf-8") as f:
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        v6_cands[p[0]] = p[1].split(",") if len(p) > 1 and p[1].strip() else []

# 3. Create Merged India Checkpoint
print("[*] Creating high-precision merged India checkpoint...", flush=True)
merged_match_path = os.path.join(OUTPUT_DIR, "matching_lgb_v7_India.tsv")
merged_cand_path = os.path.join(OUTPUT_DIR, "candidate_lgb_v7_India.tsv")

rescued_count = 0
retained_base_count = 0
total_india_matches = 0

with open(merged_match_path, "w", encoding="utf-8", newline="") as fm, \
     open(merged_cand_path, "w", encoding="utf-8", newline="") as fc:
    for eid, m3 in v3_matches.items():
        c3 = list(dict.fromkeys(v3_cands.get(eid, [])))
        if m3:
            # Keep Run 3 base
            retained_base_count += 1
            # Ensure m3 is subset of c3
            merged_c = list(dict.fromkeys(m3 + c3))[:15]
            valid_m3 = [cid for cid in m3 if cid in set(merged_c)]
            fm.write(f"{eid}\t{','.join(valid_m3)}\n")
            fc.write(f"{eid}\t{','.join(merged_c)}\n")
            total_india_matches += len(valid_m3)
        else:
            # Check if rescued by v6
            m6 = v6_matches.get(eid, [])
            c6 = list(dict.fromkeys(v6_cands.get(eid, [])))
            if m6:
                rescued_count += 1
                # Merge candidates to ensure m6 is inside candidates
                merged_c = list(dict.fromkeys(m6 + c6 + c3))[:15]
                valid_m6 = [cid for cid in m6 if cid in set(merged_c)]
                fm.write(f"{eid}\t{','.join(valid_m6)}\n")
                fc.write(f"{eid}\t{','.join(merged_c)}\n")
                total_india_matches += len(valid_m6)
            else:
                fm.write(f"{eid}\t\n")
                dedup_c = list(dict.fromkeys(c3 + c6))[:15]
                if dedup_c:
                    fc.write(f"{eid}\t{','.join(dedup_c)}\n")
                else:
                    fc.write(f"{eid}\t\n")

print(f"  India Merged: Retained Base={retained_base_count:,}, Rescued={rescued_count:,}")
print(f"  Total Non-Empty India: {retained_base_count + rescued_count:,} ({(retained_base_count+rescued_count)/len(v3_matches)*100:.1f}%)")
print(f"  Total India Matches:   {total_india_matches:,}")

# 4. Assemble Final Submission (France v3 + US v3 + India v7)
print("\n[*] Assembling final submission files (France v3 + US v3 + India v7)...", flush=True)
country_ckpts = [
    (os.path.join(OUTPUT_DIR, "matching_lgb_v3_France.tsv"), os.path.join(OUTPUT_DIR, "candidate_lgb_v3_France.tsv")),
    (os.path.join(OUTPUT_DIR, "matching_lgb_v3_US.tsv"), os.path.join(OUTPUT_DIR, "candidate_lgb_v3_US.tsv")),
    (merged_match_path, merged_cand_path)
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

print(f"\n[SUCCESS] Pipeline v7 completed in {time.time()-t0:.1f}s!", flush=True)
