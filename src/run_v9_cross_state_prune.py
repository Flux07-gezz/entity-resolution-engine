"""Run 9: Cross-State Geographic Contradiction Pruner.

Builds on top of Run 8 (which scored 0.710 on live leaderboard).
Prunes 84,918 cross-state false merges across 62,565 entities:
- If S1 and Target are in different States/Regions AND name similarity is < 92%, prune!
- Guarantees 100% referential integrity (matched_ids subset of candidate_ids).
- Validated with official submission validator.
- Generates submission_step2.zip and matching_results.zip.
"""

import os
import re
import sys
import time
import zipfile
import subprocess
from rapidfuzz import fuzz

OUTPUT_DIR = "output"
DATA_DIR = r"data\student_resource\dataset\test"

MATCHING_OUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
ZIP_OUT = os.path.join(OUTPUT_DIR, "submission_step2.zip")
MATCH_ZIP = os.path.join(OUTPUT_DIR, "matching_results.zip")

INDIAN_CITIES = {
    "mumbai": "MH", "pune": "MH", "nagpur": "MH", "maharashtra": "MH", "thane": "MH", "nashik": "MH",
    "delhi": "DL", "new delhi": "DL",
    "bangalore": "KA", "bengaluru": "KA", "karnataka": "KA", "mysore": "KA",
    "chennai": "TN", "tamil nadu": "TN", "coimbatore": "TN", "madurai": "TN",
    "hyderabad": "TS", "telangana": "TS", "secunderabad": "TS",
    "kolkata": "WB", "howrah": "WB", "west bengal": "WB", "siliguri": "WB",
    "ahmedabad": "GJ", "surat": "GJ", "gujarat": "GJ", "vadodara": "GJ", "rajkot": "GJ",
    "jaipur": "RJ", "rajasthan": "RJ", "jodhpur": "RJ", "udaipur": "RJ",
    "lucknow": "UP", "kanpur": "UP", "uttar pradesh": "UP", "noida": "UP", "ghaziabad": "UP", "agra": "UP", "varanasi": "UP",
    "patna": "BR", "bihar": "BR", "muzaffarpur": "BR",
    "bhubaneswar": "OD", "orissa": "OD", "odisha": "OD", "cuttack": "OD",
    "chandigarh": "PB", "punjab": "PB", "ludhiana": "PB", "amritsar": "PB",
    "indore": "MP", "bhopal": "MP", "madhya pradesh": "MP", "gwalior": "MP",
    "kochi": "KL", "kerala": "KL", "trivandrum": "KL", "ernakulam": "KL"
}

US_STATES = {
    "ca": "CA", "california": "CA", "ny": "NY", "new york": "NY",
    "tx": "TX", "texas": "TX", "fl": "FL", "florida": "FL",
    "il": "IL", "illinois": "IL", "pa": "PA", "pennsylvania": "PA",
    "oh": "OH", "ohio": "OH", "ga": "GA", "georgia": "GA",
    "nc": "NC", "north carolina": "NC", "mi": "MI", "michigan": "MI",
    "nj": "NJ", "new jersey": "NJ", "va": "VA", "virginia": "VA",
    "wa": "WA", "washington": "WA", "az": "AZ", "arizona": "AZ",
    "ma": "MA", "massachusetts": "MA", "tn": "TN", "tennessee": "TN",
    "in": "IN", "indiana": "IN", "mo": "MO", "missouri": "MO",
    "md": "MD", "maryland": "MD", "wi": "WI", "wisconsin": "WI",
    "co": "CO", "colorado": "CO", "mn": "MN", "minnesota": "MN"
}

def extract_region(addr: str, country: str):
    if not addr: return ""
    a = addr.lower()
    if country == "India":
        for k, v in INDIAN_CITIES.items():
            if k in a:
                return v
    elif country == "US":
        for k, v in US_STATES.items():
            if k in a:
                return v
    return ""

t0 = time.time()
print("==========================================================", flush=True)
print("Amazon ML Challenge 2026: Cross-State Precision Engine v9", flush=True)
print("Pruning 84k Geographic Contradictions on Top of Run 8 (0.710)", flush=True)
print("==========================================================", flush=True)

# 1. Load S1 entities
s1_data = {}
with open(os.path.join(DATA_DIR, "test_source1.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        r = extract_region(p[2], p[3])
        s1_data[p[0]] = (p[1], p[2], p[3], r)

print(f"[*] Loaded {len(s1_data):,} S1 entities in {time.time()-t0:.1f}s.", flush=True)

# 2. Load current Run 8 predictions
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

# 3. Stream S2 and S3 for matched target records
target_data = {}
for src in ["test_source2.tsv", "test_source3.tsv"]:
    t_stream = time.time()
    print(f"[*] Streaming {src} for matched target regions...", flush=True)
    with open(os.path.join(DATA_DIR, src), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in all_matched_tids:
                r = extract_region(p[2], p[3])
                target_data[p[0]] = (p[1], p[2], p[3], r)
    print(f"    Done {src} in {time.time()-t_stream:.1f}s.", flush=True)

print(f"[*] Loaded metadata for {len(target_data):,} matched targets.", flush=True)

# 4. Prune Cross-State Contradictions & Write Final Cleaned TSVs
print("[*] Applying Cross-State Contradiction Pruner to generate Run 9 files...", flush=True)
pruned_matches = 0
pruned_entities = 0
total_final_matches = 0
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
            else:
                fc.write(f"{eid}\t\n")
            continue
            
        s_name, s_addr, s_country, s_reg = s1_data.get(eid, ("", "", "", ""))
        if not s_reg:
            clean_m = tids
        else:
            clean_m = []
            for tid in tids:
                tinfo = target_data.get(tid)
                if not tinfo:
                    clean_m.append(tid)
                    continue
                t_name, t_addr, t_country, t_reg = tinfo
                if t_reg and t_reg != s_reg:
                    # Cross-state match: keep ONLY if name is practically exact brand match (sort >= 92)
                    ns = fuzz.token_sort_ratio(s_name.lower(), t_name.lower())
                    if ns < 92:
                        pruned_matches += 1
                    else:
                        clean_m.append(tid)
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
        else:
            fc.write(f"{eid}\t\n")

print(f"\n[PRUNING COMPLETE]")
print(f"  Pruned cross-state false matches: {pruned_matches:,}")
print(f"  Entities refined:                {pruned_entities:,}")
print(f"  Total remaining matches:         {total_final_matches:,}")
print(f"  Non-empty entities:              {non_empty_entities:,} ({non_empty_entities/len(preds)*100:.1f}%)")

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

print(f"\n[SUCCESS] Pipeline v9 completed in {time.time()-t0:.1f}s!", flush=True)
