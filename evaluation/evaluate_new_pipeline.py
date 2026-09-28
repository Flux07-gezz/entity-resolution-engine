import json
import os
import sys
sys.path.insert(0, ".")
import time
from collections import defaultdict
from rapidfuzz import fuzz
from evaluation.metrics import evaluate_entity_resolution
from src.predict_submission import clean_tokens, extract_house_number, THRESHOLD, TOP_K_CANDIDATES, MAX_POSTING, MAX_HOUSE_POSTING

TRAIN_DIR = r"data\student_resource\dataset\train"

# 1. Load Fold 0 validation entities
with open("evaluation/splits_cv5.json", "r") as f:
    splits = json.load(f)
val_s1_ids = set(splits[0]["val"])
print(f"Loaded {len(val_s1_ids)} validation S1 entities for Fold 1.")

# 2. Load ground truth for these entities
ground_truth = defaultdict(list)
with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.strip().split("\t")
        s1 = parts[0]
        if s1 in val_s1_ids:
            targets = [t.strip() for t in parts[1].split(",") if t.strip()] if len(parts) > 1 else []
            ground_truth[s1] = targets

gt_dict = {s1: sorted(ground_truth.get(s1, [])) for s1 in val_s1_ids}

print(f"Ground truth loaded. Non-singleton validation entities: {sum(1 for targets in gt_dict.values() if targets)}")
print(f"Singleton validation entities: {sum(1 for targets in gt_dict.values() if not targets)}")

# 3. Load S1 validation records
val_records = []
s1_by_id = {}
with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.strip().split("\t")
        if parts[0] in val_s1_ids:
            rec = {
                "id": parts[0],
                "name": parts[1].lower().strip(),
                "address": parts[2].lower().strip(),
                "country": parts[3],
                "tokens": set(clean_tokens(parts[1].lower().strip())),
                "house": extract_house_number(parts[2].lower().strip())
            }
            val_records.append(rec)
            s1_by_id[parts[0]] = rec

print(f"Loaded {len(val_records)} validation S1 entity records.")

# Partition validation entities by country
s1_by_country = defaultdict(list)
for idx, r in enumerate(val_records):
    s1_by_country[r["country"]].append((idx, r))

# 4. Inverted index per country
country_indices = {}
for c, entity_list in s1_by_country.items():
    raw_tokens = defaultdict(list)
    raw_houses = defaultdict(list)
    for idx, r in entity_list:
        for t in r["tokens"]:
            raw_tokens[t].append(idx)
        if r["house"]:
            raw_houses[r["house"]].append(idx)
    pruned_tokens = {t: ids for t, ids in raw_tokens.items() if len(ids) <= MAX_POSTING}
    pruned_houses = {h: ids for h, ids in raw_houses.items() if len(ids) <= MAX_HOUSE_POSTING}
    country_indices[c] = (pruned_tokens, pruned_houses)

# Candidate storage: val_idx -> dict(tid -> score)
s1_candidates = [dict() for _ in range(len(val_records))]

# 5. Stream train_source2.tsv and train_source3.tsv
t0 = time.time()
for tf in ["train_source2.tsv", "train_source3.tsv"]:
    tf_path = os.path.join(TRAIN_DIR, tf)
    print(f"Streaming {tf}...")
    with open(tf_path, "r", encoding="utf-8") as f:
        f.readline()
        scanned = 0
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            country = parts[3]
            if country not in country_indices:
                continue
            
            scanned += 1
            tid = parts[0]
            tname = parts[1].lower().strip()
            taddr = parts[2].lower().strip()
            t_tokens = clean_tokens(tname)
            if not t_tokens:
                continue

            token_idx, house_idx = country_indices[country]
            matched_indices = set()
            for tok in t_tokens:
                hits = token_idx.get(tok)
                if hits:
                    matched_indices.update(hits)
            
            if taddr:
                thouse = extract_house_number(taddr)
                if thouse:
                    hhits = house_idx.get(thouse)
                    if hhits:
                        t_set = set(t_tokens)
                        for idx in hhits:
                            if t_set & val_records[idx]["tokens"]:
                                matched_indices.add(idx)

            if not matched_indices:
                continue

            if len(matched_indices) > 50:
                t_set = set(t_tokens)
                matched_indices = sorted(
                    matched_indices,
                    key=lambda idx: len(t_set & val_records[idx]["tokens"]),
                    reverse=True
                )[:50]

            for idx in matched_indices:
                s1_rec = val_records[idx]
                sort_ratio = fuzz.token_sort_ratio(s1_rec["name"], tname)
                if sort_ratio < 70:
                    continue
                set_ratio = fuzz.token_set_ratio(s1_rec["name"], tname)
                name_sim = (sort_ratio * 0.4 + set_ratio * 0.6) / 100.0
                
                if s1_rec["address"] and taddr:
                    addr_sim = fuzz.token_set_ratio(s1_rec["address"], taddr) / 100.0
                else:
                    addr_sim = 0.50

                score = 0.65 * name_sim + 0.35 * addr_sim
                if score >= 0.70:
                    cdict = s1_candidates[idx]
                    if score > cdict.get(tid, 0.0):
                        cdict[tid] = score
                        if len(cdict) > TOP_K_CANDIDATES * 2:
                            sc = sorted(cdict.items(), key=lambda x: x[1], reverse=True)[:TOP_K_CANDIDATES]
                            s1_candidates[idx] = dict(sc)

print(f"Streaming target files completed in {time.time()-t0:.2f}s.")

# 6. Evaluate predictions at calibrated threshold tau = 0.86
predictions = {}
candidates_dict = {}
for idx, r in enumerate(val_records):
    s1_id = r["id"]
    cand_items = sorted(s1_candidates[idx].items(), key=lambda x: x[1], reverse=True)[:TOP_K_CANDIDATES]
    cand_ids = [tid for tid, _ in cand_items]
    matched_ids = [tid for tid, sc in cand_items if sc >= THRESHOLD]
    predictions[s1_id] = matched_ids
    candidates_dict[s1_id] = cand_ids

eval_results = evaluate_entity_resolution(gt_dict, predictions, list(val_s1_ids))

# Calculate Candidate Recall Ceiling
total_gt_pairs = 0
recalled_gt_pairs = 0
for s1_id, gt_targets in gt_dict.items():
    cand_set = set(candidates_dict.get(s1_id, []))
    for t in gt_targets:
        total_gt_pairs += 1
        if t in cand_set:
            recalled_gt_pairs += 1

cand_recall = recalled_gt_pairs / total_gt_pairs if total_gt_pairs > 0 else 0.0

print("\n=======================================================")
print(f"BENCHMARK RESULTS FOR NEW OPTIMIZED PIPELINE (Fold 1)")
print(f"Threshold tau* = {THRESHOLD}")
print(f"=======================================================")
print(f"Entity-Level Macro F0.5:   {eval_results['entity_macro_f05']*100:.2f}%")
print(f"Entity-Level Precision:    {eval_results['entity_macro_precision']*100:.2f}%")
print(f"Entity-Level Recall:       {eval_results['entity_macro_recall']*100:.2f}%")
print(f"Pair Precision:            {eval_results['pair_precision']*100:.2f}%")
print(f"Pair Recall:               {eval_results['pair_recall']*100:.2f}%")
print(f"Candidate Recall Ceiling:  {cand_recall*100:.2f}%")
print(f"Singleton Accuracy:        {eval_results['singleton_accuracy']*100:.2f}%")
print(f"Exact Set Accuracy:        {eval_results['exact_set_accuracy']*100:.2f}%")
print(f"False Merges per 100 S1:   {eval_results['false_merges_per_100_s1']:.2f}")
print(f"Missed Matches per 100 S1: {eval_results['missed_matches_per_100_s1']:.2f}")
print("=======================================================")
