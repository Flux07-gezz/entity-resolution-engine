"""Leakage-safe validation split module.

Guarantees:
- Strict grouping by Source 1 entity IDs (zero leakage of candidate pairs across train/validation).
- Stratification across match-count categories (singleton, single-match, multi-match).
- Fixed sample caching for iteration budget compliance.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold


def categorize_s1(matches: Sequence[str]) -> str:
    """Classify S1 entity by its ground-truth match cardinality."""
    count = len(matches)
    if count == 0:
        return "singleton"
    elif count == 1:
        return "single_match"
    else:
        return "multi_match"


def create_stratified_sample(
    all_s1_ids: Sequence[str],
    ground_truth: Dict[str, Sequence[str]],
    country_map: Optional[Dict[str, str]] = None,
    sample_size: int = 5000,
    random_seed: int = 42,
) -> List[str]:
    """Sample a balanced, fixed subset of Source 1 IDs stratified by country and cardinality."""
    if len(all_s1_ids) <= sample_size:
        return list(all_s1_ids)

    rng = np.random.RandomState(random_seed)
    by_category: Dict[str, List[str]] = defaultdict(list)

    for s1_id in all_s1_ids:
        cat = categorize_s1(ground_truth.get(s1_id, []))
        if country_map:
            country = country_map.get(s1_id, "unknown")
            stratum = f"{country}_{cat}"
        else:
            stratum = cat
        by_category[stratum].append(s1_id)

    sampled: List[str] = []
    total_records = len(all_s1_ids)

    for stratum, ids in by_category.items():
        proportion = len(ids) / total_records
        n_sample = max(1, int(round(proportion * sample_size)))
        chosen = rng.choice(ids, size=min(n_sample, len(ids)), replace=False).tolist()
        sampled.extend(chosen)

    # If minor rounding discrepancy, adjust
    if len(sampled) > sample_size:
        sampled = rng.choice(sampled, size=sample_size, replace=False).tolist()
    elif len(sampled) < sample_size:
        remaining = list(set(all_s1_ids) - set(sampled))
        fill = rng.choice(remaining, size=(sample_size - len(sampled)), replace=False).tolist()
        sampled.extend(fill)

    return sorted(sampled)


def create_grouped_kfold(
    s1_ids: Sequence[str],
    ground_truth: Dict[str, Sequence[str]],
    country_map: Optional[Dict[str, str]] = None,
    n_splits: int = 5,
    random_seed: int = 42,
) -> List[Tuple[List[str], List[str]]]:
    """Create K leakage-safe folds of Source 1 IDs, stratified by country and match category.

    Returns:
        List of (train_s1_ids, val_s1_ids) tuples.
    """
    labels = []
    for s1_id in s1_ids:
        cat = categorize_s1(ground_truth.get(s1_id, []))
        if country_map:
            labels.append(f"{country_map.get(s1_id, 'unknown')}_{cat}")
        else:
            labels.append(cat)

    label_to_id = {lbl: idx for idx, lbl in enumerate(sorted(set(labels)))}
    y = np.array([label_to_id[lbl] for lbl in labels])
    X = np.array(s1_ids)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    folds = []

    for train_idx, val_idx in skf.split(X, y):
        train_s1 = X[train_idx].tolist()
        val_s1 = X[val_idx].tolist()
        # Strict zero-leakage invariant
        assert len(set(train_s1) & set(val_s1)) == 0, "Leakage detected between train and val S1 IDs!"
        folds.append((train_s1, val_s1))

    return folds


def save_splits(folds: List[Tuple[List[str], List[str]]], path: Path):
    """Persist folds to disk for guaranteed consistency across runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [{"train": train, "val": val} for train, val in folds]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def load_splits(path: Path) -> List[Tuple[List[str], List[str]]]:
    """Load persisted folds from disk."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(fold["train"], fold["val"]) for fold in data]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate leakage-safe grouped validation splits.")
    parser.add_argument("--s1-file", default="data/student_resource/dataset/train/train_source1.tsv")
    parser.add_argument("--gt-file", default="data/student_resource/dataset/train/train_ground_truth.tsv")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--output", default="evaluation/splits_cv5.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading S1 from {args.s1_file}...")
    s1_ids = []
    country_map = {}
    with open(args.s1_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid, country = parts[0], parts[3]
            s1_ids.append(eid)
            country_map[eid] = country

    print(f"Loading ground truth from {args.gt_file}...")
    gt = {}
    with open(args.gt_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            matches = parts[1].split(",") if len(parts) > 1 and parts[1].strip() else []
            gt[s1_id] = matches

    print(f"Sampling {args.sample_size} entities stratified by (country, cardinality)...")
    sample_ids = create_stratified_sample(
        s1_ids, gt, country_map=country_map, sample_size=args.sample_size, random_seed=args.seed
    )

    print(f"Creating {args.n_splits}-fold grouped split...")
    folds = create_grouped_kfold(
        sample_ids, gt, country_map=country_map, n_splits=args.n_splits, random_seed=args.seed
    )

    out_path = Path(args.output)
    save_splits(folds, out_path)
    print(f"Successfully saved {args.n_splits} folds to {out_path}!")
    for idx, (tr, va) in enumerate(folds, 1):
        print(f"  Fold {idx}: {len(tr)} train, {len(va)} val (zero overlap: {len(set(tr) & set(va)) == 0})")


if __name__ == "__main__":
    main()

