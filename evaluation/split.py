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
    sample_size: int = 3000,
    random_seed: int = 42,
) -> List[str]:
    """Sample a balanced, fixed subset of Source 1 IDs stratified by cardinality category."""
    if len(all_s1_ids) <= sample_size:
        return list(all_s1_ids)

    rng = np.random.RandomState(random_seed)
    by_category: Dict[str, List[str]] = defaultdict(list)

    for s1_id in all_s1_ids:
        cat = categorize_s1(ground_truth.get(s1_id, []))
        by_category[cat].append(s1_id)

    sampled: List[str] = []
    total_records = len(all_s1_ids)

    for cat, ids in by_category.items():
        proportion = len(ids) / total_records
        n_cat_sample = max(1, int(round(proportion * sample_size)))
        # Sample without replacement
        chosen = rng.choice(ids, size=min(n_cat_sample, len(ids)), replace=False).tolist()
        sampled.extend(chosen)

    # If minor rounding discrepancy, adjust
    if len(sampled) > sample_size:
        sampled = rng.choice(sampled, size=sample_size, replace=False).tolist()

    return sorted(sampled)


def create_grouped_kfold(
    s1_ids: Sequence[str],
    ground_truth: Dict[str, Sequence[str]],
    n_splits: int = 5,
    random_seed: int = 42,
) -> List[Tuple[List[str], List[str]]]:
    """Create K leakage-safe folds of Source 1 IDs, stratified by match category.

    Returns:
        List of (train_s1_ids, val_s1_ids) tuples.
    """
    labels = [categorize_s1(ground_truth.get(s1_id, [])) for s1_id in s1_ids]
    cat_map = {"singleton": 0, "single_match": 1, "multi_match": 2}
    y = np.array([cat_map[lbl] for lbl in labels])
    X = np.array(s1_ids)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    folds = []

    for train_idx, val_idx in skf.split(X, y):
        train_s1 = X[train_idx].tolist()
        val_s1 = X[val_idx].tolist()
        # Verify strict zero-leakage invariant
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
