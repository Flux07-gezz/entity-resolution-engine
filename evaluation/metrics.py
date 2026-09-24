"""Evaluation metrics for Business Entity Resolution.

Focuses on Entity-Level Macro F_0.5 and the competition dashboard metrics.
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np


def compute_entity_f_beta(
    true_set: Set[str],
    pred_set: Set[str],
    beta: float = 0.5,
) -> Tuple[float, float, float, int, int]:
    """Compute precision, recall, F_beta, false_positives, false_negatives for a single entity.

    Returns:
        (precision, recall, f_beta, false_merges, missed_matches)
    """
    beta_sq = beta ** 2

    # Case 1: Both true and predicted are empty (true singleton correctly predicted)
    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0, 1.0, 1.0, 0, 0

    # Case 2: True set empty, predicted not empty (false merge on singleton)
    if len(true_set) == 0 and len(pred_set) > 0:
        return 0.0, 1.0, 0.0, len(pred_set), 0

    # Case 3: True set not empty, predicted empty (missed all matches)
    if len(true_set) > 0 and len(pred_set) == 0:
        return 0.0, 0.0, 0.0, 0, len(true_set)

    # Case 4: Both non-empty
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = tp / len(true_set) if len(true_set) > 0 else 0.0

    denom = (beta_sq * precision) + recall
    if denom == 0.0:
        f_beta = 0.0
    else:
        f_beta = ((1.0 + beta_sq) * precision * recall) / denom

    return precision, recall, f_beta, fp, fn


def evaluate_entity_resolution(
    ground_truth: Dict[str, Sequence[str]],
    predictions: Dict[str, Sequence[str]],
    all_s1_ids: Optional[Sequence[str]] = None,
    beta: float = 0.5,
) -> Dict[str, float]:
    """Calculate the full experiment dashboard metrics at entity and pair levels."""
    if all_s1_ids is None:
        eval_s1_ids = sorted(list(ground_truth.keys()))
    else:
        eval_s1_ids = list(all_s1_ids)

    n_entities = len(eval_s1_ids)
    if n_entities == 0:
        raise ValueError("Cannot evaluate on empty entity set.")

    precisions: List[float] = []
    recalls: List[float] = []
    f_betas: List[float] = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    singleton_correct = 0
    singleton_total = 0

    exact_set_matches = 0
    total_false_merges = 0
    total_missed_matches = 0

    for s1_id in eval_s1_ids:
        t_set = set(ground_truth.get(s1_id, []))
        p_set = set(predictions.get(s1_id, []))

        # Check singleton
        if len(t_set) == 0:
            singleton_total += 1
            if len(p_set) == 0:
                singleton_correct += 1

        # Check exact set
        if t_set == p_set:
            exact_set_matches += 1

        prec, rec, fb, fp, fn = compute_entity_f_beta(t_set, p_set, beta=beta)
        precisions.append(prec)
        recalls.append(rec)
        f_betas.append(fb)

        total_false_merges += fp
        total_missed_matches += fn

        tp = len(t_set & p_set)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    pair_precision = (
        (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    )
    pair_recall = (
        (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    )

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f_beta = float(np.mean(f_betas))

    singleton_accuracy = (
        (singleton_correct / singleton_total) if singleton_total > 0 else 1.0
    )
    exact_set_accuracy = exact_set_matches / n_entities

    false_merges_per_100_s1 = (total_false_merges / n_entities) * 100.0
    missed_matches_per_100_s1 = (total_missed_matches / n_entities) * 100.0

    return {
        "entity_macro_f05": macro_f_beta,
        "entity_macro_precision": macro_precision,
        "entity_macro_recall": macro_recall,
        "pair_precision": pair_precision,
        "pair_recall": pair_recall,
        "singleton_accuracy": singleton_accuracy,
        "exact_set_accuracy": exact_set_accuracy,
        "false_merges_per_100_s1": false_merges_per_100_s1,
        "missed_matches_per_100_s1": missed_matches_per_100_s1,
        "total_entities": float(n_entities),
        "total_singletons": float(singleton_total),
        "total_true_pairs": float(total_tp + total_fn),
        "total_pred_pairs": float(total_tp + total_fp),
    }


def format_dashboard(metrics: Dict[str, float], title: str = "EVALUATION DASHBOARD") -> str:
    """Format metrics dictionary according to Section 23 specifications."""
    lines = [
        f"=========================================",
        f"{title.upper()}",
        f"=========================================",
        f"Entity Macro F0.5:         {metrics.get('entity_macro_f05', 0.0) * 100:.2f}%",
        f"Entity Macro Precision:    {metrics.get('entity_macro_precision', 0.0) * 100:.2f}%",
        f"Entity Macro Recall:       {metrics.get('entity_macro_recall', 0.0) * 100:.2f}%",
        f"Pair Precision:            {metrics.get('pair_precision', 0.0) * 100:.2f}%",
        f"Pair Recall:               {metrics.get('pair_recall', 0.0) * 100:.2f}%",
        f"Singleton Accuracy:        {metrics.get('singleton_accuracy', 0.0) * 100:.2f}%",
        f"Exact Set Accuracy:        {metrics.get('exact_set_accuracy', 0.0) * 100:.2f}%",
        f"False merges / 100 S1:     {metrics.get('false_merges_per_100_s1', 0.0):.2f}",
        f"Missed matches / 100 S1:   {metrics.get('missed_matches_per_100_s1', 0.0):.2f}",
    ]
    if "candidate_recall" in metrics:
        lines.append(f"Candidate Recall:          {metrics['candidate_recall'] * 100:.2f}%")
    if "avg_candidates_per_s1" in metrics:
        lines.append(f"Avg candidates / S1:       {metrics['avg_candidates_per_s1']:.2f}")
    lines.append(f"=========================================")
    return "\n".join(lines)
