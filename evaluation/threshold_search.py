"""Threshold search module for optimizing Entity-Level Macro F_0.5.

Performs a single, bounded sweep (0.50 to 0.98 in 0.02 increments) as specified by the compute budget.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from evaluation.metrics import evaluate_entity_resolution
from src.decision import EntityDecisionEngine


def search_best_threshold(
    scored_pairs: List[Dict[str, Any]],
    ground_truth: Dict[str, Sequence[str]],
    all_s1_ids: Sequence[str],
    min_thresh: float = 0.50,
    max_thresh: float = 0.98,
    step: float = 0.02,
    enable_hard_contradictions: bool = True,
) -> Tuple[float, Dict[str, float], List[Dict[str, Any]]]:
    """Sweep decision thresholds and identify threshold maximizing Entity-Level Macro F_0.5.

    Returns:
        (best_threshold, best_metrics, threshold_sweep_table)
    """
    decision_engine = EntityDecisionEngine(
        enable_hard_contradiction_filter=enable_hard_contradictions
    )

    thresholds = np.arange(min_thresh, max_thresh + 1e-5, step)
    sweep_results: List[Dict[str, Any]] = []

    best_threshold = min_thresh
    best_f05 = -1.0
    best_metrics: Dict[str, float] = {}

    for thresh in thresholds:
        thresh_val = round(float(thresh), 4)
        preds = decision_engine.decide_entities(
            scored_pairs, all_s1_ids, threshold=thresh_val
        )
        metrics = evaluate_entity_resolution(ground_truth, preds, all_s1_ids)

        row = {
            "threshold": thresh_val,
            "entity_macro_f05": metrics["entity_macro_f05"],
            "entity_macro_precision": metrics["entity_macro_precision"],
            "entity_macro_recall": metrics["entity_macro_recall"],
            "pair_precision": metrics["pair_precision"],
            "pair_recall": metrics["pair_recall"],
            "singleton_accuracy": metrics["singleton_accuracy"],
            "exact_set_accuracy": metrics["exact_set_accuracy"],
            "false_merges_per_100_s1": metrics["false_merges_per_100_s1"],
            "missed_matches_per_100_s1": metrics["missed_matches_per_100_s1"],
        }
        sweep_results.append(row)

        if metrics["entity_macro_f05"] > best_f05:
            best_f05 = metrics["entity_macro_f05"]
            best_threshold = thresh_val
            best_metrics = metrics

    return best_threshold, best_metrics, sweep_results
