"""Error analysis and diagnostic reporting for entity resolution.

Analyzes and categorizes false merges (precision errors) and missed matches (recall errors):
- Distinguishes address collisions, name collisions, franchise branches, parser discrepancies.
- Outputs diagnostic reports for targeted iterations.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import pandas as pd


def analyze_errors(
    predictions: Dict[str, Sequence[str]],
    ground_truth: Dict[str, Sequence[str]],
    s1_records: Dict[str, Dict[str, Any]],
    target_records: Dict[str, Dict[str, Any]],
    candidate_features: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Inspect and categorize entity-level errors.

    Returns:
        Dict containing lists of 'false_merges' and 'missed_matches' with diagnostic metadata.
    """
    false_merges: List[Dict[str, Any]] = []
    missed_matches: List[Dict[str, Any]] = []

    for s1_id, pred_list in predictions.items():
        true_list = ground_truth.get(s1_id, [])
        pred_set = set(pred_list)
        true_set = set(true_list)

        s1_data = s1_records.get(s1_id, {})

        # False Merges (FP)
        for target_id in (pred_set - true_set):
            t_data = target_records.get(target_id, {})
            feats = (
                candidate_features.get((s1_id, target_id), {})
                if candidate_features
                else {}
            )

            # Heuristic category diagnosis
            category = "other"
            s1_name = str(s1_data.get("name", "")).lower()
            t_name = str(t_data.get("name", "")).lower()
            s1_addr = str(s1_data.get("address", "")).lower()
            t_addr = str(t_data.get("address", "")).lower()

            if s1_name and t_name and (s1_name == t_name or s1_name in t_name or t_name in s1_name):
                if s1_addr != t_addr:
                    category = "same_name_diff_address_or_branch"
            elif s1_addr and t_addr and s1_addr == t_addr:
                if s1_name != t_name:
                    category = "address_collision_diff_business"
            elif feats.get("house_num_conflict", 0.0) == 1.0:
                category = "house_number_conflict"
            elif feats.get("city_conflict", 0.0) == 1.0:
                category = "city_conflict"

            false_merges.append({
                "s1_id": s1_id,
                "target_id": target_id,
                "category": category,
                "s1_name": s1_data.get("name"),
                "target_name": t_data.get("name"),
                "s1_address": s1_data.get("address"),
                "target_address": t_data.get("address"),
                "model_score": feats.get("score"),
                "blocking_passes": feats.get("blocking_passes"),
            })

        # Missed Matches (FN)
        for target_id in (true_set - pred_set):
            t_data = target_records.get(target_id, {})
            feats = (
                candidate_features.get((s1_id, target_id), {})
                if candidate_features
                else {}
            )

            missed_matches.append({
                "s1_id": s1_id,
                "target_id": target_id,
                "s1_name": s1_data.get("name"),
                "target_name": t_data.get("name"),
                "s1_address": s1_data.get("address"),
                "target_address": t_data.get("address"),
                "was_candidate": bool(feats),
                "model_score": feats.get("score") if feats else None,
            })

    return {
        "false_merges": false_merges,
        "missed_matches": missed_matches,
        "total_false_merges": len(false_merges),
        "total_missed_matches": len(missed_matches),
    }
