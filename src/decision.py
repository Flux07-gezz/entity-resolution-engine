"""Entity-level decision and match resolution module.

Converts pairwise match probabilities into entity match sets for each Source 1 entity:
- Thresholding with empirical optimization
- Hard contradiction filtering (e.g., severe house number conflict)
- Deterministic deduplicated sorting
- Proper singleton handling (empty match list)
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


class EntityDecisionEngine:
    """Decision engine converting pairwise scores into final entity predictions."""

    def __init__(
        self,
        default_threshold: float = 0.75,
        enable_hard_contradiction_filter: bool = True,
    ):
        self.default_threshold = default_threshold
        self.enable_hard_contradiction_filter = enable_hard_contradiction_filter

    def decide_entities(
        self,
        scored_pairs: List[Dict[str, Any]],
        all_s1_ids: Sequence[str],
        threshold: Optional[float] = None,
        source_thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, List[str]]:
        """Produce final match set for every S1 entity.

        Args:
            scored_pairs: List of dicts with:
                - 's1_id': str
                - 'target_id': str
                - 'score': float (predicted probability or similarity)
                - 'target_source': 's2' | 's3' (optional)
                - 'features': Dict[str, float] (optional)
            all_s1_ids: All expected Source 1 entity IDs
            threshold: Global threshold (overrides default_threshold)
            source_thresholds: Optional source-specific threshold dict e.g. {'s2': 0.8, 's3': 0.75}

        Returns:
            Dict mapping s1_id -> list of matched target IDs
        """
        thresh = threshold if threshold is not None else self.default_threshold

        # Initialize every S1 entity with an empty list
        predictions: Dict[str, List[str]] = {s1_id: [] for s1_id in all_s1_ids}

        for pair in scored_pairs:
            s1_id = pair["s1_id"]
            target_id = pair["target_id"]
            score = float(pair.get("score", 0.0))
            target_src = pair.get("target_source", "unknown")

            # Determine applicable threshold
            cutoff = (
                source_thresholds.get(target_src, thresh)
                if source_thresholds
                else thresh
            )

            if score < cutoff:
                continue

            # Hard contradiction checks (if features present)
            feats = pair.get("features", {})
            if self.enable_hard_contradiction_filter and feats:
                # Severe house number conflict (both present and completely conflicting)
                if feats.get("house_num_conflict", 0.0) == 1.0:
                    continue

            predictions[s1_id].append(target_id)

        # Deduplicate and sort deterministically
        for s1_id in all_s1_ids:
            matches = predictions[s1_id]
            clean = sorted(list(set(matches)))
            predictions[s1_id] = clean

        return predictions
