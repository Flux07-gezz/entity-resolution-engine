"""Scoring and ML models for candidate pair matching.

Includes:
- BaselineScorer: Conservative heuristic similarity scorer (Phase 3 baseline).
- GBDTModel: LightGBM / CatBoost pair classification model (Phase 11).
"""

from typing import Any, Dict, List, Optional
import numpy as np
from src.features import extract_pair_features

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class BaselineScorer:
    """Trivial, conservative baseline scorer using exact/fuzzy name and address similarity."""

    def __init__(self, name_weight: float = 0.6, addr_weight: float = 0.4):
        self.name_weight = name_weight
        self.addr_weight = addr_weight

    def score_pairs(
        self,
        candidate_pairs: List[Dict[str, Any]],
        s1_records: Dict[str, Dict[str, Any]],
        target_records: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compute baseline similarity score for each candidate pair."""
        scored_pairs = []

        for pair in candidate_pairs:
            s1_id = pair["s1_id"]
            target_id = pair["target_id"]
            s1_rec = s1_records[s1_id]
            t_rec = target_records[target_id]

            feats = extract_pair_features(
                s1_rec,
                t_rec,
                blocking_passes=pair.get("blocking_passes"),
                target_source=pair.get("target_source", "s2"),
            )

            name_sim = feats.get("name_token_sort_ratio", 0.0)
            addr_sim = feats.get("addr_token_sort_ratio", 0.0)

            # Contradiction penalty
            penalty = 0.0
            if feats.get("house_num_conflict", 0.0) == 1.0:
                penalty += 0.5
            if feats.get("city_conflict", 0.0) == 1.0:
                penalty += 0.3

            score = (self.name_weight * name_sim + self.addr_weight * addr_sim) - penalty
            score = max(0.0, min(1.0, score))

            scored_pairs.append({
                "s1_id": s1_id,
                "target_id": target_id,
                "score": float(score),
                "target_source": pair.get("target_source", "unknown"),
                "features": feats,
            })

        return scored_pairs


class PairClassifier:
    """Gradient boosted tree classifier for candidate pair probability estimation."""

    def __init__(self, model_type: str = "lightgbm", params: Optional[Dict[str, Any]] = None):
        self.model_type = model_type
        self.params = params or {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42,
            "verbose": -1,
        }
        self.model = None
        self.feature_names: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """Fit classifier on pair evidence vectors."""
        self.feature_names = feature_names or []
        if self.model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMClassifier(**self.params)
            self.model.fit(X, y)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(random_state=42)
            self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict match probabilities."""
        if self.model is None:
            raise ValueError("Model is not fitted.")
        return self.model.predict_proba(X)[:, 1]
