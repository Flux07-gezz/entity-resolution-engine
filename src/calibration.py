"""Probability calibration module for entity matching.

Implements Platt scaling (sigmoid) and isotonic regression on out-of-fold predictions.
"""

from typing import Optional
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ScoreCalibrator:
    """Calibrates pairwise raw scores / probabilities against empirical true match rates."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.calibrator = None

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray):
        """Fit calibration curve on validation/OOF predictions."""
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(raw_scores, y_true)
        elif self.method == "sigmoid":
            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            self.calibrator.fit(raw_scores.reshape(-1, 1), y_true)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        """Apply calibration mapping."""
        if self.calibrator is None:
            return raw_scores

        if self.method == "isotonic":
            return self.calibrator.transform(raw_scores)
        else:
            return self.calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
