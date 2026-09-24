"""Unit tests for entity-level evaluation metrics."""

import pytest
from evaluation.metrics import compute_entity_f_beta, evaluate_entity_resolution


def test_compute_entity_f_beta_perfect_singleton():
    """Predicting empty for an empty true set earns F_0.5 = 1.0."""
    prec, rec, f_beta, fp, fn = compute_entity_f_beta(set(), set(), beta=0.5)
    assert prec == 1.0
    assert rec == 1.0
    assert f_beta == 1.0
    assert fp == 0
    assert fn == 0


def test_compute_entity_f_beta_false_merge_on_singleton():
    """Predicting a match for an empty true set incurs precision 0.0 and false merges."""
    prec, rec, f_beta, fp, fn = compute_entity_f_beta(set(), {"S2_1"}, beta=0.5)
    assert prec == 0.0
    assert f_beta == 0.0
    assert fp == 1
    assert fn == 0


def test_compute_entity_f_beta_missed_match():
    """Predicting empty for a non-empty true set yields F_0.5 = 0.0 and missed matches."""
    prec, rec, f_beta, fp, fn = compute_entity_f_beta({"S2_1"}, set(), beta=0.5)
    assert prec == 0.0
    assert rec == 0.0
    assert f_beta == 0.0
    assert fp == 0
    assert fn == 1


def test_compute_entity_f_beta_partial_match():
    """Test partial match F_0.5 calculation weighting precision over recall."""
    true_set = {"S2_1", "S3_1"}
    pred_set = {"S2_1", "S2_2"}  # 1 TP, 1 FP, 1 FN

    prec, rec, f_beta, fp, fn = compute_entity_f_beta(true_set, pred_set, beta=0.5)
    assert prec == 0.5
    assert rec == 0.5
    assert fp == 1
    assert fn == 1
    # F_0.5 with P=0.5, R=0.5:
    # (1.25 * 0.25) / (0.25 * 0.5 + 0.5) = 0.3125 / 0.625 = 0.5
    assert pytest.approx(f_beta, 1e-4) == 0.5


def test_evaluate_entity_resolution_aggregate():
    """Test full evaluation pipeline on a multi-entity synthetic set."""
    ground_truth = {
        "S1_1": [],
        "S1_2": ["S2_10"],
        "S1_3": ["S2_20", "S3_30"],
    }
    # S1_1 correctly empty
    # S1_2 correctly S2_10
    # S1_3 only predicted S2_20 (missed S3_30: P=1.0, R=0.5)
    predictions = {
        "S1_1": [],
        "S1_2": ["S2_10"],
        "S1_3": ["S2_20"],
    }

    metrics = evaluate_entity_resolution(ground_truth, predictions)
    assert metrics["singleton_accuracy"] == 1.0
    assert metrics["exact_set_accuracy"] == 2 / 3
    assert metrics["false_merges_per_100_s1"] == 0.0
    assert metrics["missed_matches_per_100_s1"] == (1 / 3) * 100.0

    # S1_3: P=1.0, R=0.5 => F_0.5 = (1.25 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.83333
    # S1_1: 1.0, S1_2: 1.0, S1_3: 0.83333 => Mean: (1.0 + 1.0 + 0.83333) / 3 = 0.94444
    assert pytest.approx(metrics["entity_macro_f05"], 1e-4) == (2.0 + 0.625 / 0.75) / 3
