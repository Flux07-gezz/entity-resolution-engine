"""Unit tests for SubmissionWriter and SubmissionValidator."""

import pytest
import tempfile
from pathlib import Path
from utils.submission import SubmissionWriter
from utils.validation import SubmissionValidator


def test_submission_writer_singletons_and_matches():
    """Test zero matches, one match, and multiple matches."""
    writer = SubmissionWriter()
    expected_s1 = ["S1_001", "S1_002", "S1_003"]
    predictions = {
        "S1_001": [],                    # Zero matches (singleton)
        "S1_002": ["S2_10"],             # One match
        "S1_003": ["S2_20", "S3_30"],    # Multiple matches
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tsv_path = Path(tmpdir) / "matching_results.tsv"
        writer.write_tsv(predictions, expected_s1, tsv_path)

        # Read back with writer
        loaded = writer.read_tsv(tsv_path)
        assert loaded["S1_001"] == []
        assert loaded["S1_002"] == ["S2_10"]
        assert loaded["S1_003"] == ["S2_20", "S3_30"]

        # Validate with SubmissionValidator
        validator = SubmissionValidator(expected_s1_ids=set(expected_s1))
        is_valid, errors, stats = validator.validate_file(tsv_path)
        assert is_valid, f"Validation errors: {errors}"
        assert stats["total_rows"] == 3
        assert stats["singletons"] == 1
        assert stats["single_match"] == 1
        assert stats["multi_match"] == 1
        assert stats["total_matches_predicted"] == 3


def test_submission_writer_duplicate_candidates():
    """Test deduplication of predicted IDs and deterministic ordering."""
    writer = SubmissionWriter()
    expected_s1 = ["S1_001"]
    # Duplicate entries and unsorted order
    predictions = {
        "S1_001": ["S3_99", "S2_10", "S3_99", "S2_10", "S2_05"],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tsv_path = Path(tmpdir) / "matching_results.tsv"
        writer.write_tsv(predictions, expected_s1, tsv_path)

        loaded = writer.read_tsv(tsv_path)
        # Should be deduplicated and sorted
        assert loaded["S1_001"] == ["S2_05", "S2_10", "S3_99"]


def test_submission_writer_missing_s1_ids():
    """Test that missing expected Source 1 IDs raises an error."""
    writer = SubmissionWriter()
    expected_s1 = ["S1_001", "S1_002", "S1_003"]
    # S1_003 is missing from predictions
    predictions = {
        "S1_001": ["S2_01"],
        "S1_002": [],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tsv_path = Path(tmpdir) / "matching_results.tsv"
        with pytest.raises(ValueError, match="Missing predictions for 1 Source 1 IDs"):
            writer.write_tsv(predictions, expected_s1, tsv_path)


def test_submission_writer_extra_s1_ids():
    """Test that extraneous IDs not in expected S1 list raise an error."""
    writer = SubmissionWriter()
    expected_s1 = ["S1_001"]
    predictions = {
        "S1_001": ["S2_01"],
        "S1_999": ["S2_02"],  # Extra
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tsv_path = Path(tmpdir) / "matching_results.tsv"
        with pytest.raises(ValueError, match="Extraneous 1 IDs"):
            writer.write_tsv(predictions, expected_s1, tsv_path)


def test_submission_self_match_removed():
    """Test that a Source 1 ID matching itself is safely removed."""
    writer = SubmissionWriter()
    expected_s1 = ["S1_001"]
    predictions = {
        "S1_001": ["S1_001", "S2_10"],  # S1_001 cannot match itself
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tsv_path = Path(tmpdir) / "matching_results.tsv"
        writer.write_tsv(predictions, expected_s1, tsv_path)

        loaded = writer.read_tsv(tsv_path)
        assert loaded["S1_001"] == ["S2_10"]


def test_submission_validator_malformed():
    """Test validator catches malformed files."""
    validator = SubmissionValidator(expected_s1_ids={"S1_001", "S1_002"})

    with tempfile.TemporaryDirectory() as tmpdir:
        tsv_path = Path(tmpdir) / "bad.tsv"
        # Duplicate S1 row and invalid target
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("source_1_id\tmatched_ids\n")
            f.write("S1_001\tS2_10,S2_10\n")  # duplicate match
            f.write("S1_001\tS2_20\n")         # duplicate S1 row

        is_valid, errors, stats = validator.validate_file(tsv_path)
        assert not is_valid
        assert any("Duplicate matches found" in err for err in errors)
        assert any("Duplicate Source 1 ID" in err for err in errors)
