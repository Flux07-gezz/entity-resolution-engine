"""Validation module for submission files."""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import csv
import os


class SubmissionValidator:
    """Validates submission files against challenge specifications."""

    def __init__(
        self,
        expected_s1_ids: Optional[Set[str]] = None,
        valid_target_ids: Optional[Set[str]] = None,
        delimiter: str = "\t",
        match_separator: str = ",",
    ):
        self.expected_s1_ids = set(expected_s1_ids) if expected_s1_ids is not None else None
        self.valid_target_ids = set(valid_target_ids) if valid_target_ids is not None else None
        self.delimiter = delimiter
        self.match_separator = match_separator

    def validate_file(self, file_path: Union[str, Path]) -> Tuple[bool, List[str], Dict[str, Union[int, float]]]:
        """Validate a submission file and return (is_valid, error_list, summary_stats)."""
        path = Path(file_path)
        errors: List[str] = []
        stats: Dict[str, Union[int, float]] = {
            "total_rows": 0,
            "singletons": 0,
            "single_match": 0,
            "multi_match": 0,
            "total_matches_predicted": 0,
            "max_matches": 0,
        }

        if not path.is_file():
            errors.append(f"File not found: {path}")
            return False, errors, stats

        seen_s1: Set[str] = set()

        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=self.delimiter)
                try:
                    header = next(reader)
                except StopIteration:
                    errors.append("File is empty.")
                    return False, errors, stats

                if len(header) < 2:
                    errors.append(f"Expected at least 2 columns in header, got: {header}")
                expected_headers = [
                    ["source1_entity_id", "matched_entity_ids"],
                    ["source1_entity_id", "candidate_entity_ids"],
                ]
                norm_header = [c.strip().lower() for c in header]
                if norm_header not in expected_headers:
                    errors.append(
                        f"Unexpected header {norm_header}. Expected exactly ['source1_entity_id', 'matched_entity_ids'] or ['source1_entity_id', 'candidate_entity_ids']."
                    )

                for line_idx, row in enumerate(reader, start=2):
                    if not row:
                        continue
                    if len(row) < 2:
                        errors.append(f"Line {line_idx}: Row has fewer than 2 columns: {row}")
                        continue

                    s1_id = row[0].strip()
                    match_str = row[1].strip()

                    if not s1_id:
                        errors.append(f"Line {line_idx}: Empty Source 1 ID.")
                        continue

                    if s1_id in seen_s1:
                        errors.append(f"Line {line_idx}: Duplicate Source 1 ID '{s1_id}'.")
                    seen_s1.add(s1_id)

                    matches = [m.strip() for m in match_str.split(self.match_separator)] if match_str else []
                    matches = [m for m in matches if m]

                    # Check for internal duplicates in matches
                    if len(matches) != len(set(matches)):
                        errors.append(f"Line {line_idx} ({s1_id}): Duplicate matches found: {matches}")

                    # Check if S1 id is in matches
                    if s1_id in matches:
                        errors.append(f"Line {line_idx} ({s1_id}): Source 1 ID cannot be its own match.")

                    # Check valid target IDs if provided
                    if self.valid_target_ids:
                        invalid_targets = [m for m in matches if m not in self.valid_target_ids]
                        if invalid_targets:
                            errors.append(
                                f"Line {line_idx} ({s1_id}): Invalid target IDs {invalid_targets[:3]}"
                            )

                    # Update stats
                    match_count = len(matches)
                    stats["total_matches_predicted"] += match_count
                    if match_count == 0:
                        stats["singletons"] += 1
                    elif match_count == 1:
                        stats["single_match"] += 1
                    else:
                        stats["multi_match"] += 1
                    if match_count > stats["max_matches"]:
                        stats["max_matches"] = match_count

            stats["total_rows"] = len(seen_s1)

            # Check expected S1 coverage if known
            if self.expected_s1_ids is not None:
                missing_s1 = self.expected_s1_ids - seen_s1
                if missing_s1:
                    errors.append(
                        f"Missing {len(missing_s1)} expected Source 1 IDs (e.g. {list(missing_s1)[:5]})"
                    )
                unexpected_s1 = seen_s1 - self.expected_s1_ids
                if unexpected_s1:
                    errors.append(
                        f"Found {len(unexpected_s1)} unexpected Source 1 IDs (e.g. {list(unexpected_s1)[:5]})"
                    )

        except Exception as e:
            errors.append(f"Exception during parsing: {str(e)}")

        is_valid = len(errors) == 0
        return is_valid, errors, stats


def run_official_validator(
    matching_path: Union[str, Path] = "output/matching_results.tsv",
    candidate_path: Optional[Union[str, Path]] = "output/candidate_pairs.tsv",
    test_dir: Union[str, Path] = "data/student_resource/dataset/test",
    check_ids: bool = False,
) -> Tuple[int, str]:
    """Run the official organizer validator script (data/student_resource/utils/validate_submission.py)."""
    import subprocess
    import sys

    validator_script = Path("data/student_resource/utils/validate_submission.py")
    if not validator_script.is_file():
        raise FileNotFoundError(f"Official validator not found at: {validator_script}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        sys.executable,
        str(validator_script),
        "--matching", str(matching_path),
        "--test-dir", str(test_dir),
    ]
    if candidate_path and Path(candidate_path).is_file():
        cmd.extend(["--candidate", str(candidate_path)])
    if check_ids:
        cmd.append("--check-ids")

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    return proc.returncode, stdout + stderr


