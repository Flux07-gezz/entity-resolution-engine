"""Submission generation and validation utilities for entity resolution.

Ensures exact compliance with submission formatting:
- Exactly one row per Source 1 entity.
- Deterministic, deduplicated matches for Source 2 and Source 3.
- Proper handling of singletons (empty matches).
- No missing or extraneous Source 1 entities.
- Valid TSV format.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Union
import csv
import io


class SubmissionWriter:
    """Format, validate, and write matching_results.tsv."""

    DEFAULT_S1_COL = "source_1_id"
    DEFAULT_MATCH_COL = "matched_ids"
    DELIMITER = "\t"
    MATCH_SEPARATOR = ","

    def __init__(
        self,
        s1_column: str = DEFAULT_S1_COL,
        match_column: str = DEFAULT_MATCH_COL,
        delimiter: str = DELIMITER,
        match_separator: str = MATCH_SEPARATOR,
    ):
        self.s1_column = s1_column
        self.match_column = match_column
        self.delimiter = delimiter
        self.match_separator = match_separator

    def clean_and_sort_matches(
        self,
        matches: Union[str, Iterable[str], None],
        s1_id: Optional[str] = None,
    ) -> List[str]:
        """Deduplicate, clean, and deterministically sort matched entity IDs."""
        if matches is None:
            return []

        if isinstance(matches, str):
            raw_items = [item.strip() for item in matches.split(self.match_separator)]
        else:
            raw_items = [str(item).strip() for item in matches]

        seen: Set[str] = set()
        clean_list: List[str] = []

        for item in raw_items:
            if not item:
                continue
            # A source 1 entity cannot be its own match
            if s1_id is not None and item == s1_id:
                continue
            if item not in seen:
                seen.add(item)
                clean_list.append(item)

        # Deterministic sorting
        clean_list.sort()
        return clean_list

    def format_match_string(self, matches: Sequence[str]) -> str:
        """Format list of matched IDs into TSV column string."""
        return self.match_separator.join(matches)

    def prepare_submission_rows(
        self,
        predictions: Dict[str, Union[Sequence[str], str, None]],
        expected_s1_ids: Sequence[str],
    ) -> List[Dict[str, str]]:
        """Validate predictions against expected Source 1 IDs and prepare sanitized rows."""
        expected_set = set(expected_s1_ids)
        pred_set = set(predictions.keys())

        missing = expected_set - pred_set
        if missing:
            raise ValueError(
                f"Missing predictions for {len(missing)} Source 1 IDs (e.g. {list(missing)[:5]})"
            )

        extra = pred_set - expected_set
        if extra:
            raise ValueError(
                f"Extraneous {len(extra)} IDs found in predictions not in expected Source 1 set (e.g. {list(extra)[:5]})"
            )

        rows = []
        for s1_id in expected_s1_ids:
            raw_matches = predictions.get(s1_id, [])
            clean_matches = self.clean_and_sort_matches(raw_matches, s1_id=s1_id)
            rows.append({
                self.s1_column: str(s1_id).strip(),
                self.match_column: self.format_match_string(clean_matches),
            })
        return rows

    def write_tsv(
        self,
        predictions: Dict[str, Union[Sequence[str], str, None]],
        expected_s1_ids: Sequence[str],
        output_path: Union[str, Path],
    ) -> Path:
        """Write predictions to TSV file, guaranteeing one row per Source 1 entity."""
        rows = self.prepare_submission_rows(predictions, expected_s1_ids)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[self.s1_column, self.match_column],
                delimiter=self.delimiter,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

        return path

    def read_tsv(self, input_path: Union[str, Path]) -> Dict[str, List[str]]:
        """Read and parse matching_results.tsv back into a dictionary of lists."""
        path = Path(input_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        results: Dict[str, List[str]] = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)
            if reader.fieldnames is None:
                raise ValueError("Empty or invalid TSV file.")
            
            # Allow flexible headers if custom
            s1_col = self.s1_column if self.s1_column in reader.fieldnames else reader.fieldnames[0]
            match_col = self.match_column if self.match_column in reader.fieldnames else (
                reader.fieldnames[1] if len(reader.fieldnames) > 1 else None
            )

            for line_no, row in enumerate(reader, start=2):
                s1_id = row.get(s1_col)
                if not s1_id:
                    raise ValueError(f"Line {line_no}: Empty Source 1 ID")
                s1_id = s1_id.strip()
                if s1_id in results:
                    raise ValueError(f"Line {line_no}: Duplicate Source 1 ID '{s1_id}'")

                match_val = row.get(match_col, "") if match_col else ""
                clean_matches = self.clean_and_sort_matches(match_val, s1_id=s1_id)
                results[s1_id] = clean_matches

        return results
