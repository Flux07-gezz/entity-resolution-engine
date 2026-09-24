"""End-to-end entity resolution pipeline.

Orchestrates:
Data Loading -> Normalization -> Address Parsing -> Blocking -> Scoring -> Decision -> Submission Output & Validation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import csv
import pandas as pd
import yaml

from evaluation.logger import ExperimentTracker
from evaluation.metrics import evaluate_entity_resolution, format_dashboard
from src.address_parser import AddressParser
from src.blocking import CandidateBlocker, evaluate_blocking
from src.decision import EntityDecisionEngine
from src.model import BaselineScorer
from src.normalize import normalize_address, normalize_name
from utils.submission import SubmissionWriter
from utils.validation import SubmissionValidator


def load_tsv_records(
    file_path: Union[str, Path],
    id_col: str,
    name_col: str = "name",
    addr_col: str = "address",
    source_tag: str = "s1",
) -> Dict[str, Dict[str, Any]]:
    """Load TSV file into dictionary of record metadata."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    records: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rec_id = row[id_col].strip()
            raw_name = row.get(name_col, "").strip()
            raw_addr = row.get(addr_col, "").strip()

            records[rec_id] = {
                "id": rec_id,
                "source": source_tag,
                "raw_name": raw_name,
                "raw_address": raw_addr,
                "name": raw_name,
                "address": raw_addr,
            }
    return records


def load_ground_truth(file_path: Union[str, Path]) -> Dict[str, List[str]]:
    """Load train_ground_truth.tsv into a mapping of s1_id -> [matched_ids]."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    gt: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # Identify column names
        s1_col = reader.fieldnames[0]
        match_col = reader.fieldnames[1] if len(reader.fieldnames) > 1 else None

        for row in reader:
            s1_id = row[s1_col].strip()
            match_str = row.get(match_col, "").strip() if match_col else ""
            if match_str:
                matches = [m.strip() for m in match_str.split(",") if m.strip()]
            else:
                matches = []
            gt[s1_id] = sorted(list(set(matches)))
    return gt


class EntityResolutionPipeline:
    """Configurable pipeline executing entity resolution end-to-end."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)

        self.address_parser = AddressParser()
        self.blocker = CandidateBlocker(
            max_candidates_per_s1=self.config.get("blocking", {}).get("max_candidates_per_s1", 50)
        )
        self.scorer = BaselineScorer()
        self.decision_engine = EntityDecisionEngine(
            default_threshold=self.config.get("decision", {}).get("default_threshold", 0.75)
        )
        self.writer = SubmissionWriter()
        self.tracker = ExperimentTracker()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        path = Path(config_path)
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def preprocess_records(self, records: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Normalize names, addresses, and extract parsed address attributes."""
        processed = {}
        for rid, rec in records.items():
            norm_name = normalize_name(rec["raw_name"])
            norm_addr = normalize_address(rec["raw_address"])
            parsed = self.address_parser.parse(norm_addr)

            processed[rid] = {
                **rec,
                "name": norm_name,
                "address": norm_addr,
                "house_number": parsed["house_number"],
                "unit": parsed["unit"],
                "street": parsed["street"],
                "city": parsed["city"],
                "state": parsed["state"],
                "postal_code": parsed["postal_code"],
            }
        return processed

    def run(
        self,
        s1_file: Union[str, Path],
        s2_file: Union[str, Path],
        s3_file: Union[str, Path],
        output_file: Union[str, Path] = "output/matching_results.tsv",
        gt_file: Optional[Union[str, Path]] = None,
        experiment_id: str = "run_baseline",
        description: str = "End-to-end baseline pipeline run",
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute the full pipeline."""
        print(f"[*] Loading data: S1={s1_file}, S2={s2_file}, S3={s3_file}")
        s1_records = load_tsv_records(s1_file, id_col="source_1_id", source_tag="s1")
        s2_records = load_tsv_records(s2_file, id_col="source_2_id", source_tag="s2")
        s3_records = load_tsv_records(s3_file, id_col="source_3_id", source_tag="s3")

        all_s1_ids = list(s1_records.keys())

        # Preprocess
        print(f"[*] Preprocessing records (normalization & address parsing)...")
        s1_proc = self.preprocess_records(s1_records)
        s2_proc = self.preprocess_records(s2_records)
        s3_proc = self.preprocess_records(s3_records)

        # Combine target records
        target_proc = {**s2_proc, **s3_proc}

        # Blocking / Candidate Generation
        print(f"[*] Generating candidate pairs...")
        candidate_pairs = self.blocker.generate_candidate_pairs(s1_proc, target_proc)
        print(f"    Total candidate pairs generated: {len(candidate_pairs)}")

        blocking_metrics = {}
        ground_truth = None
        if gt_file and Path(gt_file).is_file():
            ground_truth = load_ground_truth(gt_file)
            blocking_metrics = evaluate_blocking(candidate_pairs, ground_truth, all_s1_ids)
            print(f"    Candidate Recall: {blocking_metrics['candidate_recall'] * 100:.2f}%")
            print(f"    Avg Candidates/S1: {blocking_metrics['avg_candidates_per_s1']:.2f}")

        # Scoring
        print(f"[*] Scoring candidate pairs...")
        scored_pairs = self.scorer.score_pairs(candidate_pairs, s1_proc, target_proc)

        # Entity Decision
        print(f"[*] Resolving entity match sets...")
        preds = self.decision_engine.decide_entities(
            scored_pairs, all_s1_ids, threshold=threshold
        )

        # Write Submission
        print(f"[*] Writing submission file to {output_file}...")
        out_path = self.writer.write_tsv(preds, all_s1_ids, output_file)

        # Validate Submission File
        print(f"[*] Validating submission file...")
        validator = SubmissionValidator(
            expected_s1_ids=set(all_s1_ids),
            valid_target_ids=set(target_proc.keys()),
        )
        is_valid, val_errors, val_stats = validator.validate_file(out_path)
        if not is_valid:
            print(f"[!] Validation WARNING: {val_errors}")
        else:
            print(f"[+] Submission file is VALID! Rows={val_stats['total_rows']}, Matches={val_stats['total_matches_predicted']}")

        # Evaluation & Dashboard
        dashboard_metrics = {}
        if ground_truth is not None:
            eval_metrics = evaluate_entity_resolution(ground_truth, preds, all_s1_ids)
            dashboard_metrics = {**eval_metrics, **blocking_metrics}
            print("\n" + format_dashboard(dashboard_metrics, title=experiment_id) + "\n")

            # Persist experiment
            self.tracker.log_experiment(
                experiment_id=experiment_id,
                description=description,
                metrics=dashboard_metrics,
                config={"threshold": threshold or self.decision_engine.default_threshold},
            )

        return {
            "predictions": preds,
            "output_file": str(out_path),
            "is_valid": is_valid,
            "validation_errors": val_errors,
            "metrics": dashboard_metrics,
        }
