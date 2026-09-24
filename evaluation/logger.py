"""Experiment logging and tracking system.

Appends every experiment to evaluation/experiments.jsonl and regenerates evaluation/experiments.md.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import subprocess


def get_git_commit_hash() -> str:
    """Retrieve current git HEAD commit hash if in a git repo."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "uncommitted"


class ExperimentTracker:
    """Tracks and persists experiment runs according to Section 23 and 31.3 specifications."""

    def __init__(
        self,
        log_path: str = "evaluation/experiments.jsonl",
        summary_path: str = "evaluation/experiments.md",
    ):
        self.log_path = Path(log_path)
        self.summary_path = Path(summary_path)

    def log_experiment(
        self,
        experiment_id: str,
        description: str,
        metrics: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record an experiment entry and regenerate summary markdown."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        commit = get_git_commit_hash()
        timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "git_commit": commit,
            "description": description,
            "metrics": metrics,
            "config": config or {},
        }

        # Append to experiments.jsonl
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Regenerate experiments.md
        self.regenerate_summary_markdown()
        return record

    def regenerate_summary_markdown(self):
        """Regenerate evaluation/experiments.md table from experiments.jsonl."""
        if not self.log_path.is_file():
            return

        records = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        lines = [
            "# Experiment Tracking Summary",
            "",
            "All experiment runs persisted for reproducible comparison and audit.",
            "",
            "| ID | Timestamp (UTC) | Commit | Description | Entity Macro F0.5 | Pair Prec | Pair Rec | Cand Rec | Singleton Acc | Exact Set Acc | False Merges / 100 S1 |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]

        for r in records:
            m = r.get("metrics", {})
            f05 = f"{m.get('entity_macro_f05', 0.0) * 100:.2f}%"
            pp = f"{m.get('pair_precision', 0.0) * 100:.2f}%"
            pr = f"{m.get('pair_recall', 0.0) * 100:.2f}%"
            cr = f"{m.get('candidate_recall', 0.0) * 100:.2f}%" if "candidate_recall" in m else "N/A"
            sa = f"{m.get('singleton_accuracy', 0.0) * 100:.2f}%"
            esa = f"{m.get('exact_set_accuracy', 0.0) * 100:.2f}%"
            fm = f"{m.get('false_merges_per_100_s1', 0.0):.2f}"
            commit = r.get("git_commit", "-")
            ts = r.get("timestamp", "")[:19].replace("T", " ")
            eid = r.get("experiment_id", "")
            desc = r.get("description", "").replace("|", "-")

            lines.append(
                f"| {eid} | {ts} | `{commit}` | {desc} | **{f05}** | {pp} | {pr} | {cr} | {sa} | {esa} | {fm} |"
            )

        lines.append("")

        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
