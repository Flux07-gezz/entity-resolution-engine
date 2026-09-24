"""CLI entrypoint to run prediction pipeline.

Usage:
    python src/predict.py --s1 data/synthetic/source_1.tsv --s2 data/synthetic/source_2.tsv --s3 data/synthetic/source_3.tsv --gt data/synthetic/train_ground_truth.tsv --out output/matching_results.tsv
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import EntityResolutionPipeline


def main():
    parser = argparse.ArgumentParser(description="Run Entity Resolution Pipeline")
    parser.add_argument("--s1", required=True, help="Path to Source 1 TSV")
    parser.add_argument("--s2", required=True, help="Path to Source 2 TSV")
    parser.add_argument("--s3", required=True, help="Path to Source 3 TSV")
    parser.add_argument("--gt", default=None, help="Optional path to train_ground_truth.tsv")
    parser.add_argument("--out", default="output/matching_results.tsv", help="Path to output TSV")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold")
    parser.add_argument("--exp-id", default="run_baseline", help="Experiment identifier")
    parser.add_argument("--desc", default="Baseline pipeline run", help="Experiment description")
    args = parser.parse_args()

    pipeline = EntityResolutionPipeline()
    result = pipeline.run(
        s1_file=args.s1,
        s2_file=args.s2,
        s3_file=args.s3,
        output_file=args.out,
        gt_file=args.gt,
        experiment_id=args.exp_id,
        description=args.desc,
        threshold=args.threshold,
    )

    if not result["is_valid"]:
        print("Pipeline produced an invalid submission file!", file=sys.stderr)
        sys.exit(1)

    print(f"Pipeline executed successfully. Output written to {result['output_file']}")


if __name__ == "__main__":
    main()
