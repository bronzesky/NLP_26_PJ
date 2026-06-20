from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis import add_buckets, infer_columns, merge_predictions, read_table
from src.features import add_text_features, high_confidence_errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions_file",
        "--predictions",
        dest="predictions_file",
        type=Path,
        required=True,
    )
    parser.add_argument("--data_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--min_confidence", type=float, default=0.9)
    args = parser.parse_args()

    predictions = read_table(args.predictions_file)
    data = read_table(args.data_file) if args.data_file else None
    df = merge_predictions(predictions, data)
    if "text" not in df.columns:
        raise ValueError("features require a text column in predictions_file or merged --data_file.")

    featured = add_text_features(add_buckets(df))
    true_col, pred_col = infer_columns(featured)
    errors = high_confidence_errors(featured, true_col, pred_col, args.min_confidence)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    features_file = args.output_dir / "features.csv"
    errors_file = args.output_dir / "error_cases.csv"
    featured.to_csv(features_file, index=False, quoting=csv.QUOTE_MINIMAL, escapechar="\\")
    errors.to_csv(errors_file, index=False, quoting=csv.QUOTE_MINIMAL, escapechar="\\")
    print(f"Wrote {len(featured)} rows to {features_file}")
    print(f"Wrote {len(errors)} high-confidence errors to {errors_file}")


if __name__ == "__main__":
    main()
