from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis import add_buckets, grouped_metrics, infer_columns, merge_predictions, read_table


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
    parser.add_argument("--output_file", type=Path, default=Path("outputs/analysis/group_metrics.csv"))
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    predictions = read_table(args.predictions_file)
    data = read_table(args.data_file) if args.data_file else None
    df = add_buckets(merge_predictions(predictions, data))
    true_col, pred_col = infer_columns(df)

    group_cols = ["domain", "model", "source", "length_bucket", "confidence_bucket"]
    metrics = grouped_metrics(df, group_cols, true_col, pred_col)
    output_file = args.output_file
    if args.output_dir is not None:
        output_file = args.output_dir / "group_metrics.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_file, index=False)
    for group_col in group_cols:
        grouped_metrics(df, [group_col], true_col, pred_col).to_csv(
            output_file.parent / f"group_by_{group_col}.csv", index=False
        )
    print(f"Wrote {len(metrics)} rows to {output_file}")


if __name__ == "__main__":
    main()
