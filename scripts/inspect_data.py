import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


DEFAULT_DATA_DIR = Path(
    "/inspire/hdd/project/fdu-aidake-cfff/public/wangyanqing/NLPPJ/data/processed/semeval"
)


def inspect_csv(path: Path) -> dict:
    csv.field_size_limit(sys.maxsize)
    labels = Counter()
    models = Counter()
    sources = Counter()
    lengths = []
    first_rows = []

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for i, row in enumerate(reader):
            if i < 3:
                first_rows.append(
                    {
                        key: value[:200].replace("\n", " ")
                        if isinstance(value, str)
                        else value
                        for key, value in row.items()
                    }
                )
            labels[row.get("label", "")] += 1
            models[row.get("model", "")] += 1
            sources[row.get("source", "")] += 1
            lengths.append(len(row.get("text", "")))

    sorted_lengths = sorted(lengths)

    def pct(q: float) -> int:
        if not sorted_lengths:
            return 0
        return sorted_lengths[int((len(sorted_lengths) - 1) * q)]

    return {
        "path": str(path),
        "columns": fieldnames,
        "num_rows": len(lengths),
        "labels": labels.most_common(),
        "models": models.most_common(),
        "sources": sources.most_common(),
        "text_len": {
            "min": min(lengths) if lengths else 0,
            "p50": pct(0.5),
            "p90": pct(0.9),
            "p99": pct(0.99),
            "max": max(lengths) if lengths else 0,
            "avg": int(sum(lengths) / len(lengths)) if lengths else 0,
        },
        "first_rows": first_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train_file", default="semeval_train_full.csv")
    parser.add_argument("--dev_file", default="semeval_dev_full.csv")
    args = parser.parse_args()

    report = {
        "train": inspect_csv(args.data_dir / args.train_file),
        "dev": inspect_csv(args.data_dir / args.dev_file),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

