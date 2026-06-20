"""
scripts/build_paragraph_labels.py

Build paragraph-level soft labels from M4 training data.
AI documents: paragraph label = 0.9
Human documents: paragraph label = 0.1
Filter out paragraphs with < min_words words.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PARA_RE = re.compile(r"\n\s*\n+")

DEFAULT_TRAIN = Path(
    "/inspire/hdd/project/fdu-aidake-cfff/public/wangyanqing/NLPPJ"
    "/data/processed/semeval/semeval_train_full.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output_file", type=Path,
                        default=Path("data/paragraph_labels.jsonl"))
    parser.add_argument("--min_words", type=int, default=15)
    parser.add_argument("--ai_label", type=float, default=0.9)
    parser.add_argument("--human_label", type=float, default=0.1)
    args = parser.parse_args()

    print(f"Loading {args.train_data}...")
    df = pd.read_csv(args.train_data, encoding="utf-8-sig")
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    print(f"Loaded {len(df)} documents")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    total_kept = 0
    total_skipped = 0

    with open(args.output_file, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            doc_id = str(row["id"])
            doc_label = int(row["label"])
            soft_label = args.ai_label if doc_label == 1 else args.human_label
            text = str(row["text"])

            paras = [p.strip() for p in PARA_RE.split(text) if p.strip()]
            if not paras:
                paras = [text.strip()]

            for para_idx, para_text in enumerate(paras):
                word_count = len(para_text.split())
                if word_count < args.min_words:
                    total_skipped += 1
                    continue
                record = {
                    "doc_id": doc_id,
                    "para_idx": para_idx,
                    "para_text": para_text,
                    "doc_label": doc_label,
                    "soft_label": soft_label,
                    "word_count": word_count,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_kept += 1

    print(f"Kept {total_kept} paragraphs, skipped {total_skipped} (< {args.min_words} words)")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
