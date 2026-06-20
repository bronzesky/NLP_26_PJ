from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PARAGRAPH_RE = re.compile(r"\n\s*\n+")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*")


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in PARAGRAPH_RE.split(text) if p.strip()]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.findall(text) if s.strip()]


def split_spans(text: str, granularity: str) -> list[str]:
    if granularity == "paragraph":
        parts = split_paragraphs(text)
        return parts if len(parts) > 1 else split_sentences(text)
    return split_sentences(text)


def build_mixed_doc(
    base_spans: list[str],
    base_label: int,
    insert_pool: list[str],
    insert_label: int,
    insert_ratio: float,
    rng: random.Random,
) -> tuple[str, list[dict]]:
    n = len(base_spans)
    n_insert = max(1, round(n * insert_ratio))
    insert_positions = sorted(rng.sample(range(n), min(n_insert, n)))
    insert_sources = [rng.choice(insert_pool) for _ in insert_positions]

    span_records = []
    result_spans: list[str] = []
    pos_set = set(insert_positions)
    insert_iter = iter(zip(insert_positions, insert_sources))
    next_insert = next(insert_iter, None)

    for i, span in enumerate(base_spans):
        if next_insert is not None and next_insert[0] == i:
            ins_text = next_insert[1]
            result_spans.append(ins_text)
            span_records.append({"span_text": ins_text, "span_label": insert_label, "replaced_index": i})
            next_insert = next(insert_iter, None)
        else:
            result_spans.append(span)
            span_records.append({"span_text": span, "span_label": base_label, "replaced_index": i})

    return "\n\n".join(result_spans), span_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("data/mixed"))
    parser.add_argument("--n_docs", type=int, default=5000)
    parser.add_argument("--insert_ratio_min", type=float, default=0.1)
    parser.add_argument("--insert_ratio_max", type=float, default=0.4)
    parser.add_argument("--granularity", default="paragraph,sentence")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    granularities = [g.strip() for g in args.granularity.split(",")]

    df = pd.read_csv(args.train_data, encoding="utf-8-sig")
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    human_texts = df[df["label"] == 0]["text"].tolist()
    ai_texts = df[df["label"] == 1]["text"].tolist()

    if not human_texts or not ai_texts:
        raise ValueError("Need both human (label=0) and AI (label=1) examples in train_data.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    mixed_docs_rows = []
    span_label_rows = []
    doc_id = 0

    # half AI-base, half human-base
    n_ai_base = args.n_docs // 2
    n_human_base = args.n_docs - n_ai_base

    def make_batch(base_pool, base_label, insert_pool, insert_label, n):
        nonlocal doc_id
        for _ in range(n):
            base_text = rng.choice(base_pool)
            gran = rng.choice(granularities)
            base_spans = split_spans(base_text, gran)
            if len(base_spans) < 2:
                base_spans = [base_text]
            # pick insert spans from a separate source doc
            insert_text = rng.choice(insert_pool)
            insert_spans = split_spans(insert_text, gran)
            insert_ratio = rng.uniform(args.insert_ratio_min, args.insert_ratio_max)
            mixed_text, span_records = build_mixed_doc(
                base_spans, base_label, insert_spans, insert_label, insert_ratio, rng
            )
            n_inserted = sum(1 for s in span_records if s["span_label"] == insert_label)
            ai_ratio = sum(1 for s in span_records if s["span_label"] == 1) / max(len(span_records), 1)
            doc_label = 1 if ai_ratio >= 0.5 else 0
            is_mixed = n_inserted > 0
            mixed_docs_rows.append({
                "id": f"mixed_{doc_id}",
                "mixed_text": mixed_text,
                "document_label": doc_label,
                "is_mixed": is_mixed,
                "base_origin": "ai" if base_label == 1 else "human",
                "inserted_origin": "human" if base_label == 1 else "ai",
                "granularity": gran,
                "n_spans": len(span_records),
                "n_inserted": n_inserted,
            })
            char_offset = 0
            separator = "\n\n"
            accumulated_text = ""
            for span_idx, rec in enumerate(span_records):
                span_text = rec["span_text"]
                if span_idx > 0:
                    accumulated_text += separator
                    char_offset = len(accumulated_text)
                span_start = len(accumulated_text)
                accumulated_text += span_text
                span_end = len(accumulated_text)
                span_label_rows.append({
                    "doc_id": f"mixed_{doc_id}",
                    "span_index": span_idx,
                    "span_start": span_start,
                    "span_end": span_end,
                    "span_text": span_text,
                    "span_label": rec["span_label"],
                    "granularity": gran,
                })
            doc_id += 1

    make_batch(ai_texts, 1, human_texts, 0, n_ai_base)
    make_batch(human_texts, 0, ai_texts, 1, n_human_base)

    mixed_docs = pd.DataFrame(mixed_docs_rows)
    span_labels = pd.DataFrame(span_label_rows)

    mixed_docs.to_csv(args.output_dir / "mixed_docs.csv", index=False, quoting=csv.QUOTE_MINIMAL, escapechar="\\")
    span_labels.to_csv(args.output_dir / "span_labels.csv", index=False, quoting=csv.QUOTE_MINIMAL, escapechar="\\")

    print(json.dumps({
        "n_docs": len(mixed_docs),
        "n_spans": len(span_labels),
        "human_spans": int((span_labels["span_label"] == 0).sum()),
        "ai_spans": int((span_labels["span_label"] == 1).sum()),
        "mixed_docs_path": str(args.output_dir / "mixed_docs.csv"),
        "span_labels_path": str(args.output_dir / "span_labels.csv"),
    }, indent=2))


if __name__ == "__main__":
    main()
