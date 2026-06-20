"""
scripts/build_hc3_mixed.py

Construct same-topic human-AI mixed documents from HC3.
Each QA pair provides same-question human and ChatGPT answers.
Sentences are swapped at the sentence level with length matching,
producing span-level labels (0=human, 1=AI).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.findall(text) if len(s.strip().split()) >= 3]


def build_mixed_doc(
    base_sents: list[str],
    base_label: int,
    insert_pool: list[str],
    insert_label: int,
    insert_ratio: float,
    rng: random.Random,
) -> tuple[str, list[dict], int]:
    """
    Returns (mixed_text, span_records, doc_label).
    doc_label = 1 if AI content >= 50%, else 0.
    """
    n = len(base_sents)
    if n == 0:
        return "", [], base_label

    n_insert = max(1, round(n * insert_ratio))
    n_insert = min(n_insert, len(insert_pool), n)
    if n_insert == 0:
        return " ".join(base_sents), [
            {"span_index": i, "span_text": s, "span_label": base_label}
            for i, s in enumerate(base_sents)
        ], base_label

    # Length matching: only use insert sentences with word count in [0.5×, 2×] of base mean
    base_mean_len = sum(len(s.split()) for s in base_sents) / len(base_sents)
    lo, hi = base_mean_len * 0.4, base_mean_len * 2.5
    compatible = [s for s in insert_pool if lo <= len(s.split()) <= hi]
    if not compatible:
        compatible = insert_pool  # fallback: no filter

    insert_positions = sorted(rng.sample(range(n), n_insert))
    insert_sents = [rng.choice(compatible) for _ in insert_positions]
    pos_set = set(insert_positions)

    result_sents: list[str] = []
    span_records: list[dict] = []
    insert_iter = iter(zip(insert_positions, insert_sents))
    next_ins = next(insert_iter, None)

    for i, sent in enumerate(base_sents):
        if next_ins is not None and next_ins[0] == i:
            ins_text = next_ins[1]
            result_sents.append(ins_text)
            span_records.append({
                "span_index": len(result_sents) - 1,
                "span_text": ins_text,
                "span_label": insert_label,
            })
            next_ins = next(insert_iter, None)
        result_sents.append(sent)
        span_records.append({
            "span_index": len(result_sents) - 1,
            "span_text": sent,
            "span_label": base_label,
        })

    mixed_text = " ".join(result_sents)
    ai_count = sum(1 for r in span_records if r["span_label"] == 1)
    doc_label = 1 if ai_count / len(span_records) >= 0.5 else 0
    return mixed_text, span_records, doc_label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hc3_file", type=Path,
                        default=Path("data/hc3/all.jsonl"))
    parser.add_argument("--output_dir", type=Path,
                        default=Path("data/hc3_mixed"))
    parser.add_argument("--max_docs", type=int, default=5000)
    parser.add_argument("--insert_ratio_min", type=float, default=0.15)
    parser.add_argument("--insert_ratio_max", type=float, default=0.35)
    parser.add_argument("--min_sents", type=int, default=4,
                        help="Minimum sentences in base document")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading HC3 from {args.hc3_file}...")
    records = []
    with open(args.hc3_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            human_ans = d.get("human_answers", [])
            ai_ans = d.get("chatgpt_answers", [])
            if not human_ans or not ai_ans:
                continue
            records.append({
                "question": d.get("question", ""),
                "human": human_ans[0],
                "ai": ai_ans[0],
            })
    print(f"Loaded {len(records)} QA pairs")

    docs_out = open(args.output_dir / "docs.jsonl", "w", encoding="utf-8")
    spans_out = open(args.output_dir / "spans.jsonl", "w", encoding="utf-8")

    doc_id = 0
    skipped = 0
    for rec in rng.sample(records, min(len(records), args.max_docs * 2)):
        if doc_id >= args.max_docs:
            break

        human_sents = split_sentences(rec["human"])
        ai_sents = split_sentences(rec["ai"])

        if len(human_sents) < args.min_sents or len(ai_sents) < args.min_sents:
            skipped += 1
            continue

        insert_ratio = rng.uniform(args.insert_ratio_min, args.insert_ratio_max)

        # Type A: AI base, insert human sentences
        mixed_text_a, spans_a, doc_label_a = build_mixed_doc(
            base_sents=ai_sents,
            base_label=1,
            insert_pool=human_sents,
            insert_label=0,
            insert_ratio=insert_ratio,
            rng=rng,
        )
        doc_a = {
            "id": f"hc3_mixed_{doc_id}",
            "mixed_text": mixed_text_a,
            "document_label": doc_label_a,
            "base_origin": "ai",
            "inserted_origin": "human",
            "question": rec["question"],
            "n_spans": len(spans_a),
        }
        docs_out.write(json.dumps(doc_a, ensure_ascii=False) + "\n")
        for sp in spans_a:
            spans_out.write(json.dumps({"doc_id": doc_a["id"], **sp}, ensure_ascii=False) + "\n")
        doc_id += 1
        if doc_id >= args.max_docs:
            break

        # Type B: human base, insert AI sentences
        mixed_text_b, spans_b, doc_label_b = build_mixed_doc(
            base_sents=human_sents,
            base_label=0,
            insert_pool=ai_sents,
            insert_label=1,
            insert_ratio=insert_ratio,
            rng=rng,
        )
        doc_b = {
            "id": f"hc3_mixed_{doc_id}",
            "mixed_text": mixed_text_b,
            "document_label": doc_label_b,
            "base_origin": "human",
            "inserted_origin": "ai",
            "question": rec["question"],
            "n_spans": len(spans_b),
        }
        docs_out.write(json.dumps(doc_b, ensure_ascii=False) + "\n")
        for sp in spans_b:
            spans_out.write(json.dumps({"doc_id": doc_b["id"], **sp}, ensure_ascii=False) + "\n")
        doc_id += 1

    docs_out.close()
    spans_out.close()
    print(f"Generated {doc_id} mixed documents ({skipped} skipped, too short)")
    print(f"Docs: {args.output_dir}/docs.jsonl")
    print(f"Spans: {args.output_dir}/spans.jsonl")


if __name__ == "__main__":
    main()
