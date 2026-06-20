"""
scripts/predict_hsad.py

Batch inference with HSAD model.
Output format matches predict_transformer.py:
  predictions.csv  — id, label, pred, prob_human, prob_ai, correct
  metrics.json     — accuracy, macro_f1, auroc, ...
  sent_predictions.csv — id, sent_idx, sent_text, prob_ai
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model_hsad import HSAD
from src.metrics import binary_metrics

SENT_RE = re.compile(r"[^.!?\n]+[.!?\n]*")


def split_sentences(text: str, max_sents: int) -> list[str]:
    return [s.strip() for s in SENT_RE.findall(text) if s.strip()][:max_sents]


def get_sent_spans(text, sentences, offsets):
    spans = []
    search_from = 0
    for sent in sentences:
        char_start = text.find(sent, search_from)
        if char_start == -1:
            char_start = search_from
        char_end = char_start + len(sent)
        search_from = char_start + 1
        tok_start, tok_end = None, None
        for i, (a, b) in enumerate(offsets):
            if a == 0 and b == 0:
                continue
            if tok_start is None and b > char_start:
                tok_start = i
            if b <= char_end:
                tok_end = i
        if tok_start is None:
            tok_start = 1
        if tok_end is None or tok_end < tok_start:
            tok_end = tok_start
        spans.append((tok_start, tok_end + 1))
    return spans


def read_input(input_file: Path) -> list[dict]:
    path = str(input_file)
    if path.endswith(".jsonl"):
        rows = []
        with open(input_file) as f:
            for line in f:
                r = json.loads(line)
                rows.append({"id": r["id"], "text": str(r.get("text", "")),
                             "label": int(r.get("label", -1))})
        return rows
    df = pd.read_csv(input_file, encoding="utf-8-sig")
    df["text"] = df["text"].fillna("").astype(str)
    return df[["id", "text", "label"]].to_dict("records")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path,
                        default=REPO_ROOT / "outputs/hsad/best_model")
    parser.add_argument("--roberta_model_dir", type=Path,
                        default=REPO_ROOT / "outputs/roberta_base/best_model")
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_sents", type=int, default=64)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.roberta_model_dir), use_fast=True)
    print(f"Loading model from {args.model_dir}...")
    model = HSAD.from_pretrained(str(args.model_dir),
                                  roberta_model_dir=str(args.roberta_model_dir)).to(device)
    model.eval()

    rows = read_input(args.input_file)
    print(f"Predicting {len(rows)} documents...")

    pred_rows, sent_rows = [], []
    for i, rec in enumerate(rows):
        if i % 1000 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
        text = rec["text"]
        doc_id = rec["id"]
        true_label = rec["label"]

        try:
            sents = split_sentences(text, args.max_sents) or [text[:200]]
            enc = tokenizer(text, truncation=True, max_length=args.max_length,
                            return_tensors="pt", return_offsets_mapping=True)
            ids = enc["input_ids"].squeeze(0).to(device)
            mask = enc["attention_mask"].squeeze(0).to(device)
            offsets = enc["offset_mapping"].squeeze(0).tolist()
            spans = get_sent_spans(text, sents, offsets)

            with torch.no_grad():
                out = model(ids, mask, spans)

            probs = torch.softmax(out["doc_logits"], dim=0).cpu().numpy()
            pred = int(probs.argmax())
            sent_probs = torch.softmax(out["sent_logits"], dim=1)[:, 1].cpu().numpy()

            pred_rows.append({
                "id": doc_id, "label": true_label, "pred": pred,
                "prob_human": float(probs[0]), "prob_ai": float(probs[1]),
                "correct": bool(pred == true_label) if true_label >= 0 else None,
            })
            for j, (s, sp) in enumerate(zip(sents, sent_probs)):
                sent_rows.append({
                    "id": doc_id, "sent_idx": j,
                    "sent_text": s[:200], "prob_ai": round(float(sp), 4),
                })
        except Exception as e:
            print(f"Error at doc {doc_id}: {e}")
            pred_rows.append({"id": doc_id, "label": true_label, "pred": 0,
                              "prob_human": 1.0, "prob_ai": 0.0, "correct": False})

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(args.output_dir / "predictions.csv", index=False)

    sent_df = pd.DataFrame(sent_rows)
    sent_df.to_csv(args.output_dir / "sent_predictions.csv", index=False, escapechar="\\")

    # Metrics (only when labels are available)
    valid = pred_df[pred_df["label"] >= 0]
    if len(valid) > 0:
        metrics = binary_metrics(
            valid["label"].to_numpy(),
            valid["pred"].to_numpy(),
            valid["prob_ai"].to_numpy(),
        )
        metrics["rows"] = len(valid)
        (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"\nAccuracy: {metrics['accuracy']:.4f}  Macro-F1: {metrics['macro_f1']:.4f}"
              f"  AUROC: {metrics.get('auroc', 'N/A')}")

    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
