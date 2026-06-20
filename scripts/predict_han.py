"""
scripts/predict_han.py

Run trained HierarchicalDetector on test data.
Outputs predictions.csv + metrics.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model_han import HierarchicalDetector
from src.metrics import binary_metrics

SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*")
PARA_RE = re.compile(r"\n\s*\n+")


def split_sentences(text: str, max_sents: int = 128) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.findall(text) if s.strip()][:max_sents]


def split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in PARA_RE.split(text) if p.strip()]
    return parts if parts else [text.strip()]


def para_boundaries(sentences: list[str], paragraphs: list[str]) -> list[tuple[int, int]]:
    para_lens = [max(len(split_sentences(p)), 1) for p in paragraphs]
    total = sum(para_lens)
    n_sents = len(sentences)
    boundaries = []
    start = 0
    for pl in para_lens:
        n = max(1, round(pl / total * n_sents))
        end = min(start + n, n_sents)
        if start < end:
            boundaries.append((start, end))
        start = end
        if start >= n_sents:
            break
    if not boundaries or boundaries[-1][1] < n_sents:
        boundaries = [(0, n_sents)]
    return boundaries


def predict_one(
    text: str,
    model: HierarchicalDetector,
    tokenizer,
    device: torch.device,
    max_sent_len: int = 96,
    max_sents: int = 128,
) -> dict:
    sentences = split_sentences(text, max_sents)
    if not sentences:
        sentences = [text[:300]]
    paragraphs = split_paragraphs(text)
    bounds = para_boundaries(sentences, paragraphs)

    enc = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=max_sent_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask, bounds)

    logits = outputs["doc_logits"]
    probs = torch.softmax(logits, dim=0)
    prob_ai = float(probs[1].item())
    prob_human = float(probs[0].item())
    pred = int(logits.argmax().item())

    sent_weights = outputs["sent_attention_weights"].cpu().numpy().tolist()
    para_weights = outputs["para_attention_weights"].cpu().numpy().tolist()

    return {
        "pred": pred,
        "prob_ai": round(prob_ai, 6),
        "prob_human": round(prob_human, 6),
        "logit_ai": round(float(logits[1].item()), 6),
        "logit_human": round(float(logits[0].item()), 6),
        "sent_attention_weights": sent_weights,
        "para_attention_weights": para_weights,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", type=str,
                        default="/inspire/hdd/project/fdu-aidake-cfff/public/hanz/semeval2024_task8a_en_baseline/models/deberta-v3-large")
    parser.add_argument("--max_sent_len", type=int, default=96)
    parser.add_argument("--max_sents", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model from {args.model_dir}...")
    model = HierarchicalDetector.from_pretrained(str(args.model_dir))
    model = model.to(device)
    model.eval()

    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Loading input from {args.input_file}...")
    if args.input_file.suffix == ".jsonl":
        rows = [json.loads(l) for l in args.input_file.read_text().splitlines() if l.strip()]
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(args.input_file, encoding="utf-8-sig")
    df["text"] = df["text"].fillna("").astype(str)

    print(f"Predicting on {len(df)} documents...")
    results = []
    for i, row in df.iterrows():
        if i % 500 == 0:
            print(f"  {i}/{len(df)}...")
        try:
            out = predict_one(
                str(row["text"]), model, tokenizer, device,
                args.max_sent_len, args.max_sents
            )
        except Exception as e:
            print(f"  Warning: failed on row {i}: {e}")
            out = {"pred": 0, "prob_ai": 0.5, "prob_human": 0.5,
                   "logit_ai": 0.0, "logit_human": 0.0}
        results.append(out)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_df = df.copy()
    pred_df["pred"] = [r["pred"] for r in results]
    pred_df["prob_ai"] = [r["prob_ai"] for r in results]
    pred_df["prob_human"] = [r["prob_human"] for r in results]
    pred_df["logit_ai"] = [r["logit_ai"] for r in results]
    pred_df["logit_human"] = [r["logit_human"] for r in results]

    # Save without text to save space
    out_df = pred_df.drop(columns=["text"], errors="ignore")
    out_df.to_csv(args.output_dir / "predictions.csv", index=False)
    out_df.to_json(args.output_dir / "predictions.jsonl", orient="records", lines=True)

    # Metrics (if label available)
    if "label" in pred_df.columns:
        pred_df["label"] = pred_df["label"].astype(int)
        pred_df["correct"] = pred_df["label"] == pred_df["pred"]
        metrics = binary_metrics(pred_df["label"], pred_df["prob_ai"])
        metrics["rows"] = len(pred_df)
        (args.output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        print(json.dumps(metrics, indent=2))
    else:
        print(f"No labels found. Predictions saved to {args.output_dir}")


if __name__ == "__main__":
    main()
