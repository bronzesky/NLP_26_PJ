from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration import TemperatureScaler
from src.data import (
    build_prediction_frame,
    read_prediction_input,
    write_metrics_if_labeled,
    write_prediction_outputs,
)


class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, chunks, tokenizer, max_length: int):
        self.encodings = tokenizer(list(chunks), truncation=True, max_length=max_length)

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.encodings.items()}


def _chunk_text(text: str, tokenizer, max_length: int, stride: int) -> list[str]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_length - 2:
        return [text]
    # decode overlapping windows back to strings
    chunks = []
    step = max(stride, 1)
    for start in range(0, len(ids), step):
        end = min(start + max_length - 2, len(ids))
        chunk_ids = ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text)
        if end >= len(ids):
            break
    return chunks if chunks else [text]


def _aggregate(chunk_probs: list[float], method: str, topk: int = 3) -> float:
    if not chunk_probs:
        return 0.5
    if method == "max":
        return max(chunk_probs)
    if method == "first":
        return chunk_probs[0]
    if method == "topk_mean":
        k = min(topk, len(chunk_probs))
        return sum(sorted(chunk_probs, reverse=True)[:k]) / k
    return sum(chunk_probs) / len(chunk_probs)  # mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--aggregation", choices=["mean", "max", "first", "topk_mean"], default="mean")
    parser.add_argument("--topk", type=int, default=3,
                        help="k for topk_mean aggregation")
    parser.add_argument("--calibration_file", type=Path, default=None,
                        help="Path to temperature.json for RoBERTa calibration")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--include_text", action="store_true")
    args = parser.parse_args()

    df = read_prediction_input(args.input_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # build per-document chunks and an index mapping chunk → doc
    all_chunks: list[str] = []
    doc_chunk_map: list[int] = []  # chunk index → doc index
    for doc_idx, text in enumerate(df["text"]):
        chunks = _chunk_text(str(text), tokenizer, args.max_length, args.stride)
        for chunk in chunks:
            all_chunks.append(chunk)
            doc_chunk_map.append(doc_idx)

    dataset = ChunkDataset(all_chunks, tokenizer, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    # Load calibration scaler if provided
    cal_scaler = None
    if args.calibration_file and Path(args.calibration_file).exists():
        import json
        cal_data = json.loads(Path(args.calibration_file).read_text())
        cal_scaler = TemperatureScaler.from_dict(cal_data)

    chunk_probs_ai: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            if cal_scaler is not None:
                probs_ai = cal_scaler.predict_positive_proba(logits.cpu().numpy())
                chunk_probs_ai.extend(float(p) for p in probs_ai)
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                chunk_probs_ai.extend(float(p[1]) for p in probs)

    # aggregate chunks → doc probabilities
    n_docs = len(df)
    doc_chunk_probs: list[list[float]] = [[] for _ in range(n_docs)]
    for chunk_idx, doc_idx in enumerate(doc_chunk_map):
        doc_chunk_probs[doc_idx].append(chunk_probs_ai[chunk_idx])

    prob_ai = np.array([_aggregate(probs, args.aggregation, topk=args.topk) for probs in doc_chunk_probs])
    prob_human = 1.0 - prob_ai
    predictions = (prob_ai >= 0.5).astype(int)

    pred_df = build_prediction_frame(
        df=df,
        pred=predictions,
        prob_human=prob_human,
        prob_ai=prob_ai,
        include_text=args.include_text,
    )
    write_prediction_outputs(args.output_dir, pred_df)
    metrics = write_metrics_if_labeled(args.output_dir, pred_df)
    if metrics is not None:
        print(json.dumps(metrics, indent=2))
    else:
        print(json.dumps({"rows": int(len(df)), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
