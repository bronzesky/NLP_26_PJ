"""
scripts/train_hsad.py

Train HSAD: Hierarchical Sentence-Aware Detector.
Phase 1 (smoke / doc-only): semeval subtask A documents only.
Phase 2 (full MTL): + subtask C sentence-level supervision.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model_hsad import HSAD

SENT_RE = re.compile(r"[^.!?\n]+[.!?\n]*")
DATA_DIR = Path("/inspire/hdd/project/fdu-aidake-cfff/public/wangyanqing/NLPPJ/data/processed/semeval")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_sentences(text: str, max_sents: int) -> list[str]:
    sents = [s.strip() for s in SENT_RE.findall(text) if s.strip()]
    return sents[:max_sents]


def get_sent_spans(text: str, sentences: list[str], offsets: list[tuple]) -> list[tuple[int, int]]:
    """Map sentences back to token index spans using offset mapping."""
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
                continue  # special tokens
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


def load_subtaskC(parquet_path: Path, max_sents: int) -> list[dict]:
    """Load subtask C data and convert char boundary to sent labels."""
    df = pd.read_parquet(parquet_path)
    records = []
    for _, row in df.iterrows():
        text = str(row["text"])
        boundary = int(row["label"])
        sents = split_sentences(text, max_sents)
        if not sents:
            continue
        # assign sent labels based on char boundary
        sent_labels = []
        pos = 0
        for s in sents:
            idx = text.find(s, pos)
            if idx == -1:
                idx = pos
            mid = idx + len(s) // 2
            sent_labels.append(0 if mid < boundary else 1)
            pos = max(pos, idx + 1)
        # doc label: majority of sent labels
        doc_label = int(sum(sent_labels) > len(sent_labels) / 2)
        records.append({
            "text": text,
            "doc_label": doc_label,
            "sent_labels": sent_labels,
            "has_sent_labels": True,
        })
    return records


def load_semeval(csv_path: Path, max_sents: int, max_samples=None) -> list[dict]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    if max_samples:
        df = df.sample(min(max_samples, len(df)), random_state=42)
    records = []
    for _, row in df.iterrows():
        sents = split_sentences(str(row["text"]), max_sents)
        records.append({
            "text": str(row["text"]),
            "doc_label": int(row["label"]),
            "sent_labels": None,  # no sentence labels
            "has_sent_labels": False,
        })
    return records


def process_record(rec: dict, tokenizer, max_length: int, max_sents: int, device):
    """Tokenize a record and compute sent_spans. Returns tensors."""
    text = rec["text"]
    sentences = split_sentences(text, max_sents)
    if not sentences:
        sentences = [text[:200]]

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    input_ids = enc["input_ids"].squeeze(0).to(device)
    attention_mask = enc["attention_mask"].squeeze(0).to(device)
    offsets = enc["offset_mapping"].squeeze(0).tolist()

    sent_spans = get_sent_spans(text, sentences, offsets)

    # Truncate sent_labels to match available sentences (some may be cut by max_length)
    n_sents = len(sent_spans)
    sent_labels_tensor = None
    if rec["has_sent_labels"] and rec["sent_labels"]:
        sl = list(rec["sent_labels"])[:n_sents]
        while len(sl) < n_sents:
            sl.append(rec["doc_label"])
        sent_labels_tensor = torch.tensor(sl, dtype=torch.long, device=device)

    doc_label = torch.tensor(rec["doc_label"], dtype=torch.long, device=device)
    return input_ids, attention_mask, sent_spans, doc_label, sent_labels_tensor


def evaluate(model, records, tokenizer, max_length, max_sents, device) -> dict:
    model.eval()
    correct = total = 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for rec in records:
            try:
                ids, mask, spans, doc_label, _ = process_record(
                    rec, tokenizer, max_length, max_sents, device)
                out = model(ids, mask, spans)
                pred = int(out["doc_logits"].argmax().item())
                prob = float(torch.softmax(out["doc_logits"], dim=0)[1].item())
                correct += int(pred == doc_label.item())
                total += 1
                all_probs.append(prob)
                all_labels.append(doc_label.item())
            except Exception:
                total += 1
    acc = correct / total if total > 0 else 0.0
    return {"accuracy": acc, "total": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roberta_model_dir", type=Path,
                        default=REPO_ROOT / "outputs/roberta_base/best_model")
    parser.add_argument("--train_data", type=Path, default=DATA_DIR / "semeval_train_full.csv")
    parser.add_argument("--subtaskC_train", type=Path,
                        default=REPO_ROOT / "data/subtaskC/subtaskC/train-00000-of-00001.parquet")
    parser.add_argument("--dev_data", type=Path, default=DATA_DIR / "semeval_dev_full.csv")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs/hsad")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--roberta_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--sent_loss_weight", type=float, default=0.3)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_sents", type=int, default=64)
    parser.add_argument("--freeze_layers", type=int, default=8)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--no_subtaskC", action="store_true",
                        help="Disable subtask C sentence supervision (doc-only mode)")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer from {args.roberta_model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.roberta_model_dir, use_fast=True)

    # Load data
    print("Loading semeval train data...")
    train_records = load_semeval(args.train_data, args.max_sents, args.max_train_samples)
    print(f"Semeval train: {len(train_records)} docs")

    if not args.no_subtaskC and args.subtaskC_train.exists():
        print("Loading subtask C sentence-level data...")
        subtaskC_records = load_subtaskC(args.subtaskC_train, args.max_sents)
        if args.max_train_samples:
            subtaskC_records = subtaskC_records[:max(1, args.max_train_samples // 10)]
        print(f"SubtaskC: {len(subtaskC_records)} docs (with sent labels)")
        all_train = train_records + subtaskC_records
    else:
        print("Running in doc-only mode (no subtask C)")
        all_train = train_records
    random.shuffle(all_train)
    print(f"Total training records: {len(all_train)}")

    print("Loading dev data...")
    dev_records = load_semeval(args.dev_data, args.max_sents, args.max_eval_samples)
    print(f"Dev: {len(dev_records)} docs")

    # Initialize model
    print(f"Initializing HSAD from {args.roberta_model_dir}...")
    model = HSAD(
        str(args.roberta_model_dir),
        freeze_layers=args.freeze_layers,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")

    # Optimizer: dual lr (RoBERTa backbone vs new heads)
    roberta_param_ids = set(id(p) for p in model.encoder.parameters())
    roberta_params = [p for p in model.parameters()
                      if id(p) in roberta_param_ids and p.requires_grad]
    head_params = [p for p in model.parameters()
                   if id(p) not in roberta_param_ids and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": roberta_params, "lr": args.roberta_lr, "weight_decay": 0.01},
        {"params": head_params,    "lr": args.head_lr,    "weight_decay": 0.01},
    ], eps=1e-6)

    total_steps = (len(all_train) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    print(f"\nStarting training: {args.epochs} epochs, {len(all_train)} docs/epoch")
    print(f"Grad accum: {args.grad_accum}  RoBERTa lr: {args.roberta_lr}  Head lr: {args.head_lr}")
    print(f"Sent loss weight: {args.sent_loss_weight}  fp16: {args.fp16}\n")

    log_file = open(args.output_dir / "training_log.jsonl", "w")
    best_dev_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        random.shuffle(all_train)
        epoch_loss = 0.0
        step_count = 0
        optimizer.zero_grad()
        epoch_start = time.time()

        for doc_idx, rec in enumerate(all_train):
            try:
                ids, mask, spans, doc_label, sent_labels = process_record(
                    rec, tokenizer, args.max_length, args.max_sents, device)

                if args.fp16:
                    with autocast("cuda", dtype=torch.bfloat16):
                        out = model(ids, mask, spans)
                        loss = ce_loss(out["doc_logits"].unsqueeze(0), doc_label.unsqueeze(0))
                        if sent_labels is not None and len(sent_labels) > 0:
                            n = min(len(sent_labels), out["sent_logits"].size(0))
                            sent_loss = ce_loss(out["sent_logits"][:n], sent_labels[:n])
                            loss = loss + args.sent_loss_weight * sent_loss
                else:
                    out = model(ids, mask, spans)
                    loss = ce_loss(out["doc_logits"].unsqueeze(0), doc_label.unsqueeze(0))
                    if sent_labels is not None and len(sent_labels) > 0:
                        n = min(len(sent_labels), out["sent_logits"].size(0))
                        sent_loss = ce_loss(out["sent_logits"][:n], sent_labels[:n])
                        loss = loss + args.sent_loss_weight * sent_loss

                (loss / args.grad_accum).backward()
                epoch_loss += loss.item()

                if (doc_idx + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step_count += 1

                    if step_count % args.log_steps == 0:
                        avg = epoch_loss / max(doc_idx + 1, 1)
                        elapsed = time.time() - epoch_start
                        msg = {"epoch": epoch, "step": step_count, "doc": doc_idx,
                               "avg_loss": round(avg, 4), "elapsed_min": round(elapsed / 60, 1)}
                        print(f"Ep{epoch} step{step_count} loss={avg:.4f} t={elapsed/60:.1f}m",
                              flush=True)
                        log_file.write(json.dumps(msg) + "\n")
                        log_file.flush()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM at doc {doc_idx}, skipping")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                else:
                    raise

        # End of epoch
        avg_epoch_loss = epoch_loss / max(len(all_train), 1)
        print(f"Epoch {epoch} finished. avg_loss={avg_epoch_loss:.4f}", flush=True)

        print("Evaluating on dev...", flush=True)
        dev_metrics = evaluate(model, dev_records, tokenizer,
                               args.max_length, args.max_sents, device)
        dev_acc = dev_metrics["accuracy"]
        print(f"Dev accuracy: {dev_acc:.4f}", flush=True)

        # Save checkpoint
        ckpt_dir = args.output_dir / f"epoch_{epoch}"
        ckpt_dir.mkdir(exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        (ckpt_dir / "meta.json").write_text(
            json.dumps({"epoch": epoch, "dev_accuracy": round(dev_acc, 4),
                        "avg_loss": round(avg_epoch_loss, 4)}))

        log_file.write(json.dumps({"epoch": epoch, "avg_loss": round(avg_epoch_loss, 4),
                                   "dev_accuracy": dev_acc}) + "\n")
        log_file.flush()

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_dir = args.output_dir / "best_model"
            best_dir.mkdir(exist_ok=True)
            model.save_pretrained(str(best_dir))
            (best_dir / "meta.json").write_text(
                json.dumps({"epoch": epoch, "dev_accuracy": round(dev_acc, 4)}))
            print(f"  *** New best model saved (dev_acc={dev_acc:.4f}) ***", flush=True)

    print(f"\nTraining complete. Best dev accuracy: {best_dev_acc:.4f}", flush=True)
    log_file.close()


if __name__ == "__main__":
    main()
