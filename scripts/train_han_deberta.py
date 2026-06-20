"""
scripts/train_han_deberta.py

Train HierarchicalDetector (HAN over DeBERTa-v3-large) on M4 + HC3 mixed data.
Supports:
  - MTL loss (doc + sentence + feature buckets)
  - Dual learning rates (DeBERTa vs cross-sent heads)
  - fp16 training
  - Per-epoch checkpointing + resume
  - Smoke mode (--max_train_samples)
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
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model_han import HierarchicalDetector

SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*")
PARA_RE = re.compile(r"\n\s*\n+")

DEFAULT_TRAIN = Path(
    "/inspire/hdd/project/fdu-aidake-cfff/public/wangyanqing/NLPPJ"
    "/data/processed/semeval/semeval_train_full.csv"
)
DEFAULT_DEV = Path(
    "/inspire/hdd/project/fdu-aidake-cfff/public/wangyanqing/NLPPJ"
    "/data/processed/semeval/semeval_dev_full.csv"
)


# ── helpers ────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_sentences(text: str, max_sents: int) -> list[str]:
    sents = [s.strip() for s in SENTENCE_RE.findall(text) if s.strip()]
    return sents[:max_sents]


def split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in PARA_RE.split(text) if p.strip()]
    return paras if paras else [text.strip()]


def assign_bucket(value: float, p33: float, p67: float) -> int:
    if value <= p33:
        return 0
    elif value <= p67:
        return 1
    else:
        return 2


def get_feature_buckets(text: str, mtl_boundaries: dict) -> dict[str, int]:
    """Compute MTL auxiliary targets for a text."""
    import re as _re
    tokens = _re.findall(r"\b[a-zA-Z']+\b", text.lower())
    n_tokens = max(len(tokens), 1)
    sents = [s.strip() for s in SENTENCE_RE.findall(text) if s.strip()]
    n_sents = max(len(sents), 1)

    # discourse_total_density
    _DISCOURSE = [
        "furthermore", "moreover", "additionally", "in addition",
        "however", "nevertheless", "therefore", "thus", "consequently",
        "in conclusion", "to summarize", "in summary", "overall",
    ]
    disc_count = sum(
        1 for s in sents if any(m in s.lower() for m in _DISCOURSE)
    )
    discourse_val = disc_count / n_sents

    # contraction_ratio
    _CONT_RE = _re.compile(
        r"\b(don't|doesn't|didn't|won't|wouldn't|can't|couldn't|"
        r"isn't|aren't|wasn't|weren't|I'm|I've|I'll|it's|that's|"
        r"they're|we're|you're)\b",
        _re.IGNORECASE,
    )
    contraction_val = len(_CONT_RE.findall(text)) / n_tokens

    # first_person_ratio
    _FP = frozenset(["i", "me", "my", "mine", "we", "us", "our"])
    fp_val = sum(1 for t in tokens if t in _FP) / n_tokens

    buckets = {}
    for feat, val in [
        ("discourse", discourse_val),
        ("contraction", contraction_val),
        ("first_person", fp_val),
    ]:
        if feat in mtl_boundaries:
            b = mtl_boundaries[feat]
            buckets[feat] = assign_bucket(val, b["p33"], b["p67"])
        else:
            buckets[feat] = 1  # neutral
    return buckets


# ── Dataset ────────────────────────────────────────────────────────────────────

class HANDocumentDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_sent_len: int,
        max_sents: int,
        mtl_boundaries: dict,
        has_sent_labels: bool = False,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_sent_len = max_sent_len
        self.max_sents = max_sents
        self.mtl_boundaries = mtl_boundaries
        self.has_sent_labels = has_sent_labels

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        text = rec["text"]
        doc_label = int(rec["label"])

        sentences = split_sentences(text, self.max_sents)
        if not sentences:
            sentences = [text[:200]]

        paragraphs = split_paragraphs(text)
        # Map paragraph boundaries to sentence indices
        para_boundaries = self._compute_para_boundaries(sentences, paragraphs)

        # Tokenize all sentences
        enc = self.tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=self.max_sent_len,
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"],           # (num_sents, max_sent_len)
            "attention_mask": enc["attention_mask"],  # (num_sents, max_sent_len)
            "doc_label": torch.tensor(doc_label, dtype=torch.long),
            "para_boundaries": para_boundaries,
            "num_sents": len(sentences),
        }

        # MTL feature buckets
        feat_buckets = get_feature_buckets(text, self.mtl_boundaries)
        for feat, bucket in feat_buckets.items():
            item[f"feat_{feat}"] = torch.tensor(bucket, dtype=torch.long)

        # Sentence labels (only for HC3 mixed data)
        if self.has_sent_labels and "sent_labels" in rec:
            sent_labels = rec["sent_labels"][:self.max_sents]
            # Pad to num_sents
            while len(sent_labels) < len(sentences):
                sent_labels.append(doc_label)
            item["sent_labels"] = torch.tensor(sent_labels[:len(sentences)], dtype=torch.long)

        return item

    def _compute_para_boundaries(
        self, sentences: list[str], paragraphs: list[str]
    ) -> list[tuple[int, int]]:
        """Map paragraphs to sentence index ranges (approximate)."""
        if len(paragraphs) <= 1:
            return [(0, len(sentences))]

        # Estimate sentences per paragraph proportionally
        para_lens = [max(len(split_sentences(p, 999)), 1) for p in paragraphs]
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

        # Ensure last boundary covers all sentences
        if boundaries and boundaries[-1][1] < n_sents:
            boundaries[-1] = (boundaries[-1][0], n_sents)
        if not boundaries:
            boundaries = [(0, n_sents)]
        return boundaries


def han_collate_fn(batch: list[dict]) -> dict:
    """Custom collate: each item has variable num_sents, process individually."""
    # Return as list — training loop handles each doc independently
    return batch


# ── Training utilities ─────────────────────────────────────────────────────────

def compute_loss(
    outputs: dict,
    batch_item: dict,
    loss_sent_weight: float,
    loss_feat_weight: float,
    ce_loss: nn.CrossEntropyLoss,
) -> tuple[torch.Tensor, dict]:
    """Compute combined loss for one document."""
    losses = {}

    # Document classification loss
    doc_logits = outputs["doc_logits"].unsqueeze(0)  # (1, 2)
    doc_label = batch_item["doc_label"].unsqueeze(0)  # (1,)
    losses["doc"] = ce_loss(doc_logits, doc_label)

    total = losses["doc"]

    # Sentence-level loss (only if sent_labels provided)
    if "sent_labels" in batch_item and loss_sent_weight > 0:
        sent_weights = outputs["sent_attention_weights"]  # (num_sents,)
        sent_labels = batch_item["sent_labels"].to(sent_weights.device)
        n = min(len(sent_weights), len(sent_labels))
        # Treat attention weights as probability of being AI (soft supervision)
        # Use MSE loss with soft labels (autocast-safe)
        sent_probs = sent_weights[:n].float()
        sent_targets = sent_labels[:n].float()
        losses["sent"] = torch.nn.functional.mse_loss(sent_probs, sent_targets)
        total = total + loss_sent_weight * losses["sent"]

    # MTL auxiliary losses
    aux_logits = outputs["aux_logits"]
    for feat in ["discourse", "contraction", "first_person"]:
        key = f"feat_{feat}"
        if key in batch_item and loss_feat_weight > 0:
            feat_logit = aux_logits[feat].unsqueeze(0)  # (1, 3)
            feat_label = batch_item[key].unsqueeze(0).to(feat_logit.device)
            losses[f"feat_{feat}"] = ce_loss(feat_logit, feat_label)
            total = total + loss_feat_weight * losses[f"feat_{feat}"]

    return total, losses


def evaluate(
    model: HierarchicalDetector,
    dev_records: list[dict],
    tokenizer,
    max_sent_len: int,
    max_sents: int,
    mtl_boundaries: dict,
    device: torch.device,
) -> dict:
    model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for rec in dev_records:
            sentences = split_sentences(rec["text"], max_sents)
            if not sentences:
                sentences = [rec["text"][:200]]
            paragraphs = split_paragraphs(rec["text"])

            enc = tokenizer(
                sentences,
                padding="max_length",
                truncation=True,
                max_length=max_sent_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # Compute paragraph boundaries
            para_lens = [max(len(split_sentences(p, 999)), 1) for p in paragraphs]
            t = sum(para_lens)
            n_sents = len(sentences)
            para_boundaries = []
            start = 0
            for pl in para_lens:
                n = max(1, round(pl / t * n_sents))
                end = min(start + n, n_sents)
                if start < end:
                    para_boundaries.append((start, end))
                start = end
                if start >= n_sents:
                    break
            if not para_boundaries or para_boundaries[-1][1] < n_sents:
                para_boundaries = [(0, n_sents)]

            try:
                outputs = model(input_ids, attention_mask, para_boundaries)
                logits = outputs["doc_logits"]
                prob = torch.softmax(logits, dim=0)[1].item()
                pred = int(logits.argmax().item())
                label = int(rec["label"])
                correct += int(pred == label)
                total += 1
                all_probs.append(prob)
                all_labels.append(label)
            except Exception:
                total += 1

    acc = correct / total if total > 0 else 0.0
    return {"accuracy": acc, "total": total}


# ── Main training loop ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="/inspire/hdd/project/fdu-aidake-cfff/public/hanz/semeval2024_task8a_en_baseline/models/deberta-v3-large")
    parser.add_argument("--train_data", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev_data", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--hc3_mixed_data", type=str,
                        default="data/hc3_mixed/docs.jsonl")
    parser.add_argument("--hc3_spans", type=str,
                        default="data/hc3_mixed/spans.jsonl")
    parser.add_argument("--feature_baselines", type=Path,
                        default=Path("data/feature_baselines.json"))
    parser.add_argument("--output_dir", type=Path,
                        default=Path("outputs/han_deberta_large_full"))
    parser.add_argument("--max_sent_len", type=int, default=96)
    parser.add_argument("--max_sents", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--deberta_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--loss_sent_weight", type=float, default=0.3)
    parser.add_argument("--loss_feat_weight", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=0.3)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit training samples (smoke mode)")
    parser.add_argument("--max_hc3_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint dir to resume from")
    parser.add_argument("--log_steps", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load feature baselines ─────────────────────────────────────────────────
    mtl_boundaries: dict = {}
    if args.feature_baselines.exists():
        baselines = json.loads(args.feature_baselines.read_text())
        mtl_boundaries = baselines.get("_mtl_boundaries", {})
        print(f"Loaded MTL boundaries: {list(mtl_boundaries.keys())}")
    else:
        print("WARNING: feature_baselines.json not found. MTL auxiliary tasks disabled.")

    # ── Load tokenizer ─────────────────────────────────────────────────────────
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ── Load M4 training data ──────────────────────────────────────────────────
    print("Loading M4 training data...")
    train_df = pd.read_csv(args.train_data, encoding="utf-8-sig")
    train_df["text"] = train_df["text"].fillna("").astype(str)
    train_df["label"] = train_df["label"].astype(int)

    if args.max_train_samples:
        n_per = args.max_train_samples // 2
        ai = train_df[train_df["label"] == 1].sample(
            min(n_per, (train_df["label"] == 1).sum()), random_state=args.seed
        )
        hu = train_df[train_df["label"] == 0].sample(
            min(n_per, (train_df["label"] == 0).sum()), random_state=args.seed
        )
        train_df = pd.concat([ai, hu]).sample(frac=1, random_state=args.seed)
    print(f"M4 train: {len(train_df)} docs")

    m4_records = [
        {"text": row["text"], "label": row["label"]}
        for _, row in train_df.iterrows()
    ]

    # ── Load HC3 mixed data ────────────────────────────────────────────────────
    hc3_records: list[dict] = []
    hc3_path = Path(args.hc3_mixed_data)
    spans_path = Path(args.hc3_spans)

    if hc3_path.exists() and spans_path.exists():
        print("Loading HC3 mixed data...")
        span_by_doc: dict[str, list[int]] = {}
        with open(spans_path, encoding="utf-8") as f:
            for line in f:
                sp = json.loads(line)
                did = sp["doc_id"]
                if did not in span_by_doc:
                    span_by_doc[did] = []
                span_by_doc[did].append(sp["span_label"])

        with open(hc3_path, encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                rec = {
                    "text": doc["mixed_text"],
                    "label": doc["document_label"],
                    "sent_labels": span_by_doc.get(doc["id"], []),
                }
                hc3_records.append(rec)

        if args.max_hc3_samples:
            hc3_records = hc3_records[:args.max_hc3_samples]
        print(f"HC3 mixed: {len(hc3_records)} docs")
    else:
        print("HC3 mixed data not found — training without sentence-level supervision")

    # ── Load dev data ──────────────────────────────────────────────────────────
    dev_df = pd.read_csv(args.dev_data, encoding="utf-8-sig")
    dev_df["text"] = dev_df["text"].fillna("").astype(str)
    dev_df["label"] = dev_df["label"].astype(int)
    if args.max_eval_samples:
        dev_df = dev_df.sample(min(args.max_eval_samples, len(dev_df)), random_state=args.seed)
    dev_records = [{"text": row["text"], "label": row["label"]} for _, row in dev_df.iterrows()]
    print(f"Dev: {len(dev_records)} docs")

    # ── Build combined training list (HC3 x2 weight via repetition) ───────────
    all_train = m4_records + hc3_records * 2  # HC3 doubled for weight
    random.shuffle(all_train)
    print(f"Total training records (with HC3 x2): {len(all_train)}")

    # ── Initialize model ───────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume_from:
        print(f"Resuming from {args.resume_from}...")
        model = HierarchicalDetector.from_pretrained(args.resume_from)
        ckpt_meta = json.loads(
            (Path(args.resume_from) / "meta.json").read_text()
        )
        start_epoch = ckpt_meta.get("epoch", 0) + 1
    else:
        print(f"Initializing model from {args.model_name}...")
        model = HierarchicalDetector(args.model_name, args.max_sent_len, args.max_sents)

    model = model.to(device)

    # ── Optimizer with dual learning rates ─────────────────────────────────────
    # No-decay groups: bias and normalization parameters
    # Two learning rate groups: DeBERTa backbone vs new heads
    deberta_param_ids = set(id(p) for p in model.sentence_encoder.parameters())
    deberta_params = [p for p in model.parameters()
                      if id(p) in deberta_param_ids and p.requires_grad]
    head_params = [p for p in model.parameters()
                   if id(p) not in deberta_param_ids and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": deberta_params, "lr": args.deberta_lr, "weight_decay": 0.01},
        {"params": head_params, "lr": args.head_lr, "weight_decay": 0.01},
    ], eps=1e-6)

    total_steps = (len(all_train) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    ce_loss = nn.CrossEntropyLoss()
    scaler = None  # DeBERTa-v3 uses bf16 internally; use autocast without scaler

    # ── Training loop ──────────────────────────────────────────────────────────
    best_dev_acc = 0.0
    log_file = open(args.output_dir / "training_log.jsonl", "a")

    print(f"\nStarting training: {args.epochs} epochs, {len(all_train)} docs/epoch")
    print(f"Grad accum: {args.grad_accum}, effective batch: {args.grad_accum}")
    print(f"DeBERTa lr: {args.deberta_lr}, Head lr: {args.head_lr}")
    print(f"fp16: {args.fp16}\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        random.shuffle(all_train)
        epoch_loss = 0.0
        step_count = 0
        optimizer.zero_grad()
        epoch_start = time.time()

        for doc_idx, rec in enumerate(all_train):
            try:
                sentences = split_sentences(rec["text"], args.max_sents)
                if not sentences:
                    continue
                paragraphs = split_paragraphs(rec["text"])

                enc = tokenizer(
                    sentences,
                    padding="max_length",
                    truncation=True,
                    max_length=args.max_sent_len,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)

                # Para boundaries
                para_lens = [max(len(split_sentences(p, 999)), 1) for p in paragraphs]
                t = sum(para_lens)
                n_sents = len(sentences)
                para_boundaries = []
                start = 0
                for pl in para_lens:
                    n = max(1, round(pl / t * n_sents))
                    end = min(start + n, n_sents)
                    if start < end:
                        para_boundaries.append((start, end))
                    start = end
                    if start >= n_sents:
                        break
                if not para_boundaries or para_boundaries[-1][1] < n_sents:
                    para_boundaries = [(0, n_sents)]

                # Build batch_item tensors
                batch_item = {
                    "doc_label": torch.tensor(int(rec["label"]), dtype=torch.long).to(device),
                }
                feat_buckets = get_feature_buckets(rec["text"], mtl_boundaries)
                for feat, bucket in feat_buckets.items():
                    batch_item[f"feat_{feat}"] = torch.tensor(bucket, dtype=torch.long).to(device)
                if "sent_labels" in rec and rec["sent_labels"]:
                    sl = rec["sent_labels"][:n_sents]
                    while len(sl) < n_sents:
                        sl.append(int(rec["label"]))
                    batch_item["sent_labels"] = torch.tensor(sl[:n_sents], dtype=torch.long).to(device)

                # Forward pass
                if args.fp16:
                    with autocast("cuda", dtype=torch.bfloat16):
                        outputs = model(input_ids, attention_mask, para_boundaries)
                        loss, loss_parts = compute_loss(
                            outputs, batch_item,
                            args.loss_sent_weight, args.loss_feat_weight, ce_loss
                        )
                    (loss / args.grad_accum).backward()
                else:
                    outputs = model(input_ids, attention_mask, para_boundaries)
                    loss, loss_parts = compute_loss(
                        outputs, batch_item,
                        args.loss_sent_weight, args.loss_feat_weight, ce_loss
                    )
                    (loss / args.grad_accum).backward()

                epoch_loss += loss.item()

                # Gradient accumulation step
                if (doc_idx + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step_count += 1

                    if step_count % args.log_steps == 0:
                        avg_loss = epoch_loss / max(doc_idx + 1, 1)
                        elapsed = time.time() - epoch_start
                        msg = {
                            "epoch": epoch, "step": step_count,
                            "doc": doc_idx, "avg_loss": round(avg_loss, 4),
                            "elapsed_min": round(elapsed / 60, 1),
                        }
                        print(f"Ep{epoch} step{step_count} loss={avg_loss:.4f} t={elapsed/60:.1f}m")
                        log_file.write(json.dumps(msg) + "\n")
                        log_file.flush()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM at doc {doc_idx}, skipping. Error: {e}")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                raise

        # ── End of epoch: evaluate ─────────────────────────────────────────────
        avg_epoch_loss = epoch_loss / max(len(all_train), 1)
        print(f"\nEpoch {epoch} finished. avg_loss={avg_epoch_loss:.4f}")
        print("Evaluating on dev...")
        dev_metrics = evaluate(
            model, dev_records, tokenizer,
            args.max_sent_len, args.max_sents, mtl_boundaries, device
        )
        dev_acc = dev_metrics["accuracy"]
        print(f"Dev accuracy: {dev_acc:.4f}")

        epoch_msg = {
            "epoch": epoch,
            "avg_loss": round(avg_epoch_loss, 4),
            "dev_accuracy": round(dev_acc, 4),
        }
        log_file.write(json.dumps(epoch_msg) + "\n")
        log_file.flush()

        # Save checkpoint
        epoch_dir = args.output_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(epoch_dir))
        (epoch_dir / "meta.json").write_text(
            json.dumps({"epoch": epoch, "dev_accuracy": dev_acc, "avg_loss": avg_epoch_loss})
        )

        # Save best model
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_dir = args.output_dir / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_dir))
            (best_dir / "meta.json").write_text(
                json.dumps({"epoch": epoch, "dev_accuracy": dev_acc})
            )
            print(f"  *** New best model saved (dev_acc={dev_acc:.4f}) ***")

    log_file.close()
    print(f"\nTraining complete. Best dev accuracy: {best_dev_acc:.4f}")
    print(f"Best model saved to: {args.output_dir}/best_model")


if __name__ == "__main__":
    main()
