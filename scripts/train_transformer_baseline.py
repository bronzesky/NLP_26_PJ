import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_DATA_DIR = Path(
    "/inspire/hdd/project/fdu-aidake-cfff/public/wangyanqing/NLPPJ/data/processed/semeval"
)


def read_split(path: Path, max_samples: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"id", "label", "text"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df[["id", "label", "text"]].copy()
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].fillna("").astype(str)
    if max_samples is not None and max_samples < len(df):
        per_label = max_samples // df["label"].nunique()
        remainder = max_samples - per_label * df["label"].nunique()
        sampled = []
        for label, group in df.groupby("label", sort=True):
            take = min(len(group), per_label + (1 if remainder > 0 else 0))
            sampled.append(group.sample(n=take, random_state=42))
            remainder = max(0, remainder - 1)
        df = pd.concat(sampled).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "micro_f1": f1_score(labels, preds, average="micro"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train_file", default="semeval_train_full.csv")
    parser.add_argument("--dev_file", default="semeval_dev_full.csv")
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/roberta_base"))
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    train_df = read_split(args.data_dir / args.train_file, args.max_train_samples)
    dev_df = read_split(args.data_dir / args.dev_file, args.max_eval_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_pandas(train_df, preserve_index=False).map(
        tokenize, batched=True, remove_columns=["text", "id"]
    )
    dev_ds = Dataset.from_pandas(dev_df, preserve_index=False).map(
        tokenize, batched=True, remove_columns=["text", "id"]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "human", 1: "machine"},
        label2id={"human": 0, "machine": 1},
    )

    training_kwargs = {
        "output_dir": str(args.output_dir),
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": 50,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "fp16": True,
        "report_to": [],
        "seed": args.seed,
    }
    try:
        training_args = TrainingArguments(**training_kwargs)
    except TypeError:
        training_kwargs["evaluation_strategy"] = training_kwargs.pop("eval_strategy")
        training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": dev_ds,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    metrics = trainer.evaluate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "eval_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    predictions = trainer.predict(dev_ds).predictions.argmax(axis=-1)
    pd.DataFrame({"id": dev_df["id"], "label": predictions}).to_csv(
        args.output_dir / "dev_predictions.csv", index=False
    )
    trainer.save_model(args.output_dir / "best_model")
    tokenizer.save_pretrained(args.output_dir / "best_model")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
