import json
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

try:
    from src.metrics import binary_metrics
except Exception:
    binary_metrics = None


GENERATED_COLUMNS = {
    "pred",
    "prob_human",
    "prob_ai",
    "logit_human",
    "logit_ai",
    "correct",
}


def read_prediction_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")

    required = {"id", "text"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    if "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    return df


def build_prediction_frame(
    df: pd.DataFrame,
    pred,
    prob_human,
    prob_ai,
    logit_human=None,
    logit_ai=None,
    include_text: bool = False,
) -> pd.DataFrame:
    pred = np.asarray(pred, dtype=int)
    if len(pred) != len(df):
        raise ValueError(f"prediction count {len(pred)} does not match input rows {len(df)}")

    metadata_cols = [col for col in df.columns if col not in GENERATED_COLUMNS]
    if not include_text:
        metadata_cols = [col for col in metadata_cols if col != "text"]
    out = df[metadata_cols].copy()
    out["pred"] = pred
    out["prob_human"] = np.asarray(prob_human, dtype=float)
    out["prob_ai"] = np.asarray(prob_ai, dtype=float)

    if logit_human is not None and logit_ai is not None:
        out["logit_human"] = np.asarray(logit_human, dtype=float)
        out["logit_ai"] = np.asarray(logit_ai, dtype=float)

    if "label" in out.columns:
        out["correct"] = out["label"].astype(int).to_numpy() == pred

    preferred = [
        "id",
        "text",
        "label",
        "pred",
        "prob_human",
        "prob_ai",
        "logit_human",
        "logit_ai",
        "correct",
        "model",
        "domain",
        "source",
    ]
    ordered = [col for col in preferred if col in out.columns]
    ordered.extend(col for col in out.columns if col not in ordered)
    return out[ordered]


def write_prediction_outputs(output_dir: Path, predictions: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        output_dir / "predictions.csv",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
    )
    predictions.to_json(
        output_dir / "predictions.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )


def write_metrics_if_labeled(
    output_dir: Path, predictions: pd.DataFrame
) -> Optional[dict]:
    if "label" not in predictions.columns:
        return None

    labels = predictions["label"].astype(int)
    pred = predictions["pred"].astype(int)
    if binary_metrics is not None and "prob_ai" in predictions.columns:
        metrics = binary_metrics(labels, predictions["prob_ai"])
        metrics["rows"] = int(len(predictions))
    else:
        metrics = {
            "accuracy": float(accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, average="macro")),
            "micro_f1": float(f1_score(labels, pred, average="micro")),
            "rows": int(len(predictions)),
        }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics
