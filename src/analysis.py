from __future__ import annotations

from pathlib import Path

import pandas as pd


TRUE_LABEL_CANDIDATES = ("true_label", "gold_label", "y_true", "target", "label_true")
PRED_LABEL_CANDIDATES = ("pred_label", "prediction", "y_pred", "label_pred", "pred")
PROB_CANDIDATES = (
    "prob_ai",
    "prob_machine",
    "machine_prob",
    "prob_1",
    "p_machine",
    "score",
)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path, encoding="utf-8-sig")


def merge_predictions(predictions: pd.DataFrame, data: pd.DataFrame | None) -> pd.DataFrame:
    df = predictions.copy()
    if data is None:
        return df
    if "id" not in df.columns or "id" not in data.columns:
        raise ValueError("Both predictions and data_file must contain an id column.")

    keep = ["id"]
    keep.extend(
        col
        for col in ("label", "text", "model", "source", "domain")
        if col in data.columns and col not in df.columns
    )
    data_part = data[keep].copy()
    rename = {}
    if "label" in data_part.columns and "label" in df.columns:
        rename["label"] = "true_label"
    data_part = data_part.rename(columns=rename)
    return df.merge(data_part, on="id", how="left")


def infer_columns(df: pd.DataFrame) -> tuple[str, str]:
    true_col = _first_existing(df, TRUE_LABEL_CANDIDATES)
    pred_col = _first_existing(df, PRED_LABEL_CANDIDATES)

    if pred_col is None and "label" in df.columns:
        pred_col = "label" if true_col is not None else None
    if true_col is None and "label" in df.columns and pred_col != "label":
        true_col = "label"

    if true_col is None or pred_col is None:
        raise ValueError(
            "Could not infer true/predicted label columns. Use columns like "
            "true_label + pred_label, or pass --data_file so prediction label can be merged."
        )
    return true_col, pred_col


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "text" in out.columns:
        word_counts = out["text"].fillna("").astype(str).str.findall(r"\b\w+\b").str.len()
        out["length_bucket"] = pd.cut(
            word_counts,
            bins=[-1, 50, 100, 200, 400, 800, 10**9],
            labels=["0-50", "51-100", "101-200", "201-400", "401-800", "801+"],
        ).astype(str)
    elif "length_bucket" not in out.columns:
        out["length_bucket"] = "unknown"

    conf = confidence_series(out)
    if conf is None:
        out["confidence_bucket"] = "unknown"
    else:
        out["confidence"] = conf
        out["confidence_bucket"] = pd.cut(
            conf,
            bins=[-0.001, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001],
            labels=["<=0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"],
        ).astype(str)

    for col in ("domain", "model", "source"):
        if col not in out.columns:
            out[col] = "unknown"
        out[col] = out[col].fillna("unknown").astype(str)
    return out


def confidence_series(df: pd.DataFrame) -> pd.Series | None:
    prob_col = _first_existing(df, PROB_CANDIDATES)
    if prob_col is not None:
        probs = pd.to_numeric(df[prob_col], errors="coerce")
        return probs.apply(lambda value: max(value, 1.0 - value) if pd.notna(value) else value)
    if "confidence" in df.columns:
        return pd.to_numeric(df["confidence"], errors="coerce")
    return None


def binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    truth = pd.to_numeric(y_true, errors="coerce")
    pred = pd.to_numeric(y_pred, errors="coerce")
    valid = truth.notna() & pred.notna()
    truth = truth[valid].astype(int)
    pred = pred[valid].astype(int)
    n = int(len(truth))
    if n == 0:
        return _empty_metrics()

    tp = int(((truth == 1) & (pred == 1)).sum())
    tn = int(((truth == 0) & (pred == 0)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    f1_0 = _f1(tn, fn, fp)
    f1_1 = _f1(tp, fp, fn)
    micro_f1 = (tp + tn) / n
    return {
        "rows": n,
        "accuracy": (tp + tn) / n,
        "macro_f1": (f1_0 + f1_1) / 2.0,
        "micro_f1": micro_f1,
        "precision_machine": _safe_div(tp, tp + fp),
        "recall_machine": _safe_div(tp, tp + fn),
        "f1_machine": f1_1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def grouped_metrics(df: pd.DataFrame, group_cols: list[str], true_col: str, pred_col: str) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(binary_metrics(group[true_col], group[pred_col]))
        rows.append(row)
    return pd.DataFrame(rows)


def _first_existing(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _safe_div(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return _safe_div(2 * precision * recall, precision + recall)


def _empty_metrics() -> dict[str, float | int]:
    return {
        "rows": 0,
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "micro_f1": 0.0,
        "precision_machine": 0.0,
        "recall_machine": 0.0,
        "f1_machine": 0.0,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
    }
