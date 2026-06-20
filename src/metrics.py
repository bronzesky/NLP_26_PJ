from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class ReliabilityBin:
    bin: int
    lower: float
    upper: float
    count: int
    confidence: float
    accuracy: float
    positive_rate: float
    gap: float


def _as_1d(array, name: str) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {values.shape}")
    return values


def _as_binary_labels(y_true) -> np.ndarray:
    labels = _as_1d(y_true, "y_true").astype(int)
    unique = set(np.unique(labels).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"y_true must contain only 0/1 labels, got {sorted(unique)}")
    return labels


def sigmoid(logits) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    out = np.empty_like(logits, dtype=float)
    positive = logits >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_x = np.exp(logits[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def softmax(logits) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    if logits.ndim == 1:
        return sigmoid(logits)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"logits must have shape (n,) or (n, 2), got {logits.shape}")
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def probability_of_positive(scores) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim == 1:
        if np.nanmin(values) >= 0.0 and np.nanmax(values) <= 1.0:
            return values
        return sigmoid(values)
    return softmax(values)[:, 1]


def binary_predictions(scores, threshold: float = 0.5) -> np.ndarray:
    return (probability_of_positive(scores) >= threshold).astype(int)


def accuracy(y_true, y_pred) -> float:
    labels = _as_binary_labels(y_true)
    preds = _as_1d(y_pred, "y_pred").astype(int)
    if labels.shape != preds.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(labels == preds))


def _f1_for_label(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float:
    tp = np.sum((y_true == label) & (y_pred == label))
    fp = np.sum((y_true != label) & (y_pred == label))
    fn = np.sum((y_true == label) & (y_pred != label))
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float((2 * tp) / denom)


def macro_f1(y_true, y_pred) -> float:
    labels = _as_binary_labels(y_true)
    preds = _as_1d(y_pred, "y_pred").astype(int)
    if labels.shape != preds.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float((_f1_for_label(labels, preds, 0) + _f1_for_label(labels, preds, 1)) / 2.0)


def micro_f1(y_true, y_pred) -> float:
    return accuracy(y_true, y_pred)


def auroc(y_true, y_score) -> float:
    labels = _as_binary_labels(y_true)
    scores = probability_of_positive(y_score)
    if labels.shape != scores.shape:
        raise ValueError("y_true and y_score must have the same shape")

    positives = labels == 1
    n_pos = int(np.sum(positives))
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    rank_sum_pos = float(np.sum(ranks[positives]))
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fpr_at_95_tpr(y_true, y_score, target_tpr: float = 0.95) -> float:
    labels = _as_binary_labels(y_true)
    scores = probability_of_positive(y_score)
    if labels.shape != scores.shape:
        raise ValueError("y_true and y_score must have the same shape")

    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    best_fpr = math.inf
    for threshold in thresholds:
        pred_pos = scores >= threshold
        tpr = np.sum(pred_pos & (labels == 1)) / n_pos
        if tpr >= target_tpr:
            fpr = np.sum(pred_pos & (labels == 0)) / n_neg
            best_fpr = min(best_fpr, float(fpr))
    return float(best_fpr) if math.isfinite(best_fpr) else float("nan")


def brier(y_true, y_prob) -> float:
    labels = _as_binary_labels(y_true).astype(float)
    probs = np.clip(probability_of_positive(y_prob), 0.0, 1.0)
    if labels.shape != probs.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    return float(np.mean((probs - labels) ** 2))


def nll(y_true, y_prob) -> float:
    labels = _as_binary_labels(y_true).astype(float)
    probs = np.clip(probability_of_positive(y_prob), EPS, 1.0 - EPS)
    if labels.shape != probs.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    losses = -(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
    return float(np.mean(losses))


def reliability_bins(y_true, y_prob, n_bins: int = 15) -> list[ReliabilityBin]:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    labels = _as_binary_labels(y_true)
    probs = np.clip(probability_of_positive(y_prob), 0.0, 1.0)
    if labels.shape != probs.shape:
        raise ValueError("y_true and y_prob must have the same shape")

    bins: list[ReliabilityBin] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    preds = (probs >= 0.5).astype(int)
    confidences = np.where(preds == 1, probs, 1.0 - probs)
    correct = (preds == labels).astype(float)

    for idx in range(n_bins):
        lower = float(edges[idx])
        upper = float(edges[idx + 1])
        if idx == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        count = int(np.sum(mask))
        if count:
            conf = float(np.mean(confidences[mask]))
            acc = float(np.mean(correct[mask]))
            pos_rate = float(np.mean(labels[mask]))
            gap = abs(acc - conf)
        else:
            conf = acc = pos_rate = gap = float("nan")
        bins.append(ReliabilityBin(idx, lower, upper, count, conf, acc, pos_rate, gap))
    return bins


def ece(y_true, y_prob, n_bins: int = 15) -> float:
    labels = _as_binary_labels(y_true)
    total = len(labels)
    if total == 0:
        return float("nan")
    value = 0.0
    for item in reliability_bins(labels, y_prob, n_bins=n_bins):
        if item.count:
            value += (item.count / total) * item.gap
    return float(value)


def binary_metrics(y_true, scores, threshold: float = 0.5, n_bins: int = 15) -> dict[str, float]:
    probs = probability_of_positive(scores)
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": accuracy(y_true, preds),
        "macro_f1": macro_f1(y_true, preds),
        "micro_f1": micro_f1(y_true, preds),
        "auroc": auroc(y_true, probs),
        "fpr_at_95_tpr": fpr_at_95_tpr(y_true, probs),
        "brier": brier(y_true, probs),
        "nll": nll(y_true, probs),
        "ece": ece(y_true, probs, n_bins=n_bins),
    }
