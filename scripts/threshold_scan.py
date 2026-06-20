"""
Phase 0 diagnostic: threshold scan.
Reuse stored dev/test prob_ai. Find best threshold on dev, apply to test.
Goal: confirm RoBERTa's low test accuracy is an operating-point problem,
not a representation problem (AUROC should stay constant).
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


def metrics_at(thr, y, p):
    pred = (p >= thr).astype(int)
    return {
        "threshold": float(thr),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
    }


def scan(name, dev_csv, test_csv):
    dev = pd.read_csv(dev_csv)
    test = pd.read_csv(test_csv)
    yd, pd_ = dev["label"].values, dev["prob_ai"].values
    yt, pt = test["label"].values, test["prob_ai"].values

    auroc_dev = roc_auc_score(yd, pd_)
    auroc_test = roc_auc_score(yt, pt)

    # baseline @ 0.5
    base_dev = metrics_at(0.5, yd, pd_)
    base_test = metrics_at(0.5, yt, pt)

    # scan dev for best macro-F1 and best accuracy
    grid = np.linspace(0.01, 0.99, 197)
    dev_scores = [metrics_at(t, yd, pd_) for t in grid]
    best_f1 = max(dev_scores, key=lambda d: d["macro_f1"])
    best_acc = max(dev_scores, key=lambda d: d["accuracy"])

    # apply dev-optimal thresholds to test
    test_at_f1thr = metrics_at(best_f1["threshold"], yt, pt)
    test_at_accthr = metrics_at(best_acc["threshold"], yt, pt)

    print(f"\n========== {name} ==========")
    print(f"AUROC  dev={auroc_dev:.4f}  test={auroc_test:.4f}  (unchanged by threshold)")
    print(f"--- @ threshold 0.5 (default) ---")
    print(f"  dev : acc={base_dev['accuracy']:.4f}  macroF1={base_dev['macro_f1']:.4f}")
    print(f"  test: acc={base_test['accuracy']:.4f}  macroF1={base_test['macro_f1']:.4f}")
    print(f"--- dev-optimal threshold (by macro-F1) = {best_f1['threshold']:.3f} ---")
    print(f"  dev : acc={best_f1['accuracy']:.4f}  macroF1={best_f1['macro_f1']:.4f}")
    print(f"  test: acc={test_at_f1thr['accuracy']:.4f}  macroF1={test_at_f1thr['macro_f1']:.4f}")
    print(f"--- dev-optimal threshold (by accuracy) = {best_acc['threshold']:.3f} ---")
    print(f"  test: acc={test_at_accthr['accuracy']:.4f}  macroF1={test_at_accthr['macro_f1']:.4f}")
    # oracle: best possible on test itself (upper bound)
    test_scores = [metrics_at(t, yt, pt) for t in grid]
    oracle = max(test_scores, key=lambda d: d["macro_f1"])
    print(f"--- ORACLE test threshold = {oracle['threshold']:.3f} (upper bound) ---")
    print(f"  test: acc={oracle['accuracy']:.4f}  macroF1={oracle['macro_f1']:.4f}")


if __name__ == "__main__":
    base = "outputs"
    scan("RoBERTa-base (single-pass)",
         f"{base}/roberta_base_dev_predict/predictions.csv",
         f"{base}/roberta_base_test/predictions.csv")
    # chunked has no dev_predict csv with same schema; scan test oracle only if dev missing
    import os
    chunk_dev = f"{base}/roberta_base_chunked_dev_predict/predictions.csv"
    chunk_test = f"{base}/roberta_base_chunked_test/predictions.csv"
    if os.path.exists(chunk_dev):
        scan("RoBERTa-base (chunked)", chunk_dev, chunk_test)
    else:
        print("\n[chunked] no chunked dev_predict csv; using single-pass dev threshold on chunked test")
        scan("RoBERTa-chunked (dev thr from single-pass)",
             f"{base}/roberta_base_dev_predict/predictions.csv", chunk_test)
