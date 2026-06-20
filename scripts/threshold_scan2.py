"""
Phase 0 diagnostic v2: scan threshold in LOGIT-MARGIN space using actual
data values as candidates (proper ROC sweep). The prob-space grid in v1
saturated near 0/1 and missed the real decision point.

margin = logit_ai - logit_human  (single-pass: from stored logits;
         chunked: reconstructed as log(p_ai) - log(p_human))
accuracy/macro-F1 depend only on the margin threshold (monotonic), so
temperature scaling does NOT change them — it only affects calibration
(ECE/Brier), reported separately.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, brier_score_loss


def get_margin(df):
    if "logit_ai" in df.columns and "logit_human" in df.columns:
        return (df["logit_ai"] - df["logit_human"]).values
    p = np.clip(df["prob_ai"].values, 1e-12, 1 - 1e-12)
    return np.log(p) - np.log(1 - p)


def best_threshold(margin, y, by="macro_f1"):
    cand = np.unique(margin)
    if len(cand) > 4000:
        cand = np.quantile(margin, np.linspace(0, 1, 4000))
    mids = (cand[:-1] + cand[1:]) / 2
    cands = np.concatenate([[cand[0] - 1], mids, [cand[-1] + 1]])
    best, best_t = -1, 0.0
    for t in cands:
        pred = (margin >= t).astype(int)
        s = f1_score(y, pred, average="macro") if by == "macro_f1" else accuracy_score(y, pred)
        if s > best:
            best, best_t = s, t
    return best_t


def report(tag, margin, y):
    pred = (margin >= 0).astype(int)  # margin>=0 <=> prob_ai>=0.5
    return {
        "acc": accuracy_score(y, pred),
        "f1": f1_score(y, pred, average="macro"),
    }


def at(margin, y, t):
    pred = (margin >= t).astype(int)
    return accuracy_score(y, pred), f1_score(y, pred, average="macro")


def scan(name, dev_csv, test_csv):
    dev, test = pd.read_csv(dev_csv), pd.read_csv(test_csv)
    md, yd = get_margin(dev), dev["label"].values
    mt, yt = get_margin(test), test["label"].values

    print(f"\n========== {name} ==========")
    print(f"AUROC  dev={roc_auc_score(yd, md):.4f}  test={roc_auc_score(yt, mt):.4f}")
    b = report("", md, yd); print(f"@0.5  dev : acc={b['acc']:.4f}  macroF1={b['f1']:.4f}")
    b = report("", mt, yt); print(f"@0.5  test: acc={b['acc']:.4f}  macroF1={b['f1']:.4f}")

    t_f1 = best_threshold(md, yd, "macro_f1")
    a, f = at(mt, yt, t_f1)
    da, df_ = at(md, yd, t_f1)
    print(f"dev-opt margin thr (macroF1)={t_f1:+.3f} -> dev acc={da:.4f} F1={df_:.4f} | test acc={a:.4f} F1={f:.4f}")

    t_oracle = best_threshold(mt, yt, "macro_f1")
    a, f = at(mt, yt, t_oracle)
    print(f"ORACLE test margin thr={t_oracle:+.3f} -> test acc={a:.4f} F1={f:.4f} (upper bound)")


if __name__ == "__main__":
    base = "outputs"
    scan("RoBERTa-base (single-pass)",
         f"{base}/roberta_base_dev_predict/predictions.csv",
         f"{base}/roberta_base_test/predictions.csv")
    chunk_dev = f"{base}/roberta_base_chunked_dev_predict/predictions.csv"
    chunk_test = f"{base}/roberta_base_chunked_test/predictions.csv"
    if os.path.exists(chunk_dev):
        scan("RoBERTa-base (chunked)", chunk_dev, chunk_test)
    else:
        scan("RoBERTa-chunked (dev thr from single-pass dev)",
             f"{base}/roberta_base_dev_predict/predictions.csv", chunk_test)
