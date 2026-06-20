"""
Phase A: ceiling estimate with an IN-DISTRIBUTION calibration set.

The honest question: if we HAD a labeled calibration set drawn from the same
distribution as test (unseen generators + outfox domain), how well can the
already-trained RoBERTa be fixed by temperature scaling + threshold selection,
without retraining?

Method: stratified 50/50 split of test into calib / eval (stratify by
model x label so every generator appears in both halves). Fit temperature and
pick macro-F1-optimal threshold on calib, evaluate on eval. Average over seeds.
Also report per-generator macro-F1 after calibration to see who remains hard.

No data leakage: threshold/temperature chosen on calib, scored on disjoint eval.
This is an upper bound for "fixed model + good calib set", strictly below the
all-test oracle but realistic.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.metrics import binary_metrics, macro_f1
from src.calibration import TemperatureScaler


def best_threshold_on_margin(margin, y):
    cand = np.unique(margin)
    if len(cand) > 3000:
        cand = np.quantile(margin, np.linspace(0, 1, 3000))
    mids = (cand[:-1] + cand[1:]) / 2
    cands = np.concatenate([[cand[0] - 1], mids, [cand[-1] + 1]])
    best, best_t = -1, 0.0
    for t in cands:
        s = macro_f1(y, (margin >= t).astype(int))
        if s > best:
            best, best_t = s, t
    return best_t


def stratified_half(df, seed):
    rng = np.random.default_rng(seed)
    calib_idx = []
    for _, grp in df.groupby(["model", "label"]):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        calib_idx.extend(idx[: len(idx) // 2])
    calib_mask = df.index.isin(calib_idx)
    return df[calib_mask], df[~calib_mask]


def run(test_csv, seeds=(0, 1, 2, 3, 4)):
    df = pd.read_csv(test_csv).reset_index(drop=True)
    has_logits = "logit_ai" in df.columns and "logit_human" in df.columns
    df["margin"] = (df["logit_ai"] - df["logit_human"]) if has_logits else (
        np.log(np.clip(df["prob_ai"], 1e-12, 1)) - np.log(np.clip(1 - df["prob_ai"], 1e-12, 1)))

    base = binary_metrics(df["label"].values, df["prob_ai"].values, threshold=0.5)
    print(f"\n=== {test_csv} ===")
    print(f"[baseline @0.5, full test] acc={base['accuracy']:.4f} "
          f"macroF1={base['macro_f1']:.4f} AUROC={base['auroc']:.4f} ECE={base['ece']:.4f}")

    accs, f1s, eces, temps, thrs = [], [], [], [], []
    per_gen = {}
    for seed in seeds:
        calib, ev = stratified_half(df, seed)
        # temperature on calib (needs (n,2) logits)
        if has_logits:
            cl = np.stack([calib["logit_human"].values, calib["logit_ai"].values], 1)
            el = np.stack([ev["logit_human"].values, ev["logit_ai"].values], 1)
            ts = TemperatureScaler().fit(cl, calib["label"].values)
            ev_p = ts.predict_positive_proba(el)
            temps.append(ts.temperature)
            ev_margin = el[:, 1] / ts.temperature - el[:, 0] / ts.temperature
        else:
            ev_p = ev["prob_ai"].values
            ev_margin = ev["margin"].values
            temps.append(1.0)
        t = best_threshold_on_margin(calib["margin"].values, calib["label"].values)
        thrs.append(t)
        pred = (ev["margin"].values >= t).astype(int)
        m = binary_metrics(ev["label"].values, ev_p, threshold=0.5)  # ece on calibrated prob
        accs.append((pred == ev["label"].values).mean())
        f1s.append(macro_f1(ev["label"].values, pred))
        eces.append(m["ece"])
        for gen, g in ev.assign(_pred=pred).groupby("model"):
            per_gen.setdefault(gen, []).append(macro_f1(g["label"].values, g["_pred"].values))

    print(f"[calibrated, dev'=test-half, mean±std over {len(seeds)} seeds]")
    print(f"  acc    = {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  macroF1= {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  ECE    = {np.mean(eces):.4f} ± {np.std(eces):.4f}")
    print(f"  temp   = {np.mean(temps):.3f}   margin-thr = {np.mean(thrs):+.3f}")
    print(f"[per-generator macroF1 after calibration]")
    for gen in sorted(per_gen):
        v = per_gen[gen]
        print(f"  {gen:10s} {np.mean(v):.4f} ± {np.std(v):.4f}")


if __name__ == "__main__":
    run("outputs/roberta_base_test/predictions.csv")
    run("outputs/roberta_base_chunked_test/predictions.csv")
    run("outputs/tfidf_test/predictions.csv")
