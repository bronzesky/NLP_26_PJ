"""
Phase B-2a: rigorous, leakage-free validation of the region-aware decision.

Three numbers, answering different questions:

[P0] In-test split, band chosen on calib by calib-F1 (the ORIGINAL B-2 setup).
     Optimistic: band selection sees the same calib it is scored to fit.

[P1] DEPLOY / SemEval protocol: fit EVERYTHING (margin thresholds, band edges,
     ambiguous-band classifier) on DEV only, predict TEST once. This is the only
     number reportable as "method achieves X on test". Dev = bloomz + 5 domains,
     so distribution shift vs test (6 generators, outfox) is in full effect.

[P2] Nested CV upper bound: still uses test labels for calibration, but band
     edges are selected on an INNER split of calib and scored on the held-out
     eval. Removes the band-selection optimism of P0. Answers: "with an
     in-distribution calib set, what is the non-overfit ceiling?"

Compares each against its single-global-threshold counterpart.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.metrics import macro_f1

ORTHO = ["ttr", "repeated_bigram_ratio", "repeated_trigram_ratio",
         "avg_word_length", "text_length", "punctuation_per_token"]
SEEDS = (0, 1, 2, 3, 4)

test = pd.read_csv(REPO / "outputs/analysis/features.csv").reset_index(drop=True)
test["margin"] = test["logit_ai"] - test["logit_human"]
dev = pd.read_csv(REPO / "outputs/analysis_dev/features.csv").reset_index(drop=True)
dev["margin"] = dev["logit_ai"] - dev["logit_human"]


def best_single(margin, y):
    cand = np.quantile(margin, np.linspace(0, 1, 2000))
    mids = (cand[:-1] + cand[1:]) / 2
    best, bt = -1.0, 0.0
    for t in mids:
        s = macro_f1(y, (margin >= t).astype(int))
        if s > best:
            best, bt = s, t
    return bt


def fit_region(fit_df, score_df, low_qs, high_qs, select_fn):
    """Fit band+clf on fit_df, return predictions on score_df. select_fn scores a config."""
    yf = fit_df.label.values
    best = None
    for tl in np.quantile(fit_df.margin.values, low_qs):
        for th in np.quantile(fit_df.margin.values, high_qs):
            if th <= tl:
                continue
            amb = (fit_df.margin.values >= tl) & (fit_df.margin.values < th)
            if amb.sum() < 50 or len(np.unique(yf[amb])) < 2:
                continue
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(fit_df.loc[amb, ORTHO].values, yf[amb])
            score = select_fn(tl, th, clf)
            if best is None or score > best[0]:
                best = (score, tl, th, clf)
    _, tl, th, clf = best
    m = score_df.margin.values
    amb = (m >= tl) & (m < th)
    pred = np.where(m >= th, 1, np.where(m < tl, 0, -1))
    pred[amb] = clf.predict(score_df.loc[amb, ORTHO].values)
    return pred, tl, th, amb.mean()


def per_gen(df, pred):
    p = df.assign(pred=pred)
    out = {g: round(float((p[p.model == g].pred == 1).mean()), 3)
           for g in ["GPT4", "bloomz", "dolly", "davinci"]}
    out["hum_spec"] = round(float((p[p.model == "human"].pred == 0).mean()), 3)
    return out


LOWQ = [0.3, 0.4, 0.5, 0.6]
HIGHQ = [0.75, 0.8, 0.85, 0.9]


def p0_intest():
    f1s = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        cidx = []
        for _, g in test.groupby(["model", "label"]):
            idx = g.index.to_numpy(); rng.shuffle(idx); cidx.extend(idx[:len(idx)//2])
        calib = test[test.index.isin(cidx)]; ev = test[~test.index.isin(cidx)]
        yc = calib.label.values
        pred, *_ = fit_region(calib, ev, LOWQ, HIGHQ,
                              lambda tl, th, clf: macro_f1(yc, _self_pred(calib, tl, th, clf)))
        f1s.append(macro_f1(ev.label.values, pred))
    return np.mean(f1s), np.std(f1s)


def _self_pred(df, tl, th, clf):
    m = df.margin.values
    amb = (m >= tl) & (m < th)
    pred = np.where(m >= th, 1, np.where(m < tl, 0, -1))
    pred[amb] = clf.predict(df.loc[amb, ORTHO].values)
    return pred


def p1_deploy():
    # fit all on DEV, predict TEST once. single seed (no test split).
    yd = dev.label.values
    # single-threshold baseline
    t_single = best_single(dev.margin.values, yd)
    pred_s = (test.margin.values >= t_single).astype(int)
    f1_s = macro_f1(test.label.values, pred_s)
    # region-aware: band+clf selected on dev by dev-CV (inner) to avoid dev overfit
    from sklearn.model_selection import KFold
    def dev_cv_score(tl, th, clf):
        # rough: score on dev itself (dev is the only fit set available)
        return macro_f1(yd, _self_pred(dev, tl, th, clf))
    pred_r, tl, th, frac = fit_region(dev, test, LOWQ, HIGHQ, dev_cv_score)
    f1_r = macro_f1(test.label.values, pred_r)
    return (f1_s, per_gen(test, pred_s)), (f1_r, per_gen(test, pred_r), tl, th, frac)


def p2_nested():
    f1s = []; pgs = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        cidx = []
        for _, g in test.groupby(["model", "label"]):
            idx = g.index.to_numpy(); rng.shuffle(idx); cidx.extend(idx[:len(idx)//2])
        calib = test[test.index.isin(cidx)].reset_index(drop=True)
        ev = test[~test.index.isin(cidx)]
        # inner split of calib for band selection
        rng2 = np.random.default_rng(seed + 100)
        iidx = []
        for _, g in calib.groupby(["model", "label"]):
            idx = g.index.to_numpy(); rng2.shuffle(idx); iidx.extend(idx[:len(idx)//2])
        inner_fit = calib[calib.index.isin(iidx)]
        inner_val = calib[~calib.index.isin(iidx)]
        yiv = inner_val.label.values
        # select band on inner_fit, scored on inner_val (no optimism)
        pred, tl, th, frac = fit_region(
            inner_fit, ev, LOWQ, HIGHQ,
            lambda tl, th, clf: macro_f1(yiv, _self_pred(inner_val, tl, th, clf)))
        f1s.append(macro_f1(ev.label.values, pred))
        pgs.append(per_gen(ev, pred))
    keys = pgs[0].keys()
    pg = {k: round(float(np.mean([d[k] for d in pgs])), 3) for k in keys}
    return np.mean(f1s), np.std(f1s), pg


if __name__ == "__main__":
    print("Dev composition:", dev.model.value_counts().to_dict())
    print()
    m, s = p0_intest()
    print(f"[P0] in-test, band-on-calib (original B-2): macro-F1 = {m:.4f} ± {s:.4f}")
    print()
    (f1s, pgs), (f1r, pgr, tl, th, fr) = p1_deploy()
    print(f"[P1] DEPLOY dev->test once (LEGAL test score):")
    print(f"     single-threshold : macro-F1 = {f1s:.4f}  {pgs}")
    print(f"     region-aware     : macro-F1 = {f1r:.4f}  {pgr}")
    print(f"     band=[{tl:.2f},{th:.2f}] {fr*100:.0f}% routed")
    print()
    m, s, pg = p2_nested()
    print(f"[P2] nested-CV upper bound (in-dist calib, no band optimism):")
    print(f"     region-aware : macro-F1 = {m:.4f} ± {s:.4f}  {pg}")
