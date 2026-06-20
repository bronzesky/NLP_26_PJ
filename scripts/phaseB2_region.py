"""
Phase B-2: region-aware (two-stage) decision vs single global threshold.

Mechanism from diagnostics: in RoBERTa margin space, human upper tail
(p75=11.2, p90=11.9) overlaps bloomz (p50=10.9, p90=12.0), while clean AI
sits >=12.1. A single global threshold must trade human specificity against
bloomz recall. But bloomz is strongly separable from human in ORTHOGONAL
handcrafted features (TTR 0.83 vs 0.43, repeated_bigram 0.01 vs 0.15).

Legitimate routing (uses only observable text features, NOT the model label):
  - margin >= t_high           -> AI   (clean AI region)
  - margin <  t_low            -> human (clean human region)
  - t_low <= margin < t_high   -> AMBIGUOUS: decide with a small classifier
                                   on orthogonal handcrafted features
All thresholds + the ambiguous-region classifier are fit on the calib half;
scored on the disjoint eval half. 5 seeds. Compared against the single-global
-threshold ceiling from Phase A.
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

df = pd.read_csv(REPO / "outputs/analysis/features.csv")
df["margin"] = df["logit_ai"] - df["logit_human"]
df = df.reset_index(drop=True)


def best_single_threshold(margin, y):
    cand = np.quantile(margin, np.linspace(0, 1, 2000))
    mids = (cand[:-1] + cand[1:]) / 2
    best, bt = -1.0, 0.0
    for t in mids:
        s = macro_f1(y, (margin >= t).astype(int))
        if s > best:
            best, bt = s, t
    return bt


def per_gen(ev, pred):
    evp = ev.assign(pred=pred)
    out = {}
    for g in ["GPT4", "bloomz", "chatGPT", "cohere", "davinci", "dolly"]:
        s = evp[evp.model == g]
        out[g] = float((s.pred == 1).mean()) if len(s) else float("nan")
    out["human_spec"] = float((evp[evp.model == "human"].pred == 0).mean())
    return out


def run():
    single_f1, region_f1 = [], []
    single_pg, region_pg = [], []
    band_info = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        cidx = []
        for _, grp in df.groupby(["model", "label"]):
            idx = grp.index.to_numpy(); rng.shuffle(idx)
            cidx.extend(idx[: len(idx) // 2])
        calib = df[df.index.isin(cidx)].copy()
        ev = df[~df.index.isin(cidx)].copy()
        yc, ye = calib.label.values, ev.label.values

        # --- baseline: single global threshold (Phase A) ---
        t_single = best_single_threshold(calib.margin.values, yc)
        pred_s = (ev.margin.values >= t_single).astype(int)
        single_f1.append(macro_f1(ye, pred_s))
        single_pg.append(per_gen(ev, pred_s))

        # --- region-aware: choose ambiguous band [t_low, t_high] on calib ---
        # t_high: above this, calib is almost pure AI; t_low: below, almost pure human
        # use quantiles of margin where precision flips; search a small grid
        best = None
        for t_low in np.quantile(calib.margin.values, [0.3, 0.4, 0.5, 0.6]):
            for t_high in np.quantile(calib.margin.values, [0.75, 0.8, 0.85, 0.9]):
                if t_high <= t_low:
                    continue
                amb = (calib.margin.values >= t_low) & (calib.margin.values < t_high)
                if amb.sum() < 50 or len(np.unique(yc[amb])) < 2:
                    continue
                clf = LogisticRegression(max_iter=1000, class_weight="balanced")
                clf.fit(calib.loc[amb, ORTHO].values, yc[amb])
                # build calib prediction to score this config
                p = np.where(calib.margin.values >= t_high, 1,
                             np.where(calib.margin.values < t_low, 0, -1))
                p[amb] = clf.predict(calib.loc[amb, ORTHO].values)
                f1c = macro_f1(yc, p)
                if best is None or f1c > best[0]:
                    best = (f1c, t_low, t_high, clf)
        _, t_low, t_high, clf = best
        amb_e = (ev.margin.values >= t_low) & (ev.margin.values < t_high)
        pred_r = np.where(ev.margin.values >= t_high, 1,
                          np.where(ev.margin.values < t_low, 0, -1))
        pred_r[amb_e] = clf.predict(ev.loc[amb_e, ORTHO].values)
        region_f1.append(macro_f1(ye, pred_r))
        region_pg.append(per_gen(ev, pred_r))
        band_info.append((t_low, t_high, amb_e.mean()))

    def avg_pg(lst):
        keys = lst[0].keys()
        return {k: round(float(np.mean([d[k] for d in lst])), 4) for k in keys}

    print("=== single global threshold (Phase A baseline) ===")
    print(f"  macro-F1 = {np.mean(single_f1):.4f} ± {np.std(single_f1):.4f}")
    print(f"  per-gen  = {avg_pg(single_pg)}")
    print("\n=== region-aware two-stage (ortho features in ambiguous band) ===")
    print(f"  macro-F1 = {np.mean(region_f1):.4f} ± {np.std(region_f1):.4f}")
    print(f"  per-gen  = {avg_pg(region_pg)}")
    tl = np.mean([b[0] for b in band_info]); th = np.mean([b[1] for b in band_info])
    fr = np.mean([b[2] for b in band_info])
    print(f"\n  ambiguous band: [{tl:.2f}, {th:.2f}], {fr*100:.1f}% of eval routed to ortho classifier")


if __name__ == "__main__":
    run()
