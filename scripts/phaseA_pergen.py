"""
Phase A addendum: correct per-generator breakdown.
macro-F1 per generator is degenerate (each generator subset is single-class).
Report instead: per-generator RECALL (AI detection rate) at the calibrated
threshold, and human SPECIFICITY (true-negative rate). Averaged over seeds.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.metrics import macro_f1


def best_threshold(margin, y):
    cand = np.unique(margin)
    if len(cand) > 3000:
        cand = np.quantile(margin, np.linspace(0, 1, 3000))
    mids = (cand[:-1] + cand[1:]) / 2
    cands = np.concatenate([[cand[0] - 1], mids, [cand[-1] + 1]])
    best, bt = -1, 0.0
    for t in cands:
        s = macro_f1(y, (margin >= t).astype(int))
        if s > best:
            best, bt = s, t
    return bt


def run(name, test_csv, seeds=(0, 1, 2, 3, 4)):
    df = pd.read_csv(test_csv).reset_index(drop=True)
    if "logit_ai" in df.columns:
        df["margin"] = df["logit_ai"] - df["logit_human"]
    else:
        df["margin"] = np.log(np.clip(df["prob_ai"], 1e-12, 1)) - np.log(np.clip(1 - df["prob_ai"], 1e-12, 1))

    gens = [m for m in df["model"].unique() if m != "human"]
    rec = {g: [] for g in gens}
    spec = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        cidx = []
        for _, grp in df.groupby(["model", "label"]):
            idx = grp.index.to_numpy(); rng.shuffle(idx)
            cidx.extend(idx[:len(idx)//2])
        calib = df[df.index.isin(cidx)]; ev = df[~df.index.isin(cidx)]
        t = best_threshold(calib["margin"].values, calib["label"].values)
        ev = ev.assign(pred=(ev["margin"].values >= t).astype(int))
        for g in gens:
            sub = ev[ev["model"] == g]  # all label==1
            rec[g].append((sub["pred"] == 1).mean())
        hum = ev[ev["model"] == "human"]  # all label==0
        spec.append((hum["pred"] == 0).mean())

    print(f"\n=== {name} ===  (calibrated threshold, mean over {len(seeds)} seeds)")
    print(f"  human specificity (TNR) = {np.mean(spec):.4f}")
    print(f"  per-generator AI recall (TPR):")
    for g in sorted(gens):
        print(f"    {g:10s} {np.mean(rec[g]):.4f}")


if __name__ == "__main__":
    run("RoBERTa single", "outputs/roberta_base_test/predictions.csv")
    run("RoBERTa chunked", "outputs/roberta_base_chunked_test/predictions.csv")
    run("TF-IDF", "outputs/tfidf_test/predictions.csv")
