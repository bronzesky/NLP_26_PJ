"""
Phase A: operating-point ceiling analysis. Reproducible, writes JSON.

Question answered: given the trained model frozen, how much of the test-set
macro-F1 gap is just a wrong operating point (threshold/temperature) vs a
representation failure?

Method: stratified 50/50 split of TEST into calib/eval (stratify by model x
label so every generator is in both halves). Pick macro-F1-optimal margin
threshold + temperature on calib, score on disjoint eval. Mean over seeds.
Reports overall metrics, per-generator AI recall (TPR), human specificity.

No leakage: calib and eval are disjoint; the all-test oracle (reported for
reference) is a strict upper bound above this realistic ceiling.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.metrics import binary_metrics, macro_f1
from src.calibration import TemperatureScaler

RUNS = {
    "roberta_single": "outputs/roberta_base_test/predictions.csv",
    "roberta_chunked": "outputs/roberta_base_chunked_test/predictions.csv",
    "tfidf": "outputs/tfidf_test/predictions.csv",
    "fusion_lgbm": "outputs/fusion_lgbm_test/predictions.csv",
    "fusion_lr": "outputs/fusion_lr_test/predictions.csv",
}
SEEDS = (0, 1, 2, 3, 4)
OUT = REPO / "outputs/phaseA_ceiling"


def margin_of(df):
    if "logit_ai" in df.columns and "logit_human" in df.columns:
        return (df["logit_ai"] - df["logit_human"]).values
    p = np.clip(df["prob_ai"].values, 1e-12, 1 - 1e-12)
    return np.log(p) - np.log(1 - p)


def best_threshold(margin, y):
    cand = np.unique(margin)
    if len(cand) > 3000:
        cand = np.quantile(margin, np.linspace(0, 1, 3000))
    mids = (cand[:-1] + cand[1:]) / 2
    cands = np.concatenate([[cand[0] - 1], mids, [cand[-1] + 1]])
    best, bt = -1.0, 0.0
    for t in cands:
        s = macro_f1(y, (margin >= t).astype(int))
        if s > best:
            best, bt = s, t
    return bt


def analyze(name, csv):
    df = pd.read_csv(REPO / csv).reset_index(drop=True)
    has_logits = "logit_ai" in df.columns and "logit_human" in df.columns
    df["margin"] = margin_of(df)
    y_all = df["label"].values
    gens = sorted(m for m in df["model"].unique() if m != "human")

    base = binary_metrics(y_all, df["prob_ai"].values, threshold=0.5)
    oracle_t = best_threshold(df["margin"].values, y_all)
    oracle_f1 = macro_f1(y_all, (df["margin"].values >= oracle_t).astype(int))

    accs, f1s, eces, temps, thrs = [], [], [], [], []
    rec = {g: [] for g in gens}
    spec = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        cidx = []
        for _, grp in df.groupby(["model", "label"]):
            idx = grp.index.to_numpy(); rng.shuffle(idx)
            cidx.extend(idx[: len(idx) // 2])
        calib = df[df.index.isin(cidx)]
        ev = df[~df.index.isin(cidx)]

        if has_logits:
            cl = np.stack([calib["logit_human"].values, calib["logit_ai"].values], 1)
            el = np.stack([ev["logit_human"].values, ev["logit_ai"].values], 1)
            ts = TemperatureScaler().fit(cl, calib["label"].values)
            ev_p = ts.predict_positive_proba(el)
            temps.append(float(ts.temperature))
        else:
            ev_p = ev["prob_ai"].values
            temps.append(1.0)

        t = best_threshold(calib["margin"].values, calib["label"].values)
        thrs.append(float(t))
        pred = (ev["margin"].values >= t).astype(int)
        yv = ev["label"].values
        accs.append(float((pred == yv).mean()))
        f1s.append(macro_f1(yv, pred))
        eces.append(binary_metrics(yv, ev_p, threshold=0.5)["ece"])
        evp = ev.assign(pred=pred)
        for g in gens:
            sub = evp[evp["model"] == g]
            rec[g].append(float((sub["pred"] == 1).mean()))
        hum = evp[evp["model"] == "human"]
        spec.append(float((hum["pred"] == 0).mean()))

    return {
        "name": name,
        "baseline_at_0.5": {k: round(float(base[k]), 4) for k in
                            ["accuracy", "macro_f1", "auroc", "ece"]},
        "oracle_alltest": {"margin_threshold": round(float(oracle_t), 3),
                           "macro_f1": round(float(oracle_f1), 4)},
        "calibrated_indist_ceiling": {
            "accuracy_mean": round(float(np.mean(accs)), 4),
            "accuracy_std": round(float(np.std(accs)), 4),
            "macro_f1_mean": round(float(np.mean(f1s)), 4),
            "macro_f1_std": round(float(np.std(f1s)), 4),
            "ece_mean": round(float(np.mean(eces)), 4),
            "temperature_mean": round(float(np.mean(temps)), 3),
            "margin_threshold_mean": round(float(np.mean(thrs)), 3),
        },
        "human_specificity_mean": round(float(np.mean(spec)), 4),
        "per_generator_ai_recall": {g: round(float(np.mean(rec[g])), 4) for g in gens},
        "n_seeds": len(SEEDS),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {name: analyze(name, csv) for name, csv in RUNS.items()}
    (OUT / "ceiling.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote {OUT / 'ceiling.json'}\n")
    for name, r in results.items():
        c = r["calibrated_indist_ceiling"]
        print(f"=== {name} ===")
        print(f"  baseline@0.5 macroF1={r['baseline_at_0.5']['macro_f1']:.4f} "
              f"AUROC={r['baseline_at_0.5']['auroc']:.4f} ECE={r['baseline_at_0.5']['ece']:.4f}")
        print(f"  calibrated ceiling macroF1={c['macro_f1_mean']:.4f}±{c['macro_f1_std']:.4f} "
              f"(oracle {r['oracle_alltest']['macro_f1']:.4f})  ECE={c['ece_mean']:.4f}")
        print(f"  human specificity={r['human_specificity_mean']:.4f}  "
              f"per-gen recall={r['per_generator_ai_recall']}")


if __name__ == "__main__":
    main()
