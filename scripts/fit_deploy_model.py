"""
Fit the DEPLOY region-aware decision model on DEV and persist it.

This is the P1 (legal) protocol frozen into a reusable artifact: all params
learned on dev only, so the pipeline can score arbitrary text without touching
test labels. Saves margin thresholds, ambiguous-band edges, and the ortho-
feature logistic-regression coefficients to outputs/region_aware/deploy_model.json.
"""
import json
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

dev = pd.read_csv(REPO / "outputs/analysis_dev/features.csv").reset_index(drop=True)
dev["margin"] = dev["logit_ai"] - dev["logit_human"]
y = dev["label"].values


def self_pred(df, tl, th, clf):
    m = df.margin.values
    amb = (m >= tl) & (m < th)
    pred = np.where(m >= th, 1, np.where(m < tl, 0, -1))
    pred[amb] = clf.predict(df.loc[amb, ORTHO].values)
    return pred


best = None
for tl in np.quantile(dev.margin.values, [0.3, 0.4, 0.5, 0.6]):
    for th in np.quantile(dev.margin.values, [0.75, 0.8, 0.85, 0.9]):
        if th <= tl:
            continue
        amb = (dev.margin.values >= tl) & (dev.margin.values < th)
        if amb.sum() < 50 or len(np.unique(y[amb])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(dev.loc[amb, ORTHO].values, y[amb])
        f1 = macro_f1(y, self_pred(dev, tl, th, clf))
        if best is None or f1 > best[0]:
            best = (f1, tl, th, clf)

f1, tl, th, clf = best
# also fit a feature scaler stat for reporting (human/ai means already in baselines)
artifact = {
    "protocol": "P1_deploy_fit_on_dev",
    "temperature": 5.367681468392405,
    "ortho_features": ORTHO,
    "margin_t_low": float(tl),
    "margin_t_high": float(th),
    "ortho_lr_coef": clf.coef_[0].tolist(),
    "ortho_lr_intercept": float(clf.intercept_[0]),
    "dev_fit_macro_f1": float(f1),
    "note": "margin>=t_high->AI; margin<t_low->human; ambiguous->ortho LR. "
            "margin = (logit_ai - logit_human). Logits from roberta_base/best_model.",
}
out = REPO / "outputs/region_aware/deploy_model.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(artifact, indent=2))
print(f"Wrote {out}")
print(f"  band=[{tl:.3f}, {th:.3f}]  dev-fit macro-F1={f1:.4f}")
print(f"  ortho coef: {dict(zip(ORTHO, [round(c,4) for c in clf.coef_[0]]))}")
