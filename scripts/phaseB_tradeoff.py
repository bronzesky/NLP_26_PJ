"""
Phase B hypothesis test: is bloomz a victim of the high threshold forced by
human over-confidence, rather than an independent hard case?

Sweep the margin threshold and trace, jointly:
  - human specificity (TNR)
  - bloomz recall (TPR)
  - GPT4 recall, seen-AI recall
  - overall macro-F1
If bloomz recall and human specificity trade off steeply (bloomz only
recovers when we accept many human false positives), bloomz is collateral
damage of the single global threshold, not a distinct failure mode.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.metrics import macro_f1

df = pd.read_csv(REPO / "outputs/analysis/features.csv")
df["margin"] = df["logit_ai"] - df["logit_human"]
y = df["label"].values

def grp(name):
    return df[df.model == name] if name != "seen_AI" else df[df.model.isin(["chatGPT","cohere","davinci","dolly"])]

human, bloomz, gpt4, seen = grp("human"), grp("bloomz"), grp("GPT4"), grp("seen_AI")

print(f"{'margin_thr':>10s}{'macroF1':>9s}{'human_spec':>11s}{'bloomz_rec':>11s}{'GPT4_rec':>9s}{'seen_rec':>9s}")
for t in [-5, 0, 3, 6, 9, 10, 11, 12, 13, 14, 16, 20]:
    pred_all = (df["margin"].values >= t).astype(int)
    f1 = macro_f1(y, pred_all)
    hs = (human["margin"] < t).mean()
    br = (bloomz["margin"] >= t).mean()
    g4 = (gpt4["margin"] >= t).mean()
    sr = (seen["margin"] >= t).mean()
    print(f"{t:10.1f}{f1:9.4f}{hs:11.4f}{br:11.4f}{g4:9.4f}{sr:9.4f}")

# where does bloomz's margin actually sit vs human?
print("\n=== margin percentiles per group ===")
for name in ["human","bloomz","GPT4","seen_AI"]:
    g = grp(name)
    q = g["margin"].quantile([0.1,0.25,0.5,0.75,0.9]).values
    print(f"  {name:8s} p10={q[0]:7.2f} p25={q[1]:7.2f} p50={q[2]:7.2f} p75={q[3]:7.2f} p90={q[4]:7.2f}")
