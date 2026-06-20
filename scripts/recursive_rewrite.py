"""
Robustness eval: recursive de-AI rewriting vs the region-aware detector.
For each sample, rewrite up to K rounds (feeding output back in), tracking
margin / prob_ai / suspicion after each round. Answers whether the detector is
genuinely robust to paraphrase or just margin-saturated at round 1.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.detector_pipeline import RegionAwareDetector
from scripts.humanize import Humanizer

K = 3
N_PER_MODEL = 4
MODELS = ["chatGPT", "GPT4", "davinci"]

det = RegionAwareDetector()
det._lm = None  # skip ppl for speed
hum = Humanizer()
df = pd.read_csv(REPO / "outputs/analysis/features.csv")

GENERIC = ("Rewrite the following text to sound naturally human-written: "
           "vary sentence length, use contractions, add a personal voice, "
           "avoid formulaic transitions and formal Latinate words. "
           "Keep the meaning. Output only the rewritten text.\n\nText:\n{TEXT}")

rows = []
for model in MODELS:
    sub = df[(df.model == model) & (df.text.str.len().between(700, 2500))].head(N_PER_MODEL)
    for _, r in sub.iterrows():
        text = str(r["text"])
        a = det.analyze(text, with_suggestions=True)
        traj = [(round(a["doc_margin"], 2), round(a["doc_prob_ai"], 3), round(a["doc_suspicion"], 3))]
        cur = text
        for k in range(K):
            prompt = a.get("suggestions", {}).get("composite_prompt") or GENERIC
            cur = hum.rewrite(cur, prompt)
            a = det.analyze(cur, with_suggestions=True)
            traj.append((round(a["doc_margin"], 2), round(a["doc_prob_ai"], 3), round(a["doc_suspicion"], 3)))
        rows.append({"model": model, "trajectory": traj,
                     "final_label": a["doc_label"], "true_label": int(r["label"])})
        print(f"{model:8s} margin/prob/susp by round: {traj}  final_label={a['doc_label']}")

out = REPO / "outputs/robustness/recursive_rewrite.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

# summary: mean suspicion + flip rate per round
print("\n=== summary (mean over samples) ===")
for k in range(K + 1):
    susp = np.mean([row["trajectory"][k][2] for row in rows])
    prob = np.mean([row["trajectory"][k][1] for row in rows])
    margin = np.mean([row["trajectory"][k][0] for row in rows])
    flipped = np.mean([row["trajectory"][k][2] < 0.5 for row in rows])
    print(f"round {k}: margin={margin:6.2f}  prob_ai={prob:.3f}  suspicion={susp:.3f}  flipped_to_human={flipped*100:.0f}%")
print(f"\nWrote {out}")
