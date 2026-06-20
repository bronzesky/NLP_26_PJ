"""
Phase B diagnostic: why does the frozen detector generalize to unseen GPT4
(recall 0.89) but fail on unseen bloomz (recall 0.10)?

Compare feature distributions across groups: human, bloomz, GPT4, and the
seen-family AI (chatGPT/cohere/davinci/dolly). For each handcrafted feature
report group medians, and rank features by how much bloomz sits closer to
human than the seen-AI cluster does. Also report the model's own prob_ai
distribution per group (does bloomz look human in RoBERTa space too?).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
df = pd.read_csv(REPO / "outputs/analysis/features.csv")

FEATS = ["text_length", "word_count", "avg_sentence_length", "ttr",
         "punctuation_per_token", "repeated_bigram_ratio",
         "repeated_trigram_ratio", "avg_word_length"]

groups = {
    "human": df[df.model == "human"],
    "bloomz": df[df.model == "bloomz"],
    "GPT4": df[df.model == "GPT4"],
    "seen_AI": df[df.model.isin(["chatGPT", "cohere", "davinci", "dolly"])],
}

print("=== prob_ai (RoBERTa) per group: median / mean ===")
for g, d in groups.items():
    print(f"  {g:10s} median={d.prob_ai.median():.4f}  mean={d.prob_ai.mean():.4f}")

print("\n=== handcrafted feature medians per group ===")
hdr = f"{'feature':24s}" + "".join(f"{g:>12s}" for g in groups)
print(hdr)
for f in FEATS:
    row = f"{f:24s}" + "".join(f"{groups[g][f].median():12.3f}" for g in groups)
    print(row)

# "human-likeness" score: for each feature, is bloomz closer to human than seen_AI is?
print("\n=== which features make bloomz look human (|bloomz-human| vs |seenAI-human|) ===")
h = groups["human"]
ranked = []
for f in FEATS:
    hm = h[f].median()
    b_dist = abs(groups["bloomz"][f].median() - hm)
    s_dist = abs(groups["seen_AI"][f].median() - hm)
    g4_dist = abs(groups["GPT4"][f].median() - hm)
    # normalize by human IQR to compare across features
    iqr = h[f].quantile(0.75) - h[f].quantile(0.25) + 1e-9
    ranked.append((f, b_dist/iqr, g4_dist/iqr, s_dist/iqr))
ranked.sort(key=lambda x: x[1])  # bloomz closest to human first
print(f"{'feature':24s}{'bloomz':>10s}{'GPT4':>10s}{'seen_AI':>10s}   (dist to human, /human-IQR)")
for f, b, g4, s in ranked:
    print(f"{f:24s}{b:10.2f}{g4:10.2f}{s:10.2f}")

# length angle: bloomz length vs human
print("\n=== text_length distribution (chars) percentiles ===")
for g, d in groups.items():
    q = d.text_length.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values
    print(f"  {g:10s} p10={q[0]:.0f} p25={q[1]:.0f} p50={q[2]:.0f} p75={q[3]:.0f} p90={q[4]:.0f}")
