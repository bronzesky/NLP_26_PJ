"""
Generate a gallery of PaperPass-style reports on real test samples:
one human, one bloomz (the hard human-like case), one GPT4 (unseen generator).
Demonstrates the region-aware method end-to-end with real perplexity.
"""
import sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.detector_pipeline import RegionAwareDetector
from scripts.render_report import render

OUT = REPO / "outputs/report_gallery"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(REPO / "outputs/analysis/features.csv")
det = RegionAwareDetector()

picks = {
    "human": "人类写作样本",
    "bloomz": "bloomz 样本（拟人难例）",
    "GPT4": "GPT-4 样本（未见生成器）",
}
summary = []
for model, title in picks.items():
    # pick a reasonably long sample for a richer report
    sub = df[(df.model == model) & (df.text.str.len() > 800)]
    row = sub.iloc[0] if len(sub) else df[df.model == model].iloc[0]
    r = det.analyze(str(row["text"]), title=title)
    out = OUT / f"report_{model}.html"
    render(r, out, title)
    correct = "✓" if r["doc_label"] == int(row["label"]) else "✗"
    summary.append((model, int(row["label"]), r["doc_label"], r["doc_prob_ai"],
                    r["doc_region"], r["ppl"], r["burstiness"], correct))
    print(f"{model:8s} -> {out.name}")

print("\nmodel    true pred P(AI) region        ppl    burst  ok")
for m, t, p, s, reg, ppl, b, c in summary:
    print(f"{m:8s}  {t}    {p}   {s:.2f}  {reg:12s} {ppl:6.1f} {b:.2f}   {c}")
