"""
Run one full de-AI closed loop on a fresh sample and save BOTH before/after
full analyze() results (with paragraph/sentence breakdown) for rendering a
PaperPass reduce_aigc-style before/after report.
"""
import json
import sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.detector_pipeline import RegionAwareDetector
from scripts.humanize import APIHumanizer

OUT = REPO / "outputs/reduce_demo"
OUT.mkdir(parents=True, exist_ok=True)

det = RegionAwareDetector()
df = pd.read_csv(REPO / "outputs/analysis/features.csv")
# fresh clear-AI sample, different from earlier ones
row = df[(df.model == "chatGPT") & (df.text.str.len().between(700, 3500))].iloc[0]
text = str(row["text"])

attacker = "claude-sonnet-4-6"  # strongest attacker from the eval
hum = APIHumanizer(model=attacker)

before = det.analyze(text, title="原文", with_suggestions=True)
composite = before.get("suggestions", {}).get("composite_prompt", "")
rewritten = hum.rewrite(text, composite)
after = det.analyze(rewritten, title="降AIGC后", with_suggestions=True)

bundle = {
    "attacker": attacker,
    "before": before,
    "after": after,
    "original_text": text,
    "rewritten_text": rewritten,
    "label_before": before["doc_label"],
    "label_after": after["doc_label"],
    "prob_ai_drop": round(before["doc_prob_ai"] - after["doc_prob_ai"], 4),
}
(OUT / "bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2))
print(f"attacker={attacker}")
print(f"BEFORE prob_ai={before['doc_prob_ai']:.3f} verdict={before['doc_label']} "
      f"grade={before['doc_grade']} margin={before['doc_margin']:.2f}")
print(f"AFTER  prob_ai={after['doc_prob_ai']:.3f} verdict={after['doc_label']} "
      f"grade={after['doc_grade']} margin={after['doc_margin']:.2f}")
print(f"Wrote {OUT/'bundle.json'}")
