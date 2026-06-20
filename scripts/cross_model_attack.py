"""
Cross-model paraphrase-attack robustness eval (via external API through tunnel).

For each high-AI sample, rewrite with several attacker models (Claude/GPT/Qwen),
up to K recursive rounds, re-detecting each round. Measures how much the
region-aware detector's AI suspicion drops under paraphrase attack by stronger
models than the local Qwen3-8B. This is the detector-robustness deliverable.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.detector_pipeline import RegionAwareDetector
from scripts.humanize import APIHumanizer

ATTACKERS = ["claude-sonnet-4-6", "gpt-5.2", "qwen/qwen3-max"]
K = 2
N = 12  # high-AI samples per run (scaled for paper-grade numbers)
GENERIC = ("Rewrite the following text to sound naturally human-written: "
           "vary sentence length sharply, use contractions, add a personal voice "
           "and a concrete aside, avoid formulaic transitions and formal Latinate "
           "words. Preserve the meaning. Output ONLY the rewritten text.\n\nText:\n{TEXT}")

det = RegionAwareDetector()
det._lm = None
df = pd.read_csv(REPO / "outputs/analysis/features.csv")
# pick clear-AI samples (chatGPT, high prob) of moderate length
pool = df[(df.model == "chatGPT") & (df.text.str.len().between(800, 2200))].head(N)

rows = []
for attacker in ATTACKERS:
    hum = APIHumanizer(model=attacker)
    for _, r in pool.iterrows():
        text = str(r["text"])
        a = det.analyze(text, with_suggestions=True)
        traj = [(round(a["doc_margin"], 2), round(a["doc_prob_ai"], 3), a["doc_label"])]
        cur = text
        ok = True
        for k in range(K):
            try:
                prompt = a.get("suggestions", {}).get("composite_prompt") or GENERIC
                cur = hum.rewrite(cur, prompt)
                a = det.analyze(cur, with_suggestions=True)
                traj.append((round(a["doc_margin"], 2), round(a["doc_prob_ai"], 3), a["doc_label"]))
            except Exception as e:
                print(f"  [{attacker}] round {k} failed: {type(e).__name__} {str(e)[:80]}")
                ok = False
                break
        rows.append({"attacker": attacker, "trajectory": traj,
                     "final_label": a["doc_label"], "ok": ok,
                     "final_text": cur[:400]})
        print(f"{attacker:22s} {traj} label={a['doc_label']}")

out = REPO / "outputs/robustness/cross_model_attack.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

print("\n=== summary: mean prob_ai / flip-to-human rate by attacker & round ===")
for attacker in ATTACKERS:
    sub = [r for r in rows if r["attacker"] == attacker and r["ok"]]
    if not sub:
        print(f"{attacker}: no successful runs"); continue
    line = f"{attacker:22s}"
    for k in range(K + 1):
        probs = [r["trajectory"][k][1] for r in sub if len(r["trajectory"]) > k]
        flip = np.mean([r["trajectory"][k][2] == 0 for r in sub if len(r["trajectory"]) > k])
        line += f" | r{k}: p(AI)={np.mean(probs):.3f} flip={flip*100:.0f}%"
    print(line)
print(f"\nWrote {out}")
