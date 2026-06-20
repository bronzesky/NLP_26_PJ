"""
E-series de-AI ablation (fixed-RoBERTa AI rate, multi-sample, with controls).
Variants on the same N high-AI chatGPT docs:
  E-full : targeted (occlusion) + feature-guided + recursive  (our method)
  E1     : whole-doc rewrite + guided + recursive             (targeted ablation)
  E2     : targeted + guided + SINGLE round                   (recursion ablation)
  E4     : RANDOM sentence selection + guided + recursive      (occlusion ablation)
Metrics per variant: AI-rate before/after (same fixed RoBERTa), %edited tokens
(fidelity proxy), AI-rate drop per unit edit (efficiency), verdict flip rate.
"""
import sys, json, random
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/inspire/hdd/project/fdu-aidake-cfff/public/hanz/semeval2024_task8a_en_baseline")
sys.path.insert(0, str(REPO))
from src.detector_pipeline import RegionAwareDetector, split_sentences
from scripts.humanize import APIHumanizer, _ai_feature_guidance

N = 8
MAXR = 6
random.seed(0)
det = RegionAwareDetector(); det._lm = None
hum = APIHumanizer(model="claude-sonnet-4-6")
df = pd.read_csv(REPO / "outputs/analysis/features.csv")
pool = df[(df.model == "chatGPT") & (df.text.str.len().between(700, 2200))].head(N)

def edit_frac(a, b):
    sa, sb = a.split(), b.split()
    # crude token-level change fraction
    from difflib import SequenceMatcher
    sm = SequenceMatcher(None, sa, sb)
    same = sum(blk.size for blk in sm.get_matching_blocks())
    return 1 - same / max(len(sa), 1)

def ai_rate(txt):
    return det.analyze(txt, with_suggestions=False)["doc_prob_ai"]

def rewrite_sentences(text, guidance, targets):
    cur = text
    for s in targets:
        prompt = ("Rewrite ONLY this sentence to read as natural human writing. "
                  f"Specifically: {guidance}. Keep meaning, one sentence. Output only the sentence.\n\nSentence:\n" + s)
        try:
            rw = hum.rewrite(s, prompt, max_new_tokens=400).split("\n")[0].strip()
            if rw and rw != s:
                cur = cur.replace(s, rw, 1)
        except Exception as e:
            print(f"   rewrite fail {type(e).__name__}")
    return cur

def run_variant(name, text, mode, recursive):
    res0 = det.analyze(text, with_suggestions=True)
    guidance = _ai_feature_guidance(res0.get("feature_evidence", []))
    before = res0["doc_prob_ai"]; cur = text
    rounds = MAXR if recursive else 1
    for r in range(rounds):
        res = det.analyze(cur, with_suggestions=False)
        if res["doc_prob_ai"] < 0.5: break
        if mode == "whole":
            prompt = ("Rewrite the following text to read as natural human writing. "
                      f"Specifically: {guidance}. Keep meaning. Output only the rewritten text.\n\nText:\n" + cur)
            try: cur = hum.rewrite(cur, prompt, max_new_tokens=1500)
            except Exception as e: print(f"   whole fail {type(e).__name__}")
        else:
            sents = []
            for p in res["paragraphs"]:
                for s in p.get("sentences", []):
                    sents.append((s["text"], s.get("contrib", 0.0)))
            if not sents: break
            k = max(1, int(len(sents) * 0.4))
            if mode == "targeted":
                targets = [s for s, c in sorted(sents, key=lambda x: -x[1])[:k] if c > 0] or [sents[0][0]]
            else:  # random
                targets = [s for s, _ in random.sample(sents, min(k, len(sents)))]
            cur = rewrite_sentences(cur, guidance, targets)
    after = ai_rate(cur)
    ef = edit_frac(text, cur)
    return {"before": before, "after": after, "drop": before - after,
            "edit_frac": round(ef, 3), "flip": int(after < 0.5),
            "drop_per_edit": round((before - after) / max(ef, 1e-3), 3)}

variants = [("E-full targeted+rec", "targeted", True),
            ("E1 whole+rec", "whole", True),
            ("E2 targeted+single", "targeted", False),
            ("E4 random+rec", "random", True)]
agg = {v[0]: [] for v in variants}
for _, row in pool.iterrows():
    text = str(row["text"])
    for name, mode, rec in variants:
        r = run_variant(name, text, mode, rec)
        agg[name].append(r)
        print(f"{name:22s} before={r['before']:.3f} after={r['after']:.3f} drop={r['drop']:.3f} edit={r['edit_frac']:.2f} flip={r['flip']}")

print("\n=== SUMMARY (mean over %d docs) ===" % N)
summ = {}
for name in agg:
    rs = agg[name]
    m = {"mean_drop": round(np.mean([x["drop"] for x in rs]), 3),
         "mean_after": round(np.mean([x["after"] for x in rs]), 3),
         "flip_rate": round(np.mean([x["flip"] for x in rs]), 3),
         "mean_edit_frac": round(np.mean([x["edit_frac"] for x in rs]), 3),
         "drop_per_edit": round(np.mean([x["drop_per_edit"] for x in rs]), 3)}
    summ[name] = m
    print(f"{name:22s} drop={m['mean_drop']:.3f} after={m['mean_after']:.3f} flip={m['flip_rate']:.2f} edit={m['mean_edit_frac']:.2f} drop/edit={m['drop_per_edit']:.2f}")
(REPO/"outputs/ablation").mkdir(parents=True, exist_ok=True)
(REPO/"outputs/ablation/eseries.json").write_text(json.dumps({"summary": summ, "raw": agg, "n": N}, ensure_ascii=False, indent=2))
print("\nWrote outputs/ablation/eseries.json")
