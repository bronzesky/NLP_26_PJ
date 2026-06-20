"""
scripts/humanize.py

De-AI rewriting via local Qwen3-8B, with closed-loop verification.

Positioning (research framing): this is a DETECTOR-ROBUSTNESS / adversarial
evaluation tool. It rewrites high-AI text using suggestions from the detector's
own feature diagnosis, then re-scores with the detector to measure how much AI
suspicion drops under a paraphrase attack. It quantifies detector robustness,
not a service for evading academic-integrity checks.

Pipeline:
  text -> RegionAwareDetector.analyze() -> composite de-AI prompt (polish_advisor)
       -> Qwen3-8B rewrite -> re-analyze -> before/after AI-suspicion delta.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

QWEN_DIR = "/inspire/hdd/project/fdu-aidake-cfff/public/Group35/qwen3-8b"

import re as _re

_PREAMBLE_RE = _re.compile(
    r"^\s*(here(?:'s| is| are)[^\n:]*:?|sure[,!.][^\n]*|certainly[,!.][^\n]*|"
    r"below is[^\n:]*:?|the rewritten[^\n:]*:?|rewritten (?:text|version)[^\n:]*:?|"
    r"i(?:'ve| have) rewritten[^\n:]*:?)\s*\n+", _re.IGNORECASE)

_CODEFENCE_RE = _re.compile(r"^\s*```[a-zA-Z]*\s*\n|\n```\s*$")


def _strip_preamble(text: str) -> str:
    """Remove LLM chat preambles ('Here's the rewritten text:'), code fences,
    and trailing notes. These are assistant-framing tokens that (a) are not
    part of the document and (b) are themselves strong AI signals."""
    t = text.strip()
    t = _CODEFENCE_RE.sub("", t).strip()
    # strip a leading preamble line if present
    m = _PREAMBLE_RE.match(t)
    if m:
        t = t[m.end():].strip()
    # drop a trailing meta note like "I varied sentence length..." after a blank line
    parts = t.rsplit("\n\n", 1)
    if len(parts) == 2 and _re.match(
            r"^\s*(i\b|note:|the (?:above|rewrite)|this version|changes made)",
            parts[1].strip(), _re.IGNORECASE) and len(parts[1]) < 400:
        t = parts[0].strip()
    return t


class Humanizer:
    def __init__(self, model_dir: str = QWEN_DIR):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map="auto",
            max_memory={0: "7GiB", 1: "7GiB", "cpu": "30GiB"})
        self.model.eval()

    @torch.no_grad()
    def rewrite(self, text: str, composite_prompt: str, max_new_tokens: int = 1024) -> str:
        # composite_prompt ends with "Text to rewrite:\n{TEXT}" -> fill it
        if "{TEXT}" in composite_prompt:
            instruction = composite_prompt.replace("{TEXT}", text)
        else:
            instruction = composite_prompt + "\n\nText to rewrite:\n" + text
        msgs = [{"role": "user", "content": instruction}]
        chat = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tok(chat, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **ids, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=0.7, top_p=0.9, repetition_penalty=1.05,
            pad_token_id=self.tok.eos_token_id)
        gen = self.tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        return _strip_preamble(gen)


class APIHumanizer:
    """OpenAI-compatible API rewriter. Enables cross-model attacks (rewrite with
    GPT/Claude/Qwen-max, detect with the local RoBERTa detector). On CFFF the
    base_url points at a reverse-SSH-tunnel (http://localhost:13001)."""

    def __init__(self, model: str = "claude-sonnet-4-6",
                 base_url: str = None, api_key: str = None):
        self.model = model
        self.base_url = (base_url or os.environ.get("LLM_API_BASE", "http://localhost:13001")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")

    def rewrite(self, text: str, composite_prompt: str, max_new_tokens: int = 1500) -> str:
        import urllib.request
        import json as _json
        if "{TEXT}" in composite_prompt:
            instruction = composite_prompt.replace("{TEXT}", text)
        else:
            instruction = composite_prompt + "\n\nText to rewrite:\n" + text
        instruction += ("\n\nIMPORTANT: Output ONLY the rewritten text itself. "
                        "Do not add any preamble, explanation, quotation marks, "
                        "code fences, or notes about what you changed.")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": instruction}],
            "temperature": 0.8, "max_tokens": max_new_tokens,
        }
        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=_json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + self.api_key,
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            d = _json.load(resp)
        return _strip_preamble(d["choices"][0]["message"]["content"])


def _ai_feature_guidance(evidence: list, top_k: int = 5) -> str:
    """Turn the detector's AI-leaning linguistic features into concrete rewrite
    guidance (the 'why it looks AI' that steers the polish)."""
    tips = {
        "discourse_total_density": "remove formulaic connectors (however, furthermore, moreover, in conclusion)",
        "discourse_additive": "drop additive connectors (furthermore, moreover, additionally)",
        "discourse_causal": "drop causal connectors (therefore, thus, consequently)",
        "discourse_conclusive": "delete summary phrases (in conclusion, to summarize, overall)",
        "structural_completeness": "break the rigid topic-sentence/evidence/transition template; vary paragraph shapes",
        "mattr": "vary vocabulary; stop reusing the same adjectives and phrasings",
        "latinate_ratio": "replace formal Latinate words (utilization->use, implementation->how it works)",
        "passive_ratio": "convert passive voice to active",
        "sentence_length_cv": "vary sentence length sharply; mix very short and long sentences",
        "sentence_length_std": "vary sentence length sharply",
        "contraction_ratio": "add contractions (don't, it's, we're, I've)",
        "first_person_ratio": "add a genuine first-person perspective and a concrete personal detail",
        "hedge_density": "move hedging to where you are actually uncertain, not just start/end",
        "modal_ratio": "reduce stacked modals (may, might, could, should)",
        "avg_word_length": "prefer shorter, plainer words",
    }
    picked = []
    for e in evidence:
        if e.get("signal") == "AI-like" and e["feature"] in tips:
            picked.append(tips[e["feature"]])
        if len(picked) >= top_k:
            break
    if not picked:
        return "make phrasing less uniform and less formulaic"
    return "; ".join(picked)


def targeted_recursive_polish(text, detector=None, humanizer=None, title="sample",
                              target=0.5, max_rounds=4, top_frac=0.4,
                              max_new_tokens=400):
    """Lower the (fixed) RoBERTa AI rate by rewriting ONLY the sentences that
    contribute most to it (occlusion-located), guided by the AI-leaning
    features (why they look AI). Recurse on the new top contributors until the
    same RoBERTa AI rate drops below `target` or rounds run out.

    The detector is FIXED; AI rate before/after is the same RoBERTa prob."""
    from src.detector_pipeline import RegionAwareDetector, split_sentences
    det = detector or RegionAwareDetector()
    hum = humanizer or Humanizer()

    res0 = det.analyze(text, title=title, with_suggestions=True)
    guidance = _ai_feature_guidance(res0.get("feature_evidence", []))
    cur = text
    all_edits = []
    trajectory = [{"round": 0, "prob_ai": res0["doc_prob_ai"],
                   "verdict": res0["doc_verdict"], "rewritten_sents": 0}]

    for rnd in range(1, max_rounds + 1):
        res = det.analyze(cur, title=title, with_suggestions=False)
        if res["doc_prob_ai"] < target:
            break
        # collect (sentence, contrib) across paragraphs, rank by AI contribution
        sents = []
        for p in res["paragraphs"]:
            for s in p.get("sentences", []):
                sents.append((s["text"], s.get("contrib", 0.0)))
        if not sents:
            sents = [(s, 0.0) for s in split_sentences(cur)]
        # pick the top fraction of sentences pushing toward AI (contrib > 0)
        ranked = sorted(sents, key=lambda x: -x[1])
        n_target = max(1, int(len(ranked) * top_frac))
        targets = [s for s, c in ranked[:n_target] if c > 0] or [ranked[0][0]]

        # rewrite ONLY those sentences, in context, with feature guidance
        new_cur = cur
        round_edits = []
        for sent in targets:
            prompt = (
                "Rewrite ONLY this sentence to read as natural human writing. "
                f"Specifically: {guidance}. Keep the meaning and keep it one sentence. "
                "Output only the rewritten sentence.\n\nSentence:\n" + sent)
            try:
                rw = hum.rewrite(sent, prompt, max_new_tokens=max_new_tokens)
                rw = rw.split("\n")[0].strip()
                if rw and rw != sent:
                    new_cur = new_cur.replace(sent, rw, 1)
                    round_edits.append({"round": rnd, "original": sent, "rewritten": rw})
            except Exception as e:
                print(f"  [round {rnd}] rewrite failed: {type(e).__name__}")
        cur = new_cur
        all_edits.extend(round_edits)
        res_after = det.analyze(cur, title=title, with_suggestions=False)
        trajectory.append({"round": rnd, "prob_ai": res_after["doc_prob_ai"],
                           "verdict": res_after["doc_verdict"],
                           "rewritten_sents": len(round_edits)})
        if res_after["doc_prob_ai"] < target:
            break

    final = det.analyze(cur, title=title + " (降AI后)", with_suggestions=True)
    return {
        "title": title,
        "ai_rate_before": res0["doc_prob_ai"],
        "ai_rate_after": final["doc_prob_ai"],
        "ai_rate_drop": round(res0["doc_prob_ai"] - final["doc_prob_ai"], 4),
        "verdict_before": res0["doc_verdict"],
        "verdict_after": final["doc_verdict"],
        "rounds_used": len(trajectory) - 1,
        "trajectory": trajectory,
        "edits": all_edits,
        "guidance": guidance,
        "original_text": text,
        "rewritten_text": cur,
        "before": res0,
        "after": final,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, default=None)
    ap.add_argument("--text_file", type=Path, default=None)
    ap.add_argument("--title", type=str, default="sample")
    ap.add_argument("--out", type=Path, default=REPO / "outputs/humanize_demo/result.json")
    args = ap.parse_args()

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    r = targeted_recursive_polish(text, title=args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # keep json light: drop the heavy nested before/after analyze dicts
    slim = {k: v for k, v in r.items() if k not in ("before", "after")}
    args.out.write_text(json.dumps(slim, ensure_ascii=False, indent=2))
    print(f"AI rate (RoBERTa): {r['ai_rate_before']:.3f} -> {r['ai_rate_after']:.3f} "
          f"(drop {r['ai_rate_drop']:.3f}) in {r['rounds_used']} round(s)")
    print(f"verdict: {r['verdict_before']} -> {r['verdict_after']}")
    print(f"trajectory: {[(t['round'], round(t['prob_ai'],3), t['rewritten_sents']) for t in r['trajectory']]}")
    print(f"guidance: {r['guidance']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
