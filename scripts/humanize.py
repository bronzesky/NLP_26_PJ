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
        return gen.strip()


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
        return d["choices"][0]["message"]["content"].strip()


def closed_loop(text: str, title: str = "sample", detector=None, humanizer=None,
                with_report: bool = False) -> dict:
    """Detect -> de-AI rewrite -> re-detect. Returns before/after deltas."""
    from src.detector_pipeline import RegionAwareDetector
    det = detector or RegionAwareDetector()
    hum = humanizer or Humanizer()

    before = det.analyze(text, title=title)
    composite = before.get("suggestions", {}).get("composite_prompt", "")
    rewritten = hum.rewrite(text, composite)
    after = det.analyze(rewritten, title=title + " (降AI后)")

    result = {
        "title": title,
        "before": {
            "suspicion": before["doc_suspicion"], "prob_ai": before["doc_prob_ai"],
            "label": before["doc_label"], "grade": before["doc_grade"],
            "region": before["doc_region"], "ppl": before["ppl"],
            "burstiness": before["burstiness"],
        },
        "after": {
            "suspicion": after["doc_suspicion"], "prob_ai": after["doc_prob_ai"],
            "label": after["doc_label"], "grade": after["doc_grade"],
            "region": after["doc_region"], "ppl": after["ppl"],
            "burstiness": after["burstiness"],
        },
        "suspicion_drop": round(before["doc_suspicion"] - after["doc_suspicion"], 4),
        "prob_ai_drop": round(before["doc_prob_ai"] - after["doc_prob_ai"], 4),
        "original_text": text,
        "rewritten_text": rewritten,
    }
    if with_report:
        result["_before_full"] = before
        result["_after_full"] = after
    return result


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

    r = closed_loop(text, title=args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, ensure_ascii=False, indent=2))
    b, a = r["before"], r["after"]
    print(f"BEFORE: suspicion={b['suspicion']:.3f} prob_ai={b['prob_ai']:.3f} "
          f"grade={b['grade']} ppl={b['ppl']} burst={b['burstiness']}")
    print(f"AFTER : suspicion={a['suspicion']:.3f} prob_ai={a['prob_ai']:.3f} "
          f"grade={a['grade']} ppl={a['ppl']} burst={a['burstiness']}")
    print(f"DROP  : suspicion -{r['suspicion_drop']:.3f}  prob_ai -{r['prob_ai_drop']:.3f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
