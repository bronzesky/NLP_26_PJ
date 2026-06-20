"""
src/detector_pipeline.py

Unified inference pipeline for the region-aware AI-text detector.
Takes raw text -> full hierarchical detection result (document / paragraph /
sentence scores + linguistic-feature evidence + de-AI suggestions).

Deployment protocol (P1): all decision params were fit on DEV only and frozen
into outputs/region_aware/deploy_model.json. No test labels touched here.

Decision rule (region-aware two-stage):
    margin = logit_ai - logit_human   (from fine-tuned roberta_base/best_model)
    margin >= t_high            -> AI
    margin <  t_low             -> human
    t_low <= margin < t_high    -> ambiguous: decided by logistic regression on
                                   orthogonal handcrafted features (TTR etc.)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]

_SENT_RE = re.compile(r"[^.!?\n]+[.!?\n]*")


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()] or [text.strip()]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.findall(text) if s.strip()]


@dataclass
class DetectorConfig:
    model_dir: str = str(REPO / "outputs/roberta_base/best_model")
    deploy_model: str = str(REPO / "outputs/region_aware/deploy_model.json")
    baselines: str = str(REPO / "data/feature_baselines.json")
    max_length: int = 512
    device: Optional[str] = None


def _grade(prob_ai: float) -> str:
    """PaperPass-style grade. prob_ai in [0,1]."""
    p = prob_ai * 100
    if p >= 70:
        return "high"      # 高度疑似 (red)
    if p >= 60:
        return "middle"    # 中度疑似 (orange)
    if p >= 50:
        return "low"       # 轻度疑似 (purple)
    return "ok"            # 合格 (green)


class RegionAwareDetector:
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.cfg = config or DetectorConfig()
        self.device = torch.device(
            self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_dir).to(self.device).eval()

        dm = json.loads(Path(self.cfg.deploy_model).read_text())
        self.temperature = float(dm["temperature"])
        self.t_low = float(dm["margin_t_low"])
        self.t_high = float(dm["margin_t_high"])
        self.ortho_features = list(dm["ortho_features"])
        self.ortho_coef = np.array(dm["ortho_lr_coef"], dtype=float)
        self.ortho_intercept = float(dm["ortho_lr_intercept"])

        self.baselines = json.loads(Path(self.cfg.baselines).read_text())

    # ---- low-level scoring -------------------------------------------------
    @torch.no_grad()
    def _logits(self, texts: list[str]) -> np.ndarray:
        """Return (n, 2) logits [human, ai] for a batch of texts."""
        out = []
        for i in range(0, len(texts), 16):
            batch = texts[i:i + 16]
            enc = self.tokenizer(batch, truncation=True, max_length=self.cfg.max_length,
                                 padding=True, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits  # (b, 2)
            out.append(logits.cpu().numpy())
        return np.concatenate(out, axis=0)

    def _calibrated_prob_ai(self, logits: np.ndarray) -> np.ndarray:
        """Temperature-scaled P(AI)."""
        scaled = logits / self.temperature
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        e = np.exp(scaled)
        return e[:, 1] / e.sum(axis=1)

    @staticmethod
    def _margin(logits: np.ndarray) -> np.ndarray:
        return logits[:, 1] - logits[:, 0]

    def _ortho_prob_ai(self, feats: dict[str, float]) -> float:
        x = np.array([float(feats.get(f, 0.0)) for f in self.ortho_features])
        z = float(np.dot(self.ortho_coef, x) + self.ortho_intercept)
        return 1.0 / (1.0 + np.exp(-z))

    def _region_decide(self, margin: float, cal_prob: float,
                       feats: dict[str, float]) -> tuple[int, str, float]:
        """Return (label, region, final_prob). final_prob is the single
        coherent AI probability that BOTH drives the verdict and is shown as
        the headline, so they can never disagree:
          - clean regions: the calibrated RoBERTa P(AI) (continuous, not a
            hardcoded 0/1 -- that discrete jump was the old bug);
          - ambiguous band: the orthogonal-feature classifier P(AI), which is
            what actually decides the verdict there.
        verdict = (final_prob >= 0.5)."""
        if margin >= self.t_high:
            return int(cal_prob >= 0.5), "clean_ai", float(cal_prob)
        if margin < self.t_low:
            return int(cal_prob >= 0.5), "clean_human", float(cal_prob)
        # ambiguous band -> orthogonal-feature classifier is the deciding signal
        p = self._ortho_prob_ai(feats)
        return int(p >= 0.5), "ambiguous", float(p)

    # ---- feature evidence --------------------------------------------------
    def _feature_evidence(self, feats: dict[str, float]) -> list[dict]:
        """For each handcrafted feature, map the current value onto a
        human<->AI discriminant axis via a Gaussian log-likelihood ratio:
            LLR = logN(val; ai_mean,ai_std) - logN(val; human_mean,human_std)
            disc = tanh(LLR/2)  in [-1,+1]   (-1 = certainly human, +1 = AI)
        This accounts for both class means AND spreads, so the axis position
        means "how much this feature leans AI", not just raw value."""
        import math

        def gll(x, mu, sd):
            sd = max(float(sd), 1e-6)
            return -0.5 * ((x - mu) / sd) ** 2 - math.log(sd)

        rows = []
        for name, b in self.baselines.items():
            if name.startswith("_") or name not in feats or not isinstance(b, dict):
                continue
            h = float(b.get("human_mean", 0.0)); hs = float(b.get("human_std", 0.0))
            a = float(b.get("ai_mean", 0.0));    as_ = float(b.get("ai_std", 0.0))
            val = float(feats[name])
            if abs(a - h) < 1e-9:
                continue
            llr = gll(val, a, as_) - gll(val, h, hs)
            disc = math.tanh(llr / 2.0)                  # [-1,+1], + = AI
            disc_h = math.tanh((gll(h, a, as_) - gll(h, h, hs)) / 2.0)  # where human-mean sits
            disc_a = math.tanh((gll(a, a, as_) - gll(a, h, hs)) / 2.0)  # where AI-mean sits
            if disc >= 0.33:
                signal = "AI-like"
            elif disc <= -0.33:
                signal = "human-like"
            else:
                signal = "neutral"
            rows.append({
                "feature": name, "value": round(val, 4),
                "human_mean": round(h, 4), "ai_mean": round(a, 4),
                "human_std": round(hs, 4), "ai_std": round(as_, 4),
                "disc": round(float(disc), 4),
                "disc_human": round(float(disc_h), 4),
                "disc_ai": round(float(disc_a), 4),
                "signal": signal,
            })
        # most discriminative (|disc| largest) first
        rows.sort(key=lambda r: -abs(r["disc"]))
        return rows

    # ---- public API --------------------------------------------------------
    def analyze(self, text: str, title: str = "Untitled",
                with_suggestions: bool = True, occlusion: bool = True) -> dict:
        from src.features_v2 import full_features
        text = (text or "").strip()
        paragraphs = split_paragraphs(text)

        # document-level. THE AI RATE IS A SINGLE FIXED RoBERTa calibrated prob,
        # used identically before/after polishing. Region/ortho are metadata
        # only -- they never override the AI rate or the verdict.
        doc_logits = self._logits([text])[0]
        doc_margin = float(self._margin(doc_logits[None, :])[0])
        doc_prob = float(self._calibrated_prob_ai(doc_logits[None, :])[0])
        doc_feats = full_features(text)
        doc_label = int(doc_prob >= 0.5)
        # region info kept for the report (which decision zone the doc falls in)
        if doc_margin >= self.t_high:
            doc_region = "clean_ai"
        elif doc_margin < self.t_low:
            doc_region = "clean_human"
        else:
            doc_region = "ambiguous"

        # ---- occlusion sentence scoring (coherent with the document score) ----
        # Each sentence's AI contribution = doc_prob(full) - doc_prob(without it).
        # Positive => removing the sentence lowered AI prob => it pushed toward AI.
        all_sents, sent_owner = [], []
        for pi, para in enumerate(paragraphs):
            for s in split_sentences(para):
                all_sents.append(s)
                sent_owner.append(pi)

        occl = {}
        if occlusion and len(all_sents) >= 2:
            variants = []
            for i in range(len(all_sents)):
                variants.append(" ".join(all_sents[:i] + all_sents[i + 1:]))
            v_logits = self._logits(variants)
            v_probs = self._calibrated_prob_ai(v_logits)
            contribs = doc_prob - v_probs  # signed contribution to AI prob
            # normalize contributions to [0,1] heat by mapping around 0
            mx = max(abs(contribs).max(), 1e-6)
            for i in range(len(all_sents)):
                occl[i] = {"contrib": float(contribs[i]),
                           "heat": float(0.5 + 0.5 * contribs[i] / mx)}

        # paragraph-level
        para_results = []
        sent_global = 0
        for pi, para in enumerate(paragraphs):
            p_logits = self._logits([para])[0]
            p_margin = float(self._margin(p_logits[None, :])[0])
            p_prob = float(self._calibrated_prob_ai(p_logits[None, :])[0])
            p_feats = full_features(para)
            p_label = int(p_prob >= 0.5)
            p_region = ("clean_ai" if p_margin >= self.t_high
                        else "clean_human" if p_margin < self.t_low else "ambiguous")

            sents = split_sentences(para)
            sent_results = []
            for s in sents:
                idx = sent_global
                sent_global += 1
                if idx in occl:
                    contrib = occl[idx]["contrib"]
                    heat = occl[idx]["heat"]
                    sent_results.append({
                        "text": s, "contrib": round(contrib, 4),
                        "heat": round(heat, 4),
                        "direction": "AI" if contrib > 0.02 else ("human" if contrib < -0.02 else "neutral"),
                    })
            para_results.append({
                "index": pi, "text": para,
                "prob_ai": round(p_prob, 4),
                "label": int(p_label), "region": p_region,
                "grade": _grade(p_prob),
                "sentences": sent_results,
                "ppl": round(self._perplexity(para), 2),
                "burstiness": round(self._burstiness(para), 3),
            })

        evidence = self._feature_evidence(doc_feats)
        result = {
            "title": title,
            "doc_prob_ai": round(doc_prob, 4),
            "doc_label": int(doc_label),
            "doc_verdict": "AI" if doc_label == 1 else "human",
            "doc_region": doc_region,
            "doc_grade": _grade(doc_prob),
            "doc_margin": round(doc_margin, 3),
            "char_count": len(text),
            "word_count": len(text.split()),
            "paragraph_count": len(paragraphs),
            "ppl": round(self._perplexity(text), 2),
            "burstiness": round(self._burstiness(text), 3),
            "feature_evidence": evidence,
            "paragraphs": para_results,
            "grade_distribution": self._grade_dist(para_results),
        }
        if with_suggestions:
            from src.polish_advisor import generate_suggestions
            result["suggestions"] = generate_suggestions(
                doc_feats, self.baselines, text, doc_prob)
        return result

    # ---- ppl / burstiness (the two classic detector signals) ---------------
    def _ensure_lm(self):
        """Lazily load a small causal LM (GPT-2) for real perplexity.
        hf-mirror reachable on CFFF; falls back to None if unavailable."""
        if hasattr(self, "_lm"):
            return
        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        try:
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast
            local = REPO / "models/gpt2"
            src = str(local) if local.exists() else "gpt2"
            self._lm_tok = GPT2TokenizerFast.from_pretrained(src)
            self._lm = GPT2LMHeadModel.from_pretrained(src).to(self.device).eval()
        except Exception as e:
            print(f"[ppl] LM unavailable ({type(e).__name__}); perplexity disabled")
            self._lm = None

    @torch.no_grad()
    def _perplexity(self, text: str) -> float:
        """Real token-level perplexity under GPT-2. Lower ppl = more predictable
        = more AI-like. Returns 0.0 if no LM available."""
        self._ensure_lm()
        if getattr(self, "_lm", None) is None or not text.strip():
            return 0.0
        enc = self._lm_tok(text, return_tensors="pt", truncation=True,
                           max_length=512).to(self.device)
        ids = enc["input_ids"]
        if ids.size(1) < 2:
            return 0.0
        out = self._lm(ids, labels=ids)
        return float(torch.exp(out.loss).item())

    def _burstiness(self, text: str) -> float:
        """Burstiness = std/mean of sentence lengths (in tokens). Human text
        is burstier (higher); AI text is more uniform (lower)."""
        sents = split_sentences(text)
        if len(sents) < 2:
            return 0.0
        lengths = np.array([len(s.split()) for s in sents], dtype=float)
        m = lengths.mean()
        return float(lengths.std() / m) if m > 0 else 0.0

    @staticmethod
    def _grade_dist(paragraphs: list[dict]) -> dict:
        total = len(paragraphs) or 1
        counts = {"high": 0, "middle": 0, "low": 0, "ok": 0}
        for p in paragraphs:
            counts[p["grade"]] += 1
        return {g: round(100 * c / total, 2) for g, c in counts.items()}


def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None, help="raw text to analyze")
    parser.add_argument("--text_file", type=Path, default=None)
    parser.add_argument("--title", type=str, default="Untitled")
    parser.add_argument("--out", type=Path, default=REPO / "outputs/pipeline_demo/result.json")
    parser.add_argument("--no_ppl", action="store_true", help="skip perplexity (faster)")
    args = parser.parse_args()

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    det = RegionAwareDetector()
    if args.no_ppl:
        det._lm = None  # disable ppl
    result = det.analyze(text, title=args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"doc_prob_ai={result['doc_prob_ai']:.4f} verdict={result['doc_verdict']} "
          f"label={result['doc_label']} grade={result['doc_grade']} region={result['doc_region']} "
          f"ppl={result['ppl']} burstiness={result['burstiness']}")
    print(f"paragraphs={result['paragraph_count']} "
          f"grade_dist={result['grade_distribution']}")
    print(f"top evidence: {[(e['feature'], e['signal']) for e in result['feature_evidence'][:5]]}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
