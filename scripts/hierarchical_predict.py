"""
scripts/hierarchical_predict.py  v2
HAN + DeBERTa-v3-large based hierarchical AI text detection.
"""
from __future__ import annotations

import argparse, json, os, re, sys
from pathlib import Path

import torch

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from src.features_v2 import full_features
from src.polish_advisor import generate_suggestions
from src.viz import write_full_report_html

SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*")
PARA_RE = re.compile(r"\n\s*\n+")

DEFAULT_MODEL = PROJ_ROOT / "outputs/han_deberta_large_full/best_model"
DEFAULT_TOKENIZER = PROJ_ROOT / "models/deberta-v3-large"
DEFAULT_BASELINES = PROJ_ROOT / "data/feature_baselines.json"


def split_sentences(text, max_sents=128):
    return [s.strip() for s in SENTENCE_RE.findall(text) if s.strip()][:max_sents]


def split_paragraphs(text):
    parts = [p.strip() for p in PARA_RE.split(text) if p.strip()]
    return parts if parts else [text.strip()]


def para_boundaries(sentences, paragraphs):
    para_lens = [max(len(split_sentences(p)), 1) for p in paragraphs]
    total = sum(para_lens)
    n_sents = len(sentences)
    bounds, start = [], 0
    for pl in para_lens:
        n = max(1, round(pl / total * n_sents))
        end = min(start + n, n_sents)
        if start < end:
            bounds.append((start, end))
        start = end
        if start >= n_sents:
            break
    if not bounds or bounds[-1][1] < n_sents:
        bounds = [(0, n_sents)]
    return bounds


def run_han(text, model, tokenizer, device, max_sent_len=96, max_sents=128):
    sentences = split_sentences(text, max_sents) or [text[:300]]
    paragraphs = split_paragraphs(text)
    bounds = para_boundaries(sentences, paragraphs)
    enc = tokenizer(sentences, padding="max_length", truncation=True,
                    max_length=max_sent_len, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model(ids, mask, bounds)
    doc_prob = float(torch.softmax(out["doc_logits"], dim=0)[1].item())
    return {
        "doc_prob_ai": doc_prob,
        "sentences": sentences,
        "paragraphs": paragraphs,
        "para_boundaries": bounds,
        "sent_attention_weights": out["sent_attention_weights"].cpu().numpy().tolist(),
        "para_attention_weights": out["para_attention_weights"].cpu().numpy().tolist(),
    }


def build_result(text, han, features, baselines):
    doc_prob = han["doc_prob_ai"]
    sentences, paragraphs = han["sentences"], han["paragraphs"]
    bounds, sent_w, para_w = han["para_boundaries"], han["sent_attention_weights"], han["para_attention_weights"]

    doc_features = {}
    for feat, val in features.items():
        if feat not in baselines:
            continue
        b = baselines[feat]
        h, a = b.get("human_mean", 0), b.get("ai_mean", 0)
        dev = (val - h) / abs(a - h) if abs(a - h) > 1e-6 else 0.0
        sig = "AI-like" if dev > 0.3 else ("human-like" if dev < -0.3 else "neutral")
        doc_features[feat] = {"value": round(float(val), 4), "human_mean": round(float(h), 4),
                              "ai_mean": round(float(a), 4), "signal": sig, "deviation": round(float(dev), 3)}

    result_paragraphs = []
    for pi, (ps, pe) in enumerate(bounds):
        if pi >= len(paragraphs):
            break
        para_text = paragraphs[pi]
        pa = float(para_w[pi]) if pi < len(para_w) else 1.0 / len(bounds)
        pd = min(pa * len(bounds) * doc_prob, 1.0)
        sents_out = []
        for si in range(ps, min(pe, len(sentences))):
            sa = float(sent_w[si]) if si < len(sent_w) else 0.0
            sd = min(sa * len(sentences) * doc_prob, 1.0)
            sents_out.append({"index": si - ps, "text": sentences[si],
                              "prob_ai": round(sd, 4), "attention_weight": round(sa, 6)})
        pf = full_features(para_text)
        hints = []
        for feat in ["discourse_total_density", "contraction_ratio", "first_person_ratio"]:
            if feat in pf and feat in baselines:
                v = pf[feat]; am = baselines[feat].get("ai_mean", 0); hm = baselines[feat].get("human_mean", 0)
                if abs(am - hm) > 1e-6 and (v - hm) / abs(am - hm) > 0.5:
                    hints.append(f"{feat}={v:.3f}")
        result_paragraphs.append({
            "index": pi, "text": para_text, "prob_ai": round(pd, 4), "label": int(pd >= 0.5),
            "attention_weight": round(pa, 6), "top_features": " | ".join(hints[:3]), "sentences": sents_out,
        })

    suggestions = generate_suggestions(features, baselines, text, doc_prob)
    return {
        "doc_prob_ai": round(doc_prob, 4),
        "doc_label": "machine" if doc_prob >= 0.5 else "human",
        "doc_features": doc_features,
        "paragraphs": result_paragraphs,
        "polishing": suggestions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=None)
    parser.add_argument("--input_file", type=Path, default=None)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer_dir", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--feature_baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--output_dir", type=Path, default=PROJ_ROOT / "outputs/hierarchical_demo")
    parser.add_argument("--max_sent_len", type=int, default=96)
    parser.add_argument("--max_sents", type=int, default=128)
    parser.add_argument("--no_html", action="store_true")
    parser.add_argument("--model_type", default="han", choices=["han", "hsad"])
    parser.add_argument("--roberta_model_dir", type=Path,
                        default=PROJ_ROOT / "outputs/roberta_base/best_model")
    parser.add_argument("--compute_ig", action="store_true")
    args = parser.parse_args()

    if args.text:
        text = args.text
    elif args.input_file:
        if str(args.input_file).endswith(".jsonl"):
            text = json.loads(args.input_file.read_text().splitlines()[0]).get("text", "")
        else:
            text = args.input_file.read_text(encoding="utf-8")
    else:
        raise SystemExit("Provide --text or --input_file")

    baselines = {}
    if args.feature_baselines.exists():
        raw = json.loads(args.feature_baselines.read_text())
        baselines = {k: v for k, v in raw.items() if not k.startswith("_")}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model_dir}...")

    if not args.model_dir.exists():
        print("Model not found. Running feature-only mode.")
        han_output = None
    elif args.model_type == "hsad":
        import sys as _sys, json as _json
        from src.viz import score_sentences_hsad, write_full_report_html
        print("Running HSAD inference...")
        sent_df = score_sentences_hsad(
            text, model_dir=args.model_dir,
            roberta_model_dir=args.roberta_model_dir,
            compute_ig=args.compute_ig,
        )
        doc_prob = sent_df.attrs.get("doc_prob_ai", 0.5)
        para_sentences = []
        for _, row in sent_df.iterrows():
            para_sentences.append({"index": int(row["sentence_index"]),
                "text": row["text"], "prob_ai": float(row["prob_ai"]),
                "ig_html": row.get("ig_html", "")})
        features_local = full_features(text)
        result = {
            "doc_prob_ai": round(doc_prob, 4),
            "doc_label": "machine" if doc_prob >= 0.5 else "human",
            "doc_features": {k: {"value": round(float(v), 4), "signal": "unknown",
                "human_mean": round(baselines.get(k, {}).get("human_mean", 0), 4),
                "ai_mean": round(baselines.get(k, {}).get("ai_mean", 0), 4)}
                for k, v in features_local.items() if k in baselines},
            "paragraphs": [{"index": 0, "text": text, "prob_ai": round(doc_prob, 4),
                "label": int(doc_prob >= 0.5), "top_features": "",
                "sentences": para_sentences}],
            "polishing": generate_suggestions(features_local, baselines, text, doc_prob),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "result.json").write_text(
            _json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        if not args.no_html:
            write_full_report_html(result, args.output_dir / "report.html")
            print(f"HTML: {args.output_dir / chr(39)}report.html{chr(39)}")
        print(f"Document AI probability: {doc_prob:.1%}")
        _sys.exit(0)
    else:
        from src.model_han import HierarchicalDetector
        from transformers import AutoTokenizer
        model = HierarchicalDetector.from_pretrained(str(args.model_dir)).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
        han_output = run_han(text, model, tokenizer, device, args.max_sent_len, args.max_sents)

    print("Computing linguistic features...")
    features = full_features(text)

    if han_output:
        result = build_result(text, han_output, features, baselines)
    else:
        result = {
            "doc_prob_ai": None, "doc_label": "unknown",
            "doc_features": {k: {"value": round(float(v), 4), "signal": "unknown",
                                 "human_mean": round(baselines.get(k, {}).get("human_mean", 0), 4),
                                 "ai_mean": round(baselines.get(k, {}).get("ai_mean", 0), 4)}
                             for k, v in features.items() if k in baselines},
            "paragraphs": [],
            "polishing": generate_suggestions(features, baselines, text, 0.5),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if result["doc_prob_ai"] is not None:
        print(f"\nDocument AI probability: {result['doc_prob_ai']:.1%} ({result['doc_label']})")
    print(f"Suggestions: {result['polishing']['total_suggestions']} across 4 tiers")
    print(f"Priority: {result['polishing']['priority_features']}")

    if not args.no_html:
        html_path = args.output_dir / "report.html"
        write_full_report_html(result, html_path)
        print(f"HTML: {html_path}")

    print("\n" + "=" * 60 + "\nCOMPOSITE POLISHING PROMPT:\n" + "=" * 60)
    print(result["polishing"]["composite_prompt"].replace("{TEXT}", "[paste your text here]"))


if __name__ == "__main__":
    main()
