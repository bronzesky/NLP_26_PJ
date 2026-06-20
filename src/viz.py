from __future__ import annotations

import html
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    return paragraphs or [text.strip()]


def score_paragraphs(
    paragraphs: list[str],
    model_dir: Path,
    max_length: int = 512,
    batch_size: int = 8,
) -> pd.DataFrame:
    return score_paragraphs_transformer(
        paragraphs,
        model_dir=model_dir,
        max_length=max_length,
        batch_size=batch_size,
    )


def score_paragraphs_transformer(
    paragraphs: list[str],
    model_dir: Path,
    max_length: int = 512,
    batch_size: int = 8,
) -> pd.DataFrame:
    import pandas as pd

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    except ImportError as exc:
        raise RuntimeError("paragraph_heatmap.py requires torch and transformers.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = _ParagraphDataset(paragraphs, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    rows = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            size = len(batch["input_ids"])
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1).cpu()
            for i, prob in enumerate(probs):
                prob_machine = float(prob[1]) if prob.numel() > 1 else float(prob[0])
                label = int(prob.argmax().item())
                rows.append(
                    {
                        "paragraph_index": offset + i,
                        "label": label,
                        "prob_ai": prob_machine,
                        "prob_machine": prob_machine,
                        "confidence": max(prob_machine, 1.0 - prob_machine),
                        "text": paragraphs[offset + i],
                    }
                )
            offset += size
    return pd.DataFrame(rows)


def score_paragraphs_tfidf(paragraphs: list[str], model_file: Path) -> pd.DataFrame:
    import pandas as pd

    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("tfidf paragraph heatmaps require joblib.") from exc

    model = joblib.load(model_file)
    pred, prob_ai = _predict_ai_probabilities(model, paragraphs)
    rows = []
    for idx, text in enumerate(paragraphs):
        ai_prob = float(prob_ai[idx])
        rows.append(
            {
                "paragraph_index": idx,
                "label": int(pred[idx]),
                "prob_ai": ai_prob,
                "prob_machine": ai_prob,
                "confidence": max(ai_prob, 1.0 - ai_prob),
                "text": text,
            }
        )
    return pd.DataFrame(rows)


def write_heatmap_html(scores: pd.DataFrame, output_file: Path, title: str = "Paragraph heatmap") -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for _, row in scores.iterrows():
        prob = float(row["prob_machine"])
        color = _heat_color(prob)
        label = "machine" if int(row["label"]) == 1 else "human"
        blocks.append(
            f'<section class="para" style="background:{color}">'
            f'<div class="meta">#{int(row["paragraph_index"]) + 1} '
            f'{label} | p(machine)={prob:.3f} | conf={float(row["confidence"]):.3f}</div>'
            f'<p>{html.escape(str(row["text"]))}</p>'
            "</section>"
        )

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #17202a; }}
    main {{ max-width: 960px; margin: 0 auto; }}
    h1 {{ font-size: 28px; margin-bottom: 8px; }}
    .legend {{ display: flex; gap: 12px; align-items: center; margin: 0 0 24px; color: #53606b; }}
    .swatch {{ width: 88px; height: 12px; background: linear-gradient(90deg, #dff3e7, #fff3bf, #f8c8c8); border: 1px solid #d8dee4; }}
    .para {{ border: 1px solid #d8dee4; border-radius: 8px; padding: 16px 18px; margin: 14px 0; }}
    .meta {{ font-size: 13px; font-weight: 650; color: #334155; margin-bottom: 8px; }}
    p {{ margin: 0; line-height: 1.6; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(title)}</h1>
    <div class="legend"><span>human-like</span><span class="swatch"></span><span>machine-like</span></div>
    {''.join(blocks)}
  </main>
</body>
</html>
"""
    output_file.write_text(doc, encoding="utf-8")


class _ParagraphDataset:
    def __init__(self, paragraphs, tokenizer, max_length: int):
        self.encodings = tokenizer(list(paragraphs), truncation=True, max_length=max_length)

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.encodings.items()}


def _heat_color(prob_machine: float) -> str:
    if prob_machine < 0.5:
        alpha = 0.18 + (0.5 - prob_machine) * 0.8
        return f"rgba(46, 160, 67, {alpha:.3f})"
    alpha = 0.18 + (prob_machine - 0.5) * 0.8
    return f"rgba(218, 54, 51, {alpha:.3f})"


def _predict_ai_probabilities(model, texts: list[str]):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("tfidf paragraph heatmaps require numpy.") from exc

    pred = np.asarray(model.predict(texts), dtype=int)

    if hasattr(model, "predict_proba"):
        raw_probs = np.asarray(model.predict_proba(texts), dtype=float)
        classes = np.asarray(getattr(model, "classes_", [0, 1]), dtype=int)
        prob_ai = np.zeros(len(pred), dtype=float)
        for idx, label in enumerate(classes):
            if label == 1:
                prob_ai = raw_probs[:, idx]
        return pred, prob_ai

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(texts), dtype=float)
        if scores.ndim == 2 and scores.shape[1] == 2:
            exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            classes = np.asarray(getattr(model, "classes_", [0, 1]), dtype=int)
            ai_cols = np.where(classes == 1)[0]
            if len(ai_cols) > 0:
                return pred, probs[:, int(ai_cols[0])]
            return pred, probs[:, 1]

        prob_ai = 1.0 / (1.0 + np.exp(-scores))
        return pred, prob_ai

    return pred, pred.astype(float)
# ---- Fusion paragraph scoring ----

_FEATURE_LABELS = {
    "discourse_marker_ratio": "语篇标记词密度高（AI 典型）",
    "modal_verb_ratio": "情态动词密度高",
    "repeated_trigram_ratio": "三元组重复率高",
    "ttr": "词汇多样性低",
    "first_person_ratio": "第一人称代词比例高（human 典型）",
    "avg_sentence_length": "平均句长偏长",
    "contraction_ratio": "缩写词密度高（human 典型）",
}


def score_paragraphs_fusion(
    paragraphs: list[str],
    fusion_model_dir,
    tfidf_model_file,
    roberta_model_dir,
    max_length: int = 512,
    batch_size: int = 8,
) -> "pd.DataFrame":
    import pandas as pd
    import joblib
    import numpy as np

    fusion_model_dir = Path(fusion_model_dir)
    tfidf_model_file = Path(tfidf_model_file)
    roberta_model_dir = Path(roberta_model_dir)

    bundle = joblib.load(fusion_model_dir / "model.joblib")
    clf = bundle["clf"]
    scaler = bundle["scaler"]
    feat_names = bundle["feature_names"]

    tfidf_scores = score_paragraphs_tfidf(paragraphs, model_file=tfidf_model_file)
    roberta_scores = score_paragraphs_transformer(
        paragraphs, model_dir=roberta_model_dir, max_length=max_length, batch_size=batch_size
    )

    from src.features import text_features
    manual = pd.DataFrame([text_features(p) for p in paragraphs])

    base = pd.DataFrame({
        "tfidf_prob_ai": tfidf_scores["prob_ai"].to_numpy(),
        "roberta_prob_ai": roberta_scores["prob_ai"].to_numpy(),
        "roberta_logit_ai": np.zeros(len(paragraphs)),  # not stored in score_paragraphs output
    })

    # combine
    all_cols = ["tfidf_prob_ai", "roberta_prob_ai", "roberta_logit_ai"] + list(manual.columns)
    feat_df = pd.concat([base.reset_index(drop=True), manual.reset_index(drop=True)], axis=1)
    X = feat_df[feat_names].fillna(0.0).to_numpy(dtype=float)
    X_s = scaler.transform(X)
    prob_ai = clf.predict_proba(X_s)[:, 1]

    rows = []
    for i, (p, pa) in enumerate(zip(paragraphs, prob_ai)):
        mf = manual.iloc[i].to_dict()
        top_features = _top_feature_contributions(mf, pa)
        rows.append({
            "paragraph_index": i,
            "label": int(pa >= 0.5),
            "prob_ai": float(pa),
            "prob_machine": float(pa),
            "confidence": max(float(pa), 1.0 - float(pa)),
            "text": p,
            "top_features": top_features,
        })
    return pd.DataFrame(rows)


def _top_feature_contributions(manual_features: dict, prob_ai: float) -> str:
    """Return a human-readable summary of up to 3 notable features."""
    notes = []
    dm = manual_features.get("discourse_marker_ratio", 0)
    if dm > 0.3:
        notes.append(f"语篇标记词密度={dm:.2f}")
    modal = manual_features.get("modal_verb_ratio", 0)
    if modal > 0.08:
        notes.append(f"情态动词比例={modal:.2f}")
    ttr = manual_features.get("ttr", 1.0)
    if ttr < 0.7:
        notes.append(f"词汇多样性低 TTR={ttr:.2f}")
    rep3 = manual_features.get("repeated_trigram_ratio", 0)
    if rep3 > 0.05:
        notes.append(f"三元组重复率={rep3:.2f}")
    fp = manual_features.get("first_person_ratio", 0)
    if fp > 0.05:
        notes.append(f"第一人称比例={fp:.2f}")
    return " | ".join(notes[:3]) if notes else ""


def write_hierarchical_html(scores: "pd.DataFrame", output_file: Path, title: str = "Hierarchical AI Detection") -> None:
    """Enhanced heatmap with article-level probability and per-paragraph feature hints."""
    import html as html_module

    output_file.parent.mkdir(parents=True, exist_ok=True)
    probs = scores["prob_machine"].astype(float).tolist()
    article_prob = sum(probs) / len(probs) if probs else 0.0
    article_label = "机器生成" if article_prob >= 0.5 else "人类写作"
    article_conf = f"{article_prob:.1%}"

    blocks = []
    for _, row in scores.iterrows():
        prob = float(row["prob_machine"])
        color = _heat_color(prob)
        label = "machine" if int(row["label"]) == 1 else "human"
        feat_hint = row.get("top_features", "") or ""
        feat_block = f'<div class="feat">{html_module.escape(feat_hint)}</div>' if feat_hint else ""
        blocks.append(
            f'<section class="para" style="background:{color}">'
            f'<div class="meta">#{int(row["paragraph_index"]) + 1} '
            f'{label} | p(AI)={prob:.3f} | conf={float(row["confidence"]):.3f}</div>'
            f'<p>{html_module.escape(str(row["text"]))}</p>'
            f'{feat_block}'
            "</section>"
        )

    doc = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_module.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #17202a; }}
    main {{ max-width: 960px; margin: 0 auto; }}
    h1 {{ font-size: 28px; margin-bottom: 8px; }}
    .article-summary {{ background: #f0f4fa; border: 1px solid #c3cfe0; border-radius: 8px; padding: 14px 18px; margin: 0 0 20px; }}
    .article-summary strong {{ font-size: 18px; }}
    .model-tag {{ font-size: 12px; color: #6b7280; margin-top: 4px; }}
    .legend {{ display: flex; gap: 12px; align-items: center; margin: 0 0 24px; color: #53606b; }}
    .swatch {{ width: 88px; height: 12px; background: linear-gradient(90deg, rgba(46,160,67,0.6), rgba(218,54,51,0.6)); border: 1px solid #d8dee4; }}
    .para {{ border: 1px solid #d8dee4; border-radius: 8px; padding: 16px 18px; margin: 14px 0; }}
    .meta {{ font-size: 13px; font-weight: 650; color: #334155; margin-bottom: 8px; }}
    .feat {{ font-size: 12px; color: #6b7280; margin-top: 6px; font-style: italic; }}
    p {{ margin: 0; line-height: 1.6; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <main>
    <h1>{html_module.escape(title)}</h1>
    <div class="article-summary">
      <strong>文章整体 AI 概率：{article_conf}（{article_label}）</strong>
      <div class="model-tag">Model: Fusion (TF-IDF + RoBERTa + Linguistic Features)</div>
    </div>
    <div class="legend"><span>human-like</span><span class="swatch"></span><span>machine-like</span></div>
    {''.join(blocks)}
  </main>
</body>
</html>
"""
    output_file.write_text(doc, encoding="utf-8")

# ---- Sentence-level scoring ----

import re as _re
_SENTENCE_RE = _re.compile(r"[^.!?]+[.!?]*")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]


def score_sentences_fusion(
    sentences: list[str],
    fusion_model_dir,
    tfidf_model_file,
    roberta_model_dir,
    max_length: int = 256,
    batch_size: int = 16,
    calibration_file=None,
) -> "pd.DataFrame":
    import pandas as pd
    import joblib
    import json
    import numpy as np
    from pathlib import Path

    fusion_model_dir = Path(fusion_model_dir)
    tfidf_model_file = Path(tfidf_model_file)
    roberta_model_dir = Path(roberta_model_dir)

    bundle = joblib.load(fusion_model_dir / "model.joblib")
    clf = bundle["clf"]
    scaler = bundle["scaler"]
    feat_names = bundle["feature_names"]

    tfidf_scores = score_paragraphs_tfidf(sentences, model_file=tfidf_model_file)
    roberta_scores = score_paragraphs_transformer(
        sentences, model_dir=roberta_model_dir, max_length=max_length, batch_size=batch_size
    )

    roberta_prob_ai = roberta_scores["prob_ai"].to_numpy()
    if calibration_file is not None and Path(calibration_file).exists():
        from src.calibration import TemperatureScaler
        cal_data = json.loads(Path(calibration_file).read_text())
        cal_scaler = TemperatureScaler.from_dict(cal_data)
        eps = 1e-6
        probs_clipped = np.clip(roberta_prob_ai, eps, 1 - eps)
        logits_approx = np.log(probs_clipped / (1 - probs_clipped))
        logit_pairs = np.column_stack([-logits_approx, logits_approx])
        roberta_prob_ai = cal_scaler.predict_positive_proba(logit_pairs)

    from src.features import text_features
    manual = pd.DataFrame([text_features(s) for s in sentences])

    base = pd.DataFrame({
        "tfidf_prob_ai": tfidf_scores["prob_ai"].to_numpy(),
        "roberta_prob_ai": roberta_prob_ai,
        "roberta_logit_ai": np.zeros(len(sentences)),
    })

    feat_df = pd.concat([base.reset_index(drop=True), manual.reset_index(drop=True)], axis=1)
    X = feat_df[feat_names].fillna(0.0).to_numpy(dtype=float)
    X_s = scaler.transform(X)
    prob_ai = clf.predict_proba(X_s)[:, 1]

    rows = []
    for i, (s, pa) in enumerate(zip(sentences, prob_ai)):
        mf = manual.iloc[i].to_dict()
        rows.append({
            "sentence_index": i,
            "label": int(pa >= 0.5),
            "prob_ai": float(pa),
            "confidence": max(float(pa), 1.0 - float(pa)),
            "text": s,
            "top_features": _top_feature_contributions(mf, pa),
        })
    return pd.DataFrame(rows)


# ---- Full hierarchical HTML report ----

_KEY_FEATURES = [
    "discourse_marker_ratio",
    "modal_verb_ratio",
    "ttr",
    "repeated_trigram_ratio",
    "first_person_ratio",
    "contraction_ratio",
    "avg_sentence_length",
    "punctuation_per_token",
]

_FEATURE_DIRECTION = {
    "discourse_marker_ratio": "higher=AI",
    "modal_verb_ratio": "higher=AI",
    "ttr": "lower=AI",
    "repeated_trigram_ratio": "higher=AI",
    "first_person_ratio": "lower=AI",
    "contraction_ratio": "lower=AI",
    "avg_sentence_length": "higher=AI",
    "punctuation_per_token": "neutral",
}


def write_full_report_html(result: dict, output_file, title: str = "AI Text Detection Report") -> None:
    import html as _html
    from pathlib import Path

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    doc_prob = float(result.get("doc_prob_ai", 0.5))
    doc_label = result.get("doc_label", "unknown")
    doc_features = result.get("doc_features", {})
    paragraphs = result.get("paragraphs", [])

    feat_rows = []
    for feat in _KEY_FEATURES:
        if feat not in doc_features:
            continue
        info = doc_features[feat]
        val = float(info.get("value", 0))
        h_mean = float(info.get("human_mean", 0))
        ai_mean = float(info.get("ai_mean", 0))
        signal = info.get("signal", "")
        direction = _FEATURE_DIRECTION.get(feat, "neutral")
        signal_class = "ai-signal" if "AI" in signal else ("human-signal" if "human" in signal.lower() else "")
        feat_rows.append(
            "<tr class=\"" + signal_class + "\">"
            "<td>" + _html.escape(feat) + "</td>"
            "<td>" + f"{val:.3f}" + "</td>"
            "<td>" + f"{h_mean:.3f}" + "</td>"
            "<td>" + f"{ai_mean:.3f}" + "</td>"
            "<td>" + _html.escape(signal) + "</td>"
            "<td style=\"font-size:11px;color:#888\">" + _html.escape(direction) + "</td>"
            "</tr>"
        )
    feat_table = "\n".join(feat_rows)

    para_blocks = []
    for para in paragraphs:
        p_prob = float(para.get("prob_ai", 0.5))
        p_color = _heat_color(p_prob)
        p_label = "machine" if para.get("label", int(p_prob >= 0.5)) == 1 else "human"
        p_feat = _html.escape(str(para.get("top_features", "") or ""))
        p_idx = int(para.get("index", 0))

        sent_items = []
        for sent in para.get("sentences", []):
            s_prob = float(sent.get("prob_ai", 0.5))
            s_color = _heat_color(s_prob)
            s_text = _html.escape(str(sent.get("text", "")))
            sent_items.append(
                "<span class=\"sent\" style=\"background:" + s_color + ";border-radius:3px;padding:1px 3px;display:inline\">"
                + s_text
                + "<sup style=\"font-size:9px;color:#555;margin-left:2px\">" + f"{s_prob:.2f}" + "</sup>"
                + "</span> "
            )
        p_text_escaped = _html.escape(str(para.get("text", "")))
        sent_html = "".join(sent_items) if sent_items else "<p style=\"margin:0\">" + p_text_escaped + "</p>"
        feat_block = "<div class=\"feat\">" + p_feat + "</div>" if p_feat else ""

        para_blocks.append(
            "<section class=\"para\" style=\"background:" + p_color + "\">"
            "<div class=\"meta\">#" + str(p_idx + 1) + " " + p_label
            + " | p(AI)=" + f"{p_prob:.3f}"
            + " | conf=" + f"{max(p_prob, 1-p_prob):.3f}" + "</div>"
            "<div class=\"sent-container\">" + sent_html + "</div>"
            + feat_block
            + "</section>"
        )

    html_parts = [
        "<!doctype html>\n<html lang=\"en\">\n<head>\n",
        "  <meta charset=\"utf-8\">\n",
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n",
        "  <title>" + _html.escape(title) + "</title>\n",
        "  <style>\n",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #17202a; }\n",
        "    main { max-width: 1000px; margin: 0 auto; }\n",
        "    h1 { font-size: 26px; margin-bottom: 8px; }\n",
        "    h2 { font-size: 18px; margin: 24px 0 8px; color: #344054; }\n",
        "    .summary { background: #f0f4fa; border: 1px solid #c3cfe0; border-radius: 8px; padding: 14px 18px; margin: 0 0 20px; }\n",
        "    .summary strong { font-size: 20px; }\n",
        "    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 13px; }\n",
        "    th { background: #f1f5f9; text-align: left; padding: 6px 10px; border-bottom: 2px solid #d1d9e0; }\n",
        "    td { padding: 5px 10px; border-bottom: 1px solid #e8ecf0; }\n",
        "    .ai-signal { background: #fff5f5; }\n",
        "    .human-signal { background: #f0fff4; }\n",
        "    .legend { display: flex; gap: 12px; align-items: center; margin: 0 0 20px; color: #53606b; font-size: 13px; }\n",
        "    .swatch { width: 88px; height: 12px; background: linear-gradient(90deg, rgba(46,160,67,0.5), rgba(218,54,51,0.5)); border: 1px solid #d8dee4; }\n",
        "    .para { border: 1px solid #d8dee4; border-radius: 8px; padding: 14px 16px; margin: 12px 0; }\n",
        "    .meta { font-size: 12px; font-weight: 650; color: #334155; margin-bottom: 8px; }\n",
        "    .sent-container { line-height: 1.9; }\n",
        "    .feat { font-size: 11px; color: #6b7280; margin-top: 6px; font-style: italic; border-top: 1px solid #e8ecf0; padding-top: 4px; }\n",
        "  </style>\n</head>\n<body>\n  <main>\n",
        "    <h1>" + _html.escape(title) + "</h1>\n",
        "    <div class=\"summary\"><strong>Overall AI probability: " + f"{doc_prob:.1%}" + " (" + _html.escape(doc_label) + ")</strong></div>\n",
        "    <h2>Linguistic Feature Analysis</h2>\n",
        "    <table><thead><tr><th>Feature</th><th>This doc</th><th>Human avg</th><th>AI avg</th><th>Signal</th><th>Direction</th></tr></thead>\n",
        "    <tbody>" + feat_table + "</tbody></table>\n",
        "    <h2>Paragraph / Sentence Breakdown</h2>\n",
        "    <div class=\"legend\"><span>human-like</span><span class=\"swatch\"></span><span>AI-like</span><span style=\"margin-left:12px;font-size:11px\">superscript = sentence AI prob</span></div>\n",
        "\n".join(para_blocks) + "\n",
        "  </main>\n</body>\n</html>\n",
    ]
    output_file.write_text("".join(html_parts), encoding="utf-8")
"""
Append to src/viz.py: ngram phrase highlighting via TF-IDF coefficients.
"""


def get_ngram_highlights(
    text: str,
    tfidf_model_file,
    top_k: int = 10,
    min_contribution: float = 0.3,
):
    """Return top-k AI-like and human-like ngrams found in text, with their contributions.

    contribution = tfidf_value * logistic_regression_coef
    Positive contribution = AI-like evidence
    Negative contribution = human-like evidence

    Returns:
        dict with keys 'ai_phrases' and 'human_phrases', each a list of
        {'phrase': str, 'contribution': float, 'start': int, 'end': int}
    """
    import numpy as np
    import joblib
    from pathlib import Path

    tfidf_model_file = Path(tfidf_model_file)
    if not tfidf_model_file.exists():
        return {"ai_phrases": [], "human_phrases": []}

    model = joblib.load(tfidf_model_file)
    tfidf = model.named_steps["tfidf"]
    lr = model.named_steps["clf"]
    coef = lr.coef_[0]  # shape (vocab_size,)

    # Get TF-IDF vector for this text
    vec = tfidf.transform([text])  # (1, vocab_size) sparse
    inv_vocab = {v: k for k, v in tfidf.vocabulary_.items()}

    # Find non-zero entries
    cx = vec.tocoo()
    contributions = []
    for _, col, val in zip(cx.row, cx.col, cx.data):
        phrase = inv_vocab[col]
        contribution = float(val * coef[col])
        if abs(contribution) < min_contribution:
            continue
        contributions.append({"phrase": phrase, "contribution": contribution})

    # Sort and split
    ai_phrases = sorted(
        [c for c in contributions if c["contribution"] > 0],
        key=lambda x: -x["contribution"]
    )[:top_k]
    human_phrases = sorted(
        [c for c in contributions if c["contribution"] < 0],
        key=lambda x: x["contribution"]
    )[:top_k]

    # Find character offsets in text (case-insensitive)
    text_lower = text.lower()
    for entry_list in [ai_phrases, human_phrases]:
        for entry in entry_list:
            phrase = entry["phrase"]
            idx = text_lower.find(phrase)
            entry["start"] = idx
            entry["end"] = idx + len(phrase) if idx >= 0 else -1

    return {"ai_phrases": ai_phrases, "human_phrases": human_phrases}


def highlight_text_with_ngrams(text: str, ai_phrases: list, human_phrases: list) -> str:
    """Return HTML string with AI phrases underlined red, human phrases underlined green.

    Uses character offsets. Overlapping spans are handled by taking the highest
    |contribution| span first.
    """
    import html as _html

    # Build list of (start, end, label, contribution)
    spans = []
    for p in ai_phrases:
        if p["start"] >= 0:
            spans.append((p["start"], p["end"], "ai", p["contribution"]))
    for p in human_phrases:
        if p["start"] >= 0:
            spans.append((p["start"], p["end"], "human", p["contribution"]))

    if not spans:
        return _html.escape(text)

    # Sort by contribution magnitude descending, then greedily pick non-overlapping
    spans.sort(key=lambda x: -abs(x[2 + 1]))  # sort by abs(contribution)
    selected = []
    used = set()
    for start, end, label, contrib in spans:
        if any(i in used for i in range(start, end)):
            continue
        selected.append((start, end, label, contrib))
        for i in range(start, end):
            used.add(i)

    # Sort by position for rendering
    selected.sort(key=lambda x: x[0])

    # Build HTML with highlighted spans
    result = []
    prev = 0
    for start, end, label, contrib in selected:
        result.append(_html.escape(text[prev:start]))
        color = "#da3633" if label == "ai" else "#2ea043"
        bg = "rgba(218,54,51,0.12)" if label == "ai" else "rgba(46,160,67,0.12)"
        tooltip = f"AI-like (+{contrib:.2f})" if label == "ai" else f"Human-like ({contrib:.2f})"
        result.append(
            f'<mark style="background:{bg};border-bottom:2px solid {color};'
            f'border-radius:2px;padding:0 1px" title="{tooltip}">'
            f'{_html.escape(text[start:end])}'
            f'</mark>'
        )
        prev = end
    result.append(_html.escape(text[prev:]))
    return "".join(result)


def score_sentences_hsad(
    text: str,
    model_dir,
    roberta_model_dir,
    max_length: int = 512,
    max_sents: int = 64,
    compute_ig: bool = False,
    n_ig_steps: int = 50,
) -> "pd.DataFrame":
    """
    Run HSAD on a document and return per-sentence AI probabilities.
    Optionally compute Integrated Gradients token attribution.

    Returns DataFrame: sentence_index, text, prob_ai, label, ig_html (if compute_ig)
    """
    import re
    import torch
    import pandas as pd
    from pathlib import Path

    SENT_RE = re.compile(r"[^.!?\n]+[.!?\n]*")

    model_dir = Path(model_dir)
    roberta_model_dir = Path(roberta_model_dir)

    from src.model_hsad import HSAD
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(roberta_model_dir), use_fast=True)
    model = HSAD.from_pretrained(str(model_dir), roberta_model_dir=str(roberta_model_dir))
    model = model.to(device)
    model.eval()

    sentences = [s.strip() for s in SENT_RE.findall(text) if s.strip()][:max_sents]
    if not sentences:
        sentences = [text[:200]]

    enc = tokenizer(text, truncation=True, max_length=max_length,
                    return_tensors="pt", return_offsets_mapping=True)
    input_ids = enc["input_ids"].squeeze(0).to(device)
    attention_mask = enc["attention_mask"].squeeze(0).to(device)
    offsets = enc["offset_mapping"].squeeze(0).tolist()

    # Compute sent spans
    sent_spans = []
    search_from = 0
    for sent in sentences:
        char_start = text.find(sent, search_from)
        if char_start == -1:
            char_start = search_from
        char_end = char_start + len(sent)
        search_from = char_start + 1
        tok_start, tok_end = None, None
        for i, (a, b) in enumerate(offsets):
            if a == 0 and b == 0:
                continue
            if tok_start is None and b > char_start:
                tok_start = i
            if b <= char_end:
                tok_end = i
        if tok_start is None:
            tok_start = 1
        if tok_end is None or tok_end < tok_start:
            tok_end = tok_start
        sent_spans.append((tok_start, tok_end + 1))

    with torch.no_grad():
        out = model(input_ids, attention_mask, sent_spans)

    sent_probs = torch.softmax(out["sent_logits"], dim=1)[:, 1].cpu().numpy()
    doc_prob = float(torch.softmax(out["doc_logits"], dim=0)[1].item())

    rows = []
    for i, (sent, prob) in enumerate(zip(sentences, sent_probs)):
        row = {
            "sentence_index": i,
            "text": sent,
            "prob_ai": round(float(prob), 4),
            "label": int(float(prob) >= 0.5),
            "ig_html": "",
        }
        rows.append(row)

    if compute_ig:
        try:
            from src.ig_attribution import compute_token_attribution, attribution_to_html_spans
            attrs = compute_token_attribution(model, tokenizer, text, sent_spans,
                                              n_steps=n_ig_steps, device=device)
            ig_html = attribution_to_html_spans(attrs)
            # Assign the full-doc IG html to first row only (document-level)
            if rows:
                rows[0]["ig_html"] = ig_html
        except Exception as e:
            pass  # IG is optional

    df = pd.DataFrame(rows)
    df.attrs["doc_prob_ai"] = doc_prob
    return df
