"""
scripts/compare_before_after.py

Compare an original AI-generated text with a human-edited (polished) version.
Outputs a side-by-side HTML report showing:
  - AI probability change (before vs after)
  - Feature-by-feature comparison table
  - Phrase-level highlights for both versions
  - Which AI features were successfully reduced

Usage:
    python scripts/compare_before_after.py \
        --before path/to/original.txt \
        --after path/to/polished.txt \
        --output_dir outputs/compare_demo

Or inline:
    python scripts/compare_before_after.py \
        --before_text "AI generated text here..." \
        --after_text "Polished text here..."
"""
from __future__ import annotations

import argparse
import html as _html
import json
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.features import text_features
from src.viz import (
    score_paragraphs_fusion,
    split_paragraphs,
    get_ngram_highlights,
    highlight_text_with_ngrams,
)

_KEY_FEATURES = [
    "discourse_marker_ratio",
    "modal_verb_ratio",
    "ttr",
    "repeated_trigram_ratio",
    "first_person_ratio",
    "contraction_ratio",
    "avg_sentence_length",
    "punctuation_per_token",
    "sentence_length_std",
    "word_count",
]

_FEATURE_AI_DIRECTION = {
    "discourse_marker_ratio": "higher=AI",
    "modal_verb_ratio": "higher=AI",
    "ttr": "lower=AI",
    "repeated_trigram_ratio": "higher=AI",
    "first_person_ratio": "lower=AI",
    "contraction_ratio": "lower=AI",
    "avg_sentence_length": "higher=AI",
    "punctuation_per_token": "neutral",
    "sentence_length_std": "lower=AI",
    "word_count": "neutral",
}


def score_text(text: str, fusion_model_dir: Path, tfidf_model_file: Path,
               roberta_model_dir: Path, max_length: int, batch_size: int) -> float:
    """Return document-level AI probability via paragraph mean."""
    import numpy as np
    paragraphs = split_paragraphs(text)
    scores = score_paragraphs_fusion(
        paragraphs,
        fusion_model_dir=fusion_model_dir,
        tfidf_model_file=tfidf_model_file,
        roberta_model_dir=roberta_model_dir,
        max_length=max_length,
        batch_size=batch_size,
    )
    return float(np.mean(scores["prob_ai"].tolist()))


def feature_change_signal(feat: str, before_val: float, after_val: float) -> str:
    """Return whether the change in this feature moved away from AI."""
    direction = _FEATURE_AI_DIRECTION.get(feat, "neutral")
    if direction == "neutral":
        return "neutral"
    delta = after_val - before_val
    if direction == "higher=AI":
        return "improved" if delta < -0.001 else ("worsened" if delta > 0.001 else "unchanged")
    else:  # lower=AI
        return "improved" if delta > 0.001 else ("worsened" if delta < -0.001 else "unchanged")


def build_html(before_text: str, after_text: str,
               before_prob: float, after_prob: float,
               before_feats: dict, after_feats: dict,
               before_highlights: dict, after_highlights: dict,
               title: str = "AI Text: Before vs After Polishing") -> str:

    # Feature table rows
    feat_rows = []
    for feat in _KEY_FEATURES:
        bv = before_feats.get(feat, 0.0)
        av = after_feats.get(feat, 0.0)
        signal = feature_change_signal(feat, bv, av)
        delta = av - bv
        delta_str = f"{delta:+.3f}"
        signal_class = {"improved": "improved", "worsened": "worsened"}.get(signal, "")
        direction = _FEATURE_AI_DIRECTION.get(feat, "neutral")
        feat_rows.append(
            f'<tr class="{signal_class}">'
            f'<td>{_html.escape(feat)}</td>'
            f'<td>{bv:.3f}</td>'
            f'<td>{av:.3f}</td>'
            f'<td>{delta_str}</td>'
            f'<td style="font-size:11px;color:#888">{direction}</td>'
            f'<td>{_html.escape(signal)}</td>'
            f'</tr>'
        )

    # Highlighted text
    before_html = highlight_text_with_ngrams(
        before_text, before_highlights["ai_phrases"], before_highlights["human_phrases"]
    )
    after_html = highlight_text_with_ngrams(
        after_text, after_highlights["ai_phrases"], after_highlights["human_phrases"]
    )

    # AI phrase lists
    def phrase_list(phrases, color):
        if not phrases:
            return "<em style='color:#888'>none found</em>"
        items = "".join(
            f'<li><mark style="background:rgba({color},0.15);border-bottom:2px solid rgba({color},0.8)">'
            f'{_html.escape(p["phrase"])}</mark> ({p["contribution"]:+.2f})</li>'
            for p in phrases
        )
        return f"<ul style='margin:4px 0;padding-left:18px'>{items}</ul>"

    before_ai_list = phrase_list(before_highlights["ai_phrases"], "218,54,51")
    after_ai_list = phrase_list(after_highlights["ai_phrases"], "218,54,51")
    before_human_list = phrase_list(before_highlights["human_phrases"], "46,160,67")
    after_human_list = phrase_list(after_highlights["human_phrases"], "46,160,67")

    prob_delta = after_prob - before_prob
    prob_delta_str = f"{prob_delta:+.1%}"
    prob_color = "#2ea043" if prob_delta < 0 else "#da3633"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{_html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #17202a; }}
    main {{ max-width: 1100px; margin: 0 auto; }}
    h1 {{ font-size: 24px; margin-bottom: 6px; }}
    h2 {{ font-size: 16px; margin: 20px 0 8px; color: #344054; border-bottom: 1px solid #e8ecf0; padding-bottom: 4px; }}
    .prob-banner {{ display: flex; gap: 24px; margin: 0 0 20px; }}
    .prob-box {{ flex: 1; padding: 14px 18px; border-radius: 8px; border: 1px solid #d1d9e0; }}
    .prob-box .label {{ font-size: 12px; color: #6b7280; margin-bottom: 4px; }}
    .prob-box .value {{ font-size: 28px; font-weight: 700; }}
    .prob-box.before .value {{ color: #da3633; }}
    .prob-box.after .value {{ color: #2ea043; }}
    .delta-box {{ flex: 0.5; padding: 14px 18px; border-radius: 8px; background: #f0f4fa; border: 1px solid #c3cfe0; display: flex; flex-direction: column; justify-content: center; align-items: center; }}
    .delta-box .value {{ font-size: 28px; font-weight: 700; color: {prob_color}; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-bottom: 20px; }}
    th {{ background: #f1f5f9; text-align: left; padding: 6px 10px; border-bottom: 2px solid #d1d9e0; }}
    td {{ padding: 5px 10px; border-bottom: 1px solid #e8ecf0; }}
    .improved {{ background: #f0fff4; }}
    .worsened {{ background: #fff5f5; }}
    .side-by-side {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    .text-box {{ border: 1px solid #d8dee4; border-radius: 8px; padding: 16px; line-height: 1.8; font-size: 14px; }}
    .text-box h3 {{ margin: 0 0 10px; font-size: 14px; color: #344054; }}
    .phrase-section {{ font-size: 13px; }}
    mark {{ border-radius: 2px; padding: 0 1px; }}
    .legend {{ font-size: 12px; color: #6b7280; margin-bottom: 8px; }}
  </style>
</head>
<body><main>
  <h1>{_html.escape(title)}</h1>

  <div class="prob-banner">
    <div class="prob-box before">
      <div class="label">Before (AI original)</div>
      <div class="value">{before_prob:.1%}</div>
    </div>
    <div class="delta-box">
      <div class="label" style="font-size:12px;color:#6b7280">Change</div>
      <div class="value">{prob_delta_str}</div>
    </div>
    <div class="prob-box after">
      <div class="label">After (polished)</div>
      <div class="value">{after_prob:.1%}</div>
    </div>
  </div>

  <h2>Feature Comparison</h2>
  <table>
    <thead><tr>
      <th>Feature</th><th>Before</th><th>After</th><th>Delta</th>
      <th>AI direction</th><th>Signal</th>
    </tr></thead>
    <tbody>{"".join(feat_rows)}</tbody>
  </table>

  <h2>Text with Phrase Highlights</h2>
  <div class="legend">
    <mark style="background:rgba(218,54,51,0.12);border-bottom:2px solid #da3633;border-radius:2px;padding:0 3px">AI-like phrase</mark>
    &nbsp;
    <mark style="background:rgba(46,160,67,0.12);border-bottom:2px solid #2ea043;border-radius:2px;padding:0 3px">Human-like phrase</mark>
    &nbsp; (hover for contribution score)
  </div>
  <div class="side-by-side">
    <div class="text-box">
      <h3>Before ({before_prob:.1%} AI)</h3>
      <div>{before_html}</div>
    </div>
    <div class="text-box">
      <h3>After ({after_prob:.1%} AI)</h3>
      <div>{after_html}</div>
    </div>
  </div>

  <h2>Top AI-like Phrases</h2>
  <div class="side-by-side">
    <div class="phrase-section"><strong>Before:</strong>{before_ai_list}</div>
    <div class="phrase-section"><strong>After:</strong>{after_ai_list}</div>
  </div>

  <h2>Top Human-like Phrases</h2>
  <div class="side-by-side">
    <div class="phrase-section"><strong>Before:</strong>{before_human_list}</div>
    <div class="phrase-section"><strong>After:</strong>{after_human_list}</div>
  </div>

</main></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, default=None)
    parser.add_argument("--after", type=Path, default=None)
    parser.add_argument("--before_text", type=str, default=None)
    parser.add_argument("--after_text", type=str, default=None)
    parser.add_argument("--fusion_model_dir", type=Path,
                        default=PROJ_ROOT / "outputs/fusion_lgbm_calibrated")
    parser.add_argument("--tfidf_model_file", type=Path,
                        default=PROJ_ROOT / "outputs/tfidf/model.joblib")
    parser.add_argument("--roberta_model_dir", type=Path,
                        default=PROJ_ROOT / "outputs/roberta_base/best_model")
    parser.add_argument("--output_dir", type=Path,
                        default=PROJ_ROOT / "outputs/compare_demo")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    before_text = args.before_text or (args.before.read_text(encoding="utf-8") if args.before else None)
    after_text = args.after_text or (args.after.read_text(encoding="utf-8") if args.after else None)
    if not before_text or not after_text:
        raise SystemExit("Provide --before/--after files or --before_text/--after_text")

    print("Scoring original text...")
    before_prob = score_text(before_text, args.fusion_model_dir, args.tfidf_model_file,
                             args.roberta_model_dir, args.max_length, args.batch_size)
    print(f"  Before AI probability: {before_prob:.1%}")

    print("Scoring polished text...")
    after_prob = score_text(after_text, args.fusion_model_dir, args.tfidf_model_file,
                            args.roberta_model_dir, args.max_length, args.batch_size)
    print(f"  After AI probability:  {after_prob:.1%}")
    print(f"  Change: {after_prob - before_prob:+.1%}")

    before_feats = text_features(before_text)
    after_feats = text_features(after_text)

    before_highlights = get_ngram_highlights(before_text, args.tfidf_model_file, top_k=10)
    after_highlights = get_ngram_highlights(after_text, args.tfidf_model_file, top_k=10)

    html = build_html(before_text, after_text, before_prob, after_prob,
                      before_feats, after_feats, before_highlights, after_highlights)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.output_dir / "comparison.html"
    html_path.write_text(html, encoding="utf-8")

    result = {
        "before_prob_ai": round(before_prob, 4),
        "after_prob_ai": round(after_prob, 4),
        "delta": round(after_prob - before_prob, 4),
        "feature_before": {k: round(float(before_feats.get(k, 0)), 4) for k in _KEY_FEATURES},
        "feature_after": {k: round(float(after_feats.get(k, 0)), 4) for k in _KEY_FEATURES},
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nHTML report: {html_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
