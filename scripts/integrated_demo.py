"""
scripts/integrated_demo.py

One-page integrated demo:
  1. Sentence-level AI probability heatmap (query-style highlight)
  2. Linguistic feature analysis with human/AI comparison
  3. Natural-language polishing suggestions (4 tiers)
  4. Composite prompt ready to paste into ChatGPT for rewriting
  5. Optional before/after comparison if --after_text is provided
"""
from __future__ import annotations
import argparse, html as _html, json, sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.viz import score_sentences_fusion, split_sentences
from src.features_v2 import full_features
from src.polish_advisor import generate_suggestions

_HEAT_STOPS = [
    (0.0,  "#f0f4ff"),
    (0.3,  "#fef3cd"),
    (0.5,  "#fde8b0"),
    (0.7,  "#f8c471"),
    (0.85, "#f0a500"),
    (1.0,  "#c0392b"),
]

def heat_color(p: float) -> str:
    for i in range(len(_HEAT_STOPS) - 1):
        lo, c_lo = _HEAT_STOPS[i]
        hi, c_hi = _HEAT_STOPS[i + 1]
        if lo <= p <= hi:
            t = (p - lo) / (hi - lo)
            def lerp_hex(a, b, t):
                ar, ag, ab = int(a[1:3],16), int(a[3:5],16), int(a[5:7],16)
                br, bg, bb = int(b[1:3],16), int(b[3:5],16), int(b[5:7],16)
                return "#{:02x}{:02x}{:02x}".format(
                    int(ar + (br-ar)*t), int(ag + (bg-ag)*t), int(ab + (bb-ab)*t))
            return lerp_hex(c_lo, c_hi, t)
    return _HEAT_STOPS[-1][1]

def render_sentence_highlight(sentences, probs) -> str:
    parts = []
    for sent, p in zip(sentences, probs):
        color = heat_color(p)
        border = "#c0392b" if p >= 0.7 else ("#f0a500" if p >= 0.4 else "#ccc")
        escaped = _html.escape(sent)
        tooltip = f"AI probability: {p:.1%}"
        parts.append(
            f'<span class="sent-chip" style="background:{color};border-color:{border}" '
            f'title="{tooltip}" data-prob="{p:.3f}">{escaped}</span>'
        )
    return " ".join(parts)

def render_feature_table(feats: dict, baselines: dict) -> str:
    KEY_FEATS = [
        ("discourse_total_density", "Discourse markers", "higher=AI"),
        ("contraction_ratio",       "Contractions",      "lower=AI"),
        ("first_person_ratio",      "1st-person pronouns","lower=AI"),
        ("sentence_length_cv",      "Sentence length variation","lower=AI"),
        ("mattr",                   "Vocabulary diversity","lower=AI"),
        ("latinate_ratio",          "Latinate words",    "higher=AI"),
        ("passive_ratio",           "Passive voice",     "higher=AI"),
        ("hedge_density",           "Hedging phrases",   "higher=AI"),
    ]
    rows = []
    for key, label, direction in KEY_FEATS:
        if key not in feats or key not in baselines: continue
        val = float(feats[key])
        hm  = float(baselines[key].get("human_mean", 0))
        am  = float(baselines[key].get("ai_mean", 0))
        if abs(am - hm) < 1e-8: continue
        dev = (val - hm) / abs(am - hm)
        if "higher=AI" in direction:
            ai_signal = dev > 0.3
        else:
            ai_signal = dev < -0.3
        signal_class = "ai-sig" if ai_signal else ("human-sig" if (
            ("higher=AI" in direction and dev < -0.3) or
            ("lower=AI"  in direction and dev >  0.3)) else "neutral-sig")
        bar_pct = min(abs(dev) * 50, 100)
        bar_color = "#e74c3c" if ai_signal else "#27ae60"
        rows.append(
            f'<tr class="{signal_class}">'
            f'<td>{_html.escape(label)}</td>'
            f'<td style="font-family:monospace">{val:.4f}</td>'
            f'<td style="font-family:monospace;color:#888">{hm:.4f}</td>'
            f'<td style="font-family:monospace;color:#888">{am:.4f}</td>'
            f'<td><div class="bar" style="width:{bar_pct:.0f}%;background:{bar_color}"></div></td>'
            f'<td style="font-size:11px;color:#666">{direction}</td>'
            f'</tr>'
        )
    return "\n".join(rows)

def render_suggestions(suggestions: dict) -> str:
    tier_labels = {
        "tier1_lexical":   ("📝", "Lexical",   "#3498db"),
        "tier2_syntactic": ("🔧", "Syntactic", "#9b59b6"),
        "tier3_discourse": ("🔗", "Discourse", "#e67e22"),
        "tier4_pragmatic": ("💬", "Pragmatic", "#27ae60"),
    }
    parts = []
    for key, (icon, label, color) in tier_labels.items():
        items = suggestions.get(key, [])
        if not items: continue
        for item in items:
            sg = _html.escape(item.get("suggestion", ""))
            parts.append(
                f'<div class="suggestion-card" style="border-left:4px solid {color}">'
                f'<span class="tier-badge" style="background:{color}">{icon} {label}</span>'
                f'<p>{sg}</p>'
                f'</div>'
            )
    return "\n".join(parts) if parts else "<p style='color:#888'>No suggestions — text looks human-like!</p>"

def build_html(text: str, sentences, probs, doc_prob: float,
               feats: dict, baselines: dict, suggestions: dict,
               after_text: str = None, after_prob: float = None,
               title: str = "AI Text Analysis") -> str:

    sent_html     = render_sentence_highlight(sentences, probs)
    feature_rows  = render_feature_table(feats, baselines)
    suggestion_html = render_suggestions(suggestions)
    composite_prompt = _html.escape(suggestions.get("composite_prompt", ""))
    prob_color = "#e74c3c" if doc_prob >= 0.7 else ("#f39c12" if doc_prob >= 0.4 else "#27ae60")
    prob_label = "Likely AI" if doc_prob >= 0.6 else ("Mixed" if doc_prob >= 0.35 else "Likely Human")

    after_section = ""
    if after_text and after_prob is not None:
        delta = after_prob - doc_prob
        delta_color = "#27ae60" if delta < 0 else "#e74c3c"
        delta_str = f"{delta:+.1%}"
        after_sents = split_sentences(after_text)
        after_section = f"""
<section class="card">
  <h2>✅ After Polishing</h2>
  <div class="prob-banner" style="background:{('#27ae60' if after_prob<0.5 else '#e74c3c')}22;border-color:{('#27ae60' if after_prob<0.5 else '#e74c3c')}">
    <span class="prob-value" style="color:{('#27ae60' if after_prob<0.5 else '#e74c3c')}">{after_prob:.1%}</span>
    <span class="prob-change" style="color:{delta_color}">({delta_str})</span>
    <span class="prob-label">{'Likely Human ✓' if after_prob < 0.5 else 'Still AI-like'}</span>
  </div>
  <div class="sent-container">{_html.escape(after_text)}</div>
</section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{_html.escape(title)}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,sans-serif;background:#f7f8fa;color:#1a1a2e;line-height:1.6}}
header{{background:#1a1a2e;color:#fff;padding:20px 40px}}
header h1{{font-size:1.3rem;font-weight:600}}
header p{{opacity:.7;font-size:.85rem;margin-top:4px}}
.container{{max-width:1000px;margin:0 auto;padding:28px 20px;display:flex;flex-direction:column;gap:24px}}
.card{{background:#fff;border-radius:10px;padding:24px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
h2{{font-size:1rem;font-weight:600;margin-bottom:14px;color:#1a1a2e;border-left:4px solid #3498db;padding-left:10px}}
.prob-banner{{display:flex;align-items:center;gap:14px;padding:14px 20px;border-radius:8px;border:1px solid;margin-bottom:16px}}
.prob-value{{font-size:2.2rem;font-weight:700}}
.prob-label{{font-size:1rem;font-weight:500}}
.prob-change{{font-size:1.1rem;font-weight:600}}
.sent-container{{line-height:2.1;font-size:.95rem}}
.sent-chip{{display:inline;padding:2px 4px;border-radius:4px;border:1px solid;margin:1px;cursor:default;transition:opacity .15s}}
.sent-chip:hover{{opacity:.8;transform:scale(1.01)}}
table{{width:100%;border-collapse:collapse;font-size:.87rem}}
th{{background:#1a1a2e;color:#fff;padding:8px 12px;text-align:left;font-weight:500}}
td{{padding:7px 12px;border-bottom:1px solid #eee;vertical-align:middle}}
.ai-sig td:first-child{{color:#e74c3c;font-weight:600}}
.human-sig td:first-child{{color:#27ae60;font-weight:600}}
.bar{{height:8px;border-radius:4px;min-width:2px}}
.suggestion-card{{padding:14px 16px;margin-bottom:12px;border-radius:6px;background:#fafafa;border:1px solid #eee}}
.tier-badge{{display:inline-block;padding:2px 8px;border-radius:10px;color:#fff;font-size:.75rem;font-weight:600;margin-bottom:6px}}
.suggestion-card p{{font-size:.9rem;color:#333;margin-top:6px}}
.prompt-box{{background:#1a1a2e;color:#a8d8a8;padding:16px;border-radius:8px;font-family:monospace;font-size:.82rem;white-space:pre-wrap;word-break:break-word}}
.copy-btn{{margin-top:10px;padding:8px 16px;background:#3498db;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:.85rem}}
.copy-btn:hover{{background:#2980b9}}
.legend{{display:flex;gap:12px;flex-wrap:wrap;font-size:.8rem;margin-bottom:12px}}
.legend-item{{display:flex;align-items:center;gap:5px}}
.legend-dot{{width:14px;height:14px;border-radius:3px;border:1px solid #ccc}}
</style>
</head>
<body>
<header>
  <h1>🔍 AI Text Detector — Integrated Analysis</h1>
  <p>Sentence-level detection · Feature analysis · Polishing suggestions</p>
</header>
<div class="container">

<section class="card">
  <h2>Document-Level Verdict</h2>
  <div class="prob-banner" style="background:{prob_color}22;border-color:{prob_color}">
    <span class="prob-value" style="color:{prob_color}">{doc_prob:.1%}</span>
    <span class="prob-label">{prob_label}</span>
  </div>
</section>

<section class="card">
  <h2>Sentence-Level AI Probability (Query-Style Highlight)</h2>
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#f0f4ff;border-color:#ccc"></div><span>&lt;30% Human-like</span></div>
    <div class="legend-item"><div class="legend-dot" style="background:#fde8b0;border-color:#f0a500"></div><span>30–70% Mixed</span></div>
    <div class="legend-item"><div class="legend-dot" style="background:#c0392b;border-color:#c0392b"></div><span>&gt;85% AI-like</span></div>
  </div>
  <div class="sent-container">{sent_html}</div>
</section>

<section class="card">
  <h2>Linguistic Feature Analysis</h2>
  <table>
    <tr><th>Feature</th><th>This text</th><th>Human avg</th><th>AI avg</th><th>Signal strength</th><th>Direction</th></tr>
    {feature_rows}
  </table>
</section>

<section class="card">
  <h2>Polishing Suggestions — How to Make It Sound More Human</h2>
  {suggestion_html}
</section>

<section class="card">
  <h2>Composite Rewriting Prompt (paste into ChatGPT)</h2>
  <div class="prompt-box" id="prompt-text">{composite_prompt}</div>
  <button class="copy-btn" onclick="navigator.clipboard.writeText(document.getElementById('prompt-text').innerText).then(()=>this.textContent='Copied ✓')">Copy Prompt</button>
</section>

{after_section}

</div>
</body>
</html>"""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--input_file", type=Path, default=None)
    p.add_argument("--after_text", default=None)
    p.add_argument("--after_file", type=Path, default=None)
    p.add_argument("--fusion_model_dir", type=Path, default=PROJ_ROOT/"outputs/fusion_lgbm")
    p.add_argument("--tfidf_model_file", type=Path, default=PROJ_ROOT/"outputs/tfidf/model.joblib")
    p.add_argument("--roberta_model_dir", type=Path, default=PROJ_ROOT/"outputs/roberta_base/best_model")
    p.add_argument("--output_dir", type=Path, default=PROJ_ROOT/"outputs/integrated_demo")
    p.add_argument("--title", default="AI Text Analysis")
    args = p.parse_args()

    if args.text:
        text = args.text
    elif args.input_file:
        text = args.input_file.read_text(encoding="utf-8")
    else:
        raise SystemExit("Provide --text or --input_file")

    after_text = args.after_text
    if args.after_file and args.after_file.exists():
        after_text = args.after_file.read_text(encoding="utf-8")

    baselines = json.loads((PROJ_ROOT/"data/feature_baselines.json").read_text())

    print("Scoring sentences...", flush=True)
    sentences = split_sentences(text)
    sent_df = score_sentences_fusion(
        sentences=sentences,
        fusion_model_dir=args.fusion_model_dir,
        tfidf_model_file=args.tfidf_model_file,
        roberta_model_dir=args.roberta_model_dir,
    )
    probs = sent_df["prob_ai"].tolist()
    doc_prob = float(sent_df["prob_ai"].mean())

    print("Computing features...", flush=True)
    feats = full_features(text)
    suggestions = generate_suggestions(feats, baselines, text, doc_prob)

    after_prob = None
    if after_text:
        print("Scoring after-text...", flush=True)
        after_sents = split_sentences(after_text)
        after_df = score_sentences_fusion(
            sentences=after_sents,
            fusion_model_dir=args.fusion_model_dir,
            tfidf_model_file=args.tfidf_model_file,
            roberta_model_dir=args.roberta_model_dir,
        )
        after_prob = float(after_df["prob_ai"].mean())

    html = build_html(text, sentences, probs, doc_prob,
                      feats, baselines, suggestions,
                      after_text=after_text, after_prob=after_prob,
                      title=args.title)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "analysis.html"
    out.write_text(html, encoding="utf-8")
    print(f"Written: {out}")
    print(f"Document AI probability: {doc_prob:.1%}")
    print(f"Sentences: {len(sentences)}")
    if after_prob is not None:
        print(f"After polishing: {after_prob:.1%} (change: {after_prob-doc_prob:+.1%})")

if __name__ == "__main__":
    main()
