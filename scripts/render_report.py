"""
scripts/render_report.py

Render a RegionAwareDetector.analyze() result into a single self-contained
PaperPass-style HTML report. No external assets, no JS dependencies.

Sections:
  1. Header: title + overall AI suspicion gauge + verdict
  2. Dashboard: SVG ring gauge, grade distribution bars, ppl/burstiness, info
  3. Highlighted text: paragraphs/sentences colored by grade (red/orange/purple/green)
  4. Feature evidence: per-feature deviation bars vs human/AI baseline
  5. De-AI suggestions: tiered polish advice + composite rewrite prompt
"""
from __future__ import annotations

import html as _html
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# PaperPass grade palette
GRADE = {
    "high":   ("#da3633", "高度疑似", "≥70%"),
    "middle": ("#e8870c", "中度疑似", "60–70%"),
    "low":    ("#8957e5", "轻度疑似", "50–60%"),
    "ok":     ("#2ea043", "合格",     "<50%"),
}


def _grade_color(grade: str) -> str:
    return GRADE.get(grade, GRADE["ok"])[0]


def _esc(s) -> str:
    return _html.escape(str(s))


def _ring_gauge(pct: float, color: str, size: int = 150, label: str = "P(AI)") -> str:
    """SVG donut gauge showing pct (0-100)."""
    r = size / 2 - 12
    c = 2 * 3.14159265 * r
    filled = c * pct / 100
    cx = cy = size / 2
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#eceff3" stroke-width="14"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="14"'
        f' stroke-dasharray="{filled:.2f} {c:.2f}" stroke-linecap="round"'
        f' transform="rotate(-90 {cx} {cy})"/>'
        f'<text x="50%" y="48%" text-anchor="middle" font-size="30" font-weight="700"'
        f' fill="{color}">{pct:.1f}%</text>'
        f'<text x="50%" y="64%" text-anchor="middle" font-size="12" fill="#8a94a6">{label}</text>'
        f'</svg>'
    )


def _heat_color(heat: float) -> str:
    """heat in [0,1]: 0=human-pushing (green), 0.5=neutral, 1=AI-pushing (red)."""
    h = max(0.0, min(1.0, heat))
    if h >= 0.5:
        t = (h - 0.5) / 0.5
        return f"rgba(218,54,51,{0.12 + 0.55*t:.2f})"   # red
    t = (0.5 - h) / 0.5
    return f"rgba(46,160,67,{0.12 + 0.45*t:.2f})"        # green


def _highlighted_text(paragraphs: list[dict]) -> str:
    """Render paragraphs; sentences shaded by OCCLUSION contribution to the
    document AI score (red = removing it lowers AI prob = pushes AI;
    green = pushes human). This is coherent with the document score."""
    blocks = []
    for p in paragraphs:
        sents = p.get("sentences", [])
        if sents:
            spans = []
            for s in sents:
                heat = s.get("heat", 0.5)
                contrib = s.get("contrib", 0.0)
                bg = _heat_color(heat)
                spans.append(
                    f'<span class="sent" style="background:{bg}" '
                    f'title="对文档AI概率的贡献 {contrib:+.3f}（{s.get("direction","neutral")}）">'
                    f'{_esc(s["text"])}</span> '
                )
            inner = "".join(spans)
        else:
            inner = _esc(p["text"])
        p_prob = p.get("prob_ai", 0) * 100
        p_color = _grade_color(p["grade"])
        blocks.append(
            f'<div class="para">'
            f'<div class="para-tag" style="background:{p_color}">'
            f'#{p["index"]+1} · 段落P(AI) {p_prob:.0f}% · '
            f'ppl {p.get("ppl",0):.0f} · burst {p.get("burstiness",0):.2f}</div>'
            f'<div class="para-body">{inner}</div>'
            f'</div>'
        )
    return "\n".join(blocks)


def _evidence_bars(evidence: list[dict], top_n: int = 12) -> str:
    """Discriminant axis per feature: left = certainly human, center = decision
    boundary, right = certainly AI. The current value's marker sits at its
    Gaussian-LLR discriminant score disc in [-1,+1] mapped to [0%,100%].
    Small ticks show where the human-mean and AI-mean of the feature land."""
    rows = []
    for e in evidence[:top_n]:
        disc = max(-1.0, min(1.0, e.get("disc", 0.0)))
        pos = (disc + 1) / 2 * 100                      # -1->0%, +1->100%
        hpos = (max(-1.0, min(1.0, e.get("disc_human", -1))) + 1) / 2 * 100
        apos = (max(-1.0, min(1.0, e.get("disc_ai", 1))) + 1) / 2 * 100
        sig = e["signal"]
        mcolor = "#da3633" if disc > 0.05 else ("#2ea043" if disc < -0.05 else "#5b6573")
        lean = f"{abs(disc)*100:.0f}% 偏{'AI' if disc>0 else '人类'}" if abs(disc) > 0.05 else "中性"
        rows.append(
            f'<tr>'
            f'<td class="fname">{_esc(e["feature"])}</td>'
            f'<td class="fbar"><div class="dax">'
            f'<div class="dax-mid"></div>'
            f'<span class="tick h" style="left:{hpos:.1f}%" title="人类均值"></span>'
            f'<span class="tick a" style="left:{apos:.1f}%" title="AI均值"></span>'
            f'<span class="dot" style="left:{pos:.1f}%;background:{mcolor}" '
            f'title="判别分 {disc:+.2f}（{lean}）｜本文值 {e["value"]:.3f}"></span>'
            f'</div></td>'
            f'<td class="fval" style="color:{mcolor}">{lean}</td>'
            f'</tr>'
        )
    body = "\n".join(rows)
    return (
        '<table class="ev"><thead><tr>'
        '<th>特征</th>'
        '<th><div class="dax-head"><span>← 确定是人类</span>'
        '<span class="mid">决策边界</span><span>确定是AI →</span></div></th>'
        '<th>倾向</th></tr></thead><tbody>'
        + body + '</tbody></table>'
    )


def _suggestions_panel(sugg: dict) -> str:
    if not sugg:
        return "<p>未生成降AI建议。</p>"
    tier_labels = [
        ("tier3_discourse", "篇章 / 衔接"),
        ("tier4_pragmatic", "语用 / 人称语气"),
        ("tier2_syntactic", "句法 / 句长节奏"),
        ("tier1_lexical", "词汇 / 多样性"),
    ]
    cards = []
    for key, label in tier_labels:
        items = sugg.get(key, [])
        for it in items:
            cards.append(
                f'<div class="sug-card">'
                f'<div class="sug-head">{_esc(label)} '
                f'<span class="sug-dev">偏离 {it.get("deviation","?")}</span></div>'
                f'<div class="sug-body">{_esc(it.get("suggestion",""))}</div>'
                f'</div>'
            )
    cards_html = "\n".join(cards) if cards else "<p>文本已较接近人类写作模式。</p>"
    composite = _esc(sugg.get("composite_prompt", ""))
    return (
        f'<div class="sug-grid">{cards_html}</div>'
        f'<h3>一键降AI Prompt（粘贴到改写模型）</h3>'
        f'<pre class="composite">{composite}</pre>'
    )


_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, 'Segoe UI', 'PingFang SC', sans-serif;
       margin: 0; background: #f4f6f9; color: #1f2733; }
.wrap { max-width: 980px; margin: 0 auto; padding: 24px; }
.topbar { display:flex; justify-content:space-between; align-items:center;
          background:#fff; border-radius:12px; padding:18px 24px; margin-bottom:18px;
          box-shadow:0 1px 4px rgba(0,0,0,.06); }
.topbar h1 { font-size:18px; margin:0; }
.topbar .verdict { font-size:14px; font-weight:700; padding:6px 14px; border-radius:20px;
                   color:#fff; }
.dash { display:grid; grid-template-columns: 170px 1fr 1fr; gap:20px;
        background:#fff; border-radius:12px; padding:22px; margin-bottom:18px;
        box-shadow:0 1px 4px rgba(0,0,0,.06); align-items:center; }
.dash .gauge { text-align:center; }
.bars { }
.bars .row { display:flex; align-items:center; margin:6px 0; font-size:13px; }
.bars .lbl { width:110px; color:#5b6573; }
.bars .track { flex:1; height:12px; background:#eceff3; border-radius:6px; overflow:hidden; margin:0 8px; }
.bars .fill { height:100%; border-radius:6px; }
.bars .pc { width:48px; text-align:right; font-weight:600; }
.signals { font-size:13px; color:#5b6573; }
.signals .sig-item { display:flex; justify-content:space-between; padding:6px 0;
                     border-bottom:1px solid #eef1f5; }
.signals .sig-item b { color:#1f2733; }
.card { background:#fff; border-radius:12px; padding:22px; margin-bottom:18px;
        box-shadow:0 1px 4px rgba(0,0,0,.06); }
.card h2 { font-size:16px; margin:0 0 14px; color:#27313f; }
.legend { font-size:12px; color:#6b7480; margin-bottom:12px; }
.legend span { display:inline-block; margin-right:14px; }
.legend i { display:inline-block; width:22px; height:4px; vertical-align:middle;
            margin-right:5px; border-radius:2px; }
.para { margin:14px 0; }
.para-tag { display:inline-block; color:#fff; font-size:11px; font-weight:600;
            padding:2px 10px; border-radius:10px; margin-bottom:6px; }
.para-body { line-height:2.0; font-size:15px; }
.sent { padding:1px 2px; border-radius:3px; }
table.ev { width:100%; border-collapse:collapse; font-size:13px; }
table.ev th { font-weight:600; color:#6b7480; font-size:12px; padding:4px 8px; text-align:left; }
table.ev td { padding:10px 8px; border-bottom:1px solid #eef1f5; vertical-align:middle; }
.ev .fname { width:190px; color:#3a4452; font-family:monospace; font-size:12px; }
.ev .fbar { width:auto; min-width:260px; }
.dax-head { display:flex; justify-content:space-between; font-size:11px; color:#9aa3b0; }
.dax-head .mid { color:#b6c0cf; }
.dax { position:relative; height:24px; border-radius:12px;
       background:linear-gradient(90deg,#2ea043 0%, #d6e8d8 38%, #f1f0f2 50%, #f3dada 62%, #da3633 100%);
       border:1px solid #e2e6ec; }
.dax-mid { position:absolute; left:50%; top:0; bottom:0; width:2px;
           background:rgba(31,39,51,.35); transform:translateX(-1px); }
.dax .tick { position:absolute; top:50%; width:2px; height:14px; transform:translate(-1px,-50%);
             opacity:.65; }
.dax .tick.h { background:#0b5d20; }
.dax .tick.a { background:#7a1210; }
.dax .dot { position:absolute; top:50%; width:16px; height:16px; border-radius:50%;
            border:3px solid #fff; box-shadow:0 0 0 1px rgba(0,0,0,.25);
            transform:translate(-8px,-50%); z-index:3; }
.ev .fval { width:90px; text-align:right; font-weight:600; font-size:12px; }
.sug-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.sug-card { border:1px solid #e7ebf0; border-radius:8px; padding:12px 14px; }
.sug-head { font-weight:700; font-size:13px; color:#27313f; margin-bottom:6px; }
.sug-dev { float:right; font-weight:500; font-size:11px; color:#da3633; }
.sug-body { font-size:13px; color:#4a5562; line-height:1.6; }
.composite { background:#0f1620; color:#cfe3ff; padding:16px; border-radius:8px;
             font-size:12px; white-space:pre-wrap; line-height:1.55; overflow-x:auto; }
.foot { text-align:center; color:#9aa3b0; font-size:12px; padding:16px; }
"""


def render(result: dict, output_file, title: str = None) -> Path:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    title = title or result.get("title", "AI 文本检测报告")

    susp = float(result.get("doc_prob_ai", 0)) * 100   # headline = real calibrated P(AI)
    grade = result.get("doc_grade", "ok")
    color, gname, _ = GRADE[grade]
    region = result.get("doc_region", "")
    region_zh = {"clean_ai": "高置信AI区", "clean_human": "高置信人类区",
                 "ambiguous": "模糊带（正交特征裁决）"}.get(region, region)
    verdict = (f"最终判定：AI 生成" if result.get("doc_label", 0) == 1
               else "最终判定：人类写作") + f" · {region_zh}"

    # grade distribution bars
    gd = result.get("grade_distribution", {})
    bar_rows = []
    for g in ["high", "middle", "low", "ok"]:
        gc, gl, _ = GRADE[g]
        v = gd.get(g, 0)
        bar_rows.append(
            f'<div class="row"><span class="lbl">{gl}</span>'
            f'<div class="track"><div class="fill" style="width:{v}%;background:{gc}"></div></div>'
            f'<span class="pc">{v:.0f}%</span></div>'
        )
    bars_html = "\n".join(bar_rows)

    # signal panel
    sig_html = "".join(
        f'<div class="sig-item"><span>{k}</span><b>{v}</b></div>'
        for k, v in [
            ("困惑度 (perplexity)", f'{result.get("ppl",0):.1f}'),
            ("突发性 (burstiness)", f'{result.get("burstiness",0):.3f}'),
            ("RoBERTa 原始 P(AI)", f'{result.get("doc_prob_ai",0)*100:.1f}%'),
            ("判定区域", result.get("doc_region", "-")),
            ("字数", f'{result.get("word_count",0)} 词 / {result.get("char_count",0)} 字符'),
            ("段落数", result.get("paragraph_count", 0)),
        ]
    )

    legend = (
        '<div class="legend">'
        '<span><i style="background:rgba(218,54,51,0.6)"></i>该句推高AI概率</span>'
        '<span><i style="background:rgba(46,160,67,0.6)"></i>该句推向人类</span>'
        '<span style="color:#9aa3b0">底色＝遮挡法：删去该句后文档AI概率的变化量，悬停看数值</span></div>'
    )

    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)} · AI检测报告</title><style>{_CSS}</style></head><body><div class="wrap">
<div class="topbar"><h1>{_esc(title)}</h1>
<span class="verdict" style="background:{color}">{verdict} · {gname}</span></div>

<div class="dash">
  <div class="gauge">{_ring_gauge(susp, color)}</div>
  <div class="bars"><div style="font-size:13px;color:#5b6573;margin-bottom:6px">疑似度分布（按段落）</div>{bars_html}</div>
  <div class="signals">{sig_html}</div>
</div>

<div class="card"><h2>正文逐句高亮</h2>{legend}{_highlighted_text(result.get("paragraphs", []))}</div>

<div class="card"><h2>AI 特征检测（语言学证据）</h2>
<div class="legend"><span style="color:#9aa3b0">每个特征按高斯似然比映射到判别轴：圆点越靠左＝该特征越确定指向人类，越靠右＝越确定指向AI，正中为决策边界。竖线标出人类均值（深绿）与AI均值（深红）位置。最具区分度的特征排在前。</span></div>
{_evidence_bars(result.get("feature_evidence", []))}</div>

<div class="card"><h2>辅助降 AI 率</h2>{_suggestions_panel(result.get("suggestions"))}</div>

<div class="foot">Region-Aware AI-Text Detector · SemEval-2024 Task 8 · 校准+正交特征两阶段判定<br>
本报告仅表示文本由 AI 生成的统计可能性，与内容质量无关。</div>
</div></body></html>"""
    output_file.write_text(html, encoding="utf-8")
    return output_file


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True, help="analyze() result JSON")
    ap.add_argument("--out", type=Path, default=REPO / "outputs/pipeline_demo/report.html")
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()
    result = json.loads(Path(args.result).read_text())
    out = render(result, args.out, args.title)
    print(f"Wrote {out}")


def render_reduce(bundle: dict, output_file, title: str = "降 AIGC 前后对比报告") -> Path:
    """PaperPass reduce_aigc-style before/after report from a humanize bundle."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    b, a = bundle["before"], bundle["after"]
    attacker = bundle.get("attacker", "?")

    def pct(r, key="doc_prob_ai"):
        return float(r.get(key, 0)) * 100

    bc = _grade_color(b["doc_grade"]); ac = _grade_color(a["doc_grade"])
    b_susp, a_susp = pct(b), pct(a)
    b_prob, a_prob = b_susp, a_susp  # headline now IS the calibrated prob

    head = (
        f'<div class="rd-head">'
        f'<div class="rd-col"><div class="rd-lbl">降AIGC前</div>'
        f'<div class="rd-ring">{_ring_gauge(b_susp, bc, 130)}</div>'
        f'<div class="rd-meta">最终判定 {("AI" if b["doc_label"]==1 else "人类")} · margin {b["doc_margin"]:.1f} · '
        f'ppl {b.get("ppl",0):.0f} · burst {b.get("burstiness",0):.2f}</div></div>'
        f'<div class="rd-arrow">→<br><span>{attacker}<br>递归改写</span></div>'
        f'<div class="rd-col"><div class="rd-lbl">降AIGC后</div>'
        f'<div class="rd-ring">{_ring_gauge(a_susp, ac, 130)}</div>'
        f'<div class="rd-meta">最终判定 {("AI" if a["doc_label"]==1 else "人类")} · margin {a["doc_margin"]:.1f} · '
        f'ppl {a.get("ppl",0):.0f} · burst {a.get("burstiness",0):.2f}</div></div>'
        f'</div>'
        f'<div class="rd-delta">RoBERTa 校准 P(AI) {b_susp:.1f}% → {a_susp:.1f}%（'
        f'{"↓" if a_susp<b_susp else "→"} {abs(b_susp-a_susp):.1f} 个百分点）· '
        f'margin {b["doc_margin"]:.1f} → {a["doc_margin"]:.1f} · '
        f'最终判定 {("AI" if b["doc_label"]==1 else "人类")} → {("AI" if a["doc_label"]==1 else "人类")}</div>'
    )

    side = (
        f'<div class="rd-side">'
        f'<div class="rd-pane"><h3>原文</h3><div class="rd-text">{_esc(bundle["original_text"])}</div></div>'
        f'<div class="rd-pane"><h3>降AIGC后</h3><div class="rd-text">{_esc(bundle["rewritten_text"])}</div></div>'
        f'</div>'
    )

    extra_css = """
.rd-head { display:flex; align-items:center; justify-content:center; gap:30px; }
.rd-col { text-align:center; }
.rd-lbl { font-weight:700; color:#5b6573; margin-bottom:6px; }
.rd-meta { font-size:12px; color:#8a94a6; margin-top:6px; }
.rd-arrow { font-size:26px; color:#b6c0cf; text-align:center; }
.rd-arrow span { font-size:11px; color:#8a94a6; }
.rd-delta { text-align:center; font-size:14px; font-weight:600; color:#27313f;
            background:#f0f4fa; border-radius:8px; padding:10px; margin-top:16px; }
.rd-side { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:8px; }
.rd-pane h3 { font-size:14px; color:#344054; margin:0 0 8px; }
.rd-text { font-size:13px; line-height:1.85; white-space:pre-wrap; background:#fafbfc;
           border:1px solid #eef1f5; border-radius:8px; padding:14px; max-height:560px; overflow:auto; }
"""
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title><style>{_CSS}{extra_css}</style></head><body><div class="wrap">
<div class="topbar"><h1>{_esc(title)}</h1>
<span class="verdict" style="background:{ac}">改写后判定：{"AI" if a["doc_label"]==1 else "人类"}</span></div>
<div class="card">{head}</div>
<div class="card"><h2>原文 / 降AIGC后 全文对比</h2>{side}</div>
<div class="foot">攻防评测：改写攻击使 AI 疑似度下降，量化检测器鲁棒性。仅研究用途。</div>
</div></body></html>"""
    output_file.write_text(html, encoding="utf-8")
    return output_file


if __name__ == "__main__":
    main()
