"""Generate comprehensive analysis HTML report with inline charts."""
from __future__ import annotations
import json
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("/inspire/hdd/project/fdu-aidake-cfff/public/hanz/semeval2024_task8a_en_baseline")
OUT = BASE / "outputs" / "analysis_report"

preds = {
    "TF-IDF": pd.read_csv(BASE / "outputs/tfidf_test/predictions.csv"),
    "RoBERTa-base": pd.read_csv(BASE / "outputs/roberta_base_test/predictions.csv"),
    "RoBERTa-chunked": pd.read_csv(BASE / "outputs/roberta_base_chunked_test/predictions.csv"),
    "Fusion-LightGBM": pd.read_csv(BASE / "outputs/fusion_lgbm_test/predictions.csv"),
}

wc_map = {}
with open(BASE / "data/official/test_sets/subtaskA_monolingual.jsonl") as f:
    for line in f:
        row = json.loads(line)
        wc_map[row["id"]] = len(row["text"].split())

for df in preds.values():
    df["wc"] = df["id"].map(wc_map)
    df["len_bucket"] = df["wc"].apply(lambda w: "0-200" if w<=200 else "201-500" if w<=500 else "501-800" if w<=800 else "800+")

bucket_order = ["0-200", "201-500", "501-800", "800+"]
rows = []
for name, df in preds.items():
    g = df.groupby("len_bucket")["correct"].agg(["mean","count"]).reindex(bucket_order)
    for bucket, row in g.iterrows():
        rows.append({"model":name,"len_bucket":bucket,"accuracy":round(row["mean"],4),"n":int(row["count"])})
length_df = pd.DataFrame(rows)

# Calibration
n_bins = 10
bins = np.linspace(0,1,n_bins+1)
cal_rows = []
for name, df in preds.items():
    df2 = df.copy()
    df2["bin"] = pd.cut(df2["prob_ai"], bins=bins, labels=False, include_lowest=True)
    for b, g in df2.groupby("bin"):
        center = (bins[b]+bins[b+1])/2
        cal_rows.append({"model":name,"bin_center":round(float(center),2),"frac_positive":round(float(g["label"].mean()),4),"count":len(g)})
cal_df = pd.DataFrame(cal_rows)

# Overall metrics
metrics_files = {
    "TF-IDF": "outputs/tfidf_test/metrics.json",
    "RoBERTa-base": "outputs/roberta_base_test/metrics.json",
    "RoBERTa-chunked": "outputs/roberta_base_chunked_test/metrics.json",
    "Fusion-LR": "outputs/fusion_lr_test/metrics.json",
    "Fusion-LightGBM": "outputs/fusion_lgbm_test/metrics.json",
}
overall = []
for name, path in metrics_files.items():
    m = json.loads((BASE / path).read_text())
    overall.append({"model": name, "accuracy": m.get("accuracy"), "macro_f1": m.get("macro_f1"),
                    "auroc": m.get("auroc"), "ece": m.get("ece")})
overall_df = pd.DataFrame(overall)

# Source accuracy
src_rows = []
for name, df in {"RoBERTa-base":preds["RoBERTa-base"],"Fusion-LightGBM":preds["Fusion-LightGBM"]}.items():
    for src, g in df.groupby("model"):
        src_rows.append({"model_name":name,"ai_source":src,"accuracy":round(g["correct"].mean(),4),"n":len(g)})
src_df = pd.DataFrame(src_rows)

# JSON for charts
len_chart = {}
for _, row in length_df.iterrows():
    len_chart.setdefault(row["model"],[]).append({"x":row["len_bucket"],"y":row["accuracy"]})

cal_chart = {}
for _, row in cal_df.iterrows():
    cal_chart.setdefault(row["model"],[]).append({"x":row["bin_center"],"y":row["frac_positive"]})

src_chart = {}
for _, row in src_df.iterrows():
    src_chart.setdefault(row["model_name"],[]).append({"x":row["ai_source"],"y":row["accuracy"]})

COLORS = {"TF-IDF":"#e07b39","RoBERTa-base":"#5b8dd9","RoBERTa-chunked":"#9b59b6","Fusion-LightGBM":"#27ae60","Fusion-LR":"#f39c12"}

# ── HTML ──────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SemEval-2024 Task 8A — Analysis Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:system-ui,sans-serif;background:#f7f8fa;color:#222;line-height:1.5;}}
header{{background:#1a1a2e;color:#fff;padding:24px 40px;}}
header h1{{font-size:1.4rem;font-weight:600;}}
header p{{opacity:.7;font-size:.9rem;margin-top:4px;}}
.container{{max-width:1100px;margin:0 auto;padding:32px 24px;}}
h2{{font-size:1.1rem;font-weight:600;color:#1a1a2e;margin:36px 0 12px;border-left:4px solid #27ae60;padding-left:12px;}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px;}}
.card{{background:#fff;border-radius:8px;padding:24px;box-shadow:0 1px 4px rgba(0,0,0,.08);}}
.card.full{{grid-column:1/-1;}}
canvas{{max-height:300px;}}
table{{width:100%;border-collapse:collapse;font-size:.88rem;}}
th{{background:#1a1a2e;color:#fff;padding:8px 12px;text-align:left;font-weight:500;}}
td{{padding:7px 12px;border-bottom:1px solid #eee;}}
tr:hover td{{background:#f0f4ff;}}
.best{{font-weight:700;color:#27ae60;}}
.note{{font-size:.82rem;color:#666;margin-top:8px;font-style:italic;}}
</style>
</head>
<body>
<header>
  <h1>SemEval-2024 Task 8A — Model Analysis Report</h1>
  <p>Binary AI-text detection · outfox test set (n=34,272) · June 2026</p>
</header>
<div class="container">

<h2>Overall Metrics</h2>
<div class="card full">
<table>
<tr><th>Model</th><th>Test Accuracy</th><th>Test Macro-F1</th><th>AUROC</th><th>ECE ↓</th></tr>
"""

best_f1 = overall_df["macro_f1"].max()
best_auroc = overall_df["auroc"].max()
best_ece = overall_df["ece"].min()
for _, row in overall_df.iterrows():
    def c(val, best): return ' class="best"' if abs(val-best)<1e-9 else ''
    html += f"<tr><td>{row['model']}</td>"
    html += f"<td>{row['accuracy']:.4f}</td>"
    html += f"<td{c(row['macro_f1'],best_f1)}>{row['macro_f1']:.4f}</td>"
    html += f"<td{c(row['auroc'],best_auroc)}>{row['auroc']:.4f}</td>"
    html += f"<td{c(row['ece'],best_ece)}>{row['ece']:.4f}</td></tr>\n"

html += """</table>
<p class="note">Bold = best in column. Fusion-LightGBM leads Macro-F1; RoBERTa-chunked leads AUROC; TF-IDF leads ECE (best calibrated).</p>
</div>

<h2>Analysis 1: Accuracy by Text Length</h2>
<div class="grid">
<div class="card full"><canvas id="lenChart"></canvas>
<p class="note">RoBERTa degrades sharply on long texts (truncation at 512 tokens). TF-IDF excels on long texts (bag-of-words is length-invariant). Fusion-LightGBM matches TF-IDF on long texts while combining RoBERTa signal for medium texts. Note: Fusion underperforms on very short texts (0-200w) — dominated by bloomz short samples, a known distribution shift.</p>
</div>
</div>

<h2>Analysis 2: Reliability Diagram (Calibration)</h2>
<div class="grid">
<div class="card full"><canvas id="calChart"></canvas>
<p class="note">A perfectly calibrated model follows the diagonal. RoBERTa-base shows extreme overconfidence (high prob_ai even for human texts → ECE=0.324). TF-IDF is the best calibrated (ECE=0.030). Fusion-LightGBM (ECE=0.147) is substantially better calibrated than RoBERTa.</p>
</div>
</div>

<h2>Analysis 3: Per-AI-Source Accuracy</h2>
<div class="grid">
<div class="card full"><canvas id="srcChart"></canvas>
<p class="note">RoBERTa-base achieves near-perfect detection on all AI sources but catastrophically misclassifies human texts (acc=0.315) — it is effectively over-biased toward predicting AI. Fusion-LightGBM dramatically improves human accuracy (0.767) at a cost on bloomz (0.410), which uses short, simple sentences that mimic human style.</p>
</div>
</div>

</div>
<script>
const COLORS = """ + json.dumps(COLORS) + """;

// Chart 1: Length bucket
const lenData = """ + json.dumps(len_chart) + """;
new Chart(document.getElementById('lenChart'), {
  type: 'bar',
  data: {
    labels: ['0-200','201-500','501-800','800+'],
    datasets: Object.entries(lenData).map(([name, pts]) => ({
      label: name,
      data: pts.map(p=>p.y),
      backgroundColor: COLORS[name]+'99',
      borderColor: COLORS[name],
      borderWidth:1.5,
    }))
  },
  options: {
    responsive:true, plugins:{title:{display:true,text:'Accuracy by Text Length Bucket'}},
    scales:{y:{min:0,max:1,title:{display:true,text:'Accuracy'}},x:{title:{display:true,text:'Word Count Bucket'}}}
  }
});

// Chart 2: Reliability
const calData = """ + json.dumps(cal_chart) + """;
const calCtx = document.getElementById('calChart');
const calDatasets = Object.entries(calData).map(([name, pts]) => ({
  label: name, type:'line',
  data: pts.map(p=>({x:p.x,y:p.y})),
  borderColor: COLORS[name], backgroundColor:'transparent',
  pointRadius:4, borderWidth:2, tension:0.2,
}));
calDatasets.push({
  label:'Perfect', type:'line',
  data:[{x:0.05,y:0.05},{x:0.95,y:0.95}],
  borderColor:'#aaa',borderDash:[6,4],borderWidth:1.5,pointRadius:0,backgroundColor:'transparent'
});
new Chart(calCtx, {
  type:'scatter',
  data:{datasets:calDatasets},
  options:{
    responsive:true,
    plugins:{title:{display:true,text:'Reliability Diagram (Calibration)'}},
    scales:{
      x:{min:0,max:1,title:{display:true,text:'Mean Predicted Probability (prob_ai)'}},
      y:{min:0,max:1,title:{display:true,text:'Fraction of Positives (Actual AI ratio)'}}
    }
  }
});

// Chart 3: Source accuracy
const srcData = """ + json.dumps(src_chart) + """;
const srcLabels = ['bloomz','chatGPT','cohere','davinci','dolly','GPT4','human'];
new Chart(document.getElementById('srcChart'), {
  type:'bar',
  data:{
    labels:srcLabels,
    datasets: Object.entries(srcData).map(([name,pts])=>({
      label:name,
      data: srcLabels.map(s=>{ const p=pts.find(x=>x.x===s); return p?p.y:null;}),
      backgroundColor: COLORS[name]+'99', borderColor:COLORS[name], borderWidth:1.5,
    }))
  },
  options:{
    responsive:true,plugins:{title:{display:true,text:'Accuracy by AI Source (RoBERTa-base vs Fusion-LightGBM)'}},
    scales:{y:{min:0,max:1,title:{display:true,text:'Accuracy'}},x:{title:{display:true,text:'AI Source / human'}}}
  }
});
</script>
</body>
</html>
"""

(OUT / "analysis_report.html").write_text(html)
print("Written:", OUT / "analysis_report.html")
