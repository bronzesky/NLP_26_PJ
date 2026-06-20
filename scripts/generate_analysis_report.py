"""Generate comprehensive analysis HTML report and CSV tables."""
from __future__ import annotations
import json
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("/inspire/hdd/project/fdu-aidake-cfff/public/hanz/semeval2024_task8a_en_baseline")
OUT = BASE / "outputs" / "analysis_report"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load predictions ──────────────────────────────────────────────────────────
preds = {
    "TF-IDF": pd.read_csv(BASE / "outputs/tfidf_test/predictions.csv"),
    "RoBERTa-base": pd.read_csv(BASE / "outputs/roberta_base_test/predictions.csv"),
    "RoBERTa-chunked": pd.read_csv(BASE / "outputs/roberta_base_chunked_test/predictions.csv"),
    "Fusion-LightGBM": pd.read_csv(BASE / "outputs/fusion_lgbm_test/predictions.csv"),
}

# ── Add text length ───────────────────────────────────────────────────────────
wc_map = {}
with open(BASE / "data/official/test_sets/subtaskA_monolingual.jsonl") as f:
    for line in f:
        row = json.loads(line)
        wc_map[row["id"]] = len(row["text"].split())

def len_bucket(wc):
    if wc <= 200: return "0-200"
    if wc <= 500: return "201-500"
    if wc <= 800: return "501-800"
    return "800+"

for df in preds.values():
    df["wc"] = df["id"].map(wc_map)
    df["len_bucket"] = df["wc"].apply(len_bucket)

# ── Table 1: Length-bucket accuracy ──────────────────────────────────────────
bucket_order = ["0-200", "201-500", "501-800", "800+"]
rows = []
for name, df in preds.items():
    g = df.groupby("len_bucket")["correct"].agg(["mean", "count"]).reindex(bucket_order)
    for bucket, row in g.iterrows():
        rows.append({"model": name, "len_bucket": bucket,
                     "accuracy": round(row["mean"], 4), "n": int(row["count"])})
length_table = pd.DataFrame(rows)
length_table.to_csv(OUT / "length_bucket_comparison.csv", index=False)

# Pivot for easy reading
pivot_len = length_table.pivot(index="len_bucket", columns="model", values="accuracy").reindex(bucket_order)
pivot_len.to_csv(OUT / "length_bucket_pivot.csv")
print("Length bucket pivot:")
print(pivot_len.to_string())

# ── Table 2: Per-AI-source accuracy (RoBERTa-base vs Fusion) ─────────────────
src_rows = []
for name, df in {"RoBERTa-base": preds["RoBERTa-base"], "Fusion-LightGBM": preds["Fusion-LightGBM"]}.items():
    g = df.groupby("model")["correct"].agg(["mean", "count"])
    for src, row in g.iterrows():
        src_rows.append({"model_name": name, "ai_source": src,
                         "accuracy": round(row["mean"], 4), "n": int(row["count"])})
source_table = pd.DataFrame(src_rows)
source_table.to_csv(OUT / "source_accuracy.csv", index=False)
print("\nSource accuracy:")
print(source_table.pivot(index="ai_source", columns="model_name", values="accuracy").to_string())

# ── Table 3: Reliability (calibration) data ──────────────────────────────────
n_bins = 10
bins = np.linspace(0, 1, n_bins + 1)
cal_rows = []
for name, df in preds.items():
    df = df.copy()
    df["bin"] = pd.cut(df["prob_ai"], bins=bins, labels=False, include_lowest=True)
    for b, g in df.groupby("bin"):
        center = (bins[b] + bins[b + 1]) / 2
        cal_rows.append({
            "model": name,
            "bin_center": round(float(center), 2),
            "frac_positive": round(float(g["label"].mean()), 4),
            "count": len(g),
        })
cal_table = pd.DataFrame(cal_rows)
cal_table.to_csv(OUT / "calibration_data.csv", index=False)

# ── Overall metrics summary ───────────────────────────────────────────────────
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
pd.DataFrame(overall).to_csv(OUT / "overall_metrics.csv", index=False)

print("\nAll tables written to", OUT)
