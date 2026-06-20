"""
Paper figures, redesigned to a top-journal style:
- Okabe-Ito colorblind-safe palette (Nature-recommended)
- sans-serif, no top/right spines, hairline axes, vector PDF
- single-column width (~3.4in), description goes in the caption (minimal title)
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/inspire/hdd/project/fdu-aidake-cfff/public/hanz/semeval2024_task8a_en_baseline")
sys.path.insert(0, str(REPO))
from src.metrics import macro_f1, auroc
FIG = REPO / "paper/figs"; FIG.mkdir(parents=True, exist_ok=True)

# ---- Okabe-Ito palette ----
OI = {"black":"#000000","orange":"#E69F00","skyblue":"#56B4E9","green":"#009E73",
      "yellow":"#F0E442","blue":"#0072B2","vermillion":"#D55E00","purple":"#CC79A7",
      "grey":"#999999"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
W1 = 3.4  # single column inch

test = pd.read_csv(REPO/"outputs/analysis/features.csv")
test["margin"] = test["logit_ai"] - test["logit_human"]
yt, mt = test["label"].values, test["margin"].values

# ---- fig1: macro-F1 vs threshold ----
ths = np.linspace(mt.min(), mt.max(), 240)
f1s = [macro_f1(yt, (mt >= t).astype(int)) for t in ths]
fig, ax = plt.subplots(figsize=(W1, 2.5), constrained_layout=True)
ax.plot(ths, f1s, color=OI["blue"], lw=1.6, zorder=3)
ax.axvline(0, ls=(0,(4,2)), color=OI["vermillion"], lw=1.1, zorder=2)
best_t = ths[int(np.argmax(f1s))]
ax.axvline(best_t, ls=(0,(1,1.5)), color=OI["green"], lw=1.1, zorder=2)
f0 = macro_f1(yt,(mt>=0).astype(int))
ax.scatter([0],[f0], color=OI["vermillion"], s=16, zorder=4)
ax.scatter([best_t],[max(f1s)], color=OI["green"], s=16, zorder=4)
ax.annotate(f"default (margin=0)\nF1={f0:.2f}", (0,f0), (0.04,0.30), textcoords="axes fraction",
            fontsize=6.3, color=OI["vermillion"], ha="left",
            arrowprops=dict(arrowstyle="-", color=OI["vermillion"], lw=0.5))
ax.annotate(f"oracle thr\nF1={max(f1s):.2f}", (best_t,max(f1s)), (0.62,0.55), textcoords="axes fraction",
            fontsize=6.3, color=OI["green"], ha="left",
            arrowprops=dict(arrowstyle="-", color=OI["green"], lw=0.5))
ax.set_xlabel("decision threshold (RoBERTa margin)"); ax.set_ylabel("test macro-F1")
ax.set_title(f"AUROC = {auroc(yt,mt):.3f}, yet F1 collapses at the default threshold", fontsize=7.6)
ax.set_ylim(0.45, 0.92)
fig.savefig(FIG/"fig1_operating_point.pdf"); plt.close()

# ---- fig2: margin distributions ----
fig, ax = plt.subplots(figsize=(W1, 2.5), constrained_layout=True)
groups = [("human", OI["green"], test[test.model=="human"]),
          ("bloomz (unseen)", OI["orange"], test[test.model=="bloomz"]),
          ("GPT-4 (unseen)", OI["blue"], test[test.model=="GPT4"]),
          ("seen AI", OI["vermillion"], test[test.model.isin(["chatGPT","cohere","davinci","dolly"])])]
bins = np.linspace(-15, 14, 64)
ax.axvspan(-10.58, 10.91, color=OI["grey"], alpha=0.16, lw=0, zorder=1)
for name,c,d in groups:
    ax.hist(d["margin"], bins=bins, density=True, histtype="step", color=c, lw=1.4, zorder=3, label=name)
ax.text(0.5, 0.95, "ambiguous band", transform=ax.transAxes, ha="center",
        fontsize=6.2, color=OI["grey"])
ax.set_xlabel("RoBERTa margin  (logit$_{\\mathrm{AI}}$ − logit$_{\\mathrm{human}}$)")
ax.set_ylabel("density")
ax.set_title("bloomz overlaps the human upper tail", fontsize=7.6)
ax.legend(loc="upper left", handlelength=1.2)
fig.savefig(FIG/"fig2_margin_overlap.pdf"); plt.close()

# ---- fig3: ablation bars ----
abl = json.loads((REPO/"outputs/ablation/stage1.json").read_text())
t1 = {r["name"]: r["macro_f1"] for r in abl["table1"]}
items = [("single-thr\n@0.5","P0 single-thr @0.5 (margin>=0)",OI["grey"]),
         ("single-thr\ndev-opt","P1 single-thr dev-opt (t*=-5.87)",OI["grey"]),
         ("− orthogonal\n(RoBERTa in band)","A2 region - ortho (RoBERTa in band)",OI["grey"]),
         ("region-aware\n(full)","Region-aware FULL (P1)",OI["blue"]),
         ("single-thr\noracle","Single-thr ORACLE (test-fit)",OI["orange"])]
vals = [t1.get(k,0) for _,k,_ in items]
cols = [c for *_,c in items]
fig, ax = plt.subplots(figsize=(W1, 2.7), constrained_layout=True)
bars = ax.bar(range(len(vals)), vals, color=cols, width=0.66, zorder=3)
ax.set_xticks(range(len(vals))); ax.set_xticklabels([l for l,_,_ in items], fontsize=6.3)
ax.set_ylabel("test macro-F1"); ax.set_ylim(0.5, 0.93)
ax.grid(axis="y", lw=0.4, color="#dddddd", zorder=0)
ax.set_axisbelow(True)
for i,v in enumerate(vals): ax.text(i, v+0.006, f"{v:.3f}", ha="center", fontsize=6.5)
ax.set_title("Orthogonal stage supplies +0.27; oracle can't (sacrifices bloomz)", fontsize=7.2)
fig.savefig(FIG/"fig3_ablation.pdf"); plt.close()

print("redrew fig1/fig2/fig3 (Okabe-Ito, sans-serif, minimal spines) -> paper/figs/")

# ---- fig4: de-AI rate descent (targeted recursive, fixed RoBERTa) ----
tj = json.loads((REPO/"outputs/humanize_demo/targeted.json").read_text())
traj = tj["trajectory"]
xs = [t["round"] for t in traj]; ys = [t["prob_ai"] for t in traj]
fig, ax = plt.subplots(figsize=(W1, 2.5), constrained_layout=True)
ax.axhline(0.5, ls=(0,(4,2)), color=OI["vermillion"], lw=1.0, zorder=1)
ax.text(xs[-1], 0.52, "verdict threshold", color=OI["vermillion"], fontsize=6.2, ha="right")
ax.plot(xs, ys, "-o", color=OI["blue"], lw=1.6, ms=4, zorder=3)
# color points by verdict
for x,y in zip(xs,ys):
    ax.scatter([x],[y], color=(OI["blue"] if y>=0.5 else OI["green"]), s=20, zorder=4)
ax.annotate(f"{ys[0]:.2f}\n(AI)", (xs[0],ys[0]), (xs[0]+0.15,ys[0]-0.04), fontsize=6.3, color=OI["blue"])
ax.annotate(f"{ys[-1]:.2f}\n(human)", (xs[-1],ys[-1]), (xs[-1]-0.9,ys[-1]+0.06), fontsize=6.3, color=OI["green"])
ax.set_xlabel("targeted-rewrite round"); ax.set_ylabel("AI rate (fixed RoBERTa P$_{\\mathrm{AI}}$)")
ax.set_xticks(xs); ax.set_ylim(0.3, 0.95)
ax.set_title("Targeted recursive de-AI: same detector measures throughout", fontsize=7.4)
fig.savefig(FIG/"fig4_deai_descent.pdf"); plt.close()
print("wrote fig4_deai_descent.pdf")
