"""
Stage-1 ablations B1 (calibration) + A3 (band-width sweep). Cheap, reuse
dev/test feature CSVs. P1 protocol where fitting is needed.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

REPO = Path("/inspire/hdd/project/fdu-aidake-cfff/public/hanz/semeval2024_task8a_en_baseline")
sys.path.insert(0, str(REPO))
from src.metrics import macro_f1, auroc, ece, brier

ORTHO = ["ttr", "repeated_bigram_ratio", "repeated_trigram_ratio",
         "avg_word_length", "text_length", "punctuation_per_token"]
TEMP = 5.367681468392405

dev = pd.read_csv(REPO / "outputs/analysis_dev/features.csv")
test = pd.read_csv(REPO / "outputs/analysis/features.csv")
for d in (dev, test):
    d["margin"] = d["logit_ai"] - d["logit_human"]
yd, yt = dev["label"].values, test["label"].values
md, mt = dev["margin"].values, test["margin"].values


def best_thr(margin, y):
    cand = np.quantile(margin, np.linspace(0, 1, 1500))
    mids = (cand[:-1] + cand[1:]) / 2
    best, bt = -1, 0.0
    for t in mids:
        s = macro_f1(y, (margin >= t).astype(int))
        if s > best:
            best, bt = s, t
    return bt


def bloomz_rec(df, pred):
    m = (df.model == "bloomz").values
    return float((pred[m] == 1).mean())


# ===================== B1: calibration methods =====================
# Probabilities under each calibration (fit calibrator on DEV), then two reads:
#  (a) fixed 0.5 threshold;  (b) dev-optimal threshold on the calibrated prob.
print("===== B1: Calibration methods (fit on dev) =====")
print(f"{'method':14s}{'read':12s}{'macroF1':>9s}{'AUROC':>8s}{'ECE':>8s}{'Brier':>8s}")

def sig(x): return 1/(1+np.exp(-x))

# raw prob from margin (T=1) vs temperature T=5.37
def eval_prob(name, p_dev, p_test):
    au = auroc(yt, p_test)
    # (a) @0.5
    pred05 = (p_test >= 0.5).astype(int)
    print(f"{name:14s}{'@0.5':12s}{macro_f1(yt,pred05):>9.4f}{au:>8.4f}{ece(yt,p_test):>8.4f}{brier(yt,p_test):>8.4f}")
    # (b) dev-opt threshold on calibrated prob
    cand=np.quantile(p_dev,np.linspace(0,1,500)); mids=(cand[:-1]+cand[1:])/2
    bt=max(mids,key=lambda t:macro_f1(yd,(p_dev>=t).astype(int)))
    predb=(p_test>=bt).astype(int)
    print(f"{'':14s}{'dev-thr':12s}{macro_f1(yt,predb):>9.4f}{au:>8.4f}{ece(yt,p_test):>8.4f}{brier(yt,p_test):>8.4f}")

# none (T=1)
eval_prob("none(T=1)", sig(md), sig(mt))
# temperature T=5.37
eval_prob("temp(5.37)", sig(md/TEMP), sig(mt/TEMP))
# Platt: logistic on margin, fit dev
platt=LogisticRegression(max_iter=1000); platt.fit(md.reshape(-1,1),yd)
eval_prob("Platt", platt.predict_proba(md.reshape(-1,1))[:,1], platt.predict_proba(mt.reshape(-1,1))[:,1])
# Isotonic on margin, fit dev
iso=IsotonicRegression(out_of_bounds="clip"); iso.fit(md,yd)
eval_prob("Isotonic", iso.predict(md), iso.predict(mt))

print("\n  -> AUROC identical across all (threshold/monotonic-calib invariant).")
print("  -> dev-thr read: F1 ~equal across calibs => threshold is the lever, T redundant in margin space.")

# ===================== A3: band-width sweep =====================
print("\n===== A3: band-width sweep (region-aware, ortho LR in band, P1) =====")
print(f"{'t_low':>8s}{'t_high':>8s}{'%band':>7s}{'macroF1':>9s}{'bloomz':>8s}")
sweep=[]
qs=np.quantile(md,[0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
def region_eval(tl,th):
    bdev=(md>=tl)&(md<th)
    if bdev.sum()<30 or len(np.unique(yd[bdev]))<2:
        # degenerate: pure single threshold at tl (=th)
        pred=(mt>=tl).astype(int); return macro_f1(yt,pred),bloomz_rec(test,pred),0.0
    clf=LogisticRegression(max_iter=2000,class_weight="balanced")
    clf.fit(dev.loc[bdev,ORTHO].values,yd[bdev])
    m=mt; pred=np.where(m>=th,1,np.where(m<tl,0,-1)); band=(m>=tl)&(m<th)
    pred[band]=clf.predict(test.loc[band,ORTHO].values)
    return macro_f1(yt,pred),bloomz_rec(test,pred),float(band.mean())
# degenerate end A: band width 0 (single threshold dev-opt)
t1=best_thr(md,yd); p=(mt>=t1).astype(int)
print(f"{'(single-thr)':>16s}{0.0:>7.2f}{macro_f1(yt,p):>9.4f}{bloomz_rec(test,p):>8.3f}   <- band width 0")
sweep.append({"config":"single-thr","macro_f1":round(macro_f1(yt,p),4),"bloomz":round(bloomz_rec(test,p),3)})
# degenerate end B: full-domain band (everything -> ortho)
tl_all,th_all=md.min()-1,md.max()+1
f1a,bla,frac=region_eval(tl_all,th_all)
print(f"{tl_all:>8.1f}{th_all:>8.1f}{frac*100:>6.0f}%{f1a:>9.4f}{bla:>8.3f}   <- pure handcrafted")
sweep.append({"config":"full-band(pure-ortho)","macro_f1":round(f1a,4),"bloomz":round(bla,3),"frac":round(frac,3)})
# grid of bands
for tl in [-12,-10.58,-8,-5]:
    for th in [8,10.91,12]:
        if th<=tl: continue
        f1b,blb,fr=region_eval(tl,th)
        print(f"{tl:>8.2f}{th:>8.2f}{fr*100:>6.0f}%{f1b:>9.4f}{blb:>8.3f}")
        sweep.append({"t_low":tl,"t_high":th,"frac":round(fr,3),"macro_f1":round(f1b,4),"bloomz":round(blb,3)})

out={"a3_sweep":sweep}
(REPO/"outputs/ablation/stage1b.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
print("\nWrote outputs/ablation/stage1b.json")
