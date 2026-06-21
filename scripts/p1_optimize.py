"""
P1 + A1: optimize the ambiguous-band arbiter against the human-specificity
bottleneck, and evaluate the 1D-TTR + 2-region simplifications. All P1 (fit on
dev, apply to test), reuse feature CSVs. Fixed RoBERTa margin.

Variants:
  B0  baseline: 6-feat LR (balanced) in band, ortho thr 0.5      [current 0.857]
  B1  1D-TTR LR (balanced), ortho thr 0.5                        [elegance E1]
  B2  1D-TTR LR, sweep ortho decision threshold on dev           [param P1]
  B3  1D-TTR LR, sweep class_weight on dev                       [param P1]
  A1  drop t_low (2-region: >=t_high AI, else TTR rule)          [arch A1]
Report test macro-F1, bloomz recall, human specificity.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO = Path("/inspire/hdd/project/fdu-aidake-cfff/public/hanz/semeval2024_task8a_en_baseline")
sys.path.insert(0, str(REPO))
from src.metrics import macro_f1

ORTHO6 = ["ttr","repeated_bigram_ratio","repeated_trigram_ratio","avg_word_length","text_length","punctuation_per_token"]
TL, TH = -10.58, 10.91
dev = pd.read_csv(REPO/"outputs/analysis_dev/features.csv")
test = pd.read_csv(REPO/"outputs/analysis/features.csv")
for d in (dev,test): d["margin"]=d["logit_ai"]-d["logit_human"]
yd,yt=dev["label"].values,test["label"].values
md,mt=dev["margin"].values,test["margin"].values

def metrics(df, pred):
    f1=macro_f1(df["label"].values,pred)
    bl=(pred[(df.model=="bloomz").values]==1).mean()
    hs=(pred[(df.model=="human").values]==0).mean()
    return f1,bl,hs

def apply_region(df, clf, feats, tl, th, ortho_thr=0.5, two_region=False):
    m=df["margin"].values
    p_ortho=clf.predict_proba(df[feats].values)[:,1]
    if two_region:
        # >=th -> AI ; else TTR rule
        pred=np.where(m>=th, 1, (p_ortho>=ortho_thr).astype(int))
    else:
        pred=np.where(m>=th,1,np.where(m<tl,0,-1))
        band=(m>=tl)&(m<th)
        pred[band]=(p_ortho[band]>=ortho_thr).astype(int)
    return pred

print(f"{'variant':32s}{'F1':>8s}{'bloomz':>8s}{'human_spec':>11s}")
rows=[]

# B0 baseline 6-feat
bdev=(md>=TL)&(md<TH)
clf6=LogisticRegression(max_iter=2000,class_weight="balanced").fit(dev.loc[bdev,ORTHO6].values,yd[bdev])
f1,bl,hs=metrics(test, apply_region(test,clf6,ORTHO6,TL,TH))
print(f"{'B0 6-feat LR balanced @0.5':32s}{f1:>8.4f}{bl:>8.3f}{hs:>11.3f}"); rows.append(("B0",f1,bl,hs))

# B1 1D TTR
clf1=LogisticRegression(max_iter=2000,class_weight="balanced").fit(dev.loc[bdev,["ttr"]].values,yd[bdev])
f1,bl,hs=metrics(test, apply_region(test,clf1,["ttr"],TL,TH))
print(f"{'B1 1D-TTR LR balanced @0.5':32s}{f1:>8.4f}{bl:>8.3f}{hs:>11.3f}"); rows.append(("B1",f1,bl,hs))

# B2 1D TTR, sweep ortho threshold on dev
def dev_f1_at(clf,feats,thr):
    return macro_f1(yd, apply_region(dev,clf,feats,TL,TH,ortho_thr=thr))
best=max(np.linspace(0.3,0.8,51), key=lambda t: dev_f1_at(clf1,["ttr"],t))
f1,bl,hs=metrics(test, apply_region(test,clf1,["ttr"],TL,TH,ortho_thr=best))
print(f"{'B2 1D-TTR thr=%.2f(dev-opt)'%best:32s}{f1:>8.4f}{bl:>8.3f}{hs:>11.3f}"); rows.append(("B2",f1,bl,hs))

# B3 1D TTR sweep class_weight on dev
best_cw=None; best_f1=-1
for w in [None,"balanced",{0:1,1:0.7},{0:1,1:0.5},{0:1.5,1:1},{0:2,1:1}]:
    c=LogisticRegression(max_iter=2000,class_weight=w).fit(dev.loc[bdev,["ttr"]].values,yd[bdev])
    fd=macro_f1(yd,apply_region(dev,c,["ttr"],TL,TH))
    if fd>best_f1: best_f1,best_cw,bc=fd,w,c
f1,bl,hs=metrics(test, apply_region(test,bc,["ttr"],TL,TH))
print(f"{'B3 1D-TTR cw=%s'%str(best_cw):32s}{f1:>8.4f}{bl:>8.3f}{hs:>11.3f}"); rows.append(("B3",f1,bl,hs))

# A1 two-region (drop t_low), 1D TTR, dev-opt thr
best_a=max(np.linspace(0.3,0.8,51), key=lambda t: macro_f1(yd, apply_region(dev,clf1,["ttr"],TL,TH,ortho_thr=t,two_region=True)))
f1,bl,hs=metrics(test, apply_region(test,clf1,["ttr"],TL,TH,ortho_thr=best_a,two_region=True))
print(f"{'A1 2-region 1D-TTR thr=%.2f'%best_a:32s}{f1:>8.4f}{bl:>8.3f}{hs:>11.3f}"); rows.append(("A1",f1,bl,hs))

(REPO/"outputs/ablation/p1_optimize.json").write_text(json.dumps(
    {"rows":[{"v":v,"macro_f1":round(f,4),"bloomz":round(b,3),"human_spec":round(h,3)} for v,f,b,h in rows],
     "best_ortho_thr":round(float(best),3),"best_cw":str(best_cw),"a1_thr":round(float(best_a),3)},
    ensure_ascii=False,indent=2))
print("\nWrote outputs/ablation/p1_optimize.json")
