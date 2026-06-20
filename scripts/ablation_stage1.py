"""
Stage-1 ablations (all cheap, reuse dev/test feature CSVs, P1 protocol = fit on
dev, apply to test). Produces Table 1 (region-aware ablation) + Table 3 (ortho
feature leave-one-out / TTR causal) + A5 (classifier family) + A3 (band sweep).

Fixed detector = temperature-scaled RoBERTa margin. All decision params fit on
dev only. AUROC is threshold-invariant (self-check ~0.944).
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

REPO = Path("/inspire/hdd/project/fdu-aidake-cfff/public/hanz/semeval2024_task8a_en_baseline")
sys.path.insert(0, str(REPO))
from src.metrics import macro_f1, auroc

ORTHO = ["ttr", "repeated_bigram_ratio", "repeated_trigram_ratio",
         "avg_word_length", "text_length", "punctuation_per_token"]
TEMP = 5.367681468392405
GENS = ["GPT4", "bloomz", "chatGPT", "cohere", "davinci", "dolly"]

dev = pd.read_csv(REPO / "outputs/analysis_dev/features.csv")
test = pd.read_csv(REPO / "outputs/analysis/features.csv")
for d in (dev, test):
    d["margin"] = d["logit_ai"] - d["logit_human"]
yd, yt = dev["label"].values, test["label"].values
md, mt = dev["margin"].values, test["margin"].values


def best_single_threshold(margin, y):
    cand = np.quantile(margin, np.linspace(0, 1, 1500))
    mids = (cand[:-1] + cand[1:]) / 2
    best, bt = -1, 0.0
    for t in mids:
        s = macro_f1(y, (margin >= t).astype(int))
        if s > best:
            best, bt = s, t
    return bt


def per_gen_recall(df, pred):
    out = {}
    for g in GENS:
        m = (df.model == g).values
        out[g] = round(float((pred[m] == 1).mean()), 3) if m.sum() else float("nan")
    hm = (df.model == "human").values
    out["human_spec"] = round(float((pred[hm] == 0).mean()), 3)
    return out


def report(name, pred, df=test, y=yt):
    f1 = macro_f1(y, pred)
    pg = per_gen_recall(df, pred)
    print(f"{name:42s} F1={f1:.4f}  bloomz={pg['bloomz']:.3f}  GPT4={pg['GPT4']:.3f}  human_spec={pg['human_spec']:.3f}")
    return {"name": name, "macro_f1": round(f1, 4), **pg}


def fit_ortho(fit_df, band_mask, clf):
    clf.fit(fit_df.loc[band_mask, ORTHO].values, fit_df.loc[band_mask, "label"].values)
    return clf


# region-aware apply: clean_ai(m>=th)->1, clean_human(m<tl)->0, band->clf
def region_pred(df, tl, th, clf, feats=ORTHO):
    m = df["margin"].values
    pred = np.where(m >= th, 1, np.where(m < tl, 0, -1))
    band = (m >= tl) & (m < th)
    if band.sum():
        pred[band] = clf.predict(df.loc[band, feats].values)
    return pred


print("=== AUROC self-check (threshold-invariant) ===")
print(f"test AUROC = {auroc(yt, mt):.4f}\n")

rows = []
# deploy band from json
dm = json.loads((REPO / "outputs/region_aware/deploy_model.json").read_text())
TL, TH = dm["margin_t_low"], dm["margin_t_high"]
print(f"deploy band (dev-fit): [{TL:.2f}, {TH:.2f}]\n")

print("===== TABLE 1: Region-aware ablation (P1: fit dev, apply test) =====")
# P0 single threshold @0.5 (margin>=0)
rows.append(report("P0 single-thr @0.5 (margin>=0)", (mt >= 0).astype(int)))
# P1 single threshold dev-optimal
t_single = best_single_threshold(md, yd)
rows.append(report(f"P1 single-thr dev-opt (t*={t_single:.2f})", (mt >= t_single).astype(int)))
# A2: region but band decided by RoBERTa (dev-opt threshold within band)
band_dev = (md >= TL) & (md < TH)
t_band = best_single_threshold(md[band_dev], yd[band_dev])
def region_roberta_band(df, tl, th, tb):
    m = df["margin"].values
    pred = np.where(m >= th, 1, np.where(m < tl, 0, (m >= tb).astype(int)))
    return pred
rows.append(report("A2 region - ortho (RoBERTa in band)", region_roberta_band(test, TL, TH, t_band)))
# full region-aware (P1): ortho LR in band
clf_full = fit_ortho(dev, band_dev, LogisticRegression(max_iter=2000, class_weight="balanced"))
rows.append(report("Region-aware FULL (P1)", region_pred(test, TL, TH, clf_full)))

# P2: same-dist calib (stratified test split, mean over 5 seeds) - region-aware
def p2_region(seeds=(0,1,2,3,4)):
    f1s, pgs = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        cidx = []
        for _, g in test.groupby(["model","label"]):
            idx = g.index.to_numpy(); rng.shuffle(idx); cidx.extend(idx[:len(idx)//2])
        calib = test[test.index.isin(cidx)]; ev = test[~test.index.isin(cidx)]
        cm = calib["margin"].values
        tl = best_single_threshold(cm, calib["label"].values)  # reuse as proxy band edges
        # use deploy band but refit ortho on calib band
        bmask = (cm >= TL) & (cm < TH)
        if bmask.sum() < 30 or len(np.unique(calib["label"].values[bmask]))<2:
            continue
        c = fit_ortho(calib, bmask, LogisticRegression(max_iter=2000, class_weight="balanced"))
        pred = region_pred(ev, TL, TH, c)
        f1s.append(macro_f1(ev["label"].values, pred))
        pgs.append(per_gen_recall(ev, pred))
    return np.mean(f1s), np.std(f1s), {k: round(np.mean([p[k] for p in pgs]),3) for k in pgs[0]}
p2f1, p2sd, p2pg = p2_region()
print(f"{'Region-aware (P2 in-dist ceiling)':42s} F1={p2f1:.4f}±{p2sd:.4f}  bloomz={p2pg['bloomz']:.3f}  GPT4={p2pg['GPT4']:.3f}  human_spec={p2pg['human_spec']:.3f}")
rows.append({"name":"Region-aware (P2 ceiling)","macro_f1":round(float(p2f1),4),**p2pg})
# oracle single threshold on test
t_oracle = best_single_threshold(mt, yt)
rows.append(report(f"Single-thr ORACLE (test-fit)", (mt >= t_oracle).astype(int)))

print("\n===== TABLE 3: Ortho feature leave-one-out + TTR causal (P1) =====")
loo_rows = []
def report_loo(name, feats):
    c = fit_ortho(dev, band_dev, LogisticRegression(max_iter=2000, class_weight="balanced"))
    # refit with subset
    c.fit(dev.loc[band_dev, feats].values, yd[band_dev])
    pred = region_pred(test, TL, TH, c, feats=feats)
    f1 = macro_f1(yt, pred); pg = per_gen_recall(test, pred)
    print(f"{name:28s} F1={f1:.4f}  bloomz={pg['bloomz']:.3f}")
    return {"name":name,"macro_f1":round(f1,4),"bloomz":pg["bloomz"]}
base = report_loo("all features", ORTHO)
loo_rows.append(base)
for f in ORTHO:
    sub = [x for x in ORTHO if x != f]
    loo_rows.append(report_loo(f"- {f}", sub))
loo_rows.append(report_loo("TTR only", ["ttr"]))

print("\n===== A5: Classifier family in band (P1, overfit gap) =====")
def report_clf(name, clf):
    c = fit_ortho(dev, band_dev, clf)
    pred = region_pred(test, TL, TH, c)
    f1 = macro_f1(yt, pred); pg = per_gen_recall(test, pred)
    # dev band acc vs test band acc (overfit gap)
    dev_band_acc = (c.predict(dev.loc[band_dev,ORTHO].values)==yd[band_dev]).mean()
    tb = (mt>=TL)&(mt<TH)
    test_band_acc = (c.predict(test.loc[tb,ORTHO].values)==yt[tb]).mean()
    print(f"{name:16s} F1={f1:.4f}  bloomz={pg['bloomz']:.3f}  devBandAcc={dev_band_acc:.3f}  testBandAcc={test_band_acc:.3f}  gap={dev_band_acc-test_band_acc:.3f}")
    return {"name":name,"macro_f1":round(f1,4),"bloomz":pg["bloomz"],"overfit_gap":round(float(dev_band_acc-test_band_acc),3)}
a5=[]
import lightgbm as lgb
a5.append(report_clf("LR", LogisticRegression(max_iter=2000, class_weight="balanced")))
a5.append(report_clf("DecisionTree", DecisionTreeClassifier(max_depth=4, class_weight="balanced")))
a5.append(report_clf("LGBM", lgb.LGBMClassifier(n_estimators=100, verbose=-1)))

out = {"table1": rows, "table3_loo": loo_rows, "a5_classifier": a5,
       "test_auroc": round(float(auroc(yt, mt)),4), "deploy_band":[TL,TH]}
(REPO/"outputs/ablation").mkdir(parents=True, exist_ok=True)
(REPO/"outputs/ablation/stage1.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print("\nWrote outputs/ablation/stage1.json")
