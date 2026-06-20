"""
C1 detection-side: leave-one-linguistic-group-out on Fusion-LGBM.
Rebuild the 16 handcrafted features + tfidf/roberta signals, group by
linguistic layer, drop each group, retrain LGBM, report test macro-F1 delta.
Train subsampled for speed; test full.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

REPO = Path("/inspire/hdd/project/fdu-aidake-cfff/public/hanz/semeval2024_task8a_en_baseline")
sys.path.insert(0, str(REPO))
from src.features import text_features
from src.metrics import macro_f1

DATA = Path("/inspire/hdd/project/fdu-aidake-cfff/public/wangyanqing/NLPPJ/data/processed/semeval")

# model signals (always kept)
SIGNALS = ["tfidf_prob_ai", "roberta_prob_ai", "roberta_logit_ai"]
GROUPS = {
    "lexical": ["ttr", "avg_word_length"],
    "structural": ["text_length", "word_count", "sentence_count", "paragraph_count",
                   "avg_sentence_length", "sentence_length_std"],
    "repetition": ["repeated_bigram_ratio", "repeated_trigram_ratio", "punctuation_per_token"],
    "pragmatic": ["first_person_ratio", "second_person_ratio", "contraction_ratio", "modal_verb_ratio"],
    "discourse": ["discourse_marker_ratio"],
}
ALL_HAND = [f for g in GROUPS.values() for f in g]


def build(pred_tfidf, pred_rob, data_csv, n=None):
    tf = pd.read_csv(pred_tfidf, encoding="utf-8-sig")[["id", "prob_ai"]].rename(columns={"prob_ai": "tfidf_prob_ai"})
    rb = pd.read_csv(pred_rob, encoding="utf-8-sig")[["id", "prob_ai", "logit_ai", "label"]].rename(columns={"prob_ai": "roberta_prob_ai", "logit_ai": "roberta_logit_ai"})
    dat = pd.read_csv(data_csv, encoding="utf-8-sig")[["id", "text"]]
    df = rb.merge(tf, on="id").merge(dat, on="id")
    if n:
        df = df.sample(min(n, len(df)), random_state=42).reset_index(drop=True)
    feats = df["text"].fillna("").astype(str).map(lambda t: text_features(t))
    fdf = pd.DataFrame(list(feats))
    for c in ALL_HAND:
        if c not in fdf: fdf[c] = 0.0
    out = pd.concat([df[["id", "label"] + SIGNALS].reset_index(drop=True), fdf[ALL_HAND]], axis=1)
    return out

print("building train features (subsample 20k)...")
tr = build(REPO/"outputs/tfidf_train_predict/predictions.csv",
           REPO/"outputs/roberta_base/train_predict/predictions.csv",
           DATA/"semeval_train_full.csv", n=20000)
print("building test features (text from analysis/features.csv)...")
def build_test():
    tf = pd.read_csv(REPO/"outputs/tfidf_test/predictions.csv")[["id","prob_ai"]].rename(columns={"prob_ai":"tfidf_prob_ai"})
    rb = pd.read_csv(REPO/"outputs/roberta_base_test/predictions.csv")[["id","prob_ai","logit_ai","label"]].rename(columns={"prob_ai":"roberta_prob_ai","logit_ai":"roberta_logit_ai"})
    dat = pd.read_csv(REPO/"outputs/analysis/features.csv")[["id","text"]]
    df = rb.merge(tf,on="id").merge(dat,on="id")
    feats = df["text"].fillna("").astype(str).map(lambda t: text_features(t))
    fdf = pd.DataFrame(list(feats))
    for c in ALL_HAND:
        if c not in fdf: fdf[c]=0.0
    return pd.concat([df[["id","label"]+SIGNALS].reset_index(drop=True), fdf[ALL_HAND]],axis=1)
te = build_test()

ytr, yte = tr["label"].values, te["label"].values

def train_eval(feat_cols, name):
    m = lgb.LGBMClassifier(n_estimators=200, verbose=-1)
    m.fit(tr[feat_cols].values, ytr)
    p = m.predict_proba(te[feat_cols].values)[:, 1]
    f1 = macro_f1(yte, (p >= 0.5).astype(int))
    print(f"{name:26s} F1={f1:.4f}  (#feat={len(feat_cols)})")
    return round(f1, 4)

print()
full_cols = SIGNALS + ALL_HAND
base = train_eval(full_cols, "all features")
rows = [{"config": "all", "macro_f1": base}]
for g, cols in GROUPS.items():
    keep = [c for c in full_cols if c not in cols]
    f1 = train_eval(keep, f"- {g}")
    rows.append({"config": f"-{g}", "macro_f1": f1, "delta": round(f1 - base, 4)})
# signals only (no handcrafted)
f1s = train_eval(SIGNALS, "signals only (no handcrafted)")
rows.append({"config": "signals_only", "macro_f1": f1s, "delta": round(f1s - base, 4)})

(REPO/"outputs/ablation/c1_feature_groups.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
print("\nWrote outputs/ablation/c1_feature_groups.json")
