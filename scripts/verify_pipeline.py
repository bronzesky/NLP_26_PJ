import sys, pandas as pd
sys.path.insert(0, ".")
from src.detector_pipeline import RegionAwareDetector

df = pd.read_csv("outputs/analysis/features.csv")
det = RegionAwareDetector()
det._lm = None  # skip ppl for speed

for model in ["human", "bloomz", "GPT4", "chatGPT"]:
    sub = df[df.model == model]
    n_ok = 0
    for _, row in sub.head(20).iterrows():
        r = det.analyze(str(row["text"]), title=model, with_suggestions=False)
        n_ok += int(r["doc_label"] == row["label"])
    # show first sample detail
    row = sub.iloc[0]
    r = det.analyze(str(row["text"]), title=model, with_suggestions=False)
    tl = int(row["label"])
    print(f"{model:8s} true={tl} pred={r['doc_label']} pAI={r['doc_prob_ai']:.3f} "
          f"region={r['doc_region']:12s} margin={r['doc_margin']:.2f} | acc@20={n_ok}/20")
