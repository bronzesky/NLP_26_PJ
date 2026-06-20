from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features import text_features
from src.metrics import binary_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--span_labels", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/span_classifier"))
    parser.add_argument("--tfidf_model_file", type=Path, default=Path("outputs/tfidf/model.joblib"))
    parser.add_argument("--model", choices=["lr", "lgbm"], default="lgbm")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading span labels...")
    spans_df = pd.read_csv(args.span_labels, encoding="utf-8-sig")
    spans_df["span_text"] = spans_df["span_text"].fillna("").astype(str)
    spans_df["span_label"] = spans_df["span_label"].astype(int)

    print(f"Computing manual features for {len(spans_df)} spans...")
    manual = pd.DataFrame([text_features(t) for t in spans_df["span_text"]])

    # optional: add TF-IDF features
    tfidf_prob_ai = np.full(len(spans_df), 0.5)
    if args.tfidf_model_file.exists():
        try:
            tfidf_model = joblib.load(args.tfidf_model_file)
            raw = tfidf_model.predict_proba(spans_df["span_text"].tolist())
            classes = np.asarray(getattr(tfidf_model, "classes_", [0, 1]))
            ai_col = np.where(classes == 1)[0]
            if len(ai_col):
                tfidf_prob_ai = raw[:, int(ai_col[0])]
        except Exception as e:
            print(f"Warning: could not use TF-IDF model: {e}")

    manual["tfidf_prob_ai"] = tfidf_prob_ai
    feat_names = list(manual.columns)
    X = manual.fillna(0.0).to_numpy(dtype=float)
    y = spans_df["span_label"].to_numpy()

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(spans_df)), test_size=args.test_size, random_state=args.seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if args.model == "lr":
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="saga", random_state=args.seed, n_jobs=-1)
        clf.fit(X_train_s, y_train)
    else:
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise SystemExit("lightgbm not installed. Run: .conda/bin/pip install lightgbm")
        clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             random_state=args.seed, n_jobs=-1, verbose=-1)
        clf.fit(X_train_s, y_train)

    prob_ai_test = clf.predict_proba(X_test_s)[:, 1]
    span_metrics = binary_metrics(y_test, prob_ai_test)
    span_metrics["n_spans_train"] = int(len(X_train))
    span_metrics["n_spans_test"] = int(len(X_test))
    span_metrics["model"] = args.model

    # document-level accuracy via majority vote on test spans
    test_spans = spans_df.iloc[idx_test].copy()
    test_spans["pred"] = (prob_ai_test >= 0.5).astype(int)
    if "doc_id" in test_spans.columns:
        doc_preds = test_spans.groupby("doc_id")["pred"].mean().apply(lambda x: int(x >= 0.5))
        # mixed_docs has document_label — try to load for doc-level eval
        mixed_docs_path = args.span_labels.parent / "mixed_docs.csv"
        if mixed_docs_path.exists():
            docs_df = pd.read_csv(mixed_docs_path)[["id", "document_label"]]
            docs_df = docs_df.rename(columns={"id": "doc_id"})
            doc_eval = doc_preds.reset_index().merge(docs_df, on="doc_id", how="inner")
            if len(doc_eval):
                doc_acc = float((doc_eval["pred"] == doc_eval["document_label"]).mean())
                span_metrics["doc_accuracy_majority_vote"] = doc_acc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(span_metrics, indent=2), encoding="utf-8")
    joblib.dump({"clf": clf, "scaler": scaler, "feature_names": feat_names, "model_type": args.model},
                args.output_dir / "model.joblib")

    print(json.dumps(span_metrics, indent=2))


if __name__ == "__main__":
    main()
