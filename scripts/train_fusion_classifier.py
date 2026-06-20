from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration import TemperatureScaler
from src.features import text_features
from src.metrics import binary_metrics

DEFAULT_DATA_DIR = Path(
    "/inspire/hdd/project/fdu-aidake-cfff/public/wangyanqing/NLPPJ/data/processed/semeval"
)

FEATURE_NAMES = [
    "tfidf_prob_ai",
    "roberta_prob_ai",
    "roberta_logit_ai",
    "text_length", "word_count", "sentence_count", "paragraph_count",
    "avg_sentence_length", "sentence_length_std",
    "ttr", "avg_word_length",
    "repeated_bigram_ratio", "repeated_trigram_ratio",
    "punctuation_per_token",
    "first_person_ratio", "second_person_ratio",
    "contraction_ratio", "modal_verb_ratio", "discourse_marker_ratio",
]


def load_preds(path: Path, required_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df[["id", "label", "text"]]


def _apply_calibration(roberta_preds: pd.DataFrame, calibration_file: "Path | None") -> pd.DataFrame:
    """Replace prob_ai with calibrated probability if calibration_file is provided."""
    if calibration_file is None or not calibration_file.exists():
        return roberta_preds
    cal = json.loads(calibration_file.read_text(encoding="utf-8"))
    scaler = TemperatureScaler.from_dict(cal)
    # logit_human and logit_ai must exist
    logits = roberta_preds[["logit_human", "logit_ai"]].to_numpy(dtype=float)
    calibrated_prob = scaler.predict_positive_proba(logits)
    out = roberta_preds.copy()
    out["prob_ai"] = calibrated_prob
    return out


def build_features(tfidf_preds: pd.DataFrame, roberta_preds: pd.DataFrame, data: pd.DataFrame) -> np.ndarray:
    merged = data.merge(
        tfidf_preds[["id", "prob_ai"]].rename(columns={"prob_ai": "tfidf_prob_ai"}),
        on="id", how="left"
    ).merge(
        roberta_preds[["id", "prob_ai", "logit_ai"]].rename(
            columns={"prob_ai": "roberta_prob_ai", "logit_ai": "roberta_logit_ai"}
        ),
        on="id", how="left"
    )

    manual = pd.DataFrame([text_features(t) for t in merged["text"]])
    feat_cols = ["tfidf_prob_ai", "roberta_prob_ai", "roberta_logit_ai"] + list(manual.columns)
    feat_df = pd.concat([merged[["tfidf_prob_ai", "roberta_prob_ai", "roberta_logit_ai"]].reset_index(drop=True),
                         manual.reset_index(drop=True)], axis=1)
    return feat_df[feat_cols].fillna(0.0).to_numpy(dtype=float), list(feat_cols)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tfidf_pred", type=Path, required=True)
    parser.add_argument("--train_roberta_pred", type=Path, required=True)
    parser.add_argument("--train_data", type=Path, default=DEFAULT_DATA_DIR / "semeval_train_full.csv")
    parser.add_argument("--dev_tfidf_pred", type=Path, required=True)
    parser.add_argument("--dev_roberta_pred", type=Path, required=True)
    parser.add_argument("--dev_data", type=Path, default=DEFAULT_DATA_DIR / "semeval_dev_full.csv")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/fusion"))
    parser.add_argument("--model", choices=["lr", "lgbm"], default="lr")
    parser.add_argument("--calibration_file", type=Path, default=None,
                        help="Path to temperature.json for RoBERTa calibration")
    args = parser.parse_args()

    print(f"Loading data for model={args.model}...")
    train_df = load_data(args.train_data)
    dev_df = load_data(args.dev_data)
    train_tfidf = load_preds(args.train_tfidf_pred, ["id", "prob_ai"])
    train_roberta = load_preds(args.train_roberta_pred, ["id", "prob_ai", "logit_ai"])
    dev_tfidf = load_preds(args.dev_tfidf_pred, ["id", "prob_ai"])
    dev_roberta = load_preds(args.dev_roberta_pred, ["id", "prob_ai", "logit_ai"])

    if args.calibration_file:
        print(f"Applying RoBERTa calibration from {args.calibration_file}...")
        train_roberta = _apply_calibration(train_roberta, args.calibration_file)
        dev_roberta = _apply_calibration(dev_roberta, args.calibration_file)

    print("Building feature matrices...")
    X_train, feat_names = build_features(train_tfidf, train_roberta, train_df)
    X_dev, _ = build_features(dev_tfidf, dev_roberta, dev_df)
    y_train = train_df["label"].to_numpy()
    y_dev = dev_df["label"].to_numpy()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_dev_s = scaler.transform(X_dev)

    if args.model == "lr":
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="saga", random_state=42, n_jobs=-1)
        clf.fit(X_train_s, y_train)
        prob_ai = clf.predict_proba(X_dev_s)[:, 1]
        importance = dict(zip(feat_names, clf.coef_[0].tolist()))
    else:
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise SystemExit("lightgbm not installed. Run: .conda/bin/pip install lightgbm")
        clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             random_state=42, n_jobs=-1, verbose=-1)
        clf.fit(X_train_s, y_train)
        prob_ai = clf.predict_proba(X_dev_s)[:, 1]
        importance = dict(zip(feat_names, [float(x) for x in clf.feature_importances_]))

    preds = (prob_ai >= 0.5).astype(int)
    metrics = binary_metrics(y_dev, prob_ai)
    metrics["rows"] = int(len(y_dev))
    metrics["model"] = args.model

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "feature_importance.json").write_text(json.dumps(importance, indent=2), encoding="utf-8")
    joblib.dump({"clf": clf, "scaler": scaler, "feature_names": feat_names, "model_type": args.model},
                args.output_dir / "model.joblib")

    pred_df = dev_df.copy()
    pred_df["pred"] = preds
    pred_df["prob_ai"] = prob_ai
    pred_df["prob_human"] = 1.0 - prob_ai
    pred_df["correct"] = pred_df["label"] == pred_df["pred"]
    pred_df.drop(columns=["text"], errors="ignore").to_csv(args.output_dir / "predictions.csv", index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
