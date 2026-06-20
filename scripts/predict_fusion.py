from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration import TemperatureScaler
from src.data import (
    build_prediction_frame,
    read_prediction_input,
    write_metrics_if_labeled,
    write_prediction_outputs,
)
from src.features import text_features


def load_preds(path: Path, required_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig") if path.suffix.lower() == ".csv" else pd.read_json(path, lines=True)
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def _apply_calibration(roberta_preds, calibration_file):
    if calibration_file is None or not Path(calibration_file).exists():
        return roberta_preds
    cal = json.loads(Path(calibration_file).read_text())
    scaler = TemperatureScaler.from_dict(cal)
    logits = roberta_preds[["logit_human", "logit_ai"]].to_numpy(dtype=float)
    out = roberta_preds.copy()
    out["prob_ai"] = scaler.predict_positive_proba(logits)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--tfidf_pred", type=Path, required=True)
    parser.add_argument("--roberta_pred", type=Path, required=True)
    parser.add_argument("--data_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--include_text", action="store_true")
    parser.add_argument("--calibration_file", type=Path, default=None,
                        help="Path to temperature.json for RoBERTa calibration")
    args = parser.parse_args()

    bundle = joblib.load(args.model_dir / "model.joblib")
    clf = bundle["clf"]
    scaler = bundle["scaler"]
    feat_names = bundle["feature_names"]

    df = read_prediction_input(args.data_file)
    tfidf_preds = load_preds(args.tfidf_pred, ["id", "prob_ai"])
    roberta_preds = load_preds(args.roberta_pred, ["id", "prob_ai", "logit_ai"])

    merged = df.merge(
        tfidf_preds[["id", "prob_ai"]].rename(columns={"prob_ai": "tfidf_prob_ai"}),
        on="id", how="left"
    ).merge(
        roberta_preds[["id", "prob_ai", "logit_ai"]].rename(
            columns={"prob_ai": "roberta_prob_ai", "logit_ai": "roberta_logit_ai"}
        ),
        on="id", how="left"
    )

    if args.calibration_file:
        roberta_preds = _apply_calibration(roberta_preds, args.calibration_file)
        merged = df.merge(
            tfidf_preds[["id", "prob_ai"]].rename(columns={"prob_ai": "tfidf_prob_ai"}),
            on="id", how="left"
        ).merge(
            roberta_preds[["id", "prob_ai", "logit_ai"]].rename(
                columns={"prob_ai": "roberta_prob_ai", "logit_ai": "roberta_logit_ai"}
            ),
            on="id", how="left"
        )

    manual = pd.DataFrame([text_features(t) for t in merged["text"]])
    base_cols = ["tfidf_prob_ai", "roberta_prob_ai", "roberta_logit_ai"]
    feat_df = pd.concat([merged[base_cols].reset_index(drop=True), manual.reset_index(drop=True)], axis=1)
    X = feat_df[feat_names].fillna(0.0).to_numpy(dtype=float)
    X_s = scaler.transform(X)

    prob_ai = clf.predict_proba(X_s)[:, 1]
    prob_human = 1.0 - prob_ai
    predictions = (prob_ai >= 0.5).astype(int)

    pred_df = build_prediction_frame(
        df=df,
        pred=predictions,
        prob_human=prob_human,
        prob_ai=prob_ai,
        include_text=args.include_text,
    )
    write_prediction_outputs(args.output_dir, pred_df)
    metrics = write_metrics_if_labeled(args.output_dir, pred_df)
    if metrics is not None:
        print(json.dumps(metrics, indent=2))
    else:
        print(json.dumps({"rows": int(len(df)), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
