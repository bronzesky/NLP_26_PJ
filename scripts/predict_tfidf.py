import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import (  # noqa: E402
    build_prediction_frame,
    read_prediction_input,
    write_metrics_if_labeled,
    write_prediction_outputs,
)


def class_probabilities(model, texts) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.asarray(model.predict(texts), dtype=int)

    if hasattr(model, "predict_proba"):
        raw_probs = np.asarray(model.predict_proba(texts), dtype=float)
        classes = np.asarray(getattr(model, "classes_", [0, 1]), dtype=int)
        prob_human = np.zeros(len(pred), dtype=float)
        prob_ai = np.zeros(len(pred), dtype=float)
        for idx, label in enumerate(classes):
            if label == 0:
                prob_human = raw_probs[:, idx]
            elif label == 1:
                prob_ai = raw_probs[:, idx]
        return pred, prob_human, prob_ai

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(texts), dtype=float)
        if scores.ndim == 2 and scores.shape[1] == 2:
            exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            return pred, probs[:, 0], probs[:, 1]

        prob_ai = 1.0 / (1.0 + np.exp(-scores))
        return pred, 1.0 - prob_ai, prob_ai

    prob_ai = pred.astype(float)
    return pred, 1.0 - prob_ai, prob_ai


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        "--model_path",
        dest="model_file",
        type=Path,
        default=Path("outputs/tfidf/model.joblib"),
    )
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--include_text", action="store_true")
    args = parser.parse_args()

    df = read_prediction_input(args.input_file)
    model = joblib.load(args.model_file)
    pred, prob_human, prob_ai = class_probabilities(model, df["text"])

    pred_df = build_prediction_frame(
        df=df,
        pred=pred,
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
