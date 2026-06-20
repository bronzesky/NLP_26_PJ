import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


DEFAULT_DATA_DIR = Path(
    "/inspire/hdd/project/fdu-aidake-cfff/public/wangyanqing/NLPPJ/data/processed/semeval"
)


def read_split(path: Path, max_samples: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"id", "label", "text"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df[["id", "label", "text"]].copy()
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].fillna("").astype(str)
    if max_samples is not None and max_samples < len(df):
        per_label = max_samples // df["label"].nunique()
        remainder = max_samples - per_label * df["label"].nunique()
        sampled = []
        for label, group in df.groupby("label", sort=True):
            take = min(len(group), per_label + (1 if remainder > 0 else 0))
            sampled.append(group.sample(n=take, random_state=42))
            remainder = max(0, remainder - 1)
        df = pd.concat(sampled).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train_file", default="semeval_train_full.csv")
    parser.add_argument("--dev_file", default="semeval_dev_full.csv")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/tfidf"))
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_features", type=int, default=300_000)
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--c", type=float, default=4.0)
    args = parser.parse_args()

    train_df = read_split(args.data_dir / args.train_file, args.max_train_samples)
    dev_df = read_split(args.data_dir / args.dev_file, args.max_eval_samples)

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    analyzer="word",
                    ngram_range=(1, args.ngram_max),
                    min_df=2,
                    max_df=0.95,
                    max_features=args.max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=args.c,
                    max_iter=1000,
                    n_jobs=-1,
                    solver="saga",
                    random_state=42,
                    verbose=1,
                ),
            ),
        ]
    )

    pipe.fit(train_df["text"], train_df["label"])
    pred = pipe.predict(dev_df["text"])

    metrics = {
        "accuracy": accuracy_score(dev_df["label"], pred),
        "macro_f1": f1_score(dev_df["label"], pred, average="macro"),
        "micro_f1": f1_score(dev_df["label"], pred, average="micro"),
        "train_rows": int(len(train_df)),
        "dev_rows": int(len(dev_df)),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": dev_df["id"], "label": pred}).to_csv(
        args.output_dir / "dev_predictions.csv", index=False
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    joblib.dump(pipe, args.output_dir / "model.joblib")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
