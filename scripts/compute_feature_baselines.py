"""
scripts/compute_feature_baselines.py

Compute per-feature human/AI mean, std, and 33/67 percentile buckets
from M4 training data. Used for MTL label generation and feature comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features_v2 import full_features

DEFAULT_TRAIN = Path(
    "/inspire/hdd/project/fdu-aidake-cfff/public/wangyanqing/NLPPJ"
    "/data/processed/semeval/semeval_train_full.csv"
)

# The 3 features used for MTL auxiliary tasks
MTL_FEATURES = ["discourse_total_density", "contraction_ratio", "first_person_ratio"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output_file", type=Path,
                        default=Path("data/feature_baselines.json"))
    parser.add_argument("--n_per_label", type=int, default=5000,
                        help="Samples per label class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.train_data}...")
    df = pd.read_csv(args.train_data, encoding="utf-8-sig")
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    # Sample balanced
    ai = df[df["label"] == 1].sample(
        n=min(args.n_per_label, (df["label"] == 1).sum()), random_state=args.seed
    )
    human = df[df["label"] == 0].sample(
        n=min(args.n_per_label, (df["label"] == 0).sum()), random_state=args.seed
    )
    sample = pd.concat([ai, human]).reset_index(drop=True)
    print(f"Computing features for {len(sample)} samples (this may take ~15 min)...")

    all_features: list[dict] = []
    for i, row in sample.iterrows():
        if i % 500 == 0:
            print(f"  {i}/{len(sample)}...")
        try:
            feats = full_features(str(row["text"]))
            feats["label"] = int(row["label"])
            all_features.append(feats)
        except Exception as e:
            print(f"  Warning: failed on row {i}: {e}")

    feat_df = pd.DataFrame(all_features)
    feat_names = [c for c in feat_df.columns if c != "label"]

    baselines: dict = {}
    for feat in feat_names:
        vals = feat_df[feat].dropna()
        ai_vals = feat_df[feat_df["label"] == 1][feat].dropna()
        hu_vals = feat_df[feat_df["label"] == 0][feat].dropna()

        p33 = float(np.percentile(vals, 33))
        p67 = float(np.percentile(vals, 67))

        baselines[feat] = {
            "human_mean": float(hu_vals.mean()) if len(hu_vals) > 0 else 0.0,
            "human_std": float(hu_vals.std()) if len(hu_vals) > 0 else 0.0,
            "ai_mean": float(ai_vals.mean()) if len(ai_vals) > 0 else 0.0,
            "ai_std": float(ai_vals.std()) if len(ai_vals) > 0 else 0.0,
            "p33": p33,
            "p67": p67,
        }

    # Add MTL bucket boundaries (used in training)
    baselines["_mtl_features"] = MTL_FEATURES
    baselines["_mtl_boundaries"] = {
        feat: {
            "p33": baselines[feat]["p33"],
            "p67": baselines[feat]["p67"],
        }
        for feat in MTL_FEATURES if feat in baselines
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(baselines, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved baselines for {len(feat_names)} features to {args.output_file}")

    # Print key features for verification
    print("\nKey feature summary:")
    for feat in MTL_FEATURES + ["sentence_length_cv", "mattr", "passive_ratio"]:
        if feat in baselines:
            b = baselines[feat]
            print(f"  {feat}: human={b['human_mean']:.4f}±{b['human_std']:.4f}  "
                  f"ai={b['ai_mean']:.4f}±{b['ai_std']:.4f}  "
                  f"p33={b['p33']:.4f}  p67={b['p67']:.4f}")


if __name__ == "__main__":
    main()
