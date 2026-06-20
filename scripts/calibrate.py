from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration import TemperatureScaler
from src.metrics import binary_metrics, reliability_bins, softmax


REQUIRED_COLUMNS = {"label", "logit_human", "logit_ai"}


def read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df.copy()
    df["label"] = df["label"].astype(int)
    df["logit_human"] = df["logit_human"].astype(float)
    df["logit_ai"] = df["logit_ai"].astype(float)
    return df


def logits_from_frame(df: pd.DataFrame):
    return df[["logit_human", "logit_ai"]].to_numpy(dtype=float)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def metrics_row(split: str, stage: str, labels, probs, n_bins: int) -> dict:
    row = {"split": split, "stage": stage}
    row.update(binary_metrics(labels, probs, n_bins=n_bins))
    return row


def reliability_frame(split: str, stage: str, labels, probs, n_bins: int) -> pd.DataFrame:
    rows = []
    for item in reliability_bins(labels, probs, n_bins=n_bins):
        row = asdict(item)
        row["split"] = split
        row["stage"] = stage
        rows.append(row)
    return pd.DataFrame(rows)


def maybe_plot_reliability(reliability: pd.DataFrame, output_dir: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    for (split, stage), group in reliability.groupby(["split", "stage"], sort=True):
        non_empty = group[group["count"] > 0]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
        ax.bar(
            non_empty["confidence"],
            non_empty["accuracy"],
            width=1.0 / max(len(group), 1),
            alpha=0.75,
            align="center",
            edgecolor="black",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{split} {stage} reliability")
        fig.tight_layout()
        fig.savefig(output_dir / f"reliability_{split}_{stage}.png", dpi=160)
        plt.close(fig)
    return True


def _svg_reliability(group: pd.DataFrame, title: str) -> str:
    width = 420
    height = 360
    left = 54
    top = 28
    plot = 280
    axis_bottom = top + plot
    non_empty = group[group["count"] > 0].copy()
    max_count = max(int(non_empty["count"].max()), 1) if len(non_empty) else 1
    bar_width = max(plot / max(len(group), 1) * 0.78, 2)

    bars = []
    for _, row in non_empty.iterrows():
        conf = float(row["confidence"])
        acc = float(row["accuracy"])
        count = int(row["count"])
        x = left + conf * plot - bar_width / 2
        y = axis_bottom - acc * plot
        h = acc * plot
        opacity = 0.35 + 0.55 * (count / max_count)
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{h:.2f}" '
            f'fill="#4f7cac" opacity="{opacity:.3f}"><title>'
            f'confidence={conf:.3f}, accuracy={acc:.3f}, count={count}</title></rect>'
        )

    ticks = []
    for value in [0, 0.25, 0.5, 0.75, 1.0]:
        x = left + value * plot
        y = axis_bottom - value * plot
        ticks.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{axis_bottom}" y2="{axis_bottom + 5}" stroke="#555"/>')
        ticks.append(f'<text x="{x:.1f}" y="{axis_bottom + 20}" text-anchor="middle">{value:g}</text>')
        ticks.append(f'<line x1="{left - 5}" x2="{left}" y1="{y:.1f}" y2="{y:.1f}" stroke="#555"/>')
        ticks.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{value:g}</text>')

    return f"""
<section class="chart">
  <h2>{html.escape(title)}</h2>
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)} reliability diagram">
    <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>
    <line x1="{left}" y1="{axis_bottom}" x2="{left + plot}" y2="{axis_bottom}" stroke="#222"/>
    <line x1="{left}" y1="{top}" x2="{left}" y2="{axis_bottom}" stroke="#222"/>
    <line x1="{left}" y1="{axis_bottom}" x2="{left + plot}" y2="{top}" stroke="#888" stroke-dasharray="5 5"/>
    {''.join(ticks)}
    {''.join(bars)}
    <text x="{left + plot / 2}" y="{height - 12}" text-anchor="middle">Confidence</text>
    <text x="16" y="{top + plot / 2}" transform="rotate(-90 16 {top + plot / 2})" text-anchor="middle">Accuracy</text>
  </svg>
</section>
"""


def write_reliability_html(reliability: pd.DataFrame, output_path: Path) -> None:
    sections = []
    for (split, stage), group in reliability.groupby(["split", "stage"], sort=True):
        sections.append(_svg_reliability(group, f"{split} {stage}"))

    output_path.write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Reliability Diagrams</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; }
    h1 { font-size: 24px; margin: 0 0 16px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 20px; }
    .chart { border: 1px solid #d6dde6; border-radius: 8px; padding: 14px; background: #fff; }
    .chart h2 { font-size: 16px; margin: 0 0 8px; }
    svg { width: 100%; height: auto; }
    text { font-size: 12px; fill: #344054; }
  </style>
</head>
<body>
  <h1>Reliability Diagrams</h1>
  <div class="grid">
"""
        + "\n".join(sections)
        + """
  </div>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_calibrated_predictions(path: Path, df: pd.DataFrame, before_probs, after_probs) -> None:
    out = df.copy()
    out["prob_ai_before"] = before_probs
    out["prob_ai_after"] = after_probs
    out["pred_before"] = (out["prob_ai_before"] >= 0.5).astype(int)
    out["pred_after"] = (out["prob_ai_after"] >= 0.5).astype(int)
    out.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit temperature scaling and report calibration metrics.")
    parser.add_argument("--dev_predictions", type=Path, required=True)
    parser.add_argument("--test_predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--n_bins", type=int, default=15)
    parser.add_argument("--min_temperature", type=float, default=0.05)
    parser.add_argument("--max_temperature", type=float, default=20.0)
    args = parser.parse_args()

    dev_df = read_predictions(args.dev_predictions)
    test_df = read_predictions(args.test_predictions)
    dev_logits = logits_from_frame(dev_df)
    test_logits = logits_from_frame(test_df)

    scaler = TemperatureScaler(
        min_temperature=args.min_temperature,
        max_temperature=args.max_temperature,
    ).fit(dev_logits, dev_df["label"].to_numpy())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "temperature.json",
        {
            "temperature": scaler.temperature,
            "fit_split": "dev",
            "dev_predictions": str(args.dev_predictions),
            "test_predictions": str(args.test_predictions),
        },
    )

    dev_before = softmax(dev_logits)[:, 1]
    dev_after = scaler.predict_positive_proba(dev_logits)
    test_before = softmax(test_logits)[:, 1]
    test_after = scaler.predict_positive_proba(test_logits)

    metrics = pd.DataFrame(
        [
            metrics_row("dev", "before", dev_df["label"], dev_before, args.n_bins),
            metrics_row("dev", "after", dev_df["label"], dev_after, args.n_bins),
            metrics_row("test", "before", test_df["label"], test_before, args.n_bins),
            metrics_row("test", "after", test_df["label"], test_after, args.n_bins),
        ]
    )
    metrics.to_csv(args.output_dir / "metrics_before_after.csv", index=False)

    reliability = pd.concat(
        [
            reliability_frame("dev", "before", dev_df["label"], dev_before, args.n_bins),
            reliability_frame("dev", "after", dev_df["label"], dev_after, args.n_bins),
            reliability_frame("test", "before", test_df["label"], test_before, args.n_bins),
            reliability_frame("test", "after", test_df["label"], test_after, args.n_bins),
        ],
        ignore_index=True,
    )
    reliability.to_csv(args.output_dir / "reliability.csv", index=False)
    plotted = maybe_plot_reliability(reliability, args.output_dir)
    reliability_html = args.output_dir / "reliability.html"
    write_reliability_html(reliability, reliability_html)

    write_calibrated_predictions(
        args.output_dir / "dev_calibrated_predictions.csv",
        dev_df,
        dev_before,
        dev_after,
    )
    write_calibrated_predictions(
        args.output_dir / "test_calibrated_predictions.csv",
        test_df,
        test_before,
        test_after,
    )

    print(
        json.dumps(
            {
                "temperature": scaler.temperature,
                "metrics": str(args.output_dir / "metrics_before_after.csv"),
                "reliability": str(args.output_dir / "reliability.csv"),
                "reliability_html": str(reliability_html),
                "plots": plotted,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
