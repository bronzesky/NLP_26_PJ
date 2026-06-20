from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_JSON_PATTERNS = ("metrics.json", "eval_metrics.json", "*metrics*.json")
DEFAULT_CSV_NAMES = (
    "group_by_model.csv",
    "group_by_length_bucket.csv",
    "group_by_domain.csv",
    "group_by_source.csv",
    "group_by_confidence_bucket.csv",
    "group_metrics.csv",
    "metrics_before_after.csv",
    "reliability.csv",
)
PRIMARY_METRICS = (
    "accuracy",
    "macro_f1",
    "micro_f1",
    "auroc",
    "fpr_at_95_tpr",
    "ece",
    "brier",
    "nll",
)
COUNT_METRICS = ("rows", "train_rows", "dev_rows", "eval_samples_per_second")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize model metrics JSON files and optional analysis CSVs into Markdown."
    )
    parser.add_argument(
        "--results_dirs",
        "--input_dirs",
        dest="results_dirs",
        type=Path,
        nargs="*",
        default=[],
        help="Directories to scan for common metrics JSON and analysis CSV files.",
    )
    parser.add_argument(
        "--metrics_files",
        type=Path,
        nargs="*",
        default=[],
        help="Explicit metrics JSON files to include.",
    )
    parser.add_argument(
        "--csv_files",
        type=Path,
        nargs="*",
        default=[],
        help="Optional analysis/calibration CSV files to include.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Markdown output path.",
    )
    parser.add_argument(
        "--max_csv_rows",
        type=int,
        default=12,
        help="Maximum rows shown per CSV table.",
    )
    args = parser.parse_args()

    metrics_files = unique_paths([*args.metrics_files, *discover_metrics(args.results_dirs)])
    csv_files = unique_paths([*args.csv_files, *discover_csvs(args.results_dirs)])

    runs = [read_metrics_file(path) for path in metrics_files]
    csv_summaries = [read_csv_summary(path, args.max_csv_rows) for path in csv_files]

    output = render_markdown(runs, csv_summaries, metrics_files, csv_files)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(output, encoding="utf-8")
    print(f"Wrote summary to {args.output_file}")


def discover_metrics(results_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for directory in results_dirs:
        if directory.is_file() and directory.suffix.lower() == ".json":
            paths.append(directory)
            continue
        if not directory.exists():
            continue
        for pattern in DEFAULT_JSON_PATTERNS:
            paths.extend(path for path in directory.rglob(pattern) if path.is_file())
    return paths


def discover_csvs(results_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for directory in results_dirs:
        if directory.is_file() and directory.suffix.lower() == ".csv":
            paths.append(directory)
            continue
        if not directory.exists():
            continue
        for name in DEFAULT_CSV_NAMES:
            paths.extend(path for path in directory.rglob(name) if path.is_file())
    return paths


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path.expanduser().resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def read_metrics_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")

    normalized = normalize_metrics(payload)
    return {
        "run": infer_run_name(path),
        "path": path,
        "metrics": normalized,
        "raw": payload,
    }


def normalize_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        clean_key = key[5:] if key.startswith("eval_") else key
        out[clean_key] = value
    return out


def infer_run_name(path: Path) -> str:
    parent = path.parent.name
    if parent and parent != ".":
        return parent
    return path.stem


def read_csv_summary(path: Path, max_rows: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames or []
    return {
        "path": path,
        "kind": classify_csv(path, columns),
        "columns": columns,
        "rows": rows,
        "display_rows": choose_csv_rows(path, rows, max_rows),
    }


def classify_csv(path: Path, columns: list[str]) -> str:
    name = path.name.lower()
    column_set = set(columns)
    if {"split", "stage", "ece"}.issubset(column_set):
        return "calibration_metrics"
    if {"bin", "confidence", "accuracy", "gap"}.issubset(column_set):
        return "reliability"
    if name.startswith("group_by_") or {"accuracy", "macro_f1"}.intersection(column_set):
        return "group_metrics"
    return "csv"


def choose_csv_rows(path: Path, rows: list[dict[str, str]], max_rows: int) -> list[dict[str, str]]:
    if max_rows <= 0:
        return []
    if len(rows) <= max_rows:
        return rows

    name = path.name.lower()
    if name == "group_by_length_bucket.csv":
        return rows[:max_rows]
    if "macro_f1" in rows[0] or "accuracy" in rows[0]:
        key = "macro_f1" if "macro_f1" in rows[0] else "accuracy"
        return sorted(rows, key=lambda row: parse_float(row.get(key)), reverse=True)[:max_rows]
    return rows[:max_rows]


def render_markdown(
    runs: list[dict[str, Any]],
    csv_summaries: list[dict[str, Any]],
    metrics_files: list[Path],
    csv_files: list[Path],
) -> str:
    lines: list[str] = [
        "# Model Comparison Summary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Inputs",
        "",
        f"- Metrics JSON files: {len(metrics_files)}",
        f"- CSV files: {len(csv_files)}",
        "",
    ]

    if runs:
        lines.extend(render_overall_metrics(runs))
    else:
        lines.extend(["## Overall Metrics", "", "No metrics JSON files were found.", ""])

    if csv_summaries:
        lines.extend(["## CSV Summaries", ""])
        for summary in csv_summaries:
            lines.extend(render_csv_summary(summary))
    else:
        lines.extend(["## CSV Summaries", "", "No CSV files were provided or discovered.", ""])

    lines.extend(render_file_list("Metrics Files", metrics_files))
    lines.extend(render_file_list("CSV Files", csv_files))
    return "\n".join(lines).rstrip() + "\n"


def render_overall_metrics(runs: list[dict[str, Any]]) -> list[str]:
    metric_keys = selected_metric_keys(runs)
    table_rows = []
    for run in sorted(runs, key=run_sort_key):
        row = {"run": run["run"]}
        for key in metric_keys:
            row[key] = format_value(run["metrics"].get(key))
        row["path"] = short_path(run["path"])
        table_rows.append(row)

    lines = ["## Overall Metrics", ""]
    lines.extend(markdown_table(["run", *metric_keys, "path"], table_rows))
    lines.append("")

    best_lines = best_metric_lines(runs)
    if best_lines:
        lines.extend(["### Best Runs", "", *best_lines, ""])
    return lines


def selected_metric_keys(runs: list[dict[str, Any]]) -> list[str]:
    available = {key for run in runs for key in run["metrics"].keys()}
    keys = [key for key in PRIMARY_METRICS if key in available]
    keys.extend(key for key in COUNT_METRICS if key in available and key not in keys)

    extras = sorted(
        key
        for key in available
        if key not in keys and is_scalar_metric(any_metric_value(runs, key))
    )
    return [*keys, *extras[:6]]


def any_metric_value(runs: list[dict[str, Any]], key: str) -> Any:
    for run in runs:
        if key in run["metrics"]:
            return run["metrics"][key]
    return None


def run_sort_key(run: dict[str, Any]) -> tuple[float, float, str]:
    metrics = run["metrics"]
    macro_f1 = parse_float(metrics.get("macro_f1"))
    accuracy = parse_float(metrics.get("accuracy"))
    return (-macro_f1 if not math.isnan(macro_f1) else math.inf, -accuracy if not math.isnan(accuracy) else math.inf, run["run"])


def best_metric_lines(runs: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for key in ("macro_f1", "accuracy", "auroc"):
        candidates = [
            (parse_float(run["metrics"].get(key)), run["run"])
            for run in runs
            if key in run["metrics"]
        ]
        candidates = [(value, run_name) for value, run_name in candidates if not math.isnan(value)]
        if not candidates:
            continue
        value, run_name = max(candidates, key=lambda item: item[0])
        lines.append(f"- Best `{key}`: `{run_name}` ({format_number(value)})")
    for key in ("ece", "nll", "brier", "fpr_at_95_tpr"):
        candidates = [
            (parse_float(run["metrics"].get(key)), run["run"])
            for run in runs
            if key in run["metrics"]
        ]
        candidates = [(value, run_name) for value, run_name in candidates if not math.isnan(value)]
        if not candidates:
            continue
        value, run_name = min(candidates, key=lambda item: item[0])
        lines.append(f"- Lowest `{key}`: `{run_name}` ({format_number(value)})")
    return lines


def render_csv_summary(summary: dict[str, Any]) -> list[str]:
    path = summary["path"]
    rows = summary["rows"]
    display_rows = summary["display_rows"]
    columns = display_columns(summary["columns"])

    lines = [
        f"### {path.name}",
        "",
        f"- Path: `{short_path(path)}`",
        f"- Rows: {len(rows)}",
        "",
    ]

    if summary["kind"] == "calibration_metrics":
        deltas = calibration_delta_lines(rows)
        if deltas:
            lines.extend(deltas)
            lines.append("")

    if display_rows and columns:
        lines.extend(markdown_table(columns, display_rows))
        lines.append("")
    elif not rows:
        lines.extend(["No rows found.", ""])
    return lines


def display_columns(columns: list[str]) -> list[str]:
    preferred = [
        "model",
        "length_bucket",
        "domain",
        "source",
        "confidence_bucket",
        "split",
        "stage",
        "rows",
        "count",
        "accuracy",
        "macro_f1",
        "micro_f1",
        "auroc",
        "ece",
        "brier",
        "nll",
        "fpr_at_95_tpr",
        "confidence",
        "gap",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    selected = [column for column in preferred if column in columns]
    if selected:
        return selected
    return columns[:8]


def calibration_delta_lines(rows: list[dict[str, str]]) -> list[str]:
    by_split: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        split = row.get("split", "")
        stage = row.get("stage", "")
        if split and stage:
            by_split.setdefault(split, {})[stage] = row

    lines: list[str] = []
    for split, stages in sorted(by_split.items()):
        before = stages.get("before")
        after = stages.get("after")
        if not before or not after:
            continue
        parts = []
        for metric in ("ece", "nll", "brier", "accuracy", "macro_f1"):
            if metric not in before or metric not in after:
                continue
            delta = parse_float(after.get(metric)) - parse_float(before.get(metric))
            if math.isnan(delta):
                continue
            sign = "+" if delta >= 0 else ""
            parts.append(f"`{metric}` {sign}{format_number(delta)}")
        if parts:
            lines.append(f"- `{split}` after-minus-before: {', '.join(parts)}")
    return lines


def render_file_list(title: str, paths: list[Path]) -> list[str]:
    lines = [f"## {title}", ""]
    if not paths:
        lines.extend(["None.", ""])
        return lines
    lines.extend(f"- `{short_path(path)}`" for path in paths)
    lines.append("")
    return lines


def markdown_table(columns: list[str], rows: list[dict[str, Any]]) -> list[str]:
    if not columns:
        return []
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [format_cell(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def is_scalar_metric(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def format_cell(value: Any) -> str:
    return escape_markdown(format_value(value))


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return format_number(value)
    text = str(value)
    number = parse_float(text)
    if not math.isnan(number) and text.strip() != "":
        return format_number(number)
    return text


def format_number(value: float | int) -> str:
    if isinstance(value, bool):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    if number.is_integer() and abs(number) >= 1:
        return str(int(number))
    return f"{number:.4f}".rstrip("0").rstrip(".")


def parse_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def short_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


if __name__ == "__main__":
    main()
