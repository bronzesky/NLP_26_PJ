from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.viz import (
    score_paragraphs,
    score_paragraphs_fusion,
    score_paragraphs_tfidf,
    split_paragraphs,
    write_heatmap_html,
    write_hierarchical_html,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["transformer", "tfidf", "fusion"], default="transformer")
    parser.add_argument("--model_dir", type=Path, default=None,
                        help="Transformer model dir, or fusion model dir when --model_type fusion")
    parser.add_argument("--model_file", type=Path, default=Path("outputs/tfidf/model.joblib"))
    parser.add_argument("--tfidf_model_file", type=Path, default=Path("outputs/tfidf/model.joblib"),
                        help="TF-IDF model file when --model_type fusion")
    parser.add_argument("--roberta_model_dir", type=Path, default=None,
                        help="RoBERTa model dir when --model_type fusion")
    parser.add_argument("--text", default=None)
    parser.add_argument("--input_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/paragraph_heatmap"))
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    if not args.text and not args.input_file:
        raise ValueError("Provide --text or --input_file.")
    text = args.text if args.text is not None else args.input_file.read_text(encoding="utf-8")
    paragraphs = split_paragraphs(text)

    if args.model_type == "transformer":
        if args.model_dir is None:
            raise ValueError("Provide --model_dir when --model_type transformer.")
        scores = score_paragraphs(
            paragraphs,
            model_dir=args.model_dir,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
    elif args.model_type == "fusion":
        if args.model_dir is None:
            raise ValueError("Provide --model_dir (fusion model dir) when --model_type fusion.")
        if args.roberta_model_dir is None:
            raise ValueError("Provide --roberta_model_dir when --model_type fusion.")
        scores = score_paragraphs_fusion(
            paragraphs,
            fusion_model_dir=args.model_dir,
            tfidf_model_file=args.tfidf_model_file,
            roberta_model_dir=args.roberta_model_dir,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
    else:
        scores = score_paragraphs_tfidf(paragraphs, model_file=args.model_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = args.output_dir / "paragraph_scores.csv"
    html_file = args.output_dir / "paragraph_heatmap.html"
    scores.to_csv(csv_file, index=False)

    if args.model_type == "fusion":
        write_hierarchical_html(scores, html_file)
    else:
        write_heatmap_html(scores, html_file)

    print(f"Wrote {len(scores)} paragraph scores to {csv_file}")
    print(f"Wrote heatmap to {html_file}")


if __name__ == "__main__":
    main()
