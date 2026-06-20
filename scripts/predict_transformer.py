import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import (  # noqa: E402
    build_prediction_frame,
    read_prediction_input,
    write_metrics_if_labeled,
    write_prediction_outputs,
)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length: int):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.encodings.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--include_text", action="store_true")
    args = parser.parse_args()

    df = read_prediction_input(args.input_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = TextDataset(df["text"], tokenizer, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    predictions = []
    probabilities = []
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            logits_np = logits.cpu().numpy()
            predictions.extend(np.argmax(probs, axis=-1).tolist())
            probabilities.append(probs)
            all_logits.append(logits_np)

    probabilities = np.concatenate(probabilities, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    pred_df = build_prediction_frame(
        df=df,
        pred=predictions,
        prob_human=probabilities[:, 0],
        prob_ai=probabilities[:, 1],
        logit_human=all_logits[:, 0],
        logit_ai=all_logits[:, 1],
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
