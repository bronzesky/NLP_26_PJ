#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
.conda/bin/python scripts/train_transformer_baseline.py \
  --model_name roberta-base \
  --max_train_samples 512 \
  --max_eval_samples 512 \
  --num_train_epochs 1 \
  --max_length 256 \
  --output_dir outputs/roberta_smoke \
  "$@"

