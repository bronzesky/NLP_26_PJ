#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
.conda/bin/python scripts/train_tfidf_baseline.py "$@"

