# Test Reproduction Notes

The processed train/dev data directory only contains:

- `semeval_train_full.csv`
- `semeval_dev_full.csv`

No official test split is present under:

`/inspire/hdd/project/fdu-aidake-cfff/public/wangyanqing/NLPPJ/data/processed/semeval`

The English test split has been downloaded from a public HuggingFace mirror of
the official test data and copied to:

`data/official/test_sets/subtaskA_monolingual.jsonl`

This file has `34272` rows and includes gold `label`, so it can be evaluated
locally on CFFF.

## Official Format

For SemEval-2024 Task 8 Subtask A, the official submission/prediction format is
JSONL with one object per line:

```json
{"id": "example-id", "label": 1}
```

Labels:

- `0`: human
- `1`: machine-generated

## Predict On Test

After full training finishes, use the saved best model:

```bash
cd /inspire/hdd/project/fdu-aidake-cfff/public/hanz/semeval2024_task8a_en_baseline
.conda/bin/python scripts/predict_transformer.py \
  --model_dir outputs/roberta_base/best_model \
  --input_file data/official/test_sets/subtaskA_monolingual.jsonl \
  --output_dir outputs/roberta_base_test
```

The script accepts either `.jsonl` or `.csv` input, as long as the input has
`id` and `text` columns/fields.

Outputs:

- `predictions.jsonl`: official-style `id,label` file
- `predictions.csv`: includes `prob_machine` for analysis
- `metrics.json`: written only if the input file includes gold `label`

## Dev Prediction Check

The same script can verify a trained model on the dev CSV:

```bash
cd /inspire/hdd/project/fdu-aidake-cfff/public/hanz/semeval2024_task8a_en_baseline
.conda/bin/python scripts/predict_transformer.py \
  --model_dir outputs/roberta_base/best_model \
  --input_file /inspire/hdd/project/fdu-aidake-cfff/public/wangyanqing/NLPPJ/data/processed/semeval/semeval_dev_full.csv \
  --output_dir outputs/roberta_base_dev_predict
```
