# Region-Aware AI-Generated Text Detection

Robust, calibrated, and interpretable machine-generated text detection on
**SemEval-2024 Task 8 Subtask A (English)**, under the benchmark's realistic
distribution shift (unseen generators + unseen domain at test time).

> **TL;DR.** A fine-tuned RoBERTa reaches test AUROC **0.944** but only **0.62**
> macro-F1 at the default threshold — the gap is a *mis-placed operating point*,
> not a representation failure. Calibration + an in-distribution threshold
> recover **0.886**. A **region-aware two-stage** rule (confident-margin fast
> path + orthogonal-feature classifier for the ambiguous band) reaches a
> deployable **0.857**, beating the prior best single model (0.835), and lifts
> the hardest human-like generator (`bloomz`) recall from **0.10 → 0.70**.

## What's here

- **Detector** (`src/detector_pipeline.py`): text → calibrated document P(AI),
  region-aware verdict, paragraph/sentence breakdown, occlusion attribution,
  23-dim linguistic feature evidence, perplexity + burstiness.
- **Interpretable report** (`scripts/render_report.py`): single-file
  plagiarism-checker-style HTML — ring gauge, occlusion sentence highlight,
  per-feature human↔AI discriminant axis, de-AI suggestions.
- **De-AI robustness** (`scripts/humanize.py`, `cross_model_attack.py`):
  local Qwen3-8B or external API rewriting + closed-loop re-detection.
- **Paper** (`paper/main.tex`): ACL-format writeup.

## Reproduce

```bash
# 0. environment (conda env at .conda/)
.conda/bin/python --version   # 3.10

# 1. baselines + calibration ceiling (uses stored predictions, no GPU)
.conda/bin/python scripts/phaseA_report.py        # operating-point ceiling
.conda/bin/python scripts/phaseB2a_strict.py      # region-aware, P0/P1/P2

# 2. freeze the deployable region-aware model (fit on dev)
.conda/bin/python scripts/fit_deploy_model.py     # -> outputs/region_aware/deploy_model.json

# 3. analyze arbitrary text + render report
.conda/bin/python src/detector_pipeline.py --text_file demo_text.txt --title demo
.conda/bin/python scripts/render_report.py --result outputs/pipeline_demo/result.json

# 4. gallery on real test samples
.conda/bin/python scripts/demo_reports.py         # -> outputs/report_gallery/*.html

# 5. de-AI robustness (needs an LLM; see notes)
.conda/bin/python scripts/cross_model_attack.py   # -> outputs/robustness/
```

## Key results

| Model | base F1 | calibrated F1 | AUROC |
|---|---|---|---|
| RoBERTa single | 0.619 | **0.886** | 0.944 |
| TF-IDF + LR | 0.828 | 0.830 | 0.897 |
| Fusion LGBM | 0.835 | 0.882 | 0.930 |
| **Region-aware (P1 deploy)** | — | **0.857** | — |

See `docs/` for the full diagnosis (`phaseA_operating_point.md`), method
(`phaseB2_region_aware.md`), robustness (`phaseC3_robustness.md`), visualization
corrections (`viz_corrections.md`), and the HSAD negative result
(`hsad_failure.md`).

## Notes

- The detector model (`outputs/roberta_base/best_model`) is a standard fine-tune;
  all region-aware decision parameters are fit on dev only (leakage-free).
- Perplexity uses GPT-2 (auto-downloaded via HF mirror).
- The de-AI component is a **detector-robustness probe**, not a tool to evade
  academic-integrity systems.
