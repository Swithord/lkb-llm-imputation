# Benchmark Toolkit

This folder contains utilities to build and evaluate a high-resource vs low-resource inference benchmark without changing `code/prompting.py`.

## 1) Build benchmark prompts + gold labels

```bash
.venv/bin/python benchmark/build_benchmark.py \
  --prompting_py code/prompting.py \
  --typ output/uriel+_typological.csv \
  --meta output/metadata.csv \
  --topk_csv out_corr/topk_per_feature.csv \
  --gen output/genetic_neighbours.json \
  --geo output/geographic_neighbours.json \
  --high_n 50 \
  --low_n 50 \
  --per_language 20 \
  --top_n 10 \
  --prompts_out benchmark/prompts_eval.jsonl \
  --prompts_high_out benchmark/prompts_eval_high.jsonl \
  --prompts_low_out benchmark/prompts_eval_low.jsonl \
  --gold_out benchmark/gold_eval.jsonl \
  --groups_out benchmark/language_groups.json
```

Outputs:
- `benchmark/prompts_eval.jsonl`: prompts with strict JSON output contract (`value`, `confidence`, `rationale`).
- `benchmark/prompts_eval_high.jsonl`: prompts for high-resource languages only.
- `benchmark/prompts_eval_low.jsonl`: prompts for low-resource languages only.
- `benchmark/gold_eval.jsonl`: gold labels for masked observed cells.
- `benchmark/language_groups.json`: selected high/low language sets.

## 2) Run model inference

Use your preferred inference runner. Make sure prediction records include `id` and either:
- direct fields: `value`, `confidence`, `rationale`, or
- field `output` containing a JSON object string with those keys.

Example with the included runner:

```bash
.venv/bin/python benchmark/infer_from_prompts.py \
  --in benchmark/prompts_eval_high.jsonl \
  --out benchmark/predictions_eval_high.jsonl \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --device cuda \
  --batch_size 4

.venv/bin/python benchmark/infer_from_prompts.py \
  --in benchmark/prompts_eval_low.jsonl \
  --out benchmark/predictions_eval_low.jsonl \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --device cuda \
  --batch_size 4
```

## 3) Evaluate predictions

```bash
.venv/bin/python benchmark/evaluate_benchmark.py \
  --gold benchmark/gold_eval.jsonl \
  --pred benchmark/predictions_eval.jsonl \
  --report_out benchmark/report_eval.json
```

Metrics include:
- accuracy (overall/high/low),
- parsed rate,
- high-confidence accuracy + coverage,
- rationale quality rate (non-empty and <= 30 words),
- calibration (`brier_on_correctness`, `ece_10bin`).

The evaluator reports both:
- `normalized`: metrics on normalized prediction fields (`value/confidence/rationale`).
- `strict_raw`: metrics parsed directly from model raw text in `output`.

## 3b) Fit confidence calibration map

```bash
.venv/bin/python benchmark/fit_confidence_map.py \
  --gold benchmark/gold_eval.jsonl \
  --pred benchmark/predictions_eval.jsonl \
  --mode normalized \
  --out benchmark/confidence_map.json
```

Then re-evaluate with calibrated confidence:

```bash
.venv/bin/python benchmark/evaluate_benchmark.py \
  --gold benchmark/gold_eval.jsonl \
  --pred benchmark/predictions_eval.jsonl \
  --confidence_map benchmark/confidence_map.json \
  --report_out benchmark/report_eval_calibrated.json
```

No-leakage option (fit on train split, evaluate on held-out split):

```bash
.venv/bin/python benchmark/split_calibration_eval.py \
  --gold benchmark/gold_eval.jsonl \
  --pred benchmark/predictions_eval.jsonl \
  --mode normalized \
  --split_by language \
  --test_frac 0.2 \
  --seed 7 \
  --out benchmark/split_calibration_report.json
```

## 3c) Per-feature error analysis

```bash
.venv/bin/python benchmark/feature_error_analysis.py \
  --gold benchmark/gold_eval.jsonl \
  --pred benchmark/predictions_eval.jsonl \
  --mode normalized \
  --min_n 10 \
  --out benchmark/feature_errors.json
```

## 4) Run all-missing evaluation prompts on GPU

This runs inference over all currently missing features for the evaluation languages (high/low split).

```bash
.venv/bin/python benchmark/eval_prompting.py \
  --prompting_py code/prompting.py \
  --typ output/uriel+_typological.csv \
  --meta output/metadata.csv \
  --topk_csv out_corr/topk_per_feature.csv \
  --gen output/genetic_neighbours.json \
  --geo output/geographic_neighbours.json \
  --groups_json benchmark/language_groups.json \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --device cuda \
  --batch_size 4 \
  --max_new_tokens 96 \
  --prompts_out benchmark/eval_missing_prompts.jsonl \
  --pred_out benchmark/eval_missing_predictions.jsonl
```
