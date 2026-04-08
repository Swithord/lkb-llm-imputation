#!/bin/bash
# Cluster defaults in this script match the lab's current setup:
# - account: rrg-annielee
# - GPU:     h100:1
# - modules: python + cuda
# Override Slurm resources at submit time, e.g.
#   sbatch --account=<acct> --gres=gpu:a100:1 --mem=80G scripts/run_eval.sbatch
#SBATCH --job-name=llm-impute-uriel
#SBATCH --account=rrg-annielee
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=artifacts/logs/%x-%j.out
#SBATCH --error=artifacts/logs/%x-%j.err

set -euo pipefail

MODE="${MODE:-all}"   # benchmark | full | all
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
GOLD="${GOLD:-data/benchmark/gold_eval_2.jsonl}"
OUT_DIR="${OUT_DIR:-artifacts/prediction}"
PROMPT_VERSION="${PROMPT_VERSION:-v3_strict_json}"
PYTHON_MODULE="${PYTHON_MODULE:-python}"
CUDA_MODULE="${CUDA_MODULE:-cuda}"
EXTRA_MODULES="${EXTRA_MODULES:-}"
VENV_PATH="${VENV_PATH:-.venv}"
INFER_DEVICE="${INFER_DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p artifacts/logs "$OUT_DIR"

if [[ -n "$PYTHON_MODULE" ]]; then
  module load "$PYTHON_MODULE"
fi
if [[ -n "$CUDA_MODULE" ]]; then
  module load "$CUDA_MODULE" 2>/dev/null || true
fi
for mod in $EXTRA_MODULES; do
  module load "$mod"
done
if [[ -f "$VENV_PATH/bin/activate" ]]; then
  source "$VENV_PATH/bin/activate"
else
  echo "No virtualenv found at $VENV_PATH"
  exit 1
fi

# Debug checks
echo "PWD: $(pwd)"
echo "HOST: $(hostname)"
echo "which python: $(which python)"
python --version
python -c "import pandas; print('pandas ok')"
python -c "import torch; print('torch ok')"

nvidia-smi

export HF_HOME="${SCRATCH:-/tmp}/hf"
export TRANSFORMERS_CACHE="${SCRATCH:-/tmp}/hf/transformers"
export HF_DATASETS_CACHE="${SCRATCH:-/tmp}/hf/datasets"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TF=1

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" artifacts/logs "$OUT_DIR"

run_benchmark() {
  NO_VOTE_PROMPTS="$OUT_DIR/prompts_eval_${PROMPT_VERSION}_novote.jsonl"
  VOTE_PROMPTS="$OUT_DIR/prompts_eval_${PROMPT_VERSION}_vote.jsonl"
  NO_VOTE_PREDS="$OUT_DIR/predictions_eval_${PROMPT_VERSION}_novote.jsonl"
  VOTE_PREDS="$OUT_DIR/predictions_eval_${PROMPT_VERSION}_vote.jsonl"
  NO_VOTE_REPORT="$OUT_DIR/report_eval_${PROMPT_VERSION}_novote.json"
  VOTE_REPORT="$OUT_DIR/report_eval_${PROMPT_VERSION}_vote.json"

  python benchmark/build_prompts_from_gold.py \
    --prompting_py code/prompting.py \
    --typ data/derived/uriel+_typological.csv \
    --meta data/derived/metadata.csv \
    --topk_csv data/features/topk_per_feature.csv \
    --gen data/derived/genetic_neighbours.json \
    --geo data/derived/geographic_neighbours.json \
    --gold "$GOLD" \
    --prompt_version "$PROMPT_VERSION" \
    --no-include_vote_table \
    --out "$NO_VOTE_PROMPTS"

  python benchmark/build_prompts_from_gold.py \
    --prompting_py code/prompting.py \
    --typ data/derived/uriel+_typological.csv \
    --meta data/derived/metadata.csv \
    --topk_csv data/features/topk_per_feature.csv \
    --gen data/derived/genetic_neighbours.json \
    --geo data/derived/geographic_neighbours.json \
    --gold "$GOLD" \
    --prompt_version "$PROMPT_VERSION" \
    --include_vote_table \
    --out "$VOTE_PROMPTS"

  python benchmark/infer_from_prompts.py \
    --in "$NO_VOTE_PROMPTS" \
    --out "$NO_VOTE_PREDS" \
    --model "$MODEL" \
    --device "$INFER_DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --no-strengthen_json_contract \
    --stop_at_closing_brace

  python benchmark/infer_from_prompts.py \
    --in "$VOTE_PROMPTS" \
    --out "$VOTE_PREDS" \
    --model "$MODEL" \
    --device "$INFER_DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --no-strengthen_json_contract \
    --stop_at_closing_brace

  python benchmark/evaluate_benchmark.py \
    --gold "$GOLD" \
    --pred "$NO_VOTE_PREDS" \
    --report_out "$NO_VOTE_REPORT"

  python benchmark/evaluate_benchmark.py \
    --gold "$GOLD" \
    --pred "$VOTE_PREDS" \
    --report_out "$VOTE_REPORT"

  if [[ -f artifacts/benchmark/predictions_eval_2.jsonl ]]; then
    python benchmark/evaluate_benchmark.py \
      --gold "$GOLD" \
      --pred artifacts/benchmark/predictions_eval_2.jsonl \
      --report_out "$OUT_DIR/report_eval_baseline_from_existing.json"

    python benchmark/compare_prompt_reports.py \
      --baseline_report "$OUT_DIR/report_eval_baseline_from_existing.json" \
      --v3_no_vote_report "$NO_VOTE_REPORT" \
      --v3_vote_report "$VOTE_REPORT" \
      --prompt_version "$PROMPT_VERSION" \
      --out "$OUT_DIR/report_prompt_${PROMPT_VERSION}.json"
  fi

  python benchmark/measure_stream_a_speed.py \
    --in "$VOTE_PROMPTS" \
    --model "$MODEL" \
    --device "$INFER_DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --report_out "$OUT_DIR/speed_report_${PROMPT_VERSION}.json"
}

run_full() {
  OUT_DIR_ENV="$OUT_DIR" python - <<'PY'
import json, os, pandas as pd
out_dir = os.environ["OUT_DIR_ENV"]
df = pd.read_csv("data/derived/uriel+_typological.csv", index_col=0)
langs = [str(x) for x in df.index]
path = f"{out_dir}/language_groups_all.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump({"high": langs, "low": []}, f, ensure_ascii=False, indent=2)
print(f"Wrote {path}")
PY

  python benchmark/eval_prompting.py \
    --prompting_py code/prompting.py \
    --typ data/derived/uriel+_typological.csv \
    --meta data/derived/metadata.csv \
    --topk_csv data/features/topk_per_feature.csv \
    --gen data/derived/genetic_neighbours.json \
    --geo data/derived/geographic_neighbours.json \
    --groups_json "$OUT_DIR/language_groups_all.json" \
    --prompt_version "$PROMPT_VERSION" \
    --include_vote_table \
    --model "$MODEL" \
    --device "$INFER_DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --prompts_out "$OUT_DIR/eval_missing_prompts_all_${PROMPT_VERSION}.jsonl" \
    --pred_out "$OUT_DIR/eval_missing_predictions_all_${PROMPT_VERSION}.jsonl"
}

case "$MODE" in
  benchmark) run_benchmark ;;
  full) run_full ;;
  all) run_benchmark; run_full ;;
  *) echo "MODE must be benchmark|full|all"; exit 1 ;;
esac
