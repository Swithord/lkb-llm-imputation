#!/bin/bash
# Cluster defaults in this script match the lab's current setup:
# - account: rrg-annielee
# - GPU:     h100_3g.40gb:1
# - modules: python + cuda
# Override Slurm resources at submit time, e.g.
#   sbatch --account=<acct> --gpus=a100:1 --mem=80G scripts/run_eval.sbatch
#SBATCH --job-name=llm-impute-uriel
#SBATCH --account=rrg-annielee
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=artifacts/logs/%x-%j.out
#SBATCH --error=artifacts/logs/%x-%j.err

set -euo pipefail

MODE="${MODE:-all}"   # benchmark | full | all
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
GOLD="${GOLD:-data/benchmark/gold_eval_2.jsonl}"
OUT_DIR="${OUT_DIR:-artifacts/prediction}"
PROMPT_VERSION="${PROMPT_VERSION:-v4_strict_json}"
LEGACY_RUN_MODES="${LEGACY_RUN_MODES:-vote,novote}"   # comma-separated: vote,novote
PYTHON_MODULE="${PYTHON_MODULE:-python}"
CUDA_MODULE="${CUDA_MODULE:-cuda}"
EXTRA_MODULES="${EXTRA_MODULES:-}"
VENV_PATH="${VENV_PATH:-.venv}"
INFER_DEVICE="${INFER_DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

mode_enabled() {
  local mode="$1"
  local csv="$2"
  case ",$csv," in
    *",$mode,"*) return 0 ;;
    *) return 1 ;;
  esac
}

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
echo "Gold file: $GOLD"
echo "Legacy run modes: $LEGACY_RUN_MODES"
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

  if mode_enabled "novote" "$LEGACY_RUN_MODES"; then
    python -m lkb.scripts.build_prompts \
      --gold "$GOLD" \
      --prompt icl \
      --version "$PROMPT_VERSION" \
      --no-include_vote_table \
      --out "$NO_VOTE_PROMPTS"
  fi

  if mode_enabled "vote" "$LEGACY_RUN_MODES"; then
    python -m lkb.scripts.build_prompts \
      --gold "$GOLD" \
      --prompt icl \
      --version "$PROMPT_VERSION" \
      --include_vote_table \
      --out "$VOTE_PROMPTS"
  fi

  if mode_enabled "novote" "$LEGACY_RUN_MODES"; then
    python -m lkb.scripts.infer \
      --in "$NO_VOTE_PROMPTS" \
      --out "$NO_VOTE_PREDS" \
      --model "$MODEL" \
      --device "$INFER_DEVICE" \
      --batch_size "$BATCH_SIZE" \
      --max_new_tokens "$MAX_NEW_TOKENS"
  fi

  if mode_enabled "vote" "$LEGACY_RUN_MODES"; then
    python -m lkb.scripts.infer \
      --in "$VOTE_PROMPTS" \
      --out "$VOTE_PREDS" \
      --model "$MODEL" \
      --device "$INFER_DEVICE" \
      --batch_size "$BATCH_SIZE" \
      --max_new_tokens "$MAX_NEW_TOKENS"
  fi

  if mode_enabled "novote" "$LEGACY_RUN_MODES"; then
    python -m lkb.scripts.evaluate \
      --gold "$GOLD" \
      --pred "$NO_VOTE_PREDS" \
      --report_out "$NO_VOTE_REPORT"
  fi

  if mode_enabled "vote" "$LEGACY_RUN_MODES"; then
    python -m lkb.scripts.evaluate \
      --gold "$GOLD" \
      --pred "$VOTE_PREDS" \
      --report_out "$VOTE_REPORT"
  fi

  if [[ -f "$VOTE_PROMPTS" ]]; then
    python -m lkb.scripts.speed_bench \
      --in "$VOTE_PROMPTS" \
      --model "$MODEL" \
      --device "$INFER_DEVICE" \
      --batch_size "$BATCH_SIZE" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --report_out "$OUT_DIR/speed_report_${PROMPT_VERSION}.json"
  elif [[ -f "$NO_VOTE_PROMPTS" ]]; then
    python -m lkb.scripts.speed_bench \
      --in "$NO_VOTE_PROMPTS" \
      --model "$MODEL" \
      --device "$INFER_DEVICE" \
      --batch_size "$BATCH_SIZE" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --report_out "$OUT_DIR/speed_report_${PROMPT_VERSION}.json"
  fi
}

run_full() {
  python -m lkb.scripts.build_prompts \
    --gold "$GOLD" \
    --prompt icl \
    --version "$PROMPT_VERSION" \
    --include_vote_table \
    --out "$OUT_DIR/eval_missing_prompts_all_${PROMPT_VERSION}.jsonl"

  python -m lkb.scripts.infer \
    --in "$OUT_DIR/eval_missing_prompts_all_${PROMPT_VERSION}.jsonl" \
    --out "$OUT_DIR/eval_missing_predictions_all_${PROMPT_VERSION}.jsonl" \
    --model "$MODEL" \
    --device "$INFER_DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS"
}

case "$MODE" in
  benchmark) run_benchmark ;;
  full) run_full ;;
  all) run_benchmark; run_full ;;
  *) echo "MODE must be benchmark|full|all"; exit 1 ;;
esac
