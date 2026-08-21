#!/bin/bash
# Rerun only the unparsed (value=null) cases from a set of finished
# ICL-prompt eval runs, then splice the retried predictions back in and
# refresh each report. One array task per (model, ablation config) combo
# that had unparsed predictions as of 2026-08-21:
#
#   0: base    Llama-3.1-70B-Instruct   (non-model-slugged files)
#   1: anchors Llama-3.1-70B-Instruct
#   2: base    Llama-3.1-8B-Instruct
#   3: anchors Llama-3.1-8B-Instruct
#   4: base    gemma-4-31b-it
#   5: anchors gemma-4-31b-it
#   6: geo     gemma-4-31b-it
#   7: phylo   gemma-4-31b-it
#   8: full    gemma-4-31b-it
#
# Cluster defaults mirror scripts/run_eval.sbatch, the script that
# originally produced these files (ICL prompts, not KG).
# Override Slurm resources at submit time, e.g.
#   sbatch --account=<acct> --gpus=h100:1 --mem=80G scripts/rerun_unparsed_benchmark.sbatch
#SBATCH --job-name=rerun-unparsed
#SBATCH --account=rrg-annielee
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --gpus=h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --array=0-8
#SBATCH --output=artifacts/logs/%x-%A_%a.out
#SBATCH --error=artifacts/logs/%x-%A_%a.err

set -euo pipefail

GOLD="${GOLD:-data/benchmark/gold_eval_2.jsonl}"
OUT_DIR="${OUT_DIR:-artifacts/prediction}"
PROMPT_VERSION="${PROMPT_VERSION:-v4_strict_json}"
PYTHON_MODULE="${PYTHON_MODULE:-python}"
CUDA_MODULE="${CUDA_MODULE:-cuda}"
EXTRA_MODULES="${EXTRA_MODULES:-}"
VENV_PATH="${VENV_PATH:-.venv}"
INFER_DEVICE="${INFER_DEVICE:-auto}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

LLAMA_70B_MODEL="${LLAMA_70B_MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
LLAMA_8B_MODEL="${LLAMA_8B_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-4-31b-it}"

# Parallel arrays: MODELS[i] / FILE_SUFFIXES[i] describe one (model, config) combo.
# FILE_SUFFIXES matches the tail of the existing prompts/predictions/report
# filenames: prompts_eval_${PROMPT_VERSION}_${FILE_SUFFIX}.jsonl, etc.
MODELS=(
  "$LLAMA_70B_MODEL"
  "$LLAMA_70B_MODEL"
  "$LLAMA_8B_MODEL"
  "$LLAMA_8B_MODEL"
  "$GEMMA_MODEL"
  "$GEMMA_MODEL"
  "$GEMMA_MODEL"
  "$GEMMA_MODEL"
  "$GEMMA_MODEL"
)
FILE_SUFFIXES=(
  "base"
  "llama-3.1-70b-instruct_anchors"
  "llama-3.1-8b-instruct_base"
  "llama-3.1-8b-instruct_anchors"
  "gemma-4-31b-it_base"
  "gemma-4-31b-it_anchors"
  "gemma-4-31b-it_geo"
  "gemma-4-31b-it_phylo"
  "gemma-4-31b-it_full"
)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script is meant to run as a Slurm array job (--array=0-$((${#MODELS[@]} - 1)))."
  echo "Set SLURM_ARRAY_TASK_ID to test a single combo locally."
  exit 1
fi

idx="$SLURM_ARRAY_TASK_ID"
MODEL="${MODELS[$idx]}"
SUFFIX="${FILE_SUFFIXES[$idx]}"

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p artifacts/logs

if [[ -n "$PYTHON_MODULE" ]]; then module load "$PYTHON_MODULE"; fi
if [[ -n "$CUDA_MODULE" ]]; then module load "$CUDA_MODULE" 2>/dev/null || true; fi
for mod in $EXTRA_MODULES; do module load "$mod"; done
if [[ -f "$VENV_PATH/bin/activate" ]]; then
  source "$VENV_PATH/bin/activate"
else
  echo "No virtualenv found at $VENV_PATH"
  exit 1
fi

echo "Task $idx: model=$MODEL suffix=$SUFFIX"
python --version
nvidia-smi

export HF_HOME="${SCRATCH:-/tmp}/hf"
export TRANSFORMERS_CACHE="${SCRATCH:-/tmp}/hf/transformers"
export HF_DATASETS_CACHE="${SCRATCH:-/tmp}/hf/datasets"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TF=1
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

PROMPTS="$OUT_DIR/prompts_eval_${PROMPT_VERSION}_${SUFFIX}.jsonl"
PRED="$OUT_DIR/predictions_eval_${PROMPT_VERSION}_${SUFFIX}.jsonl"
REPORT="$OUT_DIR/report_eval_${PROMPT_VERSION}_${SUFFIX}.json"
RETRY_PROMPTS="$OUT_DIR/prompts_eval_${PROMPT_VERSION}_${SUFFIX}_retry.jsonl"
RETRY_PRED="$OUT_DIR/predictions_eval_${PROMPT_VERSION}_${SUFFIX}_retry.jsonl"
MERGED_PRED="$OUT_DIR/predictions_eval_${PROMPT_VERSION}_${SUFFIX}.merged.jsonl"

for f in "$PROMPTS" "$PRED"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing expected file: $f" >&2
    exit 1
  fi
done

python -m lkb.scripts.rerun_unparsed extract \
  --pred "$PRED" \
  --prompts "$PROMPTS" \
  --out "$RETRY_PROMPTS"

if [[ ! -s "$RETRY_PROMPTS" ]]; then
  echo "Nothing unparsed for $SUFFIX; skipping infer/merge/evaluate."
  exit 0
fi

python -m lkb.scripts.infer \
  --in "$RETRY_PROMPTS" \
  --out "$RETRY_PRED" \
  --model "$MODEL" \
  --device "$INFER_DEVICE" \
  --dtype "$DTYPE" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS"

python -m lkb.scripts.rerun_unparsed merge \
  --pred "$PRED" \
  --retry "$RETRY_PRED" \
  --out "$MERGED_PRED"

cp "$PRED" "${PRED}.bak"
mv "$MERGED_PRED" "$PRED"

python -m lkb.scripts.evaluate \
  --gold "$GOLD" \
  --pred "$PRED" \
  --report_out "$REPORT"

echo "Done: $SUFFIX (backup at ${PRED}.bak)"
