#!/bin/bash
# Cluster defaults in this script match the lab's current setup:
# - account: rrg-annielee
# - GPU:     h100_3g.40gb:1
# - modules: python + cuda
# Override Slurm resources at submit time, e.g.
#   sbatch --account=<acct> --gpus=a100:1 --mem=80G scripts/run_eval.sbatch
#SBATCH --job-name=llm-impute-benchmark
#SBATCH --account=rrg-annielee
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gpus=h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --output=artifacts/logs/%x-%j.out
#SBATCH --error=artifacts/logs/%x-%j.err
#SBATCH --mem=32G
#SBATCH --array=0-4
#SBATCH --output=artifacts/logs/%x-%A_%a.out
#SBATCH --error=artifacts/logs/%x-%A_%a.err

set -euo pipefail

CONFIGS=(base anchors phylo geo full)
cfg="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
GOLD="${GOLD:-data/benchmark/gold_eval_2.jsonl}"
OUT_DIR="${OUT_DIR:-artifacts/prediction}"
PROMPT_VERSION="${PROMPT_VERSION:-v4_strict_json}"
PYTHON_MODULE="${PYTHON_MODULE:-python}"
CUDA_MODULE="${CUDA_MODULE:-cuda}"
EXTRA_MODULES="${EXTRA_MODULES:-}"
VENV_PATH="${VENV_PATH:-.venv}"
INFER_DEVICE="${INFER_DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

# base    = metadata only
# anchors = + anchor features + correlated clues
# phylo   = anchors + phylogenetic neighbours (+ vote table + contrastive)
# geo     = anchors + geographic neighbours   (+ vote table + contrastive)
# full    = anchors + both neighbour types    (+ vote table + contrastive)
config_flags() {
  case "$1" in
    base)    echo "--no-include_anchor_features --no-include_phylo_neighbors --no-include_geo_neighbors" ;;
    anchors) echo "--include_anchor_features --no-include_phylo_neighbors --no-include_geo_neighbors" ;;
    phylo)   echo "--include_anchor_features --include_phylo_neighbors --no-include_geo_neighbors" ;;
    geo)     echo "--include_anchor_features --no-include_phylo_neighbors --include_geo_neighbors" ;;
    full)    echo "--include_anchor_features --include_phylo_neighbors --include_geo_neighbors" ;;
  esac
}

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p artifacts/logs "$OUT_DIR"

if [[ -n "$PYTHON_MODULE" ]]; then module load "$PYTHON_MODULE"; fi
if [[ -n "$CUDA_MODULE" ]]; then module load "$CUDA_MODULE" 2>/dev/null || true; fi
for mod in $EXTRA_MODULES; do module load "$mod"; done
if [[ -f "$VENV_PATH/bin/activate" ]]; then
  source "$VENV_PATH/bin/activate"
else
  echo "No virtualenv found at $VENV_PATH"; exit 1
fi

echo "PWD: $(pwd)"
echo "HOST: $(hostname)"
echo "which python: $(which python)"
python --version
echo "Ablation config: $cfg (task $SLURM_ARRAY_TASK_ID)"
echo "Gold file: $GOLD"
python -c "import pandas; print('pandas ok')"
python -c "import torch; print('torch ok')"

#nvidia-smi

#export HF_HOME="${SCRATCH:-/tmp}/hf"
export TRANSFORMERS_CACHE="${SCRATCH:-/tmp}/hf/transformers"
export HF_DATASETS_CACHE="${SCRATCH:-/tmp}/hf/datasets"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TF=1
export HF_HUB_DISABLE_XET=1
export HF_HOME=~/scratch/.cache/huggingface
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

PROMPTS="$OUT_DIR/prompts_eval_${PROMPT_VERSION}_${cfg}.jsonl"
PREDS="$OUT_DIR/predictions_eval_${PROMPT_VERSION}_${cfg}.jsonl"
REPORT="$OUT_DIR/report_eval_${PROMPT_VERSION}_${cfg}.json"

# shellcheck disable=SC2046
python -m lkb.scripts.build_prompts \
  --gold "$GOLD" \
  --prompt icl \
  --version "$PROMPT_VERSION" \
  $(config_flags "$cfg") \
  --out "$PROMPTS"

python -m lkb.scripts.infer \
  --in "$PROMPTS" \
  --out "$PREDS" \
  --model "$MODEL" \
  --device "$INFER_DEVICE" \
  --dtype "$DTYPE" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS"

python -m lkb.scripts.evaluate \
  --gold "$GOLD" \
  --pred "$PREDS" \
  --report_out "$REPORT"

# Speed benchmark only for the full config
if [[ "$cfg" == "full" ]]; then
  python -m lkb.scripts.speed_bench \
    --in "$PROMPTS" \
    --model "$MODEL" \
    --device "$INFER_DEVICE" \
    --dtype "$DTYPE" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --report_out "$OUT_DIR/speed_report_${PROMPT_VERSION}.json"
fi
