#!/bin/bash
# Cluster defaults in this script match the lab's current setup:
# - account: rrg-annielee
# - GPU:     h100_3g.40gb:1
# - modules: python + cuda
# Override Slurm resources at submit time, e.g.
#   sbatch --account=<acct> --gpus=a100:1 --mem=80G scripts/run_glottolog_benchmark.sbatch
#SBATCH --job-name=glottolog-benchmark
#SBATCH --account=rrg-annielee
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=artifacts/logs/%x-%j.out
#SBATCH --error=artifacts/logs/%x-%j.err

set -euo pipefail

MODE="${MODE:-benchmark}"   # benchmark | full | all
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
GOLD="${GOLD:-data/benchmark/gold_eval_2.jsonl}"
PROMPT_VERSION="${PROMPT_VERSION:-v5_glottolog_tree_json}"
KG_RUN_MODES="${KG_RUN_MODES:-vote}"   # comma-separated: vote,novote
GLOTTOLOG_PATH="${GLOTTOLOG_PATH:-$SLURM_SUBMIT_DIR/glottolog}"
GEN_DETAIL="${GEN_DETAIL:-data/derived/genetic_neighbours_detailed.json}"
KG_NODES="${KG_NODES:-artifacts/resources/kg_nodes.jsonl}"
KG_EDGES="${KG_EDGES:-artifacts/resources/kg_edges.jsonl}"
RETRIEVAL_BACKEND="${RETRIEVAL_BACKEND:-kg_flat}"
OUT_DIR="${OUT_DIR:-artifacts/prediction/stratified_v1/${RETRIEVAL_BACKEND}}"
PYTHON_MODULE="${PYTHON_MODULE:-python}"
CUDA_MODULE="${CUDA_MODULE:-cuda}"
EXTRA_MODULES="${EXTRA_MODULES:-}"
VENV_PATH="${VENV_PATH:-.venv}"
INFER_DEVICE="${INFER_DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
export GLOTTOLOG_PATH GEN_DETAIL KG_NODES KG_EDGES RETRIEVAL_BACKEND

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

python --version
echo "Gold file: $GOLD"
echo "KG run modes: $KG_RUN_MODES"
nvidia-smi

export HF_HOME="${SCRATCH:-/tmp}/hf"
export TRANSFORMERS_CACHE="${SCRATCH:-/tmp}/hf/transformers"
export HF_DATASETS_CACHE="${SCRATCH:-/tmp}/hf/datasets"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TF=1

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" artifacts/logs "$OUT_DIR"

ensure_gen_detail() {
  if [[ -f "$GEN_DETAIL" ]]; then
    echo "Using existing $GEN_DETAIL"
    return
  fi

  echo "Building $GEN_DETAIL from checked-out Glottolog tree"
  python -m lkb.scripts.build_gen_detail \
    --glottolog_root "$GLOTTOLOG_PATH" \
    --out "$GEN_DETAIL"
}

ensure_kg() {
  if [[ -f "$KG_NODES" && -f "$KG_EDGES" ]]; then
    echo "Using existing KG files: $KG_NODES and $KG_EDGES"
    return
  fi

  echo "Building typed KG JSONL files"
  python -m lkb.scripts.build_kg \
    --gen_detail "$GEN_DETAIL" \
    --glottolog_root "$GLOTTOLOG_PATH" \
    --nodes_out "$KG_NODES" \
    --edges_out "$KG_EDGES"
}

run_benchmark() {
  local no_vote_prompts="$OUT_DIR/prompts_eval_${PROMPT_VERSION}_novote.jsonl"
  local vote_prompts="$OUT_DIR/prompts_eval_${PROMPT_VERSION}_vote.jsonl"
  local no_vote_preds="$OUT_DIR/predictions_eval_${PROMPT_VERSION}_novote.jsonl"
  local vote_preds="$OUT_DIR/predictions_eval_${PROMPT_VERSION}_vote.jsonl"
  local no_vote_report="$OUT_DIR/report_eval_${PROMPT_VERSION}_novote.json"
  local vote_report="$OUT_DIR/report_eval_${PROMPT_VERSION}_vote.json"

  if mode_enabled "novote" "$KG_RUN_MODES"; then
    python -m lkb.scripts.build_prompts \
      --gold "$GOLD" \
      --prompt kg \
      --version "$PROMPT_VERSION" \
      --no-include_vote_table \
      --retrieval_backend "$RETRIEVAL_BACKEND" \
      --gen_detail "$GEN_DETAIL" \
      --kg_nodes "$KG_NODES" \
      --kg_edges "$KG_EDGES" \
      --out "$no_vote_prompts"
  fi

  if mode_enabled "vote" "$KG_RUN_MODES"; then
    python -m lkb.scripts.build_prompts \
      --gold "$GOLD" \
      --prompt kg \
      --version "$PROMPT_VERSION" \
      --include_vote_table \
      --retrieval_backend "$RETRIEVAL_BACKEND" \
      --gen_detail "$GEN_DETAIL" \
      --kg_nodes "$KG_NODES" \
      --kg_edges "$KG_EDGES" \
      --out "$vote_prompts"
  fi

  if mode_enabled "novote" "$KG_RUN_MODES"; then
    python -m lkb.scripts.infer \
      --in "$no_vote_prompts" \
      --out "$no_vote_preds" \
      --model "$MODEL" \
      --device "$INFER_DEVICE" \
      --batch_size "$BATCH_SIZE" \
      --max_new_tokens "$MAX_NEW_TOKENS"
  fi

  if mode_enabled "vote" "$KG_RUN_MODES"; then
    python -m lkb.scripts.infer \
      --in "$vote_prompts" \
      --out "$vote_preds" \
      --model "$MODEL" \
      --device "$INFER_DEVICE" \
      --batch_size "$BATCH_SIZE" \
      --max_new_tokens "$MAX_NEW_TOKENS"
  fi

  if mode_enabled "novote" "$KG_RUN_MODES"; then
    python -m lkb.scripts.evaluate \
      --gold "$GOLD" \
      --pred "$no_vote_preds" \
      --report_out "$no_vote_report"
  fi

  if mode_enabled "vote" "$KG_RUN_MODES"; then
    python -m lkb.scripts.evaluate \
      --gold "$GOLD" \
      --pred "$vote_preds" \
      --report_out "$vote_report"
  fi
}

run_full() {
  python -m lkb.scripts.build_prompts \
    --gold "$GOLD" \
    --prompt kg \
    --version "$PROMPT_VERSION" \
    --include_vote_table \
    --retrieval_backend "$RETRIEVAL_BACKEND" \
    --gen_detail "$GEN_DETAIL" \
    --kg_nodes "$KG_NODES" \
    --kg_edges "$KG_EDGES" \
    --out "$OUT_DIR/eval_missing_prompts_all_${PROMPT_VERSION}.jsonl"

  python -m lkb.scripts.infer \
    --in "$OUT_DIR/eval_missing_prompts_all_${PROMPT_VERSION}.jsonl" \
    --out "$OUT_DIR/eval_missing_predictions_all_${PROMPT_VERSION}.jsonl" \
    --model "$MODEL" \
    --device "$INFER_DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS"
}

ensure_gen_detail
ensure_kg

case "$MODE" in
  benchmark) run_benchmark ;;
  full) run_full ;;
  all) run_benchmark; run_full ;;
  *) echo "MODE must be benchmark|full|all"; exit 1 ;;
esac
