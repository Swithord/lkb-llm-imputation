#!/usr/bin/env bash
# Run the full evaluation pipeline for the stratified_v1 split.
#
# Prerequisites: ensure data/benchmark/gold_eval_2.jsonl is present. This is created via src/create_splits.ipynb.
# Baselines are run via src/baselines.ipynb.
#
# Usage:
#   bash scripts/run_stratified_v1.sh icl_job
#   bash scripts/run_stratified_v1.sh kg_job [kg_flat|kg_typed|hybrid_flat_kg]
#   bash scripts/run_stratified_v1.sh all

set -euo pipefail

ACTION="${1:-all}"
RETRIEVAL_BACKEND="${2:-kg_flat}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
GOLD="data/benchmark/gold_eval_2.jsonl"

check_gold() {
  if [[ ! -f "$GOLD" ]]; then
    echo "ERROR: $GOLD not found."
    exit 1
  fi
}

submit_icl_job() {
  check_gold
  sbatch \
    --export=ALL,\
MODE=benchmark,\
LEGACY_RUN_MODES=vote,novote,\
MODEL="$MODEL",\
OUT_DIR="artifacts/prediction/stratified_v1/icl" \
    scripts/run_eval.sbatch
}

submit_kg_job() {
  check_gold
  sbatch \
    --export=ALL,\
MODE=benchmark,\
KG_RUN_MODES=vote,\
MODEL="$MODEL",\
RETRIEVAL_BACKEND="$RETRIEVAL_BACKEND",\
OUT_DIR="artifacts/prediction/stratified_v1/${RETRIEVAL_BACKEND}" \
    scripts/run_glottolog_benchmark.sbatch
}

case "$ACTION" in
  icl_job) submit_icl_job ;;
  kg_job)  submit_kg_job ;;
  all)
    submit_icl_job
    submit_kg_job
    ;;
  *)
    echo "Unknown action: $ACTION"
    echo "Usage: $0 icl_job | kg_job [backend] | all"
    exit 1
    ;;
esac
