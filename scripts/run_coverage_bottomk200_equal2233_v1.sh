#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_coverage_bottomk200_equal2233_v1.sh build
#   bash scripts/run_coverage_bottomk200_equal2233_v1.sh baselines
#   bash scripts/run_coverage_bottomk200_equal2233_v1.sh legacy_job
#   bash scripts/run_coverage_bottomk200_equal2233_v1.sh kg_job kg_flat
#   bash scripts/run_coverage_bottomk200_equal2233_v1.sh all

ACTION="${1:-all}"                    # build | baselines | legacy_job | kg_job | all
RETRIEVAL_BACKEND="${2:-kg_flat}"     # used only for kg_job
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

build_benchmark() {
  ./.venv/bin/python benchmark/build_urielplus_resource_benchmark.py \
    --split_mode bottomk \
    --min_observed_for_split 5 \
    --lrl_bottom_k 200 \
    --hrl_min_observed 240 \
    --target_each 2233 \
    --min_per_family_type 0 \
    --prompts_out data/benchmark/prompts_eval_coverage_bottomk200_equal2233_v1.jsonl \
    --gold_out data/benchmark/gold_eval_coverage_bottomk200_equal2233_v1.jsonl \
    --examples_out data/benchmark/examples_eval_coverage_bottomk200_equal2233_v1.jsonl \
    --groups_out data/benchmark/language_groups_coverage_bottomk200_equal2233_v1.json \
    --manifest_out data/benchmark/manifest_coverage_bottomk200_equal2233_v1.json
}

run_baselines() {
  ./.venv/bin/python benchmark/score_baselines_on_gold.py \
    --gold data/benchmark/gold_eval_coverage_bottomk200_equal2233_v1.jsonl \
    --resource_groups_json data/benchmark/language_groups_coverage_bottomk200_equal2233_v1.json \
    --out_dir artifacts/prediction/coverage_bottomk200_equal2233_v1/baselines \
    --report_out artifacts/prediction/coverage_bottomk200_equal2233_v1/report_baselines_on_gold.json
}

submit_legacy_job() {
  sbatch --export=ALL,MODE=benchmark,BENCHMARK_VARIANT=coverage_bottomk200_equal2233_v1,LEGACY_RUN_MODES=vote,novote,MODEL="$MODEL" scripts/run_eval.sbatch
}

submit_kg_job() {
  sbatch --export=ALL,MODE=benchmark,BENCHMARK_VARIANT=coverage_bottomk200_equal2233_v1,KG_RUN_MODES=vote,MODEL="$MODEL",RETRIEVAL_BACKEND="$RETRIEVAL_BACKEND" scripts/run_glottolog_benchmark.sbatch
}

case "$ACTION" in
  build) build_benchmark ;;
  baselines) run_baselines ;;
  legacy_job) submit_legacy_job ;;
  kg_job) submit_kg_job ;;
  all)
    build_benchmark
    run_baselines
    ;;
  *)
    echo "Unknown action: $ACTION"
    exit 1
    ;;
esac

