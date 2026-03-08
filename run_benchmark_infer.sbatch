#!/bin/bash
#SBATCH --job-name=bench-infer-v3
#SBATCH --account=rrg-annielee
#SBATCH --time=05:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load python
module load cuda 2>/dev/null || true
source ~/venv/bin/activate

cd "$SLURM_SUBMIT_DIR"   # submit from repo root

export HF_HOME="$SCRATCH/hf"
export TRANSFORMERS_CACHE="$SCRATCH/hf/transformers"
export HF_DATASETS_CACHE="$SCRATCH/hf/datasets"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TF=1
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" logs benchmark

MODEL="meta-llama/Llama-3.2-3B-Instruct"
GOLD="benchmark/gold_eval_2.jsonl"

# Build v3 strict-json prompts on the same eval IDs: no-vote and vote-table
python benchmark/build_prompts_from_gold.py \
  --prompting_py code/prompting.py \
  --typ output/uriel+_typological.csv \
  --meta output/metadata.csv \
  --topk_csv out_corr/topk_per_feature.csv \
  --gen output/genetic_neighbours.json \
  --geo output/geographic_neighbours.json \
  --gold "$GOLD" \
  --prompt_version v3_strict_json \
  --no-include_vote_table \
  --out benchmark/prompts_eval_v3_strict_json_novote.jsonl

python benchmark/build_prompts_from_gold.py \
  --prompting_py code/prompting.py \
  --typ output/uriel+_typological.csv \
  --meta output/metadata.csv \
  --topk_csv out_corr/topk_per_feature.csv \
  --gen output/genetic_neighbours.json \
  --geo output/geographic_neighbours.json \
  --gold "$GOLD" \
  --prompt_version v3_strict_json \
  --include_vote_table \
  --out benchmark/prompts_eval_v3_strict_json_vote.jsonl

# Inference: v3 without vote table
python benchmark/infer_from_prompts.py \
  --in benchmark/prompts_eval_v3_strict_json_novote.jsonl \
  --out benchmark/predictions_eval_v3_strict_json_novote.jsonl \
  --model "$MODEL" \
  --device cuda \
  --batch_size 4 \
  --max_new_tokens 64 \
  --no-strengthen_json_contract \
  --stop_at_closing_brace

# Inference: v3 with vote table
python benchmark/infer_from_prompts.py \
  --in benchmark/prompts_eval_v3_strict_json_vote.jsonl \
  --out benchmark/predictions_eval_v3_strict_json_vote.jsonl \
  --model "$MODEL" \
  --device cuda \
  --batch_size 4 \
  --max_new_tokens 64 \
  --no-strengthen_json_contract \
  --stop_at_closing_brace

# Evaluate both
python benchmark/evaluate_benchmark.py \
  --gold "$GOLD" \
  --pred benchmark/predictions_eval_v3_strict_json_novote.jsonl \
  --report_out benchmark/report_eval_v3_strict_json_novote.json

python benchmark/evaluate_benchmark.py \
  --gold "$GOLD" \
  --pred benchmark/predictions_eval_v3_strict_json_vote.jsonl \
  --report_out benchmark/report_eval_v3_strict_json_vote.json

# Baseline report for comparison (uses existing predictions_eval_2.jsonl if present)
if [[ ! -f benchmark/report_eval_baseline_from_existing.json ]]; then
  python benchmark/evaluate_benchmark.py \
    --gold "$GOLD" \
    --pred benchmark/predictions_eval_2.jsonl \
    --report_out benchmark/report_eval_baseline_from_existing.json
fi

# Ablation comparison + raw parse compliance deltas
python benchmark/compare_prompt_reports.py \
  --baseline_report benchmark/report_eval_baseline_from_existing.json \
  --v3_no_vote_report benchmark/report_eval_v3_strict_json_novote.json \
  --v3_vote_report benchmark/report_eval_v3_strict_json_vote.json \
  --out report_prompt_v3.json
