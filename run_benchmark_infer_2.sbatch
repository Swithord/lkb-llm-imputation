#!/bin/bash
#SBATCH --job-name=bench-infer-2
#SBATCH --account=rrg-annielee
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "Working dir: $(pwd)"

module load python
module load cuda 2>/dev/null || true
source ~/venv/bin/activate

cd "$SLURM_SUBMIT_DIR"

export HF_HOME="$SCRATCH/hf"
export TRANSFORMERS_CACHE="$SCRATCH/hf/transformers"
export HF_DATASETS_CACHE="$SCRATCH/hf/datasets"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" logs

# High-resource prompts (_2)
python benchmark/infer_from_prompts.py \
  --in benchmark/prompts_eval_high_2.jsonl \
  --out benchmark/predictions_eval_high_2.jsonl \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --device cuda \
  --batch_size 4

# Low-resource prompts (_2)
python benchmark/infer_from_prompts.py \
  --in benchmark/prompts_eval_low_2.jsonl \
  --out benchmark/predictions_eval_low_2.jsonl \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --device cuda \
  --batch_size 4

# Merge + evaluate (_2)
cat benchmark/predictions_eval_high_2.jsonl benchmark/predictions_eval_low_2.jsonl > benchmark/predictions_eval_2.jsonl
python benchmark/evaluate_benchmark.py \
  --gold benchmark/gold_eval_2.jsonl \
  --pred benchmark/predictions_eval_2.jsonl \
  --report_out benchmark/report_eval_2.json

echo "Done at $(date)"
