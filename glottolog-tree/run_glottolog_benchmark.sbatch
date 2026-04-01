#!/bin/bash
#SBATCH --job-name=glottolog-benchmark
#SBATCH --account=rrg-annielee
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

MODE="${MODE:-benchmark}"   # benchmark | full | all
MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
GOLD="${GOLD:-benchmark/gold_eval_2.jsonl}"
PROMPT_VERSION="${PROMPT_VERSION:-v5_glottolog_tree_json}"
OUT_DIR="${OUT_DIR:-prediction/benchmark_glottolog}"
GLOTTOLOG_PATH="${GLOTTOLOG_PATH:-$SLURM_SUBMIT_DIR/glottolog}"
GEN_DETAIL="${GEN_DETAIL:-output/genetic_neighbours_detailed.json}"
KG_NODES="${KG_NODES:-glottolog-tree/kg_nodes.jsonl}"
KG_EDGES="${KG_EDGES:-glottolog-tree/kg_edges.jsonl}"
RETRIEVAL_BACKEND="${RETRIEVAL_BACKEND:-kg_flat}"
export GLOTTOLOG_PATH GEN_DETAIL KG_NODES KG_EDGES RETRIEVAL_BACKEND

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs "$OUT_DIR"

module load python
module load cuda 2>/dev/null || true
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

python --version
nvidia-smi

export HF_HOME="${SCRATCH:-/tmp}/hf"
export TRANSFORMERS_CACHE="${SCRATCH:-/tmp}/hf/transformers"
export HF_DATASETS_CACHE="${SCRATCH:-/tmp}/hf/datasets"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TF=1

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" logs "$OUT_DIR"

ensure_gen_detail() {
  if [[ -f "$GEN_DETAIL" ]]; then
    echo "Using existing $GEN_DETAIL"
    return
  fi

  echo "Building $GEN_DETAIL from checked-out Glottolog tree"
  python - <<'PY'
import configparser
import csv
import importlib.util
import json
import os
from pathlib import Path

root = Path(os.environ.get("GLOTTOLOG_PATH", "glottolog")) / "languoids" / "tree"
if not root.exists():
    raise SystemExit("Missing glottolog/languoids/tree; set GLOTTOLOG_PATH or stage the repo first.")

out_path = Path(os.environ.get("GEN_DETAIL", "output/genetic_neighbours_detailed.json"))
out_path.parent.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("phylo_tree", "glottolog-tree/phylo_tree.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

parent = {}
children = {}
level = {}
for md in root.rglob("md.ini"):
    node_dir = md.parent
    node_id = node_dir.name
    rel_parts = node_dir.relative_to(root).parts
    pid = rel_parts[-2] if len(rel_parts) > 1 else None
    parent[node_id] = pid
    children.setdefault(node_id, [])
    if pid is not None:
        children.setdefault(pid, []).append(node_id)
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(md, encoding="utf-8")
    level[node_id] = cp.get("core", "level", fallback="").strip()

langs = []
with open("output/uriel+_typological.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if row:
            langs.append(row[0])

allowed = set(langs)
details = {}
for i, lang in enumerate(langs, start=1):
    if lang not in parent:
        details[lang] = []
        continue
    details[lang] = module.phylo_neighbor_records(
        lang,
        k_neighbors=200,
        parent=parent,
        children=children,
        level=level,
        allowed_nodes=allowed,
    )
    if i % 1000 == 0:
        print(f"processed {i}/{len(langs)}")

out_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {out_path}")
PY
}

ensure_kg() {
  if [[ -f "$KG_NODES" && -f "$KG_EDGES" ]]; then
    echo "Using existing KG files: $KG_NODES and $KG_EDGES"
    return
  fi

  echo "Building typed KG JSONL files"
  python glottolog-tree/kg_builder.py \
    --typ output/uriel+_typological.csv \
    --meta output/metadata.csv \
    --geo output/geographic_neighbours.json \
    --gen output/genetic_neighbours.json \
    --gen_detail "$GEN_DETAIL" \
    --topk out_corr/topk_per_feature.csv \
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

  python benchmark/build_prompts_from_gold.py \
    --prompting_py glottolog-tree/prompting.py \
    --typ output/uriel+_typological.csv \
    --meta output/metadata.csv \
    --topk_csv out_corr/topk_per_feature.csv \
    --gen output/genetic_neighbours.json \
    --gen_detail "$GEN_DETAIL" \
    --geo output/geographic_neighbours.json \
    --gold "$GOLD" \
    --prompt_version "$PROMPT_VERSION" \
    --no-include_vote_table \
    --retrieval_backend "$RETRIEVAL_BACKEND" \
    --kg_nodes "$KG_NODES" \
    --kg_edges "$KG_EDGES" \
    --out "$no_vote_prompts"

  python benchmark/build_prompts_from_gold.py \
    --prompting_py glottolog-tree/prompting.py \
    --typ output/uriel+_typological.csv \
    --meta output/metadata.csv \
    --topk_csv out_corr/topk_per_feature.csv \
    --gen output/genetic_neighbours.json \
    --gen_detail "$GEN_DETAIL" \
    --geo output/geographic_neighbours.json \
    --gold "$GOLD" \
    --prompt_version "$PROMPT_VERSION" \
    --include_vote_table \
    --retrieval_backend "$RETRIEVAL_BACKEND" \
    --kg_nodes "$KG_NODES" \
    --kg_edges "$KG_EDGES" \
    --out "$vote_prompts"

  python benchmark/infer_from_prompts.py \
    --in "$no_vote_prompts" \
    --out "$no_vote_preds" \
    --model "$MODEL" \
    --device cuda \
    --batch_size 4 \
    --max_new_tokens 64 \
    --no-strengthen_json_contract \
    --stop_at_closing_brace

  python benchmark/infer_from_prompts.py \
    --in "$vote_prompts" \
    --out "$vote_preds" \
    --model "$MODEL" \
    --device cuda \
    --batch_size 4 \
    --max_new_tokens 64 \
    --no-strengthen_json_contract \
    --stop_at_closing_brace

  python benchmark/evaluate_benchmark.py \
    --gold "$GOLD" \
    --pred "$no_vote_preds" \
    --report_out "$no_vote_report"

  python benchmark/evaluate_benchmark.py \
    --gold "$GOLD" \
    --pred "$vote_preds" \
    --report_out "$vote_report"
}

run_full() {
  OUT_DIR_ENV="$OUT_DIR" python - <<'PY'
import json
import pandas as pd
import os

out_dir = os.environ["OUT_DIR_ENV"]
df = pd.read_csv("output/uriel+_typological.csv", index_col=0)
langs = [str(x) for x in df.index]
path = f"{out_dir}/language_groups_all.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump({"high": langs, "low": []}, f, ensure_ascii=False, indent=2)
print(f"Wrote {path}")
PY

  python benchmark/eval_prompting.py \
    --prompting_py glottolog-tree/prompting.py \
    --typ output/uriel+_typological.csv \
    --meta output/metadata.csv \
    --topk_csv out_corr/topk_per_feature.csv \
    --gen output/genetic_neighbours.json \
    --gen_detail "$GEN_DETAIL" \
    --geo output/geographic_neighbours.json \
    --groups_json "$OUT_DIR/language_groups_all.json" \
    --prompt_version "$PROMPT_VERSION" \
    --include_vote_table \
    --retrieval_backend "$RETRIEVAL_BACKEND" \
    --kg_nodes "$KG_NODES" \
    --kg_edges "$KG_EDGES" \
    --model "$MODEL" \
    --device cuda \
    --batch_size 4 \
    --max_new_tokens 64 \
    --prompts_out "$OUT_DIR/eval_missing_prompts_all_${PROMPT_VERSION}.jsonl" \
    --pred_out "$OUT_DIR/eval_missing_predictions_all_${PROMPT_VERSION}.jsonl"
}

ensure_gen_detail
ensure_kg

case "$MODE" in
  benchmark) run_benchmark ;;
  full) run_full ;;
  all) run_benchmark; run_full ;;
  *) echo "MODE must be benchmark|full|all"; exit 1 ;;
esac
