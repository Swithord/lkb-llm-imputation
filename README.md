# lkb-llm-imputation

Utilities for typological feature imputation with two retrieval paths:

- a legacy URIEL+/neighbor prompting pipeline under `code/`
- a Glottolog-tree and KG-backed prompting pipeline under `glottolog-tree/`

The repo already includes checked-in data under `data/`, a vendored `glottolog/` checkout, benchmark tooling, baseline models, and SLURM launch scripts.

## Code vs Data Boundary

Use this rule of thumb:

- `code/` = logic (Python modules that transform inputs into outputs).
- `data/` = inputs/derived tables consumed by that logic.

In this repo:

- `code/` contains the legacy URIEL+ pipeline implementation (`preprocessing.py`, `prompting.py`, neighbor builders, correlation utilities).
- `data/benchmark/` contains benchmark labels and split metadata.
- `data/derived/` contains processed tables used at runtime (metadata, neighbor JSONs, typological matrix).
- `data/features/` contains feature-level artifacts (correlation outputs, feature-name maps, top-k maps).
- `data/splits/` contains split/mask artifacts used by baselines and evaluation.

Operationally: if a file is executed (`python ...`) it belongs in `code/`; if it is loaded by those scripts as an input table/JSON/CSV it belongs in `data/`.

## Repository Structure

```text
.
├── baselines/          Classical baselines: random, mean, kNN, SoftImpute
├── benchmark/          Benchmark building, inference, evaluation, calibration, analysis
├── code/               Legacy preprocessing and prompt construction
├── data/
│   ├── benchmark/      Gold labels and language-group definitions
│   ├── derived/        Metadata and neighbor tables used by prompting
│   ├── features/       Correlation matrices, feature names, top-k feature map
│   └── splits/         Standard language splits and MCAR masks
├── glottolog/          Vendored Glottolog checkout used for metadata/tree traversal
├── glottolog-tree/     Tree-aware prompting, KG build/load/retrieval code
├── scripts/            SLURM batch wrappers
└── test/               Regression tests for retrieval, preprocessing, and baselines
```

## Environment Setup

Pinned user-space dependencies live in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then install a `torch` build that matches the cluster CUDA stack in your active environment.

The pinned file currently covers the repo's direct Python dependencies, including `urielplus` for rebuild workflows. `torch` is intentionally left out because the correct wheel depends on the cluster runtime.

Notes:

- Tests were verified in the local repo venv with Python 3.12.
- `torch` is the only dependency that must be matched to the cluster GPU/CUDA setup.
- Most benchmark and baseline commands can run from the checked-in data without rebuilding URIEL+ artifacts first.

## Cluster Assumptions

The SLURM scripts are written for the lab's current cluster defaults:

- `--account=rrg-annielee`
- `--gres=gpu:h100:1`
- `module load python`
- `module load cuda`
- repo-local virtualenv at `.venv`

Those defaults are not hard requirements. Override resources with `sbatch` flags and runtime settings with exported environment variables.

Resource override example:

```bash
sbatch \
  --account=your-account \
  --gres=gpu:a100:1 \
  --cpus-per-task=8 \
  --mem=80G \
  --export=ALL,MODEL=meta-llama/Llama-3.1-8B-Instruct \
  scripts/run_eval.sbatch
```

Runtime override example:

```bash
sbatch \
  --export=ALL,VENV_PATH=$HOME/your-venv,PYTHON_MODULE=python,CUDA_MODULE=cuda,EXTRA_MODULES="",INFER_DEVICE=cuda,BATCH_SIZE=2,MAX_NEW_TOKENS=64 \
  scripts/run_glottolog_benchmark.sbatch
```

## Quick Start

### 1. Run tests

Some tests import modules out of `code/` and `benchmark/` directly, so include both on `PYTHONPATH`:

```bash
PYTHONPATH=code:benchmark pytest -q test
```

### 2. Run the standard baseline suite

Create the language split and MCAR mask:

```bash
python benchmark/prepare_standard_split.py
```

Run random, mean, and kNN baselines:

```bash
python baselines/run_baselines.py
```

Run SoftImpute:

```bash
python baselines/run_softimpute.py
```

Outputs land in `data/splits/`, `artifacts/baseline/`, and `artifacts/reports/`.

### 3. Run the legacy prompting benchmark

Use the SLURM wrapper:

```bash
sbatch --export=ALL,MODE=benchmark,MODEL=meta-llama/Llama-3.1-8B-Instruct scripts/run_eval.sbatch
```

Run the same wrapper on the coverage-based benchmark:

```bash
sbatch --export=ALL,MODE=benchmark,BENCHMARK_VARIANT=coverage_resource_v1,MODEL=meta-llama/Llama-3.1-8B-Instruct scripts/run_eval.sbatch
```

Useful overrides:

```bash
sbatch --export=ALL,MODE=all,MODEL=meta-llama/Llama-3.1-8B-Instruct,PROMPT_VERSION=v3_strict_json,OUT_DIR=artifacts/prediction scripts/run_eval.sbatch
```

This script handles prompt building, model inference, and benchmark evaluation.

Manual step-by-step commands are still available if you want local debugging without SLURM:

```bash
python benchmark/build_prompts_from_gold.py \
  --prompting_py code/prompting.py \
  --typ data/derived/uriel+_typological.csv \
  --meta data/derived/metadata.csv \
  --topk_csv data/features/topk_per_feature.csv \
  --gen data/derived/genetic_neighbours.json \
  --geo data/derived/geographic_neighbours.json \
  --gold data/benchmark/gold_eval_2.jsonl \
  --prompt_version v3_strict_json \
  --include_vote_table \
  --out artifacts/prediction/prompts_eval_v3_strict_json_vote.jsonl
```

For local debugging on the coverage-based benchmark, swap the gold path:

```bash
python benchmark/build_prompts_from_gold.py \
  --prompting_py code/prompting.py \
  --typ data/derived/uriel+_typological.csv \
  --meta data/derived/metadata.csv \
  --topk_csv data/features/topk_per_feature.csv \
  --gen data/derived/genetic_neighbours.json \
  --geo data/derived/geographic_neighbours.json \
  --gold data/benchmark/gold_eval_coverage_resource_v1.jsonl \
  --prompt_version v3_strict_json \
  --include_vote_table \
  --out artifacts/prediction/coverage_resource_v1/prompts_eval_v3_strict_json_vote.jsonl
```

### 4. Run the Glottolog-tree or KG-backed benchmark

Use the Glottolog-tree SLURM wrapper:

```bash
sbatch --export=ALL,MODE=benchmark,MODEL=meta-llama/Llama-3.1-8B-Instruct,RETRIEVAL_BACKEND=kg_flat scripts/run_glottolog_benchmark.sbatch
```

Run the same wrapper on the coverage-based benchmark:

```bash
sbatch --export=ALL,MODE=benchmark,BENCHMARK_VARIANT=coverage_resource_v1,MODEL=meta-llama/Llama-3.1-8B-Instruct,RETRIEVAL_BACKEND=kg_flat scripts/run_glottolog_benchmark.sbatch
```

Useful overrides:

```bash
sbatch --export=ALL,MODE=all,MODEL=meta-llama/Llama-3.1-8B-Instruct,PROMPT_VERSION=v5_glottolog_tree_json,RETRIEVAL_BACKEND=kg_typed,OUT_DIR=artifacts/prediction/benchmark_glottolog scripts/run_glottolog_benchmark.sbatch
```

This script will:

- ensure `data/derived/genetic_neighbours_detailed.json` exists
- ensure `artifacts/resources/kg_nodes.jsonl` and `artifacts/resources/kg_edges.jsonl` exist
- build prompts
- run inference
- evaluate predictions

Manual step-by-step commands are still available if you want to run pieces locally:

```bash
python benchmark/build_prompts_from_gold.py \
  --prompting_py glottolog-tree/prompting.py \
  --typ data/derived/uriel+_typological.csv \
  --meta data/derived/metadata.csv \
  --topk_csv data/features/topk_per_feature.csv \
  --gen data/derived/genetic_neighbours.json \
  --gen_detail data/derived/genetic_neighbours_detailed.json \
  --geo data/derived/geographic_neighbours.json \
  --retrieval_backend kg_flat \
  --kg_nodes artifacts/resources/kg_nodes.jsonl \
  --kg_edges artifacts/resources/kg_edges.jsonl \
  --gold data/benchmark/gold_eval_2.jsonl \
  --prompt_version v5_glottolog_tree_json \
  --include_vote_table \
  --out artifacts/prediction/benchmark_glottolog/prompts_eval_v5_glottolog_tree_json_vote_kg_flat.jsonl
```

Coverage-based benchmark variant:

```bash
python benchmark/build_prompts_from_gold.py \
  --prompting_py glottolog-tree/prompting.py \
  --typ data/derived/uriel+_typological.csv \
  --meta data/derived/metadata.csv \
  --topk_csv data/features/topk_per_feature.csv \
  --gen data/derived/genetic_neighbours.json \
  --gen_detail data/derived/genetic_neighbours_detailed.json \
  --geo data/derived/geographic_neighbours.json \
  --retrieval_backend kg_flat \
  --kg_nodes artifacts/resources/kg_nodes.jsonl \
  --kg_edges artifacts/resources/kg_edges.jsonl \
  --gold data/benchmark/gold_eval_coverage_resource_v1.jsonl \
  --prompt_version v5_glottolog_tree_json \
  --include_vote_table \
  --out artifacts/prediction/coverage_resource_v1/prompts_eval_v5_glottolog_tree_json_vote_kg_flat.jsonl
```

Valid `--retrieval_backend` values used in this repo are:

- `legacy`
- `kg_flat`
- `kg_typed`

## Rebuilding Data From Scratch

You only need this section if you want to regenerate `data/features/`, `data/derived/`, or the KG files.

### 1. Recompute feature correlations and feature metadata

```bash
python code/compute_feature_corr.py \
  --out data/features \
  --integrate_all \
  --use_glottocodes \
  --compute_imputed
```

This writes `feature_names.json`, `languages.json`, `C_phi.npy`, `support.npy`, `meta.json`, and optionally `C_imputed.npy`.

Build the top-k feature map used by prompt construction:

```bash
python code/select_topk_features.py \
  --corr data/features/C_phi.npy \
  --features data/features/feature_names.json \
  --k 10 \
  --absolute \
  --out data/features/topk_per_feature.csv
```

### 2. Rebuild metadata and neighbor files

The Glottolog-tree preprocessing path is the superset because it also writes `genetic_neighbours_detailed.json`.

```bash
mkdir -p data/derived
(
  cd data/derived
  GLOTTOLOG_PATH=../../glottolog PYTHONPATH=../../code python ../../glottolog-tree/preprocessing.py
)
```

This produces:

- `data/derived/metadata.csv`
- `data/derived/uriel+_typological.csv`
- `data/derived/genetic_neighbours.json`
- `data/derived/genetic_neighbours_detailed.json`
- `data/derived/geographic_neighbours.json`

If you only want the legacy files, use `../../code/preprocessing.py` instead.

### 3. Rebuild KG node and edge JSONL files

```bash
python glottolog-tree/kg_builder.py \
  --typ data/derived/uriel+_typological.csv \
  --meta data/derived/metadata.csv \
  --geo data/derived/geographic_neighbours.json \
  --gen data/derived/genetic_neighbours.json \
  --gen_detail data/derived/genetic_neighbours_detailed.json \
  --topk data/features/topk_per_feature.csv \
  --glottolog_root glottolog \
  --nodes_out artifacts/resources/kg_nodes.jsonl \
  --edges_out artifacts/resources/kg_edges.jsonl
```

### 4. Build a fresh high/low-resource benchmark split

```bash
python benchmark/build_benchmark.py \
  --prompting_py glottolog-tree/prompting.py \
  --typ data/derived/uriel+_typological.csv \
  --meta data/derived/metadata.csv \
  --topk_csv data/features/topk_per_feature.csv \
  --gen data/derived/genetic_neighbours.json \
  --gen_detail data/derived/genetic_neighbours_detailed.json \
  --geo data/derived/geographic_neighbours.json \
  --high_n 50 \
  --low_n 50 \
  --per_language 20 \
  --top_n 10 \
  --prompt_version v5_glottolog_tree_json
```

This writes prompts, examples, gold labels, and language groups under `data/benchmark/`.

Important protocol note:
- the checked-in frozen benchmark used for reported comparisons is `data/benchmark/gold_eval_2.jsonl`
- its associated language split file is `data/benchmark/language_groups_2.json`
- if you rebuild the benchmark, do not treat the new output as directly comparable to reported numbers unless you intentionally overwrite or re-freeze the protocol

### 5. Build the coverage-based HRL/MRL/LRL benchmark

```bash
python benchmark/build_urielplus_resource_benchmark.py
```

This writes a new non-frozen benchmark snapshot under `data/benchmark/`:

- `gold_eval_coverage_resource_v1.jsonl`
- `prompts_eval_coverage_resource_v1.jsonl`
- `examples_eval_coverage_resource_v1.jsonl`
- `language_groups_coverage_resource_v1.json`
- `manifest_coverage_resource_v1.json`

## Benchmark Analysis

Use `benchmark/evaluate_benchmark.py` to score benchmark predictions:

```bash
python benchmark/evaluate_benchmark.py \
  --gold data/benchmark/gold_eval_2.jsonl \
  --pred artifacts/prediction/predictions_eval_v3_strict_json_vote.jsonl \
  --report_out artifacts/prediction/report_eval_v3_strict_json_vote.json
```

Coverage-based benchmark example:

```bash
python benchmark/evaluate_benchmark.py \
  --gold data/benchmark/gold_eval_coverage_resource_v1.jsonl \
  --pred artifacts/prediction/coverage_resource_v1/predictions_eval_v5_glottolog_tree_json_vote.jsonl \
  --report_out artifacts/prediction/coverage_resource_v1/report_eval_v5_glottolog_tree_json_vote.json
```

The evaluator reports:

- overall and subgroup accuracy
- parsed rate
- high-confidence accuracy and coverage
- rationale quality rate
- calibration metrics such as Brier score and ECE

It emits both:

- `normalized`: metrics after parsing and normalizing `value`, `confidence`, and `rationale`
- `strict_raw`: metrics parsed directly from raw model output

## Frozen Benchmark Protocol

The current repo supports many experiment paths, but only one benchmark artifact set is frozen for direct method comparison:

- gold file: `data/benchmark/gold_eval_2.jsonl`
- language groups: `data/benchmark/language_groups_2.json`
- item count: `1433`
- resource groups present: `high`, `low`

What is fixed when you use the standard benchmark scripts:
- sampled evaluation items
- high/low split membership
- JSON output schema
- benchmark metrics

What is not fully standardized by the builder:
- there is no `mrl` tier
- feature-type balance is not explicitly enforced
- family-balance or family-exclusion constraints are not explicitly enforced

So the reported KG comparisons are standardized with respect to one frozen evaluation set, but not in the stronger sense of a fully stratified benchmark design.

## Coverage-Based Benchmark Snapshot

The repo also includes a separate non-frozen benchmark snapshot for coverage-based `lrl/mrl/hrl` analysis:

- gold file: `data/benchmark/gold_eval_coverage_resource_v1.jsonl`
- language groups: `data/benchmark/language_groups_coverage_resource_v1.json`
- manifest: `data/benchmark/manifest_coverage_resource_v1.json`
- item count: `10000`
- resource groups present: `lrl`, `mrl`, `hrl`

Use this benchmark for new reruns and analysis. Do not mix its results with `gold_eval_2` in the same direct-comparison table.

## Frozen Benchmark Snapshot

The table below summarizes representative results on the frozen benchmark `data/benchmark/gold_eval_2.jsonl`.

All rows use the same:

- evaluation items
- high/low split labels
- output schema
- metric pipeline

The prompt-based rows below compare the `vote` variants.

| Method | Family | Overall Acc | Overall F1 | High Acc | Low Acc |
|---|---|---:|---:|---:|---:|
| Mean | Classical baseline | 78.65 | 67.24 | 81.20 | 72.75 |
| kNN | Classical baseline | 83.18 | 74.44 | 88.80 | 70.21 |
| SoftImpute | Classical baseline | 85.35 | 78.40 | 90.80 | 72.75 |
| `v4_8b` | Legacy prompt / RAG-style baseline | 82.83 | 76.21 | 83.10 | 82.22 |
| `kg_flat` | KG baseline-preserving | 79.69 | 73.03 | 80.00 | 78.98 |
| `hybrid_flat_kg_fixed` | Flat + KG hybrid | 82.14 | 75.57 | 82.40 | 81.52 |
| `kg_typed_fixed` | Typed KG retrieval | 90.93 | 87.45 | 89.60 | 94.00 |
| `compact_fixed` | Typed KG retrieval, compact prompt | 91.21 | 87.77 | 89.90 | 94.23 |
| `kg_typed_contrastive_fixed` | Typed KG retrieval with contrastive support | **91.49** | **88.13** | **90.20** | **94.46** |

Takeaways:

- `kg_flat` is mainly an infrastructure baseline; graph storage alone does not improve much.
- `hybrid_flat_kg_fixed` helps modestly, which suggests graph evidence is useful as augmentation.
- the large gains come from typed, feature-conditioned KG retrieval
- explicit contrastive retrieval is the strongest variant on this frozen benchmark
- gains are especially large on the low-resource split

Fit a confidence calibration map:

```bash
python benchmark/fit_confidence_map.py \
  --gold data/benchmark/gold_eval_2.jsonl \
  --pred artifacts/prediction/predictions_eval_v3_strict_json_vote.jsonl \
  --mode normalized \
  --out artifacts/benchmark/confidence_map.json
```

Evaluate with a held-out calibration split:

```bash
python benchmark/split_calibration_eval.py \
  --gold data/benchmark/gold_eval_2.jsonl \
  --pred artifacts/prediction/predictions_eval_v3_strict_json_vote.jsonl \
  --mode normalized \
  --split_by language \
  --test_frac 0.2 \
  --seed 7 \
  --out artifacts/benchmark/split_calibration_report.json
```

Run per-feature error analysis:

```bash
python benchmark/feature_error_analysis.py \
  --gold data/benchmark/gold_eval_2.jsonl \
  --pred artifacts/prediction/predictions_eval_v3_strict_json_vote.jsonl \
  --mode normalized \
  --min_n 10 \
  --out artifacts/benchmark/feature_errors.json
```

Run all-missing evaluation prompts for a benchmark language set:

```bash
python benchmark/eval_prompting.py \
  --prompting_py glottolog-tree/prompting.py \
  --typ data/derived/uriel+_typological.csv \
  --meta data/derived/metadata.csv \
  --topk_csv data/features/topk_per_feature.csv \
  --gen data/derived/genetic_neighbours.json \
  --gen_detail data/derived/genetic_neighbours_detailed.json \
  --geo data/derived/geographic_neighbours.json \
  --groups_json data/benchmark/language_groups_2.json \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --device cuda \
  --batch_size 4 \
  --max_new_tokens 96 \
  --prompts_out artifacts/benchmark/eval_missing_prompts.jsonl \
  --pred_out artifacts/benchmark/eval_missing_predictions.jsonl
```

## SLURM Entry Points

There are two batch wrappers in `scripts/`:

- `scripts/run_eval.sbatch`: legacy prompt benchmark (`code/prompting.py`)
- `scripts/run_glottolog_benchmark.sbatch`: tree/KG benchmark (`glottolog-tree/prompting.py`)

Examples:

```bash
sbatch --export=ALL,MODE=benchmark,MODEL=meta-llama/Llama-3.1-8B-Instruct scripts/run_eval.sbatch
sbatch --export=ALL,MODE=benchmark,MODEL=meta-llama/Llama-3.1-8B-Instruct,RETRIEVAL_BACKEND=kg_flat scripts/run_glottolog_benchmark.sbatch
sbatch --export=ALL,MODE=benchmark,BENCHMARK_VARIANT=coverage_resource_v1,MODEL=meta-llama/Llama-3.1-8B-Instruct scripts/run_eval.sbatch
sbatch --export=ALL,MODE=benchmark,BENCHMARK_VARIANT=coverage_resource_v1,MODEL=meta-llama/Llama-3.1-8B-Instruct,RETRIEVAL_BACKEND=kg_flat scripts/run_glottolog_benchmark.sbatch
sbatch --export=ALL,MODE=benchmark,BENCHMARK_VARIANT=coverage_sparse_resource_v1,MODEL=meta-llama/Llama-3.1-8B-Instruct scripts/run_eval.sbatch
sbatch --export=ALL,MODE=benchmark,BENCHMARK_VARIANT=coverage_sparse_resource_v1,MODEL=meta-llama/Llama-3.1-8B-Instruct,RETRIEVAL_BACKEND=kg_flat scripts/run_glottolog_benchmark.sbatch
```

Useful environment variables accepted by the scripts:

- `MODE=benchmark|full|all`
- `BENCHMARK_VARIANT=frozen_v2|coverage_resource_v1|coverage_sparse_resource_v1|custom`
- `MODEL=<hf model id>`
- `PROMPT_VERSION=<prompt version>`
- `OUT_DIR=<artifact directory>`
- `GOLD=<gold jsonl path>`
- `RETRIEVAL_BACKEND=legacy|kg_flat|kg_typed` for the Glottolog-tree script
- `VENV_PATH=<virtualenv path>`
- `PYTHON_MODULE=<module name>` and `CUDA_MODULE=<module name>`
- `EXTRA_MODULES="<space separated module names>"`
- `INFER_DEVICE=cuda|cpu`
- `BATCH_SIZE=<int>`
- `MAX_NEW_TOKENS=<int>`

Score the benchmark baselines directly on the coverage-based benchmark with:

```bash
python benchmark/score_baselines_on_gold.py \
  --gold data/benchmark/gold_eval_coverage_resource_v1.jsonl \
  --out_dir artifacts/prediction/coverage_resource_v1/baselines \
  --report_out artifacts/prediction/coverage_resource_v1/report_baselines_on_gold.json
```

## Output Conventions

- `data/` stores checked-in datasets and reusable derived resources.
- `artifacts/` is for generated predictions, reports, logs, and scratch outputs.
- `artifacts/resources/kg_nodes.jsonl` and `artifacts/resources/kg_edges.jsonl` are generated resources.
