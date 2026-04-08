# Benchmark Notes

This directory contains the scripts used to build, run, and score the repo's prompt-based imputation benchmarks.

## Frozen Benchmark Artifacts

The current frozen benchmark used for direct model and retrieval comparisons is:

- `data/benchmark/gold_eval_2.jsonl`
- `data/benchmark/language_groups_2.json`

Properties of this frozen set:

- `1433` total items
- `1000` items labeled `high`
- `433` items labeled `low`
- `50` high-resource languages
- `50` low-resource languages

## What Is Standardized

When experiments are run through `benchmark/build_prompts_from_gold.py` with `data/benchmark/gold_eval_2.jsonl`, the following are fixed:

- the sampled evaluation items
- the high/low resource-group labels
- the output JSON schema
- the evaluation metrics

This is the setup used by:

- `scripts/run_eval.sbatch`
- `scripts/run_glottolog_benchmark.sbatch`

## What Is Not Fully Standardized

The benchmark builder is not a fully stratified protocol in the stronger methodological sense:

- there is no `mrl` tier
- feature-family balance is not explicitly enforced
- family-balance or family-exclusion constraints are not explicitly enforced

So direct comparisons across reported benchmark runs are valid because they use one frozen gold file, but the benchmark should not be described as fully standardized across HRL/MRL/LRL tiers and feature/family strata.

## Benchmark Results Snapshot

Representative results on the frozen benchmark `data/benchmark/gold_eval_2.jsonl` are:

| Method | Overall Acc | Overall F1 | High Acc | Low Acc |
|---|---:|---:|---:|---:|
| Mean | 78.65 | 67.24 | 81.20 | 72.75 |
| kNN | 83.18 | 74.44 | 88.80 | 70.21 |
| SoftImpute | 85.35 | 78.40 | 90.80 | 72.75 |
| Legacy prompt / RAG-style (`v4_8b`) | 82.83 | 76.21 | 83.10 | 82.22 |
| `kg_flat` | 79.69 | 73.03 | 80.00 | 78.98 |
| `hybrid_flat_kg_fixed` | 82.14 | 75.57 | 82.40 | 81.52 |
| `kg_typed_fixed` | 90.93 | 87.45 | 89.60 | 94.00 |
| `compact_fixed` | 91.21 | 87.77 | 89.90 | 94.23 |
| `kg_typed_contrastive_fixed` | **91.49** | **88.13** | **90.20** | **94.46** |

Interpretation:

- the strongest classical baseline here is SoftImpute
- the strongest legacy prompt baseline is `v4_8b`
- the strongest KG variant is `kg_typed_contrastive_fixed`
- the main empirical jump comes from typed, feature-conditioned, contrastive retrieval rather than from graph storage alone

Short note on KG variant differences:

- `kg_flat`: baseline-preserving KG implementation of the old flat retrieval logic
- `hybrid_flat_kg_fixed`: legacy flat retrieval as the backbone, with KG evidence added as augmentation
- `kg_typed_fixed`: typed, feature-conditioned KG reranking using relation type, distance, and feature overlap signals
- `compact_fixed`: same typed KG retrieval family, but with a more compact graph serialization in the prompt
- `kg_typed_contrastive_fixed`: typed KG retrieval plus explicit contrastive support for both competing values

## Rebuilding

`benchmark/build_benchmark.py` can generate a new benchmark snapshot. Its default output names now use the `_2` suffix to match the checked-in frozen artifact naming convention.

If you regenerate these files, treat the result as a new benchmark snapshot unless you intentionally replace the frozen files.
