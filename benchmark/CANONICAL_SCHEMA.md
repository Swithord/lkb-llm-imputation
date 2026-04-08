# Canonical Example Schema

Use this JSONL schema as the shared interface for Stream A and Stream B:

```json
{
  "example_id": "high:sepa1242:S_WORD_ORDER:0",
  "language_id": "sepa1242",
  "feature_id": "S_WORD_ORDER",
  "gold": 0,
  "resource_group": "high",
  "r_L": "<structured language description text>"
}
```

Notes:
- `gold` is binary (`0` or `1`).
- `r_L` is the structured context text (target language metadata + observed facts + neighbor evidence), without task/output instructions.

## Produce canonical examples

`build_benchmark.py` now defaults to the frozen `_2` benchmark artifact names:
- `data/benchmark/examples_eval_2.jsonl`
- `data/benchmark/examples_eval_high_2.jsonl`
- `data/benchmark/examples_eval_low_2.jsonl`

The checked-in frozen benchmark used for reported comparisons is:
- `data/benchmark/gold_eval_2.jsonl`
- `data/benchmark/language_groups_2.json`

## Convert existing prompt/gold files

```bash
.venv/bin/python benchmark/convert_prompt_gold_to_examples.py \
  --prompts data/benchmark/prompts_eval_2.jsonl \
  --gold data/benchmark/gold_eval_2.jsonl \
  --out data/benchmark/examples_eval_2.jsonl
```

## Inference compatibility

`infer_from_prompts.py` accepts both:
- prompt-style rows (`system` + `user`), and
- canonical rows (`example_id/language_id/feature_id/resource_group/r_L`).
