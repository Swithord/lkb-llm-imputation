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

`build_benchmark.py` now emits:
- `benchmark/examples_eval.jsonl`
- `benchmark/examples_eval_high.jsonl`
- `benchmark/examples_eval_low.jsonl`

## Convert existing prompt/gold files

```bash
.venv/bin/python benchmark/convert_prompt_gold_to_examples.py \
  --prompts benchmark/prompts_eval_2.jsonl \
  --gold benchmark/gold_eval_2.jsonl \
  --out benchmark/examples_eval_2.jsonl
```

## Inference compatibility

`infer_from_prompts.py` accepts both:
- prompt-style rows (`system` + `user`), and
- canonical rows (`example_id/language_id/feature_id/resource_group/r_L`).
