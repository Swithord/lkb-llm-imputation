"""
A prediction is "unparsed" when its `value` is null, meaning infer.py
couldn't extract a valid JSON value from the raw model output.

Workflow:
  1) extract: pull the unparsed ids' prompts into a small subset file.
  2) run infer.py on the subset with the same model/flags as the original run.
  3) merge: splice the subset predictions back into the original file.
  4) rerun evaluate.py on the merged file to refresh the report.

Usage:
  python -m lkb.scripts.rerun_unparsed extract \\
    --pred artifacts/prediction/predictions_eval_v4_strict_json_base.jsonl \\
    --prompts artifacts/prediction/prompts_eval_v4_strict_json_base.jsonl \\
    --out artifacts/prediction/prompts_eval_v4_strict_json_base_retry.jsonl

  python -m lkb.scripts.infer \\
    --in artifacts/prediction/prompts_eval_v4_strict_json_base_retry.jsonl \\
    --out artifacts/prediction/predictions_eval_v4_strict_json_base_retry.jsonl \\
    --model meta-llama/Llama-3.1-70B-Instruct ...

  python -m lkb.scripts.rerun_unparsed merge \\
    --pred artifacts/prediction/predictions_eval_v4_strict_json_base.jsonl \\
    --retry artifacts/prediction/predictions_eval_v4_strict_json_base_retry.jsonl \\
    --out artifacts/prediction/predictions_eval_v4_strict_json_base.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _read_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _write_jsonl(path: str, items: List[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _is_unparsed(pred: dict) -> bool:
    return not pred.get("parsed_ok", pred.get("value") is not None)


def cmd_extract(args: argparse.Namespace) -> None:
    predictions = _read_jsonl(args.pred)
    prompts = _read_jsonl(args.prompts)

    unparsed_ids = {p["id"] for p in predictions if _is_unparsed(p)}
    if not unparsed_ids:
        print(f"No unparsed predictions found in {args.pred}")
        return

    prompts_by_id = {p["id"]: p for p in prompts}
    missing = unparsed_ids - prompts_by_id.keys()
    if missing:
        raise ValueError(
            f"{len(missing)} unparsed id(s) have no matching prompt in {args.prompts}, "
            f"e.g. {sorted(missing)[:5]}"
        )

    # Preserve prompts-file order for readability.
    subset = [p for p in prompts if p["id"] in unparsed_ids]
    _write_jsonl(args.out, subset)
    print(
        f"{len(unparsed_ids)}/{len(predictions)} unparsed -> wrote {len(subset)} prompts to {args.out}"
    )


def cmd_merge(args: argparse.Namespace) -> None:
    predictions = _read_jsonl(args.pred)
    retry: Dict[str, dict] = {r["id"]: r for r in _read_jsonl(args.retry)}

    unknown = set(retry) - {p["id"] for p in predictions}
    if unknown:
        raise ValueError(
            f"{len(unknown)} retry id(s) not found in {args.pred}, e.g. {sorted(unknown)[:5]}"
        )

    merged = []
    replaced = 0
    still_unparsed = 0
    for pred in predictions:
        if pred["id"] in retry:
            new_pred = retry[pred["id"]]
            merged.append(new_pred)
            replaced += 1
            if _is_unparsed(new_pred):
                still_unparsed += 1
        else:
            merged.append(pred)

    _write_jsonl(args.out, merged)
    print(
        f"Replaced {replaced} prediction(s) -> wrote {len(merged)} to {args.out} "
        f"({still_unparsed} still unparsed after retry)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Find and rerun unparsed predictions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Write a prompts subset for unparsed ids")
    extract.add_argument("--pred", required=True, help="Predictions JSONL to scan")
    extract.add_argument("--prompts", required=True, help="Full prompts JSONL used to generate --pred")
    extract.add_argument("--out", required=True, help="Output subset prompts JSONL path")
    extract.set_defaults(func=cmd_extract)

    merge = subparsers.add_parser("merge", help="Splice retried predictions back into the original file")
    merge.add_argument("--pred", required=True, help="Original predictions JSONL")
    merge.add_argument("--retry", required=True, help="Predictions JSONL produced from the retry subset")
    merge.add_argument("--out", required=True, help="Output merged predictions JSONL path (may equal --pred)")
    merge.set_defaults(func=cmd_merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
