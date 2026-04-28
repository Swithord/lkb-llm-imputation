"""CLI: run HF inference over a prompts JSONL, emit predictions JSONL.

Usage:
  python -m lkb.scripts.infer \\
    --in artifacts/prediction/prompts.jsonl \\
    --out artifacts/prediction/predictions.jsonl \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --device cuda \\
    --batch_size 4 \\
    --max_new_tokens 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

from lkb.impute.llm import HFClient
from lkb.impute.prompts.icl import ICLPrompt
from lkb.interfaces import PromptPayload


def _parse_prediction(raw: str) -> dict:
    """Best-effort JSON parse of the raw model output."""
    import re

    text = raw.strip()
    # Try to find the first {...} block
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM inference on prompts JSONL.")
    parser.add_argument("--in", dest="input", required=True, help="Prompts JSONL path")
    parser.add_argument("--out", required=True, help="Predictions JSONL output path")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--dtype", default=None)
    # Kept for CLI compatibility; not used in new code
    parser.add_argument("--stop_at_closing_brace", action="store_true", default=False)
    parser.add_argument("--no-strengthen_json_contract", dest="strengthen_json_contract",
                        action="store_false", default=True)
    args = parser.parse_args()

    items: List[dict] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if not items:
        print("No prompts found in input; exiting.")
        sys.exit(0)

    llm = HFClient(
        args.model,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
    )
    # Use ICLPrompt.parse for response parsing (it handles both ICL and KG JSON)
    _parser = ICLPrompt()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(items)
    batch_size = max(1, args.batch_size)

    with out_path.open("w", encoding="utf-8") as out_f:
        for batch_start in range(0, n, batch_size):
            batch = items[batch_start : batch_start + batch_size]
            payloads = [PromptPayload(system=it["system"], user=it["user"]) for it in batch]
            raws = llm.complete(payloads)

            for it, raw in zip(batch, raws):
                pred = _parser.parse(raw)
                record = {
                    "id": it["id"],
                    "language": it["language"],
                    "feature": it["feature"],
                    "gold_value": it.get("gold_value"),
                    "resource_group": it.get("resource_group", "unknown"),
                    "value": pred.value,
                    "confidence": pred.confidence,
                    "rationale": pred.rationale,
                    "raw": pred.raw,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            done = min(batch_start + batch_size, n)
            print(f"[{done}/{n}]", flush=True)

    print(f"Wrote {n} predictions -> {args.out}")


if __name__ == "__main__":
    main()
