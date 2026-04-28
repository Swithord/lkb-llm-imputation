"""CLI: measure LLM inference throughput from a prompts JSONL.

Usage:
  python -m lkb.scripts.speed_bench \\
    --in artifacts/prediction/prompts.jsonl \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --device cuda \\
    --batch_size 4 \\
    --max_new_tokens 64 \\
    --report_out artifacts/prediction/speed_report.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

from lkb.impute.llm import HFClient
from lkb.interfaces import PromptPayload


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure LLM inference throughput.")
    parser.add_argument("--in", dest="input", required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--report_out", required=True)
    args = parser.parse_args()

    items: List[dict] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    llm = HFClient(args.model, max_new_tokens=args.max_new_tokens, device=args.device, dtype=args.dtype)
    payloads = [PromptPayload(system=it["system"], user=it["user"]) for it in items]

    batch_size = max(1, args.batch_size)
    n = len(payloads)
    t0 = time.perf_counter()

    for i in range(0, n, batch_size):
        llm.complete(payloads[i : i + batch_size])

    elapsed = time.perf_counter() - t0
    throughput = n / elapsed if elapsed > 0 else 0.0

    report = {
        "model": args.model,
        "n_prompts": n,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "elapsed_s": round(elapsed, 3),
        "throughput_prompts_per_s": round(throughput, 4),
    }

    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"n={n}  elapsed={elapsed:.1f}s  throughput={throughput:.2f} prompts/s")
    print(f"Wrote speed report -> {args.report_out}")


if __name__ == "__main__":
    main()
