from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from schema_io import load_gold_jsonl


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _extract_structured_description(user_prompt: str) -> str:
    marker = "\nTask:"
    if marker in user_prompt:
        return user_prompt.split(marker, 1)[0].rstrip()
    return user_prompt.rstrip()


def _parse_binary(gold_value: str, item_id: str) -> int:
    try:
        val = int(float(gold_value))
    except Exception as e:
        raise ValueError(f"Gold value is not numeric for {item_id}: {gold_value}") from e
    if val not in (0, 1):
        raise ValueError(f"Gold value is not binary for {item_id}: {gold_value}")
    return val


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert prompt+gold JSONL into canonical examples JSONL."
    )
    p.add_argument("--prompts", type=str, required=True)
    p.add_argument("--gold", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    prompts_by_id: Dict[str, dict] = {}
    for rec in _read_jsonl(args.prompts):
        rid = rec.get("id")
        if rid:
            prompts_by_id[str(rid)] = rec

    gold = load_gold_jsonl(args.gold)
    out_rows: list[dict] = []
    for gid, grec in gold.items():
        prompt = prompts_by_id.get(gid)
        if prompt is None:
            raise KeyError(f"Prompt missing for id={gid}")
        user_prompt = str(prompt.get("user", ""))
        out_rows.append(
            {
                "example_id": gid,
                "language_id": str(grec.get("language", "")),
                "feature_id": str(grec.get("feature", "")),
                "gold": _parse_binary(str(grec["gold_value"]), gid),
                "resource_group": str(grec.get("resource_group", "")),
                "r_L": _extract_structured_description(user_prompt),
            }
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in out_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote canonical examples: {args.out}")
    print(f"Num examples: {len(out_rows)}")


if __name__ == "__main__":
    main()
