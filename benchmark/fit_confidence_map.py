from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple

from schema_io import load_gold_jsonl, load_predictions_jsonl

DEFAULT_CONF = {"low": 0.33, "medium": 0.66, "high": 0.90}


def _parse_from_fields(obj: dict) -> Tuple[str | None, str | None]:
    if "value" in obj:
        value = obj.get("value")
        conf = obj.get("confidence")
        return (None if value is None else str(value), None if conf is None else str(conf).lower())
    return (None, None)


def _parse_from_output(obj: dict) -> Tuple[str | None, str | None]:
    raw = obj.get("output")
    if raw is None:
        return (None, None)
    text = str(raw).strip()
    try:
        parsed = json.loads(text)
        value = parsed.get("value")
        conf = parsed.get("confidence")
        return (None if value is None else str(value), None if conf is None else str(conf).lower())
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                value = parsed.get("value")
                conf = parsed.get("confidence")
                return (None if value is None else str(value), None if conf is None else str(conf).lower())
            except Exception:
                pass
    return (None, None)


def main() -> None:
    p = argparse.ArgumentParser(description="Fit low/medium/high confidence -> probability map from labeled predictions.")
    p.add_argument("--gold", type=str, required=True)
    p.add_argument("--pred", type=str, required=True)
    p.add_argument("--mode", type=str, default="normalized", choices=["normalized", "strict_raw"])
    p.add_argument("--out", type=str, default="benchmark/confidence_map.json")
    args = p.parse_args()

    gold = load_gold_jsonl(args.gold)
    pred = load_predictions_jsonl(args.pred)

    counts = {k: {"n": 0, "correct": 0} for k in ("low", "medium", "high")}
    total_correct = 0
    total_n = 0

    for pid, grec in gold.items():
        gval = str(grec["gold_value"])
        allowed = set(str(x) for x in grec.get("allowed_values", []))
        prec = pred.get(pid, {})
        if args.mode == "strict_raw":
            pval, pconf = _parse_from_output(prec)
        else:
            pval, pconf = _parse_from_fields(prec)
            if pval is None and pconf is None:
                pval, pconf = _parse_from_output(prec)

        if pconf not in counts:
            continue
        parsed = pval is not None and (not allowed or pval in allowed)
        if not parsed:
            continue

        correct = pval == gval
        counts[pconf]["n"] += 1
        counts[pconf]["correct"] += int(correct)
        total_n += 1
        total_correct += int(correct)

    base_rate = (total_correct / total_n) if total_n else 0.5
    mapping = dict(DEFAULT_CONF)
    for key in ("low", "medium", "high"):
        n = counts[key]["n"]
        if n > 0:
            mapping[key] = counts[key]["correct"] / n
        else:
            mapping[key] = base_rate

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": args.mode,
        "base_rate": base_rate,
        "counts": counts,
        "map": mapping,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
