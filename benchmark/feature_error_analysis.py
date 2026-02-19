from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


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
    p = argparse.ArgumentParser(description="Per-feature error analysis for benchmark predictions.")
    p.add_argument("--gold", type=str, required=True)
    p.add_argument("--pred", type=str, required=True)
    p.add_argument("--mode", type=str, default="normalized", choices=["normalized", "strict_raw"])
    p.add_argument("--min_n", type=int, default=10)
    p.add_argument("--out", type=str, default="benchmark/feature_errors.json")
    args = p.parse_args()

    gold: Dict[str, dict] = {}
    for line in Path(args.gold).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        gold[str(rec["id"])] = rec

    pred: Dict[str, dict] = {}
    for line in Path(args.pred).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        pid = rec.get("id")
        if pid:
            pred[str(pid)] = rec

    stats: Dict[str, dict] = {}
    for pid, grec in gold.items():
        feat = str(grec["feature"])
        group = str(grec["resource_group"])
        gval = str(grec["gold_value"])
        allowed = set(str(x) for x in grec.get("allowed_values", []))
        prec = pred.get(pid, {})
        if args.mode == "strict_raw":
            pval, _ = _parse_from_output(prec)
        else:
            pval, _ = _parse_from_fields(prec)
            if pval is None:
                pval, _ = _parse_from_output(prec)

        parsed = pval is not None and (not allowed or pval in allowed)
        correct = parsed and pval == gval

        obj = stats.setdefault(
            feat,
            {
                "n": 0,
                "parsed": 0,
                "correct": 0,
                "high_n": 0,
                "high_correct": 0,
                "low_n": 0,
                "low_correct": 0,
            },
        )
        obj["n"] += 1
        obj["parsed"] += int(parsed)
        obj["correct"] += int(correct)
        if group == "high":
            obj["high_n"] += 1
            obj["high_correct"] += int(correct)
        elif group == "low":
            obj["low_n"] += 1
            obj["low_correct"] += int(correct)

    rows = []
    for feat, s in stats.items():
        if s["n"] < args.min_n:
            continue
        rows.append(
            {
                "feature": feat,
                "n": s["n"],
                "parsed_rate": _safe_div(s["parsed"], s["n"]),
                "accuracy": _safe_div(s["correct"], s["n"]),
                "high_n": s["high_n"],
                "high_accuracy": _safe_div(s["high_correct"], s["high_n"]),
                "low_n": s["low_n"],
                "low_accuracy": _safe_div(s["low_correct"], s["low_n"]),
                "gap_high_minus_low": _safe_div(s["high_correct"], s["high_n"]) - _safe_div(s["low_correct"], s["low_n"]),
            }
        )

    rows.sort(key=lambda x: (x["accuracy"], -x["n"]))
    payload = {
        "mode": args.mode,
        "min_n": args.min_n,
        "num_features": len(rows),
        "features": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
