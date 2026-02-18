from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple


CONF_SCORE = {"low": 0.33, "medium": 0.66, "high": 0.90}


def _parse_pred_line(obj: dict) -> Tuple[str | None, str | None, str | None]:
    if "value" in obj:
        value = obj.get("value")
        conf = obj.get("confidence")
        rat = obj.get("rationale")
        return (None if value is None else str(value), None if conf is None else str(conf).lower(), None if rat is None else str(rat))

    raw = obj.get("output")
    if raw is None:
        return (None, None, None)
    text = str(raw).strip()
    try:
        parsed = json.loads(text)
        value = parsed.get("value")
        conf = parsed.get("confidence")
        rat = parsed.get("rationale")
        return (None if value is None else str(value), None if conf is None else str(conf).lower(), None if rat is None else str(rat))
    except Exception:
        m = re.search(r"\{.*\}", text)
        if m:
            try:
                parsed = json.loads(m.group(0))
                value = parsed.get("value")
                conf = parsed.get("confidence")
                rat = parsed.get("rationale")
                return (None if value is None else str(value), None if conf is None else str(conf).lower(), None if rat is None else str(rat))
            except Exception:
                pass
    return (None, None, None)


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate benchmark predictions with confidence/rationale checks.")
    p.add_argument("--gold", type=str, default="benchmark/gold_eval.jsonl")
    p.add_argument("--pred", type=str, required=True, help="JSONL with `id` and either (`value`,`confidence`,`rationale`) or `output`.")
    p.add_argument("--report_out", type=str, default=None, help="Optional JSON report path.")
    args = p.parse_args()

    gold: Dict[str, dict] = {}
    for line in Path(args.gold).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        gold[rec["id"]] = rec

    pred: Dict[str, dict] = {}
    for line in Path(args.pred).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        pid = rec.get("id")
        if pid:
            pred[str(pid)] = rec

    groups = {
        "overall": {"n": 0, "correct": 0, "parsed": 0, "high_conf_n": 0, "high_conf_correct": 0, "rationale_ok": 0},
        "high": {"n": 0, "correct": 0, "parsed": 0, "high_conf_n": 0, "high_conf_correct": 0, "rationale_ok": 0},
        "low": {"n": 0, "correct": 0, "parsed": 0, "high_conf_n": 0, "high_conf_correct": 0, "rationale_ok": 0},
    }

    brier_sum = 0.0
    brier_n = 0
    ece_bins = [{"n": 0, "acc_sum": 0.0, "conf_sum": 0.0} for _ in range(10)]

    for gid, grec in gold.items():
        group = grec["resource_group"]
        gval = str(grec["gold_value"])
        allowed = set(str(x) for x in grec.get("allowed_values", []))
        prec = pred.get(gid, {})
        pval, pconf, prat = _parse_pred_line(prec)

        for key in ("overall", group):
            groups[key]["n"] += 1

        parsed = pval is not None and (not allowed or pval in allowed)
        correct = parsed and pval == gval
        rationale_ok = bool(prat and len(prat.split()) <= 30)

        for key in ("overall", group):
            if parsed:
                groups[key]["parsed"] += 1
            if correct:
                groups[key]["correct"] += 1
            if rationale_ok:
                groups[key]["rationale_ok"] += 1

        if pconf in CONF_SCORE and parsed:
            c = CONF_SCORE[pconf]
            y = 1.0 if correct else 0.0
            brier_sum += (c - y) ** 2
            brier_n += 1

            bi = min(9, max(0, int(c * 10)))
            ece_bins[bi]["n"] += 1
            ece_bins[bi]["acc_sum"] += y
            ece_bins[bi]["conf_sum"] += c

            for key in ("overall", group):
                if pconf == "high":
                    groups[key]["high_conf_n"] += 1
                    if correct:
                        groups[key]["high_conf_correct"] += 1

    ece = 0.0
    if brier_n:
        for b in ece_bins:
            if b["n"] == 0:
                continue
            acc = b["acc_sum"] / b["n"]
            conf = b["conf_sum"] / b["n"]
            ece += (b["n"] / brier_n) * abs(acc - conf)

    report = {
        "counts": {k: {"n": v["n"]} for k, v in groups.items()},
        "metrics": {},
        "calibration": {
            "brier_on_correctness": _safe_div(brier_sum, brier_n),
            "ece_10bin": ece,
            "n_with_valid_confidence": brier_n,
        },
    }

    for k, v in groups.items():
        report["metrics"][k] = {
            "accuracy": _safe_div(v["correct"], v["n"]),
            "parsed_rate": _safe_div(v["parsed"], v["n"]),
            "high_conf_accuracy": _safe_div(v["high_conf_correct"], v["high_conf_n"]),
            "high_conf_coverage": _safe_div(v["high_conf_n"], v["n"]),
            "rationale_ok_rate": _safe_div(v["rationale_ok"], v["n"]),
        }

    print(json.dumps(report, indent=2))

    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
