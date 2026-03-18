from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_metrics import compute_binary_metrics
from schema_io import load_gold_jsonl, load_predictions_jsonl


DEFAULT_CONF_SCORE = {"low": 0.33, "medium": 0.66, "high": 0.90}


def _parse_from_fields(obj: dict) -> Tuple[str | None, str | None, str | None]:
    if "value" in obj:
        value = obj.get("value")
        conf = obj.get("confidence")
        rat = obj.get("rationale")
        return (None if value is None else str(value), None if conf is None else str(conf).lower(), None if rat is None else str(rat))
    return (None, None, None)


def _extract_last_json_object(text: str) -> dict | None:
    s = str(text).strip()
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    candidates: list[dict] = []
    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    seg_start = -1
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                seg_start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and seg_start >= 0:
                    seg = s[seg_start : i + 1]
                    try:
                        obj = json.loads(seg)
                        if isinstance(obj, dict):
                            candidates.append(obj)
                    except Exception:
                        pass
    if not candidates:
        return None
    return candidates[-1]


def _parse_from_output(obj: dict) -> Tuple[str | None, str | None, str | None]:
    raw = obj.get("output")
    if raw is None:
        return (None, None, None)
    parsed = _extract_last_json_object(str(raw))
    if isinstance(parsed, dict):
        value = parsed.get("value")
        conf = parsed.get("confidence")
        rat = parsed.get("rationale")
        return (
            None if value is None else str(value),
            None if conf is None else str(conf).lower(),
            None if rat is None else str(rat),
        )
    return (None, None, None)


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _init_groups() -> Dict[str, Dict[str, int]]:
    return {
        "overall": {
            "n": 0,
            "correct": 0,
            "parsed": 0,
            "high_conf_n": 0,
            "high_conf_correct": 0,
            "rationale_ok": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        },
        "high": {
            "n": 0,
            "correct": 0,
            "parsed": 0,
            "high_conf_n": 0,
            "high_conf_correct": 0,
            "rationale_ok": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        },
        "low": {
            "n": 0,
            "correct": 0,
            "parsed": 0,
            "high_conf_n": 0,
            "high_conf_correct": 0,
            "rationale_ok": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        },
    }


def _evaluate_view(gold: Dict[str, dict], pred: Dict[str, dict], view: str, conf_score: Dict[str, float]) -> dict:
    groups = _init_groups()
    corr_for_cal: list[int] = []
    prob_for_cal: list[float] = []

    for gid, grec in gold.items():
        group = grec["resource_group"]
        gval = str(grec["gold_value"])
        allowed = set(str(x) for x in grec.get("allowed_values", []))
        prec = pred.get(gid, {})
        if view == "strict_raw":
            pval, pconf, prat = _parse_from_output(prec)
        else:
            pval, pconf, prat = _parse_from_fields(prec)
            if pval is None and pconf is None and prat is None:
                pval, pconf, prat = _parse_from_output(prec)

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

        if parsed and gval in {"0", "1"} and pval in {"0", "1"}:
            yi = int(gval)
            pi = int(pval)
            for key in ("overall", group):
                if yi == 1 and pi == 1:
                    groups[key]["tp"] += 1
                elif yi == 0 and pi == 1:
                    groups[key]["fp"] += 1
                elif yi == 0 and pi == 0:
                    groups[key]["tn"] += 1
                else:
                    groups[key]["fn"] += 1

        if pconf in conf_score and parsed:
            c = conf_score[pconf]
            y = 1.0 if correct else 0.0
            corr_for_cal.append(int(y))
            prob_for_cal.append(float(c))

            for key in ("overall", group):
                if pconf == "high":
                    groups[key]["high_conf_n"] += 1
                    if correct:
                        groups[key]["high_conf_correct"] += 1

    if prob_for_cal:
        # Dummy labels from probabilities are only used to satisfy API shape checks.
        cal_pred = [1 if p >= 0.5 else 0 for p in prob_for_cal]
        cal_metrics = compute_binary_metrics(corr_for_cal, cal_pred, y_prob=prob_for_cal, include_ece=True, ece_bins=10)
        brier = float(cal_metrics["brier"])
        ece = float(cal_metrics["ece"])
        brier_n = len(prob_for_cal)
    else:
        brier = 0.0
        ece = 0.0
        brier_n = 0

    report = {
        "counts": {k: {"n": v["n"]} for k, v in groups.items()},
        "metrics": {},
        "calibration": {
            "brier_on_correctness": brier,
            "ece_10bin": ece,
            "n_with_valid_confidence": brier_n,
        },
    }

    for k, v in groups.items():
        precision = _safe_div(v["tp"], v["tp"] + v["fp"])
        recall = _safe_div(v["tp"], v["tp"] + v["fn"])
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        report["metrics"][k] = {
            "accuracy": _safe_div(v["correct"], v["n"]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "parsed_rate": _safe_div(v["parsed"], v["n"]),
            "high_conf_accuracy": _safe_div(v["high_conf_correct"], v["high_conf_n"]),
            "high_conf_coverage": _safe_div(v["high_conf_n"], v["n"]),
            "rationale_ok_rate": _safe_div(v["rationale_ok"], v["n"]),
        }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate benchmark predictions with confidence/rationale checks.")
    p.add_argument("--gold", type=str, default="benchmark/gold_eval.jsonl")
    p.add_argument("--pred", type=str, required=True, help="JSONL with `id` and either (`value`,`confidence`,`rationale`) or `output`.")
    p.add_argument("--confidence_map", type=str, default=None, help="Optional JSON file mapping low/medium/high to probability.")
    p.add_argument("--report_out", type=str, default=None, help="Optional JSON report path.")
    args = p.parse_args()

    conf_score = dict(DEFAULT_CONF_SCORE)
    if args.confidence_map:
        loaded = json.loads(Path(args.confidence_map).read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and "map" in loaded and isinstance(loaded["map"], dict):
            loaded = loaded["map"]
        for key in ("low", "medium", "high"):
            if key in loaded:
                conf_score[key] = float(loaded[key])

    gold = load_gold_jsonl(args.gold)
    pred = load_predictions_jsonl(args.pred)

    normalized = _evaluate_view(gold, pred, "normalized", conf_score)
    strict_raw = _evaluate_view(gold, pred, "strict_raw", conf_score)

    report = {
        "confidence_map": conf_score,
        "normalized": normalized,
        "strict_raw": strict_raw,
        "legacy": normalized,
    }

    print(json.dumps(report, indent=2))

    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
