from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from schema_io import load_gold_jsonl, load_predictions_jsonl

DEFAULT_CONF = {"low": 0.33, "medium": 0.66, "high": 0.90}
VALID_CONF = {"low", "medium", "high"}


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _parse_from_fields(obj: dict) -> Tuple[str | None, str | None, str | None]:
    if "value" in obj:
        value = obj.get("value")
        conf = obj.get("confidence")
        rat = obj.get("rationale")
        return (
            None if value is None else str(value),
            None if conf is None else str(conf).lower(),
            None if rat is None else str(rat),
        )
    return (None, None, None)


def _parse_from_output(obj: dict) -> Tuple[str | None, str | None, str | None]:
    raw = obj.get("output")
    if raw is None:
        return (None, None, None)
    text = str(raw).strip()
    try:
        parsed = json.loads(text)
        value = parsed.get("value")
        conf = parsed.get("confidence")
        rat = parsed.get("rationale")
        return (
            None if value is None else str(value),
            None if conf is None else str(conf).lower(),
            None if rat is None else str(rat),
        )
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                value = parsed.get("value")
                conf = parsed.get("confidence")
                rat = parsed.get("rationale")
                return (
                    None if value is None else str(value),
                    None if conf is None else str(conf).lower(),
                    None if rat is None else str(rat),
                )
            except Exception:
                pass
    return (None, None, None)


def _parse_prediction(obj: dict, mode: str) -> Tuple[str | None, str | None, str | None]:
    if mode == "strict_raw":
        return _parse_from_output(obj)
    value, conf, rat = _parse_from_fields(obj)
    if value is None and conf is None and rat is None:
        return _parse_from_output(obj)
    return (value, conf, rat)


def _init_groups() -> Dict[str, Dict[str, int]]:
    return {
        "overall": {"n": 0, "correct": 0, "parsed": 0, "high_conf_n": 0, "high_conf_correct": 0, "rationale_ok": 0},
        "high": {"n": 0, "correct": 0, "parsed": 0, "high_conf_n": 0, "high_conf_correct": 0, "rationale_ok": 0},
        "low": {"n": 0, "correct": 0, "parsed": 0, "high_conf_n": 0, "high_conf_correct": 0, "rationale_ok": 0},
    }


def _evaluate_subset(
    ids: Sequence[str],
    gold: Dict[str, dict],
    pred: Dict[str, dict],
    mode: str,
    conf_map: Dict[str, float],
) -> dict:
    groups = _init_groups()
    brier_sum = 0.0
    brier_n = 0
    ece_bins = [{"n": 0, "acc_sum": 0.0, "conf_sum": 0.0} for _ in range(10)]

    for gid in ids:
        grec = gold[gid]
        group = grec["resource_group"]
        gval = str(grec["gold_value"])
        allowed = set(str(x) for x in grec.get("allowed_values", []))
        pval, pconf, prat = _parse_prediction(pred.get(gid, {}), mode)

        for key in ("overall", group):
            groups[key]["n"] += 1

        parsed = pval is not None and (not allowed or pval in allowed)
        correct = parsed and pval == gval
        rationale_ok = bool(prat and len(prat.split()) <= 30)

        for key in ("overall", group):
            groups[key]["parsed"] += int(parsed)
            groups[key]["correct"] += int(correct)
            groups[key]["rationale_ok"] += int(rationale_ok)

        if pconf in VALID_CONF and parsed:
            c = conf_map[pconf]
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
                    groups[key]["high_conf_correct"] += int(correct)

    ece = 0.0
    if brier_n:
        for b in ece_bins:
            if b["n"] == 0:
                continue
            acc = b["acc_sum"] / b["n"]
            conf = b["conf_sum"] / b["n"]
            ece += (b["n"] / brier_n) * abs(acc - conf)

    metrics = {}
    for k, v in groups.items():
        metrics[k] = {
            "accuracy": _safe_div(v["correct"], v["n"]),
            "parsed_rate": _safe_div(v["parsed"], v["n"]),
            "high_conf_accuracy": _safe_div(v["high_conf_correct"], v["high_conf_n"]),
            "high_conf_coverage": _safe_div(v["high_conf_n"], v["n"]),
            "rationale_ok_rate": _safe_div(v["rationale_ok"], v["n"]),
        }

    return {
        "counts": {k: {"n": v["n"]} for k, v in groups.items()},
        "metrics": metrics,
        "calibration": {
            "brier_on_correctness": _safe_div(brier_sum, brier_n),
            "ece_10bin": ece,
            "n_with_valid_confidence": brier_n,
        },
    }


def _fit_conf_map(
    ids: Sequence[str], gold: Dict[str, dict], pred: Dict[str, dict], mode: str
) -> Tuple[Dict[str, float], dict]:
    counts = {k: {"n": 0, "correct": 0} for k in ("low", "medium", "high")}
    total_n = 0
    total_correct = 0
    for gid in ids:
        grec = gold[gid]
        gval = str(grec["gold_value"])
        allowed = set(str(x) for x in grec.get("allowed_values", []))
        pval, pconf, _ = _parse_prediction(pred.get(gid, {}), mode)
        if pconf not in VALID_CONF:
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
    conf_map = dict(DEFAULT_CONF)
    for k in ("low", "medium", "high"):
        n = counts[k]["n"]
        conf_map[k] = (counts[k]["correct"] / n) if n else base_rate
    return conf_map, {"counts": counts, "base_rate": base_rate, "n_scored": total_n}


def _split_ids(
    gold: Dict[str, dict],
    test_frac: float,
    seed: int,
    split_by: str,
) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    ids = list(gold.keys())
    if split_by == "item":
        rng.shuffle(ids)
        cut = max(1, int(round(len(ids) * test_frac)))
        test_ids = ids[:cut]
        train_ids = ids[cut:]
        return train_ids, test_ids

    by_group_lang = {"high": [], "low": []}
    seen = {"high": set(), "low": set()}
    for rec in gold.values():
        g = rec["resource_group"]
        lang = rec["language"]
        if g in by_group_lang and lang not in seen[g]:
            seen[g].add(lang)
            by_group_lang[g].append(lang)

    test_langs = set()
    for g in ("high", "low"):
        langs = by_group_lang[g]
        rng.shuffle(langs)
        if not langs:
            continue
        cut = max(1, int(round(len(langs) * test_frac)))
        cut = min(cut, len(langs) - 1) if len(langs) > 1 else 1
        test_langs.update(langs[:cut])

    test_ids = [gid for gid, rec in gold.items() if rec["language"] in test_langs]
    train_ids = [gid for gid in ids if gid not in set(test_ids)]
    return train_ids, test_ids


def main() -> None:
    p = argparse.ArgumentParser(description="Split-based confidence calibration (train/test split, no leakage).")
    p.add_argument("--gold", type=str, required=True)
    p.add_argument("--pred", type=str, required=True)
    p.add_argument("--mode", type=str, default="normalized", choices=["normalized", "strict_raw"])
    p.add_argument("--split_by", type=str, default="language", choices=["language", "item"])
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=str, default="artifacts/benchmark/split_calibration_report.json")
    args = p.parse_args()

    gold = load_gold_jsonl(args.gold)
    pred = load_predictions_jsonl(args.pred)

    train_ids, test_ids = _split_ids(gold, args.test_frac, args.seed, args.split_by)
    if not train_ids or not test_ids:
        raise RuntimeError("Split failed: empty train or test set.")

    fitted_map, fit_info = _fit_conf_map(train_ids, gold, pred, args.mode)
    test_uncal = _evaluate_subset(test_ids, gold, pred, args.mode, DEFAULT_CONF)
    test_cal = _evaluate_subset(test_ids, gold, pred, args.mode, fitted_map)

    payload = {
        "mode": args.mode,
        "split": {
            "split_by": args.split_by,
            "test_frac": args.test_frac,
            "seed": args.seed,
            "train_n": len(train_ids),
            "test_n": len(test_ids),
        },
        "train_fit": fit_info,
        "confidence_map_fitted": fitted_map,
        "test_uncalibrated": test_uncal,
        "test_calibrated": test_cal,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
