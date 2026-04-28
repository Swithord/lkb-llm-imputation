"""Evaluator: score predictions against gold, segmented by resource group."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from lkb.eval.calibration import Calibration
from lkb.eval.metrics import compute_binary_metrics
from lkb.eval.splits import Split
from lkb.interfaces import Prediction


def _safe_div(numer: float, denom: float) -> float:
    return numer / denom if denom else 0.0


def _normalize_group(value: object) -> str:
    raw = str(value or "").strip().lower()
    aliases = {"high": "hrl", "mid": "mrl", "middle": "mrl", "medium": "mrl", "low": "lrl"}
    return aliases.get(raw, raw or "unknown")


@dataclass(frozen=True)
class GoldItem:
    id: str
    language: str
    feature: str
    gold_value: str
    resource_group: str = "unknown"
    allowed_values: tuple[str, ...] = ("0", "1")


class Evaluator:
    """Score (GoldItem, Prediction) pairs and emit overall + per-group metrics."""

    def __init__(
        self,
        *,
        calibration: Optional[Calibration] = None,
    ) -> None:
        self.calibration = calibration or Calibration.default()

    # ---- gold item construction --------------------------------------------

    @staticmethod
    def gold_items_from_mask(
        kb,
        mask,
        *,
        split: Optional[Split] = None,
        limit_to_test: bool = True,
    ) -> List[GoldItem]:
        """Gold items from a boolean MCAR mask on an ``observed`` KB.

        When a ``Split`` with a language-level test set is provided and
        ``limit_to_test=True``, only entries for test-set languages are emitted.
        """
        import numpy as np

        mask = mask.astype(bool, copy=False)
        languages = list(kb.languages)
        features = list(kb.features)
        matrix = kb.as_matrix()
        test_set = set(split.test) if (split and limit_to_test) else None

        items: List[GoldItem] = []
        rows, cols = np.where(mask)
        for r, c in zip(rows.tolist(), cols.tolist()):
            lang = languages[r]
            if test_set is not None and lang not in test_set:
                continue
            val = int(matrix[r, c])
            if val not in (0, 1):
                continue
            group = (split.resource_group_for(lang) if split else None) or "unknown"
            items.append(
                GoldItem(
                    id=f"{lang}:{features[c]}",
                    language=lang,
                    feature=features[c],
                    gold_value=str(val),
                    resource_group=_normalize_group(group),
                )
            )
        return items

    # ---- scoring ------------------------------------------------------------

    def evaluate(
        self,
        gold: Sequence[GoldItem],
        predictions: Sequence[Prediction],
    ) -> dict:
        if len(gold) != len(predictions):
            raise ValueError("gold and predictions must have identical lengths.")

        groups: Dict[str, Dict[str, int]] = {"overall": _new_group_stats()}
        corr_for_cal: List[int] = []
        prob_for_cal: List[float] = []

        for g, pred in zip(gold, predictions):
            group = _normalize_group(g.resource_group)
            groups.setdefault(group, _new_group_stats())

            pval = pred.value
            pconf = (pred.confidence or "").strip().lower()
            prat = pred.rationale
            allowed = set(g.allowed_values or ())
            parsed = pval is not None and (not allowed or pval in allowed)
            correct = parsed and pval == g.gold_value
            rationale_ok = bool(prat and len(prat.split()) <= 30)

            for key in ("overall", group):
                stats = groups[key]
                stats["n"] += 1
                stats["parsed"] += int(parsed)
                stats["correct"] += int(correct)
                stats["rationale_ok"] += int(rationale_ok)

            if parsed and g.gold_value in {"0", "1"} and pval in {"0", "1"}:
                yi = int(g.gold_value)
                pi = int(pval)
                for key in ("overall", group):
                    stats = groups[key]
                    if yi == 1 and pi == 1:
                        stats["tp"] += 1
                    elif yi == 0 and pi == 1:
                        stats["fp"] += 1
                    elif yi == 0 and pi == 0:
                        stats["tn"] += 1
                    else:
                        stats["fn"] += 1

            if parsed and pconf in {"low", "medium", "high"}:
                p_correct = self.calibration.probability(pconf)
                corr_for_cal.append(int(correct))
                prob_for_cal.append(float(p_correct))
                if pconf == "high":
                    for key in ("overall", group):
                        stats = groups[key]
                        stats["high_conf_n"] += 1
                        stats["high_conf_correct"] += int(correct)

        if prob_for_cal:
            cal_pred = [1 if p >= 0.5 else 0 for p in prob_for_cal]
            cal_metrics = compute_binary_metrics(
                corr_for_cal, cal_pred, y_prob=prob_for_cal, include_ece=True, ece_bins=10
            )
            brier = float(cal_metrics["brier"])
            ece = float(cal_metrics["ece"])
        else:
            brier = 0.0
            ece = 0.0

        report = {
            "confidence_map": dict(self.calibration.mapping),
            "counts": {k: {"n": v["n"]} for k, v in groups.items()},
            "metrics": {},
            "calibration": {
                "brier_on_correctness": brier,
                "ece_10bin": ece,
                "n_with_valid_confidence": len(prob_for_cal),
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


def _new_group_stats() -> Dict[str, int]:
    return {
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
    }


__all__ = ["Evaluator", "GoldItem"]
