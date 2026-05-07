"""Post-hoc analyses over evaluated predictions.

Three analyzers:
  - ResourceSplitAnalysis: per-resource-tier accuracy / parsed / confidence summary.
  - FamilyAreaAnalysis: bucket errors by language family and macroarea.
  - FeatureErrorAnalysis: per-feature accuracy / confusion matrix.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from lkb.eval.evaluator import GoldItem, _normalize_group
from lkb.interfaces import KnowledgeBase, Prediction


def _safe_div(numer: float, denom: float) -> float:
    return numer / denom if denom else 0.0


def _is_correct(pred: Prediction, gold: GoldItem) -> Tuple[bool, bool]:
    parsed = pred.value is not None and (
        not gold.allowed_values or pred.value in gold.allowed_values
    )
    correct = parsed and pred.value == gold.gold_value
    return parsed, correct


@dataclass
class ResourceSplitAnalysis:
    gold: Sequence[GoldItem]
    predictions: Sequence[Prediction]

    def report(self) -> Dict[str, dict]:
        groups: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"n": 0, "correct": 0, "parsed": 0,
                     "high": 0, "medium": 0, "low": 0}
        )
        for g, p in zip(self.gold, self.predictions):
            key = _normalize_group(g.resource_group)
            parsed, correct = _is_correct(p, g)
            groups[key]["n"] += 1
            groups[key]["parsed"] += int(parsed)
            groups[key]["correct"] += int(correct)
            conf = (p.confidence or "").strip().lower()
            if conf in groups[key]:
                groups[key][conf] += 1
        out: Dict[str, dict] = {}
        for key, stats in groups.items():
            n = stats["n"]
            out[key] = {
                "n": n,
                "accuracy": _safe_div(stats["correct"], n),
                "parsed_rate": _safe_div(stats["parsed"], n),
                "confidence_distribution": {
                    "low": _safe_div(stats["low"], n),
                    "medium": _safe_div(stats["medium"], n),
                    "high": _safe_div(stats["high"], n),
                },
            }
        return out


@dataclass
class FamilyAreaAnalysis:
    kb: KnowledgeBase
    gold: Sequence[GoldItem]
    predictions: Sequence[Prediction]
    min_n: int = 10

    def report(self) -> Dict[str, Dict[str, dict]]:
        family_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"n": 0, "correct": 0, "parsed": 0}
        )
        area_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"n": 0, "correct": 0, "parsed": 0}
        )
        for g, p in zip(self.gold, self.predictions):
            parsed, correct = _is_correct(p, g)
            meta = self.kb.metadata_for(g.language)
            family = meta.family_name or "Unknown"
            area = meta.macroareas[0] if meta.macroareas else "Unknown"
            for stats in (family_stats[family], area_stats[area]):
                stats["n"] += 1
                stats["parsed"] += int(parsed)
                stats["correct"] += int(correct)

        def _build(stats_map: Dict[str, Dict[str, int]]) -> Dict[str, dict]:
            out: Dict[str, dict] = {}
            for key, stats in stats_map.items():
                if stats["n"] < self.min_n:
                    continue
                out[key] = {
                    "n": stats["n"],
                    "accuracy": _safe_div(stats["correct"], stats["n"]),
                    "parsed_rate": _safe_div(stats["parsed"], stats["n"]),
                }
            return out

        return {
            "by_family": _build(family_stats),
            "by_macroarea": _build(area_stats),
        }


@dataclass
class FeatureErrorAnalysis:
    gold: Sequence[GoldItem]
    predictions: Sequence[Prediction]
    min_n: int = 10

    def report(self) -> Dict[str, dict]:
        stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"n": 0, "parsed": 0, "correct": 0,
                     "tp": 0, "fp": 0, "tn": 0, "fn": 0}
        )
        for g, p in zip(self.gold, self.predictions):
            parsed, correct = _is_correct(p, g)
            s = stats[g.feature]
            s["n"] += 1
            s["parsed"] += int(parsed)
            s["correct"] += int(correct)
            if parsed and g.gold_value in {"0", "1"} and p.value in {"0", "1"}:
                yi, pi = int(g.gold_value), int(p.value)
                if yi == 1 and pi == 1:
                    s["tp"] += 1
                elif yi == 0 and pi == 1:
                    s["fp"] += 1
                elif yi == 0 and pi == 0:
                    s["tn"] += 1
                else:
                    s["fn"] += 1

        out: Dict[str, dict] = {}
        for feat, s in stats.items():
            if s["n"] < self.min_n:
                continue
            precision = _safe_div(s["tp"], s["tp"] + s["fp"])
            recall = _safe_div(s["tp"], s["tp"] + s["fn"])
            f1 = _safe_div(2.0 * precision * recall, precision + recall)
            out[feat] = {
                "n": s["n"],
                "accuracy": _safe_div(s["correct"], s["n"]),
                "parsed_rate": _safe_div(s["parsed"], s["n"]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion": {"tp": s["tp"], "fp": s["fp"], "tn": s["tn"], "fn": s["fn"]},
            }
        return out


__all__ = ["ResourceSplitAnalysis", "FamilyAreaAnalysis", "FeatureErrorAnalysis"]
