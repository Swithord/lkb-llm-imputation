"""Confidence calibration: fit {low, medium, high} -> probability from held-out data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from lkb.eval.metrics import DEFAULT_CONFIDENCE_MAP


_VALID = ("low", "medium", "high")


@dataclass
class Calibration:
    """Map confidence labels -> probability of being correct.

    ``Calibration.default()`` uses the fixed map {low:0.33, medium:0.66, high:0.90}.
    ``Calibration.fit(...)`` learns the map empirically from (confidence, correct) pairs.
    """

    mapping: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CONFIDENCE_MAP))
    info: dict = field(default_factory=dict)

    @classmethod
    def default(cls) -> "Calibration":
        return cls(mapping=dict(DEFAULT_CONFIDENCE_MAP))

    @classmethod
    def fit(
        cls,
        confidences: Sequence[Optional[str]],
        correct: Sequence[bool],
    ) -> "Calibration":
        if len(confidences) != len(correct):
            raise ValueError("confidences and correct must have identical lengths.")

        counts = {k: {"n": 0, "correct": 0} for k in _VALID}
        total_n = 0
        total_correct = 0
        for conf, ok in zip(confidences, correct):
            key = (conf or "").strip().lower()
            if key not in counts:
                continue
            counts[key]["n"] += 1
            counts[key]["correct"] += int(bool(ok))
            total_n += 1
            total_correct += int(bool(ok))

        base_rate = (total_correct / total_n) if total_n else 0.5
        mapping = dict(DEFAULT_CONFIDENCE_MAP)
        for k in _VALID:
            n = counts[k]["n"]
            mapping[k] = (counts[k]["correct"] / n) if n else base_rate

        info = {
            "counts": counts,
            "base_rate": base_rate,
            "n_scored": total_n,
        }
        return cls(mapping=mapping, info=info)

    def probability(self, confidence: Optional[str]) -> float:
        key = (confidence or "").strip().lower()
        return float(self.mapping.get(key, 0.5))

    def apply(self, confidences: Iterable[Optional[str]]) -> list[float]:
        return [self.probability(c) for c in confidences]

    def save(self, path: str | Path) -> None:
        payload = {"mapping": dict(self.mapping), "info": dict(self.info)}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "map" in data and isinstance(data["map"], dict):
            data = {"mapping": data["map"], "info": data.get("info", {})}
        return cls(
            mapping={k: float(v) for k, v in (data.get("mapping") or {}).items()},
            info=dict(data.get("info") or {}),
        )


__all__ = ["Calibration"]
