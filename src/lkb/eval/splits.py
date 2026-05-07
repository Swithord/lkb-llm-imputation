"""Split: stratified (resource-group × feature-type) pair-level train/dev/test split.

The test (and optionally dev) set is built by sampling ``n_per_cell`` observed
(language, feature) pairs from every cell in the
    {lrl, mrl, hrl} × {feature-type prefix}
grid.  The remainder of all observed pairs is the train set.

Resource groups are assigned by ascending observed-feature count:
  - LRL: bottom ``lrl_frac`` of eligible languages
  - MRL: middle slice
  - HRL: top slice
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from lkb.interfaces import KnowledgeBase


def _feature_prefix(feature: str) -> str:
    return feature.split("_", 1)[0]


def _assign_resource_groups(
    kb: KnowledgeBase,
    *,
    lrl_frac: float = 0.33,
    mrl_frac: float = 0.34,
    min_observed: int = 1,
) -> Tuple[Dict[str, List[str]], dict]:
    """Assign languages to lrl/mrl/hrl by ascending observed-feature count."""
    if lrl_frac <= 0 or mrl_frac <= 0 or lrl_frac + mrl_frac >= 1:
        raise ValueError("Need lrl_frac>0, mrl_frac>0, lrl_frac+mrl_frac<1.")

    observed_counts = kb.observed_mask().sum(axis=1)
    order = np.argsort(observed_counts, kind="stable")
    languages = list(kb.languages)
    sorted_pairs = [(languages[i], int(observed_counts[i])) for i in order]
    eligible = [(lang, cnt) for lang, cnt in sorted_pairs if cnt >= min_observed]
    if not eligible:
        raise ValueError("No languages meet the minimum observed-feature threshold.")

    n = len(eligible)
    lrl_end = max(1, min(n - 2, int(round(n * lrl_frac))))
    mrl_end = max(lrl_end + 1, min(n - 1, int(round(n * (lrl_frac + mrl_frac)))))

    lrl = [lang for lang, _ in eligible[:lrl_end]]
    mrl = [lang for lang, _ in eligible[lrl_end:mrl_end]]
    hrl = [lang for lang, _ in eligible[mrl_end:]]
    groups = {"lrl": lrl, "mrl": mrl, "hrl": hrl}

    counts_map = dict(eligible)
    meta = {
        "fractions": {"lrl": lrl_frac, "mrl": mrl_frac, "hrl": round(1.0 - lrl_frac - mrl_frac, 6)},
        "n_languages_eligible": n,
        "groups": {
            label: {
                "n_languages": len(langs),
                "observed_min": min(counts_map[l] for l in langs),
                "observed_max": max(counts_map[l] for l in langs),
            }
            for label, langs in groups.items()
        },
    }
    return groups, meta


@dataclass(frozen=True)
class Split:
    test: List[Tuple[str, str]]   # (language, feature) pairs
    dev: List[Tuple[str, str]]    # (language, feature) pairs; empty if no dev
    resource_groups: Dict[str, List[str]]  # group name -> list of all languages in that group
    meta: dict = field(default_factory=dict)

    @classmethod
    def stratified(
        cls,
        kb: KnowledgeBase,
        *,
        n_per_cell: int = 100,
        lrl_frac: float = 0.33,
        mrl_frac: float = 0.34,
        min_observed: int = 1,
        include_dev: bool = True,
        seed: int = 42,
    ) -> "Split":
        """Build a stratified pair-level split.

        Samples ``n_per_cell`` (language, feature) pairs from each
        (resource_group, feature_type) cell for test, then another
        ``n_per_cell`` for dev (if ``include_dev``).  All remaining
        observed pairs form the implicit train set.
        """
        rng = np.random.default_rng(seed)

        resource_groups, group_meta = _assign_resource_groups(
            kb, lrl_frac=lrl_frac, mrl_frac=mrl_frac, min_observed=min_observed
        )
        lang_to_group = {
            lang: group
            for group, langs in resource_groups.items()
            for lang in langs
        }

        observed_mask = kb.observed_mask()
        languages = list(kb.languages)
        features = list(kb.features)
        rows, cols = np.where(observed_mask)

        # Bucket every observed pair into its (resource_group, feature_prefix) cell.
        cells: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        for r, c in zip(rows.tolist(), cols.tolist()):
            lang = languages[r]
            group = lang_to_group.get(lang)
            if group is None:
                continue  # below min_observed threshold, not assigned to any group
            feat = features[c]
            key = (group, _feature_prefix(feat))
            if key not in cells:
                cells[key] = []
            cells[key].append((lang, feat))

        # Shuffle each cell independently.
        for key in cells:
            arr = cells[key]
            perm = rng.permutation(len(arr))
            cells[key] = [arr[i] for i in perm]

        test_pairs: List[Tuple[str, str]] = []
        dev_pairs: List[Tuple[str, str]] = []
        cell_meta: dict = {}

        n_needed = n_per_cell * (2 if include_dev else 1)
        for key in sorted(cells):
            group, prefix = key
            pairs = cells[key]
            available = len(pairs)
            if available < n_needed:
                print(
                    f"WARNING: cell ({group}, {prefix}) has only {available} pairs "
                    f"(need {n_needed}); using all available."
                )
            test_n = min(n_per_cell, available)
            test_pairs.extend(pairs[:test_n])
            dev_n = 0
            if include_dev:
                dev_n = min(n_per_cell, available - test_n)
                dev_pairs.extend(pairs[test_n : test_n + dev_n])
            cell_meta[f"{group}/{prefix}"] = {
                "available": available,
                "test": test_n,
                "dev": dev_n,
            }

        meta = {
            "n_per_cell": n_per_cell,
            "include_dev": include_dev,
            "seed": seed,
            "resource_group_params": {
                "lrl_frac": lrl_frac,
                "mrl_frac": mrl_frac,
                "min_observed": min_observed,
            },
            "resource_groups": group_meta,
            "cells": cell_meta,
            "n_test": len(test_pairs),
            "n_dev": len(dev_pairs),
        }
        return cls(
            test=test_pairs,
            dev=dev_pairs,
            resource_groups=resource_groups,
            meta=meta,
        )

    # ---- train set ----------------------------------------------------------

    def train_pairs(self, kb: KnowledgeBase) -> List[Tuple[str, str]]:
        """All observed (language, feature) pairs not in test or dev."""
        held_out = set(self.test) | set(self.dev)
        observed_mask = kb.observed_mask()
        languages = list(kb.languages)
        features = list(kb.features)
        rows, cols = np.where(observed_mask)
        return [
            (languages[r], features[c])
            for r, c in zip(rows.tolist(), cols.tolist())
            if (languages[r], features[c]) not in held_out
        ]

    # ---- helpers ------------------------------------------------------------

    def resource_group_for(self, language: str) -> Optional[str]:
        for group, langs in self.resource_groups.items():
            if language in langs:
                return group
        return None

    # ---- serialization ------------------------------------------------------

    def to_json(self) -> dict:
        return {
            "test": [list(p) for p in self.test],
            "dev": [list(p) for p in self.dev],
            "resource_groups": {k: list(v) for k, v in self.resource_groups.items()},
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "Split":
        return cls(
            test=[tuple(p) for p in payload.get("test", [])],
            dev=[tuple(p) for p in payload.get("dev", [])],
            resource_groups={
                k: list(v)
                for k, v in (payload.get("resource_groups") or {}).items()
            },
            meta=dict(payload.get("meta") or {}),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Split":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = ["Split"]
