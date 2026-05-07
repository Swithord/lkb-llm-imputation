"""Split: language-level train/dev/test and 3-tier resource groupings.

Two kinds of splits produced here:

1. **Language-level** train/dev/test partition for cross-validation / benchmarking
   (MCAR entry mask is generated in sync with the split).

2. **Resource 3-tier** lrl/mrl/hrl grouping by per-language observed-feature
   count, used to segment benchmark reports by typological coverage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from lkb.interfaces import KnowledgeBase


@dataclass(frozen=True)
class Split:
    train: List[str]
    dev: List[str]
    test: List[str]
    resource_groups: Dict[str, List[str]] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @classmethod
    def language_split(
        cls,
        kb: KnowledgeBase,
        *,
        train_frac: float = 0.8,
        dev_frac: float = 0.1,
        seed: int = 42,
        require_observed: bool = True,
    ) -> "Split":
        if train_frac <= 0 or dev_frac <= 0 or train_frac + dev_frac >= 1:
            raise ValueError("Need train_frac>0, dev_frac>0, train_frac+dev_frac<1.")

        rng = np.random.default_rng(seed)
        observed = kb.observed_mask()
        has_obs = observed.any(axis=1)
        langs = [
            lang for i, lang in enumerate(kb.languages)
            if (not require_observed) or has_obs[i]
        ]
        arr = np.array(langs, dtype=object)
        rng.shuffle(arr)

        n = len(arr)
        n_train = int(n * train_frac)
        n_dev = int(n * dev_frac)
        train = arr[:n_train].tolist()
        dev = arr[n_train : n_train + n_dev].tolist()
        test = arr[n_train + n_dev :].tolist()

        meta = {
            "split_level": "language",
            "seed": seed,
            "fractions": {
                "train": train_frac,
                "dev": dev_frac,
                "test": 1.0 - train_frac - dev_frac,
            },
            "n_languages_total": int(len(kb.languages)),
            "n_languages_with_observed": int(has_obs.sum()),
        }
        return cls(train=train, dev=dev, test=test, meta=meta)

    def with_resource_groups(
        self,
        kb: KnowledgeBase,
        *,
        lrl_frac: float = 0.33,
        mrl_frac: float = 0.34,
        min_observed: int = 1,
    ) -> "Split":
        groups, group_meta = _coverage_groups(
            kb, lrl_frac=lrl_frac, mrl_frac=mrl_frac, min_observed=min_observed
        )
        meta = dict(self.meta)
        meta["resource_groups"] = group_meta
        return Split(
            train=list(self.train),
            dev=list(self.dev),
            test=list(self.test),
            resource_groups=groups,
            meta=meta,
        )

    # ---- MCAR mask ----------------------------------------------------------

    def mcar_mask(
        self, kb: KnowledgeBase, *, mask_rate: float = 0.2, seed: int = 42
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        observed = kb.observed_mask()
        observed_idx = np.argwhere(observed)
        n_mask = int(round(mask_rate * observed_idx.shape[0]))
        if n_mask <= 0:
            return np.zeros(observed.shape, dtype=bool)
        pick = rng.choice(observed_idx.shape[0], size=n_mask, replace=False)
        chosen = observed_idx[pick]
        mask = np.zeros(observed.shape, dtype=bool)
        mask[chosen[:, 0], chosen[:, 1]] = True
        return mask

    # ---- serialization ------------------------------------------------------

    def to_json(self) -> dict:
        return {
            "train": list(self.train),
            "dev": list(self.dev),
            "test": list(self.test),
            "resource_groups": {k: list(v) for k, v in self.resource_groups.items()},
            "meta": dict(self.meta),
        }

    @classmethod
    def from_json(cls, payload: dict) -> "Split":
        return cls(
            train=list(payload.get("train", [])),
            dev=list(payload.get("dev", [])),
            test=list(payload.get("test", [])),
            resource_groups={k: list(v) for k, v in (payload.get("resource_groups") or {}).items()},
            meta=dict(payload.get("meta") or {}),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Split":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    # ---- helpers ------------------------------------------------------------

    def resource_group_for(self, language: str) -> Optional[str]:
        for group, langs in self.resource_groups.items():
            if language in langs:
                return group
        return None


def _coverage_groups(
    kb: KnowledgeBase,
    *,
    lrl_frac: float,
    mrl_frac: float,
    min_observed: int,
) -> tuple[Dict[str, List[str]], dict]:
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

    group_meta: dict = {
        "split_definition": "coverage_quantiles_on_languages_with_observed_features",
        "min_observed_threshold": int(min_observed),
        "fractions": {
            "lrl": lrl_frac,
            "mrl": mrl_frac,
            "hrl": 1.0 - lrl_frac - mrl_frac,
        },
        "n_languages_total": int(len(languages)),
        "n_languages_eligible": int(n),
        "groups": {},
    }
    counts_map = dict(eligible)
    n_features = int(kb.observed_mask().shape[1])
    for label, langs in groups.items():
        if not langs:
            raise ValueError(f"Coverage split produced an empty group: {label}")
        gcounts = [counts_map[lang] for lang in langs]
        group_meta["groups"][label] = {
            "n_languages": len(langs),
            "observed_count_min": min(gcounts),
            "observed_count_max": max(gcounts),
            "observed_count_avg": sum(gcounts) / len(gcounts),
            "missing_count_min": n_features - max(gcounts),
            "missing_count_max": n_features - min(gcounts),
            "missing_count_avg": n_features - (sum(gcounts) / len(gcounts)),
        }
    return groups, group_meta


__all__ = ["Split"]
