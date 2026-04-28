"""Classical baselines: Random, Mean (majority), kNN."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from lkb.interfaces import Imputer, KnowledgeBase, Prediction


def _prob_to_confidence(prob_correct: float) -> str:
    if prob_correct >= 0.8:
        return "high"
    if prob_correct >= 0.6:
        return "medium"
    return "low"


def _feature_prevalence(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-feature and global P(value=1) over observed entries. -1 encodes missing."""
    observed = matrix != -1
    total = int(observed.sum())
    global_p1 = float((matrix == 1).sum() / total) if total else 0.5

    pos_counts = ((matrix == 1) & observed).sum(axis=0)
    obs_counts = observed.sum(axis=0)
    feat_p1 = np.full(matrix.shape[1], global_p1, dtype=float)
    np.divide(pos_counts, obs_counts, out=feat_p1, where=obs_counts > 0)
    return feat_p1, global_p1


class _PrevalenceImputerMixin:
    def __init__(self) -> None:
        self._feat_p1: Optional[np.ndarray] = None
        self._global_p1: float = 0.5
        self._feat_to_col: dict[str, int] = {}
        self._lang_to_row: dict[str, int] = {}

    def _fit_prevalence(self, kb: KnowledgeBase) -> None:
        self._feat_p1, self._global_p1 = _feature_prevalence(kb.as_matrix())
        self._feat_to_col = {f: j for j, f in enumerate(kb.features)}
        self._lang_to_row = {l: i for i, l in enumerate(kb.languages)}

    def _p1_for(self, feature: str) -> float:
        col = self._feat_to_col.get(feature)
        if col is None or self._feat_p1 is None:
            return self._global_p1
        return float(self._feat_p1[col])


class RandomImputer(_PrevalenceImputerMixin, Imputer):
    """Bernoulli draw with per-feature positive-rate from the fitted KB."""

    name = "random"

    def __init__(self, *, seed: int = 42) -> None:
        super().__init__()
        self._rng = np.random.default_rng(seed)

    def fit(self, kb: KnowledgeBase) -> None:
        self._fit_prevalence(kb)

    def impute(
        self, kb: KnowledgeBase, pairs: Sequence[tuple[str, str]]
    ) -> list[Prediction]:
        out: list[Prediction] = []
        for _, feat in pairs:
            p1 = self._p1_for(feat)
            pred = int(self._rng.binomial(n=1, p=p1))
            prob_correct = p1 if pred == 1 else 1.0 - p1
            out.append(
                Prediction(
                    value=str(pred),
                    confidence=_prob_to_confidence(prob_correct),
                    rationale="random baseline",
                )
            )
        return out


class MeanImputer(_PrevalenceImputerMixin, Imputer):
    """Majority prediction using per-feature positive-rate."""

    name = "mean"

    def fit(self, kb: KnowledgeBase) -> None:
        self._fit_prevalence(kb)

    def impute(
        self, kb: KnowledgeBase, pairs: Sequence[tuple[str, str]]
    ) -> list[Prediction]:
        out: list[Prediction] = []
        for _, feat in pairs:
            p1 = self._p1_for(feat)
            pred = 1 if p1 >= 0.5 else 0
            prob_correct = p1 if pred == 1 else 1.0 - p1
            out.append(
                Prediction(
                    value=str(pred),
                    confidence=_prob_to_confidence(prob_correct),
                    rationale="mean baseline",
                )
            )
        return out


class KNNImputer(_PrevalenceImputerMixin, Imputer):
    """kNN over observed feature profiles. Cosine or Jaccard over co-observed features."""

    name = "knn"

    def __init__(self, *, k: int = 5, metric: str = "cosine") -> None:
        super().__init__()
        if metric not in {"cosine", "jaccard"}:
            raise ValueError(f"Unsupported metric: {metric}")
        self.k = k
        self.metric = metric
        self._matrix: Optional[np.ndarray] = None
        self._rank_cache: dict[int, np.ndarray] = {}

    def fit(self, kb: KnowledgeBase) -> None:
        self._fit_prevalence(kb)
        self._matrix = kb.as_matrix()
        self._rank_cache = {}

    def _similarity(self, query: np.ndarray) -> np.ndarray:
        assert self._matrix is not None
        train = self._matrix
        query_pos = query == 1
        query_obs = query != -1
        train_pos = train == 1
        train_obs = train != -1
        common = train_obs & query_obs[None, :]

        inter = (train_pos & query_pos[None, :] & common).sum(axis=1).astype(float)
        q_cnt = (query_pos[None, :] & common).sum(axis=1).astype(float)
        t_cnt = (train_pos & common).sum(axis=1).astype(float)

        if self.metric == "cosine":
            denom = np.sqrt(q_cnt * t_cnt)
            return np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0)
        # jaccard
        union = q_cnt + t_cnt - inter
        return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

    def _ranked_rows(self, row: int) -> np.ndarray:
        if row not in self._rank_cache:
            assert self._matrix is not None
            sim = self._similarity(self._matrix[row])
            sim[row] = -np.inf  # exclude self
            self._rank_cache[row] = np.argsort(-sim)
        return self._rank_cache[row]

    def impute(
        self, kb: KnowledgeBase, pairs: Sequence[tuple[str, str]]
    ) -> list[Prediction]:
        if self._matrix is None:
            raise RuntimeError("KNNImputer.fit must be called before impute.")
        out: list[Prediction] = []
        for lang, feat in pairs:
            row = self._lang_to_row.get(lang)
            col = self._feat_to_col.get(feat)
            if row is None or col is None:
                p1 = self._global_p1
            else:
                order = self._ranked_rows(row)
                col_vals = self._matrix[order, col]
                valid = col_vals[col_vals != -1]
                if valid.size == 0:
                    p1 = self._p1_for(feat)
                else:
                    use = valid[: min(self.k, valid.size)]
                    p1 = float(use.mean())
            pred = 1 if p1 >= 0.5 else 0
            prob_correct = p1 if pred == 1 else 1.0 - p1
            out.append(
                Prediction(
                    value=str(pred),
                    confidence=_prob_to_confidence(prob_correct),
                    rationale=f"knn(k={self.k},{self.metric}) baseline",
                )
            )
        return out
