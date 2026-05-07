"""SoftImpute baseline: iterative low-rank matrix completion via singular-value shrinkage."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from lkb.interfaces import Imputer, KnowledgeBase, Prediction

try:
    from sklearn.utils.extmath import randomized_svd as _randomized_svd
except Exception:  # pragma: no cover
    _randomized_svd = None


def _prob_to_confidence(prob_correct: float) -> str:
    if prob_correct >= 0.8:
        return "high"
    if prob_correct >= 0.6:
        return "medium"
    return "low"


def _truncated_svd(
    X: np.ndarray, n_components: int, n_iter: int, random_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _randomized_svd is not None:
        return _randomized_svd(
            X, n_components=n_components, n_iter=n_iter, random_state=random_state
        )
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    rank = min(int(n_components), len(s))
    return U[:, :rank], s[:rank], Vt[:rank, :]


def _feature_prevalence_from_nan(X: np.ndarray) -> tuple[np.ndarray, float]:
    observed = ~np.isnan(X)
    counts = observed.sum(axis=0)
    total = int(observed.sum())
    global_p1 = float(np.nansum(X) / total) if total else 0.5
    feat_p1 = np.full(X.shape[1], global_p1, dtype=float)
    np.divide(np.nansum(X, axis=0), counts, out=feat_p1, where=counts > 0)
    return np.clip(feat_p1, 0.0, 1.0), float(np.clip(global_p1, 0.0, 1.0))


def _prepare_input(
    X: np.ndarray, input_transform: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = ~np.isnan(X)
    feat_means, _ = _feature_prevalence_from_nan(X)
    feat_scales = np.ones(X.shape[1], dtype=float)

    if input_transform == "raw":
        return np.array(X, dtype=float, copy=True), np.zeros(X.shape[1]), feat_scales

    X_work = np.array(X, dtype=float, copy=True)
    X_work[observed] = X_work[observed] - np.take(feat_means, np.where(observed)[1])

    if input_transform == "centered":
        return X_work, feat_means, feat_scales

    if input_transform == "standardized":
        counts = observed.sum(axis=0)
        centered_sq = np.zeros(X.shape[1], dtype=float)
        centered_vals = X_work[observed]
        np.add.at(centered_sq, np.where(observed)[1], centered_vals * centered_vals)
        denom = np.maximum(counts - 1, 1)
        feat_scales = np.sqrt(centered_sq / denom)
        feat_scales = np.where((counts > 1) & (feat_scales > 1e-8), feat_scales, 1.0)
        X_work[observed] = X_work[observed] / np.take(feat_scales, np.where(observed)[1])
        return X_work, feat_means, feat_scales

    raise ValueError(f"Unsupported input_transform: {input_transform}")


def _restore(X_completed: np.ndarray, offsets: np.ndarray, scales: np.ndarray) -> np.ndarray:
    restored = np.array(X_completed, dtype=float, copy=True)
    restored *= scales[None, :]
    restored += offsets[None, :]
    return np.clip(restored, 0.0, 1.0)


def _soft_impute(
    X: np.ndarray,
    max_rank: Optional[int],
    max_iters: int,
    convergence_threshold: float,
    shrinkage_value: Optional[float],
) -> np.ndarray:
    X_work = np.array(X, dtype=float, copy=True)
    missing = np.isnan(X_work)
    X_work[missing] = 0.0

    if shrinkage_value is None:
        _, s0, _ = _truncated_svd(X_work, n_components=1, n_iter=5, random_state=0)
        shrinkage = float(s0[0]) / 50.0
    else:
        shrinkage = float(shrinkage_value)

    for _ in range(max_iters):
        old_missing = X_work[missing].copy()
        if max_rank is not None:
            rank = min(int(max_rank), min(X_work.shape))
            U, s, Vt = _truncated_svd(X_work, n_components=rank, n_iter=1, random_state=0)
        else:
            U, s, Vt = np.linalg.svd(X_work, full_matrices=False)

        s_shrunk = np.maximum(s - shrinkage, 0.0)
        active = int((s_shrunk > 0).sum())
        if active == 0:
            break
        U = U[:, :active]
        Vt = Vt[:active, :]
        s_shrunk = s_shrunk[:active]
        X_recon = U @ np.diag(s_shrunk) @ Vt
        X_work[missing] = X_recon[missing]

        new_missing = X_work[missing]
        diff = old_missing - new_missing
        old_norm = float(np.sqrt((old_missing ** 2).sum()))
        delta = float(np.sqrt((diff ** 2).sum()))
        if old_norm > 0 and (delta / old_norm) < convergence_threshold:
            break

    return X_work


class SoftImputeImputer(Imputer):
    """SoftImpute-equivalent iterative low-rank matrix completion."""

    name = "softimpute"

    def __init__(
        self,
        *,
        max_rank: Optional[int] = 64,
        max_iters: int = 100,
        convergence_threshold: float = 1e-5,
        shrinkage_value: Optional[float] = None,
        input_transform: str = "centered",
        threshold_mode: str = "fixed",
        decision_threshold: float = 0.5,
    ) -> None:
        if input_transform not in {"raw", "centered", "standardized"}:
            raise ValueError(f"Unsupported input_transform: {input_transform}")
        if threshold_mode not in {"fixed", "feature_prevalence", "global_prevalence"}:
            raise ValueError(f"Unsupported threshold_mode: {threshold_mode}")
        self.max_rank = max_rank
        self.max_iters = max_iters
        self.convergence_threshold = convergence_threshold
        self.shrinkage_value = shrinkage_value
        self.input_transform = input_transform
        self.threshold_mode = threshold_mode
        self.decision_threshold = decision_threshold

        self._probs: Optional[np.ndarray] = None
        self._thresholds: Optional[np.ndarray] = None
        self._feat_to_col: dict[str, int] = {}
        self._lang_to_row: dict[str, int] = {}
        self._global_p1: float = 0.5

    def fit(self, kb: KnowledgeBase) -> None:
        matrix = kb.as_matrix().astype(float)
        X = np.where(matrix == -1, np.nan, matrix)

        feat_p1, global_p1 = _feature_prevalence_from_nan(X)
        if self.threshold_mode == "fixed":
            thresholds = np.full(X.shape[1], float(np.clip(self.decision_threshold, 0.0, 1.0)))
        elif self.threshold_mode == "feature_prevalence":
            thresholds = feat_p1.astype(float, copy=True)
        else:
            thresholds = np.full(X.shape[1], global_p1, dtype=float)
        self._thresholds = np.clip(thresholds, 0.0, 1.0)
        self._global_p1 = global_p1

        X_prepared, offsets, scales = _prepare_input(X, input_transform=self.input_transform)
        completed = _soft_impute(
            X_prepared,
            max_rank=self.max_rank,
            max_iters=self.max_iters,
            convergence_threshold=self.convergence_threshold,
            shrinkage_value=self.shrinkage_value,
        )
        self._probs = _restore(completed, offsets=offsets, scales=scales)
        self._feat_to_col = {f: j for j, f in enumerate(kb.features)}
        self._lang_to_row = {l: i for i, l in enumerate(kb.languages)}

    def impute(
        self, kb: KnowledgeBase, pairs: Sequence[tuple[str, str]]
    ) -> list[Prediction]:
        if self._probs is None or self._thresholds is None:
            raise RuntimeError("SoftImputeImputer.fit must be called before impute.")
        out: list[Prediction] = []
        for lang, feat in pairs:
            row = self._lang_to_row.get(lang)
            col = self._feat_to_col.get(feat)
            if row is None or col is None:
                p = self._global_p1
                thr = 0.5
            else:
                p = float(np.clip(self._probs[row, col], 0.0, 1.0))
                thr = float(self._thresholds[col])
            pred = 1 if p >= thr else 0
            prob_correct = p if pred == 1 else 1.0 - p
            out.append(
                Prediction(
                    value=str(pred),
                    confidence=_prob_to_confidence(prob_correct),
                    rationale="softimpute matrix completion",
                )
            )
        return out
