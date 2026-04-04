import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.run_softimpute import _resolve_decision_thresholds, _soft_impute_probabilities


def test_softimpute_probabilities_preserve_observed_entries_across_transforms() -> None:
    X = np.array(
        [
            [1.0, np.nan, 0.0],
            [1.0, 1.0, np.nan],
            [0.0, 0.0, 0.0],
            [np.nan, 1.0, 1.0],
        ],
        dtype=float,
    )
    observed = ~np.isnan(X)

    for transform in ["raw", "centered", "standardized"]:
        probs = _soft_impute_probabilities(
            X,
            max_rank=2,
            max_iters=10,
            convergence_threshold=1e-6,
            shrinkage_value=0.01,
            input_transform=transform,
        )
        assert probs.shape == X.shape
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        assert np.allclose(probs[observed], X[observed])


def test_resolve_decision_thresholds_supports_prevalence_modes() -> None:
    train_matrix = np.array(
        [
            [1.0, 0.0, np.nan],
            [1.0, np.nan, 0.0],
            [0.0, 0.0, 0.0],
            [np.nan, 1.0, 1.0],
        ],
        dtype=float,
    )

    feature_thresholds, global_rate = _resolve_decision_thresholds(
        train_matrix,
        threshold_mode="feature_prevalence",
        fixed_threshold=0.5,
    )
    global_thresholds, _ = _resolve_decision_thresholds(
        train_matrix,
        threshold_mode="global_prevalence",
        fixed_threshold=0.5,
    )
    fixed_thresholds, _ = _resolve_decision_thresholds(
        train_matrix,
        threshold_mode="fixed",
        fixed_threshold=0.3,
    )

    assert np.allclose(feature_thresholds, np.array([2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))
    assert np.allclose(global_thresholds, np.full(3, global_rate))
    assert np.allclose(fixed_thresholds, np.full(3, 0.3))
