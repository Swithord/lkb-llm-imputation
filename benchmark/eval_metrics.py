from __future__ import annotations

from typing import Sequence


def _safe_div(numer: float, denom: float) -> float:
    return numer / denom if denom else 0.0


def confidence_label_to_probability(label: str, mapping: dict[str, float] | None = None) -> float:
    """Map low/medium/high confidence labels to probabilities."""
    default_map = {"low": 0.33, "medium": 0.66, "high": 0.90}
    scores = mapping or default_map
    return float(scores.get(str(label).lower(), 0.5))


def expected_calibration_error(y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10) -> float:
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have identical lengths.")
    n = len(y_true)
    if n == 0:
        return 0.0

    bins = [{"n": 0, "correct": 0.0, "conf": 0.0} for _ in range(n_bins)]
    for yt, yp in zip(y_true, y_prob):
        p = min(1.0, max(0.0, float(yp)))
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx]["n"] += 1
        bins[idx]["correct"] += float(int(yt))
        bins[idx]["conf"] += p

    ece = 0.0
    for b in bins:
        if b["n"] == 0:
            continue
        avg_acc = b["correct"] / b["n"]
        avg_conf = b["conf"] / b["n"]
        ece += (b["n"] / n) * abs(avg_acc - avg_conf)
    return ece


def compute_binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float] | None = None,
    include_ece: bool = True,
    ece_bins: int = 10,
) -> dict[str, float]:
    """
    Compute shared binary classification metrics:
    accuracy, precision, recall, F1, Brier, and optional ECE.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have identical lengths.")

    n = len(y_true)
    tp = fp = tn = fn = 0
    for yt_raw, yp_raw in zip(y_true, y_pred):
        yt = int(yt_raw)
        yp = int(yp_raw)
        if yt not in (0, 1) or yp not in (0, 1):
            raise ValueError("Binary metrics require labels in {0,1}.")
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        else:
            fn += 1

    accuracy = _safe_div(tp + tn, n)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    result: dict[str, float] = {
        "n": float(n),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": 0.0,
    }

    if y_prob is not None:
        if len(y_prob) != n:
            raise ValueError("y_prob must have same length as y_true.")
        clipped = [min(1.0, max(0.0, float(p))) for p in y_prob]
        brier = _safe_div(sum((p - float(y)) ** 2 for y, p in zip(y_true, clipped)), n)
        result["brier"] = brier
        if include_ece:
            result["ece"] = expected_calibration_error(y_true, clipped, n_bins=ece_bins)
    else:
        result["brier"] = float("nan")
        if include_ece:
            result["ece"] = float("nan")

    return result
