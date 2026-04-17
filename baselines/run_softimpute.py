from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.utils.extmath import randomized_svd as _sklearn_randomized_svd
except Exception:
    _sklearn_randomized_svd = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BENCHMARK_DIR = ROOT / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from eval_metrics import compute_binary_metrics


def _truncated_svd(
    X: np.ndarray,
    n_components: int,
    n_iter: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _sklearn_randomized_svd is not None:
        return _sklearn_randomized_svd(X, n_components=n_components, n_iter=n_iter, random_state=random_state)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    rank = min(int(n_components), len(s))
    return U[:, :rank], s[:rank], Vt[:rank, :]


def _feature_prevalence_from_nan_matrix(X: np.ndarray) -> tuple[np.ndarray, float]:
    observed = ~np.isnan(X)
    counts = np.sum(observed, axis=0)
    total_observed = int(np.sum(observed))
    global_p1 = float(np.nansum(X) / total_observed) if total_observed else 0.5
    feat_p1 = np.full(X.shape[1], global_p1, dtype=float)
    np.divide(np.nansum(X, axis=0), counts, out=feat_p1, where=counts > 0)
    return np.clip(feat_p1, 0.0, 1.0), float(np.clip(global_p1, 0.0, 1.0))


def _prepare_softimpute_input(
    X: np.ndarray,
    input_transform: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = ~np.isnan(X)
    feat_means, _ = _feature_prevalence_from_nan_matrix(X)
    feat_scales = np.ones(X.shape[1], dtype=float)

    if input_transform == "raw":
        X_work = np.array(X, dtype=float, copy=True)
        return X_work, np.zeros(X.shape[1], dtype=float), feat_scales

    X_work = np.array(X, dtype=float, copy=True)
    X_work[observed] = X_work[observed] - np.take(feat_means, np.where(observed)[1])

    if input_transform == "centered":
        return X_work, feat_means, feat_scales

    if input_transform == "standardized":
        counts = np.sum(observed, axis=0)
        centered_sq = np.zeros(X.shape[1], dtype=float)
        centered_vals = X_work[observed]
        np.add.at(centered_sq, np.where(observed)[1], centered_vals * centered_vals)
        denom = np.maximum(counts - 1, 1)
        feat_scales = np.sqrt(centered_sq / denom)
        feat_scales = np.where((counts > 1) & (feat_scales > 1e-8), feat_scales, 1.0)
        X_work[observed] = X_work[observed] / np.take(feat_scales, np.where(observed)[1])
        return X_work, feat_means, feat_scales

    raise ValueError(f"Unsupported input_transform: {input_transform}")


def _restore_softimpute_output(
    X_completed: np.ndarray,
    offsets: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    restored = np.array(X_completed, dtype=float, copy=True)
    restored *= scales[None, :]
    restored += offsets[None, :]
    return np.clip(restored, 0.0, 1.0)


def _soft_impute_fit_transform(
    X: np.ndarray,
    max_rank: int | None = 64,
    max_iters: int = 100,
    convergence_threshold: float = 1e-5,
    shrinkage_value: float | None = None,
) -> np.ndarray:
    """
    SoftImpute-equivalent iterative low-rank matrix completion.
    """
    X_work = np.array(X, dtype=float, copy=True)
    missing_mask = np.isnan(X_work)
    X_work[missing_mask] = 0.0

    if shrinkage_value is None:
        _, s0, _ = _truncated_svd(X_work, n_components=1, n_iter=5, random_state=0)
        shrinkage = float(s0[0]) / 50.0
    else:
        shrinkage = float(shrinkage_value)

    for _ in range(max_iters):
        old_missing = X_work[missing_mask].copy()
        if max_rank is not None:
            rank = min(int(max_rank), min(X_work.shape))
            U, s, Vt = _truncated_svd(X_work, n_components=rank, n_iter=1, random_state=0)
        else:
            U, s, Vt = np.linalg.svd(X_work, full_matrices=False)

        s_shrunk = np.maximum(s - shrinkage, 0.0)
        active = int(np.sum(s_shrunk > 0))
        if active == 0:
            break
        U = U[:, :active]
        Vt = Vt[:active, :]
        s_shrunk = s_shrunk[:active]
        X_recon = U @ np.diag(s_shrunk) @ Vt
        X_work[missing_mask] = X_recon[missing_mask]

        new_missing = X_work[missing_mask]
        diff = old_missing - new_missing
        old_norm = np.sqrt(np.sum(old_missing**2))
        delta = np.sqrt(np.sum(diff**2))
        if old_norm > 0 and (delta / old_norm) < convergence_threshold:
            break

    return X_work


def _soft_impute_probabilities(
    X: np.ndarray,
    max_rank: int | None = 64,
    max_iters: int = 100,
    convergence_threshold: float = 1e-5,
    shrinkage_value: float | None = None,
    input_transform: str = "centered",
) -> np.ndarray:
    X_prepared, offsets, scales = _prepare_softimpute_input(X, input_transform=input_transform)
    X_completed = _soft_impute_fit_transform(
        X_prepared,
        max_rank=max_rank,
        max_iters=max_iters,
        convergence_threshold=convergence_threshold,
        shrinkage_value=shrinkage_value,
    )
    return _restore_softimpute_output(X_completed, offsets=offsets, scales=scales)


def _load_mask(path: str, expected_shape: tuple[int, int]) -> np.ndarray:
    mask_path = Path(path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")

    if mask_path.suffix == ".npy":
        mask = np.load(mask_path).astype(bool)
    elif mask_path.suffix == ".json":
        payload = json.loads(mask_path.read_text(encoding="utf-8"))
        if "mask" in payload:
            mask = np.array(payload["mask"], dtype=bool)
        elif "indices" in payload:
            mask = np.zeros(expected_shape, dtype=bool)
            for item in payload["indices"]:
                i, j = int(item[0]), int(item[1])
                mask[i, j] = True
        else:
            raise ValueError(f"Unsupported JSON mask schema in {path}")
    else:
        raise ValueError(f"Unsupported mask format: {path}")

    if mask.shape != expected_shape:
        raise ValueError(f"Mask shape mismatch for {path}: got {mask.shape}, expected {expected_shape}")
    return mask


def _langs_to_row_mask(index: pd.Index, langs: list[str]) -> np.ndarray:
    lang_set = set(langs)
    return np.array([str(x) in lang_set for x in index], dtype=bool)


def _to_binary_array(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, -1, dtype=np.int8)
    observed = values != -1
    casted = values[observed].astype(int)
    if casted.size and (~np.isin(casted, [0, 1])).any():
        raise ValueError("SoftImpute baseline currently supports binary typology values {0,1,-1} only.")
    out[observed] = casted
    return out


def _max_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return float(rss) / (1024.0 * 1024.0)
    return float(rss) / 1024.0


def _write_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _resolve_decision_thresholds(
    train_matrix: np.ndarray,
    threshold_mode: str,
    fixed_threshold: float,
) -> tuple[np.ndarray, float]:
    feat_p1, global_p1 = _feature_prevalence_from_nan_matrix(train_matrix)
    if threshold_mode == "fixed":
        thresholds = np.full(train_matrix.shape[1], float(np.clip(fixed_threshold, 0.0, 1.0)), dtype=float)
    elif threshold_mode == "feature_prevalence":
        thresholds = feat_p1.astype(float, copy=True)
    elif threshold_mode == "global_prevalence":
        thresholds = np.full(train_matrix.shape[1], global_p1, dtype=float)
    else:
        raise ValueError(f"Unsupported threshold_mode: {threshold_mode}")
    return np.clip(thresholds, 0.0, 1.0), global_p1


def _metrics_for_subset(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, languages: np.ndarray, subset_langs: set[str]
) -> dict[str, float]:
    keep = np.array([x in subset_langs for x in languages], dtype=bool)
    if not keep.any():
        return {
            "n": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "brier": 0.0,
            "ece": 0.0,
        }
    return compute_binary_metrics(y_true[keep], y_pred[keep], y_prob[keep], include_ece=True, ece_bins=10)


def _build_records_for_mask(
    typology_df: pd.DataFrame,
    eval_mask: np.ndarray,
    probs: np.ndarray,
    baseline_name: str,
    decision_thresholds: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    binary = _to_binary_array(typology_df.to_numpy())
    row_labels = typology_df.index.astype(str).to_numpy()
    col_labels = typology_df.columns.astype(str).to_numpy()

    indices = np.argwhere(eval_mask)
    records: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    langs: list[str] = []

    for idx, (ri, ci) in enumerate(indices):
        p = float(np.clip(probs[ri, ci], 0.0, 1.0))
        pred = 1 if p >= float(decision_thresholds[ci]) else 0
        gold = int(binary[ri, ci])

        lang = str(row_labels[ri])
        feat = str(col_labels[ci])
        example_id = f"baseline:{baseline_name}:{lang}:{feat}:{idx}"

        if pred == 1:
            prob_correct = p
        else:
            prob_correct = 1.0 - p
        if prob_correct >= 0.8:
            conf = "high"
        elif prob_correct >= 0.6:
            conf = "medium"
        else:
            conf = "low"

        records.append(
            {
                "id": example_id,
                "example_id": example_id,
                "language_id": lang,
                "feature_id": feat,
                "value": str(pred),
                "confidence": conf,
                "probability": p,
                "rationale": "softimpute matrix completion prediction",
            }
        )
        y_true.append(gold)
        y_pred.append(pred)
        y_prob.append(p)
        langs.append(lang)

    return (
        records,
        np.asarray(y_true, dtype=np.int8),
        np.asarray(y_pred, dtype=np.int8),
        np.asarray(y_prob, dtype=float),
        np.asarray(langs, dtype=object),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SoftImpute baseline with masked-entry evaluation.")
    parser.add_argument("--typology_csv", type=str, default="data/derived/uriel+_typological.csv")
    parser.add_argument("--splits_json", type=str, default="data/splits/splits_v1.json")
    parser.add_argument("--mask", type=str, default="data/splits/mask_mcar20_seed42.npy")
    parser.add_argument("--groups_json", type=str, default="data/benchmark/language_groups_2.json")
    parser.add_argument("--out_dir", type=str, default="artifacts/baseline")
    parser.add_argument("--pred_out", type=str, default="artifacts/baseline/predictions_baseline_softimpute.jsonl")
    parser.add_argument("--metrics_out", type=str, default="artifacts/baseline/metrics_baseline_softimpute.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rank", type=int, default=64)
    parser.add_argument("--max_iters", type=int, default=100)
    parser.add_argument("--convergence_threshold", type=float, default=1e-5)
    parser.add_argument("--shrinkage_value", type=float, default=None)
    parser.add_argument(
        "--input_transform",
        type=str,
        default="centered",
        choices=["raw", "centered", "standardized"],
        help="Transform observed 0/1 values before SoftImpute, then invert back to probability space.",
    )
    parser.add_argument(
        "--threshold_mode",
        type=str,
        default="fixed",
        choices=["fixed", "feature_prevalence", "global_prevalence"],
        help="Decision rule for mapping SoftImpute probabilities to binary predictions.",
    )
    parser.add_argument(
        "--decision_threshold",
        type=float,
        default=0.5,
        help="Used when --threshold_mode=fixed.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    typology_df = pd.read_csv(args.typology_csv, index_col=0)
    typology_df.index = typology_df.index.astype(str)

    values = typology_df.to_numpy().astype(float)
    observed = values != -1
    shape = values.shape

    splits = json.loads(Path(args.splits_json).read_text(encoding="utf-8"))["splits"]
    mask = _load_mask(args.mask, expected_shape=shape)

    train_rows = _langs_to_row_mask(typology_df.index, splits["train"])
    dev_rows = _langs_to_row_mask(typology_df.index, splits["dev"])
    test_rows = _langs_to_row_mask(typology_df.index, splits["test"])

    visible = observed & (~mask)
    train_observed = visible & train_rows[:, None]

    dev_eval_mask = mask & observed & dev_rows[:, None]
    test_eval_mask = mask & observed & test_rows[:, None]

    # Strict protocol requested: fit on train-observed entries only.
    train_matrix = np.where(train_observed, values, np.nan)
    decision_thresholds, global_train_positive_rate = _resolve_decision_thresholds(
        train_matrix,
        threshold_mode=args.threshold_mode,
        fixed_threshold=args.decision_threshold,
    )

    mem_before = _max_rss_mb()
    t0 = time.perf_counter()
    imputed = _soft_impute_probabilities(
        train_matrix,
        max_rank=args.max_rank,
        max_iters=args.max_iters,
        convergence_threshold=args.convergence_threshold,
        shrinkage_value=args.shrinkage_value,
        input_transform=args.input_transform,
    )
    fit_seconds = time.perf_counter() - t0
    mem_after = _max_rss_mb()

    pred_records, y_true_test, y_pred_test, y_prob_test, langs_test = _build_records_for_mask(
        typology_df=typology_df,
        eval_mask=test_eval_mask,
        probs=imputed,
        baseline_name="softimpute",
        decision_thresholds=decision_thresholds,
    )
    _, y_true_dev, y_pred_dev, y_prob_dev, _ = _build_records_for_mask(
        typology_df=typology_df,
        eval_mask=dev_eval_mask,
        probs=imputed,
        baseline_name="softimpute_dev",
        decision_thresholds=decision_thresholds,
    )

    dev_metrics = compute_binary_metrics(y_true_dev, y_pred_dev, y_prob_dev, include_ece=True, ece_bins=10)
    test_metrics = compute_binary_metrics(y_true_test, y_pred_test, y_prob_test, include_ece=True, ece_bins=10)

    groups_path = Path(args.groups_json)
    hr_lr = None
    if groups_path.exists():
        groups = json.loads(groups_path.read_text(encoding="utf-8"))
        hr_lr = {
            label: _metrics_for_subset(y_true_test, y_pred_test, y_prob_test, langs_test, set(langs))
            for label, langs in groups.items()
        }

    metrics_payload: dict[str, Any] = {
        "baseline": "softimpute",
        "setup": {
            "typology_csv": args.typology_csv,
            "splits_json": args.splits_json,
            "mask": args.mask,
            "fit_protocol": "fit on train-observed entries only",
            "implementation": "SoftImpute-equivalent iterative singular-value shrinkage",
            "input_transform": args.input_transform,
            "threshold_mode": args.threshold_mode,
            "decision_threshold": args.decision_threshold if args.threshold_mode == "fixed" else None,
            "params": {
                "max_rank": args.max_rank,
                "max_iters": args.max_iters,
                "convergence_threshold": args.convergence_threshold,
                "shrinkage_value": args.shrinkage_value,
            },
            "counts": {
                "train_observed_n": int(np.sum(train_observed)),
                "dev_masked_n": int(np.sum(dev_eval_mask)),
                "test_masked_n": int(np.sum(test_eval_mask)),
            },
        },
        "runtime": {
            "fit_seconds": fit_seconds,
            "max_rss_mb_before": mem_before,
            "max_rss_mb_after": mem_after,
            "max_rss_mb_delta": max(0.0, mem_after - mem_before),
        },
        "train_distribution": {
            "global_positive_rate": global_train_positive_rate,
            "feature_threshold_min": float(np.min(decision_thresholds)),
            "feature_threshold_max": float(np.max(decision_thresholds)),
            "feature_threshold_avg": float(np.mean(decision_thresholds)),
        },
        "metrics": {
            "dev": dev_metrics,
            "test": test_metrics,
        },
    }
    if hr_lr is not None:
        metrics_payload["hr_lr_breakdown_test"] = hr_lr

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    pred_out = Path(args.pred_out)
    metrics_out = Path(args.metrics_out)

    _write_predictions(pred_out, pred_records)
    metrics_out.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"Wrote predictions: {pred_out}")
    print(f"Wrote metrics: {metrics_out}")
    print(json.dumps(metrics_payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
