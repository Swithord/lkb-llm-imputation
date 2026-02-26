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
from sklearn.utils.extmath import randomized_svd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_metrics import compute_binary_metrics


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
        _, s0, _ = randomized_svd(X_work, n_components=1, n_iter=5, random_state=0)
        shrinkage = float(s0[0]) / 50.0
    else:
        shrinkage = float(shrinkage_value)

    for _ in range(max_iters):
        old_missing = X_work[missing_mask].copy()
        if max_rank is not None:
            rank = min(int(max_rank), min(X_work.shape))
            U, s, Vt = randomized_svd(X_work, n_components=rank, n_iter=1, random_state=0)
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
        pred = 1 if p >= 0.5 else 0
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
    parser.add_argument("--typology_csv", type=str, default="output/uriel+_typological.csv")
    parser.add_argument("--splits_json", type=str, default="splits_v1.json")
    parser.add_argument("--mask", type=str, default="mask_mcar20_seed42.npy")
    parser.add_argument("--groups_json", type=str, default="benchmark/language_groups_2.json")
    parser.add_argument("--out_dir", type=str, default="baseline")
    parser.add_argument("--pred_out", type=str, default="baseline/predictions_baseline_softimpute.jsonl")
    parser.add_argument("--metrics_out", type=str, default="baseline/metrics_baseline_softimpute.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rank", type=int, default=64)
    parser.add_argument("--max_iters", type=int, default=100)
    parser.add_argument("--convergence_threshold", type=float, default=1e-5)
    parser.add_argument("--shrinkage_value", type=float, default=None)
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

    mem_before = _max_rss_mb()
    t0 = time.perf_counter()
    imputed = _soft_impute_fit_transform(
        train_matrix,
        max_rank=args.max_rank,
        max_iters=args.max_iters,
        convergence_threshold=args.convergence_threshold,
        shrinkage_value=args.shrinkage_value,
    )
    fit_seconds = time.perf_counter() - t0
    mem_after = _max_rss_mb()

    pred_records, y_true_test, y_pred_test, y_prob_test, langs_test = _build_records_for_mask(
        typology_df=typology_df,
        eval_mask=test_eval_mask,
        probs=imputed,
        baseline_name="softimpute",
    )
    _, y_true_dev, y_pred_dev, y_prob_dev, _ = _build_records_for_mask(
        typology_df=typology_df,
        eval_mask=dev_eval_mask,
        probs=imputed,
        baseline_name="softimpute_dev",
    )

    dev_metrics = compute_binary_metrics(y_true_dev, y_pred_dev, y_prob_dev, include_ece=True, ece_bins=10)
    test_metrics = compute_binary_metrics(y_true_test, y_pred_test, y_prob_test, include_ece=True, ece_bins=10)

    groups_path = Path(args.groups_json)
    hr_lr = None
    if groups_path.exists():
        groups = json.loads(groups_path.read_text(encoding="utf-8"))
        high = set(groups.get("high", []))
        low = set(groups.get("low", []))
        hr_lr = {
            "high": _metrics_for_subset(y_true_test, y_pred_test, y_prob_test, langs_test, high),
            "low": _metrics_for_subset(y_true_test, y_pred_test, y_prob_test, langs_test, low),
        }

    metrics_payload: dict[str, Any] = {
        "baseline": "softimpute",
        "setup": {
            "typology_csv": args.typology_csv,
            "splits_json": args.splits_json,
            "mask": args.mask,
            "fit_protocol": "fit on train-observed entries only",
            "implementation": "SoftImpute-equivalent iterative singular-value shrinkage",
            "threshold": 0.5,
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
