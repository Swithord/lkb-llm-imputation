from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BENCHMARK_DIR = ROOT / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from eval_metrics import compute_binary_metrics


@dataclass
class BaselineRunResult:
    predictions: list[dict[str, Any]]
    metrics: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    languages: list[str]


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


def _to_binary_array(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, -1, dtype=np.int8)
    observed = values != -1
    casted = values[observed].astype(int)
    if casted.size and (~np.isin(casted, [0, 1])).any():
        raise ValueError("Current baselines support binary typology values {0,1,-1} only.")
    out[observed] = casted
    return out


def _prob_to_confidence(prob_correct: float) -> str:
    if prob_correct >= 0.8:
        return "high"
    if prob_correct >= 0.6:
        return "medium"
    return "low"


def _feature_prevalence(binary_matrix: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, float]:
    train_vals = binary_matrix[train_mask]
    global_p1 = float(np.mean(train_vals)) if train_vals.size else 0.5

    feat_p1 = np.full(binary_matrix.shape[1], global_p1, dtype=float)
    for col in range(binary_matrix.shape[1]):
        col_vals = binary_matrix[:, col]
        col_train = col_vals[train_mask[:, col]]
        if col_train.size:
            feat_p1[col] = float(np.mean(col_train))
    return feat_p1, global_p1


def _knn_similarity(query: np.ndarray, train_visible: np.ndarray, metric: str) -> np.ndarray:
    query_obs = query != -1
    train_obs = train_visible != -1
    common = train_obs & query_obs[None, :]

    query_pos = query == 1
    train_pos = train_visible == 1

    inter = np.sum(train_pos & query_pos[None, :] & common, axis=1).astype(float)
    q_cnt = np.sum(query_pos[None, :] & common, axis=1).astype(float)
    t_cnt = np.sum(train_pos & common, axis=1).astype(float)

    if metric == "cosine":
        denom = np.sqrt(q_cnt * t_cnt)
        sim = np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0)
        return sim
    if metric == "jaccard":
        union = q_cnt + t_cnt - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        return sim
    raise ValueError(f"Unsupported distance metric: {metric}")


def _predict_knn_entry(
    train_order: np.ndarray,
    train_feature_values: np.ndarray,
    k: int,
    fallback_p1: float,
) -> tuple[int, float]:
    ordered = train_feature_values[train_order]
    valid = ordered[ordered != -1]
    if valid.size == 0:
        p1 = fallback_p1
    else:
        use = valid[: min(k, valid.size)]
        p1 = float(np.mean(use))
    pred = 1 if p1 >= 0.5 else 0
    return pred, p1


def run_baseline(
    typology_df: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    baseline_name: str,
    seed: int = 42,
    visible_mask: np.ndarray | None = None,
    k: int = 5,
    knn_metric: str = "cosine",
) -> BaselineRunResult:
    """
    Unified baseline API:
      input: typology matrix + train mask + test mask
      output: benchmark-aligned predictions and masked-entry metrics.
    """
    values = typology_df.to_numpy()
    observed = values != -1
    binary = _to_binary_array(values)

    train_eff = np.asarray(train_mask, dtype=bool) & observed
    test_eff = np.asarray(test_mask, dtype=bool) & observed
    if train_eff.shape != observed.shape or test_eff.shape != observed.shape:
        raise ValueError("train_mask/test_mask shape must match typology matrix shape.")

    if visible_mask is None:
        vis_eff = observed & (~test_eff)
    else:
        vis_eff = np.asarray(visible_mask, dtype=bool) & observed

    feat_p1, global_p1 = _feature_prevalence(binary, train_eff)
    rng = np.random.default_rng(seed)

    row_labels = typology_df.index.astype(str).to_numpy()
    col_labels = typology_df.columns.astype(str).to_numpy()
    test_indices = np.argwhere(test_eff)

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    pred_langs: list[str] = []
    predictions: list[dict[str, Any]] = []

    train_rows = np.where(np.any(train_eff, axis=1))[0]
    train_visible = np.where(train_eff[train_rows], binary[train_rows], -1)

    # Cache nearest-neighbor ordering per query language for efficiency.
    knn_rank_cache: dict[int, np.ndarray] = {}

    for idx, (row_idx, col_idx) in enumerate(test_indices):
        lang = str(row_labels[row_idx])
        feat = str(col_labels[col_idx])
        gold = int(binary[row_idx, col_idx])

        if baseline_name == "mean":
            p1 = float(feat_p1[col_idx])
            pred = 1 if p1 >= 0.5 else 0
        elif baseline_name == "random":
            p1 = float(feat_p1[col_idx])
            pred = int(rng.binomial(n=1, p=p1))
        elif baseline_name == "knn":
            if row_idx not in knn_rank_cache:
                query_visible = np.where(vis_eff[row_idx], binary[row_idx], -1)
                sim = _knn_similarity(query_visible, train_visible, metric=knn_metric)
                knn_rank_cache[row_idx] = np.argsort(-sim)
            rank = knn_rank_cache[row_idx]
            pred, p1 = _predict_knn_entry(
                train_order=rank,
                train_feature_values=train_visible[:, col_idx],
                k=k,
                fallback_p1=float(feat_p1[col_idx]),
            )
        else:
            raise ValueError(f"Unknown baseline: {baseline_name}")

        prob_correct = p1 if pred == 1 else (1.0 - p1)
        conf = _prob_to_confidence(prob_correct)
        example_id = f"baseline:{baseline_name}:{lang}:{feat}:{idx}"

        predictions.append(
            {
                "id": example_id,
                "example_id": example_id,
                "language_id": lang,
                "feature_id": feat,
                "value": str(pred),
                "confidence": conf,
                "probability": float(p1),
                "rationale": f"{baseline_name} baseline prediction",
            }
        )

        y_true.append(gold)
        y_pred.append(pred)
        y_prob.append(p1)
        pred_langs.append(lang)

    metrics = compute_binary_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, include_ece=True, ece_bins=10)
    metrics["train_observed_n"] = float(np.sum(train_eff))
    metrics["test_masked_n"] = float(np.sum(test_eff))
    metrics["global_train_positive_rate"] = float(global_p1)

    return BaselineRunResult(
        predictions=predictions,
        metrics=metrics,
        y_true=np.asarray(y_true, dtype=np.int8),
        y_pred=np.asarray(y_pred, dtype=np.int8),
        y_prob=np.asarray(y_prob, dtype=float),
        languages=pred_langs,
    )


def _langs_to_row_mask(index: pd.Index, langs: list[str]) -> np.ndarray:
    lang_set = set(langs)
    return np.array([str(x) in lang_set for x in index], dtype=bool)


def _metrics_for_language_subset(result: BaselineRunResult, language_ids: set[str]) -> dict[str, float]:
    keep = np.array([lang in language_ids for lang in result.languages], dtype=bool)
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
    return compute_binary_metrics(
        y_true=result.y_true[keep],
        y_pred=result.y_pred[keep],
        y_prob=result.y_prob[keep],
        include_ece=True,
        ece_bins=10,
    )


def _write_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _parse_k_values(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    vals = [v for v in vals if v > 0]
    if not vals:
        raise ValueError("k_values must include at least one positive integer.")
    return sorted(set(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random/mean/kNN baselines with split-aware evaluation.")
    parser.add_argument("--typology_csv", type=str, default="data/derived/uriel+_typological.csv")
    parser.add_argument("--splits_json", type=str, default="data/splits/splits_v1.json")
    parser.add_argument("--mask", type=str, default="data/splits/mask_mcar20_seed42.npy")
    parser.add_argument("--groups_json", type=str, default="data/benchmark/language_groups_2.json")
    parser.add_argument("--k_values", type=str, default="1,3,5,7,11,21")
    parser.add_argument("--knn_metric", type=str, default="cosine", choices=["cosine", "jaccard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="artifacts/baseline")
    parser.add_argument("--report_out", type=str, default="artifacts/reports/report_baselines_v1.json")
    args = parser.parse_args()

    typology_df = pd.read_csv(args.typology_csv, index_col=0)
    typology_df.index = typology_df.index.astype(str)

    values = typology_df.to_numpy()
    observed = values != -1
    shape = typology_df.shape

    splits = json.loads(Path(args.splits_json).read_text(encoding="utf-8"))["splits"]
    mask = _load_mask(args.mask, expected_shape=shape)

    train_rows = _langs_to_row_mask(typology_df.index, splits["train"])
    dev_rows = _langs_to_row_mask(typology_df.index, splits["dev"])
    test_rows = _langs_to_row_mask(typology_df.index, splits["test"])

    visible_mask = observed & (~mask)
    train_mask = visible_mask & train_rows[:, None]
    dev_eval_mask = mask & observed & dev_rows[:, None]
    test_eval_mask = mask & observed & test_rows[:, None]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline: mean
    mean_dev = run_baseline(
        typology_df=typology_df,
        train_mask=train_mask,
        test_mask=dev_eval_mask,
        baseline_name="mean",
        seed=args.seed,
        visible_mask=visible_mask,
    )
    mean_test = run_baseline(
        typology_df=typology_df,
        train_mask=train_mask,
        test_mask=test_eval_mask,
        baseline_name="mean",
        seed=args.seed,
        visible_mask=visible_mask,
    )

    # Baseline: random (Bernoulli with per-feature train prevalence)
    random_dev = run_baseline(
        typology_df=typology_df,
        train_mask=train_mask,
        test_mask=dev_eval_mask,
        baseline_name="random",
        seed=args.seed,
        visible_mask=visible_mask,
    )
    random_test = run_baseline(
        typology_df=typology_df,
        train_mask=train_mask,
        test_mask=test_eval_mask,
        baseline_name="random",
        seed=args.seed,
        visible_mask=visible_mask,
    )

    # Baseline: kNN with k tuned on dev.
    k_values = _parse_k_values(args.k_values)
    knn_dev_by_k: dict[int, BaselineRunResult] = {}
    for k in k_values:
        knn_dev_by_k[k] = run_baseline(
            typology_df=typology_df,
            train_mask=train_mask,
            test_mask=dev_eval_mask,
            baseline_name="knn",
            seed=args.seed,
            visible_mask=visible_mask,
            k=k,
            knn_metric=args.knn_metric,
        )

    best_k = max(k_values, key=lambda kk: knn_dev_by_k[kk].metrics["f1"])
    knn_test = run_baseline(
        typology_df=typology_df,
        train_mask=train_mask,
        test_mask=test_eval_mask,
        baseline_name="knn",
        seed=args.seed,
        visible_mask=visible_mask,
        k=best_k,
        knn_metric=args.knn_metric,
    )

    # Write prediction files aligned to benchmark-style JSONL.
    _write_predictions(out_dir / "predictions_baseline_mean.jsonl", mean_test.predictions)
    _write_predictions(out_dir / "predictions_baseline_random.jsonl", random_test.predictions)
    _write_predictions(out_dir / "predictions_baseline_knn.jsonl", knn_test.predictions)

    report: dict[str, Any] = {
        "setup": {
            "typology_csv": args.typology_csv,
            "splits_json": args.splits_json,
            "mask": args.mask,
            "seed": args.seed,
            "knn_metric": args.knn_metric,
            "candidate_k": k_values,
            "selected_k": best_k,
            "eval_protocol": "Evaluate on masked entries only.",
            "counts": {
                "train_visible_n": int(np.sum(train_mask)),
                "dev_masked_n": int(np.sum(dev_eval_mask)),
                "test_masked_n": int(np.sum(test_eval_mask)),
            },
        },
        "baselines": {
            "mean": {
                "dev": mean_dev.metrics,
                "test": mean_test.metrics,
            },
            "random": {
                "dev": random_dev.metrics,
                "test": random_test.metrics,
            },
            "knn": {
                "dev_by_k": {str(k): knn_dev_by_k[k].metrics for k in k_values},
                "test": knn_test.metrics,
            },
        },
        "outputs": {
            "mean_predictions": str(out_dir / "predictions_baseline_mean.jsonl"),
            "random_predictions": str(out_dir / "predictions_baseline_random.jsonl"),
            "knn_predictions": str(out_dir / "predictions_baseline_knn.jsonl"),
        },
    }

    groups_path = Path(args.groups_json)
    if groups_path.exists():
        groups = json.loads(groups_path.read_text(encoding="utf-8"))
        high = set(groups.get("high", []))
        low = set(groups.get("low", []))
        report["hr_lr_breakdown"] = {
            "mean_test": {
                "high": _metrics_for_language_subset(mean_test, high),
                "low": _metrics_for_language_subset(mean_test, low),
            },
            "random_test": {
                "high": _metrics_for_language_subset(random_test, high),
                "low": _metrics_for_language_subset(random_test, low),
            },
            "knn_test": {
                "high": _metrics_for_language_subset(knn_test, high),
                "low": _metrics_for_language_subset(knn_test, low),
            },
        }

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote report: {report_path}")
    print(json.dumps(report["baselines"], indent=2))


if __name__ == "__main__":
    main()
