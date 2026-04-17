from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.run_baselines import run_baseline
from baselines.run_softimpute import _resolve_decision_thresholds, _soft_impute_probabilities
from eval_metrics import compute_binary_metrics


def _read_gold_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "id" in rec and "gold_value" in rec:
            rows.append(
                {
                    "id": str(rec["id"]),
                    "resource_group": str(rec.get("resource_group", "")),
                    "language": str(rec["language"]),
                    "feature": str(rec["feature"]),
                    "gold_value": str(rec["gold_value"]),
                    "allowed_values": [str(v) for v in rec.get("allowed_values", ["0", "1"])],
                }
            )
        elif "example_id" in rec and "gold" in rec:
            rows.append(
                {
                    "id": str(rec["example_id"]),
                    "resource_group": str(rec.get("resource_group", "")),
                    "language": str(rec.get("language", rec.get("language_id", ""))),
                    "feature": str(rec.get("feature", rec.get("feature_id", ""))),
                    "gold_value": str(rec["gold"]),
                    "allowed_values": [str(v) for v in rec.get("allowed_values", ["0", "1"])],
                }
            )
        else:
            raise ValueError("Unsupported gold schema in input file.")
    return rows


def _stable_row_seed(seed: int, language: str) -> int:
    digest = hashlib.sha256(f"{seed}:{language}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _budget_map_from_args(
    enabled: bool,
    budget_lrl: int,
    budget_mrl: int,
    budget_hrl: int,
) -> dict[str, int]:
    if not enabled:
        return {}
    out = {
        "lrl": int(budget_lrl),
        "mrl": int(budget_mrl),
        "hrl": int(budget_hrl),
    }
    for group, budget in out.items():
        if budget < 0:
            raise ValueError(f"Budget for {group} must be >= 0; got {budget}.")
    return out


def _load_language_to_group(gold_rows: list[dict[str, Any]], groups_json: str | None) -> dict[str, str]:
    if groups_json:
        payload = json.loads(Path(groups_json).read_text(encoding="utf-8"))
        lang_to_group: dict[str, str] = {}
        for group, langs in payload.items():
            if not isinstance(langs, list):
                continue
            for lang in langs:
                key = str(lang)
                prev = lang_to_group.get(key)
                if prev is not None and prev != str(group):
                    raise ValueError(f"Language appears in multiple groups in {groups_json}: {key}")
                lang_to_group[key] = str(group)
        return lang_to_group

    lang_to_group: dict[str, str] = {}
    for rec in gold_rows:
        lang = str(rec["language"])
        group = str(rec.get("resource_group", ""))
        prev = lang_to_group.get(lang)
        if prev is not None and prev != group:
            raise ValueError(f"Inconsistent resource_group for language {lang}: {prev} vs {group}")
        lang_to_group[lang] = group
    return lang_to_group


def _build_budget_visible_mask(
    observed: np.ndarray,
    index: pd.Index,
    lang_to_group: dict[str, str],
    budgets_by_group: dict[str, int],
    budget_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    visible = observed.copy()
    row_idx = {str(lang): i for i, lang in enumerate(index.astype(str).tolist())}
    n_cols = observed.shape[1]

    group_stats: dict[str, Any] = {}
    for group, budget in budgets_by_group.items():
        affected = 0
        originally_observed = 0
        kept_observed = 0
        hidden_observed = 0
        for lang, lang_group in lang_to_group.items():
            if lang_group != group:
                continue
            ri = row_idx.get(lang)
            if ri is None:
                continue
            obs_cols = np.flatnonzero(observed[ri])
            n_obs = int(obs_cols.size)
            if n_obs == 0:
                continue
            affected += 1
            originally_observed += n_obs
            keep_n = min(int(budget), n_obs)
            if keep_n == n_obs:
                kept_observed += n_obs
                continue
            rng = random.Random(_stable_row_seed(budget_seed, lang))
            keep_cols = rng.sample(obs_cols.tolist(), keep_n) if keep_n > 0 else []
            row_visible = np.zeros(n_cols, dtype=bool)
            if keep_cols:
                row_visible[np.asarray(keep_cols, dtype=int)] = True
            visible[ri] = row_visible
            kept_observed += keep_n
            hidden_observed += (n_obs - keep_n)

        group_stats[group] = {
            "budget": int(budget),
            "languages_affected": int(affected),
            "originally_observed_cells": int(originally_observed),
            "kept_observed_cells": int(kept_observed),
            "hidden_observed_cells": int(hidden_observed),
        }

    meta = {
        "enabled": bool(budgets_by_group),
        "budgets_by_group": {k: int(v) for k, v in budgets_by_group.items()},
        "seed": int(budget_seed),
        "total_observed_cells": int(observed.sum()),
        "total_visible_observed_cells": int((observed & visible).sum()),
        "total_hidden_observed_cells": int((observed & (~visible)).sum()),
        "group_stats": group_stats,
    }
    return visible, meta


def _prob_to_confidence(prob_correct: float) -> str:
    if prob_correct >= 0.8:
        return "high"
    if prob_correct >= 0.6:
        return "medium"
    return "low"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _raw_json_output(value: str, confidence: str, rationale: str) -> str:
    return json.dumps(
        {
            "value": str(value),
            "confidence": str(confidence),
            "rationale": str(rationale),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _subset_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, groups: list[str], group_name: str
) -> dict[str, float]:
    keep = np.array([g == group_name for g in groups], dtype=bool)
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


def _metrics_by_resource_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    groups: list[str],
) -> dict[str, dict[str, float]]:
    labels = sorted({label for label in groups if label})
    return {
        label: _subset_metrics(y_true, y_pred, y_prob, groups, label)
        for label in labels
    }


def _records_from_run_result(
    baseline_name: str,
    gold_rows: list[dict[str, Any]],
    by_key: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    out: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    groups: list[str] = []

    for rec in gold_rows:
        key = (rec["language"], rec["feature"])
        pred = by_key.get(key)
        if pred is None:
            raise KeyError(f"Missing prediction for {key} in {baseline_name}")
        p = float(np.clip(float(pred.get("probability", 0.5)), 0.0, 1.0))
        val = str(pred["value"])
        conf = str(pred.get("confidence", "low"))
        rationale = f"{baseline_name} baseline on benchmark gold ids"
        out.append(
            {
                "id": rec["id"],
                "resource_group": rec["resource_group"],
                "language": rec["language"],
                "feature": rec["feature"],
                "value": val,
                "confidence": conf,
                "rationale": rationale,
                "probability": p,
                "output": _raw_json_output(val, conf, rationale),
            }
        )
        y_true.append(int(float(rec["gold_value"])))
        y_pred.append(int(float(val)))
        y_prob.append(p)
        groups.append(rec["resource_group"])
    return (
        out,
        np.asarray(y_true, dtype=np.int8),
        np.asarray(y_pred, dtype=np.int8),
        np.asarray(y_prob, dtype=float),
        groups,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Score baselines directly on benchmark gold IDs for apples-to-apples fairness.")
    p.add_argument("--typology_csv", type=str, default="data/derived/uriel+_typological.csv")
    p.add_argument("--gold", type=str, default="data/benchmark/gold_eval_2.jsonl")
    p.add_argument(
        "--resource_groups_json",
        type=str,
        default=None,
        help="Optional mapping of all languages to resource groups (group -> language list) for budgeting.",
    )
    p.add_argument("--out_dir", type=str, default="artifacts/prediction/benchmark")
    p.add_argument("--knn_k", type=int, default=11)
    p.add_argument("--knn_metric", type=str, default="cosine", choices=["cosine", "jaccard"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--softimpute_max_rank", type=int, default=64)
    p.add_argument("--softimpute_max_iters", type=int, default=100)
    p.add_argument("--softimpute_convergence_threshold", type=float, default=1e-5)
    p.add_argument("--softimpute_shrinkage_value", type=float, default=None)
    p.add_argument(
        "--softimpute_input_transform",
        type=str,
        default="centered",
        choices=["raw", "centered", "standardized"],
    )
    p.add_argument(
        "--softimpute_threshold_mode",
        type=str,
        default="fixed",
        choices=["fixed", "feature_prevalence", "global_prevalence"],
    )
    p.add_argument("--softimpute_decision_threshold", type=float, default=0.5)
    p.add_argument(
        "--budget_protocol",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply per-group per-language context budgets before baseline fitting.",
    )
    p.add_argument("--budget_lrl", type=int, default=5)
    p.add_argument("--budget_mrl", type=int, default=20)
    p.add_argument("--budget_hrl", type=int, default=80)
    p.add_argument("--budget_seed", type=int, default=17)
    p.add_argument(
        "--budget_eval_requirement",
        type=str,
        default="strict",
        choices=["none", "warn", "strict"],
        help="How to enforce that eval rows are masked by the budget protocol.",
    )
    p.add_argument("--report_out", type=str, default="artifacts/prediction/benchmark/report_baselines_on_gold_eval_2.json")
    args = p.parse_args()

    typology_df = pd.read_csv(args.typology_csv, index_col=0)
    typology_df.index = typology_df.index.astype(str)
    values = typology_df.to_numpy().astype(float)
    observed = (values != -1) & (~pd.isna(values))
    n_rows, n_cols = typology_df.shape

    gold_rows = _read_gold_rows(args.gold)
    if not gold_rows:
        raise RuntimeError("No gold rows loaded.")

    row_idx = {str(lang): i for i, lang in enumerate(typology_df.index.astype(str).tolist())}
    col_idx = {str(feat): j for j, feat in enumerate(typology_df.columns.astype(str).tolist())}
    eval_mask = np.zeros((n_rows, n_cols), dtype=bool)
    for rec in gold_rows:
        lang = rec["language"]
        feat = rec["feature"]
        if lang not in row_idx:
            raise KeyError(f"Language not found in typology: {lang}")
        if feat not in col_idx:
            raise KeyError(f"Feature not found in typology: {feat}")
        ri = row_idx[lang]
        ci = col_idx[feat]
        if not observed[ri, ci]:
            raise ValueError(f"Gold entry is not observed in typology matrix: {lang}/{feat}")
        eval_mask[ri, ci] = True

    budgets_by_group = _budget_map_from_args(
        enabled=args.budget_protocol,
        budget_lrl=args.budget_lrl,
        budget_mrl=args.budget_mrl,
        budget_hrl=args.budget_hrl,
    )
    if budgets_by_group:
        lang_to_group = _load_language_to_group(gold_rows, args.resource_groups_json)
        budget_visible_mask, budget_meta = _build_budget_visible_mask(
            observed=observed,
            index=typology_df.index,
            lang_to_group=lang_to_group,
            budgets_by_group=budgets_by_group,
            budget_seed=args.budget_seed,
        )
        budget_hidden_mask = observed & (~budget_visible_mask)
        budget_violations: list[str] = []
        for rec in gold_rows:
            ri = row_idx[rec["language"]]
            ci = col_idx[rec["feature"]]
            if not budget_hidden_mask[ri, ci]:
                budget_violations.append(rec["id"])
        if budget_violations:
            msg = (
                f"{len(budget_violations)} gold rows are not masked by the budget protocol. "
                f"Example IDs: {budget_violations[:5]}"
            )
            if args.budget_eval_requirement == "strict":
                raise ValueError(msg)
            if args.budget_eval_requirement == "warn":
                print(f"[WARN] {msg}")
        visible_base_mask = budget_visible_mask
    else:
        budget_meta = {
            "enabled": False,
            "total_observed_cells": int(observed.sum()),
            "total_visible_observed_cells": int(observed.sum()),
            "total_hidden_observed_cells": 0,
            "group_stats": {},
        }
        budget_violations = []
        visible_base_mask = observed.copy()

    train_mask = visible_base_mask & (~eval_mask)
    visible_mask = train_mask.copy()

    # Mean / Random / kNN baselines on the same eval IDs.
    mean_res = run_baseline(
        typology_df=typology_df,
        train_mask=train_mask,
        test_mask=eval_mask,
        baseline_name="mean",
        seed=args.seed,
        visible_mask=visible_mask,
    )
    rnd_res = run_baseline(
        typology_df=typology_df,
        train_mask=train_mask,
        test_mask=eval_mask,
        baseline_name="random",
        seed=args.seed,
        visible_mask=visible_mask,
    )
    knn_res = run_baseline(
        typology_df=typology_df,
        train_mask=train_mask,
        test_mask=eval_mask,
        baseline_name="knn",
        seed=args.seed,
        visible_mask=visible_mask,
        k=args.knn_k,
        knn_metric=args.knn_metric,
    )

    def to_map(preds: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
        return {(str(r["language_id"]), str(r["feature_id"])): r for r in preds}

    mean_rows, mean_true, mean_pred, mean_prob, groups = _records_from_run_result("mean", gold_rows, to_map(mean_res.predictions))
    rnd_rows, rnd_true, rnd_pred, rnd_prob, _ = _records_from_run_result("random", gold_rows, to_map(rnd_res.predictions))
    knn_rows, knn_true, knn_pred, knn_prob, _ = _records_from_run_result("knn", gold_rows, to_map(knn_res.predictions))

    # SoftImpute on same eval IDs.
    train_matrix = np.where(train_mask, values, np.nan)
    soft_thresholds, soft_global_train_positive_rate = _resolve_decision_thresholds(
        train_matrix,
        threshold_mode=args.softimpute_threshold_mode,
        fixed_threshold=args.softimpute_decision_threshold,
    )
    imputed = _soft_impute_probabilities(
        train_matrix,
        max_rank=args.softimpute_max_rank,
        max_iters=args.softimpute_max_iters,
        convergence_threshold=args.softimpute_convergence_threshold,
        shrinkage_value=args.softimpute_shrinkage_value,
        input_transform=args.softimpute_input_transform,
    )
    soft_rows: list[dict[str, Any]] = []
    soft_true: list[int] = []
    soft_pred: list[int] = []
    soft_prob: list[float] = []
    for rec in gold_rows:
        ri = row_idx[rec["language"]]
        ci = col_idx[rec["feature"]]
        p1 = float(np.clip(imputed[ri, ci], 0.0, 1.0))
        pred = 1 if p1 >= float(soft_thresholds[ci]) else 0
        prob_correct = p1 if pred == 1 else (1.0 - p1)
        conf = _prob_to_confidence(prob_correct)
        rationale = "softimpute baseline on benchmark gold ids"
        soft_rows.append(
            {
                "id": rec["id"],
                "resource_group": rec["resource_group"],
                "language": rec["language"],
                "feature": rec["feature"],
                "value": str(pred),
                "confidence": conf,
                "rationale": rationale,
                "probability": p1,
                "output": _raw_json_output(str(pred), conf, rationale),
            }
        )
        soft_true.append(int(float(rec["gold_value"])))
        soft_pred.append(pred)
        soft_prob.append(p1)
    soft_true_arr = np.asarray(soft_true, dtype=np.int8)
    soft_pred_arr = np.asarray(soft_pred, dtype=np.int8)
    soft_prob_arr = np.asarray(soft_prob, dtype=float)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_files = {
        "mean": out_dir / "predictions_baseline_mean_gold_eval_2.jsonl",
        "random": out_dir / "predictions_baseline_random_gold_eval_2.jsonl",
        "knn": out_dir / "predictions_baseline_knn_gold_eval_2.jsonl",
        "softimpute": out_dir / "predictions_baseline_softimpute_gold_eval_2.jsonl",
    }
    _write_jsonl(pred_files["mean"], mean_rows)
    _write_jsonl(pred_files["random"], rnd_rows)
    _write_jsonl(pred_files["knn"], knn_rows)
    _write_jsonl(pred_files["softimpute"], soft_rows)

    protocol_text = (
        "Train context uses budgeted visible entries per language/resource group and excludes target gold IDs."
        if budgets_by_group
        else "Train context uses all observed entries except the target gold IDs."
    )

    report: dict[str, Any] = {
        "setup": {
            "typology_csv": args.typology_csv,
            "gold": args.gold,
            "n_gold": len(gold_rows),
            "protocol": protocol_text,
            "resource_groups_json": args.resource_groups_json,
            "knn_k": args.knn_k,
            "knn_metric": args.knn_metric,
            "seed": args.seed,
            "train_visible_cells_after_protocol": int(train_mask.sum()),
            "eval_cells": int(eval_mask.sum()),
            "budget_protocol": budget_meta,
            "budget_eval_requirement": args.budget_eval_requirement if budgets_by_group else "none",
            "budget_eval_violations_n": int(len(budget_violations)),
            "budget_eval_violation_examples": budget_violations[:10],
            "softimpute_input_transform": args.softimpute_input_transform,
            "softimpute_threshold_mode": args.softimpute_threshold_mode,
            "softimpute_decision_threshold": (
                args.softimpute_decision_threshold if args.softimpute_threshold_mode == "fixed" else None
            ),
            "softimpute_global_train_positive_rate": soft_global_train_positive_rate,
        },
        "predictions": {k: str(v) for k, v in pred_files.items()},
        "metrics_binary": {
            "mean": compute_binary_metrics(mean_true, mean_pred, mean_prob, include_ece=True, ece_bins=10),
            "random": compute_binary_metrics(rnd_true, rnd_pred, rnd_prob, include_ece=True, ece_bins=10),
            "knn": compute_binary_metrics(knn_true, knn_pred, knn_prob, include_ece=True, ece_bins=10),
            "softimpute": compute_binary_metrics(soft_true_arr, soft_pred_arr, soft_prob_arr, include_ece=True, ece_bins=10),
        },
        "resource_group_breakdown_binary": {
            "mean": _metrics_by_resource_group(mean_true, mean_pred, mean_prob, groups),
            "random": _metrics_by_resource_group(rnd_true, rnd_pred, rnd_prob, groups),
            "knn": _metrics_by_resource_group(knn_true, knn_pred, knn_prob, groups),
            "softimpute": _metrics_by_resource_group(soft_true_arr, soft_pred_arr, soft_prob_arr, groups),
        },
    }

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote report: {report_out}")
    print(json.dumps(report["metrics_binary"], indent=2))


if __name__ == "__main__":
    main()
