from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_metrics import compute_binary_metrics
from schema_io import load_gold_jsonl, load_predictions_jsonl


DEFAULT_CONFIDENCE_MAP = {"low": 0.33, "medium": 0.66, "high": 0.90}


def _extract_last_json_object(text: str) -> dict[str, Any] | None:
    s = str(text).strip()
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    candidates: list[dict[str, Any]] = []
    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    seg_start = -1
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                seg_start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and seg_start >= 0:
                    seg = s[seg_start : i + 1]
                    try:
                        obj = json.loads(seg)
                        if isinstance(obj, dict):
                            candidates.append(obj)
                    except Exception:
                        pass
    if not candidates:
        return None
    return candidates[-1]


def _parse_prediction_fields(rec: dict[str, Any]) -> tuple[str | None, str | None]:
    value = rec.get("value")
    confidence = rec.get("confidence")
    if value is not None:
        return str(value), None if confidence is None else str(confidence).lower()

    raw = rec.get("output")
    if raw is None:
        return None, None
    parsed = _extract_last_json_object(str(raw))
    if not isinstance(parsed, dict):
        return None, None
    value = parsed.get("value")
    confidence = parsed.get("confidence")
    return (
        None if value is None else str(value),
        None if confidence is None else str(confidence).lower(),
    )


def _safe_probability(text: Any) -> float | None:
    if text is None:
        return None
    try:
        val = float(text)
    except Exception:
        return None
    if math.isnan(val):
        return None
    return min(1.0, max(0.0, val))


def _value_is_missing(raw: str) -> bool:
    text = raw.strip()
    if text == "":
        return True
    if text.lower() == "nan":
        return True
    try:
        return float(text) == -1.0
    except Exception:
        return False


def load_language_metadata(metadata_csv: str) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    with Path(metadata_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            glottocode = str(row.get("glottocode") or "").strip()
            if not glottocode:
                continue
            family_name = str(row.get("family_name") or "").strip() or "Unknown"
            raw_macroareas = str(row.get("macroareas") or "")
            macroareas = [part.strip() for part in raw_macroareas.split(";") if part.strip()]
            metadata[glottocode] = {
                "family_name": family_name,
                "macroareas": macroareas,
                "primary_macroarea": macroareas[0] if macroareas else "Unknown",
            }
    return metadata


def load_observed_feature_counts(typology_csv: str) -> tuple[dict[str, int], dict[str, int]]:
    counts: dict[str, int] = {}
    order: dict[str, int] = {}
    with Path(typology_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Typology CSV is empty: {typology_csv}")
        for idx, row in enumerate(reader):
            if not row:
                continue
            lang = str(row[0])
            observed = 0
            for cell in row[1:]:
                if not _value_is_missing(cell):
                    observed += 1
            counts[lang] = observed
            order[lang] = idx
    return counts, order


def _sort_languages(languages: list[str], counts: dict[str, int], order: dict[str, int], reverse: bool = False) -> list[str]:
    if reverse:
        return sorted(languages, key=lambda lang: (-counts[lang], order[lang], lang))
    return sorted(languages, key=lambda lang: (counts[lang], order[lang], lang))


def _round_bucket_size(n_languages: int, fraction: float) -> int:
    size = int(round(n_languages * fraction))
    return max(1, min(n_languages, size))


def _bucket_observed_stats(bucket_langs: list[str], counts: dict[str, int]) -> dict[str, Any]:
    values = [counts[lang] for lang in bucket_langs]
    if not values:
        return {
            "n_languages": 0,
            "observed_count_min": None,
            "observed_count_max": None,
            "observed_count_avg": None,
            "languages": [],
        }
    return {
        "n_languages": len(bucket_langs),
        "observed_count_min": min(values),
        "observed_count_max": max(values),
        "observed_count_avg": sum(values) / len(values),
        "languages": bucket_langs,
    }


def build_extreme_splits(
    benchmark_languages: list[str],
    counts: dict[str, int],
    order: dict[str, int],
    fractions: list[float],
) -> dict[str, dict[str, Any]]:
    ascending = _sort_languages(benchmark_languages, counts, order, reverse=False)
    descending = _sort_languages(benchmark_languages, counts, order, reverse=True)
    splits: dict[str, dict[str, Any]] = {}
    n_languages = len(benchmark_languages)
    for fraction in fractions:
        k = _round_bucket_size(n_languages, fraction)
        low_langs = ascending[:k]
        high_langs = descending[:k]
        split_name = f"extremes_bottom_top_{int(round(fraction * 100)):02d}pct"
        splits[split_name] = {
            "kind": "extremes",
            "fraction": fraction,
            "k_languages_per_side": k,
            "groups": {
                "low": _bucket_observed_stats(low_langs, counts),
                "high": _bucket_observed_stats(high_langs, counts),
            },
        }
    return splits


def build_tertile_split(
    benchmark_languages: list[str],
    counts: dict[str, int],
    order: dict[str, int],
) -> dict[str, Any]:
    ascending = _sort_languages(benchmark_languages, counts, order, reverse=False)
    n_languages = len(ascending)
    low_end = n_languages // 3
    mid_end = (2 * n_languages) // 3
    low_langs = ascending[:low_end]
    mid_langs = ascending[low_end:mid_end]
    high_langs = ascending[mid_end:]
    return {
        "kind": "tertiles_within_benchmark",
        "groups": {
            "low": _bucket_observed_stats(low_langs, counts),
            "mid": _bucket_observed_stats(mid_langs, counts),
            "high": _bucket_observed_stats(high_langs, counts),
        },
    }


def build_metadata_group_defs(
    languages: list[str],
    counts: dict[str, int],
    order: dict[str, int],
    metadata: dict[str, dict[str, Any]],
    metadata_field: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[str]] = {}
    for lang in languages:
        value = metadata.get(lang, {}).get(metadata_field)
        label = str(value).strip() if value is not None else ""
        label = label or "Unknown"
        grouped.setdefault(label, []).append(lang)

    return {
        label: _bucket_observed_stats(_sort_languages(group_langs, counts, order, reverse=False), counts)
        for label, group_langs in sorted(
            grouped.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
    }


@dataclass
class PredictionRow:
    item_id: str
    language: str
    gold_value: int
    parsed_value: int | None
    confidence: str | None
    probability: float | None


def normalize_predictions(
    gold: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    for item_id, gold_rec in gold.items():
        pred_rec = predictions.get(item_id, {})
        raw_value, confidence = _parse_prediction_fields(pred_rec)
        parsed_value: int | None = None
        if raw_value in {"0", "1"}:
            parsed_value = int(raw_value)
        probability = _safe_probability(pred_rec.get("probability"))
        rows.append(
            PredictionRow(
                item_id=item_id,
                language=str(gold_rec["language"]),
                gold_value=int(str(gold_rec["gold_value"])),
                parsed_value=parsed_value,
                confidence=confidence,
                probability=probability,
            )
        )
    return rows


def compute_group_metrics(
    rows: list[PredictionRow],
    confidence_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    total_n = len(rows)
    if total_n == 0:
        return {
            "n_items": 0,
            "n_languages": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "parsed_rate": 0.0,
            "brier": None,
            "ece": None,
        }

    correct = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []

    for row in rows:
        if row.parsed_value is None:
            continue
        if row.parsed_value == row.gold_value:
            correct += 1
        y_true.append(row.gold_value)
        y_pred.append(row.parsed_value)
        if row.probability is not None:
            y_prob.append(row.probability)
        elif confidence_map and row.confidence in confidence_map:
            conf = confidence_map[row.confidence]
            y_prob.append(conf if row.parsed_value == 1 else 1.0 - conf)

    parsed_n = len(y_pred)
    accuracy = correct / total_n
    if parsed_n:
        metrics = compute_binary_metrics(
            y_true,
            y_pred,
            y_prob=y_prob if len(y_prob) == parsed_n else None,
            include_ece=True,
            ece_bins=10,
        )
    else:
        metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "brier": float("nan"),
            "ece": float("nan"),
        }

    out: dict[str, Any] = {
        "n_items": total_n,
        "n_languages": len({row.language for row in rows}),
        "accuracy": accuracy,
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "parsed_rate": parsed_n / total_n,
        "brier": None if math.isnan(float(metrics["brier"])) else float(metrics["brier"]),
        "ece": None if math.isnan(float(metrics["ece"])) else float(metrics["ece"]),
    }
    return out


def _collect_rows_for_languages(rows: list[PredictionRow], languages: list[str]) -> list[PredictionRow]:
    keep = set(languages)
    return [row for row in rows if row.language in keep]


def _count_gold_items_by_language(gold: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in gold.values():
        lang = str(rec["language"])
        counts[lang] = counts.get(lang, 0) + 1
    return counts


def _attach_group_item_counts(
    group_def: dict[str, Any],
    gold_items_by_language: dict[str, int],
) -> dict[str, Any]:
    out = dict(group_def)
    langs = list(group_def.get("languages", []))
    out["n_items"] = sum(gold_items_by_language.get(lang, 0) for lang in langs)
    return out


def _accuracy_leaderboard(model_metrics: dict[str, dict[str, Any]], group_names: list[str]) -> dict[str, list[dict[str, Any]]]:
    leaderboard: dict[str, list[dict[str, Any]]] = {}
    for group in group_names:
        ranked = sorted(
            (
                {"model": model_name, "accuracy": metrics[group]["accuracy"], "f1": metrics[group]["f1"]}
                for model_name, metrics in model_metrics.items()
            ),
            key=lambda rec: (-rec["accuracy"], -rec["f1"], rec["model"]),
        )
        leaderboard[group] = ranked
    return leaderboard


def _rank_models_by_accuracy(metrics_by_model: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (
            {"model": model_name, "accuracy": metrics["accuracy"], "f1": metrics["f1"]}
            for model_name, metrics in metrics_by_model.items()
        ),
        key=lambda rec: (-rec["accuracy"], -rec["f1"], rec["model"]),
    )


def compute_focus_model_gain(
    metrics_by_model: dict[str, dict[str, Any]],
    focus_model: str,
) -> dict[str, Any] | None:
    if focus_model not in metrics_by_model:
        return None

    baseline_rows = [
        (model_name, metrics)
        for model_name, metrics in metrics_by_model.items()
        if model_name != focus_model
    ]
    if not baseline_rows:
        return None

    focus_metrics = metrics_by_model[focus_model]
    best_acc_model, best_acc_metrics = max(
        baseline_rows,
        key=lambda item: (item[1]["accuracy"], item[1]["f1"], item[0]),
    )
    best_f1_model, best_f1_metrics = max(
        baseline_rows,
        key=lambda item: (item[1]["f1"], item[1]["accuracy"], item[0]),
    )
    return {
        "focus_model": focus_model,
        "best_baseline_by_accuracy": best_acc_model,
        "best_baseline_accuracy": best_acc_metrics["accuracy"],
        "focus_minus_best_baseline_accuracy": focus_metrics["accuracy"] - best_acc_metrics["accuracy"],
        "best_baseline_by_f1": best_f1_model,
        "best_baseline_f1": best_f1_metrics["f1"],
        "focus_minus_best_baseline_f1": focus_metrics["f1"] - best_f1_metrics["f1"],
    }


def summarize_focus_model_gains(
    group_reports: dict[str, dict[str, Any]],
    focus_model: str,
    min_languages_for_summary: int = 2,
) -> dict[str, Any]:
    eligible = [
        {"group": group_name, **group_report}
        for group_name, group_report in group_reports.items()
        if group_report["n_languages"] >= min_languages_for_summary
        and group_report.get("focus_model_gain_vs_best_baseline") is not None
    ]
    if not eligible:
        return {
            "focus_model": focus_model,
            "min_languages_for_summary": min_languages_for_summary,
            "n_groups_considered": 0,
            "groups_with_positive_accuracy_gain": 0,
            "groups_with_positive_f1_gain": 0,
            "weighted_item_share_positive_accuracy_gain": None,
            "weighted_item_share_positive_f1_gain": None,
            "focus_model_best_by_accuracy_in_groups": 0,
            "focus_model_best_by_f1_in_groups": 0,
            "top_accuracy_gains": [],
            "lowest_accuracy_gains": [],
        }

    total_items = sum(int(group["n_items"]) for group in eligible)
    positive_acc = [
        group for group in eligible if group["focus_model_gain_vs_best_baseline"]["focus_minus_best_baseline_accuracy"] > 0
    ]
    positive_f1 = [group for group in eligible if group["focus_model_gain_vs_best_baseline"]["focus_minus_best_baseline_f1"] > 0]
    focus_best_acc = [
        group for group in eligible if group["leaderboard_by_accuracy"] and group["leaderboard_by_accuracy"][0]["model"] == focus_model
    ]
    focus_best_f1 = [
        group
        for group in eligible
        if group["leaderboard_by_f1"] and group["leaderboard_by_f1"][0]["model"] == focus_model
    ]

    def _compact_rows(
        groups: list[dict[str, Any]],
        gain_key: str,
        limit: int,
        descending: bool = True,
    ) -> list[dict[str, Any]]:
        ranked = sorted(
            groups,
            key=lambda group: (
                group["focus_model_gain_vs_best_baseline"][gain_key],
                group["n_items"],
                group["group"],
            ),
            reverse=descending,
        )
        return [
            {
                "group": group["group"],
                "n_languages": group["n_languages"],
                "n_items": group["n_items"],
                "gain_accuracy": group["focus_model_gain_vs_best_baseline"]["focus_minus_best_baseline_accuracy"],
                "gain_f1": group["focus_model_gain_vs_best_baseline"]["focus_minus_best_baseline_f1"],
                "best_baseline_by_accuracy": group["focus_model_gain_vs_best_baseline"]["best_baseline_by_accuracy"],
                "best_baseline_by_f1": group["focus_model_gain_vs_best_baseline"]["best_baseline_by_f1"],
            }
            for group in ranked[:limit]
        ]

    return {
        "focus_model": focus_model,
        "min_languages_for_summary": min_languages_for_summary,
        "n_groups_considered": len(eligible),
        "groups_with_positive_accuracy_gain": len(positive_acc),
        "groups_with_positive_f1_gain": len(positive_f1),
        "weighted_item_share_positive_accuracy_gain": (
            sum(int(group["n_items"]) for group in positive_acc) / total_items if total_items else None
        ),
        "weighted_item_share_positive_f1_gain": (
            sum(int(group["n_items"]) for group in positive_f1) / total_items if total_items else None
        ),
        "focus_model_best_by_accuracy_in_groups": len(focus_best_acc),
        "focus_model_best_by_f1_in_groups": len(focus_best_f1),
        "top_accuracy_gains": _compact_rows(eligible, "focus_minus_best_baseline_accuracy", limit=5),
        "lowest_accuracy_gains": _compact_rows(
            eligible,
            "focus_minus_best_baseline_accuracy",
            limit=5,
            descending=False,
        ),
    }


def analyze_low_resource_robustness(
    low_languages: list[str],
    counts: dict[str, int],
    order: dict[str, int],
    gold_items_by_language: dict[str, int],
    normalized_by_model: dict[str, list[PredictionRow]],
    confidence_map: dict[str, float] | None,
    metadata: dict[str, dict[str, Any]],
    focus_model: str,
) -> dict[str, Any]:
    group_specs = {
        "family": {
            "metadata_field": "family_name",
            "description": "Glottolog family_name on low-resource benchmark languages.",
        },
        "macroarea": {
            "metadata_field": "primary_macroarea",
            "description": "Primary macro-area derived from the first Glottolog macroareas entry to avoid double-counting languages.",
        },
    }

    report: dict[str, Any] = {
        "focus_model": focus_model,
        "scope": "Low-resource benchmark languages only.",
        "summary_note": "Gain is measured as focus-model minus strongest non-focus baseline within each group.",
        "groupings": {},
    }

    for group_name, spec in group_specs.items():
        group_defs = build_metadata_group_defs(
            languages=low_languages,
            counts=counts,
            order=order,
            metadata=metadata,
            metadata_field=spec["metadata_field"],
        )
        group_reports: dict[str, dict[str, Any]] = {}
        for label, group_def in group_defs.items():
            group_info = _attach_group_item_counts(group_def, gold_items_by_language)
            metrics_by_model: dict[str, dict[str, Any]] = {}
            for model_name, rows in normalized_by_model.items():
                bucket_rows = _collect_rows_for_languages(rows, group_info["languages"])
                metrics_by_model[model_name] = compute_group_metrics(bucket_rows, confidence_map=confidence_map)

            group_reports[label] = {
                **group_info,
                "metrics": metrics_by_model,
                "leaderboard_by_accuracy": _rank_models_by_accuracy(metrics_by_model),
                "leaderboard_by_f1": sorted(
                    (
                        {"model": model_name, "accuracy": metrics["accuracy"], "f1": metrics["f1"]}
                        for model_name, metrics in metrics_by_model.items()
                    ),
                    key=lambda rec: (-rec["f1"], -rec["accuracy"], rec["model"]),
                ),
                "focus_model_gain_vs_best_baseline": compute_focus_model_gain(metrics_by_model, focus_model=focus_model),
            }

        report["groupings"][group_name] = {
            "description": spec["description"],
            "n_groups_total": len(group_reports),
            "n_groups_with_at_least_2_languages": sum(
                1 for group_report in group_reports.values() if group_report["n_languages"] >= 2
            ),
            "groups": group_reports,
            "summary": summarize_focus_model_gains(group_reports, focus_model=focus_model, min_languages_for_summary=2),
        }

    return report


def analyze_resource_splits(
    typology_csv: str,
    gold_path: str,
    model_paths: dict[str, str],
    metadata_csv: str = "data/derived/metadata.csv",
    confidence_map: dict[str, float] | None = None,
    focus_model: str | None = None,
) -> dict[str, Any]:
    gold = load_gold_jsonl(gold_path)
    benchmark_languages = sorted({str(rec["language"]) for rec in gold.values()})
    low_languages = sorted({str(rec["language"]) for rec in gold.values() if str(rec.get("resource_group")) == "low"})
    counts, order = load_observed_feature_counts(typology_csv)
    metadata = load_language_metadata(metadata_csv)
    if focus_model is None:
        focus_model = next(iter(model_paths))
    if focus_model not in model_paths:
        raise KeyError(f"focus_model={focus_model} is not present in model_paths")

    missing_languages = [lang for lang in benchmark_languages if lang not in counts]
    if missing_languages:
        raise KeyError(f"Missing benchmark languages in typology CSV: {missing_languages[:5]}")

    benchmark_count_values = [counts[lang] for lang in benchmark_languages]
    gold_items_by_language = _count_gold_items_by_language(gold)
    benchmark_stats = {
        "n_languages": len(benchmark_languages),
        "observed_count_min": min(benchmark_count_values),
        "observed_count_max": max(benchmark_count_values),
        "observed_count_avg": sum(benchmark_count_values) / len(benchmark_count_values),
    }

    split_defs = build_extreme_splits(
        benchmark_languages=benchmark_languages,
        counts=counts,
        order=order,
        fractions=[0.25, 1.0 / 3.0, 0.50],
    )
    split_defs["tertiles_within_benchmark"] = build_tertile_split(
        benchmark_languages=benchmark_languages,
        counts=counts,
        order=order,
    )

    report: dict[str, Any] = {
        "setup": {
            "typology_csv": typology_csv,
            "metadata_csv": metadata_csv,
            "gold": gold_path,
            "n_gold_items": len(gold),
            "n_models": len(model_paths),
            "focus_model_for_robustness": focus_model,
            "confidence_map_for_nonprobabilistic_models": confidence_map or DEFAULT_CONFIDENCE_MAP,
            "note": "Splits are recomputed from observed feature counts per language using only the benchmark languages already present in gold_eval_2.",
            "caution": "Because gold_eval_2 was built from top-50 and bottom-50 languages, the tertile split is relative to this benchmark subset rather than the full language inventory.",
        },
        "benchmark_language_stats": benchmark_stats,
        "splits": {},
    }

    normalized_by_model: dict[str, list[PredictionRow]] = {}
    for model_name, pred_path in model_paths.items():
        normalized_by_model[model_name] = normalize_predictions(gold, load_predictions_jsonl(pred_path))

    for split_name, split_def in split_defs.items():
        group_names = list(split_def["groups"].keys())
        groups_with_items = {
            group_name: _attach_group_item_counts(group_info, gold_items_by_language)
            for group_name, group_info in split_def["groups"].items()
        }
        model_metrics: dict[str, dict[str, Any]] = {}
        for model_name, rows in normalized_by_model.items():
            model_metrics[model_name] = {}
            for group_name in group_names:
                bucket_langs = groups_with_items[group_name]["languages"]
                bucket_rows = _collect_rows_for_languages(rows, bucket_langs)
                model_metrics[model_name][group_name] = compute_group_metrics(bucket_rows, confidence_map=confidence_map)

        split_report: dict[str, Any] = {
            "definition": {
                "kind": split_def["kind"],
            },
            "groups": groups_with_items,
            "metrics": model_metrics,
            "leaderboard_by_accuracy": _accuracy_leaderboard(model_metrics, group_names),
        }
        if "fraction" in split_def:
            split_report["definition"]["fraction"] = split_def["fraction"]
            split_report["definition"]["k_languages_per_side"] = split_def["k_languages_per_side"]
            for model_name, metrics in model_metrics.items():
                split_report["metrics"][model_name]["high_minus_low_accuracy"] = (
                    metrics["high"]["accuracy"] - metrics["low"]["accuracy"]
                )
                split_report["metrics"][model_name]["high_minus_low_f1"] = (
                    metrics["high"]["f1"] - metrics["low"]["f1"]
                )
        elif split_name == "tertiles_within_benchmark":
            for model_name, metrics in model_metrics.items():
                split_report["metrics"][model_name]["trend"] = {
                    "low_to_mid_accuracy": metrics["mid"]["accuracy"] - metrics["low"]["accuracy"],
                    "mid_to_high_accuracy": metrics["high"]["accuracy"] - metrics["mid"]["accuracy"],
                    "low_to_high_accuracy": metrics["high"]["accuracy"] - metrics["low"]["accuracy"],
                }

        report["splits"][split_name] = split_report

    report["low_resource_robustness"] = analyze_low_resource_robustness(
        low_languages=low_languages,
        counts=counts,
        order=order,
        gold_items_by_language=gold_items_by_language,
        normalized_by_model=normalized_by_model,
        confidence_map=confidence_map,
        metadata=metadata,
        focus_model=focus_model,
    )

    return report


def _parse_model_specs(specs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Model spec must look like name=path, got: {spec}")
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid model spec: {spec}")
        out[name] = path
    return out


def _build_stdout_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split_name, split_report in report["splits"].items():
        leaders = {}
        for group_name, rows in split_report["leaderboard_by_accuracy"].items():
            leaders[group_name] = rows[:3]
        summary[split_name] = leaders
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute multiple resource splits on existing benchmark predictions.")
    parser.add_argument("--typology_csv", type=str, default="data/derived/uriel+_typological.csv")
    parser.add_argument("--metadata_csv", type=str, default="data/derived/metadata.csv")
    parser.add_argument("--gold", type=str, default="data/benchmark/gold_eval_2.jsonl")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Repeated name=path entries for prediction JSONL files.",
    )
    parser.add_argument(
        "--focus_model",
        type=str,
        default=None,
        help="Model name used as the focal system in family/macro-area robustness summaries. Defaults to the first --model entry.",
    )
    parser.add_argument("--report_out", type=str, required=True)
    args = parser.parse_args()

    if not args.model:
        raise ValueError("At least one --model name=path entry is required.")

    model_paths = _parse_model_specs(args.model)
    report = analyze_resource_splits(
        typology_csv=args.typology_csv,
        metadata_csv=args.metadata_csv,
        gold_path=args.gold,
        model_paths=model_paths,
        confidence_map=DEFAULT_CONFIDENCE_MAP,
        focus_model=args.focus_model,
    )

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote report: {report_out}")
    print(json.dumps(_build_stdout_summary(report), indent=2))


if __name__ == "__main__":
    main()
