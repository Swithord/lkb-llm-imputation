from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from analyze_resource_splits import (
    build_extreme_splits,
    build_metadata_group_defs,
    build_tertile_split,
    summarize_focus_model_gains,
)


def test_build_extreme_splits_respects_sparse_and_dense_ends() -> None:
    benchmark_languages = ["lang_a", "lang_b", "lang_c", "lang_d", "lang_e", "lang_f"]
    counts = {
        "lang_a": 2,
        "lang_b": 5,
        "lang_c": 8,
        "lang_d": 12,
        "lang_e": 20,
        "lang_f": 25,
    }
    order = {lang: i for i, lang in enumerate(benchmark_languages)}

    splits = build_extreme_splits(benchmark_languages, counts, order, fractions=[1 / 3, 0.5])

    assert splits["extremes_bottom_top_33pct"]["groups"]["low"]["languages"] == ["lang_a", "lang_b"]
    assert splits["extremes_bottom_top_33pct"]["groups"]["high"]["languages"] == ["lang_f", "lang_e"]
    assert splits["extremes_bottom_top_50pct"]["groups"]["low"]["languages"] == ["lang_a", "lang_b", "lang_c"]
    assert splits["extremes_bottom_top_50pct"]["groups"]["high"]["languages"] == ["lang_f", "lang_e", "lang_d"]


def test_build_tertile_split_uses_within_benchmark_order() -> None:
    benchmark_languages = ["lang_a", "lang_b", "lang_c", "lang_d", "lang_e", "lang_f", "lang_g"]
    counts = {
        "lang_a": 1,
        "lang_b": 2,
        "lang_c": 3,
        "lang_d": 4,
        "lang_e": 5,
        "lang_f": 6,
        "lang_g": 7,
    }
    order = {lang: i for i, lang in enumerate(benchmark_languages)}

    split = build_tertile_split(benchmark_languages, counts, order)

    assert split["groups"]["low"]["languages"] == ["lang_a", "lang_b"]
    assert split["groups"]["mid"]["languages"] == ["lang_c", "lang_d"]
    assert split["groups"]["high"]["languages"] == ["lang_e", "lang_f", "lang_g"]


def test_build_metadata_group_defs_uses_family_or_unknown() -> None:
    languages = ["lang_a", "lang_b", "lang_c", "lang_d"]
    counts = {
        "lang_a": 3,
        "lang_b": 8,
        "lang_c": 5,
        "lang_d": 1,
    }
    order = {lang: i for i, lang in enumerate(languages)}
    metadata = {
        "lang_a": {"family_name": "Fam1", "primary_macroarea": "AreaX"},
        "lang_b": {"family_name": "Fam1", "primary_macroarea": "AreaX"},
        "lang_c": {"family_name": "Fam2", "primary_macroarea": "AreaY"},
    }

    group_defs = build_metadata_group_defs(
        languages=languages,
        counts=counts,
        order=order,
        metadata=metadata,
        metadata_field="family_name",
    )

    assert list(group_defs.keys()) == ["Fam1", "Fam2", "Unknown"]
    assert group_defs["Fam1"]["languages"] == ["lang_a", "lang_b"]
    assert group_defs["Fam2"]["languages"] == ["lang_c"]
    assert group_defs["Unknown"]["languages"] == ["lang_d"]


def test_summarize_focus_model_gains_filters_to_multi_language_groups() -> None:
    group_reports = {
        "Fam1": {
            "n_languages": 3,
            "n_items": 30,
            "leaderboard_by_accuracy": [{"model": "llm"}],
            "leaderboard_by_f1": [{"model": "llm"}],
            "focus_model_gain_vs_best_baseline": {
                "focus_minus_best_baseline_accuracy": 0.10,
                "focus_minus_best_baseline_f1": 0.04,
                "best_baseline_by_accuracy": "knn",
                "best_baseline_by_f1": "knn",
            },
        },
        "Fam2": {
            "n_languages": 2,
            "n_items": 20,
            "leaderboard_by_accuracy": [{"model": "knn"}],
            "leaderboard_by_f1": [{"model": "knn"}],
            "focus_model_gain_vs_best_baseline": {
                "focus_minus_best_baseline_accuracy": -0.05,
                "focus_minus_best_baseline_f1": -0.02,
                "best_baseline_by_accuracy": "knn",
                "best_baseline_by_f1": "knn",
            },
        },
        "Fam3": {
            "n_languages": 1,
            "n_items": 50,
            "leaderboard_by_accuracy": [{"model": "llm"}],
            "leaderboard_by_f1": [{"model": "llm"}],
            "focus_model_gain_vs_best_baseline": {
                "focus_minus_best_baseline_accuracy": 0.30,
                "focus_minus_best_baseline_f1": 0.20,
                "best_baseline_by_accuracy": "knn",
                "best_baseline_by_f1": "knn",
            },
        },
    }

    summary = summarize_focus_model_gains(group_reports, focus_model="llm", min_languages_for_summary=2)

    assert summary["n_groups_considered"] == 2
    assert summary["groups_with_positive_accuracy_gain"] == 1
    assert summary["groups_with_positive_f1_gain"] == 1
    assert summary["weighted_item_share_positive_accuracy_gain"] == 30 / 50
    assert summary["focus_model_best_by_accuracy_in_groups"] == 1
    assert summary["focus_model_best_by_f1_in_groups"] == 1
    assert summary["top_accuracy_gains"][0]["group"] == "Fam1"
    assert summary["lowest_accuracy_gains"][0]["group"] == "Fam2"
