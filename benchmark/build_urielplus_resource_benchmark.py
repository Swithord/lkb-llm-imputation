from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_prompting_module(path: str):
    spec = importlib.util.spec_from_file_location("prompting_impl", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load prompting module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _allowed_values_from_df(df: pd.DataFrame, feature: str) -> list[str]:
    if feature not in df.columns:
        return ["0", "1"]
    vals = [v for v in df[feature].unique().tolist() if v != -1 and not pd.isna(v)]
    if not vals:
        return ["0", "1"]
    out: list[str] = []
    for v in vals:
        if isinstance(v, float) and v.is_integer():
            out.append(str(int(v)))
        else:
            out.append(str(v))
    return sorted(set(out))


def _rewrite_output_contract(user_prompt: str) -> str:
    prefix = user_prompt.rstrip()
    for marker in ("Output format (STRICT JSON):", "Output format (STRICT):"):
        if marker in user_prompt:
            prefix = user_prompt.split(marker, 1)[0].rstrip()
            break
    return (
        f"{prefix}\n"
        "Reasoning guidance:\n"
        "- Compare the support for value 0 versus value 1.\n"
        "- Neighbor counts are useful, but do not follow majority vote blindly.\n"
        "- A smaller number of closer or more relevant neighbors may outweigh a larger but weaker group.\n"
        "- It is acceptable to predict a minority value if it is better supported by the overall evidence.\n"
        "- Use prevalence only as a weak tie-breaker if the evidence is otherwise balanced.\n"
        "Output format (STRICT JSON):\n"
        "Output ONLY valid JSON.\n"
        "Return exactly one minified JSON object on one line with keys in this exact order:\n"
        '{"value":"<one allowed value>","confidence":"low|medium|high","rationale":"<max 20 words>"}\n'
        "Use double quotes only. No Markdown, no code fences, no extra text.\n"
        "Few-shot examples:\n"
        '{"value":"1","confidence":"medium","rationale":"Most neighbors are 0, but the closest and most similar languages support 1."}\n'
        '{"value":"1","confidence":"high","rationale":"Observed features and nearest phylogenetic evidence align strongly with value 1."}\n'
        '{"value":"0","confidence":"low","rationale":"Evidence is balanced, so the weak prevalence prior favors 0."}'
    )


def _extract_structured_description(user_prompt: str) -> str:
    marker = "\nTask:"
    if marker in user_prompt:
        return user_prompt.split(marker, 1)[0].rstrip()
    return user_prompt.rstrip()


def _parse_binary_gold(gold_value: str, item_id: str) -> int:
    try:
        val = int(float(gold_value))
    except Exception as e:
        raise ValueError(f"Gold value is not numeric for {item_id}: {gold_value}") from e
    if val not in (0, 1):
        raise ValueError(f"Gold value is not binary for {item_id}: {gold_value}")
    return val


def _feature_type(feature: str) -> str:
    return feature.split("_", 1)[0]


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


def _build_budget_visible_mask(
    typ_df: pd.DataFrame,
    resource_groups: dict[str, list[str]],
    budgets_by_group: dict[str, int],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    values = typ_df.to_numpy()
    observed = (values != -1) & (~pd.isna(values))
    visible = observed.copy()
    row_idx = {str(lang): i for i, lang in enumerate(typ_df.index.astype(str).tolist())}

    group_stats: dict[str, Any] = {}
    for group, langs in resource_groups.items():
        budget = budgets_by_group.get(group)
        if budget is None:
            continue
        affected = 0
        originally_observed = 0
        kept_observed = 0
        hidden_observed = 0
        for lang in langs:
            ri = row_idx.get(str(lang))
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
            rng = random.Random(_stable_row_seed(seed, str(lang)))
            keep_cols = rng.sample(obs_cols.tolist(), keep_n) if keep_n > 0 else []
            visible[ri, :] = False
            if keep_cols:
                visible[ri, keep_cols] = True
            kept_observed += keep_n
            hidden_observed += (n_obs - keep_n)

        group_stats[group] = {
            "budget": int(budget),
            "languages_affected": int(affected),
            "originally_observed_cells": int(originally_observed),
            "kept_observed_cells": int(kept_observed),
            "hidden_observed_cells": int(hidden_observed),
        }

    observed_df = pd.DataFrame(observed, index=typ_df.index, columns=typ_df.columns)
    visible_df = pd.DataFrame(visible, index=typ_df.index, columns=typ_df.columns)
    meta = {
        "enabled": bool(budgets_by_group),
        "budgets_by_group": {k: int(v) for k, v in budgets_by_group.items()},
        "seed": int(seed),
        "protocol": "Keep at most N observed features per language by resource group; evaluate on hidden observed cells.",
        "total_observed_cells": int(observed.sum()),
        "total_visible_observed_cells": int((observed & visible).sum()),
        "total_hidden_observed_cells": int((observed & (~visible)).sum()),
        "group_stats": group_stats,
    }
    return visible_df, observed_df, meta


def _load_coverage_groups(
    typ_df: pd.DataFrame,
    lrl_frac: float,
    mrl_frac: float,
    min_observed: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    if lrl_frac <= 0 or mrl_frac <= 0 or lrl_frac + mrl_frac >= 1:
        raise ValueError("Need lrl_frac>0, mrl_frac>0, and lrl_frac + mrl_frac < 1.")

    observed_counts = (typ_df != -1).sum(axis=1).astype(int)
    eligible = observed_counts[observed_counts >= min_observed].sort_values(kind="stable")
    if eligible.empty:
        raise ValueError("No languages meet the minimum observed-feature threshold.")

    languages = eligible.index.astype(str).tolist()
    counts = eligible.tolist()
    n = len(languages)
    lrl_end = int(round(n * lrl_frac))
    mrl_end = int(round(n * (lrl_frac + mrl_frac)))
    lrl_end = max(1, min(n - 2, lrl_end))
    mrl_end = max(lrl_end + 1, min(n - 1, mrl_end))

    groups = {
        "lrl": languages[:lrl_end],
        "mrl": languages[lrl_end:mrl_end],
        "hrl": languages[mrl_end:],
    }
    meta: dict[str, Any] = {
        "split_definition": "coverage_quantiles_on_languages_with_observed_features",
        "min_observed_threshold": min_observed,
        "fractions": {
            "lrl": lrl_frac,
            "mrl": mrl_frac,
            "hrl": 1.0 - lrl_frac - mrl_frac,
        },
        "n_languages_total": int(typ_df.shape[0]),
        "n_languages_eligible": int(n),
        "n_languages_excluded_below_threshold": int(typ_df.shape[0] - n),
        "groups": {},
    }
    for label, langs in groups.items():
        group_counts = [int(observed_counts[lang]) for lang in langs]
        if not group_counts:
            raise ValueError(f"Coverage split produced an empty group: {label}")
        meta["groups"][label] = {
            "n_languages": len(langs),
            "observed_count_min": min(group_counts),
            "observed_count_max": max(group_counts),
            "observed_count_avg": sum(group_counts) / len(group_counts),
            "missing_count_min": int(typ_df.shape[1]) - max(group_counts),
            "missing_count_max": int(typ_df.shape[1]) - min(group_counts),
            "missing_count_avg": int(typ_df.shape[1]) - (sum(group_counts) / len(group_counts)),
        }
    return groups, meta


def _load_threshold_coverage_groups(
    typ_df: pd.DataFrame,
    lrl_max_observed: int,
    hrl_min_observed: int,
    min_observed: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    if min_observed < 1:
        raise ValueError("Need min_observed >= 1 for threshold-based splits.")
    if lrl_max_observed < min_observed:
        raise ValueError("Need lrl_max_observed >= min_observed.")
    if hrl_min_observed <= lrl_max_observed:
        raise ValueError("Need hrl_min_observed > lrl_max_observed.")

    observed_counts = (typ_df != -1).sum(axis=1).astype(int)
    eligible = observed_counts[observed_counts >= min_observed].sort_values(kind="stable")
    if eligible.empty:
        raise ValueError("No languages meet the minimum observed-feature threshold.")

    lrl = eligible[eligible <= lrl_max_observed]
    mrl = eligible[(eligible > lrl_max_observed) & (eligible < hrl_min_observed)]
    hrl = eligible[eligible >= hrl_min_observed]
    groups = {
        "lrl": lrl.index.astype(str).tolist(),
        "mrl": mrl.index.astype(str).tolist(),
        "hrl": hrl.index.astype(str).tolist(),
    }
    for label, langs in groups.items():
        if not langs:
            raise ValueError(f"Threshold split produced an empty group: {label}")

    meta: dict[str, Any] = {
        "split_definition": "coverage_absolute_thresholds_on_languages_with_observed_features",
        "min_observed_threshold": min_observed,
        "thresholds": {
            "lrl_max_observed": lrl_max_observed,
            "mrl_min_observed": lrl_max_observed + 1,
            "mrl_max_observed": hrl_min_observed - 1,
            "hrl_min_observed": hrl_min_observed,
        },
        "n_languages_total": int(typ_df.shape[0]),
        "n_languages_eligible": int(len(eligible)),
        "n_languages_excluded_below_threshold": int(typ_df.shape[0] - len(eligible)),
        "groups": {},
    }
    for label, langs in groups.items():
        group_counts = [int(observed_counts[lang]) for lang in langs]
        meta["groups"][label] = {
            "n_languages": len(langs),
            "observed_count_min": min(group_counts),
            "observed_count_max": max(group_counts),
            "observed_count_avg": sum(group_counts) / len(group_counts),
            "missing_count_min": int(typ_df.shape[1]) - max(group_counts),
            "missing_count_max": int(typ_df.shape[1]) - min(group_counts),
            "missing_count_avg": int(typ_df.shape[1]) - (sum(group_counts) / len(group_counts)),
        }
    return groups, meta


def _load_bottomk_coverage_groups(
    typ_df: pd.DataFrame,
    lrl_bottom_k: int,
    hrl_min_observed: int,
    min_observed: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    if min_observed < 1:
        raise ValueError("Need min_observed >= 1 for bottom-k splits.")
    if lrl_bottom_k < 1:
        raise ValueError("Need lrl_bottom_k >= 1.")

    observed_counts = (typ_df != -1).sum(axis=1).astype(int)
    eligible = observed_counts[observed_counts >= min_observed]
    if eligible.empty:
        raise ValueError("No languages meet the minimum observed-feature threshold.")
    if lrl_bottom_k >= len(eligible):
        raise ValueError(
            f"Need lrl_bottom_k < number of eligible languages; got lrl_bottom_k={lrl_bottom_k}, "
            f"eligible={len(eligible)}."
        )

    ranked_low = eligible.reset_index()
    ranked_low.columns = ["language", "observed_count"]
    ranked_low["language"] = ranked_low["language"].astype(str)
    ranked_low = ranked_low.sort_values(["observed_count", "language"], ascending=[True, True], kind="stable")
    lrl_langs = ranked_low.head(lrl_bottom_k)["language"].tolist()
    lrl_set = set(lrl_langs)

    ranked_high = eligible[eligible >= hrl_min_observed].sort_values(ascending=False, kind="stable")
    hrl_langs = [str(lang) for lang in ranked_high.index.astype(str).tolist() if str(lang) not in lrl_set]
    hrl_set = set(hrl_langs)

    mrl_langs = [
        str(lang)
        for lang in eligible.index.astype(str).tolist()
        if str(lang) not in lrl_set and str(lang) not in hrl_set
    ]
    groups = {
        "lrl": sorted(lrl_langs),
        "mrl": sorted(mrl_langs),
        "hrl": sorted(hrl_langs),
    }
    for label, langs in groups.items():
        if not langs:
            raise ValueError(f"Bottom-k split produced an empty group: {label}")

    observed_values_low = [int(observed_counts[lang]) for lang in lrl_langs]
    lrl_cutoff = max(observed_values_low)
    meta: dict[str, Any] = {
        "split_definition": "coverage_bottomk_low_with_dense_high_threshold",
        "min_observed_threshold": min_observed,
        "bottomk": {
            "lrl_bottom_k": int(lrl_bottom_k),
            "lrl_ordering": "observed_count_asc_then_glottocode_asc",
            "lrl_observed_cutoff_at_k": int(lrl_cutoff),
            "hrl_min_observed": int(hrl_min_observed),
            "mrl_definition": "eligible_minus_lrl_minus_hrl",
        },
        "n_languages_total": int(typ_df.shape[0]),
        "n_languages_eligible": int(len(eligible)),
        "n_languages_excluded_below_threshold": int(typ_df.shape[0] - len(eligible)),
        "groups": {},
    }
    for label, langs in groups.items():
        group_counts = [int(observed_counts[lang]) for lang in langs]
        meta["groups"][label] = {
            "n_languages": len(langs),
            "observed_count_min": min(group_counts),
            "observed_count_max": max(group_counts),
            "observed_count_avg": sum(group_counts) / len(group_counts),
            "missing_count_min": int(typ_df.shape[1]) - max(group_counts),
            "missing_count_max": int(typ_df.shape[1]) - min(group_counts),
            "missing_count_avg": int(typ_df.shape[1]) - (sum(group_counts) / len(group_counts)),
        }
    return groups, meta


def _load_topk_map(topk_df: pd.DataFrame) -> dict[str, list[str]]:
    tmp: dict[str, list[tuple[int, str]]] = {}
    for _, row in topk_df.iterrows():
        feat = str(row["feature"])
        other = str(row["other_feature"])
        rank = int(row["rank"]) if "rank" in row and not pd.isna(row["rank"]) else 0
        tmp.setdefault(feat, []).append((rank, other))
    return {feat: [other for _, other in sorted(items, key=lambda x: x[0])] for feat, items in tmp.items()}


def _build_candidate_index(
    typ_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    resource_groups: dict[str, list[str]],
    feature_types: list[str],
    candidate_mask: pd.DataFrame | None = None,
) -> dict[str, dict[str, dict[str, list[dict[str, str]]]]]:
    candidates: dict[str, dict[str, dict[str, list[dict[str, str]]]]] = {
        group: {ftype: defaultdict(list) for ftype in feature_types}
        for group in resource_groups
    }
    for group, langs in resource_groups.items():
        for lang in langs:
            if lang not in typ_df.index:
                continue
            family = "Unknown"
            if lang in meta_df.index:
                raw_family = meta_df.at[lang, "family_name"]
                family = "Unknown" if pd.isna(raw_family) or str(raw_family).strip() == "" else str(raw_family).strip()
            row = typ_df.loc[lang]
            mask_row = candidate_mask.loc[lang] if candidate_mask is not None else None
            for feat, value in row.items():
                if value == -1 or pd.isna(value):
                    continue
                if mask_row is not None and not bool(mask_row.at[feat]):
                    continue
                gold_value = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
                ftype = _feature_type(str(feat))
                candidates[group][ftype][family].append(
                    {
                        "language": lang,
                        "feature": str(feat),
                        "family": family,
                        "gold_value": gold_value,
                    }
                )
    return candidates


def _sample_group_items(
    group: str,
    requested_total: int,
    feature_types: list[str],
    min_per_family_type: int,
    candidates_by_type: dict[str, dict[str, list[dict[str, str]]]],
    seed: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rng = random.Random(seed)
    available = {
        ftype: sum(len(bucket) for bucket in candidates_by_type.get(ftype, {}).values())
        for ftype in feature_types
    }
    family_floor = {
        ftype: sum(
            min(min_per_family_type, len(items))
            for items in candidates_by_type.get(ftype, {}).values()
            if items
        )
        for ftype in feature_types
    }
    reserved_total = sum(family_floor.values())
    max_total = sum(available.values())
    realized_total = min(requested_total, max_total)
    if realized_total < reserved_total:
        raise ValueError(
            f"Requested total for {group} is too small to satisfy the per-family feature floor: "
            f"requested={requested_total}, reserved_floor={reserved_total}"
        )

    selected: list[dict[str, str]] = []
    remaining_pool: list[dict[str, str]] = []
    by_type_summary: dict[str, Any] = {}
    languages_used: set[str] = set()
    families_used: set[str] = set()

    for ftype in feature_types:
        family_buckets = candidates_by_type.get(ftype, {})
        preselected: list[dict[str, str]] = []
        remaining: list[dict[str, str]] = []
        family_summary: dict[str, Any] = {}
        for family, bucket in sorted(family_buckets.items()):
            items = list(bucket)
            rng.shuffle(items)
            reserve_n = min(min_per_family_type, len(items))
            preselected.extend(items[:reserve_n])
            remaining.extend(items[reserve_n:])
            family_summary[family] = {
                "available": len(items),
                "reserved_floor": reserve_n,
            }

        selected.extend(preselected)
        remaining_pool.extend(remaining)
        for item in preselected:
            languages_used.add(item["language"])
            families_used.add(item["family"])
        sampled_by_family = defaultdict(int)
        for item in preselected:
            sampled_by_family[item["family"]] += 1
        by_type_summary[ftype] = {
            "available": available[ftype],
            "family_floor_requirement": family_floor[ftype],
            "reserved_floor_sampled": len(preselected),
            "sampled_by_family_after_floor": dict(sorted(sampled_by_family.items())),
            "family_bucket_summary": family_summary,
        }

    extra_needed = realized_total - len(selected)
    rng.shuffle(remaining_pool)
    extra = remaining_pool[:extra_needed]
    selected.extend(extra)
    for item in extra:
        languages_used.add(item["language"])
        families_used.add(item["family"])

    final_by_type = defaultdict(int)
    final_by_family_type = defaultdict(lambda: defaultdict(int))
    for item in selected:
        ftype = _feature_type(item["feature"])
        final_by_type[ftype] += 1
        final_by_family_type[ftype][item["family"]] += 1
    for ftype in feature_types:
        by_type_summary[ftype]["sampled"] = int(final_by_type.get(ftype, 0))
        by_type_summary[ftype]["sampled_by_family"] = dict(sorted(final_by_family_type[ftype].items()))

    rng.shuffle(selected)
    summary = {
        "group": group,
        "requested_items": requested_total,
        "realized_items": realized_total,
        "available_by_type": available,
        "family_floor_by_type": family_floor,
        "reserved_floor_total": reserved_total,
        "n_languages_used": len(languages_used),
        "n_families_used": len(families_used),
        "languages_used": sorted(languages_used),
        "families_used": sorted(families_used),
        "feature_types": by_type_summary,
    }
    return selected[:realized_total], summary


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build a new benchmark with HRL/MRL/LRL defined by observed URIEL+ feature coverage."
    )
    p.add_argument("--prompting_py", type=str, default="glottolog-tree/prompting.py")
    p.add_argument("--typ", type=str, default="data/derived/uriel+_typological.csv")
    p.add_argument("--meta", type=str, default="data/derived/metadata.csv")
    p.add_argument("--topk_csv", type=str, default="data/features/topk_per_feature.csv")
    p.add_argument("--gen", type=str, default="data/derived/genetic_neighbours.json")
    p.add_argument("--gen_detail", type=str, default="data/derived/genetic_neighbours_detailed.json")
    p.add_argument("--geo", type=str, default="data/derived/geographic_neighbours.json")
    p.add_argument("--retrieval_backend", type=str, default="legacy")
    p.add_argument("--kg_nodes", type=str, default=None)
    p.add_argument("--kg_edges", type=str, default=None)
    p.add_argument("--split_mode", choices=("quantile", "threshold", "bottomk"), default="threshold")
    p.add_argument("--lrl_frac", type=float, default=0.50)
    p.add_argument("--mrl_frac", type=float, default=0.30)
    p.add_argument("--min_observed_for_split", type=int, default=1)
    p.add_argument("--lrl_max_observed", type=int, default=15)
    p.add_argument("--hrl_min_observed", type=int, default=240)
    p.add_argument("--lrl_bottom_k", type=int, default=200)
    p.add_argument(
        "--budget_protocol",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply per-group per-language context budgets and sample eval items from budget-hidden observed cells.",
    )
    p.add_argument("--budget_lrl", type=int, default=5)
    p.add_argument("--budget_mrl", type=int, default=20)
    p.add_argument("--budget_hrl", type=int, default=80)
    p.add_argument("--budget_seed", type=int, default=17)
    p.add_argument("--target_hrl", type=int, default=2000)
    p.add_argument("--target_mrl", type=int, default=3000)
    p.add_argument("--target_lrl", type=int, default=5000)
    p.add_argument(
        "--target_each",
        type=int,
        default=None,
        help="If set, override target_hrl/target_mrl/target_lrl to the same value.",
    )
    p.add_argument("--min_per_family_type", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--top_n", type=int, default=10)
    p.add_argument("--prompt_version", type=str, default="v5_glottolog_tree_json")
    p.add_argument(
        "--emit_prompt_artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to build prompt/example JSONL artifacts in addition to gold/groups/manifest.",
    )
    p.add_argument(
        "--include_vote_table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include compact vote summary table in prompt context.",
    )
    p.add_argument("--prompts_out", type=str, default="data/benchmark/prompts_eval_coverage_resource_v1.jsonl")
    p.add_argument("--gold_out", type=str, default="data/benchmark/gold_eval_coverage_resource_v1.jsonl")
    p.add_argument("--examples_out", type=str, default="data/benchmark/examples_eval_coverage_resource_v1.jsonl")
    p.add_argument("--groups_out", type=str, default="data/benchmark/language_groups_coverage_resource_v1.json")
    p.add_argument("--manifest_out", type=str, default="data/benchmark/manifest_coverage_resource_v1.json")
    args = p.parse_args()

    typ_df = pd.read_csv(args.typ, index_col=0)
    typ_df.index = typ_df.index.astype(str)
    meta_df = pd.read_csv(args.meta, index_col=0)
    meta_df.index = meta_df.index.astype(str)
    topk_df = pd.read_csv(args.topk_csv)

    with open(args.gen, "r", encoding="utf-8") as f:
        genetic = {str(k): v for k, v in json.load(f).items()}
    gen_detail_path = Path(args.gen_detail)
    if gen_detail_path.exists():
        with gen_detail_path.open("r", encoding="utf-8") as f:
            genetic_detail = {str(k): v for k, v in json.load(f).items()}
    else:
        genetic_detail = {}
    with open(args.geo, "r", encoding="utf-8") as f:
        geographic = {str(k): v for k, v in json.load(f).items()}

    feature_types: list[str] = []
    seen_feature_types: set[str] = set()
    for feature in typ_df.columns.astype(str).tolist():
        ftype = _feature_type(feature)
        if ftype not in seen_feature_types:
            feature_types.append(ftype)
            seen_feature_types.add(ftype)

    if args.target_each is not None:
        if args.target_each < 1:
            raise ValueError(f"Need target_each >= 1; got {args.target_each}.")
        args.target_hrl = int(args.target_each)
        args.target_mrl = int(args.target_each)
        args.target_lrl = int(args.target_each)

    if args.split_mode == "threshold":
        resource_groups, resource_meta = _load_threshold_coverage_groups(
            typ_df=typ_df,
            lrl_max_observed=args.lrl_max_observed,
            hrl_min_observed=args.hrl_min_observed,
            min_observed=args.min_observed_for_split,
        )
    elif args.split_mode == "quantile":
        resource_groups, resource_meta = _load_coverage_groups(
            typ_df=typ_df,
            lrl_frac=args.lrl_frac,
            mrl_frac=args.mrl_frac,
            min_observed=args.min_observed_for_split,
        )
    else:
        resource_groups, resource_meta = _load_bottomk_coverage_groups(
            typ_df=typ_df,
            lrl_bottom_k=args.lrl_bottom_k,
            hrl_min_observed=args.hrl_min_observed,
            min_observed=args.min_observed_for_split,
        )
    budgets_by_group = _budget_map_from_args(
        enabled=args.budget_protocol,
        budget_lrl=args.budget_lrl,
        budget_mrl=args.budget_mrl,
        budget_hrl=args.budget_hrl,
    )
    if budgets_by_group:
        visible_observed_mask, observed_mask, budget_meta = _build_budget_visible_mask(
            typ_df=typ_df,
            resource_groups=resource_groups,
            budgets_by_group=budgets_by_group,
            seed=args.budget_seed,
        )
        candidate_observed_mask = observed_mask & (~visible_observed_mask)
    else:
        observed_np = (typ_df.to_numpy() != -1) & (~pd.isna(typ_df.to_numpy()))
        observed_mask = pd.DataFrame(observed_np, index=typ_df.index, columns=typ_df.columns)
        visible_observed_mask = observed_mask.copy()
        candidate_observed_mask = observed_mask.copy()
        budget_meta = {
            "enabled": False,
            "protocol": "No per-language budgeting; candidate pool is all observed cells.",
            "total_observed_cells": int(observed_np.sum()),
            "total_visible_observed_cells": int(observed_np.sum()),
            "total_hidden_observed_cells": 0,
            "group_stats": {},
        }

    candidates = _build_candidate_index(
        typ_df=typ_df,
        meta_df=meta_df,
        resource_groups=resource_groups,
        feature_types=feature_types,
        candidate_mask=candidate_observed_mask,
    )

    requested_targets = {
        "hrl": args.target_hrl,
        "mrl": args.target_mrl,
        "lrl": args.target_lrl,
    }
    sampled: dict[str, list[dict[str, str]]] = {}
    group_summaries: dict[str, Any] = {}
    for offset, group in enumerate(("hrl", "mrl", "lrl")):
        sampled[group], group_summaries[group] = _sample_group_items(
            group=group,
            requested_total=requested_targets[group],
            feature_types=feature_types,
            min_per_family_type=args.min_per_family_type,
            candidates_by_type=candidates[group],
            seed=args.seed + offset,
        )

    selected_rows: list[tuple[str, dict[str, str]]] = []
    for group in ("hrl", "mrl", "lrl"):
        for item in sampled[group]:
            selected_rows.append((group, item))
    random.Random(args.seed).shuffle(selected_rows)

    for out_path in (args.prompts_out, args.gold_out, args.examples_out, args.groups_out, args.manifest_out):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    benchmark_languages = {
        group: sorted({str(lang) for lang in resource_groups.get(group, [])})
        for group in ("hrl", "mrl", "lrl")
    }

    with open(args.gold_out, "w", encoding="utf-8") as fg:
        for i, (group, item) in enumerate(selected_rows):
            lang = item["language"]
            feat = item["feature"]
            gold_value = item["gold_value"]
            item_id = f"{group}:{lang}:{feat}:{i}"
            allowed_values = _allowed_values_from_df(typ_df, feat)
            gold_rec = {
                "id": item_id,
                "resource_group": group,
                "language": lang,
                "feature": feat,
                "gold_value": gold_value,
                "allowed_values": allowed_values,
            }
            fg.write(json.dumps(gold_rec, ensure_ascii=False) + "\n")

    if args.emit_prompt_artifacts:
        prompting = _load_prompting_module(args.prompting_py)
        prompting.typ_df = typ_df.where(visible_observed_mask, other=-1).copy()
        prompting.metadata_df = meta_df
        prompting.genetic_neighbours = genetic
        if hasattr(prompting, "genetic_neighbour_details"):
            prompting.genetic_neighbour_details = genetic_detail
        prompting.geographic_neighbours = geographic
        prompting.top_n_features = args.top_n
        prompting.topk_map = _load_topk_map(topk_df)
        prompting.set_prompt_options(args.prompt_version, args.include_vote_table)
        if hasattr(prompting, "set_retrieval_options"):
            prompting.set_retrieval_options(args.retrieval_backend, args.kg_nodes, args.kg_edges)

        with open(args.prompts_out, "w", encoding="utf-8") as fp, open(args.examples_out, "w", encoding="utf-8") as fe:
            for i, (group, item) in enumerate(selected_rows):
                lang = item["language"]
                feat = item["feature"]
                gold_value = item["gold_value"]
                item_id = f"{group}:{lang}:{feat}:{i}"
                allowed_values = _allowed_values_from_df(typ_df, feat)

                original = prompting.typ_df.at[lang, feat]
                prompting.typ_df.at[lang, feat] = -1
                try:
                    system, user = prompting.construct_prompt(lang, feat)
                finally:
                    prompting.typ_df.at[lang, feat] = original
                user = _rewrite_output_contract(user)

                prompt_rec = {
                    "id": item_id,
                    "resource_group": group,
                    "language": lang,
                    "feature": feat,
                    "allowed_values": allowed_values,
                    "system": system,
                    "user": user,
                }
                example_rec = {
                    "example_id": item_id,
                    "language_id": lang,
                    "feature_id": feat,
                    "gold": _parse_binary_gold(gold_value, item_id),
                    "resource_group": group,
                    "r_L": _extract_structured_description(user),
                }
                fp.write(json.dumps(prompt_rec, ensure_ascii=False) + "\n")
                fe.write(json.dumps(example_rec, ensure_ascii=False) + "\n")

    manifest = {
        "name": "coverage_resource_v1",
        "resource_definition": {
            "metadata": resource_meta,
        },
        "sampling": {
            "seed": args.seed,
            "feature_types": feature_types,
            "min_per_family_type": args.min_per_family_type,
            "requested_targets": requested_targets,
            "candidate_pool_definition": (
                "observed cells hidden by per-language budget protocol"
                if budgets_by_group
                else "all observed cells"
            ),
            "context_budget_protocol": budget_meta,
            "realized_total_items": sum(summary["realized_items"] for summary in group_summaries.values()),
            "group_summaries": group_summaries,
        },
        "artifacts": {
            "prompts": args.prompts_out if args.emit_prompt_artifacts else None,
            "gold": args.gold_out,
            "examples": args.examples_out if args.emit_prompt_artifacts else None,
            "language_groups": args.groups_out,
        },
    }

    Path(args.groups_out).write_text(json.dumps(benchmark_languages, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.manifest_out).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote prompts: {args.prompts_out}")
    print(f"Wrote gold: {args.gold_out}")
    print(f"Wrote examples: {args.examples_out}")
    print(f"Wrote groups: {args.groups_out}")
    print(f"Wrote manifest: {args.manifest_out}")
    print(f"Num prompts: {len(selected_rows)}")


if __name__ == "__main__":
    main()
