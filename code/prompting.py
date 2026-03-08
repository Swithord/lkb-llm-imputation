from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd


SYSTEM_MESSAGE = (
    "You are a linguistics expert specializing in typology.\n"
    "Infer missing typological features using:\n"
    "(1) observed facts about the target language,\n"
    "(2) evidence from phylogenetic and geographic neighbors,\n"
    "and (3) well-established linguistic universals.\n"
    "If evidence conflicts, prefer phylogenetic evidence over geographic evidence.\n"
    "If uncertainty remains, choose the most typologically common value."
)

typ_df: Optional[pd.DataFrame]
metadata_df: pd.DataFrame
genetic_neighbours: dict
geographic_neighbours: dict
top_n_features: int
topk_map: Dict[str, List[str]]
PROMPT_VERSION: str = "v3_strict_json"
INCLUDE_VOTE_TABLE: bool = True


def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass
    return v == -1


def _format_value(v) -> str:
    if _is_missing(v):
        return "Unknown"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _get_meta_value(row: Optional[pd.Series], col: str, default: str) -> str:
    if row is None or col not in row.index:
        return default
    val = row[col]
    if _is_missing(val) or (isinstance(val, str) and val.strip() == ""):
        return default
    return str(val)


def _family_lineage(row: Optional[pd.Series]) -> str:
    if row is None:
        return "Isolate"
    family = _get_meta_value(row, "family_name", "")
    parent = _get_meta_value(row, "parent_name", "")
    if not family and not parent:
        return "Isolate"
    if family and parent and parent != family:
        return f"{family} > {parent}"
    return family or parent or "Isolate"


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def _top_correlated_features(feature: str, top_n: int) -> List[str]:
    feats = topk_map.get(feature, [])
    if not feats:
        return []
    return feats[:top_n] if top_n > 0 else feats


def _effective_correlated_features(language: str, feature: str, primary_n: int, fallback_n: int = 15) -> List[str]:
    """
    Adaptive correlated-feature selection:
    start from top-N, and expand to top-15 when observed coverage is too sparse.
    """
    ranked = _top_correlated_features(feature, 0)
    filtered = [f for f in ranked if f != feature and (typ_df is None or f in typ_df.columns)]
    if not filtered:
        return []
    if primary_n <= 0:
        return filtered

    primary = filtered[: min(primary_n, len(filtered))]
    if typ_df is None or language not in typ_df.index:
        return primary

    observed_primary = sum(1 for f in primary if _lang_has_feature_value(language, f))
    min_needed = 3 if primary_n >= 5 else 1
    if observed_primary >= min_needed:
        return primary

    if fallback_n <= primary_n:
        return primary

    extended = filtered[: min(fallback_n, len(filtered))]
    observed_extended = sum(1 for f in extended if _lang_has_feature_value(language, f))
    if observed_extended > observed_primary:
        return extended
    return primary


def _neighbor_k_for_language(neighbour_map: dict, language: str, fallback: int = 5) -> int:
    direct = neighbour_map.get(language, [])
    if isinstance(direct, list) and len(direct) > 0:
        return len(direct)
    lengths = [len(v) for v in neighbour_map.values() if isinstance(v, list) and len(v) > 0]
    if lengths:
        return max(lengths)
    return fallback


def _lang_has_feature_value(lang: str, feat: str) -> bool:
    if typ_df is None or lang not in typ_df.index or feat not in typ_df.columns:
        return False
    return not _is_missing(typ_df.at[lang, feat])


def _observed_correlated_set(lang: str, correlated: Sequence[str]) -> set[str]:
    observed = set()
    for feat in correlated:
        if _lang_has_feature_value(lang, feat):
            observed.add(feat)
    return observed


def _all_language_codes() -> List[str]:
    if typ_df is not None:
        return [str(x) for x in typ_df.index.tolist()]
    return [str(x) for x in metadata_df.index.tolist()]


def _meta_lat_lon(gc: str) -> Tuple[Optional[float], Optional[float]]:
    if gc not in metadata_df.index:
        return None, None
    row = metadata_df.loc[gc]
    lat = _get_meta_value(row, "latitude", "None")
    lon = _get_meta_value(row, "longitude", "None")
    if lat == "None" or lon == "None":
        return None, None
    return float(lat), float(lon)


def _ranked_phylo_candidates(language: str, pool_limit: int = 400) -> List[str]:
    ranked: List[str] = []
    seen = {language}

    direct = genetic_neighbours.get(language, [])
    if isinstance(direct, list):
        for nb in direct:
            nb = str(nb)
            if nb in seen:
                continue
            seen.add(nb)
            ranked.append(nb)
            if len(ranked) >= pool_limit:
                return ranked

    idx = 0
    while idx < len(ranked) and len(ranked) < pool_limit:
        cur = ranked[idx]
        idx += 1
        nxt_list = genetic_neighbours.get(cur, [])
        if not isinstance(nxt_list, list):
            continue
        for nxt in nxt_list:
            nxt = str(nxt)
            if nxt in seen:
                continue
            seen.add(nxt)
            ranked.append(nxt)
            if len(ranked) >= pool_limit:
                return ranked

    for gc in _all_language_codes():
        if gc in seen:
            continue
        seen.add(gc)
        ranked.append(gc)
        if len(ranked) >= pool_limit:
            break
    return ranked


def _ranked_geo_candidates(language: str, pool_limit: int = 1200) -> List[str]:
    seen = {language}
    ranked: List[str] = []
    lat0, lon0 = _meta_lat_lon(language)

    if lat0 is not None and lon0 is not None:
        scored: List[Tuple[float, str]] = []
        for gc in _all_language_codes():
            if gc == language:
                continue
            lat, lon = _meta_lat_lon(gc)
            if lat is None or lon is None:
                continue
            scored.append((_haversine_km(lat0, lon0, lat, lon), gc))
        scored.sort(key=lambda x: (x[0], x[1]))
        for _, gc in scored:
            if gc in seen:
                continue
            seen.add(gc)
            ranked.append(gc)
            if len(ranked) >= pool_limit:
                return ranked

    direct = geographic_neighbours.get(language, [])
    if isinstance(direct, list):
        for nb in direct:
            nb = str(nb)
            if nb in seen:
                continue
            seen.add(nb)
            ranked.append(nb)
            if len(ranked) >= pool_limit:
                return ranked

    for gc in _all_language_codes():
        if gc in seen:
            continue
        seen.add(gc)
        ranked.append(gc)
        if len(ranked) >= pool_limit:
            break
    return ranked


def _select_neighbors_with_feature_coverage(
    candidates: Sequence[str], correlated: Sequence[str], target_feature: str, k: int
) -> List[str]:
    if k <= 0:
        return []

    feature_targets = [f for f in correlated if typ_df is not None and f in typ_df.columns]
    if typ_df is not None and target_feature in typ_df.columns and target_feature not in feature_targets:
        feature_targets.append(target_feature)
    selected: List[str] = []
    selected_set: set[str] = set()
    covered: set[str] = set()

    for nb in candidates:
        if nb in selected_set:
            continue
        obs = _observed_correlated_set(nb, feature_targets)
        if not obs:
            continue
        if obs - covered:
            selected.append(nb)
            selected_set.add(nb)
            covered.update(obs)
            if len(selected) >= k and (not feature_targets or covered.issuperset(feature_targets)):
                return selected[:k]

    if len(selected) < k:
        for nb in candidates:
            if nb in selected_set:
                continue
            if _observed_correlated_set(nb, feature_targets):
                selected.append(nb)
                selected_set.add(nb)
                if len(selected) >= k:
                    return selected[:k]

    if len(selected) < k:
        for nb in candidates:
            if nb in selected_set:
                continue
            selected.append(nb)
            selected_set.add(nb)
            if len(selected) >= k:
                return selected[:k]

    return selected[:k]


def _collect_neighbor_facts(
    glottocode: str, target_feature: str, correlated: Sequence[str], limit: int = 4, max_correlated_fallback: int = 2
) -> List[Tuple[str, str]]:
    if typ_df is None or glottocode not in typ_df.index:
        return []
    facts: List[Tuple[str, str]] = []

    # Put direct target-feature evidence first when available.
    if target_feature in typ_df.columns:
        val = typ_df.at[glottocode, target_feature]
        if not _is_missing(val):
            facts.append((target_feature, _format_value(val)))
            if len(facts) >= limit:
                return facts

    fallback_used = 0
    for feat in correlated:
        if feat == target_feature or feat not in typ_df.columns:
            continue
        val = typ_df.at[glottocode, feat]
        if not _is_missing(val):
            facts.append((feat, _format_value(val)))
            fallback_used += 1
        if len(facts) >= limit:
            return facts
        if fallback_used >= max_correlated_fallback:
            return facts

    return facts


def _prioritize_neighbors_with_target_value(neighbors: Sequence[str], target_feature: str) -> List[str]:
    indexed = list(enumerate(neighbors))
    indexed.sort(
        key=lambda x: (0 if _lang_has_feature_value(str(x[1]), target_feature) else 1, x[0])
    )
    return [str(nb) for _, nb in indexed]


def _allowed_values(feature: str) -> List[str]:
    if typ_df is None or feature not in typ_df.columns:
        return ["0", "1"]
    col = typ_df[feature]
    vals = [v for v in col.unique().tolist() if not _is_missing(v)]
    if not vals:
        return ["0", "1"]
    if all(isinstance(v, float) and v.is_integer() for v in vals):
        vals = [int(v) for v in vals]
    return [str(v) for v in sorted(vals, key=lambda x: str(x))]


def _iter_missing_pairs(df: pd.DataFrame) -> Iterator[Tuple[str, str]]:
    for lang, row in df.iterrows():
        for feat, val in row.items():
            if _is_missing(val):
                yield str(lang), str(feat)


def set_prompt_options(prompt_version: str | None = None, include_vote_table: bool | None = None) -> None:
    global PROMPT_VERSION, INCLUDE_VOTE_TABLE
    if prompt_version is not None:
        PROMPT_VERSION = str(prompt_version)
    if include_vote_table is not None:
        INCLUDE_VOTE_TABLE = bool(include_vote_table)


def _count_votes(neighbors: Sequence[str], feature: str) -> Dict[str, int]:
    counts = {"yes": 0, "no": 0, "missing": 0}
    if typ_df is None or feature not in typ_df.columns:
        counts["missing"] = len(neighbors)
        return counts
    for nb in neighbors:
        if nb not in typ_df.index:
            counts["missing"] += 1
            continue
        val = typ_df.at[nb, feature]
        if _is_missing(val):
            counts["missing"] += 1
        elif int(float(val)) == 1:
            counts["yes"] += 1
        else:
            counts["no"] += 1
    return counts


def compute_neighbor_votes(
    language: str,
    feature: str,
    phylo_neighbors: Sequence[str] | None = None,
    geo_neighbors: Sequence[str] | None = None,
) -> Dict[str, Dict[str, float | int | str]]:
    if phylo_neighbors is None:
        phylo_k = _neighbor_k_for_language(genetic_neighbours, language)
        phylo_candidates = _ranked_phylo_candidates(language)
        correlated = _effective_correlated_features(language, feature, top_n_features)
        phylo_neighbors = _select_neighbors_with_feature_coverage(phylo_candidates, correlated, feature, phylo_k)
    if geo_neighbors is None:
        geo_k = _neighbor_k_for_language(geographic_neighbours, language)
        geo_candidates = _ranked_geo_candidates(language)
        correlated = _effective_correlated_features(language, feature, top_n_features)
        geo_neighbors = _select_neighbors_with_feature_coverage(geo_candidates, correlated, feature, geo_k)

    phylo_counts = _count_votes(phylo_neighbors, feature)
    geo_counts = _count_votes(geo_neighbors, feature)
    combined = {
        "yes": phylo_counts["yes"] + geo_counts["yes"],
        "no": phylo_counts["no"] + geo_counts["no"],
        "missing": phylo_counts["missing"] + geo_counts["missing"],
    }

    def _yes_ratio(c: Dict[str, int]) -> float:
        denom = c["yes"] + c["no"]
        return (c["yes"] / denom) if denom else 0.0

    def _majority(c: Dict[str, int]) -> str:
        if c["yes"] > c["no"]:
            return "yes"
        if c["no"] > c["yes"]:
            return "no"
        return "tie"

    def _agreement_ratio(c: Dict[str, int]) -> float:
        denom = c["yes"] + c["no"]
        return (max(c["yes"], c["no"]) / denom) if denom else 0.0

    return {
        "genetic": {
            **phylo_counts,
            "yes_ratio": _yes_ratio(phylo_counts),
            "majority": _majority(phylo_counts),
            "agreement_ratio": _agreement_ratio(phylo_counts),
        },
        "geographic": {
            **geo_counts,
            "yes_ratio": _yes_ratio(geo_counts),
            "majority": _majority(geo_counts),
            "agreement_ratio": _agreement_ratio(geo_counts),
        },
        "overall": {
            **combined,
            "yes_ratio": _yes_ratio(combined),
            "majority": _majority(combined),
            "agreement_ratio": _agreement_ratio(combined),
        },
    }


def _target_feature_value(glottocode: str, feature: str) -> str | None:
    if typ_df is None or glottocode not in typ_df.index or feature not in typ_df.columns:
        return None
    val = typ_df.at[glottocode, feature]
    if _is_missing(val):
        return None
    return _format_value(val)


def _feature_prevalence_prior(feature: str) -> tuple[str, float]:
    """
    Return majority value and its observed fraction for the target feature.
    Tie-break is deterministic by lexical value order.
    """
    if typ_df is None or feature not in typ_df.columns:
        return "0", 0.5
    counts: Dict[str, int] = {}
    total = 0
    for v in typ_df[feature].tolist():
        if _is_missing(v):
            continue
        key = _format_value(v)
        counts[key] = counts.get(key, 0) + 1
        total += 1
    if total == 0 or not counts:
        return "0", 0.5
    best_val, best_count = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return best_val, (best_count / total)


def construct_prompt(language: str, feature: str) -> Tuple[str, str]:
    """
    Construct the prompt for the given language and feature.
    :param language: str, Glottocode of the language to impute
    :param feature: str, the typological feature to impute
    :return: (str, str), the system and user prompts
    """
    system = SYSTEM_MESSAGE

    meta_row = metadata_df.loc[language] if language in metadata_df.index else None
    lang_name = _get_meta_value(meta_row, "name", language)
    iso = _get_meta_value(meta_row, "iso639_3", "None")
    lineage = _family_lineage(meta_row)
    macro = _get_meta_value(meta_row, "macroareas", "None")
    lat = _get_meta_value(meta_row, "latitude", "None")
    lon = _get_meta_value(meta_row, "longitude", "None")

    user_lines = []
    user_lines.append("Target language:")
    user_lines.append(f"- Name: {lang_name}")
    user_lines.append(f"- Glottocode: {language}")
    user_lines.append(f"- ISO639-3: {iso}")
    user_lines.append(f"- Family lineage: {lineage}")
    user_lines.append(f"- Macro-area: {macro}")
    user_lines.append(f"- Location: latitude={lat}, longitude={lon}")

    phylo_k = _neighbor_k_for_language(genetic_neighbours, language)
    phylo_candidates = _ranked_phylo_candidates(language)
    correlated = _effective_correlated_features(language, feature, top_n_features)
    phylo_neighbors = _select_neighbors_with_feature_coverage(phylo_candidates, correlated, feature, phylo_k)
    geo_k = _neighbor_k_for_language(geographic_neighbours, language)
    geo_candidates = _ranked_geo_candidates(language)
    geo_neighbors = _select_neighbors_with_feature_coverage(geo_candidates, correlated, feature, geo_k)
    phylo_neighbors = _prioritize_neighbors_with_target_value(phylo_neighbors, feature)
    geo_neighbors = _prioritize_neighbors_with_target_value(geo_neighbors, feature)

    votes = compute_neighbor_votes(language, feature, phylo_neighbors=phylo_neighbors, geo_neighbors=geo_neighbors)
    prior_value, prior_ratio = _feature_prevalence_prior(feature)

    if INCLUDE_VOTE_TABLE:
        g = votes["genetic"]
        geo = votes["geographic"]
        ov = votes["overall"]
        user_lines.append("Target-feature vote counts (primary evidence):")
        user_lines.append(
            f"- Genetic vote: {g['yes']} yes / {g['no']} no / {g['missing']} unk ({g['yes_ratio']:.0%} yes)"
        )
        user_lines.append(
            f"- Geo vote: {geo['yes']} yes / {geo['no']} no / {geo['missing']} unk ({geo['yes_ratio']:.0%} yes)"
        )
        user_lines.append(
            f"- Overall: {ov['yes']} yes / {ov['no']} no ({ov['yes_ratio']:.0%} yes; "
            f"majority={ov['majority']}; agreement={ov['agreement_ratio']:.0%})"
        )
        user_lines.append(f"- Feature prevalence prior: value={prior_value} ({prior_ratio:.0%} of observed)")

    user_lines.append("Observed typological facts (anchor features):")
    anchor_limit = 5
    observed_anchor_count = 0
    for feat in correlated:
        if observed_anchor_count >= anchor_limit:
            break
        if feat == feature:
            continue
        if typ_df is not None and feat not in typ_df.columns:
            continue
        val = typ_df.at[language, feat] if (typ_df is not None and language in typ_df.index) else None
        if _is_missing(val):
            continue
        user_lines.append(f"- {feat}: {_format_value(val)}")
        observed_anchor_count += 1
    if observed_anchor_count == 0:
        user_lines.append("- (no observed anchor facts)")
    elif len(correlated) > anchor_limit:
        user_lines.append("- (truncated to top observed anchors)")

    user_lines.append("Phylogenetic neighbors (top-k):")
    for i, nb in enumerate(phylo_neighbors, start=1):
        nb_meta = metadata_df.loc[nb] if nb in metadata_df.index else None
        nb_name = _get_meta_value(nb_meta, "name", nb)
        user_lines.append(f"{i}) {nb_name} (glottocode={nb}, dist={i}):")
        target_val = _target_feature_value(nb, feature)
        if target_val is None:
            user_lines.append("- target feature: unobserved")
        else:
            user_lines.append(f"- {feature}: {target_val}")

    user_lines.append("Geographic neighbors (top-k):")
    lat_val = None if lat == "None" else float(lat)
    lon_val = None if lon == "None" else float(lon)
    for i, nb in enumerate(geo_neighbors, start=1):
        nb_meta = metadata_df.loc[nb] if nb in metadata_df.index else None
        nb_name = _get_meta_value(nb_meta, "name", nb)
        nb_lat = _get_meta_value(nb_meta, "latitude", "None")
        nb_lon = _get_meta_value(nb_meta, "longitude", "None")
        km = "unknown"
        if lat_val is not None and lon_val is not None and nb_lat != "None" and nb_lon != "None":
            km = f"{_haversine_km(lat_val, lon_val, float(nb_lat), float(nb_lon)):.1f}"
        user_lines.append(f"{i}) {nb_name} (glottocode={nb}, km={km}):")
        target_val = _target_feature_value(nb, feature)
        if target_val is None:
            user_lines.append("- target feature: unobserved")
        else:
            user_lines.append(f"- {feature}: {target_val}")

    allowed = _allowed_values(feature)
    if PROMPT_VERSION == "v3_strict_json":
        user_lines.append("Prompt version: v3_strict_json")
        user_lines.append("Task:")
        user_lines.append("Predict the missing value for the following feature:")
        user_lines.append(f"- Feature: {feature}")
        user_lines.append(f"- Allowed values: {' | '.join(allowed)}")
        user_lines.append("Output format (STRICT JSON):")
        user_lines.append("Output ONLY valid JSON.")
        user_lines.append("Deterministic decision rule:")
        user_lines.append("1) Use overall vote majority on target-feature yes/no counts (ignore unknown).")
        user_lines.append("2) If tied or no overall evidence, use phylogenetic vote majority.")
        user_lines.append(f"3) If still tied or no evidence, use feature prevalence prior (choose {prior_value}).")
        user_lines.append("Return exactly one minified JSON object on one line with keys: value, confidence, rationale.")
        user_lines.append("- value: one of the allowed values above")
        user_lines.append("- confidence: low, medium, or high")
        user_lines.append("- rationale: at most 20 words")
        user_lines.append("No Markdown, no prose, no code fences, no trailing text.")
        user_lines.append("Example output:")
        user_lines.append(
            '{"value":"0","confidence":"high","rationale":"Closest genetic and geographic neighbors mostly support value 0."}'
        )
    else:
        user_lines.append("Task:")
        user_lines.append("Predict the missing value for the following feature:")
        user_lines.append(f"- Feature: {feature}")
        user_lines.append(f"- Allowed values: {' | '.join(allowed)}")
        user_lines.append("Output format (STRICT):")
        user_lines.append("Return exactly one allowed value.")
        user_lines.append("Do not provide explanations.")

    user = "\n".join(user_lines)
    return system, user


def run_llama(system: str, user: str, tokenizer, model, max_new_tokens: int = 8) -> str:
    """
    Run the LLaMA model with the given prompt.
    :param system: str, system message
    :param user: str, user message
    :param tokenizer: tokenizer
    :param model: model
    :param max_new_tokens: int, generation cap
    :return: str, model output
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"SYSTEM:\n{system}\nUSER:\n{user}\nASSISTANT:\n"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )
    gen = outputs[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    return text.strip()


def _load_impute_pairs(path: str) -> List[Tuple[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Impute list not found: {path}")
    data = json.loads(p.read_text())
    pairs: List[Tuple[str, str]] = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, str):
                pairs.append((k, v))
            elif isinstance(v, list):
                for feat in v:
                    pairs.append((k, feat))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                pairs.append((item[0], item[1]))
            elif isinstance(item, dict):
                lang = item.get("language") or item.get("glottocode")
                feat = item.get("feature")
                if lang and feat:
                    pairs.append((lang, feat))
    else:
        raise ValueError("Unsupported impute list format")
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(description="Prompt builder for typological feature imputation.")
    p.add_argument("--mode", type=str, choices=["prompts", "predict"], default="prompts")
    p.add_argument("--typ", type=str, required=True, help="Typological CSV (glottocode index).")
    p.add_argument("--meta", type=str, required=True, help="Metadata CSV (glottocode index).")
    p.add_argument("--topk_csv", type=str, required=True, help="CSV from select_topk_features.py.")
    p.add_argument("--gen", type=str, required=True, help="Genetic neighbours JSON.")
    p.add_argument("--geo", type=str, required=True, help="Geographic neighbours JSON.")
    p.add_argument("--top_n", type=int, default=10, help="Top-N correlated features to include.")
    p.add_argument("--impute", type=str, default=None, help="Optional JSON list/dict of (language, feature) pairs.")
    p.add_argument("--out", type=str, required=True, help="Output file. JSONL for prompts mode, JSON for predict mode.")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--prompt_version", type=str, default="v3_strict_json")
    p.add_argument(
        "--include_vote_table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include compact genetic/geo vote summary table for target feature.",
    )
    args = p.parse_args()

    global typ_df, metadata_df, genetic_neighbours, geographic_neighbours, top_n_features, topk_map

    typ_df = pd.read_csv(args.typ, index_col=0)
    typ_df.index = typ_df.index.astype(str)
    metadata_df = pd.read_csv(args.meta, index_col=0)
    metadata_df.index = metadata_df.index.astype(str)
    topk_df = pd.read_csv(args.topk_csv)

    with open(args.gen, "r", encoding="utf-8") as f:
        genetic_neighbours = json.load(f)
    with open(args.geo, "r", encoding="utf-8") as f:
        geographic_neighbours = json.load(f)
    genetic_neighbours = {str(k): v for k, v in genetic_neighbours.items()}
    geographic_neighbours = {str(k): v for k, v in geographic_neighbours.items()}

    top_n_features = args.top_n
    tmp: Dict[str, List[Tuple[int, str]]] = {}
    for _, row in topk_df.iterrows():
        feat = str(row["feature"])
        other = str(row["other_feature"])
        rank = int(row["rank"]) if "rank" in row and not pd.isna(row["rank"]) else 0
        tmp.setdefault(feat, []).append((rank, other))
    topk_map = {}
    for feat, items in tmp.items():
        items.sort(key=lambda x: x[0])
        topk_map[feat] = [other for _, other in items]

    set_prompt_options(args.prompt_version, args.include_vote_table)

    if args.impute:
        impute_pairs: Iterable[Tuple[str, str]] = _load_impute_pairs(args.impute)
    else:
        impute_pairs = _iter_missing_pairs(typ_df)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "prompts":
        with open(args.out, "w", encoding="utf-8") as f:
            for lang, feat in impute_pairs:
                system, user = construct_prompt(lang, feat)
                rec = {
                    "language": lang,
                    "feature": feat,
                    "system": system,
                    "user": user,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)

        results = {}
        for lang, feat in impute_pairs:
            system, user = construct_prompt(lang, feat)
            pred = run_llama(system, user, tokenizer, model, max_new_tokens=args.max_new_tokens)
            results.setdefault(lang, {})[feat] = pred

        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
