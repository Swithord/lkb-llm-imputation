from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_BASE = _load_module("glottolog_tree_base_prompting", _ROOT / "code" / "prompting.py")
_TREE = _load_module("glottolog_tree_phylo_tree", _HERE / "phylo_tree.py")
_KG_LOADER = _load_module("glottolog_tree_kg_loader", _HERE / "kg_loader.py")
_KG_RETRIEVAL = _load_module("glottolog_tree_kg_retrieval", _HERE / "kg_retrieval.py")
_BASE_RANKED_GEO_CANDIDATES = _BASE._ranked_geo_candidates


SYSTEM_MESSAGE = _BASE.SYSTEM_MESSAGE

typ_df: Optional[pd.DataFrame] = None
metadata_df: pd.DataFrame
genetic_neighbours: dict = {}
genetic_neighbour_details: dict = {}
geographic_neighbours: dict = {}
top_n_features: int = 10
topk_map: Dict[str, List[str]] = {}
PROMPT_VERSION: str = "v5_glottolog_tree_json"
INCLUDE_VOTE_TABLE: bool = True
clue_support_cache: Dict[Tuple[str, str, str], Tuple[int, int]] = {}
RETRIEVAL_BACKEND: str = "legacy"
KG_NODES_PATH: Optional[str] = None
KG_EDGES_PATH: Optional[str] = None
_KG_GRAPH = None


def _selection_variant() -> str:
    if PROMPT_VERSION in {"v4_strict_json", "v5_glottolog_tree_json", "v5_glottolog_tree_compact_json"}:
        return "v4"
    return "v3"


def _is_compact_prompt() -> bool:
    return PROMPT_VERSION == "v5_glottolog_tree_compact_json"


def _sync_base_globals() -> None:
    _BASE.typ_df = typ_df
    _BASE.metadata_df = metadata_df
    _BASE.genetic_neighbours = genetic_neighbours
    _BASE.geographic_neighbours = geographic_neighbours
    _BASE.top_n_features = top_n_features
    _BASE.topk_map = topk_map
    _BASE.PROMPT_VERSION = PROMPT_VERSION
    _BASE.INCLUDE_VOTE_TABLE = INCLUDE_VOTE_TABLE
    _BASE.clue_support_cache = clue_support_cache
    _BASE._selection_variant = _selection_variant
    _BASE._ranked_phylo_candidates = _ranked_phylo_candidates
    _BASE._ranked_geo_candidates = _ranked_geo_candidates


def set_retrieval_options(
    retrieval_backend: str | None = None,
    kg_nodes_path: str | None = None,
    kg_edges_path: str | None = None,
) -> None:
    global RETRIEVAL_BACKEND, KG_NODES_PATH, KG_EDGES_PATH, _KG_GRAPH
    if retrieval_backend is not None:
        RETRIEVAL_BACKEND = str(retrieval_backend)
    if kg_nodes_path is not None:
        KG_NODES_PATH = str(kg_nodes_path)
    if kg_edges_path is not None:
        KG_EDGES_PATH = str(kg_edges_path)
    _KG_GRAPH = None


def _ensure_kg_graph():
    global _KG_GRAPH
    if _KG_GRAPH is not None:
        return _KG_GRAPH
    if RETRIEVAL_BACKEND not in {"kg_flat", "kg_typed", "kg_typed_contrastive", "hybrid_flat_kg"}:
        return None
    if not KG_NODES_PATH or not KG_EDGES_PATH:
        raise RuntimeError("KG retrieval requires both KG_NODES_PATH and KG_EDGES_PATH.")
    _KG_GRAPH = _KG_LOADER.load_kg(KG_NODES_PATH, KG_EDGES_PATH)
    return _KG_GRAPH


def _coerce_non_negative_int(value, default: int) -> int:
    try:
        out = int(value)
    except Exception:
        return default
    return out if out >= 0 else default


def _normalized_phylo_record(raw, fallback_rank: int) -> Optional[Dict[str, object]]:
    if isinstance(raw, dict):
        glottocode = raw.get("glottocode") or raw.get("language") or raw.get("id")
        if glottocode is None:
            return None
        return {
            "glottocode": str(glottocode),
            "tree_distance": _coerce_non_negative_int(raw.get("tree_distance"), fallback_rank),
            "shared_ancestor_depth": raw.get("shared_ancestor_depth"),
            "relation_type": str(raw.get("relation_type") or "phylogenetic_neighbor"),
        }

    if raw is None:
        return None
    return {
        "glottocode": str(raw),
        "tree_distance": fallback_rank,
        "shared_ancestor_depth": None,
        "relation_type": "phylogenetic_neighbor",
    }


def _legacy_ranked_phylo_records(language: str, pool_limit: int = 400) -> List[Dict[str, object]]:
    ranked: List[Dict[str, object]] = []
    seen = {language}

    direct_details = genetic_neighbour_details.get(language, []) if isinstance(genetic_neighbour_details, dict) else []
    if isinstance(direct_details, list):
        for idx, raw in enumerate(direct_details, start=1):
            rec = _normalized_phylo_record(raw, idx)
            if rec is None:
                continue
            nb = str(rec["glottocode"])
            if nb in seen:
                continue
            seen.add(nb)
            ranked.append(rec)
            if len(ranked) >= pool_limit:
                return ranked

    direct = genetic_neighbours.get(language, [])
    if isinstance(direct, list):
        for idx, nb in enumerate(direct, start=len(ranked) + 1):
            rec = _normalized_phylo_record(nb, idx)
            if rec is None:
                continue
            nb_code = str(rec["glottocode"])
            if nb_code in seen:
                continue
            seen.add(nb_code)
            ranked.append(rec)
            if len(ranked) >= pool_limit:
                return ranked

    idx = 0
    while idx < len(ranked) and len(ranked) < pool_limit:
        cur = str(ranked[idx]["glottocode"])
        idx += 1
        nxt_list = genetic_neighbours.get(cur, [])
        if not isinstance(nxt_list, list):
            continue
        for nxt_rank, nxt in enumerate(nxt_list, start=1):
            rec = _normalized_phylo_record(nxt, len(ranked) + nxt_rank)
            if rec is None:
                continue
            nxt_code = str(rec["glottocode"])
            if nxt_code in seen:
                continue
            seen.add(nxt_code)
            ranked.append(rec)
            if len(ranked) >= pool_limit:
                return ranked

    for gc in _BASE._all_language_codes():
        if gc in seen:
            continue
        seen.add(gc)
        ranked.append(
            {
                "glottocode": gc,
                "tree_distance": len(ranked) + 1,
                "shared_ancestor_depth": None,
                "relation_type": "phylogenetic_fallback",
            }
        )
        if len(ranked) >= pool_limit:
            break
    return ranked


def _ranked_phylo_records(language: str, pool_limit: int = 400) -> List[Dict[str, object]]:
    if RETRIEVAL_BACKEND == "kg_flat":
        graph = _ensure_kg_graph()
        return _KG_RETRIEVAL.ranked_phylo_records(graph, language, pool_limit=pool_limit)
    return _legacy_ranked_phylo_records(language, pool_limit=pool_limit)


def _ranked_phylo_candidates(language: str, pool_limit: int = 400) -> List[str]:
    return [str(rec["glottocode"]) for rec in _ranked_phylo_records(language, pool_limit=pool_limit)]


def _legacy_ranked_geo_candidates(language: str, pool_limit: int = 1200) -> List[str]:
    return _BASE_RANKED_GEO_CANDIDATES(language, pool_limit=pool_limit)


def _ranked_geo_candidates(language: str, pool_limit: int = 1200) -> List[str]:
    if RETRIEVAL_BACKEND == "kg_flat":
        graph = _ensure_kg_graph()
        return _KG_RETRIEVAL.ranked_geo_candidates(graph, language, pool_limit=pool_limit)
    return _legacy_ranked_geo_candidates(language, pool_limit=pool_limit)


def _merge_record_lists(
    primary: Sequence[Dict[str, object]],
    supplemental: Sequence[Dict[str, object]],
    pool_limit: int = 400,
    preserve_prefix: int = 2,
) -> List[Dict[str, object]]:
    merged: List[Dict[str, object]] = []
    seen: set[str] = set()

    def add_record(rec: Dict[str, object]) -> None:
        glottocode = str(rec.get("glottocode", "")).strip()
        if not glottocode or glottocode in seen or len(merged) >= pool_limit:
            return
        seen.add(glottocode)
        merged.append(rec)

    for rec in list(primary)[:preserve_prefix]:
        add_record(rec)
    for rec in supplemental:
        add_record(rec)
    for rec in primary:
        add_record(rec)
    return merged


def _merge_candidate_lists(
    primary: Sequence[str],
    supplemental: Sequence[str],
    pool_limit: int = 1200,
    preserve_prefix: int = 2,
) -> List[str]:
    merged: List[str] = []
    seen: set[str] = set()

    def add_candidate(candidate: str) -> None:
        code = str(candidate)
        if not code or code in seen or len(merged) >= pool_limit:
            return
        seen.add(code)
        merged.append(code)

    for candidate in list(primary)[:preserve_prefix]:
        add_candidate(candidate)
    for candidate in supplemental:
        add_candidate(candidate)
    for candidate in primary:
        add_candidate(candidate)
    return merged


def _feature_conditioned_phylo_records(language: str, feature: str, correlated: Sequence[str], pool_limit: int = 400) -> List[Dict[str, object]]:
    if RETRIEVAL_BACKEND in {"kg_typed", "kg_typed_contrastive"}:
        graph = _ensure_kg_graph()
        return _KG_RETRIEVAL.ranked_phylo_records_typed(
            graph,
            language=language,
            target_feature=feature,
            correlated=correlated,
            pool_limit=pool_limit,
        )
    if RETRIEVAL_BACKEND == "hybrid_flat_kg":
        graph = _ensure_kg_graph()
        legacy_records = _legacy_ranked_phylo_records(language, pool_limit=pool_limit)
        typed_records = _KG_RETRIEVAL.ranked_phylo_records_typed(
            graph,
            language=language,
            target_feature=feature,
            correlated=correlated,
            pool_limit=min(pool_limit, 32),
        )
        return _merge_record_lists(legacy_records, typed_records, pool_limit=pool_limit, preserve_prefix=2)
    return _ranked_phylo_records(language, pool_limit=pool_limit)


def _feature_conditioned_geo_candidates(language: str, feature: str, correlated: Sequence[str], pool_limit: int = 1200) -> List[str]:
    if RETRIEVAL_BACKEND in {"kg_typed", "kg_typed_contrastive"}:
        graph = _ensure_kg_graph()
        return _KG_RETRIEVAL.ranked_geo_candidates_typed(
            graph,
            language=language,
            target_feature=feature,
            correlated=correlated,
            pool_limit=pool_limit,
        )
    if RETRIEVAL_BACKEND == "hybrid_flat_kg":
        graph = _ensure_kg_graph()
        legacy_candidates = _legacy_ranked_geo_candidates(language, pool_limit=pool_limit)
        typed_candidates = _KG_RETRIEVAL.ranked_geo_candidates_typed(
            graph,
            language=language,
            target_feature=feature,
            correlated=correlated,
            pool_limit=min(pool_limit, 32),
        )
        return _merge_candidate_lists(legacy_candidates, typed_candidates, pool_limit=pool_limit, preserve_prefix=2)
    return _ranked_geo_candidates(language, pool_limit=pool_limit)


def _relation_label(value: object) -> str:
    if value is None:
        return "unknown"
    return str(value).replace("_", " ")


def _compact_relation_label(value: object) -> str:
    relation = str(value or "unknown")
    mapping = {
        "same_immediate_branch": "same branch",
        "sibling_branch": "sibling branch",
        "nearby_cousin_branch": "cousin branch",
        "higher_shared_ancestor": "higher ancestor",
        "phylogenetic_neighbor": "nearby branch",
        "phylogenetic_fallback": "fallback",
    }
    return mapping.get(relation, relation.replace("_", " "))


def _format_shared_ancestor_depth(value: object) -> str:
    if value is None or value == "":
        return "unknown"
    try:
        return str(int(value))
    except Exception:
        return str(value)


def _format_phylo_neighbor_block(
    neighbors: Sequence[str],
    phylo_record_map: Dict[str, Dict[str, object]],
    target_feature: str,
    correlated: Sequence[str],
    compact: bool = False,
) -> List[str]:
    lines: List[str] = []
    for idx, nb in enumerate(neighbors, start=1):
        label = _BASE._language_label(nb)
        rec = phylo_record_map.get(str(nb), {})
        tree_distance = _coerce_non_negative_int(rec.get("tree_distance"), idx)
        if compact:
            relation_type = _compact_relation_label(rec.get("relation_type"))
            lines.append(f"{idx}) {label} (relation={relation_type}, tree_distance={tree_distance}):")
        else:
            relation_type = _relation_label(rec.get("relation_type"))
            shared_depth = _format_shared_ancestor_depth(rec.get("shared_ancestor_depth"))
            lines.append(
                f"{idx}) {label} (glottocode={nb}, relation={relation_type}, tree_distance={tree_distance}, shared_ancestor_depth={shared_depth}):"
            )
        facts = _BASE._collect_neighbor_facts(nb, target_feature, correlated, limit=4, max_correlated_fallback=3)
        if facts:
            for feat, value in facts:
                lines.append(f"- {feat}: {value}")
        else:
            lines.append("- No observed target or anchor facts available.")
    return lines


def _contrastive_force_include(candidates: Sequence[str], feature: str) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()
    for desired_value in ("1", "0"):
        candidate = _BASE._nearest_candidate_with_value(candidates, feature, desired_value)
        if candidate is None or candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


def _ordered_unique(values: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value)
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _nearest_phylo_supporting_neighbor(
    candidates: Sequence[str],
    phylo_record_map: Dict[str, Dict[str, object]],
    feature: str,
    desired_value: str,
) -> str | None:
    for idx, nb in enumerate(candidates, start=1):
        if _BASE._target_feature_value(nb, feature) != desired_value:
            continue
        rec = phylo_record_map.get(str(nb), {})
        tree_distance = _coerce_non_negative_int(rec.get("tree_distance"), idx)
        relation_type = _relation_label(rec.get("relation_type"))
        return f"{_BASE._language_label(nb)} (relation={relation_type}, tree_distance={tree_distance})"
    return None


def set_prompt_options(prompt_version: str | None = None, include_vote_table: bool | None = None) -> None:
    global PROMPT_VERSION, INCLUDE_VOTE_TABLE
    if prompt_version is not None:
        PROMPT_VERSION = str(prompt_version)
    if include_vote_table is not None:
        INCLUDE_VOTE_TABLE = bool(include_vote_table)


def compute_neighbor_votes(
    language: str,
    feature: str,
    phylo_neighbors: Sequence[str] | None = None,
    geo_neighbors: Sequence[str] | None = None,
) -> Dict[str, Dict[str, float | int | str]]:
    _sync_base_globals()
    return _BASE.compute_neighbor_votes(
        language,
        feature,
        phylo_neighbors=phylo_neighbors,
        geo_neighbors=geo_neighbors,
    )


def construct_prompt(language: str, feature: str) -> Tuple[str, str]:
    _sync_base_globals()
    system = SYSTEM_MESSAGE

    meta_row = metadata_df.loc[language] if language in metadata_df.index else None
    lang_name = _BASE._get_meta_value(meta_row, "name", language)
    iso = _BASE._get_meta_value(meta_row, "iso639_3", "None")
    lineage = _BASE._family_lineage(meta_row)
    macro = _BASE._get_meta_value(meta_row, "macroareas", "None")
    lat = _BASE._get_meta_value(meta_row, "latitude", "None")
    lon = _BASE._get_meta_value(meta_row, "longitude", "None")

    user_lines = [
        "Target language:",
        f"- Name: {lang_name}",
        f"- Glottocode: {language}",
        f"- ISO639-3: {iso}",
        f"- Family lineage: {lineage}",
        f"- Macro-area: {macro}",
        f"- Location: latitude={lat}, longitude={lon}",
    ]

    correlated = _BASE._effective_correlated_features(language, feature, top_n_features, fallback_n=15)
    phylo_k = _BASE._neighbor_k_for_language(genetic_neighbours, language)
    legacy_phylo_candidates = []
    if RETRIEVAL_BACKEND == "hybrid_flat_kg":
        legacy_phylo_candidates = [str(rec["glottocode"]) for rec in _legacy_ranked_phylo_records(language)]
    phylo_records = _feature_conditioned_phylo_records(language, feature, correlated)
    phylo_candidates = [str(rec["glottocode"]) for rec in phylo_records]
    phylo_record_map = {str(rec["glottocode"]): rec for rec in phylo_records}
    if RETRIEVAL_BACKEND in {"kg_typed_contrastive", "hybrid_flat_kg"}:
        contrastive_force = _contrastive_force_include(phylo_candidates, feature)
        if RETRIEVAL_BACKEND == "hybrid_flat_kg":
            phylo_force = _ordered_unique(list(legacy_phylo_candidates[: min(2, phylo_k)]) + contrastive_force)
        else:
            phylo_force = contrastive_force
    else:
        phylo_yes_nb = _BASE._nearest_candidate_with_value(phylo_candidates, feature, "1")
        phylo_force = [phylo_yes_nb] if phylo_yes_nb is not None else []
    phylo_neighbors = _BASE._select_neighbors_with_feature_coverage(
        phylo_candidates,
        correlated,
        feature,
        phylo_k,
        force_include=phylo_force,
        reference_language=language,
        selection_variant=_selection_variant(),
    )

    geo_k = _BASE._neighbor_k_for_language(geographic_neighbours, language)
    legacy_geo_candidates = []
    if RETRIEVAL_BACKEND == "hybrid_flat_kg":
        legacy_geo_candidates = _legacy_ranked_geo_candidates(language)
    geo_candidates = _feature_conditioned_geo_candidates(language, feature, correlated)
    if RETRIEVAL_BACKEND in {"kg_typed_contrastive", "hybrid_flat_kg"}:
        contrastive_geo_force = _contrastive_force_include(geo_candidates, feature)
        if RETRIEVAL_BACKEND == "hybrid_flat_kg":
            geo_force = _ordered_unique(list(legacy_geo_candidates[: min(1, geo_k)]) + contrastive_geo_force)
        else:
            geo_force = contrastive_geo_force
    else:
        geo_yes_nb = _BASE._nearest_candidate_with_value(geo_candidates, feature, "1")
        geo_force = [geo_yes_nb] if geo_yes_nb is not None else []
    geo_neighbors = _BASE._select_neighbors_with_feature_coverage(
        geo_candidates,
        correlated,
        feature,
        geo_k,
        force_include=geo_force,
        reference_language=language,
        selection_variant=_selection_variant(),
    )

    votes = compute_neighbor_votes(language, feature, phylo_neighbors=phylo_neighbors, geo_neighbors=geo_neighbors)
    clues, clue_summary = _BASE._collect_target_correlated_clues(language, feature, top_m=2, min_support=12)
    prior_value, prior_ratio = _BASE._feature_prevalence_prior(feature)
    anchors = _BASE._observed_anchor_facts(language, feature, max_items=5)
    phylo_yes = _nearest_phylo_supporting_neighbor(phylo_candidates, phylo_record_map, feature, "1")
    phylo_no = _nearest_phylo_supporting_neighbor(phylo_candidates, phylo_record_map, feature, "0")
    geo_yes = _BASE._nearest_supporting_neighbor(language, geo_candidates, feature, "1", mode="geographic")
    geo_no = _BASE._nearest_supporting_neighbor(language, geo_candidates, feature, "0", mode="geographic")

    user_lines.append("Observed typological facts (anchor features):")
    if anchors:
        for feat_name, feat_value in anchors:
            user_lines.append(f"- {feat_name}: {feat_value} (observed)")
    else:
        user_lines.append("- (no observed anchor facts)")

    allowed = _BASE._allowed_values(feature)
    if PROMPT_VERSION in {"v5_glottolog_tree_json", "v5_glottolog_tree_compact_json"}:
        user_lines.append(
            "Glottolog-tree retrieved evidence (compact evidence):"
            if _is_compact_prompt()
            else "Glottolog-tree retrieved evidence (detailed evidence):"
        )
        user_lines.extend(
            _format_phylo_neighbor_block(
                phylo_neighbors,
                phylo_record_map,
                feature,
                correlated,
                compact=_is_compact_prompt(),
            )
        )
        user_lines.append("Selected geographic neighbors (detailed evidence):")
        user_lines.extend(
            _BASE._format_neighbor_block(
                language,
                geo_neighbors,
                geo_candidates,
                feature,
                correlated,
                mode="geographic",
            )
        )

        if INCLUDE_VOTE_TABLE:
            g = votes["genetic"]
            geo = votes["geographic"]
            ov = votes["overall"]
            user_lines.append("Target-feature vote counts (useful but not decisive):")
            user_lines.append(
                f"- Genetic vote: {g['yes']} yes / {g['no']} no / {g['missing']} unk ({g['yes_ratio']:.0%} yes)"
            )
            user_lines.append(
                f"- Geo vote: {geo['yes']} yes / {geo['no']} no / {geo['missing']} unk ({geo['yes_ratio']:.0%} yes)"
            )
            user_lines.append(
                f"- Overall observed votes: {ov['yes']} yes / {ov['no']} no "
                f"({ov['yes_ratio']:.0%} yes; agreement={ov['agreement_ratio']:.0%})"
            )
            user_lines.append(
                f"- Vote evidence coverage: {ov['yes'] + ov['no']} observed target-feature votes "
                f"(unknown ignored in decision)."
            )
            user_lines.append(
                f"- Weak prevalence prior (tie-breaker only): value={prior_value} ({prior_ratio:.0%} of observed)"
            )

        user_lines.append("Nearest contrastive neighbor evidence:")
        user_lines.append(f"- Closest phylogenetic support for 1: {phylo_yes or 'none observed'}")
        user_lines.append(f"- Closest phylogenetic support for 0: {phylo_no or 'none observed'}")
        user_lines.append(f"- Closest geographic support for 1: {geo_yes or 'none observed'}")
        user_lines.append(f"- Closest geographic support for 0: {geo_no or 'none observed'}")

        user_lines.append("Target-specific correlated clues (compact):")
        if clues:
            for idx, clue in enumerate(clues, start=1):
                user_lines.append(
                    f"{idx}) {clue['feature']}={clue['value']} -> target support {clue['yes']} yes / {clue['no']} no"
                )
            user_lines.append(
                f"- Correlated clues leaning: {clue_summary['yes']} yes / {clue_summary['no']} no / {clue_summary['tie']} tie"
            )
        else:
            user_lines.append("- No reliable correlated clues with enough support.")

        user_lines.append(f"Prompt version: {PROMPT_VERSION}")
        user_lines.append("Task:")
        user_lines.append("Predict the missing value for the following feature:")
        user_lines.append(f"- Feature: {feature}")
        user_lines.append(f"- Allowed values: {' | '.join(allowed)}")
        user_lines.append("Reasoning guidance:")
        user_lines.append("- Compare the support for value 0 versus value 1.")
        user_lines.append("- Weigh observed anchor features, Glottolog-tree relations, geographic evidence, and correlated clues together.")
        user_lines.append("- Prefer closer genealogical evidence from the same branch or sibling branches before higher shared ancestors.")
        user_lines.append("- Neighbor counts are useful, but do not follow majority vote blindly.")
        user_lines.append("- A smaller number of closer or more relevant neighbors may outweigh a larger but weaker group.")
        user_lines.append("- It is acceptable to predict a minority value if it is better supported by the overall evidence.")
        user_lines.append(f"- Use feature prevalence only as a weak tie-breaker when evidence is otherwise balanced (prior={prior_value}).")
        user_lines.append("Output format (STRICT JSON):")
        user_lines.append("Output ONLY valid JSON.")
        user_lines.append("Return exactly one minified JSON object on one line with keys: value, confidence, rationale.")
        user_lines.append("- value: one of the allowed values above")
        user_lines.append("- confidence: low, medium, or high")
        user_lines.append("- rationale: at most 20 words")
        user_lines.append("No Markdown, no prose, no code fences, no trailing text.")
        user_lines.append("Few-shot examples:")
        user_lines.append(
            '{"value":"1","confidence":"medium","rationale":"Most distant languages favor 0, but the closest branch evidence supports 1."}'
        )
        user_lines.append(
            '{"value":"1","confidence":"high","rationale":"Closest branch evidence and observed features align strongly with value 1."}'
        )
        user_lines.append(
            '{"value":"0","confidence":"low","rationale":"Evidence is balanced, so the weak prevalence prior favors 0."}'
        )
    else:
        _BASE.set_prompt_options(PROMPT_VERSION, INCLUDE_VOTE_TABLE)
        return _BASE.construct_prompt(language, feature)

    return system, "\n".join(user_lines)


def run_llama(system: str, user: str, tokenizer, model, max_new_tokens: int = 8) -> str:
    return _BASE.run_llama(system, user, tokenizer, model, max_new_tokens=max_new_tokens)


def main() -> None:
    p = argparse.ArgumentParser(description="Glottolog-tree prompt builder for typological feature imputation.")
    p.add_argument("--mode", type=str, choices=["prompts", "predict"], default="prompts")
    p.add_argument("--typ", type=str, required=True, help="Typological CSV (glottocode index).")
    p.add_argument("--meta", type=str, required=True, help="Metadata CSV (glottocode index).")
    p.add_argument("--topk_csv", type=str, required=True, help="CSV from select_topk_features.py.")
    p.add_argument("--gen", type=str, required=True, help="Genetic neighbours JSON.")
    p.add_argument("--gen_detail", type=str, default="output/genetic_neighbours_detailed.json")
    p.add_argument("--geo", type=str, required=True, help="Geographic neighbours JSON.")
    p.add_argument(
        "--retrieval_backend",
        type=str,
        default="legacy",
        choices=["legacy", "kg_flat", "kg_typed", "kg_typed_contrastive", "hybrid_flat_kg"],
    )
    p.add_argument("--kg_nodes", type=str, default=None, help="Optional KG nodes JSONL path for KG-backed retrieval.")
    p.add_argument("--kg_edges", type=str, default=None, help="Optional KG edges JSONL path for KG-backed retrieval.")
    p.add_argument("--top_n", type=int, default=10, help="Top-N correlated features to include.")
    p.add_argument("--impute", type=str, default=None, help="Optional JSON list/dict of (language, feature) pairs.")
    p.add_argument("--out", type=str, required=True, help="Output file. JSONL for prompts mode, JSON for predict mode.")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--prompt_version", type=str, default="v5_glottolog_tree_json")
    p.add_argument(
        "--include_vote_table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include compact genetic/geo vote summary table for target feature.",
    )
    args = p.parse_args()

    global typ_df, metadata_df, genetic_neighbours, genetic_neighbour_details, geographic_neighbours, top_n_features, topk_map, clue_support_cache

    typ_df = pd.read_csv(args.typ, index_col=0)
    typ_df.index = typ_df.index.astype(str)
    metadata_df = pd.read_csv(args.meta, index_col=0)
    metadata_df.index = metadata_df.index.astype(str)
    topk_df = pd.read_csv(args.topk_csv)

    with open(args.gen, "r", encoding="utf-8") as f:
        genetic_neighbours = {str(k): v for k, v in json.load(f).items()}
    gen_detail_path = Path(args.gen_detail)
    if gen_detail_path.exists():
        with gen_detail_path.open("r", encoding="utf-8") as f:
            genetic_neighbour_details = {str(k): v for k, v in json.load(f).items()}
    else:
        genetic_neighbour_details = {}
    with open(args.geo, "r", encoding="utf-8") as f:
        geographic_neighbours = {str(k): v for k, v in json.load(f).items()}

    top_n_features = args.top_n
    clue_support_cache = {}
    tmp: Dict[str, List[Tuple[int, str]]] = {}
    for _, row in topk_df.iterrows():
        feat = str(row["feature"])
        other = str(row["other_feature"])
        rank = int(row["rank"]) if "rank" in row and not pd.isna(row["rank"]) else 0
        tmp.setdefault(feat, []).append((rank, other))
    topk_map = {feat: [other for _, other in sorted(items, key=lambda x: x[0])] for feat, items in tmp.items()}

    set_prompt_options(args.prompt_version, args.include_vote_table)
    set_retrieval_options(args.retrieval_backend, args.kg_nodes, args.kg_edges)
    _sync_base_globals()

    if args.impute:
        impute_pairs: Iterable[Tuple[str, str]] = _BASE._load_impute_pairs(args.impute)
    else:
        impute_pairs = _BASE._iter_missing_pairs(typ_df)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "prompts":
        with open(args.out, "w", encoding="utf-8") as f:
            for lang, feat in impute_pairs:
                system, user = construct_prompt(lang, feat)
                f.write(json.dumps({"language": lang, "feature": feat, "system": system, "user": user}, ensure_ascii=False) + "\n")
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
