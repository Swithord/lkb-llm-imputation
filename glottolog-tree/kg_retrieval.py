from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def _coerce_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_non_negative_int(value, default: int) -> int:
    try:
        out = int(value)
    except Exception:
        return default
    return out if out >= 0 else default


def _relation_type(edge: dict) -> str:
    relation = edge.get("relation_type")
    if relation:
        return str(relation)
    return "phylogenetic_neighbor"


def _tree_distance(edge: dict, default: int) -> int:
    return _coerce_non_negative_int(edge.get("tree_distance"), default)


def _shared_ancestor_depth(edge: dict):
    value = edge.get("shared_ancestor_depth")
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return value


def _edge_rank_key(edge: dict) -> tuple[int, int, str]:
    source_kind = str(edge.get("source_kind") or "")
    priority = 0 if source_kind == "detail" else 1
    rank = _coerce_non_negative_int(edge.get("rank"), 10**9)
    target = str(edge.get("target", ""))
    return (priority, rank, target)


def _flat_edge_rank_key(edge: dict) -> tuple[int, str]:
    rank = _coerce_non_negative_int(edge.get("rank"), 10**9)
    target = str(edge.get("target", ""))
    return (rank, target)


def ranked_phylo_records(graph, language: str, pool_limit: int = 400) -> List[Dict[str, object]]:
    lang_node = graph.language_node(language)
    if lang_node is None:
        return []

    lang_id = str(lang_node["id"])
    ranked: List[Dict[str, object]] = []
    seen = {str(language)}

    direct_edges = sorted(graph.outgoing(lang_id, "PHYLO_NEAR"), key=_edge_rank_key)
    for edge in direct_edges:
        target_id = str(edge.get("target", ""))
        target_node = graph.node(target_id)
        if not target_node or target_node.get("type") != "Language":
            continue
        nb = str(target_node.get("glottocode"))
        if not nb or nb in seen:
            continue
        seen.add(nb)
        ranked.append(
            {
                "glottocode": nb,
                "tree_distance": _tree_distance(edge, len(ranked) + 1),
                "shared_ancestor_depth": _shared_ancestor_depth(edge),
                "relation_type": _relation_type(edge),
            }
        )
        if len(ranked) >= pool_limit:
            return ranked

    idx = 0
    while idx < len(ranked) and len(ranked) < pool_limit:
        cur = str(ranked[idx]["glottocode"])
        idx += 1
        cur_node = graph.language_node(cur)
        if cur_node is None:
            continue
        nxt_edges = [
            edge
            for edge in graph.outgoing(str(cur_node["id"]), "PHYLO_NEAR")
            if str(edge.get("source_kind") or "flat") == "flat"
        ]
        for nxt_rank, edge in enumerate(sorted(nxt_edges, key=_flat_edge_rank_key), start=1):
            target_id = str(edge.get("target", ""))
            target_node = graph.node(target_id)
            if not target_node or target_node.get("type") != "Language":
                continue
            nb = str(target_node.get("glottocode"))
            if not nb or nb in seen:
                continue
            seen.add(nb)
            ranked.append(
                {
                    "glottocode": nb,
                    "tree_distance": _tree_distance(edge, len(ranked) + nxt_rank),
                    "shared_ancestor_depth": _shared_ancestor_depth(edge),
                    "relation_type": _relation_type(edge),
                }
            )
            if len(ranked) >= pool_limit:
                return ranked

    for gc in graph.all_language_codes():
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


def ranked_phylo_candidates(graph, language: str, pool_limit: int = 400) -> List[str]:
    return [str(rec["glottocode"]) for rec in ranked_phylo_records(graph, language, pool_limit=pool_limit)]


_RELATION_PRIORITY = {
    "same_immediate_branch": 5,
    "sibling_branch": 4,
    "nearby_cousin_branch": 3,
    "higher_shared_ancestor": 2,
    "phylogenetic_neighbor": 2,
    "phylogenetic_fallback": 0,
}


def _observed_value(graph, language: str, feature: str) -> Optional[str]:
    return graph.language_observations.get(str(language), {}).get(str(feature))


def _has_observed_value(graph, language: str, feature: str) -> bool:
    return _observed_value(graph, language, feature) is not None


def _observed_feature_set(graph, language: str, features: Sequence[str]) -> set[str]:
    observed = graph.language_observations.get(str(language), {})
    return {str(feature) for feature in features if str(feature) in observed}


def _shared_feature_value_count(graph, reference_language: str, candidate_language: str, features: Sequence[str]) -> int:
    reference = graph.language_observations.get(str(reference_language), {})
    candidate = graph.language_observations.get(str(candidate_language), {})
    score = 0
    for feature in features:
        feat = str(feature)
        if feat in reference and feat in candidate and reference[feat] == candidate[feat]:
            score += 1
    return score


def _typed_phylo_score(
    graph,
    reference_language: str,
    candidate_language: str,
    record: Dict[str, object],
    target_feature: str,
    correlated: Sequence[str],
    original_rank: int,
) -> tuple:
    relation_type = str(record.get("relation_type") or "phylogenetic_neighbor")
    tree_distance = _coerce_non_negative_int(record.get("tree_distance"), original_rank + 1)
    shared_depth = record.get("shared_ancestor_depth")
    try:
        shared_depth_num = int(shared_depth) if shared_depth is not None else 999
    except Exception:
        shared_depth_num = 999

    feature_targets = [str(feature) for feature in correlated if str(feature) != str(target_feature)]
    if str(target_feature) not in feature_targets:
        feature_targets.append(str(target_feature))

    has_target = 1 if _has_observed_value(graph, candidate_language, target_feature) else 0
    ref_observed = _observed_feature_set(graph, reference_language, feature_targets)
    cand_observed = _observed_feature_set(graph, candidate_language, feature_targets)
    anchor_overlap = len((cand_observed & ref_observed) - ({str(target_feature)}))
    new_coverage = len(cand_observed - ref_observed)
    shared_values = _shared_feature_value_count(graph, reference_language, candidate_language, feature_targets)
    relation_priority = _RELATION_PRIORITY.get(relation_type, 1)
    fallback_penalty = 1 if relation_type == "phylogenetic_fallback" else 0

    return (
        -fallback_penalty,
        has_target,
        relation_priority,
        anchor_overlap,
        shared_values,
        new_coverage,
        -tree_distance,
        -shared_depth_num,
        -original_rank,
        str(candidate_language),
    )


def ranked_phylo_records_typed(
    graph,
    language: str,
    target_feature: str,
    correlated: Sequence[str],
    pool_limit: int = 400,
) -> List[Dict[str, object]]:
    base_records = ranked_phylo_records(graph, language, pool_limit=pool_limit)
    scored: List[tuple[tuple, int, Dict[str, object]]] = []
    for original_rank, record in enumerate(base_records):
        candidate_language = str(record["glottocode"])
        score = _typed_phylo_score(
            graph,
            reference_language=language,
            candidate_language=candidate_language,
            record=record,
            target_feature=target_feature,
            correlated=correlated,
            original_rank=original_rank,
        )
        scored.append((score, original_rank, record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, _, record in scored[:pool_limit]]


def ranked_phylo_candidates_typed(
    graph,
    language: str,
    target_feature: str,
    correlated: Sequence[str],
    pool_limit: int = 400,
) -> List[str]:
    return [
        str(record["glottocode"])
        for record in ranked_phylo_records_typed(
            graph,
            language=language,
            target_feature=target_feature,
            correlated=correlated,
            pool_limit=pool_limit,
        )
    ]


def ranked_geo_candidates(graph, language: str, pool_limit: int = 1200) -> List[str]:
    lang_node = graph.language_node(language)
    if lang_node is None:
        return []

    seen = {str(language)}
    ranked: List[str] = []
    lat0 = _coerce_float(lang_node.get("latitude"))
    lon0 = _coerce_float(lang_node.get("longitude"))

    if lat0 is not None and lon0 is not None:
        scored: List[tuple[float, str]] = []
        for gc in graph.all_language_codes():
            if gc == language:
                continue
            node = graph.language_node(gc)
            if node is None:
                continue
            lat = _coerce_float(node.get("latitude"))
            lon = _coerce_float(node.get("longitude"))
            if lat is None or lon is None:
                continue
            scored.append((_haversine_km(lat0, lon0, lat, lon), gc))
        scored.sort(key=lambda item: (item[0], item[1]))
        for _, gc in scored:
            if gc in seen:
                continue
            seen.add(gc)
            ranked.append(gc)
            if len(ranked) >= pool_limit:
                return ranked

    geo_edges = sorted(
        graph.outgoing(str(lang_node["id"]), "GEO_NEAR"),
        key=lambda edge: (_coerce_non_negative_int(edge.get("rank"), 10**9), str(edge.get("target", ""))),
    )
    for edge in geo_edges:
        target_node = graph.node(str(edge.get("target", "")))
        if not target_node or target_node.get("type") != "Language":
            continue
        gc = str(target_node.get("glottocode"))
        if not gc or gc in seen:
            continue
        seen.add(gc)
        ranked.append(gc)
        if len(ranked) >= pool_limit:
            return ranked

    for gc in graph.all_language_codes():
        if gc in seen:
            continue
        seen.add(gc)
        ranked.append(gc)
        if len(ranked) >= pool_limit:
            break
    return ranked


def _typed_geo_score(
    graph,
    reference_language: str,
    candidate_language: str,
    target_feature: str,
    correlated: Sequence[str],
    original_rank: int,
) -> tuple:
    feature_targets = [str(feature) for feature in correlated if str(feature) != str(target_feature)]
    if str(target_feature) not in feature_targets:
        feature_targets.append(str(target_feature))
    has_target = 1 if _has_observed_value(graph, candidate_language, target_feature) else 0
    ref_observed = _observed_feature_set(graph, reference_language, feature_targets)
    cand_observed = _observed_feature_set(graph, candidate_language, feature_targets)
    anchor_overlap = len((cand_observed & ref_observed) - ({str(target_feature)}))
    shared_values = _shared_feature_value_count(graph, reference_language, candidate_language, feature_targets)
    return (has_target, anchor_overlap, shared_values, -original_rank, str(candidate_language))


def ranked_geo_candidates_typed(
    graph,
    language: str,
    target_feature: str,
    correlated: Sequence[str],
    pool_limit: int = 1200,
) -> List[str]:
    base_candidates = ranked_geo_candidates(graph, language, pool_limit=pool_limit)
    scored: List[tuple[tuple, str]] = []
    for original_rank, candidate_language in enumerate(base_candidates):
        scored.append(
            (
                _typed_geo_score(
                    graph,
                    reference_language=language,
                    candidate_language=str(candidate_language),
                    target_feature=target_feature,
                    correlated=correlated,
                    original_rank=original_rank,
                ),
                str(candidate_language),
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in scored[:pool_limit]]
