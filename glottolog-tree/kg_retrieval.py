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


def _reference_values(graph, language: str, reference_observations: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    if reference_observations is None:
        return graph.language_observations.get(str(language), {})
    return {str(feature): str(value) for feature, value in reference_observations.items()}


def _observed_value(graph, language: str, feature: str) -> Optional[str]:
    return graph.language_observations.get(str(language), {}).get(str(feature))


def _has_observed_value(graph, language: str, feature: str) -> bool:
    return _observed_value(graph, language, feature) is not None


def _text_or_none(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _macroareas(node: Optional[dict]) -> set[str]:
    if not node:
        return set()
    raw = node.get("macroareas")
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set)):
        return {str(item).strip() for item in raw if str(item).strip()}
    text = str(raw).strip()
    return {text} if text else set()


def _categorical_match_score(reference: Optional[str], candidate: Optional[str]) -> int:
    if not reference or not candidate:
        return 0
    return 1 if reference == candidate else -1


def _macroarea_match_score(reference_node: Optional[dict], candidate_node: Optional[dict]) -> int:
    reference_macroareas = _macroareas(reference_node)
    candidate_macroareas = _macroareas(candidate_node)
    if not reference_macroareas or not candidate_macroareas:
        return 0
    return 1 if reference_macroareas & candidate_macroareas else -1


def _geo_distance_km(
    graph,
    reference_language: str,
    candidate_language: str,
    original_rank: int,
) -> float:
    reference_node = graph.language_node(reference_language)
    candidate_node = graph.language_node(candidate_language)
    if reference_node is not None and candidate_node is not None:
        lat0 = _coerce_float(reference_node.get("latitude"))
        lon0 = _coerce_float(reference_node.get("longitude"))
        lat1 = _coerce_float(candidate_node.get("latitude"))
        lon1 = _coerce_float(candidate_node.get("longitude"))
        if lat0 is not None and lon0 is not None and lat1 is not None and lon1 is not None:
            return _haversine_km(lat0, lon0, lat1, lon1)

        reference_id = str(reference_node.get("id", ""))
        candidate_id = str(candidate_node.get("id", ""))
        for edge in graph.outgoing(reference_id, "GEO_NEAR"):
            if str(edge.get("target", "")) != candidate_id:
                continue
            km = _coerce_float(edge.get("km"))
            if km is not None:
                return km

    # Preserve original ordering as a weak fallback when distance is unavailable.
    return float(original_rank + 1) * 1_000_000.0


def _observed_feature_set_from_values(observed: Dict[str, str], features: Sequence[str]) -> set[str]:
    return {str(feature) for feature in features if str(feature) in observed}


def _shared_feature_value_count(reference: Dict[str, str], candidate: Dict[str, str], features: Sequence[str]) -> int:
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
    reference_observations: Optional[Dict[str, str]] = None,
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
    reference_values = _reference_values(graph, reference_language, reference_observations=reference_observations)
    candidate_values = graph.language_observations.get(str(candidate_language), {})
    ref_observed = _observed_feature_set_from_values(reference_values, feature_targets)
    cand_observed = _observed_feature_set_from_values(candidate_values, feature_targets)
    anchor_overlap = len((cand_observed & ref_observed) - ({str(target_feature)}))
    new_coverage = len(cand_observed - ref_observed)
    shared_values = _shared_feature_value_count(reference_values, candidate_values, feature_targets)
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
    reference_observations: Optional[Dict[str, str]] = None,
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
            reference_observations=reference_observations,
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
    reference_observations: Optional[Dict[str, str]] = None,
) -> List[str]:
    return [
        str(record["glottocode"])
        for record in ranked_phylo_records_typed(
            graph,
            language=language,
            target_feature=target_feature,
            correlated=correlated,
            pool_limit=pool_limit,
            reference_observations=reference_observations,
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
    reference_observations: Optional[Dict[str, str]] = None,
) -> tuple:
    feature_targets = [str(feature) for feature in correlated if str(feature) != str(target_feature)]
    if str(target_feature) not in feature_targets:
        feature_targets.append(str(target_feature))
    has_target = 1 if _has_observed_value(graph, candidate_language, target_feature) else 0
    reference_values = _reference_values(graph, reference_language, reference_observations=reference_observations)
    candidate_values = graph.language_observations.get(str(candidate_language), {})
    ref_observed = _observed_feature_set_from_values(reference_values, feature_targets)
    cand_observed = _observed_feature_set_from_values(candidate_values, feature_targets)
    anchor_overlap = len((cand_observed & ref_observed) - ({str(target_feature)}))
    shared_values = _shared_feature_value_count(reference_values, candidate_values, feature_targets)
    reference_node = graph.language_node(reference_language)
    candidate_node = graph.language_node(candidate_language)
    same_parent = _categorical_match_score(
        _text_or_none(reference_node.get("parent_id") if reference_node else None)
        or _text_or_none(reference_node.get("parent_name") if reference_node else None),
        _text_or_none(candidate_node.get("parent_id") if candidate_node else None)
        or _text_or_none(candidate_node.get("parent_name") if candidate_node else None),
    )
    same_family = _categorical_match_score(
        _text_or_none(reference_node.get("family_id") if reference_node else None)
        or _text_or_none(reference_node.get("family_name") if reference_node else None),
        _text_or_none(candidate_node.get("family_id") if candidate_node else None)
        or _text_or_none(candidate_node.get("family_name") if candidate_node else None),
    )
    macro_match = _macroarea_match_score(reference_node, candidate_node)
    distance_km = _geo_distance_km(graph, reference_language, candidate_language, original_rank)
    return (
        has_target,
        anchor_overlap,
        same_parent,
        same_family,
        macro_match,
        shared_values,
        -distance_km,
        -original_rank,
        str(candidate_language),
    )


def ranked_geo_candidates_typed(
    graph,
    language: str,
    target_feature: str,
    correlated: Sequence[str],
    pool_limit: int = 1200,
    reference_observations: Optional[Dict[str, str]] = None,
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
                    reference_observations=reference_observations,
                ),
                str(candidate_language),
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in scored[:pool_limit]]
