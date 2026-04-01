from __future__ import annotations

import math
from typing import Dict, List, Optional


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
