from __future__ import annotations

import argparse
import configparser
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


def _parse_multi_value(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return value == -1


def _format_value(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _feature_family(feature: str) -> str:
    return feature.split("_", 1)[0] if "_" in feature else feature


def _safe_float(value) -> Optional[float]:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def build_tree_from_glottolog_fs(root: Path) -> Tuple[Dict[str, Optional[str]], Dict[str, List[str]], Dict[str, str], Dict[str, str]]:
    tree_root = root / "languoids" / "tree"
    if not tree_root.exists():
        raise FileNotFoundError(f"Expected {tree_root} to exist.")

    parent: Dict[str, Optional[str]] = {}
    children: Dict[str, List[str]] = {}
    level: Dict[str, str] = {}
    names: Dict[str, str] = {}

    for md in tree_root.rglob("md.ini"):
        node_dir = md.parent
        node_id = node_dir.name
        rel_parts = node_dir.relative_to(tree_root).parts
        pid = rel_parts[-2] if len(rel_parts) > 1 else None
        parent[node_id] = pid
        children.setdefault(node_id, [])
        if pid is not None:
            children.setdefault(pid, []).append(node_id)

        cp = configparser.ConfigParser(interpolation=None)
        cp.read(md, encoding="utf-8")
        level[node_id] = cp.get("core", "level", fallback="").strip()
        names[node_id] = cp.get("core", "name", fallback=node_id).strip() or node_id

    return parent, children, level, names


def _language_node(glottocode: str, row: pd.Series, observed_feature_count: int) -> dict:
    return {
        "id": f"lang:{glottocode}",
        "type": "Language",
        "glottocode": glottocode,
        "name": None if _is_missing(row.get("name")) else str(row.get("name")),
        "iso639_3": None if _is_missing(row.get("iso639_3")) else str(row.get("iso639_3")),
        "level": None if _is_missing(row.get("level")) else str(row.get("level")),
        "family_id": None if _is_missing(row.get("family_id")) else str(row.get("family_id")),
        "family_name": None if _is_missing(row.get("family_name")) else str(row.get("family_name")),
        "parent_id": None if _is_missing(row.get("parent_id")) else str(row.get("parent_id")),
        "parent_name": None if _is_missing(row.get("parent_name")) else str(row.get("parent_name")),
        "macroareas": _parse_multi_value(row.get("macroareas")),
        "countries": _parse_multi_value(row.get("countries")),
        "latitude": _safe_float(row.get("latitude")),
        "longitude": _safe_float(row.get("longitude")),
        "observed_feature_count": observed_feature_count,
    }


def build_kg_records(
    typ_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    geographic_neighbours: dict,
    genetic_neighbours: dict,
    genetic_neighbour_details: dict,
    topk_map: Dict[str, List[str]],
    parent: Optional[Dict[str, Optional[str]]] = None,
    children: Optional[Dict[str, List[str]]] = None,
    level: Optional[Dict[str, str]] = None,
    names: Optional[Dict[str, str]] = None,
) -> Tuple[List[dict], List[dict]]:
    nodes: List[dict] = []
    edges: List[dict] = []
    node_ids: set[str] = set()

    parent = parent or {}
    children = children or {}
    level = level or {}
    names = names or {}

    # Language nodes in typology-table order preserve the current fallback order.
    for index_pos, glottocode in enumerate(typ_df.index.astype(str).tolist()):
        row = metadata_df.loc[glottocode] if glottocode in metadata_df.index else pd.Series(dtype=object)
        observed_feature_count = int((typ_df.loc[glottocode] != -1).sum()) if glottocode in typ_df.index else 0
        rec = _language_node(glottocode, row, observed_feature_count)
        rec["order_index"] = index_pos
        nodes.append(rec)
        node_ids.add(rec["id"])

    # Tree nodes.
    for node_id, node_level in level.items():
        if node_id in typ_df.index.astype(str):
            continue
        node_type = "Dialect" if "dialect" in str(node_level).lower() else "Clade"
        rec = {
            "id": f"{node_type.lower()}:{node_id}",
            "type": node_type,
            "node_id": node_id,
            "name": names.get(node_id, node_id),
            "level": node_level,
        }
        if rec["id"] not in node_ids:
            nodes.append(rec)
            node_ids.add(rec["id"])

    # Feature and value nodes.
    for feature in typ_df.columns.astype(str).tolist():
        feat_node = {
            "id": f"feat:{feature}",
            "type": "Feature",
            "feature_id": feature,
            "feature_family": _feature_family(feature),
            "allowed_values": sorted({_format_value(v) for v in typ_df[feature].tolist() if not _is_missing(v)}),
        }
        nodes.append(feat_node)
        node_ids.add(feat_node["id"])
        for value in feat_node["allowed_values"]:
            fval_node = {
                "id": f"fval:{feature}:{value}",
                "type": "FeatureValue",
                "feature_id": feature,
                "value": value,
            }
            nodes.append(fval_node)
            node_ids.add(fval_node["id"])

    # Tree edges.
    for child_id, pid in parent.items():
        if pid is None:
            continue
        src_prefix = "lang" if pid in typ_df.index.astype(str) else ("dialect" if "dialect" in str(level.get(pid, "")).lower() else "clade")
        tgt_prefix = "lang" if child_id in typ_df.index.astype(str) else ("dialect" if "dialect" in str(level.get(child_id, "")).lower() else "clade")
        edges.append(
            {
                "source": f"{src_prefix}:{pid}",
                "target": f"{tgt_prefix}:{child_id}",
                "type": "PARENT_OF",
                "distance": 1,
            }
        )

    # Language-to-clade ancestry edges.
    for glottocode in typ_df.index.astype(str).tolist():
        cur = parent.get(glottocode)
        depth = 1
        while cur is not None:
            if cur not in typ_df.index.astype(str) and "dialect" not in str(level.get(cur, "")).lower():
                edges.append(
                    {
                        "source": f"lang:{glottocode}",
                        "target": f"clade:{cur}",
                        "type": "IN_CLADE",
                        "depth_to_clade": depth,
                    }
                )
            cur = parent.get(cur)
            depth += 1

    # Observed features.
    for glottocode in typ_df.index.astype(str).tolist():
        row = typ_df.loc[glottocode]
        for feature, value in row.items():
            if _is_missing(value):
                continue
            formatted = _format_value(value)
            edges.append(
                {
                    "source": f"lang:{glottocode}",
                    "target": f"fval:{feature}:{formatted}",
                    "type": "OBSERVED_AS",
                    "feature_id": str(feature),
                    "value": formatted,
                    "source_db": "uriel+",
                }
            )

    # Geographic neighbors.
    for glottocode, neighbors in geographic_neighbours.items():
        if glottocode not in metadata_df.index:
            continue
        lat0 = _safe_float(metadata_df.loc[glottocode].get("latitude"))
        lon0 = _safe_float(metadata_df.loc[glottocode].get("longitude"))
        for rank, nb in enumerate(neighbors or [], start=1):
            nb = str(nb)
            km = None
            if nb in metadata_df.index and lat0 is not None and lon0 is not None:
                lat1 = _safe_float(metadata_df.loc[nb].get("latitude"))
                lon1 = _safe_float(metadata_df.loc[nb].get("longitude"))
                if lat1 is not None and lon1 is not None:
                    km = round(_haversine_km(lat0, lon0, lat1, lon1), 4)
            edges.append(
                {
                    "source": f"lang:{glottocode}",
                    "target": f"lang:{nb}",
                    "type": "GEO_NEAR",
                    "rank": rank,
                    "km": km,
                }
            )

    # Phylogenetic neighbors. Keep both detailed and flat edges so Stage 3 can
    # reproduce the current retrieval order through the KG abstraction.
    for glottocode, records in (genetic_neighbour_details or {}).items():
        if not isinstance(records, list):
            continue
        for rank, rec in enumerate(records, start=1):
            nb = str(rec.get("glottocode"))
            edges.append(
                {
                    "source": f"lang:{glottocode}",
                    "target": f"lang:{nb}",
                    "type": "PHYLO_NEAR",
                    "source_kind": "detail",
                    "rank": rank,
                    "tree_distance": rec.get("tree_distance"),
                    "shared_ancestor_depth": rec.get("shared_ancestor_depth"),
                    "relation_type": rec.get("relation_type"),
                }
            )
    for glottocode, neighbors in genetic_neighbours.items():
        for rank, nb in enumerate(neighbors or [], start=1):
            edges.append(
                {
                    "source": f"lang:{glottocode}",
                    "target": f"lang:{str(nb)}",
                    "type": "PHYLO_NEAR",
                    "source_kind": "flat",
                    "rank": rank,
                    "tree_distance": rank,
                    "shared_ancestor_depth": None,
                    "relation_type": "phylogenetic_neighbor",
                }
            )

    # Feature correlations.
    for feature, others in topk_map.items():
        for rank, other in enumerate(others, start=1):
            edges.append(
                {
                    "source": f"feat:{feature}",
                    "target": f"feat:{other}",
                    "type": "FEATURE_CORRELATED",
                    "rank": rank,
                }
            )

    return nodes, edges


def write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _build_topk_map(df: pd.DataFrame) -> Dict[str, List[str]]:
    tmp: Dict[str, List[Tuple[int, str]]] = {}
    for _, row in df.iterrows():
        feature = str(row["feature"])
        other = str(row["other_feature"])
        rank = int(row["rank"]) if "rank" in row and not pd.isna(row["rank"]) else 0
        tmp.setdefault(feature, []).append((rank, other))
    return {feature: [other for _, other in sorted(items, key=lambda item: item[0])] for feature, items in tmp.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build typed KG node/edge JSONL files for Glottolog-grounded retrieval.")
    parser.add_argument("--typ", type=str, default="data/derived/uriel+_typological.csv")
    parser.add_argument("--meta", type=str, default="data/derived/metadata.csv")
    parser.add_argument("--geo", type=str, default="data/derived/geographic_neighbours.json")
    parser.add_argument("--gen", type=str, default="data/derived/genetic_neighbours.json")
    parser.add_argument("--gen_detail", type=str, default="data/derived/genetic_neighbours_detailed.json")
    parser.add_argument("--topk", type=str, default="data/features/topk_per_feature.csv")
    parser.add_argument("--glottolog_root", type=str, default="glottolog")
    parser.add_argument("--nodes_out", type=str, default="artifacts/resources/kg_nodes.jsonl")
    parser.add_argument("--edges_out", type=str, default="artifacts/resources/kg_edges.jsonl")
    args = parser.parse_args()

    typ_df = pd.read_csv(args.typ, index_col=0)
    typ_df.index = typ_df.index.astype(str)
    metadata_df = pd.read_csv(args.meta, index_col=0)
    metadata_df.index = metadata_df.index.astype(str)
    geographic_neighbours = json.loads(Path(args.geo).read_text(encoding="utf-8"))
    genetic_neighbours = json.loads(Path(args.gen).read_text(encoding="utf-8"))
    genetic_detail_path = Path(args.gen_detail)
    genetic_neighbour_details = (
        json.loads(genetic_detail_path.read_text(encoding="utf-8")) if genetic_detail_path.exists() else {}
    )
    topk_map = _build_topk_map(pd.read_csv(args.topk))
    parent, children, level, names = build_tree_from_glottolog_fs(Path(args.glottolog_root))

    nodes, edges = build_kg_records(
        typ_df=typ_df,
        metadata_df=metadata_df,
        geographic_neighbours=geographic_neighbours,
        genetic_neighbours=genetic_neighbours,
        genetic_neighbour_details=genetic_neighbour_details,
        topk_map=topk_map,
        parent=parent,
        children=children,
        level=level,
        names=names,
    )

    write_jsonl(Path(args.nodes_out), nodes)
    write_jsonl(Path(args.edges_out), edges)
    print(f"Wrote {len(nodes)} nodes -> {args.nodes_out}")
    print(f"Wrote {len(edges)} edges -> {args.edges_out}")


if __name__ == "__main__":
    main()
