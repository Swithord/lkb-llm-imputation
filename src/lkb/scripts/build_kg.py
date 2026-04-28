"""CLI: build typed KG node/edge JSONL files from URIEL+ and Glottolog data.

Usage:
  python -m lkb.scripts.build_kg \\
    --typ data/derived/uriel+_typological.csv \\
    --meta data/derived/metadata.csv \\
    --geo data/derived/geographic_neighbours.json \\
    --gen data/derived/genetic_neighbours.json \\
    --gen_detail data/derived/genetic_neighbours_detailed.json \\
    --topk data/features/topk_per_feature.csv \\
    --glottolog_root glottolog \\
    --nodes_out artifacts/resources/kg_nodes.jsonl \\
    --edges_out artifacts/resources/kg_edges.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lkb.kb._kg_builder import build_kg_records, build_topk_map, build_tree_from_glottolog_fs, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build typed KG node/edge JSONL files.")
    parser.add_argument("--typ", default="data/derived/uriel+_typological.csv")
    parser.add_argument("--meta", default="data/derived/metadata.csv")
    parser.add_argument("--geo", default="data/derived/geographic_neighbours.json")
    parser.add_argument("--gen", default="data/derived/genetic_neighbours.json")
    parser.add_argument("--gen_detail", default="data/derived/genetic_neighbours_detailed.json")
    parser.add_argument("--topk", default="data/features/topk_per_feature.csv")
    parser.add_argument("--glottolog_root", default="glottolog")
    parser.add_argument("--nodes_out", default="artifacts/resources/kg_nodes.jsonl")
    parser.add_argument("--edges_out", default="artifacts/resources/kg_edges.jsonl")
    args = parser.parse_args()

    typ_df = pd.read_csv(args.typ, index_col=0)
    typ_df.index = typ_df.index.astype(str)
    metadata_df = pd.read_csv(args.meta, index_col=0)
    metadata_df.index = metadata_df.index.astype(str)
    geographic_neighbours = json.loads(Path(args.geo).read_text(encoding="utf-8"))
    genetic_neighbours = json.loads(Path(args.gen).read_text(encoding="utf-8"))
    gen_detail_path = Path(args.gen_detail)
    genetic_neighbour_details = (
        json.loads(gen_detail_path.read_text(encoding="utf-8")) if gen_detail_path.exists() else {}
    )
    topk_map = build_topk_map(pd.read_csv(args.topk))
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
