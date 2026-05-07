"""CLI: build genetic_neighbours_detailed.json from a checked-out Glottolog tree.

Usage:
  python -m lkb.scripts.build_gen_detail \\
    --typ data/derived/uriel+_typological.csv \\
    --glottolog_root glottolog \\
    --out data/derived/genetic_neighbours_detailed.json \\
    --k 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lkb.kb._kg_builder import build_tree_from_glottolog_fs
from lkb.kb._phylo_tree import phylo_neighbor_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build genetic_neighbours_detailed.json.")
    parser.add_argument("--typ", default="data/derived/uriel+_typological.csv")
    parser.add_argument("--glottolog_root", default="glottolog")
    parser.add_argument("--out", default="data/derived/genetic_neighbours_detailed.json")
    parser.add_argument("--k", type=int, default=200, help="Neighbors per language")
    args = parser.parse_args()

    typ_df = pd.read_csv(args.typ, index_col=0)
    typ_df.index = typ_df.index.astype(str)
    langs = typ_df.index.tolist()
    allowed = set(langs)

    parent, children, level, _ = build_tree_from_glottolog_fs(Path(args.glottolog_root))

    details = {}
    n = len(langs)
    for i, lang in enumerate(langs, start=1):
        details[lang] = phylo_neighbor_records(
            lang,
            k_neighbors=args.k,
            parent=parent,
            children=children,
            level=level,
            allowed_nodes=allowed,
        )
        if i % 1000 == 0:
            print(f"processed {i}/{n}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({n} languages)")


if __name__ == "__main__":
    main()
