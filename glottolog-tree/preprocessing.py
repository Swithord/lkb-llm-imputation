from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyglottolog import Glottolog


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_BASE = _load_module("glottolog_tree_base_preprocessing", _ROOT / "code" / "preprocessing.py")
_TREE = _load_module("glottolog_tree_phylo_tree", _HERE / "phylo_tree.py")


def get_metadata(languages, glottolog, output_path="metadata.csv"):
    return _BASE.get_metadata(languages, glottolog, output_path=output_path)


def compute_correlation_matrix(df, output_path="correlation_matrix.csv"):
    return _BASE.compute_correlation_matrix(df, output_path=output_path)


def compute_genetic_neighbours(languages, glottolog, k=5, output_path="genetic_neighbours.json"):
    if k <= 0:
        result = {lang: [] for lang in languages}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    parent, children, level = _TREE.build_glottolog_tree_maps(glottolog)
    lang_set = set(languages)
    neighbours = {}
    for lang in languages:
        if lang not in parent:
            neighbours[lang] = []
            continue
        neighbours[lang] = _TREE.phylo_neighbors(
            lang,
            k_neighbors=k,
            parent=parent,
            children=children,
            level=level,
            allowed_nodes=lang_set,
        )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(neighbours, f, ensure_ascii=False, indent=2)
    return neighbours


def compute_genetic_neighbour_details(
    languages,
    glottolog,
    k=200,
    output_path="genetic_neighbours_detailed.json",
):
    if k <= 0:
        result = {lang: [] for lang in languages}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    parent, children, level = _TREE.build_glottolog_tree_maps(glottolog)
    lang_set = set(languages)
    details = {}
    for lang in languages:
        if lang not in parent:
            details[lang] = []
            continue
        details[lang] = _TREE.phylo_neighbor_records(
            lang,
            k_neighbors=k,
            parent=parent,
            children=children,
            level=level,
            allowed_nodes=lang_set,
        )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    return details


def compute_geographic_neighbours(languages, glottolog, k=5, output_path="geographic_neighbours.json"):
    _BASE.glottolog = glottolog
    return _BASE.compute_geographic_neighbours(languages, k=k, output_path=output_path)


def main() -> None:
    from urielplus.urielplus import URIELPlus

    parser = argparse.ArgumentParser(description="Precompute URIEL+ resources with Glottolog-tree retrieval outputs.")
    parser.add_argument("--metadata-only", action="store_true", help="Only generate metadata.csv")
    parser.add_argument(
        "--use-integrate-databases",
        action="store_true",
        help="Use URIELPlus.integrate_databases() (may call sys.exit on some setups)",
    )
    parser.add_argument("--gen_k", type=int, default=5)
    parser.add_argument("--gen_detail_k", type=int, default=200)
    args = parser.parse_args()

    _orig_exit = sys.exit
    sys.exit = lambda code=0: (_ for _ in ()).throw(SystemExit(code))
    try:
        u = URIELPlus()
    except SystemExit:
        sys.exit = _orig_exit
        raise

    input_path = os.environ.get("GLOTTOLOG_PATH", "./glottolog")
    glottolog = Glottolog(input_path)
    if not args.metadata_only:
        if args.use_integrate_databases:
            u.integrate_databases()
        else:
            for method in getattr(_BASE, "_PER_SOURCE_INTEGRATIONS", ()):
                _BASE._safe_call(u, method)

    if args.metadata_only:
        try:
            languages = u.get_typological_languages_array()
        except SystemExit:
            languages = []
        typ_df = pd.DataFrame()
    else:
        try:
            X = u.get_typological_data_array()
            features = u.get_typological_features_array()
            languages = u.get_typological_languages_array()
        except SystemExit:
            X = np.empty((0, 0))
            features = []
            languages = []

        if X.ndim == 3:
            X = np.max(X, axis=-1)
        typ_df = pd.DataFrame(X, index=languages, columns=features)

    if not os.path.exists("metadata.csv"):
        get_metadata(languages, glottolog, output_path="metadata.csv")

    languages = list(languages)
    compute_genetic_neighbours(languages, glottolog, k=args.gen_k, output_path="genetic_neighbours.json")
    compute_genetic_neighbour_details(
        languages,
        glottolog,
        k=args.gen_detail_k,
        output_path="genetic_neighbours_detailed.json",
    )

    if _BASE._should_regenerate_neighbours("geographic_neighbours.json"):
        compute_geographic_neighbours(languages, glottolog, k=5, output_path="geographic_neighbours.json")
    if not args.metadata_only:
        typ_df.to_csv("uriel+_typological.csv")

    sys.exit = _orig_exit


if __name__ == "__main__":
    main()
