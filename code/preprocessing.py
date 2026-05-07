import json
import os
import sys

import numpy as np
import pandas as pd
from pyglottolog import Glottolog


def _safe_call(obj, method, *args, **kwargs):
    fn = getattr(obj, method, None)
    if fn is None:
        return False
    try:
        fn(*args, **kwargs)
    except SystemExit:
        return False
    except FileNotFoundError as e:
        if "duplicate_feature_sets.json" not in str(e):
            raise
        return False
    except Exception as e:
        if "already integrated" not in str(e).lower():
            raise
        return False
    return True


_PER_SOURCE_INTEGRATIONS = (
    "integrate_saphon",
    "integrate_bdproto",
    "integrate_grambank",
    "integrate_apics",
    "integrate_ewave",
    "inferred_features",
)


def _integrate_per_source(u):
    for method in _PER_SOURCE_INTEGRATIONS:
        _safe_call(u, method)


def _should_regenerate_neighbours(path):
    if not os.path.exists(path):
        return True
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return True
    if not isinstance(data, dict) or not data:
        return True

    lists = [v for v in data.values() if isinstance(v, list)]
    if not lists:
        return True
    # Regenerate stale outputs where every language has an empty neighbor list.
    if all(len(v) == 0 for v in lists):
        return True
    return False


def get_metadata(languages, glottolog, output_path="metadata.csv"):
    try:
        from metadata_utils import get_metadata as _get_metadata
    except Exception as e:
        raise ImportError(
            "metadata_utils.get_metadata not available. Ensure metadata_utils.py is in the project."
        ) from e

    return _get_metadata(languages, glottolog, output_path=output_path)


def compute_correlation_matrix(df, output_path="correlation_matrix.csv"):
    """
    Compute the correlation matrix between different features in the dataframe, and save it as a CSV file.
    :param df: pd.DataFrame, typological dataset
    :param output_path: str, path to save the correlation matrix CSV file
    """
    if not os.path.exists(output_path):
        raise FileNotFoundError(
            f"{output_path} not found. Compute it once with compute_feature_corr.py "
            "and then call this function to load the result."
        )
    corr_df = pd.read_csv(output_path, index_col=0)
    return corr_df   


def compute_genetic_neighbours(languages, k=5, output_path="genetic_neighbours.json"):
    """
    Compute the k nearest genetic neighbours using Glottolog.
    :param languages: List[str], list of glottocodes (of languages covered by URIEL+)
    :param k: int, number of nearest neighbours to retrieve
    :param output_path: str, path to save the genetic neighbours JSON file
    """
    # For each language in languages, retrieve its Glottolog tree
    # You can use glottolog.newick_tree(glottocode) to get the Newick tree for a language
    # Retrieve the k nearest genetic neighbours for each language
    # Save the dictionary as a JSON (key: glottocode, value: list of k nearest neighbours)
    if k <= 0:
        result = {lang: [] for lang in languages}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    try:
        from phylo_neighbors import phylo_neighbors
    except Exception as e:
        raise ImportError(
            "phylo_neighbors not available. Ensure phylo_neighbors.py is in the project."
        ) from e

    if "glottolog" not in globals():
        raise RuntimeError(
            "Global 'glottolog' is not initialized. "
            "Create it first, e.g., glottolog = Glottolog(input_path)."
        )

    # Build parent/children maps from Glottolog languoids.
    parent = {}
    children = {}
    level = {}
    if hasattr(glottolog, "languoids"):
        for l in glottolog.languoids():
            lid = getattr(l, "id", None)
            if lid is None:
                continue
            plevel = getattr(l, "level", None)
            if plevel is not None:
                level[lid] = (
                    getattr(plevel, "id", None)
                    or getattr(plevel, "name", None)
                    or str(plevel)
                )
            p = getattr(l, "parent", None)
            pid = getattr(p, "id", None) if p is not None else None
            parent[lid] = pid
            if pid is not None:
                children.setdefault(pid, []).append(lid)
            children.setdefault(lid, [])
    else:
        raise RuntimeError("Glottolog object does not expose languoids().")

    lang_set = set(languages)
    neighbours = {}
    for lang in languages:
        if lang not in parent:
            neighbours[lang] = []
            continue
        cand = phylo_neighbors(
            lang,
            k_neighbors=k,
            parent=parent,
            children=children,
            level=level,
            allowed_nodes=lang_set,
        )
        neighbours[lang] = [c for c in cand if c != lang][:k]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(neighbours, f, ensure_ascii=False, indent=2)

    return neighbours



def compute_geographic_neighbours(languages, k=5, output_path="geographic_neighbours.json"):
    """
    Compute the k nearest geographic neighbours using Glottolog.
    :param languages: List[str], list of glottocodes (of languages covered by URIEL+)
    :param k: int, number of nearest neighbours to retrieve
    :param output_path: str, path to save the geographic neighbours JSON file
    """
    try:
        from geo_neighbors import compute_geographic_neighbours as _compute_geo
    except Exception as e:
        raise ImportError(
            "geo_neighbors.compute_geographic_neighbours not available. "
            "Ensure geo_neighbors.py is in the project."
        ) from e

    return _compute_geo(
        languages,
        k=k,
        output_path=output_path,
        glottolog=globals().get("glottolog"),
    )


if __name__ == '__main__':
    from urielplus.urielplus import URIELPlus
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-only", action="store_true", help="Only generate metadata.csv")
    parser.add_argument(
        "--use-integrate-databases",
        action="store_true",
        help="Use URIELPlus.integrate_databases() (may call sys.exit on some setups)",
    )
    args = parser.parse_args()

    # Prevent any dependency from terminating the process via sys.exit.
    _orig_exit = sys.exit
    sys.exit = lambda code=0: (_ for _ in ()).throw(SystemExit(code))
    try:
        u = URIELPlus()
    except SystemExit:
        sys.exit = _orig_exit
        raise
    input_path = os.environ.get("GLOTTOLOG_PATH", "./glottolog")
    glottolog = Glottolog(input_path)
    # Construct the URIEL+ typological dataframe, 
    # with glottocodes as the index and typological features as columns
    if not args.metadata_only:
        if args.use_integrate_databases:
            u.integrate_databases()
        else:
            _integrate_per_source(u)
    # Skip set_glottocodes(): URIEL+ is already configured for glottocodes in this setup.

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
            # Default: union across sources (max). Missing stays -1 if all missing.
            X = np.max(X, axis=-1)

        typ_df = pd.DataFrame(X, index=languages, columns=features)

    # Compute and save the language metadata (skip if already present)
    if not os.path.exists("metadata.csv"):
        get_metadata(languages, glottolog, output_path="metadata.csv")
    # Extract the list of languages (in glottocode) in URIEL+
    languages = list(languages)
    # Compute the genetic neighbours and save it
    compute_genetic_neighbours(languages, k=5, output_path="genetic_neighbours.json")

    # Compute the geographic neighbours and save it
    if _should_regenerate_neighbours("geographic_neighbours.json"):
        compute_geographic_neighbours(languages, k=5, output_path="geographic_neighbours.json")
    if not args.metadata_only:
        typ_df.to_csv("uriel+_typological.csv") # Save the typological data

    sys.exit = _orig_exit
