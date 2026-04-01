import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_tree():
    parent = {
        "root": None,
        "famA": "root",
        "famB": "root",
        "langA": "famA",
        "langB": "famA",
        "langC": "famB",
        "diaA": "langA",
    }
    children = {
        "root": ["famA", "famB"],
        "famA": ["langA", "langB"],
        "famB": ["langC"],
        "langA": ["diaA"],
        "langB": [],
        "langC": [],
        "diaA": [],
    }
    level = {
        "root": "family",
        "famA": "family",
        "famB": "family",
        "langA": "language",
        "langB": "language",
        "langC": "language",
        "diaA": "dialect",
    }
    return parent, children, level


def test_glottolog_tree_records_include_relation_fields():
    module = _load_module("glottolog_tree_phylo_tree_test", "glottolog-tree/phylo_tree.py")
    parent, children, level = _build_tree()
    records = module.phylo_neighbor_records(
        "langA",
        k_neighbors=2,
        parent=parent,
        children=children,
        level=level,
    )
    assert [rec["glottocode"] for rec in records] == ["langB", "langC"]
    assert records[0]["relation_type"] == "same_immediate_branch"
    assert records[0]["tree_distance"] == 2
    assert records[0]["shared_ancestor_depth"] == 1


def test_glottolog_tree_prompt_surfaces_relation_metadata():
    module = _load_module("glottolog_tree_prompting_test", "glottolog-tree/prompting.py")
    module.typ_df = module.pd.DataFrame(
        {
            "feat_target": [-1, 1, 0],
            "anchor_feat": [1, 1, 0],
        },
        index=["langA", "langB", "langC"],
    )
    module.metadata_df = module.pd.DataFrame(
        {
            "name": ["Lang A", "Lang B", "Lang C"],
            "iso639_3": ["aaa", "bbb", "ccc"],
            "family_name": ["Fam", "Fam", "Fam"],
            "parent_name": ["Branch A", "Branch A", "Branch B"],
            "macroareas": ["Area", "Area", "Area"],
            "latitude": [0.0, 0.1, 0.2],
            "longitude": [0.0, 0.1, 0.2],
        },
        index=["langA", "langB", "langC"],
    )
    module.genetic_neighbours = {
        "langA": ["langB", "langC"],
        "langB": ["langA", "langC"],
        "langC": ["langA", "langB"],
    }
    module.genetic_neighbour_details = {
        "langA": [
            {
                "glottocode": "langB",
                "tree_distance": 2,
                "shared_ancestor_depth": 1,
                "relation_type": "same_immediate_branch",
            },
            {
                "glottocode": "langC",
                "tree_distance": 4,
                "shared_ancestor_depth": 2,
                "relation_type": "sibling_branch",
            },
        ]
    }
    module.geographic_neighbours = {
        "langA": ["langC", "langB"],
        "langB": ["langA", "langC"],
        "langC": ["langA", "langB"],
    }
    module.top_n_features = 1
    module.topk_map = {"feat_target": ["anchor_feat"], "anchor_feat": ["feat_target"]}
    module.clue_support_cache = {}
    module.set_prompt_options("v5_glottolog_tree_json", True)

    _, user = module.construct_prompt("langA", "feat_target")
    assert "Glottolog-tree retrieved evidence (detailed evidence):" in user
    assert "relation=same immediate branch" in user
    assert "tree_distance=2" in user
    assert "shared_ancestor_depth=1" in user
