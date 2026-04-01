import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kg_builder_emits_flat_and_detail_phylo_edges():
    builder = _load_module("glottolog_tree_kg_builder_test", "glottolog-tree/kg_builder.py")
    typ_df = pd.DataFrame(
        {"feat_target": [-1, 1, 0]},
        index=["langA", "langB", "langC"],
    )
    meta_df = pd.DataFrame(
        {
            "name": ["Lang A", "Lang B", "Lang C"],
            "latitude": [0.0, 0.0, 1.0],
            "longitude": [0.0, 1.0, 0.0],
        },
        index=["langA", "langB", "langC"],
    )
    nodes, edges = builder.build_kg_records(
        typ_df=typ_df,
        metadata_df=meta_df,
        geographic_neighbours={"langA": ["langB"], "langB": ["langA"], "langC": ["langA"]},
        genetic_neighbours={"langA": ["langC"], "langB": ["langC"], "langC": ["langB"]},
        genetic_neighbour_details={
            "langA": [
                {
                    "glottocode": "langB",
                    "tree_distance": 2,
                    "shared_ancestor_depth": 1,
                    "relation_type": "same_immediate_branch",
                }
            ]
        },
        topk_map={"feat_target": []},
        parent={"langA": None, "langB": None, "langC": None},
        children={"langA": [], "langB": [], "langC": []},
        level={"langA": "language", "langB": "language", "langC": "language"},
        names={"langA": "Lang A", "langB": "Lang B", "langC": "Lang C"},
    )

    assert any(node["id"] == "lang:langA" and node["type"] == "Language" for node in nodes)
    phylo_edges = [edge for edge in edges if edge["type"] == "PHYLO_NEAR" and edge["source"] == "lang:langA"]
    assert any(edge["source_kind"] == "detail" and edge["target"] == "lang:langB" for edge in phylo_edges)
    assert any(edge["source_kind"] == "flat" and edge["target"] == "lang:langC" for edge in phylo_edges)


def test_kg_flat_retrieval_preserves_legacy_order(tmp_path: Path):
    loader = _load_module("glottolog_tree_kg_loader_test", "glottolog-tree/kg_loader.py")
    retrieval = _load_module("glottolog_tree_kg_retrieval_test", "glottolog-tree/kg_retrieval.py")

    nodes = [
        {"id": "lang:langA", "type": "Language", "glottocode": "langA", "order_index": 0, "latitude": 0.0, "longitude": 0.0},
        {"id": "lang:langB", "type": "Language", "glottocode": "langB", "order_index": 1, "latitude": 0.0, "longitude": 1.0},
        {"id": "lang:langC", "type": "Language", "glottocode": "langC", "order_index": 2, "latitude": 0.0, "longitude": 2.0},
        {"id": "lang:langD", "type": "Language", "glottocode": "langD", "order_index": 3, "latitude": 0.0, "longitude": 3.0},
    ]
    edges = [
        {
            "source": "lang:langA",
            "target": "lang:langB",
            "type": "PHYLO_NEAR",
            "source_kind": "detail",
            "rank": 1,
            "tree_distance": 2,
            "shared_ancestor_depth": 1,
            "relation_type": "same_immediate_branch",
        },
        {
            "source": "lang:langA",
            "target": "lang:langC",
            "type": "PHYLO_NEAR",
            "source_kind": "flat",
            "rank": 1,
            "tree_distance": 3,
            "shared_ancestor_depth": None,
            "relation_type": "phylogenetic_neighbor",
        },
        {
            "source": "lang:langB",
            "target": "lang:langD",
            "type": "PHYLO_NEAR",
            "source_kind": "flat",
            "rank": 1,
            "tree_distance": 4,
            "shared_ancestor_depth": None,
            "relation_type": "phylogenetic_neighbor",
        },
        {"source": "lang:langA", "target": "lang:langD", "type": "GEO_NEAR", "rank": 1, "km": 1.0},
    ]
    nodes_path = tmp_path / "kg_nodes.jsonl"
    edges_path = tmp_path / "kg_edges.jsonl"
    nodes_path.write_text("".join(json.dumps(node) + "\n" for node in nodes), encoding="utf-8")
    edges_path.write_text("".join(json.dumps(edge) + "\n" for edge in edges), encoding="utf-8")

    graph = loader.load_kg(nodes_path, edges_path)
    records = retrieval.ranked_phylo_records(graph, "langA", pool_limit=4)
    assert [rec["glottocode"] for rec in records] == ["langB", "langC", "langD"]
    assert records[0]["relation_type"] == "same_immediate_branch"
    assert retrieval.ranked_geo_candidates(graph, "langA", pool_limit=3) == ["langB", "langC", "langD"]


def test_prompting_kg_flat_uses_kg_records(tmp_path: Path):
    module = _load_module("glottolog_tree_prompting_kg_test", "glottolog-tree/prompting.py")
    module.typ_df = module.pd.DataFrame(
        {"feat_target": [-1, 1, 0], "anchor_feat": [1, 1, 0]},
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
    module.genetic_neighbours = {"langA": ["langC"], "langB": ["langC"], "langC": ["langB"]}
    module.genetic_neighbour_details = {
        "langA": [
            {
                "glottocode": "langB",
                "tree_distance": 2,
                "shared_ancestor_depth": 1,
                "relation_type": "same_immediate_branch",
            }
        ]
    }
    module.geographic_neighbours = {"langA": ["langC", "langB"], "langB": ["langA"], "langC": ["langA"]}
    module.top_n_features = 1
    module.topk_map = {"feat_target": ["anchor_feat"], "anchor_feat": ["feat_target"]}
    module.clue_support_cache = {}
    nodes = [
        {"id": "lang:langA", "type": "Language", "glottocode": "langA", "order_index": 0, "latitude": 0.0, "longitude": 0.0},
        {"id": "lang:langB", "type": "Language", "glottocode": "langB", "order_index": 1, "latitude": 0.1, "longitude": 0.1},
        {"id": "lang:langC", "type": "Language", "glottocode": "langC", "order_index": 2, "latitude": 0.2, "longitude": 0.2},
    ]
    edges = [
        {
            "source": "lang:langA",
            "target": "lang:langB",
            "type": "PHYLO_NEAR",
            "source_kind": "detail",
            "rank": 1,
            "tree_distance": 2,
            "shared_ancestor_depth": 1,
            "relation_type": "same_immediate_branch",
        },
        {
            "source": "lang:langA",
            "target": "lang:langC",
            "type": "PHYLO_NEAR",
            "source_kind": "flat",
            "rank": 1,
            "tree_distance": 3,
            "shared_ancestor_depth": None,
            "relation_type": "phylogenetic_neighbor",
        },
        {"source": "lang:langA", "target": "lang:langC", "type": "GEO_NEAR", "rank": 1, "km": 1.0},
    ]
    nodes_path = tmp_path / "kg_nodes.jsonl"
    edges_path = tmp_path / "kg_edges.jsonl"
    nodes_path.write_text("".join(json.dumps(node) + "\n" for node in nodes), encoding="utf-8")
    edges_path.write_text("".join(json.dumps(edge) + "\n" for edge in edges), encoding="utf-8")

    module.set_prompt_options("v5_glottolog_tree_json", True)
    module.set_retrieval_options("kg_flat", str(nodes_path), str(edges_path))
    _, user = module.construct_prompt("langA", "feat_target")
    assert "Glottolog-tree retrieved evidence (detailed evidence):" in user
    assert "relation=same immediate branch" in user
    assert "tree_distance=2" in user
