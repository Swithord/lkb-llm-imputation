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


def test_kg_typed_prioritizes_feature_supported_candidates(tmp_path: Path):
    loader = _load_module("glottolog_tree_kg_loader_typed_test", "glottolog-tree/kg_loader.py")
    retrieval = _load_module("glottolog_tree_kg_retrieval_typed_test", "glottolog-tree/kg_retrieval.py")

    nodes = [
        {"id": "lang:langA", "type": "Language", "glottocode": "langA", "order_index": 0, "latitude": 0.0, "longitude": 0.0},
        {"id": "lang:langB", "type": "Language", "glottocode": "langB", "order_index": 1, "latitude": 0.0, "longitude": 1.0},
        {"id": "lang:langC", "type": "Language", "glottocode": "langC", "order_index": 2, "latitude": 0.0, "longitude": 2.0},
        {"id": "feat:feat_target", "type": "Feature", "feature_id": "feat_target"},
        {"id": "feat:anchor_feat", "type": "Feature", "feature_id": "anchor_feat"},
        {"id": "fval:feat_target:1", "type": "FeatureValue", "feature_id": "feat_target", "value": "1"},
        {"id": "fval:anchor_feat:1", "type": "FeatureValue", "feature_id": "anchor_feat", "value": "1"},
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
            "source_kind": "detail",
            "rank": 2,
            "tree_distance": 4,
            "shared_ancestor_depth": 2,
            "relation_type": "higher_shared_ancestor",
        },
        {"source": "lang:langA", "target": "fval:anchor_feat:1", "type": "OBSERVED_AS", "feature_id": "anchor_feat", "value": "1"},
        {"source": "lang:langC", "target": "fval:feat_target:1", "type": "OBSERVED_AS", "feature_id": "feat_target", "value": "1"},
        {"source": "lang:langC", "target": "fval:anchor_feat:1", "type": "OBSERVED_AS", "feature_id": "anchor_feat", "value": "1"},
        {"source": "feat:feat_target", "target": "feat:anchor_feat", "type": "FEATURE_CORRELATED", "rank": 1},
    ]
    nodes_path = tmp_path / "kg_nodes.jsonl"
    edges_path = tmp_path / "kg_edges.jsonl"
    nodes_path.write_text("".join(json.dumps(node) + "\n" for node in nodes), encoding="utf-8")
    edges_path.write_text("".join(json.dumps(edge) + "\n" for edge in edges), encoding="utf-8")

    graph = loader.load_kg(nodes_path, edges_path)
    records = retrieval.ranked_phylo_records_typed(
        graph,
        language="langA",
        target_feature="feat_target",
        correlated=["anchor_feat"],
        pool_limit=3,
    )
    assert [rec["glottocode"] for rec in records[:2]] == ["langC", "langB"]


def test_kg_typed_geo_prefers_closer_candidates_when_other_signals_tie(tmp_path: Path):
    loader = _load_module("glottolog_tree_kg_loader_geo_distance_test", "glottolog-tree/kg_loader.py")
    retrieval = _load_module("glottolog_tree_kg_retrieval_geo_distance_test", "glottolog-tree/kg_retrieval.py")

    nodes = [
        {
            "id": "lang:langA",
            "type": "Language",
            "glottocode": "langA",
            "order_index": 0,
            "latitude": 0.0,
            "longitude": 0.0,
            "family_name": "Fam",
            "parent_name": "Branch",
            "macroareas": ["Area"],
        },
        {
            "id": "lang:langB",
            "type": "Language",
            "glottocode": "langB",
            "order_index": 1,
            "latitude": 0.0,
            "longitude": 0.1,
            "family_name": "Fam",
            "parent_name": "Branch",
            "macroareas": ["Area"],
        },
        {
            "id": "lang:langC",
            "type": "Language",
            "glottocode": "langC",
            "order_index": 2,
            "latitude": 0.0,
            "longitude": 2.0,
            "family_name": "Fam",
            "parent_name": "Branch",
            "macroareas": ["Area"],
        },
        {"id": "feat:feat_target", "type": "Feature", "feature_id": "feat_target"},
        {"id": "fval:feat_target:1", "type": "FeatureValue", "feature_id": "feat_target", "value": "1"},
    ]
    edges = [
        {"source": "lang:langA", "target": "lang:langB", "type": "GEO_NEAR", "rank": 2, "km": 11.1},
        {"source": "lang:langA", "target": "lang:langC", "type": "GEO_NEAR", "rank": 1, "km": 222.4},
        {"source": "lang:langB", "target": "fval:feat_target:1", "type": "OBSERVED_AS", "feature_id": "feat_target", "value": "1"},
        {"source": "lang:langC", "target": "fval:feat_target:1", "type": "OBSERVED_AS", "feature_id": "feat_target", "value": "1"},
    ]
    nodes_path = tmp_path / "kg_nodes.jsonl"
    edges_path = tmp_path / "kg_edges.jsonl"
    nodes_path.write_text("".join(json.dumps(node) + "\n" for node in nodes), encoding="utf-8")
    edges_path.write_text("".join(json.dumps(edge) + "\n" for edge in edges), encoding="utf-8")

    graph = loader.load_kg(nodes_path, edges_path)
    candidates = retrieval.ranked_geo_candidates_typed(
        graph,
        language="langA",
        target_feature="feat_target",
        correlated=[],
        pool_limit=3,
    )
    assert candidates[:2] == ["langB", "langC"]


def test_kg_typed_geo_penalizes_family_and_macroarea_mismatch(tmp_path: Path):
    loader = _load_module("glottolog_tree_kg_loader_geo_penalty_test", "glottolog-tree/kg_loader.py")
    retrieval = _load_module("glottolog_tree_kg_retrieval_geo_penalty_test", "glottolog-tree/kg_retrieval.py")

    nodes = [
        {
            "id": "lang:langA",
            "type": "Language",
            "glottocode": "langA",
            "order_index": 0,
            "latitude": 0.0,
            "longitude": 0.0,
            "family_name": "FamA",
            "parent_name": "BranchA",
            "macroareas": ["AreaA"],
        },
        {
            "id": "lang:langB",
            "type": "Language",
            "glottocode": "langB",
            "order_index": 1,
            "latitude": 0.0,
            "longitude": 0.2,
            "family_name": "FamA",
            "parent_name": "BranchA",
            "macroareas": ["AreaA"],
        },
        {
            "id": "lang:langC",
            "type": "Language",
            "glottocode": "langC",
            "order_index": 2,
            "latitude": 0.0,
            "longitude": 0.1,
            "family_name": "FamB",
            "parent_name": "BranchB",
            "macroareas": ["AreaB"],
        },
        {"id": "feat:feat_target", "type": "Feature", "feature_id": "feat_target"},
        {"id": "fval:feat_target:1", "type": "FeatureValue", "feature_id": "feat_target", "value": "1"},
    ]
    edges = [
        {"source": "lang:langA", "target": "lang:langB", "type": "GEO_NEAR", "rank": 1, "km": 22.2},
        {"source": "lang:langA", "target": "lang:langC", "type": "GEO_NEAR", "rank": 2, "km": 11.1},
        {"source": "lang:langB", "target": "fval:feat_target:1", "type": "OBSERVED_AS", "feature_id": "feat_target", "value": "1"},
        {"source": "lang:langC", "target": "fval:feat_target:1", "type": "OBSERVED_AS", "feature_id": "feat_target", "value": "1"},
    ]
    nodes_path = tmp_path / "kg_nodes.jsonl"
    edges_path = tmp_path / "kg_edges.jsonl"
    nodes_path.write_text("".join(json.dumps(node) + "\n" for node in nodes), encoding="utf-8")
    edges_path.write_text("".join(json.dumps(edge) + "\n" for edge in edges), encoding="utf-8")

    graph = loader.load_kg(nodes_path, edges_path)
    candidates = retrieval.ranked_geo_candidates_typed(
        graph,
        language="langA",
        target_feature="feat_target",
        correlated=[],
        pool_limit=3,
    )
    assert candidates[:2] == ["langB", "langC"]


def test_prompting_kg_typed_surfaces_reordered_evidence(tmp_path: Path):
    module = _load_module("glottolog_tree_prompting_kg_typed_test", "glottolog-tree/prompting.py")
    module.typ_df = module.pd.DataFrame(
        {"feat_target": [-1, -1, 1], "anchor_feat": [1, -1, 1]},
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
    module.genetic_neighbours = {"langA": ["langB", "langC"], "langB": ["langA"], "langC": ["langA"]}
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
                "relation_type": "higher_shared_ancestor",
            },
        ]
    }
    module.geographic_neighbours = {"langA": ["langB", "langC"], "langB": ["langA"], "langC": ["langA"]}
    module.top_n_features = 1
    module.topk_map = {"feat_target": ["anchor_feat"], "anchor_feat": ["feat_target"]}
    module.clue_support_cache = {}
    nodes = [
        {"id": "lang:langA", "type": "Language", "glottocode": "langA", "order_index": 0, "latitude": 0.0, "longitude": 0.0},
        {"id": "lang:langB", "type": "Language", "glottocode": "langB", "order_index": 1, "latitude": 0.1, "longitude": 0.1},
        {"id": "lang:langC", "type": "Language", "glottocode": "langC", "order_index": 2, "latitude": 0.2, "longitude": 0.2},
        {"id": "feat:feat_target", "type": "Feature", "feature_id": "feat_target"},
        {"id": "feat:anchor_feat", "type": "Feature", "feature_id": "anchor_feat"},
        {"id": "fval:feat_target:1", "type": "FeatureValue", "feature_id": "feat_target", "value": "1"},
        {"id": "fval:anchor_feat:1", "type": "FeatureValue", "feature_id": "anchor_feat", "value": "1"},
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
            "source_kind": "detail",
            "rank": 2,
            "tree_distance": 4,
            "shared_ancestor_depth": 2,
            "relation_type": "higher_shared_ancestor",
        },
        {"source": "lang:langA", "target": "fval:anchor_feat:1", "type": "OBSERVED_AS", "feature_id": "anchor_feat", "value": "1"},
        {"source": "lang:langC", "target": "fval:feat_target:1", "type": "OBSERVED_AS", "feature_id": "feat_target", "value": "1"},
        {"source": "lang:langC", "target": "fval:anchor_feat:1", "type": "OBSERVED_AS", "feature_id": "anchor_feat", "value": "1"},
        {"source": "feat:feat_target", "target": "feat:anchor_feat", "type": "FEATURE_CORRELATED", "rank": 1},
        {"source": "lang:langA", "target": "lang:langB", "type": "GEO_NEAR", "rank": 1, "km": 1.0},
        {"source": "lang:langA", "target": "lang:langC", "type": "GEO_NEAR", "rank": 2, "km": 2.0},
    ]
    nodes_path = tmp_path / "kg_nodes.jsonl"
    edges_path = tmp_path / "kg_edges.jsonl"
    nodes_path.write_text("".join(json.dumps(node) + "\n" for node in nodes), encoding="utf-8")
    edges_path.write_text("".join(json.dumps(edge) + "\n" for edge in edges), encoding="utf-8")

    module.set_prompt_options("v5_glottolog_tree_json", True)
    module.set_retrieval_options("kg_typed", str(nodes_path), str(edges_path))
    _, user = module.construct_prompt("langA", "feat_target")
    evidence_start = user.index("Glottolog-tree retrieved evidence")
    lang_c_pos = user.index("Lang C", evidence_start)
    lang_b_pos = user.index("Lang B", evidence_start)
    assert lang_c_pos < lang_b_pos


def test_prompting_kg_typed_contrastive_includes_yes_and_no_support(tmp_path: Path):
    module = _load_module("glottolog_tree_prompting_kg_contrastive_test", "glottolog-tree/prompting.py")
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
    module.genetic_neighbours = {"langA": ["langB", "langC"], "langB": ["langA"], "langC": ["langA"]}
    module.genetic_neighbour_details = {
        "langA": [
            {"glottocode": "langB", "tree_distance": 2, "shared_ancestor_depth": 1, "relation_type": "same_immediate_branch"},
            {"glottocode": "langC", "tree_distance": 3, "shared_ancestor_depth": 1, "relation_type": "sibling_branch"},
        ]
    }
    module.geographic_neighbours = {"langA": ["langB", "langC"], "langB": ["langA"], "langC": ["langA"]}
    module.top_n_features = 1
    module.topk_map = {"feat_target": ["anchor_feat"], "anchor_feat": ["feat_target"]}
    module.clue_support_cache = {}
    nodes = [
        {"id": "lang:langA", "type": "Language", "glottocode": "langA", "order_index": 0, "latitude": 0.0, "longitude": 0.0},
        {"id": "lang:langB", "type": "Language", "glottocode": "langB", "order_index": 1, "latitude": 0.1, "longitude": 0.1},
        {"id": "lang:langC", "type": "Language", "glottocode": "langC", "order_index": 2, "latitude": 0.2, "longitude": 0.2},
        {"id": "feat:feat_target", "type": "Feature", "feature_id": "feat_target"},
        {"id": "feat:anchor_feat", "type": "Feature", "feature_id": "anchor_feat"},
        {"id": "fval:feat_target:1", "type": "FeatureValue", "feature_id": "feat_target", "value": "1"},
        {"id": "fval:feat_target:0", "type": "FeatureValue", "feature_id": "feat_target", "value": "0"},
    ]
    edges = [
        {"source": "lang:langA", "target": "lang:langB", "type": "PHYLO_NEAR", "source_kind": "detail", "rank": 1, "tree_distance": 2, "shared_ancestor_depth": 1, "relation_type": "same_immediate_branch"},
        {"source": "lang:langA", "target": "lang:langC", "type": "PHYLO_NEAR", "source_kind": "detail", "rank": 2, "tree_distance": 2, "shared_ancestor_depth": 1, "relation_type": "same_immediate_branch"},
        {"source": "lang:langB", "target": "fval:feat_target:1", "type": "OBSERVED_AS", "feature_id": "feat_target", "value": "1"},
        {"source": "lang:langC", "target": "fval:feat_target:0", "type": "OBSERVED_AS", "feature_id": "feat_target", "value": "0"},
        {"source": "lang:langA", "target": "lang:langB", "type": "GEO_NEAR", "rank": 1, "km": 1.0},
        {"source": "lang:langA", "target": "lang:langC", "type": "GEO_NEAR", "rank": 2, "km": 2.0},
    ]
    nodes_path = tmp_path / "kg_nodes.jsonl"
    edges_path = tmp_path / "kg_edges.jsonl"
    nodes_path.write_text("".join(json.dumps(node) + "\n" for node in nodes), encoding="utf-8")
    edges_path.write_text("".join(json.dumps(edge) + "\n" for edge in edges), encoding="utf-8")

    module.set_prompt_options("v5_glottolog_tree_json", True)
    module.set_retrieval_options("kg_typed_contrastive", str(nodes_path), str(edges_path))
    _, user = module.construct_prompt("langA", "feat_target")
    assert "Contrastive decision summary:" in user
    assert "Closest phylogenetic support for 1: Lang B" in user
    assert "Closest phylogenetic support for 0: Lang C" in user
    assert "Selected 1-supporters: Lang B (same branch, d=2), Lang B (15.7 km)" in user
    assert "Selected 0-supporters: Lang C (same branch, d=2), Lang C (31.5 km)" in user


def test_prompting_kg_typed_contrastive_contrast_prompt_uses_v4_style_layout(tmp_path: Path):
    module = _load_module("glottolog_tree_prompting_kg_contrastive_v5_test", "glottolog-tree/prompting.py")
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
    module.genetic_neighbours = {"langA": ["langB", "langC"], "langB": ["langA"], "langC": ["langA"]}
    module.genetic_neighbour_details = {
        "langA": [
            {"glottocode": "langB", "tree_distance": 2, "shared_ancestor_depth": 1, "relation_type": "same_immediate_branch"},
            {"glottocode": "langC", "tree_distance": 3, "shared_ancestor_depth": 1, "relation_type": "sibling_branch"},
        ]
    }
    module.geographic_neighbours = {"langA": ["langB", "langC"], "langB": ["langA"], "langC": ["langA"]}
    module.top_n_features = 1
    module.topk_map = {"feat_target": ["anchor_feat"], "anchor_feat": ["feat_target"]}
    module.clue_support_cache = {}
    nodes = [
        {"id": "lang:langA", "type": "Language", "glottocode": "langA", "order_index": 0, "latitude": 0.0, "longitude": 0.0},
        {"id": "lang:langB", "type": "Language", "glottocode": "langB", "order_index": 1, "latitude": 0.1, "longitude": 0.1},
        {"id": "lang:langC", "type": "Language", "glottocode": "langC", "order_index": 2, "latitude": 0.2, "longitude": 0.2},
        {"id": "feat:feat_target", "type": "Feature", "feature_id": "feat_target"},
        {"id": "feat:anchor_feat", "type": "Feature", "feature_id": "anchor_feat"},
        {"id": "fval:feat_target:1", "type": "FeatureValue", "feature_id": "feat_target", "value": "1"},
        {"id": "fval:feat_target:0", "type": "FeatureValue", "feature_id": "feat_target", "value": "0"},
    ]
    edges = [
        {"source": "lang:langA", "target": "lang:langB", "type": "PHYLO_NEAR", "source_kind": "detail", "rank": 1, "tree_distance": 2, "shared_ancestor_depth": 1, "relation_type": "same_immediate_branch"},
        {"source": "lang:langA", "target": "lang:langC", "type": "PHYLO_NEAR", "source_kind": "detail", "rank": 2, "tree_distance": 2, "shared_ancestor_depth": 1, "relation_type": "same_immediate_branch"},
        {"source": "lang:langB", "target": "fval:feat_target:1", "type": "OBSERVED_AS", "feature_id": "feat_target", "value": "1"},
        {"source": "lang:langC", "target": "fval:feat_target:0", "type": "OBSERVED_AS", "feature_id": "feat_target", "value": "0"},
        {"source": "lang:langA", "target": "lang:langB", "type": "GEO_NEAR", "rank": 1, "km": 1.0},
        {"source": "lang:langA", "target": "lang:langC", "type": "GEO_NEAR", "rank": 2, "km": 2.0},
    ]
    nodes_path = tmp_path / "kg_nodes.jsonl"
    edges_path = tmp_path / "kg_edges.jsonl"
    nodes_path.write_text("".join(json.dumps(node) + "\n" for node in nodes), encoding="utf-8")
    edges_path.write_text("".join(json.dumps(edge) + "\n" for edge in edges), encoding="utf-8")

    module.set_prompt_options("v5_glottolog_tree_contrast_json", True)
    module.set_retrieval_options("kg_typed_contrastive", str(nodes_path), str(edges_path))
    _, user = module.construct_prompt("langA", "feat_target")
    assert "Prompt version: v5_glottolog_tree_contrast_json" in user
    assert "Competing evidence for value 1:" in user
    assert "Competing evidence for value 0:" in user
    assert "Selected phylogenetic/geographic 1-neighbors: Lang B (same branch, d=2), Lang B (15.7 km)" in user
    assert "Selected phylogenetic/geographic 0-neighbors: Lang C (same branch, d=2), Lang C (31.5 km)" in user
    assert "Secondary vote snapshot:" in user
    assert "Glottolog-tree retrieved evidence" not in user


def test_compact_prompt_omits_extra_graph_metadata(tmp_path: Path):
    module = _load_module("glottolog_tree_prompting_compact_test", "glottolog-tree/prompting.py")
    module.typ_df = module.pd.DataFrame(
        {"feat_target": [-1, 1], "anchor_feat": [1, 1]},
        index=["langA", "langB"],
    )
    module.metadata_df = module.pd.DataFrame(
        {
            "name": ["Lang A", "Lang B"],
            "iso639_3": ["aaa", "bbb"],
            "family_name": ["Fam", "Fam"],
            "parent_name": ["Branch A", "Branch A"],
            "macroareas": ["Area", "Area"],
            "latitude": [0.0, 0.1],
            "longitude": [0.0, 0.1],
        },
        index=["langA", "langB"],
    )
    module.genetic_neighbours = {"langA": ["langB"], "langB": ["langA"]}
    module.genetic_neighbour_details = {
        "langA": [{"glottocode": "langB", "tree_distance": 2, "shared_ancestor_depth": 1, "relation_type": "same_immediate_branch"}]
    }
    module.geographic_neighbours = {"langA": ["langB"], "langB": ["langA"]}
    module.top_n_features = 1
    module.topk_map = {"feat_target": ["anchor_feat"], "anchor_feat": ["feat_target"]}
    module.clue_support_cache = {}

    module.set_prompt_options("v5_glottolog_tree_compact_json", True)
    module.set_retrieval_options("legacy", None, None)
    _, user = module.construct_prompt("langA", "feat_target")
    assert "Prompt version: v5_glottolog_tree_compact_json" in user
    phylo_section = user.split("Selected geographic neighbors", 1)[0]
    assert "glottocode=langB" not in phylo_section
    assert "shared_ancestor_depth=" not in phylo_section
    assert "tree_distance=2" in user


def test_hybrid_flat_kg_keeps_legacy_prefix_and_adds_graph_support(tmp_path: Path):
    module = _load_module("glottolog_tree_prompting_hybrid_test", "glottolog-tree/prompting.py")
    module.typ_df = module.pd.DataFrame(
        {"feat_target": [-1, -1, 1], "anchor_feat": [1, -1, 1]},
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
    module.genetic_neighbours = {"langA": ["langB"], "langB": ["langA"], "langC": ["langA"]}
    module.genetic_neighbour_details = {"langA": [{"glottocode": "langB", "tree_distance": 2, "shared_ancestor_depth": 1, "relation_type": "same_immediate_branch"}]}
    module.geographic_neighbours = {"langA": ["langB", "langC"], "langB": ["langA"], "langC": ["langA"]}
    module.top_n_features = 1
    module.topk_map = {"feat_target": ["anchor_feat"], "anchor_feat": ["feat_target"]}
    module.clue_support_cache = {}
    nodes = [
        {"id": "lang:langA", "type": "Language", "glottocode": "langA", "order_index": 0, "latitude": 0.0, "longitude": 0.0},
        {"id": "lang:langB", "type": "Language", "glottocode": "langB", "order_index": 1, "latitude": 0.1, "longitude": 0.1},
        {"id": "lang:langC", "type": "Language", "glottocode": "langC", "order_index": 2, "latitude": 0.2, "longitude": 0.2},
        {"id": "feat:feat_target", "type": "Feature", "feature_id": "feat_target"},
        {"id": "feat:anchor_feat", "type": "Feature", "feature_id": "anchor_feat"},
        {"id": "fval:feat_target:1", "type": "FeatureValue", "feature_id": "feat_target", "value": "1"},
        {"id": "fval:anchor_feat:1", "type": "FeatureValue", "feature_id": "anchor_feat", "value": "1"},
    ]
    edges = [
        {"source": "lang:langA", "target": "lang:langB", "type": "PHYLO_NEAR", "source_kind": "detail", "rank": 1, "tree_distance": 2, "shared_ancestor_depth": 1, "relation_type": "same_immediate_branch"},
        {"source": "lang:langA", "target": "lang:langC", "type": "PHYLO_NEAR", "source_kind": "detail", "rank": 2, "tree_distance": 4, "shared_ancestor_depth": 2, "relation_type": "higher_shared_ancestor"},
        {"source": "lang:langA", "target": "fval:anchor_feat:1", "type": "OBSERVED_AS", "feature_id": "anchor_feat", "value": "1"},
        {"source": "lang:langC", "target": "fval:feat_target:1", "type": "OBSERVED_AS", "feature_id": "feat_target", "value": "1"},
        {"source": "lang:langC", "target": "fval:anchor_feat:1", "type": "OBSERVED_AS", "feature_id": "anchor_feat", "value": "1"},
        {"source": "feat:feat_target", "target": "feat:anchor_feat", "type": "FEATURE_CORRELATED", "rank": 1},
        {"source": "lang:langA", "target": "lang:langB", "type": "GEO_NEAR", "rank": 1, "km": 1.0},
        {"source": "lang:langA", "target": "lang:langC", "type": "GEO_NEAR", "rank": 2, "km": 2.0},
    ]
    nodes_path = tmp_path / "kg_nodes.jsonl"
    edges_path = tmp_path / "kg_edges.jsonl"
    nodes_path.write_text("".join(json.dumps(node) + "\n" for node in nodes), encoding="utf-8")
    edges_path.write_text("".join(json.dumps(edge) + "\n" for edge in edges), encoding="utf-8")

    module.set_prompt_options("v5_glottolog_tree_json", True)
    module.set_retrieval_options("hybrid_flat_kg", str(nodes_path), str(edges_path))
    _, user = module.construct_prompt("langA", "feat_target")
    phylo_section = user.split("Selected geographic neighbors", 1)[0]
    assert "Lang B" in phylo_section
    assert "Closest phylogenetic support for 1: Lang C" in user


def test_kg_typed_does_not_leak_masked_target_value_from_kg(tmp_path: Path):
    loader = _load_module("glottolog_tree_kg_loader_leak_test", "glottolog-tree/kg_loader.py")
    retrieval = _load_module("glottolog_tree_kg_retrieval_leak_test", "glottolog-tree/kg_retrieval.py")

    nodes = [
        {"id": "lang:langA", "type": "Language", "glottocode": "langA", "order_index": 0},
        {"id": "lang:langB", "type": "Language", "glottocode": "langB", "order_index": 1},
        {"id": "lang:langC", "type": "Language", "glottocode": "langC", "order_index": 2},
        {"id": "feat:feat_target", "type": "Feature", "feature_id": "feat_target"},
        {"id": "feat:anchor_feat", "type": "Feature", "feature_id": "anchor_feat"},
        {"id": "fval:feat_target:1", "type": "FeatureValue", "feature_id": "feat_target", "value": "1"},
        {"id": "fval:feat_target:0", "type": "FeatureValue", "feature_id": "feat_target", "value": "0"},
        {"id": "fval:anchor_feat:1", "type": "FeatureValue", "feature_id": "anchor_feat", "value": "1"},
    ]
    edges = [
        {"source": "lang:langA", "target": "lang:langB", "type": "PHYLO_NEAR", "source_kind": "detail", "rank": 1, "tree_distance": 2, "shared_ancestor_depth": 1, "relation_type": "same_immediate_branch"},
        {"source": "lang:langA", "target": "lang:langC", "type": "PHYLO_NEAR", "source_kind": "detail", "rank": 2, "tree_distance": 2, "shared_ancestor_depth": 1, "relation_type": "same_immediate_branch"},
        # KG still contains the hidden target gold for langA.
        {"source": "lang:langA", "target": "fval:feat_target:0", "type": "OBSERVED_AS", "feature_id": "feat_target", "value": "0"},
        {"source": "lang:langA", "target": "fval:anchor_feat:1", "type": "OBSERVED_AS", "feature_id": "anchor_feat", "value": "1"},
        {"source": "lang:langB", "target": "fval:feat_target:0", "type": "OBSERVED_AS", "feature_id": "feat_target", "value": "0"},
        {"source": "lang:langC", "target": "fval:feat_target:1", "type": "OBSERVED_AS", "feature_id": "feat_target", "value": "1"},
        {"source": "lang:langC", "target": "fval:anchor_feat:1", "type": "OBSERVED_AS", "feature_id": "anchor_feat", "value": "1"},
    ]
    nodes_path = tmp_path / "kg_nodes.jsonl"
    edges_path = tmp_path / "kg_edges.jsonl"
    nodes_path.write_text("".join(json.dumps(node) + "\n" for node in nodes), encoding="utf-8")
    edges_path.write_text("".join(json.dumps(edge) + "\n" for edge in edges), encoding="utf-8")

    graph = loader.load_kg(nodes_path, edges_path)
    # Reference observations come from the masked typ_df state: target value hidden.
    records = retrieval.ranked_phylo_records_typed(
        graph,
        language="langA",
        target_feature="feat_target",
        correlated=["anchor_feat"],
        pool_limit=3,
        reference_observations={"anchor_feat": "1"},
    )
    assert [rec["glottocode"] for rec in records[:2]] == ["langC", "langB"]
