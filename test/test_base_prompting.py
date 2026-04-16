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


def _build_module():
    module = _load_module("base_prompting_test", "code/prompting.py")
    module.typ_df = module.pd.DataFrame(
        {
            "feat_target": [-1, 1, 0],
            "anchor_feat": [1, 1, 0],
            "clue_feat": [1, 1, 0],
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
    module.geographic_neighbours = {
        "langA": ["langB", "langC"],
        "langB": ["langA", "langC"],
        "langC": ["langA", "langB"],
    }
    module.top_n_features = 2
    module.topk_map = {
        "feat_target": ["anchor_feat", "clue_feat"],
        "anchor_feat": ["feat_target"],
        "clue_feat": ["feat_target"],
    }
    module.clue_support_cache = {}
    return module


def test_v4_rag_only_prompt_excludes_non_rag_extras():
    module = _build_module()
    module.set_prompt_options("v4_rag_only_json", False)

    _, user = module.construct_prompt("langA", "feat_target")

    assert "Prompt version: v4_rag_only_json" in user
    assert "Selected phylogenetic neighbors (detailed evidence):" in user
    assert "Selected geographic neighbors (detailed evidence):" in user
    assert "Observed typological facts (anchor features):" not in user
    assert "Nearest contrastive neighbor evidence:" not in user
    assert "Target-specific correlated clues (compact):" not in user
    assert "Weak prevalence prior" not in user
    assert "Target-feature vote counts (useful but not decisive):" not in user
    assert "Use only the retrieved phylogenetic and geographic neighbor evidence shown here." in user


def test_v4_rag_vote_prompt_adds_votes_but_not_other_extras():
    module = _build_module()
    module.set_prompt_options("v4_rag_vote_json", True)

    _, user = module.construct_prompt("langA", "feat_target")

    assert "Prompt version: v4_rag_vote_json" in user
    assert "Target-feature vote counts (useful but not decisive):" in user
    assert "Overall observed votes:" in user
    assert "Observed typological facts (anchor features):" not in user
    assert "Nearest contrastive neighbor evidence:" not in user
    assert "Target-specific correlated clues (compact):" not in user
    assert "Weak prevalence prior" not in user
