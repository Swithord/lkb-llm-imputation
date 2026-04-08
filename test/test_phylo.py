import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from phylo_neighbors import phylo_neighbors


def _build_tree():
    # Tree:
    # root
    #  ├─ langA
    #  │   └─ diaA1
    #  ├─ famB
    #  │   └─ langB
    #  │       └─ diaB1
    #  └─ famC
    #      └─ langC
    #          └─ diaC1
    parent = {
        "root": None,
        "famB": "root",
        "famC": "root",
        "langA": "root",
        "langB": "famB",
        "langC": "famC",
        "diaA1": "langA",
        "diaB1": "langB",
        "diaC1": "langC",
    }
    children = {
        "root": ["langA", "famB", "famC"],
        "famB": ["langB"],
        "famC": ["langC"],
        "langA": ["diaA1"],
        "langB": ["diaB1"],
        "langC": ["diaC1"],
        "diaA1": [],
        "diaB1": [],
        "diaC1": [],
    }
    level = {
        "root": "family",
        "famB": "family",
        "famC": "family",
        "langA": "language",
        "langB": "language",
        "langC": "language",
        "diaA1": "dialect",
        "diaB1": "dialect",
        "diaC1": "dialect",
    }
    return parent, children, level


def test_neighbors_basic():
    parent, children, level = _build_tree()
    # For langA, the nearest neighbor should be langB (same family, sibling branch).
    n = phylo_neighbors("langA", k_neighbors=1, parent=parent, children=children, level=level)
    assert n == ["langB"]


def test_neighbors_from_dialect():
    parent, children, level = _build_tree()
    # Dialect should normalize to its parent language.
    n = phylo_neighbors("diaA1", k_neighbors=1, parent=parent, children=children, level=level)
    assert n == ["langB"]


def test_neighbors_expand_to_next_level():
    parent, children, level = _build_tree()
    # Ask for two neighbors: langB (same family) then langC (next ancestor level).
    n = phylo_neighbors("langA", k_neighbors=2, parent=parent, children=children, level=level)
    assert n == ["langB", "langC"]


def test_k_zero():
    parent, children, level = _build_tree()
    n = phylo_neighbors("langA", k_neighbors=0, parent=parent, children=children, level=level)
    assert n == []
