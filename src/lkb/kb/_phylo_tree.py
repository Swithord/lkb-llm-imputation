from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple


def _level_id(level: Optional[Dict[str, str]], node: str) -> Optional[str]:
    if level is None:
        return None
    raw = level.get(node)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raw = getattr(raw, "id", None) or getattr(raw, "name", None) or str(raw)
    txt = raw.lower()
    if "dialect" in txt:
        return "dialect"
    if "language" in txt:
        return "language"
    return txt


def _is_dialect(level: Optional[Dict[str, str]], node: str) -> bool:
    return _level_id(level, node) == "dialect"


def _is_language(level: Optional[Dict[str, str]], node: str) -> bool:
    return _level_id(level, node) == "language"


def normalize_to_language(
    target_id: str,
    parent: Dict[str, Optional[str]],
    level: Optional[Dict[str, str]] = None,
) -> str:
    cur = target_id
    while cur is not None and not _is_language(level, cur):
        cur = parent.get(cur)
    return cur if cur is not None else target_id


def ancestor_depths(node: str, parent: Dict[str, Optional[str]]) -> Dict[str, int]:
    depths: Dict[str, int] = {}
    cur = node
    depth = 0
    while cur is not None and cur not in depths:
        depths[cur] = depth
        cur = parent.get(cur)
        depth += 1
    return depths


def shared_ancestor_depth(
    target_language: str,
    candidate_language: str,
    parent: Dict[str, Optional[str]],
) -> Optional[int]:
    target_depths = ancestor_depths(target_language, parent)
    cur = candidate_language
    while cur is not None:
        if cur in target_depths:
            return target_depths[cur]
        cur = parent.get(cur)
    return None


def relation_type(
    target_language: str,
    candidate_language: str,
    parent: Dict[str, Optional[str]],
    shared_depth: Optional[int],
) -> str:
    target_parent = parent.get(target_language)
    candidate_parent = parent.get(candidate_language)
    if target_parent is not None and target_parent == candidate_parent:
        return "same_immediate_branch"

    target_grandparent = parent.get(target_parent) if target_parent is not None else None
    candidate_grandparent = parent.get(candidate_parent) if candidate_parent is not None else None
    if (
        target_parent is not None
        and candidate_parent is not None
        and target_grandparent is not None
        and target_grandparent == candidate_grandparent
        and target_parent != candidate_parent
    ):
        return "sibling_branch"

    if shared_depth is not None and shared_depth <= 3:
        return "nearby_cousin_branch"
    return "higher_shared_ancestor"


def phylo_neighbor_records(
    target_id: str,
    k_neighbors: int,
    parent: Dict[str, Optional[str]],
    children: Dict[str, Sequence[str]],
    level: Optional[Dict[str, str]] = None,
    allowed_nodes: Optional[Set[str]] = None,
) -> List[dict]:
    if k_neighbors <= 0:
        return []

    lnode = normalize_to_language(target_id, parent=parent, level=level)
    if lnode not in children and lnode not in parent:
        return []

    allowed = allowed_nodes if allowed_nodes is not None else None
    neighbors: List[dict] = []
    seen: Set[str] = {lnode}
    queue: deque[Tuple[str, int]] = deque([(lnode, 0)])

    while queue and len(neighbors) < k_neighbors:
        u, dist = queue.popleft()
        nxt: List[str] = []
        p = parent.get(u)
        if p is not None:
            nxt.append(p)
        nxt.extend(children.get(u, []))
        nxt.sort()

        for v in nxt:
            if v in seen or _is_dialect(level, v):
                continue
            seen.add(v)
            queue.append((v, dist + 1))

            if not _is_language(level, v):
                continue
            if allowed is not None and v not in allowed:
                continue
            shared_depth = shared_ancestor_depth(lnode, v, parent)
            neighbors.append(
                {
                    "glottocode": v,
                    "tree_distance": dist + 1,
                    "shared_ancestor_depth": shared_depth,
                    "relation_type": relation_type(lnode, v, parent, shared_depth),
                }
            )
            if len(neighbors) == k_neighbors:
                break

    return neighbors


def phylo_neighbors(
    target_id: str,
    k_neighbors: int,
    parent: Dict[str, Optional[str]],
    children: Dict[str, Sequence[str]],
    level: Optional[Dict[str, str]] = None,
    allowed_nodes: Optional[Set[str]] = None,
) -> List[str]:
    return [
        str(rec["glottocode"])
        for rec in phylo_neighbor_records(
            target_id,
            k_neighbors=k_neighbors,
            parent=parent,
            children=children,
            level=level,
            allowed_nodes=allowed_nodes,
        )
    ]


def build_glottolog_tree_maps(glottolog) -> Tuple[Dict[str, Optional[str]], Dict[str, List[str]], Dict[str, str]]:
    if glottolog is None:
        raise RuntimeError("glottolog object is required.")
    if not hasattr(glottolog, "languoids"):
        raise RuntimeError("Glottolog object does not expose languoids().")

    parent: Dict[str, Optional[str]] = {}
    children: Dict[str, List[str]] = {}
    level: Dict[str, str] = {}
    for languoid in glottolog.languoids():
        lid = getattr(languoid, "id", None)
        if lid is None:
            continue
        plevel = getattr(languoid, "level", None)
        if plevel is not None:
            level[lid] = (
                getattr(plevel, "id", None)
                or getattr(plevel, "name", None)
                or str(plevel)
            )
        p = getattr(languoid, "parent", None)
        pid = getattr(p, "id", None) if p is not None else None
        parent[lid] = pid
        if pid is not None:
            children.setdefault(pid, []).append(lid)
        children.setdefault(lid, [])
    return parent, children, level
