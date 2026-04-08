from collections import deque
from typing import Dict, List, Optional, Sequence, Set


def phylo_neighbors(
    target_id: str,
    k_neighbors: int,
    parent: Dict[str, Optional[str]],
    children: Dict[str, Sequence[str]],
    level: Optional[Dict[str, str]] = None,
    allowed_nodes: Optional[Set[str]] = None,
) -> List[str]:
    """
    Return up to k_neighbors nearest language nodes by tree-edge distance.

    Node typing:
    - Uses explicit `level` labels ("language" / "dialect") from Glottolog.
    - Nodes with unknown or other levels are traversed but never emitted.
    """

    if k_neighbors <= 0:
        return []

    def level_id(n: str) -> Optional[str]:
        if level is None:
            return None
        raw = level.get(n)
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

    def is_dialect(n: str) -> bool:
        return level_id(n) == "dialect"

    def is_language(n: str) -> bool:
        return level_id(n) == "language"

    def normalize_to_language(n: str) -> str:
        # If target is a dialect or unknown node, climb to nearest language ancestor.
        cur = n
        while cur is not None and not is_language(cur):
            cur = parent.get(cur)
        return cur if cur is not None else n

    lnode = normalize_to_language(target_id)
    if lnode not in children and lnode not in parent:
        return []

    allowed = allowed_nodes if allowed_nodes is not None else None
    neighbors: List[str] = []
    seen: Set[str] = {lnode}
    queue: deque = deque([lnode])

    while queue and len(neighbors) < k_neighbors:
        u = queue.popleft()
        nxt = []
        p = parent.get(u)
        if p is not None:
            nxt.append(p)
        nxt.extend(children.get(u, []))
        nxt.sort()

        for v in nxt:
            if v in seen or is_dialect(v):
                continue
            seen.add(v)
            queue.append(v)

            if not is_language(v):
                continue
            if allowed is not None and v not in allowed:
                continue
            neighbors.append(v)
            if len(neighbors) == k_neighbors:
                break

    return neighbors
