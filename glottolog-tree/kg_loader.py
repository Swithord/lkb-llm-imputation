from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional


def _read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


class KGGraph:
    def __init__(
        self,
        *,
        nodes_by_id: Dict[str, dict],
        edges: List[dict],
        outgoing_all: Dict[str, List[dict]],
        outgoing_by_type: Dict[str, Dict[str, List[dict]]],
        incoming_by_type: Dict[str, Dict[str, List[dict]]],
        glottocode_to_language_id: Dict[str, str],
        feature_to_id: Dict[str, str],
        language_order: List[str],
        language_observations: Dict[str, Dict[str, str]],
        feature_correlations: Dict[str, List[str]],
    ) -> None:
        self.nodes_by_id = nodes_by_id
        self.edges = edges
        self.outgoing_all = outgoing_all
        self.outgoing_by_type = outgoing_by_type
        self.incoming_by_type = incoming_by_type
        self.glottocode_to_language_id = glottocode_to_language_id
        self.feature_to_id = feature_to_id
        self.language_order = language_order
        self.language_observations = language_observations
        self.feature_correlations = feature_correlations

    def node(self, node_id: str) -> Optional[dict]:
        return self.nodes_by_id.get(node_id)

    def language_node(self, glottocode: str) -> Optional[dict]:
        node_id = self.glottocode_to_language_id.get(str(glottocode))
        if node_id is None:
            return None
        return self.nodes_by_id.get(node_id)

    def all_language_codes(self) -> List[str]:
        return list(self.language_order)

    def outgoing(self, node_id: str, edge_type: str | None = None) -> List[dict]:
        if edge_type is None:
            return list(self.outgoing_all.get(node_id, []))
        return list(self.outgoing_by_type.get(edge_type, {}).get(node_id, []))

    def incoming(self, node_id: str, edge_type: str | None = None) -> List[dict]:
        if edge_type is None:
            merged: List[dict] = []
            for by_node in self.incoming_by_type.values():
                merged.extend(by_node.get(node_id, []))
            return merged
        return list(self.incoming_by_type.get(edge_type, {}).get(node_id, []))


def build_kg_graph(nodes: Iterable[dict], edges: Iterable[dict]) -> KGGraph:
    node_list = list(nodes)
    edge_list = list(edges)
    nodes_by_id = {str(node["id"]): node for node in node_list}
    outgoing_all: DefaultDict[str, List[dict]] = defaultdict(list)
    outgoing_by_type: DefaultDict[str, DefaultDict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    incoming_by_type: DefaultDict[str, DefaultDict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    glottocode_to_language_id: Dict[str, str] = {}
    feature_to_id: Dict[str, str] = {}
    language_with_order: List[tuple[int, str]] = []
    language_id_to_glottocode: Dict[str, str] = {}
    feature_id_to_name: Dict[str, str] = {}

    for node in node_list:
        node_id = str(node["id"])
        node_type = str(node.get("type", ""))
        if node_type == "Language":
            glottocode = str(node.get("glottocode", "")).strip()
            if glottocode:
                glottocode_to_language_id[glottocode] = node_id
                language_id_to_glottocode[node_id] = glottocode
                try:
                    order_index = int(node.get("order_index", len(language_with_order)))
                except Exception:
                    order_index = len(language_with_order)
                language_with_order.append((order_index, glottocode))
        elif node_type == "Feature":
            feature_id = str(node.get("feature_id", "")).strip()
            if feature_id:
                feature_to_id[feature_id] = node_id
                feature_id_to_name[node_id] = feature_id

    language_observations: DefaultDict[str, Dict[str, str]] = defaultdict(dict)
    feature_correlations: DefaultDict[str, List[tuple[int, str]]] = defaultdict(list)
    for edge in edge_list:
        source = str(edge["source"])
        target = str(edge["target"])
        edge_type = str(edge.get("type", ""))
        outgoing_all[source].append(edge)
        outgoing_by_type[edge_type][source].append(edge)
        incoming_by_type[edge_type][target].append(edge)
        if edge_type == "OBSERVED_AS":
            glottocode = language_id_to_glottocode.get(source)
            feature_id = str(edge.get("feature_id") or "")
            value = edge.get("value")
            if glottocode and feature_id and value is not None:
                language_observations[glottocode][feature_id] = str(value)
        elif edge_type == "FEATURE_CORRELATED":
            source_feature = feature_id_to_name.get(source)
            target_feature = feature_id_to_name.get(target)
            if source_feature and target_feature:
                try:
                    rank = int(edge.get("rank", len(feature_correlations[source_feature]) + 1))
                except Exception:
                    rank = len(feature_correlations[source_feature]) + 1
                feature_correlations[source_feature].append((rank, target_feature))

    language_order = [gc for _, gc in sorted(language_with_order, key=lambda item: (item[0], item[1]))]
    return KGGraph(
        nodes_by_id=nodes_by_id,
        edges=edge_list,
        outgoing_all=dict(outgoing_all),
        outgoing_by_type={k: dict(v) for k, v in outgoing_by_type.items()},
        incoming_by_type={k: dict(v) for k, v in incoming_by_type.items()},
        glottocode_to_language_id=glottocode_to_language_id,
        feature_to_id=feature_to_id,
        language_order=language_order,
        language_observations={gc: dict(obs) for gc, obs in language_observations.items()},
        feature_correlations={
            feature: [other for _, other in sorted(items, key=lambda item: (item[0], item[1]))]
            for feature, items in feature_correlations.items()
        },
    )


def load_kg(nodes_path: str | Path, edges_path: str | Path) -> KGGraph:
    return build_kg_graph(_read_jsonl(Path(nodes_path)), _read_jsonl(Path(edges_path)))
