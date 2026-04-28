"""URIEL+ knowledge base: runtime loader over pre-computed artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from lkb.interfaces import KnowledgeBase, LanguageMeta, NeighborRecord
from lkb.kb._kg_graph import KGGraph, load_kg


class URIELPlus(KnowledgeBase):
    """Typological KB over URIEL+ with Glottolog metadata and KG support.

    Values in the matrix are {-1, 0, 1} where -1 means missing.
    """

    def __init__(
        self,
        *,
        matrix: np.ndarray,
        languages: Sequence[str],
        features: Sequence[str],
        metadata: Optional[pd.DataFrame] = None,
        genetic_neighbours: Optional[Mapping[str, Sequence[str]]] = None,
        genetic_neighbour_details: Optional[Mapping[str, Sequence[Mapping]]] = None,
        geographic_neighbours: Optional[Mapping[str, Sequence[str]]] = None,
        feature_correlations: Optional[Mapping[str, Sequence[str]]] = None,
        kg: Optional[KGGraph] = None,
    ) -> None:
        matrix = np.asarray(matrix)
        if matrix.ndim != 2:
            raise ValueError(f"matrix must be 2D (N, F); got {matrix.shape}")
        if matrix.shape[0] != len(languages):
            raise ValueError(
                f"matrix rows ({matrix.shape[0]}) != len(languages) ({len(languages)})"
            )
        if matrix.shape[1] != len(features):
            raise ValueError(
                f"matrix cols ({matrix.shape[1]}) != len(features) ({len(features)})"
            )

        self._matrix = matrix.astype(np.int8, copy=False)
        self._languages: List[str] = [str(x) for x in languages]
        self._features: List[str] = [str(x) for x in features]
        self._lang_to_row: Dict[str, int] = {lang: i for i, lang in enumerate(self._languages)}
        self._feat_to_col: Dict[str, int] = {feat: j for j, feat in enumerate(self._features)}

        self._metadata = metadata
        self._genetic_neighbours = {k: list(v) for k, v in (genetic_neighbours or {}).items()}
        self._genetic_neighbour_details = {
            k: [dict(r) for r in v] for k, v in (genetic_neighbour_details or {}).items()
        }
        self._geographic_neighbours = {k: list(v) for k, v in (geographic_neighbours or {}).items()}
        self._feature_correlations = {k: list(v) for k, v in (feature_correlations or {}).items()}
        self._kg = kg

    # ---- KnowledgeBase ABC --------------------------------------------------

    @property
    def languages(self) -> Sequence[str]:
        return self._languages

    @property
    def features(self) -> Sequence[str]:
        return self._features

    def value(self, language: str, feature: str) -> Optional[int]:
        row = self._lang_to_row.get(language)
        col = self._feat_to_col.get(feature)
        if row is None or col is None:
            return None
        v = int(self._matrix[row, col])
        return None if v == -1 else v

    def is_observed(self, language: str, feature: str) -> bool:
        row = self._lang_to_row.get(language)
        col = self._feat_to_col.get(feature)
        if row is None or col is None:
            return False
        return bool(self._matrix[row, col] != -1)

    def as_matrix(self) -> np.ndarray:
        return self._matrix

    def observed_mask(self) -> np.ndarray:
        return self._matrix != -1

    # ---- URIEL+ extras ------------------------------------------------------

    def observations(self, language: str) -> Dict[str, str]:
        """Observed features for a language as {feature: "0"|"1"}."""
        row = self._lang_to_row.get(language)
        if row is None:
            return {}
        out: Dict[str, str] = {}
        vals = self._matrix[row]
        for j, v in enumerate(vals):
            if v != -1:
                out[self._features[j]] = str(int(v))
        return out

    def genetic_neighbors(self, language: str) -> List[str]:
        return list(self._genetic_neighbours.get(language, []))

    def genetic_neighbor_details(self, language: str) -> List[NeighborRecord]:
        records = self._genetic_neighbour_details.get(language, [])
        out: List[NeighborRecord] = []
        for rank, rec in enumerate(records, start=1):
            gc = rec.get("glottocode")
            if gc is None:
                continue
            out.append(
                NeighborRecord(
                    glottocode=str(gc),
                    rank=int(rec.get("rank", rank)),
                    tree_distance=rec.get("tree_distance"),
                    shared_ancestor_depth=rec.get("shared_ancestor_depth"),
                    relation_type=rec.get("relation_type"),
                )
            )
        return out

    def geographic_neighbors(self, language: str) -> List[str]:
        return list(self._geographic_neighbours.get(language, []))

    def metadata_for(self, language: str) -> LanguageMeta:
        if self._metadata is None or language not in self._metadata.index:
            return LanguageMeta(glottocode=language)
        row = self._metadata.loc[language]

        def _str(key: str) -> Optional[str]:
            v = row.get(key)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            s = str(v).strip()
            return s or None

        def _float(key: str) -> Optional[float]:
            v = row.get(key)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        def _tuple(key: str) -> tuple:
            s = _str(key)
            if not s:
                return ()
            return tuple(part.strip() for part in s.split(";") if part.strip())

        return LanguageMeta(
            glottocode=language,
            name=_str("name"),
            family_id=_str("family_id"),
            family_name=_str("family_name"),
            parent_id=_str("parent_id"),
            parent_name=_str("parent_name"),
            iso639_3=_str("iso639_3"),
            level=_str("level"),
            latitude=_float("latitude"),
            longitude=_float("longitude"),
            macroareas=_tuple("macroareas"),
            countries=_tuple("countries"),
        )

    def top_correlated_features(self, feature: str) -> List[str]:
        return list(self._feature_correlations.get(feature, []))

    @property
    def kg(self) -> Optional[KGGraph]:
        return self._kg

    # ---- Loaders ------------------------------------------------------------

    @classmethod
    def from_artifacts(
        cls,
        data_root: str | Path = ".",
        *,
        typology_path: str | Path = "data/derived/uriel+_typological.csv",
        metadata_path: str | Path = "data/derived/metadata.csv",
        genetic_neighbours_path: str | Path = "data/derived/genetic_neighbours.json",
        genetic_neighbour_details_path: str | Path = "data/derived/genetic_neighbours_detailed.json",
        geographic_neighbours_path: str | Path = "data/derived/geographic_neighbours.json",
        feature_topk_path: str | Path = "data/features/topk_per_feature.csv",
        kg_nodes_path: str | Path = "artifacts/resources/kg_nodes.jsonl",
        kg_edges_path: str | Path = "artifacts/resources/kg_edges.jsonl",
        load_kg_graph: bool = True,
    ) -> "URIELPlus":
        root = Path(data_root)

        typ_df = pd.read_csv(root / typology_path, index_col=0)
        typ_df.index = typ_df.index.astype(str)
        languages = typ_df.index.tolist()
        features = [str(c) for c in typ_df.columns]
        matrix = typ_df.to_numpy(dtype=np.int8)

        metadata_df: Optional[pd.DataFrame] = None
        meta_file = root / metadata_path
        if meta_file.exists():
            metadata_df = pd.read_csv(meta_file, index_col=0)
            metadata_df.index = metadata_df.index.astype(str)

        genetic = _load_json_dict(root / genetic_neighbours_path)
        genetic_details = _load_json_dict(root / genetic_neighbour_details_path)
        geographic = _load_json_dict(root / geographic_neighbours_path)

        feature_corr: Dict[str, List[str]] = {}
        topk_file = root / feature_topk_path
        if topk_file.exists():
            topk_df = pd.read_csv(topk_file)
            for feat, sub in topk_df.groupby("feature", sort=False):
                ordered = sub.sort_values("rank")["other_feature"].astype(str).tolist()
                feature_corr[str(feat)] = ordered

        kg: Optional[KGGraph] = None
        if load_kg_graph:
            nodes_file = root / kg_nodes_path
            edges_file = root / kg_edges_path
            if nodes_file.exists() and edges_file.exists():
                kg = load_kg(nodes_file, edges_file)

        return cls(
            matrix=matrix,
            languages=languages,
            features=features,
            metadata=metadata_df,
            genetic_neighbours=genetic,
            genetic_neighbour_details=genetic_details,
            geographic_neighbours=geographic,
            feature_correlations=feature_corr,
            kg=kg,
        )


def _load_json_dict(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}
