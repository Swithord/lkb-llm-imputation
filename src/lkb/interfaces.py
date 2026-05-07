"""Central interfaces: ABCs and shared dataclasses."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from lkb.kb.uriel_plus import URIELPlus


@dataclass(frozen=True)
class Prediction:
    value: Optional[str]
    confidence: Optional[str] = None
    rationale: Optional[str] = None
    parsed_ok: bool = True
    raw: Optional[str] = None


@dataclass(frozen=True)
class PromptPayload:
    system: str
    user: str
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LanguageMeta:
    glottocode: str
    name: Optional[str] = None
    family_id: Optional[str] = None
    family_name: Optional[str] = None
    parent_id: Optional[str] = None
    parent_name: Optional[str] = None
    iso639_3: Optional[str] = None
    level: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    macroareas: tuple[str, ...] = ()
    countries: tuple[str, ...] = ()


@dataclass(frozen=True)
class NeighborRecord:
    glottocode: str
    rank: int
    tree_distance: Optional[float] = None
    shared_ancestor_depth: Optional[int] = None
    relation_type: Optional[str] = None
    km: Optional[float] = None


class KnowledgeBase(ABC):
    @property
    @abstractmethod
    def languages(self) -> Sequence[str]: ...

    @property
    @abstractmethod
    def features(self) -> Sequence[str]: ...

    @abstractmethod
    def value(self, language: str, feature: str) -> Optional[int]: ...

    @abstractmethod
    def is_observed(self, language: str, feature: str) -> bool: ...

    @abstractmethod
    def as_matrix(self) -> np.ndarray: ...

    @abstractmethod
    def observed_mask(self) -> np.ndarray: ...


class Prompt(ABC):
    name: str
    version: str

    @abstractmethod
    def build(self, kb: "URIELPlus", language: str, feature: str) -> PromptPayload: ...

    @abstractmethod
    def parse(self, raw: str) -> Prediction: ...


class KGRetriever(ABC):
    backend: str

    @abstractmethod
    def phylo_records(
        self,
        kb: "URIELPlus",
        language: str,
        feature: str,
        correlated: Sequence[str],
        pool_limit: int = 400,
    ) -> list[dict]: ...

    @abstractmethod
    def geo_candidates(
        self,
        kb: "URIELPlus",
        language: str,
        feature: str,
        correlated: Sequence[str],
        pool_limit: int = 1200,
    ) -> list[str]: ...


class Imputer(ABC):
    name: str

    @abstractmethod
    def fit(self, kb: KnowledgeBase) -> None: ...

    @abstractmethod
    def impute(
        self, kb: KnowledgeBase, pairs: Sequence[tuple[str, str]]
    ) -> list[Prediction]: ...


class LLMClient(ABC):
    @abstractmethod
    def complete(self, payloads: Sequence[PromptPayload]) -> list[str]: ...
