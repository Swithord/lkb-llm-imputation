"""KG prompt strategy (v5 glottolog-tree variants).

Four versions:
  - v5_glottolog_tree_json (default, full evidence)
  - v5_glottolog_tree_compact_json (compact phylo block)
  - v5_glottolog_tree_contrast_json (contrastive framing, no detailed block)
  - v5_glottolog_tree_retrieval_only_json (no anchors / clues / prior)

Retrieval is performed by an injected ``KGRetriever`` (flat, typed,
typed+contrastive, or hybrid). ICL-style helpers (anchors, clues, vote table,
selection) are reused here; the only substantive difference from ICLPrompt is
the retrieval backend, neighbor block formatting (with relation_type/tree_distance),
and per-version template variations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from lkb.impute.prompts.icl import (
    ICLPrompt,
    _extract_json_fields,
    _haversine_km,
    _normalize_confidence,
    _normalize_rationale,
    _normalize_value,
)
from lkb.interfaces import KGRetriever, KnowledgeBase, Prediction, Prompt, PromptPayload


_SUPPORTED_VERSIONS = {
    "v5_glottolog_tree_json",
    "v5_glottolog_tree_compact_json",
    "v5_glottolog_tree_contrast_json",
    "v5_glottolog_tree_retrieval_only_json",
}


_COMPACT_RELATION_LABELS = {
    "same_immediate_branch": "same branch",
    "sibling_branch": "sibling branch",
    "nearby_cousin_branch": "cousin branch",
    "higher_shared_ancestor": "higher ancestor",
    "phylogenetic_neighbor": "nearby branch",
    "phylogenetic_fallback": "fallback",
}


def _coerce_non_negative_int(value, default: int) -> int:
    try:
        out = int(value)
    except Exception:
        return default
    return out if out >= 0 else default


def _relation_label(value: object) -> str:
    if value is None:
        return "unknown"
    return str(value).replace("_", " ")


def _compact_relation_label(value: object) -> str:
    relation = str(value or "unknown")
    return _COMPACT_RELATION_LABELS.get(relation, relation.replace("_", " "))


def _format_shared_ancestor_depth(value: object) -> str:
    if value is None or value == "":
        return "unknown"
    try:
        return str(int(value))
    except Exception:
        return str(value)


class KGPrompt(Prompt):
    """Glottolog-tree KG prompt (v5 variants) with injected retriever."""

    name = "kg"

    def __init__(
        self,
        retriever: KGRetriever,
        *,
        version: str = "v5_glottolog_tree_json",
        top_n_features: int = 10,
        include_vote_table: bool = True,
        fallback_top_n: int = 15,
        min_clue_support: int = 12,
        max_clues: int = 2,
        phylo_pool_limit: int = 400,
        geo_pool_limit: int = 1200,
    ) -> None:
        if version not in _SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported KGPrompt version: {version}")
        self.version = version
        self.retriever = retriever
        self.include_vote_table = include_vote_table
        self.phylo_pool_limit = phylo_pool_limit
        self.geo_pool_limit = geo_pool_limit

        # Reuse ICLPrompt helpers (anchors, clues, selection, votes, priors).
        self._icl = ICLPrompt(
            top_n_features=top_n_features,
            include_vote_table=include_vote_table,
            fallback_top_n=fallback_top_n,
            min_clue_support=min_clue_support,
            max_clues=max_clues,
        )

    # ---- helpers ------------------------------------------------------------

    def _is_compact(self) -> bool:
        return self.version == "v5_glottolog_tree_compact_json"

    def _is_contrast(self) -> bool:
        return self.version == "v5_glottolog_tree_contrast_json"

    def _is_retrieval_only(self) -> bool:
        return self.version == "v5_glottolog_tree_retrieval_only_json"

    def _contrastive_force_include(
        self, kb: KnowledgeBase, candidates: Sequence[str], feature: str
    ) -> List[str]:
        ordered: List[str] = []
        seen: set[str] = set()
        for desired in (1, 0):
            nb = self._icl._nearest_with_value(kb, candidates, feature, desired)
            if nb is None or nb in seen:
                continue
            seen.add(nb)
            ordered.append(nb)
        return ordered

    def _format_phylo_neighbor_block(
        self,
        kb: KnowledgeBase,
        neighbors: Sequence[str],
        phylo_record_map: Dict[str, Dict[str, object]],
        target_feature: str,
        correlated: Sequence[str],
        compact: bool = False,
    ) -> List[str]:
        lines: List[str] = []
        for idx, nb in enumerate(neighbors, start=1):
            label = self._icl._language_label(kb, nb)
            rec = phylo_record_map.get(str(nb), {})
            tree_distance = _coerce_non_negative_int(rec.get("tree_distance"), idx)
            if compact:
                relation = _compact_relation_label(rec.get("relation_type"))
                lines.append(f"{idx}) {label} (relation={relation}, tree_distance={tree_distance}):")
            else:
                relation = _relation_label(rec.get("relation_type"))
                shared_depth = _format_shared_ancestor_depth(rec.get("shared_ancestor_depth"))
                lines.append(
                    f"{idx}) {label} (glottocode={nb}, relation={relation}, "
                    f"tree_distance={tree_distance}, shared_ancestor_depth={shared_depth}):"
                )
            facts = self._icl._collect_neighbor_facts(
                kb, nb, target_feature, correlated, limit=4, max_correlated_fallback=3
            )
            if facts:
                for feat, value in facts:
                    lines.append(f"- {feat}: {value}")
            else:
                lines.append("- No observed target or anchor facts available.")
        return lines

    def _nearest_phylo_supporting_neighbor(
        self,
        kb: KnowledgeBase,
        candidates: Sequence[str],
        phylo_record_map: Dict[str, Dict[str, object]],
        feature: str,
        desired_value: int,
    ) -> Optional[str]:
        for idx, nb in enumerate(candidates, start=1):
            if kb.value(nb, feature) != desired_value:
                continue
            rec = phylo_record_map.get(str(nb), {})
            tree_distance = _coerce_non_negative_int(rec.get("tree_distance"), idx)
            relation = _relation_label(rec.get("relation_type"))
            label = self._icl._language_label(kb, nb)
            return f"{label} (relation={relation}, tree_distance={tree_distance})"
        return None

    def _supporting_phylo_summaries(
        self,
        kb: KnowledgeBase,
        neighbors: Sequence[str],
        phylo_record_map: Dict[str, Dict[str, object]],
        feature: str,
        desired_value: int,
        max_items: int = 2,
    ) -> List[str]:
        summaries: List[str] = []
        for idx, nb in enumerate(neighbors, start=1):
            if kb.value(nb, feature) != desired_value:
                continue
            rec = phylo_record_map.get(str(nb), {})
            relation = _compact_relation_label(rec.get("relation_type"))
            tree_distance = _coerce_non_negative_int(rec.get("tree_distance"), idx)
            summaries.append(f"{self._icl._language_label(kb, nb)} ({relation}, d={tree_distance})")
            if len(summaries) >= max_items:
                break
        return summaries

    def _supporting_geo_summaries(
        self,
        kb: KnowledgeBase,
        language: str,
        neighbors: Sequence[str],
        feature: str,
        desired_value: int,
        max_items: int = 2,
    ) -> List[str]:
        summaries: List[str] = []
        lat0, lon0 = self._icl._latlon(kb, language)
        for nb in neighbors:
            if kb.value(nb, feature) != desired_value:
                continue
            label = self._icl._language_label(kb, nb)
            if lat0 is not None and lon0 is not None:
                lat1, lon1 = self._icl._latlon(kb, nb)
                if lat1 is not None and lon1 is not None:
                    summaries.append(f"{label} ({_haversine_km(lat0, lon0, lat1, lon1):.1f} km)")
                else:
                    summaries.append(label)
            else:
                summaries.append(label)
            if len(summaries) >= max_items:
                break
        return summaries

    # ---- Prompt interface ---------------------------------------------------

    def build(self, kb: KnowledgeBase, language: str, feature: str) -> PromptPayload:
        self._icl._ensure_cache(kb)
        meta = kb.metadata_for(language)

        lang_name = meta.name or language
        iso = meta.iso639_3 or "None"
        family = meta.family_name or ""
        parent = meta.parent_name or ""
        if not family and not parent:
            lineage = "Isolate"
        elif family and parent and parent != family:
            lineage = f"{family} > {parent}"
        else:
            lineage = family or parent or "Isolate"
        macro = ";".join(meta.macroareas) if meta.macroareas else "None"
        lat = "None" if meta.latitude is None else str(meta.latitude)
        lon = "None" if meta.longitude is None else str(meta.longitude)

        user_lines: List[str] = [
            "Target language:",
            f"- Name: {lang_name}",
            f"- Glottocode: {language}",
            f"- ISO639-3: {iso}",
            f"- Family lineage: {lineage}",
            f"- Macro-area: {macro}",
            f"- Location: latitude={lat}, longitude={lon}",
        ]

        correlated = self._icl._effective_correlated(kb, language, feature)

        phylo_records = self.retriever.phylo_records(
            kb, language, feature, correlated, pool_limit=self.phylo_pool_limit
        )
        phylo_candidates = [str(rec["glottocode"]) for rec in phylo_records]
        phylo_record_map = {str(rec["glottocode"]): rec for rec in phylo_records}

        geo_candidates = self.retriever.geo_candidates(
            kb, language, feature, correlated, pool_limit=self.geo_pool_limit
        )

        contrastive = self.retriever.backend in {"kg_typed_contrastive", "hybrid_flat_kg"}
        if contrastive:
            phylo_force = self._contrastive_force_include(kb, phylo_candidates, feature)
            geo_force = self._contrastive_force_include(kb, geo_candidates, feature)
        else:
            phylo_yes = self._icl._nearest_with_value(kb, phylo_candidates, feature, 1)
            geo_yes = self._icl._nearest_with_value(kb, geo_candidates, feature, 1)
            phylo_force = [phylo_yes] if phylo_yes else []
            geo_force = [geo_yes] if geo_yes else []

        phylo_k = self._icl._neighbor_k(len(kb.genetic_neighbors(language)))
        geo_k = self._icl._neighbor_k(len(kb.geographic_neighbors(language)))
        phylo_neighbors = self._icl._select_neighbors(
            kb, phylo_candidates, correlated, feature, phylo_k,
            force_include=phylo_force, reference_language=language,
        )
        geo_neighbors = self._icl._select_neighbors(
            kb, geo_candidates, correlated, feature, geo_k,
            force_include=geo_force, reference_language=language,
        )

        anchors = self._icl._observed_anchor_facts(kb, language, feature)
        clues, clue_summary = self._icl._collect_clues(kb, language, feature)
        prior_value, prior_ratio = self._icl._feature_prevalence_prior(kb, feature)
        pv = self._icl._count_votes(kb, phylo_neighbors, feature)
        gv = self._icl._count_votes(kb, geo_neighbors, feature)
        overall = {
            "yes": pv["yes"] + gv["yes"],
            "no": pv["no"] + gv["no"],
            "missing": pv["missing"] + gv["missing"],
        }
        p_denom = pv["yes"] + pv["no"]
        g_denom = gv["yes"] + gv["no"]
        ov_denom = overall["yes"] + overall["no"]
        p_ratio = (pv["yes"] / p_denom) if p_denom else 0.0
        g_ratio = (gv["yes"] / g_denom) if g_denom else 0.0
        ov_ratio = (overall["yes"] / ov_denom) if ov_denom else 0.0
        agreement = (max(overall["yes"], overall["no"]) / ov_denom) if ov_denom else 0.0

        phylo_yes_nb = self._nearest_phylo_supporting_neighbor(
            kb, phylo_candidates, phylo_record_map, feature, 1
        )
        phylo_no_nb = self._nearest_phylo_supporting_neighbor(
            kb, phylo_candidates, phylo_record_map, feature, 0
        )
        geo_yes_nb = self._icl._nearest_supporting_neighbor(
            kb, language, geo_candidates, feature, 1, "geographic"
        )
        geo_no_nb = self._icl._nearest_supporting_neighbor(
            kb, language, geo_candidates, feature, 0, "geographic"
        )

        if self._is_contrast():
            user_lines.append("Observed typological facts (anchor features):")
            if anchors:
                for feat_name, feat_value in anchors:
                    user_lines.append(f"- {feat_name}: {feat_value} (observed)")
            else:
                user_lines.append("- (no observed anchor facts)")

            yes_phylo = self._supporting_phylo_summaries(kb, phylo_neighbors, phylo_record_map, feature, 1)
            no_phylo = self._supporting_phylo_summaries(kb, phylo_neighbors, phylo_record_map, feature, 0)
            yes_geo = self._supporting_geo_summaries(kb, language, geo_neighbors, feature, 1)
            no_geo = self._supporting_geo_summaries(kb, language, geo_neighbors, feature, 0)
            yes_clues = [c for c in clues if c.get("majority") == "yes"][:2]
            no_clues = [c for c in clues if c.get("majority") == "no"][:2]

            user_lines.append(f"Prompt version: {self.version}")
            user_lines.append("Task:")
            user_lines.append("Predict the missing value for the following feature:")
            user_lines.append(f"- Feature: {feature}")
            user_lines.append("- Allowed values: 0 | 1")
            user_lines.append("Competing evidence for value 1:")
            user_lines.append(f"- Closest phylogenetic support for 1: {phylo_yes_nb or 'none observed'}")
            user_lines.append(f"- Closest geographic support for 1: {geo_yes_nb or 'none observed'}")
            user_lines.append(
                "- Selected phylogenetic/geographic 1-neighbors: "
                + (", ".join(yes_phylo + yes_geo) if (yes_phylo or yes_geo) else "none in selected evidence")
            )
            if yes_clues:
                for clue in yes_clues:
                    user_lines.append(
                        f"- Clue for 1: {clue['feature']}={clue['value']} -> "
                        f"{clue['yes']} yes / {clue['no']} no"
                    )
            else:
                user_lines.append("- Clue for 1: no strong correlated clue leaning toward 1")

            user_lines.append("Competing evidence for value 0:")
            user_lines.append(f"- Closest phylogenetic support for 0: {phylo_no_nb or 'none observed'}")
            user_lines.append(f"- Closest geographic support for 0: {geo_no_nb or 'none observed'}")
            user_lines.append(
                "- Selected phylogenetic/geographic 0-neighbors: "
                + (", ".join(no_phylo + no_geo) if (no_phylo or no_geo) else "none in selected evidence")
            )
            if no_clues:
                for clue in no_clues:
                    user_lines.append(
                        f"- Clue for 0: {clue['feature']}={clue['value']} -> "
                        f"{clue['yes']} yes / {clue['no']} no"
                    )
            else:
                user_lines.append("- Clue for 0: no strong correlated clue leaning toward 0")

            if self.include_vote_table:
                user_lines.append("Secondary vote snapshot:")
                user_lines.append(f"- Genetic votes: {pv['yes']} yes / {pv['no']} no / {pv['missing']} unk")
                user_lines.append(f"- Geographic votes: {gv['yes']} yes / {gv['no']} no / {gv['missing']} unk")
                user_lines.append(f"- Overall votes: {overall['yes']} yes / {overall['no']} no")
                user_lines.append(f"- Weak prevalence prior: {prior_value} ({prior_ratio:.0%} of observed)")

            user_lines.append("Reasoning guidance:")
            user_lines.append("- Compare which side has stronger, closer, and more consistent evidence.")
            user_lines.append("- Use anchor features and closest neighbors first; use votes only as secondary evidence.")
            user_lines.append("- Prefer closer genealogical evidence from the same branch or sibling branches before higher shared ancestors.")
            user_lines.append("- A minority value is acceptable if its evidence is clearly stronger.")
            user_lines.append("- Use prevalence only as a weak tie-breaker when the two sides are otherwise balanced.")
            _append_output_format(user_lines)
            _append_contrast_examples(user_lines)

        elif self._is_retrieval_only():
            user_lines.append("Glottolog-tree retrieved evidence (detailed evidence):")
            user_lines.extend(
                self._format_phylo_neighbor_block(
                    kb, phylo_neighbors, phylo_record_map, feature, correlated, compact=False
                )
            )
            user_lines.append("Selected geographic neighbors (detailed evidence):")
            user_lines.extend(
                self._icl._format_neighbor_block(
                    kb, language, geo_neighbors, geo_candidates, feature, correlated, mode="geographic"
                )
            )
            if self.include_vote_table:
                user_lines.append("Target-feature vote counts (useful but not decisive):")
                user_lines.append(
                    f"- Genetic vote: {pv['yes']} yes / {pv['no']} no / {pv['missing']} unk ({p_ratio:.0%} yes)"
                )
                user_lines.append(
                    f"- Geo vote: {gv['yes']} yes / {gv['no']} no / {gv['missing']} unk ({g_ratio:.0%} yes)"
                )
                user_lines.append(
                    f"- Overall observed votes: {overall['yes']} yes / {overall['no']} no "
                    f"({ov_ratio:.0%} yes; agreement={agreement:.0%})"
                )
                user_lines.append(
                    f"- Vote evidence coverage: {ov_denom} observed target-feature votes "
                    f"(unknown ignored in decision)."
                )

            user_lines.append(f"Prompt version: {self.version}")
            user_lines.append("Task:")
            user_lines.append("Predict the missing value for the following feature:")
            user_lines.append(f"- Feature: {feature}")
            user_lines.append("- Allowed values: 0 | 1")
            user_lines.append("Reasoning guidance:")
            user_lines.append("- Use only the Glottolog-tree retrieved phylogenetic and geographic evidence shown here.")
            user_lines.append("- Compare the support for value 0 versus value 1.")
            user_lines.append("- Prefer closer genealogical evidence from the same branch or sibling branches before higher shared ancestors.")
            user_lines.append("- Neighbor counts are useful, but do not follow majority vote blindly.")
            user_lines.append("- A smaller number of closer or more relevant neighbors may outweigh a larger but weaker group.")
            user_lines.append("- Do not rely on anchor facts, correlated clues, or prevalence priors.")
            _append_output_format(user_lines)
            _append_retrieval_only_examples(user_lines)

        else:  # v5_glottolog_tree_json or v5_glottolog_tree_compact_json
            user_lines.append("Observed typological facts (anchor features):")
            if anchors:
                for feat_name, feat_value in anchors:
                    user_lines.append(f"- {feat_name}: {feat_value} (observed)")
            else:
                user_lines.append("- (no observed anchor facts)")

            user_lines.append(
                "Glottolog-tree retrieved evidence (compact evidence):"
                if self._is_compact()
                else "Glottolog-tree retrieved evidence (detailed evidence):"
            )
            user_lines.extend(
                self._format_phylo_neighbor_block(
                    kb, phylo_neighbors, phylo_record_map, feature, correlated,
                    compact=self._is_compact(),
                )
            )
            user_lines.append("Selected geographic neighbors (detailed evidence):")
            user_lines.extend(
                self._icl._format_neighbor_block(
                    kb, language, geo_neighbors, geo_candidates, feature, correlated,
                    mode="geographic",
                )
            )

            if self.include_vote_table:
                user_lines.append("Target-feature vote counts (useful but not decisive):")
                user_lines.append(
                    f"- Genetic vote: {pv['yes']} yes / {pv['no']} no / {pv['missing']} unk ({p_ratio:.0%} yes)"
                )
                user_lines.append(
                    f"- Geo vote: {gv['yes']} yes / {gv['no']} no / {gv['missing']} unk ({g_ratio:.0%} yes)"
                )
                user_lines.append(
                    f"- Overall observed votes: {overall['yes']} yes / {overall['no']} no "
                    f"({ov_ratio:.0%} yes; agreement={agreement:.0%})"
                )
                user_lines.append(
                    f"- Vote evidence coverage: {ov_denom} observed target-feature votes "
                    f"(unknown ignored in decision)."
                )
                user_lines.append(
                    f"- Weak prevalence prior (tie-breaker only): value={prior_value} "
                    f"({prior_ratio:.0%} of observed)"
                )

            user_lines.append("Nearest contrastive neighbor evidence:")
            user_lines.append(f"- Closest phylogenetic support for 1: {phylo_yes_nb or 'none observed'}")
            user_lines.append(f"- Closest phylogenetic support for 0: {phylo_no_nb or 'none observed'}")
            user_lines.append(f"- Closest geographic support for 1: {geo_yes_nb or 'none observed'}")
            user_lines.append(f"- Closest geographic support for 0: {geo_no_nb or 'none observed'}")

            user_lines.append("Target-specific correlated clues (compact):")
            if clues:
                for idx, clue in enumerate(clues, start=1):
                    user_lines.append(
                        f"{idx}) {clue['feature']}={clue['value']} -> target support "
                        f"{clue['yes']} yes / {clue['no']} no"
                    )
                user_lines.append(
                    f"- Correlated clues leaning: {clue_summary['yes']} yes / "
                    f"{clue_summary['no']} no / {clue_summary['tie']} tie"
                )
            else:
                user_lines.append("- No reliable correlated clues with enough support.")

            user_lines.append(f"Prompt version: {self.version}")
            user_lines.append("Task:")
            user_lines.append("Predict the missing value for the following feature:")
            user_lines.append(f"- Feature: {feature}")
            user_lines.append("- Allowed values: 0 | 1")
            user_lines.append("Reasoning guidance:")
            user_lines.append("- Compare the support for value 0 versus value 1.")
            user_lines.append(
                "- Weigh observed anchor features, Glottolog-tree relations, geographic evidence, and correlated clues together."
            )
            user_lines.append("- Prefer closer genealogical evidence from the same branch or sibling branches before higher shared ancestors.")
            user_lines.append("- Neighbor counts are useful, but do not follow majority vote blindly.")
            user_lines.append("- A smaller number of closer or more relevant neighbors may outweigh a larger but weaker group.")
            user_lines.append("- It is acceptable to predict a minority value if it is better supported by the overall evidence.")
            user_lines.append(
                f"- Use feature prevalence only as a weak tie-breaker when evidence is otherwise balanced (prior={prior_value})."
            )
            _append_output_format(user_lines)
            _append_tree_examples(user_lines)

        system = _SYSTEM_MESSAGE
        return PromptPayload(
            system=system,
            user="\n".join(user_lines),
            meta={
                "language": language,
                "feature": feature,
                "version": self.version,
                "backend": self.retriever.backend,
                "phylo_neighbors": list(phylo_neighbors),
                "geo_neighbors": list(geo_neighbors),
            },
        )

    def parse(self, raw: str) -> Prediction:
        value, confidence, rationale = _extract_json_fields(raw)
        value = _normalize_value(value, raw, ["0", "1"])
        conf = _normalize_confidence(confidence)
        rat = _normalize_rationale(rationale)
        return Prediction(
            value=value,
            confidence=conf,
            rationale=rat,
            parsed_ok=value is not None,
            raw=raw,
        )


_SYSTEM_MESSAGE = (
    "You are a linguistics expert specializing in typology.\n"
    "Infer missing typological features using:\n"
    "(1) observed facts about the target language,\n"
    "(2) evidence from phylogenetic and geographic neighbors,\n"
    "and (3) well-established linguistic universals.\n"
    "Compare the evidence for value 0 versus value 1.\n"
    "Neighbor counts are useful, but do not follow majority vote blindly.\n"
    "A smaller number of closer or more relevant neighbors may outweigh a larger number of weaker neighbors.\n"
    "It is acceptable to predict a minority value if it is better supported by the target language's observed properties and closest evidence.\n"
    "Use feature prevalence only as a weak tie-breaker when the evidence is otherwise balanced."
)


def _append_output_format(lines: List[str]) -> None:
    lines.append("Output format (STRICT JSON):")
    lines.append("Output ONLY valid JSON.")
    lines.append("Return exactly one minified JSON object on one line with keys: value, confidence, rationale.")
    lines.append("- value: one of the allowed values above")
    lines.append("- confidence: low, medium, or high")
    lines.append("- rationale: at most 20 words")
    lines.append("No Markdown, no prose, no code fences, no trailing text.")
    lines.append("Few-shot examples:")


def _append_tree_examples(lines: List[str]) -> None:
    lines.append(
        '{"value":"1","confidence":"medium","rationale":"Most distant languages favor 0, but the closest branch evidence supports 1."}'
    )
    lines.append(
        '{"value":"1","confidence":"high","rationale":"Closest branch evidence and observed features align strongly with value 1."}'
    )
    lines.append(
        '{"value":"0","confidence":"low","rationale":"Evidence is balanced, so the weak prevalence prior favors 0."}'
    )


def _append_contrast_examples(lines: List[str]) -> None:
    lines.append(
        '{"value":"1","confidence":"medium","rationale":"Closer branch and geographic evidence for 1 outweigh broader support for 0."}'
    )
    lines.append(
        '{"value":"0","confidence":"medium","rationale":"Closer evidence for 0 is stronger despite a few supporting 1 neighbors."}'
    )
    lines.append(
        '{"value":"0","confidence":"low","rationale":"Both sides are weak, so the prevalence prior favors 0."}'
    )


def _append_retrieval_only_examples(lines: List[str]) -> None:
    lines.append(
        '{"value":"1","confidence":"medium","rationale":"Closer retrieved branch and geographic evidence support 1 more strongly than 0."}'
    )
    lines.append(
        '{"value":"0","confidence":"medium","rationale":"Retrieved phylogenetic and geographic evidence consistently favors 0."}'
    )
    lines.append(
        '{"value":"0","confidence":"low","rationale":"Retrieved evidence is weak and mixed, so 0 is a cautious choice."}'
    )
