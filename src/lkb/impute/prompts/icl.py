"""ICL prompt strategy (v4_strict_json).

Few-shot in-context prompt: target language header, observed anchor features,
selected phylogenetic + geographic neighbors with evidence, optional vote table,
contrastive nearest-neighbor evidence, correlated clues, and strict-JSON output.
"""

from __future__ import annotations

import json
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from lkb.interfaces import KnowledgeBase, Prediction, Prompt, PromptPayload


SYSTEM_MESSAGE = (
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

VALID_CONFIDENCE = {"low", "medium", "high"}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


class ICLPrompt(Prompt):
    """In-context learning prompt (v4_strict_json)."""

    name = "icl"
    version = "v4_strict_json"

    def __init__(
        self,
        *,
        top_n_features: int = 10,
        include_vote_table: bool = True,
        fallback_top_n: int = 15,
        min_clue_support: int = 12,
        max_clues: int = 2,
    ) -> None:
        self.top_n_features = top_n_features
        self.include_vote_table = include_vote_table
        self.fallback_top_n = fallback_top_n
        self.min_clue_support = min_clue_support
        self.max_clues = max_clues

        # Caches keyed by id(kb) to avoid cross-kb leakage.
        self._kb_id: Optional[int] = None
        self._geo_rank_cache: Dict[str, List[str]] = {}
        self._clue_support_cache: Dict[Tuple[str, str, str], Tuple[int, int]] = {}
        self._prior_cache: Dict[str, Tuple[str, float]] = {}

    # ---- caching helpers ----------------------------------------------------

    def _ensure_cache(self, kb: KnowledgeBase) -> None:
        if id(kb) != self._kb_id:
            self._kb_id = id(kb)
            self._geo_rank_cache = {}
            self._clue_support_cache = {}
            self._prior_cache = {}

    # ---- target anchor ------------------------------------------------------

    def _effective_correlated(self, kb: KnowledgeBase, language: str, feature: str) -> List[str]:
        ranked = kb.top_correlated_features(feature)
        features_set = set(kb.features)
        filtered = [f for f in ranked if f != feature and f in features_set]
        if not filtered or self.top_n_features <= 0:
            return filtered

        primary = filtered[: min(self.top_n_features, len(filtered))]
        observed_primary = sum(1 for f in primary if kb.is_observed(language, f))
        min_needed = 3 if self.top_n_features >= 5 else 1
        if observed_primary >= min_needed:
            return primary
        if self.fallback_top_n <= self.top_n_features:
            return primary
        extended = filtered[: min(self.fallback_top_n, len(filtered))]
        observed_extended = sum(1 for f in extended if kb.is_observed(language, f))
        if observed_extended > observed_primary:
            return extended
        return primary

    def _observed_anchor_facts(
        self, kb: KnowledgeBase, language: str, target_feature: str, max_items: int = 5
    ) -> List[Tuple[str, str]]:
        anchors: List[Tuple[str, str]] = []
        for feat in self._effective_correlated(kb, language, target_feature):
            if feat == target_feature:
                continue
            v = kb.value(language, feat)
            if v is None:
                continue
            anchors.append((feat, str(int(v))))
            if len(anchors) >= max_items:
                break
        return anchors

    # ---- candidate ranking --------------------------------------------------

    def _ranked_phylo_candidates(
        self, kb: KnowledgeBase, language: str, pool_limit: int = 400
    ) -> List[str]:
        ranked: List[str] = []
        seen = {language}
        for nb in kb.genetic_neighbors(language):
            nb = str(nb)
            if nb in seen:
                continue
            seen.add(nb)
            ranked.append(nb)
            if len(ranked) >= pool_limit:
                return ranked
        idx = 0
        while idx < len(ranked) and len(ranked) < pool_limit:
            cur = ranked[idx]
            idx += 1
            for nxt in kb.genetic_neighbors(cur):
                nxt = str(nxt)
                if nxt in seen:
                    continue
                seen.add(nxt)
                ranked.append(nxt)
                if len(ranked) >= pool_limit:
                    return ranked
        for gc in kb.languages:
            if gc in seen:
                continue
            seen.add(gc)
            ranked.append(gc)
            if len(ranked) >= pool_limit:
                break
        return ranked

    def _latlon(self, kb: KnowledgeBase, glottocode: str) -> Tuple[Optional[float], Optional[float]]:
        meta = kb.metadata_for(glottocode)
        return meta.latitude, meta.longitude

    def _ranked_geo_candidates(
        self, kb: KnowledgeBase, language: str, pool_limit: int = 1200
    ) -> List[str]:
        if language in self._geo_rank_cache:
            return self._geo_rank_cache[language]

        seen = {language}
        ranked: List[str] = []
        lat0, lon0 = self._latlon(kb, language)

        if lat0 is not None and lon0 is not None:
            scored: List[Tuple[float, str]] = []
            for gc in kb.languages:
                if gc == language:
                    continue
                lat, lon = self._latlon(kb, gc)
                if lat is None or lon is None:
                    continue
                scored.append((_haversine_km(lat0, lon0, lat, lon), gc))
            scored.sort(key=lambda x: (x[0], x[1]))
            for _, gc in scored:
                if gc in seen:
                    continue
                seen.add(gc)
                ranked.append(gc)
                if len(ranked) >= pool_limit:
                    break

        if len(ranked) < pool_limit:
            for nb in kb.geographic_neighbors(language):
                nb = str(nb)
                if nb in seen:
                    continue
                seen.add(nb)
                ranked.append(nb)
                if len(ranked) >= pool_limit:
                    break

        if len(ranked) < pool_limit:
            for gc in kb.languages:
                if gc in seen:
                    continue
                seen.add(gc)
                ranked.append(gc)
                if len(ranked) >= pool_limit:
                    break

        self._geo_rank_cache[language] = ranked
        return ranked

    # ---- neighbor selection -------------------------------------------------

    @staticmethod
    def _neighbor_k(neighbour_map_size: int, fallback: int = 5) -> int:
        return neighbour_map_size if neighbour_map_size > 0 else fallback

    def _shared_feature_count(
        self,
        kb: KnowledgeBase,
        ref: str,
        cand: str,
        features: Sequence[str],
    ) -> int:
        score = 0
        for feat in features:
            rv = kb.value(ref, feat)
            cv = kb.value(cand, feat)
            if rv is None or cv is None:
                continue
            if rv == cv:
                score += 1
        return score

    def _observed_correlated_set(
        self, kb: KnowledgeBase, lang: str, features: Sequence[str]
    ) -> set[str]:
        return {f for f in features if kb.is_observed(lang, f)}

    def _nearest_with_value(
        self, kb: KnowledgeBase, candidates: Sequence[str], feature: str, desired: int
    ) -> Optional[str]:
        for nb in candidates:
            v = kb.value(nb, feature)
            if v == desired:
                return str(nb)
        return None

    def _select_neighbors(
        self,
        kb: KnowledgeBase,
        candidates: Sequence[str],
        correlated: Sequence[str],
        target_feature: str,
        k: int,
        force_include: Sequence[str] = (),
        reference_language: Optional[str] = None,
    ) -> List[str]:
        """v4 tie-breaker variant: marginal coverage + target coverage + similarity."""
        if k <= 0:
            return []

        features_set = set(kb.features)
        feature_targets: List[str] = [f for f in correlated if f in features_set]
        if target_feature in features_set and target_feature not in feature_targets:
            feature_targets.append(target_feature)

        ordered = [str(nb) for nb in candidates]
        if reference_language is not None:
            indexed = list(enumerate(ordered))
            indexed.sort(
                key=lambda x: (
                    0 if kb.is_observed(str(x[1]), target_feature) else 1,
                    -self._shared_feature_count(kb, reference_language, str(x[1]), feature_targets),
                    x[0],
                )
            )
            ordered = [str(nb) for _, nb in indexed]

        selected: List[str] = []
        selected_set: set[str] = set()
        covered: set[str] = set()

        for nb in force_include:
            nb = str(nb)
            if nb in selected_set or nb not in ordered:
                continue
            selected.append(nb)
            selected_set.add(nb)
            covered.update(self._observed_correlated_set(kb, nb, feature_targets))
            if len(selected) >= k:
                return selected[:k]

        while len(selected) < k:
            best_nb: Optional[str] = None
            best_obs: Optional[set[str]] = None
            best_key: Optional[tuple] = None
            for rank, nb in enumerate(ordered):
                if nb in selected_set:
                    continue
                obs = self._observed_correlated_set(kb, nb, feature_targets)
                if not obs:
                    continue
                new_obs = obs - covered
                if not new_obs:
                    continue
                new_obs_shared = (
                    self._shared_feature_count(kb, reference_language, nb, tuple(sorted(new_obs)))
                    if reference_language is not None
                    else 0
                )
                all_shared = (
                    self._shared_feature_count(kb, reference_language, nb, feature_targets)
                    if reference_language is not None
                    else 0
                )
                key = (
                    len(new_obs),
                    1 if kb.is_observed(nb, target_feature) else 0,
                    new_obs_shared,
                    all_shared,
                    -rank,
                )
                if best_key is None or key > best_key:
                    best_nb = nb
                    best_obs = obs
                    best_key = key

            if best_nb is None:
                break
            selected.append(best_nb)
            selected_set.add(best_nb)
            covered.update(best_obs or set())
            if len(selected) >= k and (
                not feature_targets or covered.issuperset(feature_targets)
            ):
                return selected[:k]

        if len(selected) < k:
            for nb in ordered:
                if nb in selected_set:
                    continue
                if self._observed_correlated_set(kb, nb, feature_targets):
                    selected.append(nb)
                    selected_set.add(nb)
                    if len(selected) >= k:
                        return selected[:k]

        if len(selected) < k:
            for nb in ordered:
                if nb in selected_set:
                    continue
                selected.append(nb)
                selected_set.add(nb)
                if len(selected) >= k:
                    return selected[:k]

        return selected[:k]

    # ---- evidence assembly --------------------------------------------------

    def _language_label(self, kb: KnowledgeBase, glottocode: str) -> str:
        return kb.metadata_for(glottocode).name or glottocode

    def _collect_neighbor_facts(
        self,
        kb: KnowledgeBase,
        glottocode: str,
        target_feature: str,
        correlated: Sequence[str],
        limit: int = 4,
        max_correlated_fallback: int = 3,
    ) -> List[Tuple[str, str]]:
        facts: List[Tuple[str, str]] = []
        tv = kb.value(glottocode, target_feature)
        if tv is not None:
            facts.append((target_feature, str(int(tv))))
            if len(facts) >= limit:
                return facts

        fallback = 0
        for feat in correlated:
            if feat == target_feature:
                continue
            v = kb.value(glottocode, feat)
            if v is None:
                continue
            facts.append((feat, str(int(v))))
            fallback += 1
            if len(facts) >= limit or fallback >= max_correlated_fallback:
                return facts
        return facts

    def _format_neighbor_block(
        self,
        kb: KnowledgeBase,
        language: str,
        neighbors: Sequence[str],
        candidates: Sequence[str],
        target_feature: str,
        correlated: Sequence[str],
        mode: str,
    ) -> List[str]:
        lines: List[str] = []
        rank = {str(nb): idx for idx, nb in enumerate(candidates, start=1)}
        lat0, lon0 = self._latlon(kb, language)

        for idx, nb in enumerate(neighbors, start=1):
            label = self._language_label(kb, nb)
            if mode == "geographic" and lat0 is not None and lon0 is not None:
                lat1, lon1 = self._latlon(kb, nb)
                if lat1 is not None and lon1 is not None:
                    header = f"{idx}) {label} (glottocode={nb}, km={_haversine_km(lat0, lon0, lat1, lon1):.1f}):"
                else:
                    header = f"{idx}) {label} (glottocode={nb}, rank={rank.get(str(nb), '?')}):"
            else:
                header = f"{idx}) {label} (glottocode={nb}, rank={rank.get(str(nb), '?')}):"
            lines.append(header)

            facts = self._collect_neighbor_facts(kb, nb, target_feature, correlated)
            if facts:
                for feat, val in facts:
                    lines.append(f"- {feat}: {val}")
            else:
                lines.append("- No observed target or anchor facts available.")
        return lines

    def _count_votes(
        self, kb: KnowledgeBase, neighbors: Sequence[str], feature: str
    ) -> Dict[str, int]:
        counts = {"yes": 0, "no": 0, "missing": 0}
        for nb in neighbors:
            v = kb.value(nb, feature)
            if v is None:
                counts["missing"] += 1
            elif v == 1:
                counts["yes"] += 1
            else:
                counts["no"] += 1
        return counts

    def _feature_prevalence_prior(
        self, kb: KnowledgeBase, feature: str
    ) -> Tuple[str, float]:
        if feature in self._prior_cache:
            return self._prior_cache[feature]
        col = None
        try:
            col_idx = kb.features.index(feature)
        except ValueError:
            self._prior_cache[feature] = ("0", 0.5)
            return self._prior_cache[feature]
        col = kb.as_matrix()[:, col_idx]
        observed = col[col != -1]
        if observed.size == 0:
            self._prior_cache[feature] = ("0", 0.5)
            return self._prior_cache[feature]
        pos = int((observed == 1).sum())
        neg = int((observed == 0).sum())
        total = pos + neg
        if pos > neg:
            best = ("1", pos / total)
        elif neg > pos:
            best = ("0", neg / total)
        else:
            best = ("0", 0.5)
        self._prior_cache[feature] = best
        return best

    def _clue_support(
        self, kb: KnowledgeBase, target_feature: str, clue_feature: str, clue_value: str
    ) -> Tuple[int, int]:
        key = (target_feature, clue_feature, clue_value)
        if key in self._clue_support_cache:
            return self._clue_support_cache[key]
        try:
            ti = kb.features.index(target_feature)
            ci = kb.features.index(clue_feature)
        except ValueError:
            self._clue_support_cache[key] = (0, 0)
            return self._clue_support_cache[key]
        M = kb.as_matrix()
        target_col = M[:, ti]
        clue_col = M[:, ci]
        cv = int(clue_value) if clue_value in {"0", "1"} else -99
        mask = (clue_col == cv) & (target_col != -1)
        target_vals = target_col[mask]
        yes = int((target_vals == 1).sum())
        no = int((target_vals == 0).sum())
        self._clue_support_cache[key] = (yes, no)
        return (yes, no)

    def _collect_clues(
        self, kb: KnowledgeBase, language: str, target_feature: str
    ) -> Tuple[List[Dict], Dict[str, int]]:
        clues: List[Dict] = []
        summary = {"yes": 0, "no": 0, "tie": 0, "n_used": 0}
        for clue_feat in self._effective_correlated(kb, language, target_feature):
            if clue_feat == target_feature:
                continue
            cv = kb.value(language, clue_feat)
            if cv is None:
                continue
            clue_val = str(int(cv))
            yes, no = self._clue_support(kb, target_feature, clue_feat, clue_val)
            support = yes + no
            if support < self.min_clue_support:
                continue
            majority = "yes" if yes > no else ("no" if no > yes else "tie")
            clues.append(
                {
                    "feature": clue_feat,
                    "value": clue_val,
                    "yes": yes,
                    "no": no,
                    "support_n": support,
                    "majority": majority,
                }
            )
            summary[majority] += 1
            summary["n_used"] += 1
            if len(clues) >= self.max_clues:
                break
        return clues, summary

    def _nearest_supporting_neighbor(
        self,
        kb: KnowledgeBase,
        language: str,
        candidates: Sequence[str],
        feature: str,
        desired: int,
        mode: str,
    ) -> Optional[str]:
        lat0, lon0 = self._latlon(kb, language)
        for idx, nb in enumerate(candidates, start=1):
            if kb.value(nb, feature) != desired:
                continue
            label = self._language_label(kb, nb)
            if mode == "geographic" and lat0 is not None and lon0 is not None:
                lat1, lon1 = self._latlon(kb, nb)
                if lat1 is not None and lon1 is not None:
                    return f"{label} ({_haversine_km(lat0, lon0, lat1, lon1):.1f} km)"
            return f"{label} (rank {idx})"
        return None

    # ---- Prompt interface ---------------------------------------------------

    def build(self, kb: KnowledgeBase, language: str, feature: str) -> PromptPayload:
        self._ensure_cache(kb)
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

        lines: List[str] = [
            "Target language:",
            f"- Name: {lang_name}",
            f"- Glottocode: {language}",
            f"- ISO639-3: {iso}",
            f"- Family lineage: {lineage}",
            f"- Macro-area: {macro}",
            f"- Location: latitude={lat}, longitude={lon}",
        ]

        correlated = self._effective_correlated(kb, language, feature)

        phylo_candidates = self._ranked_phylo_candidates(kb, language)
        phylo_k = self._neighbor_k(len(kb.genetic_neighbors(language)))
        phylo_yes_nb = self._nearest_with_value(kb, phylo_candidates, feature, 1)
        phylo_neighbors = self._select_neighbors(
            kb,
            phylo_candidates,
            correlated,
            feature,
            phylo_k,
            force_include=[phylo_yes_nb] if phylo_yes_nb else (),
            reference_language=language,
        )

        geo_candidates = self._ranked_geo_candidates(kb, language)
        geo_k = self._neighbor_k(len(kb.geographic_neighbors(language)))
        geo_yes_nb = self._nearest_with_value(kb, geo_candidates, feature, 1)
        geo_neighbors = self._select_neighbors(
            kb,
            geo_candidates,
            correlated,
            feature,
            geo_k,
            force_include=[geo_yes_nb] if geo_yes_nb else (),
            reference_language=language,
        )

        anchors = self._observed_anchor_facts(kb, language, feature)
        lines.append("Observed typological facts (anchor features):")
        if anchors:
            for feat_name, feat_value in anchors:
                lines.append(f"- {feat_name}: {feat_value} (observed)")
        else:
            lines.append("- (no observed anchor facts)")

        lines.append("Selected phylogenetic neighbors (detailed evidence):")
        lines.extend(
            self._format_neighbor_block(
                kb, language, phylo_neighbors, phylo_candidates, feature, correlated,
                mode="phylogenetic",
            )
        )

        lines.append("Selected geographic neighbors (detailed evidence):")
        lines.extend(
            self._format_neighbor_block(
                kb, language, geo_neighbors, geo_candidates, feature, correlated,
                mode="geographic",
            )
        )

        prior_value, prior_ratio = self._feature_prevalence_prior(kb, feature)

        if self.include_vote_table:
            pv = self._count_votes(kb, phylo_neighbors, feature)
            gv = self._count_votes(kb, geo_neighbors, feature)
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
            lines.append("Target-feature vote counts (useful but not decisive):")
            lines.append(
                f"- Genetic vote: {pv['yes']} yes / {pv['no']} no / {pv['missing']} unk ({p_ratio:.0%} yes)"
            )
            lines.append(
                f"- Geo vote: {gv['yes']} yes / {gv['no']} no / {gv['missing']} unk ({g_ratio:.0%} yes)"
            )
            lines.append(
                f"- Overall observed votes: {overall['yes']} yes / {overall['no']} no "
                f"({ov_ratio:.0%} yes; agreement={agreement:.0%})"
            )
            lines.append(
                f"- Vote evidence coverage: {ov_denom} observed target-feature votes "
                f"(unknown ignored in decision)."
            )
            lines.append(
                f"- Weak prevalence prior (tie-breaker only): value={prior_value} ({prior_ratio:.0%} of observed)"
            )

        phylo_yes = self._nearest_supporting_neighbor(kb, language, phylo_candidates, feature, 1, "phylogenetic")
        phylo_no = self._nearest_supporting_neighbor(kb, language, phylo_candidates, feature, 0, "phylogenetic")
        geo_yes = self._nearest_supporting_neighbor(kb, language, geo_candidates, feature, 1, "geographic")
        geo_no = self._nearest_supporting_neighbor(kb, language, geo_candidates, feature, 0, "geographic")
        lines.append("Nearest contrastive neighbor evidence:")
        lines.append(f"- Closest phylogenetic support for 1: {phylo_yes or 'none observed'}")
        lines.append(f"- Closest phylogenetic support for 0: {phylo_no or 'none observed'}")
        lines.append(f"- Closest geographic support for 1: {geo_yes or 'none observed'}")
        lines.append(f"- Closest geographic support for 0: {geo_no or 'none observed'}")

        clues, clue_summary = self._collect_clues(kb, language, feature)
        lines.append("Target-specific correlated clues (compact):")
        if clues:
            for idx, clue in enumerate(clues, start=1):
                lines.append(
                    f"{idx}) {clue['feature']}={clue['value']} -> target support "
                    f"{clue['yes']} yes / {clue['no']} no"
                )
            lines.append(
                f"- Correlated clues leaning: {clue_summary['yes']} yes / "
                f"{clue_summary['no']} no / {clue_summary['tie']} tie"
            )
        else:
            lines.append("- No reliable correlated clues with enough support.")

        lines.append(f"Prompt version: {self.version}")
        lines.append("Task:")
        lines.append("Predict the missing value for the following feature:")
        lines.append(f"- Feature: {feature}")
        lines.append("- Allowed values: 0 | 1")
        lines.append("Reasoning guidance:")
        lines.append("- Compare the support for value 0 versus value 1.")
        lines.append(
            "- Weigh observed anchor features, nearest phylogenetic evidence, nearest geographic evidence, and correlated clues together."
        )
        lines.append("- Neighbor counts are useful, but do not follow majority vote blindly.")
        lines.append(
            "- A smaller number of closer or more relevant neighbors may outweigh a larger but weaker group."
        )
        lines.append(
            "- It is acceptable to predict a minority value if it is better supported by the overall evidence."
        )
        lines.append(
            f"- Use feature prevalence only as a weak tie-breaker when evidence is otherwise balanced (prior={prior_value})."
        )
        lines.append("Output format (STRICT JSON):")
        lines.append("Output ONLY valid JSON.")
        lines.append(
            "Return exactly one minified JSON object on one line with keys: value, confidence, rationale."
        )
        lines.append("- value: one of the allowed values above")
        lines.append("- confidence: low, medium, or high")
        lines.append("- rationale: at most 20 words")
        lines.append("No Markdown, no prose, no code fences, no trailing text.")
        lines.append("Few-shot examples:")
        lines.append(
            '{"value":"1","confidence":"medium","rationale":"Most neighbors are 0, but the closest and most similar languages support 1."}'
        )
        lines.append(
            '{"value":"1","confidence":"high","rationale":"Observed features and nearest phylogenetic evidence align strongly with value 1."}'
        )
        lines.append(
            '{"value":"0","confidence":"low","rationale":"Evidence is balanced, so the weak prevalence prior favors 0."}'
        )

        return PromptPayload(
            system=SYSTEM_MESSAGE,
            user="\n".join(lines),
            meta={
                "language": language,
                "feature": feature,
                "version": self.version,
                "phylo_neighbors": list(phylo_neighbors),
                "geo_neighbors": list(geo_neighbors),
            },
        )

    def parse(self, raw: str) -> Prediction:
        value, confidence, rationale = _extract_json_fields(raw)
        value = _normalize_value(value, raw, ["0", "1"])
        conf = _normalize_confidence(confidence)
        rat = _normalize_rationale(rationale)
        parsed_ok = value is not None
        return Prediction(
            value=value,
            confidence=conf,
            rationale=rat,
            parsed_ok=parsed_ok,
            raw=raw,
        )


def _extract_json_fields(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    raw = text.strip()
    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None

    if not isinstance(parsed, dict):
        candidates: List[dict] = []
        start = raw.find("{")
        if start >= 0:
            depth = 0
            in_string = False
            escaped = False
            seg_start = -1
            for i in range(start, len(raw)):
                ch = raw[i]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                    continue
                if ch == "{":
                    if depth == 0:
                        seg_start = i
                    depth += 1
                elif ch == "}":
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and seg_start >= 0:
                            segment = raw[seg_start : i + 1]
                            try:
                                obj = json.loads(segment)
                                if isinstance(obj, dict):
                                    candidates.append(obj)
                            except Exception:
                                pass
        if candidates:
            parsed = candidates[-1]

    if not isinstance(parsed, dict):
        return None, None, None
    value = parsed.get("value")
    confidence = parsed.get("confidence")
    rationale = parsed.get("rationale")
    return (
        None if value is None else str(value),
        None if confidence is None else str(confidence).lower(),
        None if rationale is None else str(rationale),
    )


def _normalize_confidence(value: Optional[str]) -> str:
    if not value:
        return "low"
    v = value.strip().lower()
    if "|" in v or "<" in v or ">" in v:
        return "low"
    if v in VALID_CONFIDENCE:
        return v
    if "high" in v:
        return "high"
    if "med" in v:
        return "medium"
    if "low" in v:
        return "low"
    return "low"


def _normalize_rationale(value: Optional[str]) -> str:
    text = "" if value is None else " ".join(str(value).split())
    lower = text.lower()
    if "<max" in lower or "<one" in lower:
        return "Insufficient direct evidence."
    words = text.split()
    if not words:
        return "Insufficient direct evidence."
    return " ".join(words[:30])


def _normalize_value(
    parsed: Optional[str], raw_output: str, allowed_values: Sequence[str]
) -> Optional[str]:
    allowed = [str(v) for v in allowed_values]
    if not allowed:
        return parsed

    candidates: List[str] = []
    if parsed is not None:
        candidates.append(str(parsed).strip())
    for pattern in (
        r'"value"\s*:\s*"([^"]+)"',
        r'"value"\s*:\s*([^\s,}]+)',
    ):
        for m in re.finditer(pattern, raw_output, flags=re.IGNORECASE):
            candidates.append(m.group(1).strip().strip('"').strip("'"))

    for cand in candidates:
        if cand in allowed:
            return cand
        if re.fullmatch(r"-?\d+(?:\.0+)?", cand):
            normalized = str(int(float(cand)))
            if normalized in allowed:
                return normalized

    for val in sorted(set(allowed), key=len, reverse=True):
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(val)}(?![A-Za-z0-9_])", raw_output):
            return val

    return None
