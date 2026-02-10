from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd


SYSTEM_MESSAGE = (
    "You are a linguistics expert specializing in typology.\n"
    "Infer missing typological features using:\n"
    "(1) observed facts about the target language,\n"
    "(2) evidence from phylogenetic and geographic neighbors,\n"
    "and (3) well-established linguistic universals.\n"
    "If evidence conflicts, prefer phylogenetic evidence over geographic evidence.\n"
    "If uncertainty remains, choose the most typologically common value."
)

typ_df: Optional[pd.DataFrame]
metadata_df: pd.DataFrame
genetic_neighbours: dict
geographic_neighbours: dict
top_n_features: int
topk_map: Dict[str, List[str]]


def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass
    return v == -1


def _format_value(v) -> str:
    if _is_missing(v):
        return "Unknown"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _get_meta_value(row: Optional[pd.Series], col: str, default: str) -> str:
    if row is None or col not in row.index:
        return default
    val = row[col]
    if _is_missing(val) or (isinstance(val, str) and val.strip() == ""):
        return default
    return str(val)


def _family_lineage(row: Optional[pd.Series]) -> str:
    if row is None:
        return "Isolate"
    family = _get_meta_value(row, "family_name", "")
    parent = _get_meta_value(row, "parent_name", "")
    if not family and not parent:
        return "Isolate"
    if family and parent and parent != family:
        return f"{family} > {parent}"
    return family or parent or "Isolate"


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def _top_correlated_features(feature: str, top_n: int) -> List[str]:
    feats = topk_map.get(feature, [])
    if not feats:
        return []
    return feats[:top_n] if top_n > 0 else feats


def _collect_neighbor_facts(
    glottocode: str, target_feature: str, correlated: Sequence[str], limit: int = 12
) -> List[Tuple[str, str]]:
    if typ_df is None or glottocode not in typ_df.index:
        return []
    facts: List[Tuple[str, str]] = []

    for feat in correlated:
        if feat == target_feature or feat not in typ_df.columns:
            continue
        val = typ_df.at[glottocode, feat]
        if not _is_missing(val):
            facts.append((feat, _format_value(val)))
        if len(facts) >= limit:
            return facts

    if target_feature in typ_df.columns:
        val = typ_df.at[glottocode, target_feature]
        if not _is_missing(val) and len(facts) < limit:
            facts.append((target_feature, _format_value(val)))

    return facts


def _allowed_values(feature: str) -> List[str]:
    if typ_df is None or feature not in typ_df.columns:
        return ["0", "1"]
    col = typ_df[feature]
    vals = [v for v in col.unique().tolist() if not _is_missing(v)]
    if not vals:
        return ["0", "1"]
    if all(isinstance(v, float) and v.is_integer() for v in vals):
        vals = [int(v) for v in vals]
    return [str(v) for v in sorted(vals, key=lambda x: str(x))]


def _iter_missing_pairs(df: pd.DataFrame) -> Iterator[Tuple[str, str]]:
    for lang, row in df.iterrows():
        for feat, val in row.items():
            if _is_missing(val):
                yield str(lang), str(feat)


def construct_prompt(language: str, feature: str) -> Tuple[str, str]:
    """
    Construct the prompt for the given language and feature.
    :param language: str, Glottocode of the language to impute
    :param feature: str, the typological feature to impute
    :return: (str, str), the system and user prompts
    """
    system = SYSTEM_MESSAGE

    meta_row = metadata_df.loc[language] if language in metadata_df.index else None
    lang_name = _get_meta_value(meta_row, "name", language)
    iso = _get_meta_value(meta_row, "iso639_3", "None")
    lineage = _family_lineage(meta_row)
    macro = _get_meta_value(meta_row, "macroareas", "None")
    lat = _get_meta_value(meta_row, "latitude", "None")
    lon = _get_meta_value(meta_row, "longitude", "None")

    user_lines = []
    user_lines.append("Target language:")
    user_lines.append(f"- Name: {lang_name}")
    user_lines.append(f"- Glottocode: {language}")
    user_lines.append(f"- ISO639-3: {iso}")
    user_lines.append(f"- Family lineage: {lineage}")
    user_lines.append(f"- Macro-area: {macro}")
    user_lines.append(f"- Location: latitude={lat}, longitude={lon}")

    user_lines.append("Observed typological facts (anchor features):")
    correlated = _top_correlated_features(feature, top_n_features)
    for feat in correlated:
        if feat == feature:
            continue
        if typ_df is not None and feat not in typ_df.columns:
            continue
        val = typ_df.at[language, feat] if (typ_df is not None and language in typ_df.index) else None
        status = "missing" if _is_missing(val) else "observed"
        user_lines.append(f"- {feat}: {_format_value(val)} ({status})")

    user_lines.append("Phylogenetic neighbors (top-k):")
    for i, nb in enumerate(genetic_neighbours.get(language, []), start=1):
        nb_meta = metadata_df.loc[nb] if nb in metadata_df.index else None
        nb_name = _get_meta_value(nb_meta, "name", nb)
        user_lines.append(f"{i}) {nb_name} (glottocode={nb}, dist={i}):")
        facts = _collect_neighbor_facts(nb, feature, correlated)
        if not facts:
            user_lines.append("- (no observed anchor facts)")
        else:
            for feat_name, feat_val in facts:
                user_lines.append(f"- {feat_name}: {feat_val}")

    user_lines.append("Geographic neighbors (top-k):")
    lat_val = None if lat == "None" else float(lat)
    lon_val = None if lon == "None" else float(lon)
    if lat_val is not None and lon_val is not None:
        for i, nb in enumerate(geographic_neighbours.get(language, []), start=1):
            nb_meta = metadata_df.loc[nb] if nb in metadata_df.index else None
            nb_name = _get_meta_value(nb_meta, "name", nb)
            nb_lat = _get_meta_value(nb_meta, "latitude", "None")
            nb_lon = _get_meta_value(nb_meta, "longitude", "None")
            km = "unknown"
            if nb_lat != "None" and nb_lon != "None":
                km = f"{_haversine_km(lat_val, lon_val, float(nb_lat), float(nb_lon)):.1f}"
            user_lines.append(f"{i}) {nb_name} (glottocode={nb}, km={km}):")
            facts = _collect_neighbor_facts(nb, feature, correlated)
            if not facts:
                user_lines.append("- (no observed anchor facts)")
            else:
                for feat_name, feat_val in facts:
                    user_lines.append(f"- {feat_name}: {feat_val}")

    allowed = _allowed_values(feature)
    user_lines.append("Task:")
    user_lines.append("Predict the missing value for the following feature:")
    user_lines.append(f"- Feature: {feature}")
    user_lines.append(f"- Allowed values: {' | '.join(allowed)}")
    user_lines.append("Output format (STRICT):")
    user_lines.append("Return exactly one allowed value.")
    user_lines.append("Do not provide explanations.")

    user = "\n".join(user_lines)
    return system, user


def run_llama(system: str, user: str, tokenizer, model, max_new_tokens: int = 8) -> str:
    """
    Run the LLaMA model with the given prompt.
    :param system: str, system message
    :param user: str, user message
    :param tokenizer: tokenizer
    :param model: model
    :param max_new_tokens: int, generation cap
    :return: str, model output
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"SYSTEM:\n{system}\nUSER:\n{user}\nASSISTANT:\n"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )
    gen = outputs[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    return text.strip()


def _load_impute_pairs(path: str) -> List[Tuple[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Impute list not found: {path}")
    data = json.loads(p.read_text())
    pairs: List[Tuple[str, str]] = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, str):
                pairs.append((k, v))
            elif isinstance(v, list):
                for feat in v:
                    pairs.append((k, feat))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                pairs.append((item[0], item[1]))
            elif isinstance(item, dict):
                lang = item.get("language") or item.get("glottocode")
                feat = item.get("feature")
                if lang and feat:
                    pairs.append((lang, feat))
    else:
        raise ValueError("Unsupported impute list format")
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(description="Prompt builder for typological feature imputation.")
    p.add_argument("--mode", type=str, choices=["prompts", "predict"], default="prompts")
    p.add_argument("--typ", type=str, required=True, help="Typological CSV (glottocode index).")
    p.add_argument("--meta", type=str, required=True, help="Metadata CSV (glottocode index).")
    p.add_argument("--topk_csv", type=str, required=True, help="CSV from select_topk_features.py.")
    p.add_argument("--gen", type=str, required=True, help="Genetic neighbours JSON.")
    p.add_argument("--geo", type=str, required=True, help="Geographic neighbours JSON.")
    p.add_argument("--top_n", type=int, default=10, help="Top-N correlated features to include.")
    p.add_argument("--impute", type=str, default=None, help="Optional JSON list/dict of (language, feature) pairs.")
    p.add_argument("--out", type=str, required=True, help="Output file. JSONL for prompts mode, JSON for predict mode.")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--max_new_tokens", type=int, default=8)
    args = p.parse_args()

    global typ_df, metadata_df, genetic_neighbours, geographic_neighbours, top_n_features, topk_map

    typ_df = pd.read_csv(args.typ, index_col=0)
    typ_df.index = typ_df.index.astype(str)
    metadata_df = pd.read_csv(args.meta, index_col=0)
    metadata_df.index = metadata_df.index.astype(str)
    topk_df = pd.read_csv(args.topk_csv)

    with open(args.gen, "r", encoding="utf-8") as f:
        genetic_neighbours = json.load(f)
    with open(args.geo, "r", encoding="utf-8") as f:
        geographic_neighbours = json.load(f)
    genetic_neighbours = {str(k): v for k, v in genetic_neighbours.items()}
    geographic_neighbours = {str(k): v for k, v in geographic_neighbours.items()}

    top_n_features = args.top_n
    tmp: Dict[str, List[Tuple[int, str]]] = {}
    for _, row in topk_df.iterrows():
        feat = str(row["feature"])
        other = str(row["other_feature"])
        rank = int(row["rank"]) if "rank" in row and not pd.isna(row["rank"]) else 0
        tmp.setdefault(feat, []).append((rank, other))
    topk_map = {}
    for feat, items in tmp.items():
        items.sort(key=lambda x: x[0])
        topk_map[feat] = [other for _, other in items]

    if args.impute:
        impute_pairs: Iterable[Tuple[str, str]] = _load_impute_pairs(args.impute)
    else:
        impute_pairs = _iter_missing_pairs(typ_df)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "prompts":
        with open(args.out, "w", encoding="utf-8") as f:
            for lang, feat in impute_pairs:
                system, user = construct_prompt(lang, feat)
                rec = {
                    "language": lang,
                    "feature": feat,
                    "system": system,
                    "user": user,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)

        results = {}
        for lang, feat in impute_pairs:
            system, user = construct_prompt(lang, feat)
            pred = run_llama(system, user, tokenizer, model, max_new_tokens=args.max_new_tokens)
            results.setdefault(lang, {})[feat] = pred

        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
