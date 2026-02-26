from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _load_prompting_module(path: str):
    spec = importlib.util.spec_from_file_location("prompting_impl", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load prompting module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _allowed_values_from_df(df: pd.DataFrame, feature: str) -> List[str]:
    if feature not in df.columns:
        return ["0", "1"]
    vals = [v for v in df[feature].unique().tolist() if v != -1 and not pd.isna(v)]
    if not vals:
        return ["0", "1"]
    out: List[str] = []
    for v in vals:
        if isinstance(v, float) and v.is_integer():
            out.append(str(int(v)))
        else:
            out.append(str(v))
    return sorted(set(out))


def _rewrite_output_contract(user_prompt: str) -> str:
    marker = "Output format (STRICT):"
    prefix = user_prompt.split(marker)[0].rstrip()
    return (
        f"{prefix}\n"
        "Prompt version: v3_strict_json\n"
        "Output format (STRICT JSON):\n"
        "Output ONLY valid JSON.\n"
        "Return exactly one minified JSON object on one line with keys in this exact order:\n"
        '{"value":"<one allowed value>","confidence":"low|medium|high","rationale":"<max 30 words>"}\n'
        "Use double quotes only. No Markdown, no code fences, no extra text.\n"
        "Few-shot examples:\n"
        '{"value":"0","confidence":"high","rationale":"Neighbor evidence strongly favors value 0 with consistent signals."}\n'
        '{"value":"1","confidence":"medium","rationale":"Phylogenetic evidence favors 1 but geographic evidence is mixed."}\n'
        '{"value":"0","confidence":"low","rationale":"Evidence is sparse and conflicting; defaulting to the more common value."}'
    )


def _extract_structured_description(user_prompt: str) -> str:
    marker = "\nTask:"
    if marker in user_prompt:
        return user_prompt.split(marker, 1)[0].rstrip()
    return user_prompt.rstrip()


def _parse_binary_gold(gold_value: str, item_id: str) -> int:
    try:
        val = int(float(gold_value))
    except Exception as e:
        raise ValueError(f"Gold value is not numeric for {item_id}: {gold_value}") from e
    if val not in (0, 1):
        raise ValueError(f"Gold value is not binary for {item_id}: {gold_value}")
    return val


def _pick_language_groups(
    typ_df: pd.DataFrame, high_n: int, low_n: int, min_observed_low: int
) -> Dict[str, List[str]]:
    obs_counts = (typ_df != -1).sum(axis=1).sort_values(ascending=False)
    high = obs_counts.head(high_n).index.astype(str).tolist()

    low_pool = obs_counts[obs_counts >= min_observed_low].sort_values(ascending=True)
    low = [gc for gc in low_pool.index.astype(str).tolist() if gc not in set(high)]
    low = low[:low_n]

    if len(low) < low_n:
        fill = [gc for gc in obs_counts.sort_values(ascending=True).index.astype(str).tolist() if gc not in set(high) and gc not in set(low)]
        low.extend(fill[: (low_n - len(low))])

    return {"high": high, "low": low}


def _pick_mask_pairs(
    typ_df: pd.DataFrame,
    groups: Dict[str, List[str]],
    per_language: int,
    seed: int,
) -> List[Tuple[str, str, str]]:
    rng = random.Random(seed)
    pairs: List[Tuple[str, str, str]] = []
    for group in ("high", "low"):
        for lang in groups[group]:
            if lang not in typ_df.index:
                continue
            row = typ_df.loc[lang]
            observed = [feat for feat, val in row.items() if val != -1 and not pd.isna(val)]
            if not observed:
                continue
            rng.shuffle(observed)
            take = observed[: min(per_language, len(observed))]
            for feat in take:
                pairs.append((group, lang, feat))
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(description="Build high/low resource benchmark prompts and gold labels.")
    p.add_argument("--prompting_py", type=str, default="code/prompting.py")
    p.add_argument("--typ", type=str, default="output/uriel+_typological.csv")
    p.add_argument("--meta", type=str, default="output/metadata.csv")
    p.add_argument("--topk_csv", type=str, default="out_corr/topk_per_feature.csv")
    p.add_argument("--gen", type=str, default="output/genetic_neighbours.json")
    p.add_argument("--geo", type=str, default="output/geographic_neighbours.json")
    p.add_argument("--high_n", type=int, default=50)
    p.add_argument("--low_n", type=int, default=50)
    p.add_argument("--min_observed_low", type=int, default=5)
    p.add_argument("--per_language", type=int, default=20)
    p.add_argument("--top_n", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--prompts_out", type=str, default="benchmark/prompts_eval.jsonl")
    p.add_argument("--prompts_high_out", type=str, default="benchmark/prompts_eval_high.jsonl")
    p.add_argument("--prompts_low_out", type=str, default="benchmark/prompts_eval_low.jsonl")
    p.add_argument("--gold_out", type=str, default="benchmark/gold_eval.jsonl")
    p.add_argument("--examples_out", type=str, default="benchmark/examples_eval.jsonl")
    p.add_argument("--examples_high_out", type=str, default="benchmark/examples_eval_high.jsonl")
    p.add_argument("--examples_low_out", type=str, default="benchmark/examples_eval_low.jsonl")
    p.add_argument("--groups_out", type=str, default="benchmark/language_groups.json")
    args = p.parse_args()

    typ_df = pd.read_csv(args.typ, index_col=0)
    typ_df.index = typ_df.index.astype(str)
    meta_df = pd.read_csv(args.meta, index_col=0)
    meta_df.index = meta_df.index.astype(str)
    topk_df = pd.read_csv(args.topk_csv)

    with open(args.gen, "r", encoding="utf-8") as f:
        genetic = json.load(f)
    with open(args.geo, "r", encoding="utf-8") as f:
        geographic = json.load(f)
    genetic = {str(k): v for k, v in genetic.items()}
    geographic = {str(k): v for k, v in geographic.items()}

    tmp: Dict[str, List[Tuple[int, str]]] = {}
    for _, row in topk_df.iterrows():
        feat = str(row["feature"])
        other = str(row["other_feature"])
        rank = int(row["rank"]) if "rank" in row and not pd.isna(row["rank"]) else 0
        tmp.setdefault(feat, []).append((rank, other))
    topk_map = {feat: [other for _, other in sorted(items, key=lambda x: x[0])] for feat, items in tmp.items()}

    groups = _pick_language_groups(typ_df, args.high_n, args.low_n, args.min_observed_low)
    pairs = _pick_mask_pairs(typ_df, groups, args.per_language, args.seed)

    prompting = _load_prompting_module(args.prompting_py)
    prompting.typ_df = typ_df.copy()
    prompting.metadata_df = meta_df
    prompting.genetic_neighbours = genetic
    prompting.geographic_neighbours = geographic
    prompting.top_n_features = args.top_n
    prompting.topk_map = topk_map

    Path(args.prompts_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.prompts_high_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.prompts_low_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.gold_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.examples_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.examples_high_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.examples_low_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.groups_out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.prompts_out, "w", encoding="utf-8") as fp, \
        open(args.prompts_high_out, "w", encoding="utf-8") as fph, \
        open(args.prompts_low_out, "w", encoding="utf-8") as fpl, \
        open(args.gold_out, "w", encoding="utf-8") as fg, \
        open(args.examples_out, "w", encoding="utf-8") as fe, \
        open(args.examples_high_out, "w", encoding="utf-8") as feh, \
        open(args.examples_low_out, "w", encoding="utf-8") as fel:
        for i, (group, lang, feat) in enumerate(pairs):
            gold_raw = typ_df.at[lang, feat]
            gold_value = str(int(gold_raw)) if isinstance(gold_raw, float) and gold_raw.is_integer() else str(gold_raw)
            allowed_values = _allowed_values_from_df(typ_df, feat)
            item_id = f"{group}:{lang}:{feat}:{i}"

            original = prompting.typ_df.at[lang, feat]
            prompting.typ_df.at[lang, feat] = -1
            try:
                system, user = prompting.construct_prompt(lang, feat)
            finally:
                prompting.typ_df.at[lang, feat] = original

            user = _rewrite_output_contract(user)

            prompt_rec = {
                "id": item_id,
                "resource_group": group,
                "language": lang,
                "feature": feat,
                "allowed_values": allowed_values,
                "system": system,
                "user": user,
            }
            gold_rec = {
                "id": item_id,
                "resource_group": group,
                "language": lang,
                "feature": feat,
                "gold_value": gold_value,
                "allowed_values": allowed_values,
            }
            example_rec = {
                "example_id": item_id,
                "language_id": lang,
                "feature_id": feat,
                "gold": _parse_binary_gold(gold_value, item_id),
                "resource_group": group,
                "r_L": _extract_structured_description(user),
            }
            prompt_line = json.dumps(prompt_rec, ensure_ascii=False) + "\n"
            fp.write(prompt_line)
            if group == "high":
                fph.write(prompt_line)
            elif group == "low":
                fpl.write(prompt_line)
            fg.write(json.dumps(gold_rec, ensure_ascii=False) + "\n")
            example_line = json.dumps(example_rec, ensure_ascii=False) + "\n"
            fe.write(example_line)
            if group == "high":
                feh.write(example_line)
            elif group == "low":
                fel.write(example_line)

    Path(args.groups_out).write_text(json.dumps(groups, ensure_ascii=False, indent=2))

    print(f"Wrote prompts: {args.prompts_out}")
    print(f"Wrote high prompts: {args.prompts_high_out}")
    print(f"Wrote low prompts:  {args.prompts_low_out}")
    print(f"Wrote gold:    {args.gold_out}")
    print(f"Wrote examples: {args.examples_out}")
    print(f"Wrote high examples: {args.examples_high_out}")
    print(f"Wrote low examples:  {args.examples_low_out}")
    print(f"Wrote groups:  {args.groups_out}")
    print(f"Num prompts:   {len(pairs)}")


if __name__ == "__main__":
    main()
