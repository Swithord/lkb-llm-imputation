from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict

import pandas as pd


def _load_prompting_module(path: str):
    spec = importlib.util.spec_from_file_location("prompting_impl", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load prompting module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _build_topk_map(topk_df: pd.DataFrame) -> Dict[str, list[str]]:
    tmp: Dict[str, list[tuple[int, str]]] = {}
    for _, row in topk_df.iterrows():
        feat = str(row["feature"])
        other = str(row["other_feature"])
        rank = int(row["rank"]) if "rank" in row and not pd.isna(row["rank"]) else 0
        tmp.setdefault(feat, []).append((rank, other))
    return {feat: [other for _, other in sorted(items, key=lambda x: x[0])] for feat, items in tmp.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="Build prompt JSONL for an existing gold id set.")
    p.add_argument("--prompting_py", type=str, default="glottolog-tree/prompting.py")
    p.add_argument("--typ", type=str, default="data/derived/uriel+_typological.csv")
    p.add_argument("--meta", type=str, default="data/derived/metadata.csv")
    p.add_argument("--topk_csv", type=str, default="data/features/topk_per_feature.csv")
    p.add_argument("--gen", type=str, default="data/derived/genetic_neighbours.json")
    p.add_argument("--gen_detail", type=str, default="data/derived/genetic_neighbours_detailed.json")
    p.add_argument("--geo", type=str, default="data/derived/geographic_neighbours.json")
    p.add_argument("--retrieval_backend", type=str, default="legacy")
    p.add_argument("--kg_nodes", type=str, default=None)
    p.add_argument("--kg_edges", type=str, default=None)
    p.add_argument("--gold", type=str, default="data/benchmark/gold_eval_2.jsonl")
    p.add_argument("--top_n", type=int, default=10)
    p.add_argument("--prompt_version", type=str, default="v5_glottolog_tree_json")
    p.add_argument(
        "--include_vote_table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include compact vote summary table in prompt context.",
    )
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    typ_df = pd.read_csv(args.typ, index_col=0)
    typ_df.index = typ_df.index.astype(str)
    meta_df = pd.read_csv(args.meta, index_col=0)
    meta_df.index = meta_df.index.astype(str)
    topk_df = pd.read_csv(args.topk_csv)

    with open(args.gen, "r", encoding="utf-8") as f:
        genetic = {str(k): v for k, v in json.load(f).items()}
    gen_detail_path = Path(args.gen_detail)
    if gen_detail_path.exists():
        with gen_detail_path.open("r", encoding="utf-8") as f:
            genetic_detail = {str(k): v for k, v in json.load(f).items()}
    else:
        genetic_detail = {}
    with open(args.geo, "r", encoding="utf-8") as f:
        geographic = {str(k): v for k, v in json.load(f).items()}

    prompting = _load_prompting_module(args.prompting_py)
    prompting.typ_df = typ_df.copy()
    prompting.metadata_df = meta_df
    prompting.genetic_neighbours = genetic
    if hasattr(prompting, "genetic_neighbour_details"):
        prompting.genetic_neighbour_details = genetic_detail
    prompting.geographic_neighbours = geographic
    prompting.top_n_features = args.top_n
    prompting.topk_map = _build_topk_map(topk_df)
    prompting.set_prompt_options(args.prompt_version, args.include_vote_table)
    if hasattr(prompting, "set_retrieval_options"):
        prompting.set_retrieval_options(args.retrieval_backend, args.kg_nodes, args.kg_edges)

    rows = _read_jsonl(args.gold)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in rows:
            item_id = str(rec["id"])
            group = str(rec.get("resource_group", ""))
            lang = str(rec["language"])
            feat = str(rec["feature"])
            allowed_values = [str(x) for x in rec.get("allowed_values", ["0", "1"])]

            original = prompting.typ_df.at[lang, feat]
            prompting.typ_df.at[lang, feat] = -1
            try:
                system, user = prompting.construct_prompt(lang, feat)
            finally:
                prompting.typ_df.at[lang, feat] = original

            prompt_rec = {
                "id": item_id,
                "resource_group": group,
                "language": lang,
                "feature": feat,
                "allowed_values": allowed_values,
                "system": system,
                "user": user,
            }
            f.write(json.dumps(prompt_rec, ensure_ascii=False) + "\n")

    print(f"Wrote prompts: {out_path}")
    print(f"Num prompts: {len(rows)}")
    print(f"Prompt version: {args.prompt_version}")
    print(f"Include vote table: {args.include_vote_table}")


if __name__ == "__main__":
    main()
