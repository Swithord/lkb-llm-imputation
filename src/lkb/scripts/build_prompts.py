"""CLI: build a prompts JSONL from a gold JSONL using ICLPrompt or KGPrompt.

Usage:
  python -m lkb.scripts.build_prompts \\
    --gold data/benchmark/gold_eval_coverage_bottomk200_equal2233_v1.jsonl \\
    --prompt icl \\
    --version v4_strict_json \\
    --include_vote_table \\
    --out artifacts/prediction/prompts.jsonl

  python -m lkb.scripts.build_prompts \\
    --gold data/benchmark/gold_eval_coverage_bottomk200_equal2233_v1.jsonl \\
    --prompt kg \\
    --version v5_glottolog_tree_json \\
    --retrieval_backend hybrid_flat_kg \\
    --include_vote_table \\
    --out artifacts/prediction/prompts_kg.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lkb.kb.uriel_plus import URIELPlus
from lkb.impute.prompts.icl import ICLPrompt
from lkb.impute.prompts.kg import KGPrompt
from lkb.impute.retrievers import (
    HybridFlatKGRetriever,
    KGFlatRetriever,
    KGTypedContrastiveRetriever,
    KGTypedRetriever,
)

_RETRIEVERS = {
    "kg_flat": KGFlatRetriever,
    "kg_typed": KGTypedRetriever,
    "kg_typed_contrastive": KGTypedContrastiveRetriever,
    "hybrid_flat_kg": HybridFlatKGRetriever,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prompts JSONL from gold JSONL.")
    parser.add_argument("--gold", required=True, help="Gold JSONL path")
    parser.add_argument("--prompt", default="icl", choices=["icl", "kg"])
    parser.add_argument("--version", default="v4_strict_json")
    parser.add_argument("--include_vote_table", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retrieval_backend", default="hybrid_flat_kg", choices=list(_RETRIEVERS))
    parser.add_argument("--top_n_features", type=int, default=10)
    # Data paths forwarded to URIELPlus.from_artifacts
    parser.add_argument("--data_root", default=".")
    parser.add_argument("--typ", default="data/derived/uriel+_typological.csv")
    parser.add_argument("--meta", default="data/derived/metadata.csv")
    parser.add_argument("--topk_csv", default="data/features/topk_per_feature.csv")
    parser.add_argument("--gen", default="data/derived/genetic_neighbours.json")
    parser.add_argument("--gen_detail", default="data/derived/genetic_neighbours_detailed.json")
    parser.add_argument("--geo", default="data/derived/geographic_neighbours.json")
    parser.add_argument("--kg_nodes", default="artifacts/resources/kg_nodes.jsonl")
    parser.add_argument("--kg_edges", default="artifacts/resources/kg_edges.jsonl")
    parser.add_argument("--out", required=True, help="Output prompts JSONL path")
    args = parser.parse_args()

    kb = URIELPlus.from_artifacts(
        args.data_root,
        typology_path=args.typ,
        metadata_path=args.meta,
        genetic_neighbours_path=args.gen,
        genetic_neighbour_details_path=args.gen_detail,
        geographic_neighbours_path=args.geo,
        feature_topk_path=args.topk_csv,
        kg_nodes_path=args.kg_nodes,
        kg_edges_path=args.kg_edges,
        load_kg_graph=(args.prompt == "kg"),
    )

    if args.prompt == "icl":
        prompt = ICLPrompt(
            top_n_features=args.top_n_features,
            include_vote_table=args.include_vote_table,
        )
    else:
        retriever = _RETRIEVERS[args.retrieval_backend]()
        prompt = KGPrompt(
            retriever,
            version=args.version,
            top_n_features=args.top_n_features,
            include_vote_table=args.include_vote_table,
        )

    gold_items = []
    with open(args.gold, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                gold_items.append(json.loads(line))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for item in gold_items:
            lang = item["language"]
            feat = item["feature"]
            try:
                payload = prompt.build(kb, lang, feat)
            except Exception as exc:
                print(f"WARNING: build failed for {lang}:{feat}: {exc}")
                payload = None

            record = {
                "id": item["id"],
                "language": lang,
                "feature": feat,
                "gold_value": item.get("gold_value"),
                "resource_group": item.get("resource_group", "unknown"),
                "system": payload.system if payload else "",
                "user": payload.user if payload else "",
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"Wrote {n_ok} prompts -> {args.out}")


if __name__ == "__main__":
    main()
