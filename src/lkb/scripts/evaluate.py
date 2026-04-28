"""CLI: evaluate predictions JSONL against gold JSONL; write report JSON.

Usage:
  python -m lkb.scripts.evaluate \\
    --gold data/benchmark/gold_eval_coverage_bottomk200_equal2233_v1.jsonl \\
    --pred artifacts/prediction/predictions.jsonl \\
    --report_out artifacts/prediction/report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from lkb.eval.evaluator import Evaluator, GoldItem
from lkb.interfaces import Prediction


def _load_gold(path: str) -> List[GoldItem]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(
                GoldItem(
                    id=d["id"],
                    language=d["language"],
                    feature=d["feature"],
                    gold_value=str(d["gold_value"]),
                    resource_group=d.get("resource_group", "unknown"),
                    allowed_values=tuple(d.get("allowed_values", ("0", "1"))),
                )
            )
    return items


def _load_predictions(path: str) -> List[Prediction]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(
                Prediction(
                    value=d.get("value"),
                    confidence=d.get("confidence"),
                    rationale=d.get("rationale"),
                    parsed_ok=d.get("parsed_ok", d.get("value") is not None),
                    raw=d.get("raw"),
                )
            )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions against gold.")
    parser.add_argument("--gold", required=True, help="Gold JSONL path")
    parser.add_argument("--pred", required=True, help="Predictions JSONL path")
    parser.add_argument("--report_out", required=True, help="Output report JSON path")
    args = parser.parse_args()

    gold = _load_gold(args.gold)
    predictions = _load_predictions(args.pred)

    if len(gold) != len(predictions):
        raise ValueError(
            f"Gold ({len(gold)}) and predictions ({len(predictions)}) lengths differ. "
            "Ensure they were generated from the same gold file in the same order."
        )

    evaluator = Evaluator()
    report = evaluator.evaluate(gold, predictions)

    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote report -> {args.report_out}")

    overall = report["metrics"].get("overall", {})
    print(
        f"Overall: acc={overall.get('accuracy', 0):.4f}  "
        f"f1={overall.get('f1', 0):.4f}  "
        f"parsed={overall.get('parsed_rate', 0):.4f}  "
        f"n={report['counts'].get('overall', {}).get('n', 0)}"
    )


if __name__ == "__main__":
    main()
