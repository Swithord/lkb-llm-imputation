from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _extract_metrics(report: dict) -> Dict[str, float]:
    return {
        "raw_parse_rate": float(report["strict_raw"]["metrics"]["overall"]["parsed_rate"]),
        "raw_accuracy": float(report["strict_raw"]["metrics"]["overall"]["accuracy"]),
        "normalized_accuracy": float(report["normalized"]["metrics"]["overall"]["accuracy"]),
    }


def _delta(new: float, old: float) -> dict:
    abs_delta = new - old
    rel = (abs_delta / old) if old else None
    return {"absolute": abs_delta, "relative": rel}


def main() -> None:
    p = argparse.ArgumentParser(description="Compare baseline vs prompt-v3 ablation evaluation reports.")
    p.add_argument("--baseline_report", type=str, required=True)
    p.add_argument("--v3_no_vote_report", type=str, required=True)
    p.add_argument("--v3_vote_report", type=str, required=True)
    p.add_argument("--out", type=str, default="report_prompt_v3.json")
    args = p.parse_args()

    baseline = _extract_metrics(_load(args.baseline_report))
    no_vote = _extract_metrics(_load(args.v3_no_vote_report))
    vote = _extract_metrics(_load(args.v3_vote_report))

    compare_vs_baseline = {
        "v3_no_vote": {
            k: _delta(no_vote[k], baseline[k]) for k in baseline.keys()
        },
        "v3_vote": {
            k: _delta(vote[k], baseline[k]) for k in baseline.keys()
        },
    }
    ablation_vote_vs_no_vote = {k: _delta(vote[k], no_vote[k]) for k in no_vote.keys()}

    out = {
        "prompt_version": "v3_strict_json",
        "metrics": {
            "baseline": baseline,
            "v3_no_vote_table": no_vote,
            "v3_with_vote_table": vote,
        },
        "raw_parse_compliance_improvement": {
            "v3_no_vote_vs_baseline": compare_vs_baseline["v3_no_vote"]["raw_parse_rate"],
            "v3_vote_vs_baseline": compare_vs_baseline["v3_vote"]["raw_parse_rate"],
            "v3_vote_vs_no_vote": ablation_vote_vs_no_vote["raw_parse_rate"],
        },
        "comparison_vs_baseline": compare_vs_baseline,
        "ablation_vote_table_vs_no_vote_table": ablation_vote_vs_no_vote,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote comparison report: {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
