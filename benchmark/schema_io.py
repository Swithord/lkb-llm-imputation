from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def normalize_gold_record(rec: dict) -> dict:
    if "id" in rec and "gold_value" in rec:
        out = dict(rec)
        out["id"] = str(out["id"])
        out["gold_value"] = str(out["gold_value"])
        out["allowed_values"] = [str(v) for v in out.get("allowed_values", ["0", "1"])]
        return out

    if "example_id" in rec and "gold" in rec:
        return {
            "id": str(rec["example_id"]),
            "resource_group": str(rec.get("resource_group", "")),
            "language": str(rec.get("language", rec.get("language_id", ""))),
            "feature": str(rec.get("feature", rec.get("feature_id", ""))),
            "gold_value": str(rec["gold"]),
            "allowed_values": [str(v) for v in rec.get("allowed_values", ["0", "1"])],
        }

    raise ValueError("Unrecognized gold/example record schema.")


def normalize_prediction_record(rec: dict) -> dict:
    out = dict(rec)
    pid = out.get("id", out.get("example_id"))
    if pid is None:
        return {}
    out["id"] = str(pid)
    if "language" not in out and "language_id" in out:
        out["language"] = str(out["language_id"])
    if "feature" not in out and "feature_id" in out:
        out["feature"] = str(out["feature_id"])
    return out


def load_gold_jsonl(path: str) -> Dict[str, dict]:
    gold: Dict[str, dict] = {}
    for rec in _read_jsonl(path):
        normalized = normalize_gold_record(rec)
        gold[normalized["id"]] = normalized
    return gold


def load_predictions_jsonl(path: str) -> Dict[str, dict]:
    pred: Dict[str, dict] = {}
    for rec in _read_jsonl(path):
        normalized = normalize_prediction_record(rec)
        pid = normalized.get("id")
        if pid:
            pred[str(pid)] = normalized
    return pred
