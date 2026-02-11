from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Select top-k correlated features per feature.")
    p.add_argument("--corr", type=str, required=True, help="Path to correlation matrix .npy")
    p.add_argument("--features", type=str, required=True, help="Path to feature_names.json")
    p.add_argument("--k", type=int, default=5, help="Top-k per feature")
    p.add_argument("--out", type=str, required=True, help="Output CSV path")
    p.add_argument("--absolute", action="store_true", help="Rank by absolute correlation")
    p.add_argument("--min_corr", type=float, default=None, help="Optional minimum correlation threshold")
    args = p.parse_args()

    C = np.load(args.corr)
    feats = json.loads(Path(args.features).read_text())

    if C.shape[0] != C.shape[1]:
        raise ValueError(f"Correlation matrix must be square; got {C.shape}")
    if C.shape[0] != len(feats):
        raise ValueError(f"Feature length mismatch: C is {C.shape[0]} but features is {len(feats)}")

    rows = []
    for i, feat in enumerate(feats):
        vals = C[i].copy()
        vals[i] = np.nan  # exclude self

        if args.absolute:
            order = np.argsort(-np.abs(vals))
        else:
            order = np.argsort(-vals)

        count = 0
        for j in order:
            v = vals[j]
            if np.isnan(v):
                continue
            if args.min_corr is not None and v < args.min_corr:
                continue
            rows.append(
                {
                    "feature": feat,
                    "rank": count + 1,
                    "other_feature": feats[j],
                    "corr": float(v),
                }
            )
            count += 1
            if count >= args.k:
                break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
