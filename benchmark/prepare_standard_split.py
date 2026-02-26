from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _split_languages(languages: list[str], seed: int, train_frac: float, dev_frac: float) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(languages, dtype=object)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_dev = int(n * dev_frac)
    n_test = n - n_train - n_dev

    train = shuffled[:n_train].tolist()
    dev = shuffled[n_train : n_train + n_dev].tolist()
    test = shuffled[n_train + n_dev :].tolist()
    if len(test) != n_test:
        raise RuntimeError("Split size mismatch.")
    return {"train": train, "dev": dev, "test": test}


def _make_mcar_mask(observed: np.ndarray, mask_rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    observed_idx = np.argwhere(observed)
    n_observed = observed_idx.shape[0]
    n_mask = int(round(mask_rate * n_observed))
    pick = rng.choice(n_observed, size=n_mask, replace=False)

    mask = np.zeros(observed.shape, dtype=bool)
    chosen = observed_idx[pick]
    mask[chosen[:, 0], chosen[:, 1]] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Create standardized language split and MCAR mask.")
    parser.add_argument("--typology_csv", type=str, default="output/uriel+_typological.csv")
    parser.add_argument("--split_out", type=str, default="splits_v1.json")
    parser.add_argument("--mask_out", type=str, default="mask_mcar20_seed42.npy")
    parser.add_argument("--mask_meta_out", type=str, default="mask_mcar20_seed42.json")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--mask_seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--dev_frac", type=float, default=0.1)
    parser.add_argument("--mask_rate", type=float, default=0.2)
    args = parser.parse_args()

    if args.train_frac <= 0 or args.dev_frac <= 0 or args.train_frac + args.dev_frac >= 1:
        raise ValueError("Invalid split fractions; need train_frac>0, dev_frac>0, train_frac+dev_frac<1.")

    typology_df = pd.read_csv(args.typology_csv, index_col=0)
    typology_df.index = typology_df.index.astype(str)
    observed = typology_df.to_numpy() != -1

    lang_has_observed = observed.any(axis=1)
    split_languages = typology_df.index[lang_has_observed].tolist()
    splits = _split_languages(split_languages, args.split_seed, args.train_frac, args.dev_frac)

    split_payload = {
        "version": "v1",
        "split_level": "language",
        "typology_csv": args.typology_csv,
        "split_seed": args.split_seed,
        "fractions": {
            "train": args.train_frac,
            "dev": args.dev_frac,
            "test": 1.0 - args.train_frac - args.dev_frac,
        },
        "n_languages_total": int(typology_df.shape[0]),
        "n_languages_with_observed": int(len(split_languages)),
        "n_languages_without_observed": int(np.sum(~lang_has_observed)),
        "splits": splits,
    }

    split_path = Path(args.split_out)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    mask = _make_mcar_mask(observed, mask_rate=args.mask_rate, seed=args.mask_seed)
    mask_path = Path(args.mask_out)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mask_path, mask)

    n_observed = int(np.sum(observed))
    n_masked = int(np.sum(mask))
    meta_payload = {
        "name": f"mcar_{int(args.mask_rate * 100)}",
        "typology_csv": args.typology_csv,
        "mask_file": str(mask_path),
        "mask_seed": args.mask_seed,
        "mask_rate": args.mask_rate,
        "shape": [int(mask.shape[0]), int(mask.shape[1])],
        "n_observed_entries": n_observed,
        "n_masked_entries": n_masked,
        "masked_fraction_of_observed": (n_masked / n_observed) if n_observed else 0.0,
    }
    meta_path = Path(args.mask_meta_out)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    print(f"Wrote split file: {split_path}")
    print(f"Wrote mask file: {mask_path}")
    print(f"Wrote mask meta: {meta_path}")


if __name__ == "__main__":
    main()
