from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# Logging
# -----------------------------
def setup_logger(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# -----------------------------
# URIEL+ Fetch
# -----------------------------
def _safe_call(obj: Any, method: str, *args, **kwargs) -> bool:
    """Call obj.method(*args, **kwargs) if it exists. Return True if called."""
    if hasattr(obj, method):
        getattr(obj, method)(*args, **kwargs)
        return True
    return False


def fetch_from_urielplus(
    integrate_all: bool,
    skip_glottolog: bool,
    use_glottocodes: bool,
    aggregation: Optional[str],
    fill_with_base_lang: bool,
    limit_langs: Optional[int],
    combine_sources: Optional[str],
) -> Tuple[np.ndarray, List[str], List[str], Dict[str, Any]]:
    """
    Fetch typological modality from URIEL+.

    Returns:
      X: (N, F) in {-1,0,1}
      feature_names: length F
      languages: length N
      info: dict with notes about which methods were available/called
    """
    try:
        from urielplus import urielplus  
    except Exception as e:
        raise ImportError(
            "Failed to import urielplus. Install it with `pip install urielplus` "
            "and ensure your Python version is compatible."
        ) from e

    info: Dict[str, Any] = {"called": {}, "notes": []}

    u = urielplus.URIELPlus()

    if aggregation is not None:
        called = _safe_call(u, "set_aggregation", aggregation)
        info["called"]["set_aggregation"] = called
        if not called:
            info["notes"].append("URielPlus.set_aggregation not found; skipped.")

    called_fill = _safe_call(u, "set_fill_with_base_lang", fill_with_base_lang)
    if not called_fill:
        called_fill = _safe_call(u, "set_fill_with_base_language", fill_with_base_lang)
    info["called"]["set_fill_with_base_lang"] = called_fill
    if not called_fill:
        info["notes"].append("URielPlus.set_fill_with_base_lang(u) not found; skipped.")

    if integrate_all:
        if skip_glottolog:
            # Integrate typological sources only; skip Glottolog (slow, and affects phylogeny).
            for m in ("integrate_saphon", "integrate_bdproto", "integrate_grambank", "integrate_apics", "integrate_ewave"):
                called = _safe_call(u, m)
                info["called"][m] = called
                if not called:
                    info["notes"].append(f"URielPlus.{m} not found; skipped.")
            called_inf = _safe_call(u, "inferred_features")
            info["called"]["inferred_features"] = called_inf
            if not called_inf:
                info["notes"].append("URielPlus.inferred_features not found; skipped.")
        else:
            called = _safe_call(u, "integrate_databases")
            info["called"]["integrate_databases"] = called
            if not called:
                info["notes"].append("URielPlus.integrate_databases not found; skipped.")

    if use_glottocodes:
        called = _safe_call(u, "set_glottocodes")
        info["called"]["set_glottocodes"] = called
        if not called:
            info["notes"].append("URielPlus.set_glottocodes not found; skipped.")

    # Retrieve typological modality arrays.
    # URIEL+ README commonly mentions these exact methods, but versions can differ.
    # Try a few fallbacks.
    X_raw = None
    feature_names = None
    languages = None

    # Data
    for m in ("get_typological_data_array", "get_typology_data_array", "get_typological_data"):
        if hasattr(u, m):
            X_raw = getattr(u, m)()
            info["called"][m] = True
            break
    if X_raw is None:
        raise AttributeError(
            "Could not find a URIEL+ method to fetch typological data. "
            "Tried: get_typological_data_array / get_typology_data_array / get_typological_data."
        )

    # Feature names
    for m in ("get_typological_features_array", "get_typology_features_array", "get_typological_features"):
        if hasattr(u, m):
            feature_names = list(getattr(u, m)())
            info["called"][m] = True
            break
    if feature_names is None:
        raise AttributeError(
            "Could not find a URIEL+ method to fetch typological feature names. "
            "Tried: get_typological_features_array / get_typology_features_array / get_typological_features."
        )

    # Language list
    for m in ("get_typological_languages_array", "get_typology_languages_array", "get_typological_languages"):
        if hasattr(u, m):
            languages = list(getattr(u, m)())
            info["called"][m] = True
            break
    if languages is None:
        raise AttributeError(
            "Could not find a URIEL+ method to fetch typological language codes. "
            "Tried: get_typological_languages_array / get_typology_languages_array / get_typological_languages."
        )

    X_raw = np.asarray(X_raw)

    # Some URIEL+ versions return typological data as (N, F, S) over sources.
    # Aggregate across sources to produce (N, F), unless combining by source later.
    if X_raw.ndim == 3 and not combine_sources:
        agg = aggregation
        if agg is None and hasattr(u, "get_aggregation"):
            try:
                agg = u.get_aggregation()
            except Exception:
                agg = None
        if agg is None:
            agg = "U"
            info["notes"].append("X was 3D; defaulted aggregation to 'U' (union).")
        if agg == "U":
            # Union: max over sources, missing=-1 stays -1 if all missing.
            X_raw = np.max(X_raw, axis=-1)
        elif agg == "A":
            # Average: mean over observed sources, then binarize by 0.5 threshold.
            X_avg = np.where(X_raw == -1, np.nan, X_raw)
            X_avg = np.nanmean(X_avg, axis=-1)
            X_avg = np.where(np.isnan(X_avg), -1, X_avg)
            X_raw = np.where(X_avg == -1, -1, (X_avg >= 0.5).astype(X_avg.dtype))
        else:
            raise ValueError(f"Unknown aggregation: {agg}. Use 'U' or 'A'.")

    # Optional language limiting (useful for quick tests)
    if limit_langs is not None and limit_langs > 0:
        languages = languages[:limit_langs]
        X_raw = X_raw[:limit_langs, ...]

    # Convert to {-1,0,1}
    if np.issubdtype(X_raw.dtype, np.floating):
        # handle NaN -> -1
        X = np.full(X_raw.shape, -1, dtype=np.int8)
        mask = np.isfinite(X_raw)
        X[mask] = X_raw[mask].astype(np.int8)
    else:
        X = X_raw.astype(np.int8)

    # Sanity check
    if not np.isin(X, [-1, 0, 1]).all():
        bad_vals = np.unique(X[~np.isin(X, [-1, 0, 1])])
        raise ValueError(f"Unexpected values in URIEL+ typology matrix: {bad_vals[:20]}")

    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"Feature name length mismatch: len(feature_names)={len(feature_names)} vs X.shape[1]={X.shape[1]}"
        )
    if len(languages) != X.shape[0]:
        raise ValueError(
            f"Language length mismatch: len(languages)={len(languages)} vs X.shape[0]={X.shape[0]}"
        )

    return X, feature_names, languages, info


# -----------------------------
# Phi correlation (missing-aware)
# -----------------------------
def compute_phi_corr(
    X: np.ndarray,
    min_support: int = 50,
    alpha: float = 0.0,
    support_shrinkage: float = 0.0,
    chunk: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute missing-aware pairwise Phi correlation between binary features.

    X: (N, F) with values in {-1,0,1}
    min_support: minimum co-observed languages needed to compute phi; else corr=0
    alpha: Laplace smoothing added to contingency counts (e.g., 0.5); 0 disables
    support_shrinkage: shrink low-support correlations toward 0 using
        phi *= support / (support + support_shrinkage). 0 disables.
    chunk: feature chunk size for memory-friendly computation

    Returns:
      C_phi: (F, F) float32
      support: (F, F) int32 (co-observed counts)
    """
    assert X.ndim == 2, "X must be 2D (N, F)"
    N, F = X.shape
    logging.info(
        "Computing Phi correlation: "
        f"N={N}, F={F}, min_support={min_support}, alpha={alpha}, support_shrinkage={support_shrinkage}"
    )

    obs = (X != -1)
    is1 = (X == 1)
    is0 = (X == 0)

    C = np.zeros((F, F), dtype=np.float32)
    S = np.zeros((F, F), dtype=np.int32)

    for i0 in range(0, F, chunk):
        i1 = min(F, i0 + chunk)
        logging.info(f"  block rows [{i0}:{i1})")

        is1_i = is1[:, i0:i1]
        is0_i = is0[:, i0:i1]

        for j0 in range(i0, F, chunk):
            j1 = min(F, j0 + chunk)

            is1_j = is1[:, j0:j1]
            is0_j = is0[:, j0:j1]

            # Counts over co-observed languages (missing values are excluded because is1/is0 are False on missing)
            n11 = (is1_i.T @ is1_j).astype(np.float64)
            n10 = (is1_i.T @ is0_j).astype(np.float64)
            n01 = (is0_i.T @ is1_j).astype(np.float64)
            n00 = (is0_i.T @ is0_j).astype(np.float64)

            sup = (n11 + n10 + n01 + n00).astype(np.int32)

            if alpha > 0:
                n11 += alpha
                n10 += alpha
                n01 += alpha
                n00 += alpha

            n1_ = n11 + n10
            n0_ = n01 + n00
            n_1 = n11 + n01
            n_0 = n10 + n00

            denom = np.sqrt(n1_ * n0_ * n_1 * n_0)
            num = (n11 * n00) - (n10 * n01)

            phi = np.zeros_like(num, dtype=np.float64)
            valid = (denom > 0) & (sup >= min_support)
            phi[valid] = num[valid] / denom[valid]
            if support_shrinkage > 0:
                shrink = sup.astype(np.float64) / (sup.astype(np.float64) + support_shrinkage)
                phi[valid] *= shrink[valid]

            C[i0:i1, j0:j1] = phi.astype(np.float32)
            S[i0:i1, j0:j1] = sup

            if j0 != i0:
                C[j0:j1, i0:i1] = C[i0:i1, j0:j1].T
                S[j0:j1, i0:i1] = S[i0:i1, j0:j1].T

    np.fill_diagonal(C, 1.0)
    # Put per-feature observed counts on diagonal of support
    np.fill_diagonal(S, np.sum(obs, axis=0).astype(np.int32))
    return C, S


def compute_phi_corr_by_source(
    X: np.ndarray,
    min_support: int = 50,
    alpha: float = 0.0,
    support_shrinkage: float = 0.0,
    chunk: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute phi correlations per source and combine via support-weighted average.

    X: (N, F, S) with values in {-1,0,1}
    Returns:
      C: (F, F) float32
      S: (F, F) int32 total support across sources
    """
    assert X.ndim == 3, "X must be 3D (N, F, S)"
    N, F, S = X.shape
    logging.info(
        "Computing Phi correlation by source: "
        f"N={N}, F={F}, S={S}, min_support={min_support}, alpha={alpha}, support_shrinkage={support_shrinkage}"
    )

    C_sum = np.zeros((F, F), dtype=np.float64)
    S_sum = np.zeros((F, F), dtype=np.float64)

    for s in range(S):
        Xs = X[:, :, s]
        # Skip empty sources
        if not np.any(Xs != -1):
            continue
        C_s, S_s = compute_phi_corr(
            Xs,
            min_support=min_support,
            alpha=alpha,
            support_shrinkage=support_shrinkage,
            chunk=chunk,
        )
        C_sum += C_s.astype(np.float64) * S_s.astype(np.float64)
        S_sum += S_s.astype(np.float64)

    C = np.zeros((F, F), dtype=np.float32)
    mask = S_sum > 0
    C[mask] = (C_sum[mask] / S_sum[mask]).astype(np.float32)
    np.fill_diagonal(C, 1.0)
    return C, S_sum.astype(np.int32)


# -----------------------------
# Impute + correlate (SoftImpute preferred)
# -----------------------------
def softimpute_or_fallback(
    X: np.ndarray,
    max_iters: int = 200,
    rank: int = 50,
    shrinkage: Optional[float] = None,
    random_state: int = 0,
) -> np.ndarray:
    """
    Impute missing values and return X_hat in [0,1].

    Preferred: fancyimpute.SoftImpute
    Fallback : mean-impute + truncated SVD reconstruction (scikit-learn)
    """
    Xf = X.astype(np.float32).copy()
    Xf[Xf == -1] = np.nan

    try:
        from fancyimpute import SoftImpute  # type: ignore

        logging.info("Using fancyimpute.SoftImpute.")
        model = SoftImpute(
            max_iters=max_iters,
            shrinkage_value=shrinkage,
            init_fill_method="mean",
            verbose=False,
        )
        X_hat = model.fit_transform(Xf)
        X_hat = np.clip(X_hat, 0.0, 1.0).astype(np.float32)
        return X_hat
    except Exception as e:
        logging.warning(f"SoftImpute not available or failed ({e}). Falling back to mean+SVD.")

    # Fallback: mean-impute then truncated SVD reconstruction
    try:
        from sklearn.impute import SimpleImputer  # type: ignore
        from sklearn.decomposition import TruncatedSVD  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Fallback imputation requires scikit-learn. Install it with `pip install scikit-learn`, "
            "or install fancyimpute for SoftImpute."
        ) from e

    imp = SimpleImputer(strategy="mean")
    X_filled = imp.fit_transform(Xf)

    n_components = min(rank, X_filled.shape[1] - 1)
    if n_components < 1:
        # Degenerate case: only 1 feature
        X_hat = np.clip(X_filled, 0.0, 1.0).astype(np.float32)
        return X_hat

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    Z = svd.fit_transform(X_filled)
    X_hat = (Z @ svd.components_).astype(np.float32)
    X_hat = np.clip(X_hat, 0.0, 1.0)
    return X_hat


def compute_corr_after_impute(
    X: np.ndarray,
    max_iters: int,
    rank: int,
    shrinkage: Optional[float],
    random_state: int,
) -> np.ndarray:
    """
    Impute missing values then compute Pearson correlation between feature columns.
    """
    X_hat = softimpute_or_fallback(
        X,
        max_iters=max_iters,
        rank=rank,
        shrinkage=shrinkage,
        random_state=random_state,
    )

    # Pearson correlation, vectorized
    Xc = X_hat - X_hat.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(1, (Xc.shape[0] - 1))
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    C = cov / (std[:, None] * std[None, :])
    C = np.clip(C, -1.0, 1.0).astype(np.float32)
    np.fill_diagonal(C, 1.0)
    return C


# -----------------------------
# Output helpers
# -----------------------------
def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()

    # URIEL+ fetch options
    p.add_argument("--out", type=str, required=True, help="Output directory")
    p.add_argument("--integrate_all", action="store_true",
                   help="Call u.integrate_databases() to expand typology features/sources")
    p.add_argument("--skip_glottolog", action="store_true",
                   help="When --integrate_all is set, skip Glottolog integration (slower, affects phylogeny only)")
    p.add_argument("--use_glottocodes", action="store_true",
                   help="Call u.set_glottocodes() so language codes are glottocodes")
    p.add_argument("--aggregation", type=str, default=None, choices=[None, "U", "A"],
                   help="URIEL+ aggregation if supported: U=union, A=average (version-dependent)")
    p.add_argument("--combine_sources", type=str, default=None, choices=[None, "weighted"],
                   help="If typological data is 3D (N,F,S), combine per-source correlations. "
                        "Use 'weighted' for support-weighted averaging; disables aggregation.")
    p.add_argument("--fill_with_base_lang", action="store_true",
                   help="URIEL+ fill missing from base language if supported (version-dependent)")
    p.add_argument("--limit_langs", type=int, default=None,
                   help="Optional: limit number of languages for quick tests")

    # Phi correlation params
    p.add_argument("--min_support", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.0, help="Laplace smoothing (e.g., 0.5)")
    p.add_argument("--support_shrinkage", type=float, default=0.0,
                   help="Shrink low-support correlations toward 0; 0 disables")
    p.add_argument("--chunk", type=int, default=128)

    # Imputation params
    p.add_argument("--compute_imputed", action="store_true",
                   help="Also compute imputed correlation matrix")
    p.add_argument("--softimpute_max_iters", type=int, default=200)
    p.add_argument("--softimpute_rank", type=int, default=50)
    p.add_argument("--softimpute_shrinkage", type=float, default=-1.0,
                   help="SoftImpute shrinkage_value; -1 means None")
    p.add_argument("--seed", type=int, default=0)

    # Misc
    p.add_argument("--log_level", type=str, default="INFO")

    args = p.parse_args()
    setup_logger(args.log_level)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Fetch from URIEL+
    X, feature_names, languages, info = fetch_from_urielplus(
        integrate_all=args.integrate_all,
        skip_glottolog=args.skip_glottolog,
        use_glottocodes=args.use_glottocodes,
        aggregation=args.aggregation,
        fill_with_base_lang=args.fill_with_base_lang,
        limit_langs=args.limit_langs,
        combine_sources=args.combine_sources,
    )

    # Save raw fetch results
    np.save(out_dir / "X.npy", X)
    save_json(out_dir / "feature_names.json", feature_names)
    save_json(out_dir / "languages.json", languages)

    # 2) Phi correlation + support
    if args.combine_sources:
        if X.ndim != 3:
            raise ValueError("combine_sources requires 3D typological data (N, F, S).")
        C_phi, support = compute_phi_corr_by_source(
            X,
            min_support=args.min_support,
            alpha=args.alpha,
            support_shrinkage=args.support_shrinkage,
            chunk=args.chunk,
        )
    else:
        C_phi, support = compute_phi_corr(
            X,
            min_support=args.min_support,
            alpha=args.alpha,
            support_shrinkage=args.support_shrinkage,
            chunk=args.chunk,
        )
    np.save(out_dir / "C_phi.npy", C_phi)
    np.save(out_dir / "support.npy", support)

    # 3) Optional: impute then correlate
    C_imputed = None
    if args.compute_imputed:
        shrinkage = None if args.softimpute_shrinkage < 0 else float(args.softimpute_shrinkage)
        C_imputed = compute_corr_after_impute(
            X,
            max_iters=args.softimpute_max_iters,
            rank=args.softimpute_rank,
            shrinkage=shrinkage,
            random_state=args.seed,
        )
        np.save(out_dir / "C_imputed.npy", C_imputed)

    meta: Dict[str, Any] = {
        "urielplus": {
            "integrate_all": bool(args.integrate_all),
            "skip_glottolog": bool(args.skip_glottolog),
            "use_glottocodes": bool(args.use_glottocodes),
            "aggregation": args.aggregation,
            "combine_sources": args.combine_sources,
            "fill_with_base_lang": bool(args.fill_with_base_lang),
            "limit_langs": args.limit_langs,
            "fetch_info": info,
        },
        "phi": {
            "min_support": args.min_support,
            "alpha": args.alpha,
            "support_shrinkage": args.support_shrinkage,
            "chunk": args.chunk,
        },
        "imputed": {
            "enabled": bool(args.compute_imputed),
            "softimpute_max_iters": args.softimpute_max_iters,
            "softimpute_rank": args.softimpute_rank,
            "softimpute_shrinkage": None if args.softimpute_shrinkage < 0 else float(args.softimpute_shrinkage),
            "seed": args.seed,
        },
        "shapes": {
            "X": [int(X.shape[0]), int(X.shape[1])],
            "C_phi": [int(C_phi.shape[0]), int(C_phi.shape[1])],
            "support": [int(support.shape[0]), int(support.shape[1])],
            "C_imputed": None if C_imputed is None else [int(C_imputed.shape[0]), int(C_imputed.shape[1])],
        },
        "value_counts": {
            "-1_missing": int(np.sum(X == -1)),
            "0_absent": int(np.sum(X == 0)),
            "1_present": int(np.sum(X == 1)),
        },
    }
    save_json(out_dir / "meta.json", meta)

    logging.info(f"Done. Wrote outputs to: {out_dir}")
    logging.info("Files: X.npy, feature_names.json, languages.json, C_phi.npy, support.npy"
                 + (", C_imputed.npy" if args.compute_imputed else ""))


if __name__ == "__main__":
    main()
