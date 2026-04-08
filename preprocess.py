from __future__ import annotations

import importlib.util
from pathlib import Path


_IMPL_PATH = Path(__file__).resolve().parent / "code" / "preprocessing.py"
_SPEC = importlib.util.spec_from_file_location("legacy_preprocess_impl", _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load preprocessing module from {_IMPL_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

glottolog = getattr(_MODULE, "glottolog", None)


def compute_genetic_neighbours(*args, **kwargs):
    _MODULE.glottolog = globals().get("glottolog")
    return _MODULE.compute_genetic_neighbours(*args, **kwargs)


def compute_geographic_neighbours(*args, **kwargs):
    _MODULE.glottolog = globals().get("glottolog")
    return _MODULE.compute_geographic_neighbours(*args, **kwargs)


for _name in dir(_MODULE):
    if _name.startswith("__") or _name in {"compute_genetic_neighbours", "compute_geographic_neighbours"}:
        continue
    globals()[_name] = getattr(_MODULE, _name)

__all__ = [name for name in globals() if not name.startswith("_")]
