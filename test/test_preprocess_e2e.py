import os
import sys
import json

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import preprocess
import geo_neighbors


class _Languoid:
    def __init__(
        self,
        lid,
        level=None,
        parent=None,
        name=None,
        iso=None,
        latitude=None,
        longitude=None,
        macroareas=None,
        countries=None,
    ):
        self.id = lid
        self.level = level
        self.parent = parent
        self.name = name or lid
        self.iso = iso
        self.latitude = latitude
        self.longitude = longitude
        self.macroareas = macroareas or []
        self.countries = countries or []


class _Glottolog:
    def __init__(self, languoids):
        self._languoids = {l.id: l for l in languoids}

    def languoid(self, gc):
        return self._languoids.get(gc)

    def languoids(self):
        return list(self._languoids.values())


def _build_stub_glottolog():
    root = _Languoid("root", level="family")
    famB = _Languoid("famB", level="family", parent=root)
    famC = _Languoid("famC", level="family", parent=root)

    langA = _Languoid(
        "langA",
        level="language",
        parent=root,
        latitude=0.0,
        longitude=0.0,
        macroareas=["X"],
        countries=["AA"],
        iso="aaa",
    )
    langB = _Languoid(
        "langB",
        level="language",
        parent=famB,
        latitude=0.0,
        longitude=1.0,
        macroareas=["X"],
        countries=["BB"],
        iso="bbb",
    )
    langC = _Languoid(
        "langC",
        level="language",
        parent=famC,
        latitude=0.0,
        longitude=2.0,
        macroareas=["X"],
        countries=["CC"],
        iso="ccc",
    )

    diaA = _Languoid("diaA", level="dialect", parent=langA)
    diaB = _Languoid("diaB", level="dialect", parent=langB)
    diaC = _Languoid("diaC", level="dialect", parent=langC)

    return _Glottolog([root, famB, famC, langA, langB, langC, diaA, diaB, diaC])


def test_preprocess_end_to_end(tmp_path):
    glottolog = _build_stub_glottolog()
    preprocess.glottolog = glottolog
    geo_neighbors.glottolog = glottolog

    langs = ["langA", "langB", "langC"]

    # 1) metadata
    meta_path = tmp_path / "metadata.csv"
    meta_df = preprocess.get_metadata(langs, glottolog, output_path=str(meta_path))
    assert meta_path.exists()
    assert list(meta_df.index) == langs
    assert meta_df.loc["langA", "iso639_3"] == "aaa"

    # 2) correlation matrix (precomputed)
    corr_path = tmp_path / "correlation_matrix.csv"
    corr_df_in = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], columns=["f1", "f2"], index=["f1", "f2"])
    corr_df_in.to_csv(corr_path)
    corr_df = preprocess.compute_correlation_matrix(pd.DataFrame(), output_path=str(corr_path))
    assert corr_df.shape == (2, 2)

    # 3) genetic neighbours
    gen_path = tmp_path / "genetic.json"
    gen = preprocess.compute_genetic_neighbours(langs, k=1, output_path=str(gen_path))
    assert gen_path.exists()
    assert set(gen.keys()) == set(langs)
    assert all(len(v) <= 1 for v in gen.values())

    # 4) geographic neighbours
    geo_path = tmp_path / "geo.json"
    geo = preprocess.compute_geographic_neighbours(langs, k=1, output_path=str(geo_path))
    assert geo_path.exists()
    assert geo["langA"] == ["langB"]
    assert geo["langB"] == ["langA"]
    assert geo["langC"] == ["langB"]

    # JSON outputs are valid
    with open(gen_path, "r", encoding="utf-8") as f:
        json.load(f)
    with open(geo_path, "r", encoding="utf-8") as f:
        json.load(f)
