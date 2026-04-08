import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geo_neighbors


class _Languoid:
    def __init__(self, gc, lat, lon):
        self.id = gc
        self.latitude = lat
        self.longitude = lon


class _Glottolog:
    def __init__(self, languoids):
        self._languoids = {l.id: l for l in languoids}

    def languoid(self, gc):
        return self._languoids.get(gc)


def test_compute_geographic_neighbours_basic(tmp_path):
    # Arrange a simple triangle: A(0,0), B(0,1), C(0,2)
    g = _Glottolog(
        [
            _Languoid("A", 0.0, 0.0),
            _Languoid("B", 0.0, 1.0),
            _Languoid("C", 0.0, 2.0),
        ]
    )
    geo_neighbors.glottolog = g

    languages = ["A", "B", "C"]
    out_file = tmp_path / "geo.json"

    # Act
    res = geo_neighbors.compute_geographic_neighbours(languages, k=1, output_path=str(out_file))

    # Assert: each node's nearest neighbor by longitude
    assert res["A"] == ["B"]
    assert res["B"] == ["A"]
    assert res["C"] == ["B"]

    # File written
    assert out_file.exists()
