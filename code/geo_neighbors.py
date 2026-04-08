import json
import logging
import numpy as np


def compute_geographic_neighbours(
    languages, k=5, output_path="geographic_neighbours.json", glottolog=None
):
    """
    Compute the k nearest geographic neighbours using Glottolog.
    :param languages: List[str], list of glottocodes (of languages covered by URIEL+)
    :param k: int, number of nearest neighbours to retrieve
    :param output_path: str, path to save the geographic neighbours JSON file
    """
    if k <= 0:
        result = {lang: [] for lang in languages}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    if glottolog is None:
        glottolog = globals().get("glottolog")
    if glottolog is None:
        raise RuntimeError(
            "Glottolog object is not initialized. "
            "Pass it via compute_geographic_neighbours(..., glottolog=...)."
        )

    logger = logging.getLogger(__name__)

    languoid_map = {}
    if hasattr(glottolog, "languoids"):
        # Build one lookup table instead of repeatedly querying per language.
        for l in glottolog.languoids():
            lid = getattr(l, "id", None)
            if lid is not None:
                languoid_map[lid] = l

    def _get_languoid(gc):
        if gc in languoid_map:
            return languoid_map[gc]
        if hasattr(glottolog, "languoid"):
            return glottolog.languoid(gc)
        return None

    def _get_lat_lon(languoid_obj):
        if languoid_obj is None:
            return None, None
        lat = getattr(languoid_obj, "latitude", None)
        lon = getattr(languoid_obj, "longitude", None)
        if lat is None or lon is None:
            return None, None
        return float(lat), float(lon)

    logger.info("Geographic neighbours: loading coordinates for %d languages", len(languages))
    coords = {lang: _get_lat_lon(_get_languoid(lang)) for lang in languages}

    valid_langs = []
    valid_coords = []
    for lang in languages:
        lat, lon = coords.get(lang, (None, None))
        if lat is None or lon is None:
            continue
        valid_langs.append(lang)
        valid_coords.append((lat, lon))

    neighbours = {lang: [] for lang in languages}
    if not valid_langs:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(neighbours, f, ensure_ascii=False, indent=2)
        return neighbours

    arr = np.asarray(valid_coords, dtype=np.float64)
    lat = np.radians(arr[:, 0])
    lon = np.radians(arr[:, 1])
    cos_lat = np.cos(lat)
    earth_r = 6371.0

    n = len(valid_langs)
    logger.info("Geographic neighbours: %d/%d languages have coordinates", n, len(languages))
    k_eff = min(k, max(n - 1, 0))
    if k_eff == 0:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(neighbours, f, ensure_ascii=False, indent=2)
        return neighbours

    # Fast path via BallTree (haversine), fallback to vectorized brute force.
    try:
        from sklearn.neighbors import BallTree  # type: ignore

        logger.info("Geographic neighbours: querying BallTree")
        coords_rad = np.column_stack([lat, lon])
        tree = BallTree(coords_rad, metric="haversine")
        dist_rad, ind = tree.query(coords_rad, k=min(k_eff + 1, n))
        for i, lang in enumerate(valid_langs):
            cand = []
            for d_r, j in zip(dist_rad[i], ind[i]):
                if j == i:
                    continue
                cand.append((float(d_r * earth_r), valid_langs[int(j)]))
                if len(cand) == k_eff:
                    break
            cand.sort(key=lambda x: (x[0], x[1]))
            neighbours[lang] = [gc for _, gc in cand]
    except Exception:
        logger.info("Geographic neighbours: BallTree unavailable, using NumPy fallback")
        for i, lang in enumerate(valid_langs):
            dlat = lat - lat[i]
            dlon = lon - lon[i]
            a = (
                np.sin(dlat / 2.0) ** 2
                + cos_lat[i] * cos_lat * (np.sin(dlon / 2.0) ** 2)
            )
            d = 2.0 * earth_r * np.arcsin(np.minimum(1.0, np.sqrt(a)))
            d[i] = np.inf

            idx = np.argpartition(d, k_eff)[:k_eff]
            cand = [(float(d[j]), valid_langs[j]) for j in idx]
            cand.sort(key=lambda x: (x[0], x[1]))
            neighbours[lang] = [gc for _, gc in cand[:k_eff]]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(neighbours, f, ensure_ascii=False, indent=2)
    logger.info("Geographic neighbours: wrote %s", output_path)

    return neighbours
