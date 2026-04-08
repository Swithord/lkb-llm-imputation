from pathlib import Path

import pandas as pd


def get_metadata(languages, glottolog, output_path="metadata.csv"):
    """
    Get Glottolog metadata, and save it as a CSV file.
    :param languages: List[str], list of glottocodes (of languages covered by URIEL+)
    :param output_path: str, path to save the metadata CSV file
    :return:
    """
    if glottolog is None:
        raise RuntimeError("glottolog object is required.")

    base_path = (
        getattr(glottolog, "repos", None)
        or getattr(glottolog, "dir", None)
        or getattr(glottolog, "path", None)
    )
    if base_path is not None:
        base_path = Path(str(base_path))
        if not (base_path / "languoids" / "tree").exists():
            raise RuntimeError(
                f"Invalid Glottolog repo at {base_path}. "
                "Expected languoids/tree to exist."
            )

    def _get_languoid(gc):
        if hasattr(glottolog, "languoid"):
            return glottolog.languoid(gc)
        if hasattr(glottolog, "languoids"):
            for l in glottolog.languoids():
                if getattr(l, "id", None) == gc:
                    return l
        return None

    def _as_name_list(values):
        if not values:
            return []
        out = []
        for v in values:
            if isinstance(v, str):
                out.append(v)
            else:
                out.append(getattr(v, "name", None) or getattr(v, "id", None) or str(v))
        return out

    rows = []
    for gc in languages:
        l = _get_languoid(gc)
        if l is None:
            rows.append(
                {
                    "glottocode": gc,
                    "name": None,
                    "level": None,
                    "iso639_3": None,
                    "family_id": None,
                    "family_name": None,
                    "parent_id": None,
                    "parent_name": None,
                    "macroareas": None,
                    "countries": None,
                    "latitude": None,
                    "longitude": None,
                }
            )
            continue

        family = getattr(l, "family", None)
        parent = getattr(l, "parent", None)

        macroareas = _as_name_list(getattr(l, "macroareas", None))
        countries = _as_name_list(getattr(l, "countries", None))

        rows.append(
            {
                "glottocode": gc,
                "name": getattr(l, "name", None),
                "level": getattr(l, "level", None),
                "iso639_3": getattr(l, "iso", None),
                "family_id": getattr(family, "id", None) if family else None,
                "family_name": getattr(family, "name", None) if family else None,
                "parent_id": getattr(parent, "id", None) if parent else None,
                "parent_name": getattr(parent, "name", None) if parent else None,
                "macroareas": ";".join(macroareas) if macroareas else None,
                "countries": ";".join(countries) if countries else None,
                "latitude": getattr(l, "latitude", None),
                "longitude": getattr(l, "longitude", None),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("glottocode")

    df.to_csv(output_path)
    return df
