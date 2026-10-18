"""ICD-10-CM and SNOMED CT dictionary loaders, cached to parquet with a source-file hash stamp."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

_TAG_RE = re.compile(r"\(([^()]+)\)\s*$")

# Official SNOMED CT semantic tags, SNOMED CT Editorial Guide, "Semantic Tag" table:
# https://docs.snomed.org/snomed-ct-specifications/snomed-ct-editorial-guide/readme/authoring/general-naming-conventions/descriptions/fully-specified-name/index
SNOMED_SEMANTIC_TAGS = frozenset({
    "body structure", "cell", "cell structure", "morphologic abnormality", "finding", "disorder",
    "environment", "geographic location", "event", "observable entity", "organism", "clinical drug",
    "medicinal product", "medicinal product form", "physical object", "product", "physical force",
    "procedure", "regime/therapy", "qualifier value", "administration method", "basic dose form",
    "calculation", "disposition", "dose form", "intended site", "product name", "release characteristic",
    "role", "state of matter", "transformation", "supplier", "unit of presentation", "record artifact",
    "situation", "attribute", "core metadata concept", "foundation metadata concept", "link assertion",
    "linkage concept", "namespace concept", "OWL metadata concept", "social concept", "ethnic group",
    "occupation", "person", "racial group", "religion/philosophy", "navigational concept", "specimen",
    "staging scale", "assessment scale", "tumor staging", "substance",
})


def file_hash(path: str | Path) -> str:
    """SHA-256 of a file, used to stamp caches and indexes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _cached(src: str, cache_dir: str, name: str, loader) -> pd.DataFrame:
    """Return a parquet cache of loader(src), rebuilding when the source hash changes."""
    cache_dir_p = Path(cache_dir)
    cache_dir_p.mkdir(parents=True, exist_ok=True)
    pq, stamp = cache_dir_p / f"{name}.parquet", cache_dir_p / f"{name}.stamp.json"
    h = file_hash(src)
    if pq.exists() and stamp.exists() and json.loads(stamp.read_text()).get("hash") == h:
        return pd.read_parquet(pq)
    df = loader(src)
    df.to_parquet(pq, index=False)
    stamp.write_text(json.dumps({"hash": h, "source": Path(src).name}))
    return df


def _load_icd_raw(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, dtype=str).fillna("")
    df.columns = ["code", "description"]
    df["code"] = df["code"].str.strip()
    df["description"] = df["description"].str.strip()
    return df


def _load_snomed_raw(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, dtype={"snomed_code": str, "description": str})
    df = df.rename(columns={"snomed_code": "code"})
    df["code"] = df["code"].astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df["description_order"] = pd.to_numeric(df["description_order"], errors="coerce").fillna(0).astype(int)
    return df


@dataclass
class Dictionary:
    """One coding system: description rows plus per-code metadata."""

    system: str
    rows: pd.DataFrame                      # code, description (one row per description)
    codes: set[str] = field(default_factory=set)
    meta: pd.DataFrame | None = None        # indexed by code

    def __contains__(self, code: str) -> bool:
        return code in self.codes


def official_icd_titles(rows: pd.DataFrame) -> pd.Series:
    """Per code, the supplied description row that best matches the official CMS long description.

    Official text from simple-icd-10-cm (MIT, CMS ICD-10-CM data). Exact match after level-1
    normalization wins; otherwise the row with the highest rapidfuzz token_sort_ratio.
    """
    import simple_icd_10_cm as cm
    from rapidfuzz import fuzz

    from termmap.normalize import level1

    out = {}
    for code, g in rows.groupby("code", sort=True):
        descs = g["description"].tolist()
        try:
            off = level1(cm.get_description(code)) if cm.is_valid_item(code) else ""
        except Exception:  # code unknown to the package's release
            off = ""
        if not off or len(descs) == 1:
            out[code] = descs[0]
            continue
        norm = [level1(d) for d in descs]
        if off in norm:
            out[code] = descs[norm.index(off)]
        else:
            scores = [fuzz.token_sort_ratio(off, n) for n in norm]
            out[code] = descs[max(range(len(descs)), key=lambda i: (scores[i], -i))]
    return pd.Series(out)


def load_icd(path: str, cache_dir: str, title_mode: str = "first") -> Dictionary:
    """ICD-10-CM: drop range rows (codes containing '-'), dedupe, add hierarchy metadata.

    title_mode 'first' (v1) takes the first row per code; 'official' takes the row matching the
    official CMS long description (v2, audit item A5).
    """
    raw = _cached(path, cache_dir, "icd_raw", _load_icd_raw)
    rows = raw[~raw["code"].str.contains("-") & (raw["code"] != "")].drop_duplicates().reset_index(drop=True)
    codes = set(rows["code"])
    meta = pd.DataFrame({"code": sorted(codes)})
    meta["category"] = meta["code"].str[:3]
    meta["chapter"] = meta["code"].str[0]
    meta["depth"] = meta["code"].str.replace(".", "", regex=False).str.len() - 3
    prefixes = {c[:i] for c in codes for i in range(3, len(c))}
    meta["is_leaf"] = ~meta["code"].isin(prefixes)
    first = rows.drop_duplicates("code").set_index("code")["description"]
    if title_mode == "official":
        first = official_icd_titles(rows)
    meta["title"] = meta["code"].map(first)
    meta["category_title"] = meta["category"].map(first).fillna("")
    return Dictionary("ICD10CM", rows, codes, meta.set_index("code"))


def load_snomed(path: str, cache_dir: str, tag_mode: str = "any") -> Dictionary:
    """SNOMED CT: all descriptions plus semantic tag and preferred title per concept.

    tag_mode 'any' (v1) takes any trailing '(...)'. 'official' (v2, audit item A6) accepts a tag only
    from a description whose trailing parenthesis is an official semantic tag (the fully specified
    name). The preferred title stays description_order 1: it matches the gold display for 4,049 of
    4,335 gold concepts versus 3,593 for the tag-stripped FSN (measured 2026-10-18).
    """
    rows = _cached(path, cache_dir, "snomed_raw", _load_snomed_raw)
    rows = rows[rows["code"] != ""].reset_index(drop=True)
    tags = rows["description"].str.extract(_TAG_RE)[0]
    if tag_mode == "official":
        tags = tags.where(tags.isin(SNOMED_SEMANTIC_TAGS))
    tag_by_code = (pd.DataFrame({"code": rows["code"], "tag": tags}).dropna()
                   .drop_duplicates("code", keep="last").set_index("code")["tag"])
    codes = set(rows["code"])
    meta = pd.DataFrame({"code": sorted(codes)})
    meta["semantic_tag"] = meta["code"].map(tag_by_code).fillna("unknown")
    pref = rows.sort_values("description_order").drop_duplicates("code").set_index("code")["description"]
    meta["title"] = meta["code"].map(pref)
    return Dictionary("SNOMEDCT_US", rows[["code", "description", "description_order"]], codes, meta.set_index("code"))
