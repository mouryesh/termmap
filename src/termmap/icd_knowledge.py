"""ICD-10-CM knowledge synonyms (v2, item C1): extra description rows for codes in the supplied dictionary.

- inclusion: tabular inclusion terms and 'includes' notes, simple-icd-10-cm 1.5.0 (MIT; CMS/CDC
  ICD-10-CM April 2026 tabular XML bundled with the package).
- index: CDC ICD-10-CM FY2026 Alphabetic Index to Diseases (icd10cm-index-2026.xml from
  icd10cm-table and index-2026.zip, NCHS FTP; US government work, public domain). Each main term +
  subterm path with a code becomes one synonym, written subterms-first ("Pneumonia > lobar" ->
  "lobar pneumonia"). 'see' cross-references take the code of the referenced path when it has one;
  'see also' adds nothing (the term already has its own code). Codes with '-' (incomplete) or not in
  the dictionary are dropped, as are manifestation codes.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET  # stdlib: no external entities; expat >= 2.4.1 limits entity expansion

import pandas as pd

_PARENS = re.compile(r"\s*\([^()]*\)")


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", _PARENS.sub("", s)).strip(" ,")


def inclusion_terms(codes: set[str]) -> pd.DataFrame:
    import simple_icd_10_cm as cm
    out = []
    for c in sorted(codes):
        if not cm.is_valid_item(c):
            continue
        for t in cm.get_inclusion_term(c) + cm.get_includes(c):
            if _clean(t):
                out.append((c, _clean(t)))
    return pd.DataFrame(out, columns=["code", "description"]).assign(source="inclusion")


def _title(el) -> str:
    """Title text without nemod (nonessential modifiers), notes, or alternate forms ("Failure, failed")."""
    t = el.find("title")
    text = re.split(r"\s*Note:", t.text or "")[0] if t is not None else ""
    return _clean(text.split(",")[0])


def index_terms(xml_path: str, codes: set[str]) -> pd.DataFrame:
    root = ET.parse(xml_path).getroot()
    by_path: dict[tuple[str, ...], str] = {}
    sees: list[tuple[tuple[str, ...], str]] = []

    def walk(el, path):
        path = path + (_title(el),)
        c = el.find("code")
        if c is not None and c.text:
            by_path[path] = c.text.strip()
        s = el.find("see")
        if s is not None and s.text:
            sees.append((path, s.text.strip()))
        for sub in el.findall("term"):
            walk(sub, path)

    for mt in root.iter("mainTerm"):
        walk(mt, ())
    lower = {tuple(p.lower() for p in k): v for k, v in by_path.items()}
    resolved = dict(by_path)
    for path, target in sees:  # "Pain, abdominal" -> ("pain", "abdominal")
        key = tuple(_clean(x).lower() for x in target.split(","))
        if path not in resolved and key in lower:
            resolved[path] = lower[key]
    out = [(code, " ".join(reversed(path))) for path, code in resolved.items()
           if "-" not in code and code in codes and all(path)]
    return pd.DataFrame(out, columns=["code", "description"]).assign(source="index")


def knowledge_terms(codes: set[str], sources: list[str], index_xml: str | None) -> pd.DataFrame:
    parts = []
    if "inclusion" in sources:
        parts.append(inclusion_terms(codes))
    if "index" in sources:
        parts.append(index_terms(index_xml, codes))
    df = pd.concat(parts, ignore_index=True)
    return df[df["description"].str.len() >= 3].reset_index(drop=True)
