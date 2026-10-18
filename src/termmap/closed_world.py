"""Closed-world output check: every prediction row is present, non-blank and a real dictionary code."""
from __future__ import annotations

import re

import pandas as pd

OUTPUT_COLUMNS = ["entity", "chunk_text", "icd10cm_code", "snomed_code"]
# ICD-10-CM: letter, digit, alphanumeric, then optionally "." and 1-4 alphanumerics
ICD_RE = re.compile(r"^[A-Z][0-9][0-9A-Z](\.[0-9A-Z]{1,4})?$")


def check(pred: pd.DataFrame, inp: pd.DataFrame, icd_codes: set[str], snomed_codes: set[str]) -> dict:
    """Raise AssertionError on any violation; return counts for the log."""
    assert list(pred.columns) == OUTPUT_COLUMNS, f"columns {list(pred.columns)}"
    assert len(pred) == len(inp), f"row count {len(pred)} != input {len(inp)}"
    assert (pred["entity"].values == inp["entity"].values).all(), "row order / entity changed"
    assert (pred["chunk_text"].values == inp["chunk_text"].values).all(), "row order / chunk changed"
    for col in ("icd10cm_code", "snomed_code"):
        v = pred[col]
        assert not v.isna().any() and (v.astype(str).str.strip() != "").all(), f"blank {col}"
    bad_icd = sorted(set(pred["icd10cm_code"]) - icd_codes)
    bad_snomed = sorted(set(pred["snomed_code"]) - snomed_codes)
    assert not bad_icd, f"ICD codes not in dictionary: {bad_icd[:5]}"
    assert not bad_snomed, f"SNOMED codes not in dictionary: {bad_snomed[:5]}"
    bad_fmt = sorted(c for c in set(pred["icd10cm_code"]) if not ICD_RE.match(c))
    assert not bad_fmt, f"malformed ICD codes: {bad_fmt[:5]}"
    return {"rows": len(pred), "distinct_icd": int(pred["icd10cm_code"].nunique()),
            "distinct_snomed": int(pred["snomed_code"].nunique())}
