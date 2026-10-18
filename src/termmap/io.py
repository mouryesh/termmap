"""Reading training data and inference input; writing predictions."""
from __future__ import annotations

import re

import pandas as pd

ICD = "ICD10CM"
SNOMED = "SNOMEDCT_US"
SYSTEMS = (ICD, SNOMED)

_GROUP_RE = re.compile(r"[^a-z0-9]+")


def load_train(path: str) -> pd.DataFrame:
    """Load the long-format training CSV with every column as string (codes keep leading zeros)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected = {"entity", "chunk_text", "coding_system", "code", "display", "subconcept"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"training file missing columns: {sorted(missing)}")
    df["code"] = df["code"].str.strip()
    return df


def pair_instances(train: pd.DataFrame) -> pd.DataFrame:
    """Pivot long rows to one row per raw (entity, chunk_text) with icd_code / snomed_code columns.

    If a term instance has several codes for one system, the first in file order is kept.
    """
    df = train[["entity", "chunk_text", "coding_system", "code", "display"]].copy()
    df["_order"] = range(len(df))
    out = (
        df.sort_values("_order")
        .groupby(["entity", "chunk_text", "coding_system"], sort=False)
        .first()
        .reset_index()
        .pivot_table(index=["entity", "chunk_text"], columns="coding_system",
                     values=["code", "display", "_order"], aggfunc="first")
    )
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out = out.reset_index()
    out["_order"] = out[[c for c in out.columns if c.startswith("_order_")]].min(axis=1)
    out = out.sort_values("_order").drop(columns=[c for c in out.columns if c.startswith("_order")])
    return out.rename(columns={
        f"code_{ICD}": "icd_code", f"code_{SNOMED}": "snomed_code",
        f"display_{ICD}": "icd_display", f"display_{SNOMED}": "snomed_display",
    }).reset_index(drop=True)


def group_key(chunk_text: str, row_index: int) -> str:
    """In-memory CV group key: lowercase alphanumeric-only chunk text; empty chunks are singletons."""
    key = _GROUP_RE.sub("", str(chunk_text).lower())
    return key if key else f"__empty_{row_index}"


def load_inference_input(path: str) -> pd.DataFrame:
    """Read an inference CSV with columns entity (alias clinical_entity) and optional chunk_text."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "entity" not in df.columns and "clinical_entity" in df.columns:
        df = df.rename(columns={"clinical_entity": "entity"})
    if "entity" not in df.columns:
        raise ValueError("input must contain an 'entity' (or 'clinical_entity') column")
    if "chunk_text" not in df.columns:
        df["chunk_text"] = ""
    return df[["entity", "chunk_text"]].reset_index(drop=True)
