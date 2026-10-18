"""Stage 3 Layer B features: one row per (term, candidate code).

Fold-dependent statistics (training frequency, exact memory match) are computed from the
training rows passed in, so out-of-fold features never see their own fold's labels.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from termmap.io import ICD, SNOMED
from termmap.normalize import tokens

# ---------------------------------------------------------------- qualifier lexicon
_QUAL_WORDS = {
    "laterality": {"left": "left", "lt": "left", "right": "right", "rt": "right",
                   "bilateral": "bilateral", "bilat": "bilateral", "both": "bilateral"},
    "acuity": {"acute": "acute", "chronic": "chronic", "subacute": "subacute"},
    "relation": {"with": "with", "without": "without", "w/o": "without", "due": "due_to"},
    "encounter": {"initial": "initial", "subsequent": "subsequent", "sequela": "sequela"},
    "modifier": {"hypoxia": "hypoxia", "hypoxic": "hypoxia", "hypercapnia": "hypercapnia",
                 "hypercapnic": "hypercapnia", "exacerbation": "exacerbation", "uncontrolled": "uncontrolled",
                 "recurrent": "recurrent", "severe": "severe", "mild": "mild", "moderate": "moderate",
                 "complicated": "complicated", "uncomplicated": "uncomplicated", "malignant": "malignant",
                 "benign": "benign", "primary": "primary", "secondary": "secondary"},
}
QUAL_CLASSES = tuple(_QUAL_WORDS) + ("number",)
_NUM = re.compile(r"\b(stage|type|grade|class|level|degree)\s*([0-9]+|i{1,3}v?|iv|v)\b")
_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5"}


def qualifiers(text: str) -> set[tuple[str, str]]:
    """Qualifier (class, value) pairs in a normalized text; 'acute on chronic' counts as both."""
    out: set[tuple[str, str]] = set()
    words = tokens(text)
    for cls, table in _QUAL_WORDS.items():
        for w in words:
            if w in table:
                out.add((cls, table[w]))
    if "w/o" in text:
        out.add(("relation", "without"))
    for kind, num in _NUM.findall(text):
        out.add(("number", f"{kind}_{_ROMAN.get(num, num)}"))
    return out


def qualifier_features(term_q: set, code_q: set) -> dict[str, float]:
    """code_extra = qualifiers only the code has (over-coding); term_extra = only the term has."""
    code_extra, term_extra = code_q - term_q, term_q - code_q
    f = {"code_extra": len(code_extra), "term_extra": len(term_extra), "qual_shared": len(term_q & code_q)}
    for cls in QUAL_CLASSES:
        t = {v for c, v in term_q if c == cls}
        c = {v for k, v in code_q if k == cls}
        f[f"q_{cls}_conflict"] = int(bool(t) and bool(c) and not (t & c))
        f[f"q_{cls}_code_only"] = int(bool(c) and not t)
        f[f"q_{cls}_term_only"] = int(bool(t) and not c)
    return f


# ---------------------------------------------------------------- feature builder
class FeatureBuilder:
    """Builds the LambdaRank feature table for one system."""

    def __init__(self, system: str, table: pd.DataFrame, meta: pd.DataFrame, ctext):
        self.system, self.meta, self.ctext = system, meta, ctext
        self.text_to_codes = table.groupby("text")["code"].apply(frozenset).to_dict()
        unspec = table[table["text"].str.contains(r"\bunspecified\b|\bnos\b", regex=True)]["code"]
        self.unspecified = set(unspec)
        if system == SNOMED:
            tags = sorted(meta["semantic_tag"].dropna().unique())
            self.tag_id = {t: i for i, t in enumerate(tags)}
        self._qcache: dict[str, set] = {}

    def _code_q(self, code: str) -> set:
        if code not in self._qcache:
            self._qcache[code] = qualifiers(self.ctext(code))
        return self._qcache[code]

    def build(self, cands: pd.DataFrame, rows: pd.DataFrame, train_rows: pd.DataFrame,
              ce_col: str | None) -> pd.DataFrame:
        """Feature rows for `cands` (candidates of `rows`), statistics taken from `train_rows`."""
        freq = Counter(train_rows["code"])
        mem: dict[str, set[str]] = {}
        for t, d, c in zip(train_rows["term_norm"], train_rows["display_norm"], train_rows["code"]):
            mem.setdefault(t, set()).add(c)
            mem.setdefault(d, set()).add(c)
        term = rows["term_norm"]
        term_q = {rid: qualifiers(t) for rid, t in term.items()}
        f = pd.DataFrame(index=cands.index)
        f["row"], f["code"] = cands["row"].values, cands["code"].values
        for col in ("fused_rank", "rank_char", "rank_dense", "rank_memory"):
            f[col] = np.log1p(cands[col].astype(float).values)
        f["rrf"] = cands["rrf"].values
        f["expanded"], f["embedded"] = cands["expanded"].values, cands["embedded"].values
        if ce_col is not None:
            ce = cands[ce_col].astype(float)
            f["ce"] = ce.fillna(-20.0).values
            f["ce_missing"] = ce.isna().astype(int).values
            grp = ce.groupby(cands["row"])
            f["ce_rank"] = np.log1p(grp.rank(ascending=False, method="first").fillna(999).values)
            f["ce_gap"] = (ce - grp.transform("max")).fillna(-20.0).values
        t_of = cands["row"].map(term)
        f["exact_dict"] = [int(c in self.text_to_codes.get(t, ())) for t, c in zip(t_of, cands["code"])]
        f["exact_memory"] = [int(c in mem.get(t, ())) for t, c in zip(t_of, cands["code"])]
        f["log_train_freq"] = [math.log1p(freq.get(c, 0)) for c in cands["code"]]
        f["seen_in_train"] = (f["log_train_freq"] > 0).astype(int)
        f["unspecified"] = [int(c in self.unspecified) for c in cands["code"]]
        m = self.meta.reindex(cands["code"])
        if self.system == ICD:
            f["is_leaf"] = m["is_leaf"].fillna(True).astype(int).values
            f["depth"] = m["depth"].fillna(0).astype(int).values
            f["chapter"] = [ord(c[0]) - ord("A") for c in cands["code"]]
        else:
            f["semantic_tag"] = [self.tag_id.get(t, -1) for t in m["semantic_tag"].fillna("unknown")]
        q = [qualifier_features(term_q[r], self._code_q(c)) for r, c in zip(cands["row"], cands["code"])]
        f = pd.concat([f, pd.DataFrame(q, index=cands.index)], axis=1)
        f["label"] = cands["gold"].values
        return f

    @property
    def categorical(self) -> list[str]:
        return ["chapter"] if self.system == ICD else ["semantic_tag"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("row", "code", "label", "fold")]
