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
from termmap.normalize import level1, strip_semantic_tag, tokens

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


# ---------------------------------------------------------------- v2 lexicon (audit A7)
_BODY = ("knee|hip|shoulder|arm|leg|foot|feet|hand|ankle|wrist|elbow|eye|ear|breast|lung|kidney|"
         "thigh|calf|toe|finger|heel|forearm|groin|flank|chest|ovary|testis|testicle|tibia|femur|fibula|"
         "humerus|radius|ulna|clavicle|rib|lower extremity|upper extremity|le|ue")
_V2_RULES = [
    (re.compile(r"\b(with and without|with/without|w/wo|w and wo|w/ and w/o)\s+contrast\b"), ("contrast", "with_and_without")),
    (re.compile(r"\b(without|w/o|wo)\s+(iv\s+)?contrast\b"), ("contrast", "without")),
    (re.compile(r"\b(with|w/|w)\s+(iv\s+)?contrast\b"), ("contrast", "with")),
    (re.compile(r"(?<![a-z])w/o(?![a-z])|\bwo\b|\bwithout\b"), ("relation", "without")),
    (re.compile(r"(?<![a-z])w/(?![a-z/])|\bwith\b"), ("relation", "with")),
    (re.compile(r"\bdue to\b|\bsecondary to\b|(?<![a-z])2/2(?![0-9])"), ("relation", "due_to")),
    (re.compile(r"(?<![a-z])s/p(?![a-z])|\bstatus post\b"), ("history", "status_post")),
    (re.compile(r"(?<![a-z])h/o(?![a-z])|\bhistory of\b|\bhx of\b"), ("history", "history_of")),
    (re.compile(r"(?<![a-z])r/o(?![a-z])|\brule out\b"), ("certainty", "rule_out")),
    (re.compile(r"\b(bilateral|bilat|both)\b|(?<![a-z])b/l(?![a-z])"), ("laterality", "bilateral")),
    (re.compile(r"\b(left|lt)\b|(?<![a-z/])l(?=\.?\s+(" + _BODY + r")\b)"), ("laterality", "left")),
    (re.compile(r"\b(right|rt)\b|(?<![a-z/])r(?=\.?\s+(" + _BODY + r")\b)"), ("laterality", "right")),
    (re.compile(r"^(?:no|neg|negative|denies|without|ø)\b|\b(?:absence of|absent|negative for|free of|nad)\b"),
     ("negation", "negated")),
    (re.compile(r"\bacute on chronic\b"), ("acuity", "acute_on_chronic")),
    (re.compile(r"\bacute\b"), ("acuity", "acute")),
    (re.compile(r"\bchronic\b"), ("acuity", "chronic")),
    (re.compile(r"\bsubacute\b"), ("acuity", "subacute")),
    (re.compile(r"\binitial encounter\b|\binitial\b"), ("encounter", "initial")),
    (re.compile(r"\bsubsequent\b"), ("encounter", "subsequent")),
    (re.compile(r"\bsequela\b"), ("encounter", "sequela")),
]
_NUM_V2 = re.compile(r"\b(stage|type|grade|class|level|degree|ckd)\s*([0-9]+[a-c]?|iv|v|i{1,3})\b")


def qualifiers_v2(text: str) -> set[tuple[str, str]]:
    """v2 qualifier set: v1 modifiers plus clinical shorthand, contrast, history/rule-out, and
    stage/type values with letter suffixes or roman numerals ('ckd 3a' reads as stage 3a)."""
    out: set[tuple[str, str]] = set()
    for rx, q in _V2_RULES:
        if rx.search(text):
            out.add(q)
    for w in tokens(text):
        if w in _QUAL_WORDS["modifier"]:
            out.add(("modifier", _QUAL_WORDS["modifier"][w]))
    if ("contrast", "with_and_without") in out:
        out -= {("contrast", "with"), ("contrast", "without")}
    for kind, num in _NUM_V2.findall(text):
        kind = "stage" if kind == "ckd" else kind
        out.add(("number", f"{kind}_{_ROMAN.get(num, num)}"))
    return out


QUAL_CLASSES_V2 = QUAL_CLASSES + ("contrast", "history", "certainty", "negation")


def qualifier_features(term_q: set, code_q: set, classes: tuple[str, ...] = QUAL_CLASSES) -> dict[str, float]:
    """code_extra = qualifiers only the code has (over-coding); term_extra = only the term has."""
    code_extra, term_extra = code_q - term_q, term_q - code_q
    f = {"code_extra": len(code_extra), "term_extra": len(term_extra), "qual_shared": len(term_q & code_q)}
    for cls in classes:
        t = {v for c, v in term_q if c == cls}
        c = {v for k, v in code_q if k == cls}
        f[f"q_{cls}_conflict"] = int(bool(t) and bool(c) and not (t & c))
        f[f"q_{cls}_code_only"] = int(bool(c) and not t)
        f[f"q_{cls}_term_only"] = int(bool(t) and not c)
    return f


# ---------------------------------------------------------------- feature builder
class FeatureBuilder:
    """Builds the LambdaRank feature table for one system."""

    def __init__(self, system: str, table: pd.DataFrame, meta: pd.DataFrame, ctext, qualifier_version: int = 1,
                 groups: tuple[str, ...] = ()):
        self.system, self.meta, self.ctext, self.groups = system, meta, ctext, tuple(groups)
        self.qfun = qualifiers_v2 if qualifier_version == 2 else qualifiers
        self.qclasses = QUAL_CLASSES_V2 if qualifier_version == 2 else QUAL_CLASSES
        self.knowledge = {}  # item C1: source -> {text: codes}, exact_index / exact_inclusion features
        if "source" in table:
            for src, g in table[table["source"] != "dict"].groupby("source"):
                self.knowledge[src] = g.groupby("text")["code"].apply(frozenset).to_dict()
            table = table[table["source"] == "dict"]
        self.text_to_codes = table.groupby("text")["code"].apply(frozenset).to_dict()
        self.desc = table[["code", "text"]]
        unspec = table[table["text"].str.contains(r"\bunspecified\b|\bnos\b", regex=True)]["code"]
        self.unspecified = set(unspec)
        if system == SNOMED:
            tags = sorted(meta["semantic_tag"].dropna().unique())
            self.tag_id = {t: i for i, t in enumerate(tags)}
        self._qcache: dict[str, set] = {}

    def _code_q(self, code: str) -> set:
        if code not in self._qcache:
            self._qcache[code] = self.qfun(self.ctext(code))
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
        term_q = {rid: self.qfun(t) for rid, t in term.items()}
        f = pd.DataFrame(index=cands.index)
        f["row"], f["code"] = cands["row"].values, cands["code"].values
        for col in ["fused_rank", "rank_char", "rank_dense", "rank_memory"] + \
                [c for c in ("rank_bm25", "rank_dense2") if c in cands]:
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
        for src, t2c in sorted(self.knowledge.items()):
            f[f"exact_{src}"] = [int(c in t2c.get(t, ())) for t, c in zip(t_of, cands["code"])]
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
        q = [qualifier_features(term_q[r], self._code_q(c), self.qclasses) for r, c in zip(cands["row"], cands["code"])]
        f = pd.concat([f, pd.DataFrame(q, index=cands.index)], axis=1)
        if self.groups:
            f = pd.concat([f, self._groups(cands, rows, f)], axis=1)
        f["label"] = cands["gold"].values
        return f

    # ------------------------------------------------------------ Track F feature groups (vectorised)
    def _title(self, codes: pd.Series) -> pd.Series:
        t = self.meta["title"].reindex(codes.unique()).fillna("").map(level1)
        if self.system == SNOMED:
            t = t.map(strip_semantic_tag)
        return codes.map(t)

    def _groups(self, cands: pd.DataFrame, rows: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
        """string (F2), term (F3), structure (F5) and pool (F4) features; all computable at predict time."""
        from rapidfuzz import fuzz
        from rapidfuzz.process import cpdist
        g = pd.DataFrame(index=cands.index)
        term = cands["row"].map(rows["term_norm"]).fillna("").values
        title = self._title(cands["code"]).values
        if "string" in self.groups:
            for name, sc in (("ratio", fuzz.ratio), ("tset", fuzz.token_set_ratio), ("partial", fuzz.partial_ratio)):
                g[f"s_title_{name}"] = cpdist(term, title, scorer=sc, workers=-1)
            pairs = cands[["code"]].assign(i=np.arange(len(cands)), term=term).merge(self.desc, on="code")
            best = cpdist(pairs["term"].values, pairs["text"].values, scorer=fuzz.token_set_ratio, workers=-1)
            g["s_desc_tset_max"] = pd.Series(best).groupby(pairs["i"].values).max().reindex(range(len(cands))).fillna(0).values
            tt = [set(tokens(x)) for x in term]
            ct = [set(tokens(x)) for x in title]
            g["s_tok_recall"] = [len(a & b) / len(a) if a else 0.0 for a, b in zip(tt, ct)]
            g["s_tok_precision"] = [len(a & b) / len(b) if b else 0.0 for a, b in zip(tt, ct)]
            g["s_term_in_title"] = [int(bool(a) and a in b) for a, b in zip(term, title)]
            g["s_title_in_term"] = [int(bool(b) and b in a) for a, b in zip(term, title)]
            g["s_len_diff"] = [len(b) - len(a) for a, b in zip(tt, ct)]
        if "term" in self.groups:
            tr = rows.reindex(cands["row"].unique())
            tok = tr["term_norm"].fillna("").map(tokens)
            chunk = tr["chunk_text"].fillna("").map(level1)
            tf = pd.DataFrame({"t_tokens": tok.map(len), "t_chars": tr["term_norm"].fillna("").str.len(),
                               "t_digit": tr["term_norm"].fillna("").str.contains(r"\d").astype(int),
                               "t_short": (tr["term_norm"].fillna("").str.len() <= 4).astype(int),
                               "t_vowelless": tr["term_norm"].fillna("").str.fullmatch(r"[^aeiou\s]+").fillna(False).astype(int),
                               "t_chunk_chars": chunk.str.len(),
                               "t_table": tr["chunk_text"].fillna("").str.contains("header_|<br>").astype(int),
                               "t_in_chunk": [int(bool(t) and t in c) for t, c in zip(tr["term_norm"].fillna(""), chunk)],
                               "t_n_qual": tr["term_norm"].fillna("").map(lambda x: len({c for c, _ in self.qfun(x)}))},
                              index=tr.index)
            g = g.join(tf.reindex(cands["row"]).set_index(cands.index))
        if "structure" in self.groups:
            n_desc = self.desc.groupby("code").size()
            g["c_n_desc"] = cands["code"].map(n_desc).fillna(0).values
            g["c_title_len"] = [len(x) for x in title]
            for flag, rx in (("other", r"\bother\b"), ("nec", r"\bnec\b|not elsewhere classified"), ("nos", r"\bnos\b")):
                g[f"c_{flag}"] = pd.Series(title).str.contains(rx).astype(int).values
            if self.system == ICD:
                g["c_7th"] = (cands["code"].str.replace(".", "", regex=False).str.len() == 7).astype(int).values
            nq = cands["row"].map(rows["term_norm"].fillna("").map(lambda x: len({c for c, _ in self.qfun(x)})))
            g["c_unspec_x_nqual"] = f["unspecified"].values * nq.values
            g["c_extra_x_digit"] = f["code_extra"].values * cands["row"].map(rows["term_norm"].fillna("").str.contains(r"\d")).astype(int).values
        if "pool" in self.groups and "ce" in f:
            ce = pd.Series(np.where(f["ce_missing"].values == 1, -np.inf, f["ce"].values), index=cands.index)
            e = np.exp(ce - ce.groupby(cands["row"]).transform("max"))
            share = e / e.groupby(cands["row"]).transform("sum")
            g["p_ce_share"] = share.values
            key = cands["code"].str[:3] if self.system == ICD else \
                cands["code"].map(self.meta["semantic_tag"]).fillna("unknown")
            g["p_group_mass"] = share.groupby([cands["row"], key.values]).transform("sum").values
            g["p_group_size"] = cands.groupby([cands["row"], key.values])["code"].transform("size").values
        return g

    @property
    def categorical(self) -> list[str]:
        return ["chapter"] if self.system == ICD else ["semantic_tag"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("row", "code", "label", "fold")]
