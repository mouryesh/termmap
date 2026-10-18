"""Stage 0 normalization. Level 1 is applied identically to terms and dictionary descriptions."""
from __future__ import annotations

import re
import unicodedata

_BR = re.compile(r"<br\s*/?>", re.I)
_HEADER = re.compile(r"header_\d+\s*:")
_JUNK = re.compile(r"::|_{2,}")
_HCC = re.compile(r"\(\s*hcc\s*\)")
_SPACE_PUNCT = re.compile(r"\s+([.,;:!?)\]])")
_WS = re.compile(r"\s+")
_EDGE = re.compile(r"^[\s\W_]+|[\s\W_]+$")
_WORD = re.compile(r"[a-z0-9]+")


def level1(text: str) -> str:
    """NFKC, lowercase, strip markup/header noise and (hcc), tidy spacing and edge punctuation."""
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = _BR.sub(" ", t)
    t = _HEADER.sub(" ", t)
    t = _JUNK.sub(" ", t)
    t = _HCC.sub(" ", t)
    t = _SPACE_PUNCT.sub(r"\1", t)
    t = _WS.sub(" ", t).strip()
    # keep a trailing ')' or leading '(' only when balanced inside the string
    stripped = _EDGE.sub("", t)
    if t.endswith(")") and stripped.count("(") > stripped.count(")"):
        stripped += ")"
    return stripped


def tokens(text: str) -> list[str]:
    """Alphanumeric word tokens of an already-normalized string."""
    return _WORD.findall(text)


# ---------------------------------------------------------------- Level 2: query variants
_TAG_TAIL = re.compile(r"\s*\((?:[a-z /-]+)\)$")
_ICD_IN_TERM = re.compile(r"\b([a-z][0-9][0-9a-z](?:\.[0-9a-z]{1,4})?)\b")
_SCT_IN_TERM = re.compile(r"\b([0-9]{6,18})\b")
_ICD_UNDOTTED = re.compile(r"\b([a-z][0-9]{2}[0-9a-z]{1,4})\b")
_CLAUSE = re.compile(r"[,;:.]")


def strip_semantic_tag(desc_norm: str) -> str:
    """Remove a trailing SNOMED semantic tag, e.g. 'hypertension (disorder)' -> 'hypertension'."""
    return _TAG_TAIL.sub("", desc_norm)


class SpellCorrector:
    """Proposes the unique closest vocabulary word for out-of-vocabulary words (symspell)."""

    def __init__(self, word_counts: dict[str, int]):
        from symspellpy import SymSpell

        self.vocab = set(word_counts)
        self.sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        for w in sorted(word_counts):
            self.sym.create_dictionary_entry(w, word_counts[w])

    def correct_word(self, w: str) -> str | None:
        """Unique closest word at edit distance 1 (2 for 8+ letters); None on ties or no match."""
        from symspellpy import Verbosity

        if w in self.vocab or not w.isalpha():
            return None
        max_d = 2 if len(w) >= 8 else 1
        sugg = self.sym.lookup(w, Verbosity.CLOSEST, max_edit_distance=max_d)
        if not sugg:
            return None
        best = sugg[0].distance
        top = [s.term for s in sugg if s.distance == best]
        return top[0] if len(top) == 1 else None

    def variant(self, text: str) -> str | None:
        """Text with every correctable OOV word replaced, or None if nothing changed."""
        words = text.split(" ")
        changed = False
        out = []
        for w in words:
            core = w.strip(".,;:()[]")
            fix = self.correct_word(core) if core else None
            if fix:
                out.append(w.replace(core, fix))
                changed = True
            else:
                out.append(w)
        return " ".join(out) if changed else None


def mine_abbreviations(terms: list[str], displays: list[str], min_support: int = 1) -> dict[str, str]:
    """Mine short form -> expansion from (normalized term, normalized gold display) training pairs.

    A short form is a 2-6 letter token. Its expansion is either the whole display (when the term
    is that single token) or a run of display words whose initials spell the token. Only short
    forms with exactly one distinct expansion are kept. min_support (v2: 2) is the number of training
    pairs that must support a whole-display expansion; initials-matched expansions need one (v1: 1).
    """
    seen: dict[str, set[str]] = {}
    support: dict[tuple[str, str], int] = {}
    initials: set[tuple[str, str]] = set()
    for term, disp in zip(terms, displays):
        disp = strip_semantic_tag(disp)
        dwords = tokens(disp)
        tw = tokens(term)
        for t in tw:
            if not (2 <= len(t) <= 6 and t.isalpha()) or t in dwords:
                continue
            exp = None
            if len(tw) == 1:
                exp = disp
            else:
                for i in range(len(dwords) - len(t) + 1):
                    span = dwords[i:i + len(t)]
                    if "".join(x[0] for x in span) == t:
                        exp = " ".join(span)
                        initials.add((t, exp))
                        break
            if exp:
                seen.setdefault(t, set()).add(exp)
                support[(t, exp)] = support.get((t, exp), 0) + 1
    out = {}
    for k, v in sorted(seen.items()):
        if len(v) != 1:
            continue
        exp = next(iter(v))
        if (k, exp) in initials or support[(k, exp)] >= min_support:
            out[k] = exp
    return out


def abbreviation_variant(text: str, table: dict[str, str]) -> str | None:
    """Expand known short forms word by word; None if nothing was expanded."""
    words = text.split(" ")
    out, changed = [], False
    for w in words:
        core = w.strip(".,;:()[]")
        if core in table:
            out.append(w.replace(core, table[core]))
            changed = True
        else:
            out.append(w)
    return " ".join(out) if changed else None


def embedded_codes(raw_term: str, icd_codes: set[str], snomed_codes: set[str],
                   undotted: bool = False) -> tuple[list[str], list[str]]:
    """Codes written inside the term that exist in the dictionaries: (icd, snomed).

    undotted (v2): also read ICD codes written without the dot, e.g. 'E119' -> E11.9.
    """
    t = str(raw_term).lower()
    found = {m.upper() for m in _ICD_IN_TERM.findall(t)}
    if undotted:
        found |= {f"{m[:3]}.{m[3:]}".upper() for m in _ICD_UNDOTTED.findall(t)}
    icd = sorted(c for c in found if c in icd_codes)
    sct = sorted({m for m in _SCT_IN_TERM.findall(t) if m in snomed_codes})
    return icd, sct


def encoder_text(text: str, max_words: int = 32) -> str:
    """Encoders see only the first clause of very long terms; keyword search sees the full term."""
    if len(text.split()) <= max_words:
        return text
    first = _CLAUSE.split(text, maxsplit=1)[0].strip()
    return first if first else " ".join(text.split()[:max_words])


def query_variants(term_norm: str, spell: SpellCorrector | None, abbr: dict[str, str] | None) -> dict[str, str]:
    """All Stage 0 variants of a normalized term; 'orig' is always present."""
    v = {"orig": term_norm}
    if spell is not None:
        s = spell.variant(term_norm)
        if s and s != term_norm:
            v["spell"] = s
    if abbr:
        a = abbreviation_variant(term_norm, abbr)
        if a and a != term_norm:
            v["abbr"] = a
    return v
