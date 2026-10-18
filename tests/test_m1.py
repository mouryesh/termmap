"""M1 unit tests: Level 2 variants, fusion, collapse, sibling expansion (synthetic data only)."""
import numpy as np

from termmap.normalize import (SpellCorrector, abbreviation_variant, embedded_codes, encoder_text,
                               mine_abbreviations, query_variants, strip_semantic_tag)
from termmap.retrieval.chargram import CharGramIndex
from termmap.retrieval.fusion import best_rank_merge, collapse, rrf


def test_strip_tag():
    assert strip_semantic_tag("hypertension (disorder)") == "hypertension"
    assert strip_semantic_tag("pain in left knee") == "pain in left knee"


def test_abbreviation_mining_unique_only():
    terms = ["chf exacerbation", "htn", "ms", "ms"]
    disps = ["congestive heart failure", "essential hypertension", "multiple sclerosis", "mitral stenosis"]
    t = mine_abbreviations(terms, disps)
    assert t["chf"] == "congestive heart failure" and t["htn"] == "essential hypertension"
    assert "ms" not in t  # two expansions -> not expanded
    assert abbreviation_variant("chf, acute", t) == "congestive heart failure, acute"


def test_spell_unique_closest_and_ties():
    sp = SpellCorrector({"pneumonia": 10, "fracture": 5, "cat": 1, "bat": 1})
    assert sp.variant("pnuemonia") == "pneumonia"   # transposition, 9 letters -> distance <= 2
    assert sp.variant("hat") is None               # tie between cat and bat
    assert sp.variant("fracture") is None          # in vocabulary


def test_variants_keep_original_and_negation():
    v = query_variants("no chf", None, {"chf": "congestive heart failure"})
    assert v["orig"] == "no chf" and v["abbr"] == "no congestive heart failure"


def test_embedded_codes_and_long_terms():
    icd, sct = embedded_codes("Hypertension I10 (38341003)", {"I10"}, {"38341003"})
    assert icd == ["I10"] and sct == ["38341003"]
    long = "pain in knee, " + " ".join(["word"] * 40)
    assert encoder_text(long, 32) == "pain in knee"


def test_rrf_and_ties_by_code():
    a = [("B", 0.9), ("A", 0.8)]
    b = [("A", 0.7), ("B", 0.6)]
    fused = rrf([a, b], k=60)
    assert [c for c, _ in fused] == ["A", "B"]  # equal scores -> code order
    assert [c for c, _ in best_rank_merge([a, b])] == ["A", "B"]


def test_collapse_best_per_code():
    codes = np.array(["X", "X", "Y"], dtype=object)
    r = collapse(np.array([0, 1, 2]), np.array([0.2, 0.9, 0.5]), codes, 10)
    assert r == [("X", 0.9), ("Y", 0.5)]


def test_chargram_finds_typo():
    idx = CharGramIndex(["atrial fibrillation", "atrial flutter", "pneumonia"], ["1", "2", "3"])
    assert idx.search(["atrial fibrilation"])[0][0][0] == "1"
