"""M2 unit tests: cross-encoder inputs and training-pair construction (synthetic data only)."""
import pandas as pd

from termmap.rerank.cross_encoder import (CandidateText, build_training_pairs, context_window, query_text,
                                          top1_accuracy)


def test_context_window_marks_term():
    ctx = context_window("chf", "header_1: assessment:: admitted for CHF exacerbation", chars=20)
    assert "[ENT] chf [/ENT]" in ctx and "admitted for" in ctx
    assert context_window("chf", "") == ""
    assert query_text("chf", "") == "chf"


def _icd_fixture():
    table = pd.DataFrame({"code": ["I50", "I50.9", "I50.9", "I50.1"],
                          "text": ["heart failure", "heart failure, unspecified", "cardiac failure nos", "left ventricular failure"]})
    meta = pd.DataFrame({"code": ["I50", "I50.9", "I50.1"], "title": ["Heart failure", "Heart failure, unspecified", "Left ventricular failure"],
                         "category_title": ["Heart failure"] * 3}).set_index("code")
    return table, meta


def test_candidate_text_has_synonyms_and_category():
    table, meta = _icd_fixture()
    ct = CandidateText("ICD10CM", table, meta)
    s = ct("I50.9")
    assert s.startswith("heart failure, unspecified") and "cardiac failure nos" in s and s.endswith("heart failure")


def test_training_pairs_inject_gold_and_siblings():
    table, meta = _icd_fixture()
    ct = CandidateText("ICD10CM", table, meta)
    rows = pd.DataFrame({"term_norm": ["chf"], "chunk_text": [""], "code": ["I50.9"]}, index=[7])
    cands = pd.DataFrame({"row": [7], "code": ["I50"], "fused_rank": [1], "in_ce": [1]})  # gold not retrieved
    a, b, y = build_training_pairs(rows, cands, ct, "ICD10CM", 15, 0.0, 100, 0, {"I50": ["I50.1", "I50.9"]})
    assert y.sum() == 1 and y[0] == 1.0 and len(a) == 3  # gold + sibling I50.1 + retrieved I50


def test_top1_accuracy_tie_breaks_by_code():
    c = pd.DataFrame({"row": [0, 0, 1, 1], "code": ["A", "B", "A", "B"], "gold": [0, 1, 1, 0], "s": [1.0, 1.0, 2.0, 0.0]})
    assert top1_accuracy(c, "s", 2) == 0.5
