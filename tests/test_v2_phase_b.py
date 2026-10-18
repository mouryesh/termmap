"""Phase B: tests for modules that had none (hierarchy, coupling, confidence) and for each audit fix."""
import numpy as np
import pandas as pd

from termmap.confidence import coverage_curve, expected_calibration_error, threshold_for_precision
from termmap.coupling import CooccurrenceTable, couple, coupled_icd
from termmap.hierarchy import choose_argmax, choose_icd, fit_temperature, group_softmax, nll
from termmap.normalize import mine_abbreviations
from termmap.rerank.features import qualifiers_v2
from termmap.retrieval.engine import _stamp, _writable


# ---------------------------------------------------------------- hierarchy
def test_group_softmax_sums_to_one_per_term_and_temperature_sharpens():
    df = pd.DataFrame({"row": [0, 0, 1, 1, 1], "s": [2.0, 1.0, 0.0, 0.0, 3.0], "label": [1, 0, 0, 0, 1]})
    p = group_softmax(df, "s", 1.0)
    assert np.allclose(pd.Series(p).groupby(df["row"]).sum(), 1.0)
    assert group_softmax(df, "s", 0.5)[0] > p[0]          # lower T -> sharper
    assert fit_temperature(df, "s") <= 0.5 and nll(df, "s", 0.1) < nll(df, "s", 5.0)


def test_choose_icd_prefers_category_mass_and_respects_margin():
    codes = ["R26.89", "R26.9", "M25.50"]
    probs = np.array([0.30, 0.25, 0.45])                    # R26 mass 0.55 > M25 0.45
    code, info = choose_icd(codes, probs, margin=0.05)
    assert code == "R26.89" and info["changed_from_argmax"] == 1
    code2, info2 = choose_icd(codes, probs, margin=0.2)     # runner-up kept -> plain argmax
    assert code2 == "M25.50" and "M25" in info2["kept"]
    assert choose_argmax(["B", "A"], np.array([0.5, 0.5])) == "A"   # tie -> code order


# ---------------------------------------------------------------- coupling
def test_cooccurrence_is_smoothed_and_symmetric_in_counts():
    t = CooccurrenceTable([("I10", "38341003"), ("I10", "38341003"), ("I10", "59621000")])
    assert t("I10", "38341003") == 2 / (min(3, 2) + 1) and t("I10", "999") == 0.0


def test_couple_and_coupled_icd_keep_independent_choice_unless_moved():
    cooc = CooccurrenceTable([("I50.9", "84114007")] * 5)
    icd = [("R06.00", 0.5), ("I50.9", 0.4)]
    sct = [("84114007", 0.9)]
    assert couple(icd, sct, 0.0, cooc, None)[0] == "R06.00"           # lambda 0 = independent
    ic, sc, fit, co = couple(icd, sct, 8.0, cooc, None)
    assert (ic, sc) == ("I50.9", "84114007") and co > 0
    assert coupled_icd(icd, "R06.09", sct, 8.0, cooc, None) == ("I50.9", True)
    assert coupled_icd(icd, "R06.09", sct, 0.0, cooc, None) == ("R06.09", False)   # hierarchy pick kept
    assert coupled_icd([], "X", sct, 8.0, cooc, None) == ("X", False)


# ---------------------------------------------------------------- confidence
def test_threshold_curve_and_ece():
    conf = np.array([0.95, 0.9, 0.8, 0.6, 0.3])
    ok = np.array([1, 1, 1, 0, 0])
    t = threshold_for_precision(conf, ok, 0.98)
    assert t == 0.8                                          # top 3 are all correct
    assert threshold_for_precision(np.array([0.9]), np.array([0]), 0.98) == np.inf
    curve = coverage_curve(conf, ok, points=5)
    assert curve[0]["accuracy"] == 1.0 and curve[-1]["accuracy"] == 0.6
    assert expected_calibration_error(np.array([1.0, 1.0]), np.array([1, 1])) == 0.0


# ---------------------------------------------------------------- Phase B fixes
def test_abbreviation_min_support_needs_two_pairs_for_whole_display_expansions():
    terms = ["abd", "chf exacerbation", "htn"]
    disps = ["abdominal binder device", "congestive heart failure", "essential hypertension"]
    assert mine_abbreviations(terms, disps, 1).keys() == {"abd", "chf", "htn"}          # v1
    assert mine_abbreviations(terms, disps, 2) == {"chf": "congestive heart failure"}  # initials need 1
    assert mine_abbreviations(terms + ["htn"], disps + ["essential hypertension"], 2)["htn"] == "essential hypertension"


def test_negation_qualifier():
    assert ("negation", "negated") in qualifiers_v2("no fever")
    assert ("negation", "negated") in qualifiers_v2("negative for covid")
    assert ("negation", "negated") not in qualifiers_v2("fever")
    assert ("negation", "negated") not in qualifiers_v2("knot in neck")


def test_build_index_never_writes_through_symlinks(tmp_path):
    target = tmp_path / "v1.json"
    target.write_text("v1")
    link = tmp_path / "v2.json"
    link.symlink_to(target)
    _writable(link).write_text("v2")
    assert target.read_text() == "v1" and link.read_text() == "v2" and not link.is_symlink()


def test_stamp_records_dense_length_only_when_non_default(tmp_path):
    f = tmp_path / "d.xlsx"
    f.write_bytes(b"x")
    cfg = {"paths": {"icd_dict": str(f), "snomed_dict": str(f)}, "retrieval": {"dense_model": "m"}}
    assert "dense_max_len" not in _stamp(cfg)
    cfg["retrieval"]["dense_max_len"] = 64
    assert _stamp(cfg)["dense_max_len"] == 64


def test_embedded_undotted_icd_codes_only_with_flag():
    from termmap.normalize import embedded_codes
    assert embedded_codes("dm2 e119", {"E11.9"}, set()) == ([], [])
    assert embedded_codes("dm2 e119", {"E11.9"}, set(), undotted=True) == (["E11.9"], [])
    assert embedded_codes("E11.9 type 2", {"E11.9"}, set(), undotted=True) == (["E11.9"], [])
