"""Phase A (v2) fixes: tests per audit item, synthetic data only."""
import numpy as np
import pandas as pd

from termmap.build_dataset import candidate_frame, ce_scope
from termmap.dictionaries import SNOMED_SEMANTIC_TAGS, official_icd_titles
from termmap.rerank.cross_encoder import CandidateText, build_training_pairs
from termmap.rerank.features import qualifier_features, qualifiers_v2, QUAL_CLASSES_V2
from termmap.retrieval.engine import Candidates
from termmap.retrieval.fusion import rrf


def test_a3_negatives_follow_ltr_order_and_groups_are_contiguous():
    table = pd.DataFrame({"code": ["A", "B", "C", "D"], "text": ["a", "b", "c", "d"]})
    meta = pd.DataFrame({"code": list("ABCD"), "title": list("abcd"), "category_title": [""] * 4}).set_index("code")
    ct = CandidateText("SNOMEDCT_US", table, meta.assign(semantic_tag="finding"))
    rows = pd.DataFrame({"term_norm": ["x", "y"], "chunk_text": ["", ""], "code": ["A", "B"]}, index=[0, 1])
    cands = pd.DataFrame({"row": [0, 0, 0, 1, 1], "code": ["B", "C", "D", "C", "D"], "fused_rank": [1, 2, 3, 1, 2],
                          "in_ce": [1, 1, 1, 1, 1]})
    ltr = pd.DataFrame({"row": [0, 0, 0], "code": ["B", "C", "D"], "ltr": [0.1, 0.9, 0.5]})
    a, b, y, g = build_training_pairs(rows, cands, ct, "SNOMEDCT_US", 2, 0.0, 100, 0, None, "ltr", ltr, True)
    assert list(g) == [0, 0, 0, 1, 1, 1] and y.tolist() == [1, 0, 0, 1, 0, 0]
    assert b[1].startswith("c") and b[2].startswith("d")  # LTR order C (0.9) > D (0.5) > B (0.1)


def test_a4_scope_scores_icd_siblings_and_per_system_top():
    rc = {"ce_top": {"ICD10CM": 30, "SNOMEDCT_US": 40}, "ce_siblings": True}
    assert ce_scope(rc, "ICD10CM") == (30, True) and ce_scope(rc, "SNOMEDCT_US") == (40, False)
    assert ce_scope({"ce_top": 20}, "ICD10CM") == (20, False)  # v1 behaviour
    c = Candidates(fused=[("I50.9", 0.1)], ranks={"I50.9": {"char": 1}}, pool=["I50.9", "I50.1"], embedded=[])
    f = candidate_frame("ICD10CM", [0], ["I50.9"], [c], 30, True)
    assert f.set_index("code")["in_ce"].to_dict() == {"I50.9": 1, "I50.1": 1}
    f1 = candidate_frame("ICD10CM", [0], ["I50.9"], [c], 30, False)
    assert f1.set_index("code")["in_ce"].to_dict() == {"I50.9": 1, "I50.1": 0}


def test_a5_official_title_picks_the_cms_long_description():
    rows = pd.DataFrame({"code": ["A01.4", "A01.4", "A00.0", "A00.0"],
                         "description": ["Infection due to Salmonella paratyphi NOS", "Paratyphoid fever, unspecified",
                                         "Classical cholera", "Cholera due to Vibrio cholerae 01, biovar cholerae"]})
    t = official_icd_titles(rows)
    assert t["A01.4"] == "Paratyphoid fever, unspecified"
    assert t["A00.0"] == "Cholera due to Vibrio cholerae 01, biovar cholerae"


def test_a6_official_tag_list():
    assert {"disorder", "finding", "procedure", "morphologic abnormality", "regime/therapy"} <= SNOMED_SEMANTIC_TAGS
    assert "chronic obstructive pulmonary disease" not in SNOMED_SEMANTIC_TAGS and "AJCC" not in SNOMED_SEMANTIC_TAGS


def test_a7_shorthand_contrast_and_stage_suffixes():
    assert ("contrast", "without") in qualifiers_v2("ct cervical spine wo contrast")
    assert ("contrast", "with") in qualifiers_v2("mri brain w/ contrast")
    assert ("contrast", "with_and_without") in qualifiers_v2("ct head w/ and w/o contrast")
    assert qualifiers_v2("b/l le edema") >= {("laterality", "bilateral")} and ("laterality", "left") not in qualifiers_v2("b/l le edema")
    assert ("laterality", "left") in qualifiers_v2("l knee pain") and ("laterality", "left") not in qualifiers_v2("l5 radiculopathy")
    assert ("number", "stage_3a") in qualifiers_v2("ckd 3a") and ("number", "type_2") in qualifiers_v2("type ii diabetes")
    assert {("history", "status_post"), ("certainty", "rule_out")} <= qualifiers_v2("s/p cabg, r/o mi")
    f = qualifier_features(qualifiers_v2("ct wo contrast"), qualifiers_v2("computed tomography with contrast"), QUAL_CLASSES_V2)
    assert f["q_contrast_conflict"] == 1


def test_a9_weighted_rrf_zero_weight_drops_channel_and_default_is_equal():
    a, b = [("X", 0), ("Y", 0)], [("Y", 0), ("Z", 0)]
    assert [c for c, _ in rrf([a, b], 60, [0.0, 1.0])] == ["Y", "Z"]
    assert rrf([a, b], 60) == rrf([a, b], 60, [1.0, 1.0])
