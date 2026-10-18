"""M0 unit tests: normalization, group key, input aliasing, pairing (no proprietary data needed)."""
import pandas as pd

from termmap.io import group_key, load_inference_input, pair_instances
from termmap.normalize import level1


def test_level1_cleans_markup_and_keeps_qualifiers():
    assert level1("Atrial Fibrillation (HCC)") == "atrial fibrillation"
    assert level1("header_1: banding .<br>") == "banding"
    assert level1("No acute distress") == "no acute distress"
    assert level1("Left knee pain, chronic") == "left knee pain, chronic"
    assert level1("stage 3 ckd") == "stage 3 ckd"


def test_group_key_collapses_spacing_and_isolates_empty():
    assert group_key("banding .", 0) == group_key("Banding.", 1)
    assert group_key("", 0) != group_key("", 1)


def test_inference_alias_and_missing_chunk(tmp_path):
    f = tmp_path / "in.csv"
    pd.DataFrame({"clinical_entity": ["htn"]}).to_csv(f, index=False)
    df = load_inference_input(str(f))
    assert list(df.columns) == ["entity", "chunk_text"] and df.chunk_text[0] == ""


def test_pairing_long_to_wide():
    long = pd.DataFrame({
        "entity": ["a", "a", "b"], "chunk_text": ["x", "x", "y"],
        "coding_system": ["ICD10CM", "SNOMEDCT_US", "SNOMEDCT_US"],
        "code": ["I10", "38341003", "1"], "display": ["d1", "d2", "d3"], "subconcept": ["", "", ""],
    })
    w = pair_instances(long)
    assert len(w) == 2 and w.loc[0, "icd_code"] == "I10" and pd.isna(w.loc[1, "icd_code"])
