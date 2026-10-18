import pandas as pd
import pytest

from termmap.closed_world import check

ICD = {"E11.9", "I10", "J96.00"}
SNOMED = {"44054006", "38341003"}


def _frames(icd, snomed):
    inp = pd.DataFrame({"entity": ["dm2", "htn"], "chunk_text": ["a", ""]})
    pred = inp.assign(icd10cm_code=icd, snomed_code=snomed)
    return pred, inp


def test_valid_output_passes():
    pred, inp = _frames(["E11.9", "I10"], ["44054006", "38341003"])
    assert check(pred, inp, ICD, SNOMED)["rows"] == 2


@pytest.mark.parametrize("icd,snomed", [
    (["E11.9", ""], ["44054006", "38341003"]),          # blank
    (["E11.9", "Z99.99"], ["44054006", "38341003"]),    # not in dictionary
    (["E11.9", "I10"], ["44054006", "123"]),            # SNOMED not in dictionary
])
def test_violations_raise(icd, snomed):
    pred, inp = _frames(icd, snomed)
    with pytest.raises(AssertionError):
        check(pred, inp, ICD, SNOMED)


def test_row_order_is_enforced():
    pred, inp = _frames(["E11.9", "I10"], ["44054006", "38341003"])
    with pytest.raises(AssertionError):
        check(pred.iloc[::-1].reset_index(drop=True), inp, ICD, SNOMED)
