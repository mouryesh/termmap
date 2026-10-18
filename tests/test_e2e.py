"""End-to-end predict on a few synthetic rows: format, closed world, determinism. Needs built artifacts."""
from pathlib import Path

import pandas as pd
import pytest

from termmap.config import REPO_ROOT, load_config, set_seeds

ART = REPO_ROOT / "artifacts"
pytestmark = pytest.mark.skipif(not (ART / "model" / "SNOMEDCT_US_memory.joblib").exists(),
                                reason="final artifacts not built")

ROWS = pd.DataFrame({
    "entity": ["htn", "type 2 diabetes mellitus", "a-fib", "", "E11.9", "chf exacerbation"],
    "chunk_text": ["pmh: htn, hld", "", "ekg shows a-fib with rvr", "", "dx E11.9", "admitted for chf exacerbation"],
})


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    from termmap.pipeline import predict_file

    d = tmp_path_factory.mktemp("e2e")
    ROWS.to_csv(d / "in.csv", index=False)
    cfg = load_config()
    res = []
    for i in range(2):
        set_seeds(cfg["seed"])
        predict_file(cfg, str(d / "in.csv"), str(d / f"out{i}.csv"))
        res.append(Path(d / f"out{i}.csv").read_bytes())
    return d, res


def test_format_and_order(outputs):
    d, _ = outputs
    p = pd.read_csv(d / "out0.csv", dtype=str, keep_default_na=False)
    assert list(p.columns) == ["entity", "chunk_text", "icd10cm_code", "snomed_code"]
    assert p["entity"].tolist() == ROWS["entity"].tolist()
    assert (p["icd10cm_code"] != "").all() and (p["snomed_code"] != "").all()


def test_deterministic(outputs):
    _, (a, b) = outputs
    assert a == b
