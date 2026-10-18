"""End-to-end predict on a synthetic fixture: schema, validity, and byte-identical reruns.

Skipped when the built artifacts are absent (they are not committed).
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sample_input.csv"
needs_artifacts = pytest.mark.skipif(not (ROOT / "artifacts" / "model" / "abbreviations.json").exists(),
                                     reason="artifacts not built")


def _predict(out: Path) -> None:
    subprocess.run([sys.executable, "-m", "termmap.cli", "predict", "--input", str(FIXTURE), "--output", str(out)],
                   check=True, cwd=ROOT, env={**os.environ, "TERMMAP_USE_GPU": "0"})


@needs_artifacts
def test_predict_schema_and_determinism(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _predict(a)
    _predict(b)
    pa = pd.read_csv(a, dtype=str, keep_default_na=False)
    src = pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)
    assert list(pa.columns) == ["entity", "chunk_text", "icd10cm_code", "snomed_code"]
    assert len(pa) == len(src) and (pa["entity"] == src["entity"]).all()
    assert (pa["icd10cm_code"] != "").all() and (pa["snomed_code"] != "").all()
    assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()
    assert (tmp_path / "a_debug.csv").exists()


@needs_artifacts
def test_predict_runs_from_artifacts_only(tmp_path):
    """Dictionary .xlsx files absent -> predict still works from artifacts/ (index stamp trusted)."""
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    cfg["paths"]["icd_dict"] = str(tmp_path / "missing_icd.xlsx")
    cfg["paths"]["snomed_dict"] = str(tmp_path / "missing_snomed.xlsx")
    for k in ("train", "artifacts", "reports"):
        cfg["paths"][k] = str(ROOT / cfg["paths"][k])
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "p.csv"
    subprocess.run([sys.executable, "-m", "termmap.cli", "--config", str(cfg_path), "predict", "--input", str(FIXTURE),
                    "--output", str(out)], check=True, cwd=ROOT, env={**os.environ, "TERMMAP_USE_GPU": "0"})
    p = pd.read_csv(out, dtype=str, keep_default_na=False)
    dbg = pd.read_csv(tmp_path / "p_debug.csv", dtype=str, keep_default_na=False)
    empty = p["entity"] == ""
    assert empty.any() and (p["icd10cm_code"] != "").all()
    assert dbg.loc[empty, "icd_flags"].str.contains("fallback").all()  # empty term -> explicit fallback
