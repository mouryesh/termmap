"""S3 independent format check: python scripts/check_predictions.py INPUT PREDICTIONS [PREDICTIONS_2]

Asserts same row count and order as the input, exact columns, no blanks, every code in the supplied
dictionaries (read directly from the xlsx files, not from the pipeline's index); prints SHA-256 hashes.
"""
import hashlib
import sys

import pandas as pd

inp, pred = sys.argv[1], sys.argv[2]
X = pd.read_csv(inp, dtype=str, keep_default_na=False)
P = pd.read_csv(pred, dtype=str, keep_default_na=False)
assert list(P.columns) == ["entity", "chunk_text", "icd10cm_code", "snomed_code"], list(P.columns)
assert len(P) == len(X), (len(P), len(X))
assert (P["entity"].values == X["entity"].values).all() and (P["chunk_text"].values == X["chunk_text"].values).all(), "order"
assert (P["icd10cm_code"] != "").all() and (P["snomed_code"] != "").all(), "blank code"
icd = set(pd.read_excel("data/icd10cm_codes.xlsx", dtype=str).iloc[:, 0].str.strip())
sct = set(pd.read_excel("data/SNOMEDCT_US_codes_and_descriptions.xlsx", dtype={"snomed_code": str})["snomed_code"].astype(str).str.strip())
bad_i, bad_s = (~P["icd10cm_code"].isin(icd)).sum(), (~P["snomed_code"].isin(sct)).sum()
assert bad_i == 0 and bad_s == 0, (bad_i, bad_s)
print(f"OK: {len(P)} rows, order and columns match, no blanks, all codes in the dictionaries")
for p in sys.argv[2:]:
    print(hashlib.sha256(open(p, "rb").read()).hexdigest(), p)
