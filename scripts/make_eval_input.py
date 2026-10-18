"""eval_input.csv = unique (entity, chunk_text) of the training CSV in first-seen order, every label column
(coding_system, code, display, subconcept) removed. Usage: python scripts/make_eval_input.py [train.csv] [out.csv]"""
import sys

import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "data/hackathon_train.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "eval_input.csv"
df = pd.read_csv(src, dtype=str, keep_default_na=False)
ev = df[["entity", "chunk_text"]].drop_duplicates(keep="first").reset_index(drop=True)
ev.to_csv(out, index=False)
print(f"{len(df)} training rows -> {len(ev)} eval rows -> {out}")
