# TermMap K1 (item C2): Qwen3-8B rewrite of every unique (term, context) training pair.
# Private notebook, GPU T4 x2, Internet ON. Input: private dataset mouryeshreddy/termmap-k1.
# Output: /kaggle/working/llm_rewrite.parquet (term, ctx_hash, llm), checkpointed every 20 batches;
# a rerun resumes from a checkpoint attached as input (a previous version's output).
import glob
import os
import subprocess
import sys
import zipfile

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.51.3", "accelerate==1.6.0"], check=True)
import pandas as pd  # noqa: E402


def find(name, required=True):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    if not hits and required:
        raise FileNotFoundError(name)
    return hits[0] if hits else None


WORK = "/kaggle/tmp/termmap"
zipfile.ZipFile(find("termmap_src.bin")).extractall(WORK)
sys.path.insert(0, f"{WORK}/src")
import logging  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
from termmap.config import set_seeds  # noqa: E402
from termmap.llm_rewrite import MODEL, Rewriter  # noqa: E402

set_seeds(42)
pairs = pd.read_parquet(find("rewrite_pairs.parquet"))
OUT = "/kaggle/working/llm_rewrite.parquet"
prev = find("llm_rewrite.parquet", required=False)
done = pd.read_parquet(prev) if prev else pd.DataFrame(columns=["term", "ctx_hash", "llm"])
print(f"pairs {len(pairs)}, already done {len(done)}, model {MODEL}")
todo = pairs.merge(done[["term", "ctx_hash"]], how="left", indicator=True)
todo = todo[todo["_merge"] == "left_only"].drop(columns="_merge").reset_index(drop=True)
rw = Rewriter()
BS, CHUNK = 32, 32 * 20
for s in range(0, len(todo), CHUNK):
    part = todo.iloc[s:s + CHUNK]
    llm = rw(list(zip(part["term"], part["ctx"])), batch_size=BS)
    done = pd.concat([done, part[["term", "ctx_hash"]].assign(llm=llm)], ignore_index=True)
    done.to_parquet(OUT, index=False)
    print(f"checkpoint {len(done)}/{len(pairs)}")
assert len(done) == len(pairs) and done["llm"].notna().all()
print(done.sample(20, random_state=0).to_string())
