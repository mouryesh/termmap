# TermMap K2 (item C3): SapBERT fine-tunes, 3 fold models (out-of-fold) + 1 final model on all rows.
# Private notebook, GPU T4, Internet ON. Input: private dataset mouryeshreddy/termmap-k2.
# Output: /kaggle/working/k2/{fold0,fold1,fold2,final}/ (model + tokenizer) and k2_history.json.
# Resume: attach a previous version's output; finished folds are copied, not retrained.
import glob
import json
import os
import shutil
import subprocess
import sys
import zipfile

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.46.3", "faiss-cpu==1.9.0"], check=True)
import numpy as np  # noqa: E402
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
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

from termmap.config import load_config, set_seeds  # noqa: E402
from termmap.normalize import encoder_text  # noqa: E402
from termmap.retrieval.dense import DenseIndex, Encoder  # noqa: E402
from termmap.retrieval.finetune import finetune, mine_negatives, synonym_pairs, training_pairs  # noqa: E402

cfg = load_config(f"{WORK}/configs/v2.yaml")
SEED, BASE, MAXLEN, EPOCHS = cfg["seed"], cfg["retrieval"]["dense_model"], cfg["retrieval"]["dense_max_len"], 5
set_seeds(SEED)
OUT = "/kaggle/working/k2"
os.makedirs(OUT, exist_ok=True)
rows = pd.read_parquet(find("rows.parquet"))
SYS = ("ICD10CM", "SNOMEDCT_US")
tables = {s: pd.read_parquet(find(f"{s}_table.parquet")) for s in SYS}
metas = {s: pd.read_parquet(find(f"{s}_meta.parquet")) for s in SYS}
syn = {s: synonym_pairs(tables[s], metas[s]) for s in SYS}
print("rows", len(rows), "synonym pairs", {s: len(v) for s, v in syn.items()},
      "dictionary texts", {s: len(t) for s, t in tables.items()})

# hard negatives, mined once with the base model (each anchor's own gold code excluded)
base = Encoder(BASE, max_len=MAXLEN)
neg = {}
for s in SYS:
    t = tables[s]
    emb = base.encode([encoder_text(x) for x in t["text"]], 512).astype(np.float16)
    trn = training_pairs(rows[rows.coding_system == s]).set_index(rows.index[rows.coding_system == s])
    allp = pd.concat([syn[s], trn])
    n, nc = mine_negatives(base, allp["anchor"].tolist(), allp["code"].tolist(), emb, t["text"].values, t["code"].values)
    allp = allp.assign(negative=n, neg_code=nc)
    neg[s] = (allp.iloc[:len(syn[s])], allp.iloc[len(syn[s]):])
    print(s, "negatives mined:", len(allp))
del base


def pairs_for(train_idx):
    parts = []
    for s in SYS:
        sp, tp = neg[s]
        parts += [sp, tp[tp.index.isin(train_idx)]]
    return pd.concat(parts, ignore_index=True)


def recall10(enc, val_rows):
    hits = 0
    for s in SYS:
        v = val_rows[val_rows.coding_system == s]
        if not len(v):
            continue
        idx = DenseIndex(enc.encode([encoder_text(x) for x in tables[s]["text"]], 512), tables[s]["code"].tolist())
        ranks = idx.search(enc.encode([encoder_text(x) for x in v["term_norm"]]), top_codes=10)
        hits += sum(g in [c for c, _ in r] for g, r in zip(v["code"], ranks))
    return round(100 * hits / len(val_rows), 2)


def save(enc, d):
    enc.tok.save_pretrained(d)
    enc.model.save_pretrained(d)


prev = find("k2_history.json", required=False)
history = json.load(open(prev)) if prev else {}
for f in sorted(rows["fold"].unique()):
    d = f"{OUT}/fold{f}"
    if f"fold{f}" in history and find(f"k2/fold{f}/config.json", required=False):
        shutil.copytree(os.path.dirname(find(f"k2/fold{f}/config.json")), d, dirs_exist_ok=True)
        continue
    tr = rows[rows.fold != f]
    gi, vi = next(GroupShuffleSplit(1, test_size=0.1, random_state=SEED).split(tr, groups=tr["chunk_text"]))
    fit_rows, val_rows = tr.iloc[gi], tr.iloc[vi]
    enc = Encoder(BASE, max_len=MAXLEN)
    print(f"fold {f}: base R@10 on inner val ({len(val_rows)} rows) = {recall10(enc, val_rows)}")
    res = finetune(enc, pairs_for(fit_rows.index), EPOCHS, SEED + int(f), val_fn=lambda: recall10(enc, val_rows),
                   save_best=lambda ep: save(enc, d))
    res.update(fit_rows=len(fit_rows), val_rows=len(val_rows))
    history[f"fold{f}"] = res
    json.dump(history, open(f"{OUT}/../k2_history.json", "w"), indent=1)
    del enc

final_epochs = int(np.median([history[f"fold{f}"]["best_epoch"] for f in sorted(rows["fold"].unique())]))
enc = Encoder(BASE, max_len=MAXLEN)
history["final"] = finetune(enc, pairs_for(rows.index), final_epochs, SEED)
save(enc, f"{OUT}/final")
json.dump(history, open(f"{OUT}/../k2_history.json", "w"), indent=1)
print(json.dumps(history, indent=1))
