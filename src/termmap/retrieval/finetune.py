"""SapBERT fine-tuning (v2, item C3), run per fold on Kaggle (job K2) and once on all rows.

Recipe follows SapBERT (Liu et al., NAACL 2021, arXiv 2010.11784): AdamW, lr 2e-5, 256 pairs per batch,
[CLS] representation, task fine-tuning for a few epochs. The loss is multiple-negatives ranking (InfoNCE,
scale 20) over in-batch positives plus one mined hard negative per anchor, as the v2 brief asks, instead of
SapBERT's multi-similarity loss. In-batch candidates with the anchor's own code are masked (not negatives).

Pairs: training-fold term -> gold display, plus every non-preferred dictionary description -> the
preferred title of the same code (ICD and SNOMED, all codes with more than one description, no sampling).
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from termmap.normalize import encoder_text, level1, strip_semantic_tag

log = logging.getLogger(__name__)


def synonym_pairs(table: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """(anchor, positive, code): each description that is not the code's preferred title -> that title."""
    title = meta["title"].map(lambda t: strip_semantic_tag(level1(t)) if isinstance(t, str) else "")
    t = table.assign(positive=table["code"].map(title))
    t = t[(t["positive"] != "") & t["positive"].notna() & (t["text"] != t["positive"])]
    return pd.DataFrame({"anchor": t["text"].values, "positive": t["positive"].values, "code": t["code"].values})


def training_pairs(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"anchor": rows["term_norm"].values, "positive": rows["display_norm"].values,
                         "code": rows["code"].values})


def mine_negatives(enc, anchors: list[str], codes: list[str], dict_emb, dict_texts: np.ndarray,
                   dict_codes: np.ndarray, top: int = 10, chunk: int = 1024) -> tuple[list[str], list[str]]:
    """Per anchor, the highest-scoring dictionary text of a different code (exact search on the GPU)."""
    import torch
    D = torch.from_numpy(dict_emb).to(enc.device).half()
    negs, ncodes = [], []
    for s in range(0, len(anchors), chunk):
        q = torch.from_numpy(enc.encode([encoder_text(a) for a in anchors[s:s + chunk]], 512)).to(enc.device).half()
        idx = (q @ D.T).topk(top, dim=1).indices.cpu().numpy()
        for row, code in zip(idx, codes[s:s + chunk]):
            j = next((j for j in row if dict_codes[j] != code), row[-1])
            negs.append(dict_texts[j]); ncodes.append(dict_codes[j])
    return negs, ncodes


def finetune(enc, pairs: pd.DataFrame, epochs: int, seed: int, val_fn=None, patience: int = 1,
             batch: int = 256, lr: float = 2e-5, scale: float = 20.0, save_best=None) -> dict:
    """Train enc.model in place on pairs (anchor, positive, negative, code, neg_code); after each epoch
    val_fn() -> score, best epoch kept via save_best(epoch) when given. Returns the history."""
    import torch
    from transformers import get_linear_schedule_with_warmup
    model, tok, dev = enc.model.float().train(), enc.tok, enc.device
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = epochs * ((len(pairs) + batch - 1) // batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    rng = np.random.default_rng(seed)
    code_ids = pd.factorize(pd.concat([pairs["code"], pairs["neg_code"]]))[0]
    pc, nc = code_ids[:len(pairs)], code_ids[len(pairs):]
    log.info("finetune: %d pairs, %d epochs, %d steps, batch %d", len(pairs), epochs, steps, batch)

    def emb(texts):
        x = tok([encoder_text(t) for t in texts], padding=True, truncation=True, max_length=enc.max_len,
                return_tensors="pt").to(dev)
        return torch.nn.functional.normalize(model(**x).last_hidden_state[:, 0], dim=-1)

    hist, best, bad = [], -1.0, 0
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        order = rng.permutation(len(pairs))
        for s in range(0, len(order), batch):
            i = order[s:s + batch]
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                a = emb(pairs["anchor"].values[i].tolist())
                c = emb(pairs["positive"].values[i].tolist() + pairs["negative"].values[i].tolist())
                logits = scale * (a @ c.T).float()
            cand = torch.from_numpy(np.concatenate([pc[i], nc[i]])).to(dev)
            mine = torch.from_numpy(pc[i]).to(dev)
            same = mine[:, None] == cand[None, :]
            same[torch.arange(len(i)), torch.arange(len(i))] = False   # the target itself stays
            loss = torch.nn.functional.cross_entropy(logits.masked_fill(same, -1e4),
                                                     torch.arange(len(i), device=dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(i)
        model.eval()
        rec = {"epoch": ep, "loss": round(tot / len(pairs), 4), "seconds": round(time.time() - t0)}
        if val_fn is not None:
            rec["val"] = val_fn()
            if rec["val"] > best:
                best, bad = rec["val"], 0
                if save_best:
                    save_best(ep)
            else:
                bad += 1
        hist.append(rec)
        log.info("finetune epoch %s", rec)
        if val_fn is not None and bad > patience - 1:
            break
    if dev == "cuda":
        model.half()
    return {"history": hist, "best_epoch": max(hist, key=lambda r: r.get("val", 0))["epoch"] if val_fn else epochs,
            "pairs": len(pairs)}
