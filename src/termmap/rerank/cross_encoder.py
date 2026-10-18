"""Stage 3 Layer A: per-system cross-encoder (MedCPT) fine-tuned on retrieval hard negatives.

Input pair: (term [SEP] context window with [ENT] marks, candidate description | synonyms | category).
Context is dropped for 50% of training rows so one model serves both with- and without-context inputs.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from termmap.io import ICD, SNOMED
from termmap.normalize import level1, strip_semantic_tag

log = logging.getLogger(__name__)
ENT, ENT_END = "[ENT]", "[/ENT]"


# ---------------------------------------------------------------- input text
def context_window(term_norm: str, chunk: str, chars: int = 100) -> str:
    """About `chars` characters either side of the term inside the normalized chunk, term marked."""
    c = level1(chunk)
    if not c:
        return ""
    i = c.find(term_norm) if term_norm else -1
    if i < 0:
        return c[: 2 * chars]
    left = c[max(0, i - chars):i].strip()
    right = c[i + len(term_norm): i + len(term_norm) + chars].strip()
    return f"{left} {ENT} {term_norm} {ENT_END} {right}".strip()


def query_text(term_norm: str, ctx: str) -> str:
    return f"{term_norm} [SEP] {ctx}" if ctx else term_norm


class CandidateText:
    """Cross-encoder text for a code: preferred term | up to 3 synonyms | ICD category title / SNOMED tag.

    With knowledge synonyms in the table (column `source`, item C1) and a query term, the knowledge
    synonym closest to the term (rapidfuzz token_set_ratio, ties by text) goes right after the
    preferred term.
    """

    def __init__(self, system: str, table: pd.DataFrame, meta: pd.DataFrame, n_syn: int = 3):
        self.system, self.meta, self.n_syn = system, meta, n_syn
        know = table["source"] != "dict" if "source" in table else pd.Series(False, index=table.index)
        self.texts = table[~know].groupby("code", sort=False)["text"].apply(list).to_dict()
        self.know = table[know].groupby("code", sort=False)["text"].apply(sorted).to_dict()
        self._cache: dict[str, str] = {}

    def __call__(self, code: str, query: str | None = None) -> str:
        base = self._base(code)
        if query and code in self.know:
            from rapidfuzz import fuzz
            best = max(self.know[code], key=lambda t: fuzz.token_set_ratio(query, t))  # ties: first in sorted order
            head, sep, tail = base.partition(" | ")
            return f"{head} | {best}{sep}{tail}" if sep else f"{head} | {best}"
        return base

    def _base(self, code: str) -> str:
        if code in self._cache:
            return self._cache[code]
        m = self.meta.loc[code] if code in self.meta.index else None
        pref = level1(m["title"]) if m is not None and isinstance(m["title"], str) else ""
        if self.system == SNOMED:
            pref = strip_semantic_tag(pref)
        syns = sorted({t for t in self.texts.get(code, []) if t != pref}, key=lambda t: (len(t), t))[: self.n_syn]
        parts = [pref or (syns.pop(0) if syns else code)] + syns
        if self.system == ICD and m is not None and m["category_title"] and code[:3] != code:
            parts.append(level1(m["category_title"]))
        if self.system == SNOMED and m is not None:
            parts.append(f"({m['semantic_tag']})")
        s = " | ".join(parts)
        self._cache[code] = s
        return s


# ---------------------------------------------------------------- model
class CrossEncoder:
    """Thin wrapper over a BERT sequence-classification head with one logit."""

    def __init__(self, name_or_path: str, device: str | None = None, max_len: int = 128):
        local = Path(name_or_path).exists()
        self.tok = AutoTokenizer.from_pretrained(name_or_path, local_files_only=local)
        if ENT not in self.tok.get_vocab():
            self.tok.add_special_tokens({"additional_special_tokens": [ENT, ENT_END]})
        self.model = AutoModelForSequenceClassification.from_pretrained(name_or_path, num_labels=1,
                                                                        local_files_only=local)
        self.model.resize_token_embeddings(len(self.tok))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_len = max_len

    def save(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.tok.save_pretrained(path)
        self.model.save_pretrained(path)

    def _batch(self, a: list[str], b: list[str]):
        return self.tok(a, b, padding=True, truncation="longest_first", max_length=self.max_len,
                        return_tensors="pt").to(self.device)

    @torch.inference_mode()
    def score(self, a: list[str], b: list[str], batch_size: int = 128) -> np.ndarray:
        """Logits for (a[i], b[i]) pairs; length-sorted batches, fp16 autocast on GPU."""
        self.model.eval()
        order = np.argsort([len(x) + len(y) for x, y in zip(a, b)], kind="stable")
        out = np.zeros(len(a), dtype=np.float32)
        for s in range(0, len(a), batch_size):
            idx = order[s:s + batch_size]
            enc = self._batch([a[i] for i in idx], [b[i] for i in idx])
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                logits = self.model(**enc).logits.squeeze(-1)
            out[idx] = logits.float().cpu().numpy()
        return out

    def train_pairs(self, a: list[str], b: list[str], y: np.ndarray, cfg: dict, seed: int,
                    val_fn=None, epochs: int | None = None, groups: np.ndarray | None = None) -> dict:
        """Fine-tune; after each epoch val_fn() -> accuracy, best epoch's weights kept.

        loss 'bce' (v1): pointwise BCE over shuffled pairs. loss 'listwise' (v2, audit A2): softmax
        cross-entropy (InfoNCE) over each term's group of 1 positive + negatives, whole groups per batch
        (Morand et al. 2026, arXiv 2603.03010: comparison-based objectives beat pointwise BCE).
        Early stopping (v2, audit A1): stop after `patience` epochs without validation gain.
        """
        rc = cfg["rerank"]
        epochs = epochs or rc["epochs"]
        patience = rc.get("patience") if val_fn is not None else None
        listwise = rc.get("loss", "bce") == "listwise"
        if listwise and groups is None:
            raise ValueError("listwise loss needs group ids")
        bs = rc["batch_size"]
        if listwise:
            order = np.unique(groups, return_index=True)
            starts = np.sort(order[1])
            spans = [(st, en) for st, en in zip(starts, list(starts[1:]) + [len(a)])]
            gpb = max(1, bs // max(en - st for st, en in spans))
            n_steps = (len(spans) + gpb - 1) // gpb
        else:
            n_steps = (len(a) + bs - 1) // bs
        opt = torch.optim.AdamW(self.model.parameters(), lr=rc["lr"], weight_decay=0.01,
                                fused=self.device == "cuda")
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * epochs * n_steps), epochs * n_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=self.device == "cuda")
        bce = torch.nn.BCEWithLogitsLoss()
        g = torch.Generator().manual_seed(seed)
        yt = torch.tensor(y, dtype=torch.float32)
        hist, best, best_state, since_best = [], -1.0, None, 0
        log.info("training: %d pairs, %s, %d steps/epoch, max %d epochs, patience %s",
                 len(a), f"{len(spans)} groups x {gpb}/batch listwise" if listwise else "pointwise BCE",
                 n_steps, epochs, patience)
        for ep in range(epochs):
            self.model.train()
            t0, tot = time.time(), 0.0
            if listwise:
                gperm = torch.randperm(len(spans), generator=g).tolist()
                batches = [[spans[j] for j in gperm[k:k + gpb]] for k in range(0, len(gperm), gpb)]
            else:
                perm = torch.randperm(len(a), generator=g).tolist()
                batches = [perm[k:k + bs] for k in range(0, len(perm), bs)]
            for bt in batches:
                idx = [i for st, en in bt for i in range(st, en)] if listwise else bt
                enc = self._batch([a[i] for i in idx], [b[i] for i in idx])
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                    logits = self.model(**enc).logits.squeeze(-1)
                logits = logits.float()
                if listwise:
                    width = max(en - st for st, en in bt)
                    mat = torch.full((len(bt), width), float("-inf"), device=logits.device)
                    target = torch.zeros(len(bt), dtype=torch.long, device=logits.device)
                    pos = 0
                    for r, (st, en) in enumerate(bt):
                        mat[r, : en - st] = logits[pos:pos + en - st]
                        target[r] = int(np.argmax(y[st:en]))
                        pos += en - st
                    loss = torch.nn.functional.cross_entropy(mat, target)
                    tot += loss.item() * len(bt)
                else:
                    loss = bce(logits, yt[idx].to(self.device))
                    tot += loss.item() * len(idx)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
            acc = val_fn() if val_fn else float("nan")
            denom = len(spans) if listwise else len(a)
            hist.append({"epoch": ep + 1, "loss": round(tot / denom, 4), "val_acc": round(acc, 4),
                         "seconds": round(time.time() - t0, 1)})
            log.info("epoch %s", hist[-1])
            if val_fn is None or acc > best:
                best, since_best = acc, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                since_best += 1
                if patience is not None and since_best >= patience:
                    log.info("early stopping after epoch %d (best %.4f)", ep + 1, best)
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        best_epoch = max(hist, key=lambda h: (h["val_acc"] if val_fn else h["epoch"], -h["epoch"]))["epoch"]
        return {"history": hist, "best_epoch": best_epoch, "pairs": len(a)}


# ---------------------------------------------------------------- training data
def icd_children_map(codes: set[str]) -> dict[str, list[str]]:
    from termmap.retrieval.engine import _parent

    ch: dict[str, list[str]] = {}
    for c in sorted(codes):
        ch.setdefault(_parent(c), []).append(c)
    return ch


def build_training_pairs(rows: pd.DataFrame, cands: pd.DataFrame, ctext: CandidateText, system: str,
                         n_neg: int, ctx_dropout: float, ctx_chars: int, seed: int,
                         icd_children: dict[str, list[str]] | None = None, neg_source: str = "fused",
                         ltr_scores: pd.DataFrame | None = None, return_groups: bool = False):
    """1 positive + n_neg hard negatives per row (gold injected if retrieval missed it).

    ICD: up to 5 dictionary siblings of the gold code are always among the negatives.
    neg_source 'fused' (v1): the rest in fused-retrieval order from the cross-encoder set.
    neg_source 'ltr' (v2, audit A3): the rest in out-of-fold LambdaRank order over the whole pool,
    then fused order — the confusions the downstream ranker actually makes.
    Returns (queries, candidate texts, labels) and, with return_groups, a group id per pair.
    """
    rng = np.random.default_rng(seed)
    pool = cands if neg_source == "ltr" else cands[cands.in_ce == 1]
    if neg_source == "ltr":
        if ltr_scores is None:
            raise ValueError("neg_source 'ltr' needs ltr_scores (row, code, ltr)")
        pool = pool.merge(ltr_scores[["row", "code", "ltr"]], on=["row", "code"], how="left")
        pool["ltr"] = pool["ltr"].fillna(-np.inf)
        by_row = {r: g.sort_values(["ltr", "fused_rank", "code"], ascending=[False, True, True])
                  for r, g in pool.groupby("row")}
    else:
        by_row = {r: g.sort_values(["fused_rank", "code"]) for r, g in pool.groupby("row")}
    A, B, Y, G = [], [], [], []
    for gid, (rid, r) in enumerate(rows.iterrows()):
        gold = r["code"]
        ctx = "" if rng.random() < ctx_dropout else context_window(r["term_norm"], r["chunk_text"], ctx_chars)
        q = query_text(r["term_norm"], ctx)
        negs: list[str] = []
        if system == ICD and icd_children is not None:
            from termmap.retrieval.engine import _parent
            sib = [c for c in icd_children.get(_parent(gold), []) if c != gold]
            negs.extend(sib[:5])
        g = by_row.get(rid)
        if g is not None:
            for c in g["code"]:
                if len(negs) >= n_neg:
                    break
                if c != gold and c not in negs:
                    negs.append(c)
        for c in [gold] + negs[:n_neg]:
            A.append(q)
            B.append(ctext(c, r["term_norm"]))
            Y.append(1.0 if c == gold else 0.0)
            G.append(gid)
    out = (A, B, np.asarray(Y, dtype=np.float32))
    return out + (np.asarray(G),) if return_groups else out


def score_candidates(ce: CrossEncoder, rows: pd.DataFrame, cands: pd.DataFrame, ctext: CandidateText,
                     with_context: bool, ctx_chars: int) -> np.ndarray:
    """Cross-encoder logit for every in_ce candidate row of `cands` (NaN elsewhere)."""
    sel = cands.index[cands.in_ce == 1]
    q = {}
    for rid, r in rows.loc[cands.loc[sel, "row"].unique()].iterrows():
        ctx = context_window(r["term_norm"], r["chunk_text"], ctx_chars) if with_context else ""
        q[rid] = query_text(r["term_norm"], ctx)
    a = [q[x] for x in cands.loc[sel, "row"]]
    terms = rows["term_norm"]
    b = [ctext(c, terms[x]) for c, x in zip(cands.loc[sel, "code"], cands.loc[sel, "row"])]
    out = np.full(len(cands), np.nan, dtype=np.float32)
    out[cands.index.get_indexer(sel)] = ce.score(a, b)
    return out


def top1_accuracy(cands: pd.DataFrame, score_col: str, n_rows: int) -> float:
    """Share of rows whose highest-scoring candidate (ties by code) is gold."""
    c = cands[cands[score_col].notna()].sort_values(["row", score_col, "code"], ascending=[True, False, True])
    top = c.drop_duplicates("row")
    return float(top["gold"].sum()) / max(n_rows, 1)
