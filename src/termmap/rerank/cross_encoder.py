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
    """Cross-encoder text for a code: preferred term | up to 3 synonyms | ICD category title / SNOMED tag."""

    def __init__(self, system: str, table: pd.DataFrame, meta: pd.DataFrame, n_syn: int = 3):
        self.system, self.meta, self.n_syn = system, meta, n_syn
        self.texts = table.groupby("code", sort=False)["text"].apply(list).to_dict()
        self._cache: dict[str, str] = {}

    def __call__(self, code: str) -> str:
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
                    val_fn=None, epochs: int | None = None) -> dict:
        """BCE fine-tuning; after each epoch val_fn() -> accuracy, best epoch's weights kept."""
        rc = cfg["rerank"]
        epochs = epochs or rc["epochs"]
        bs = rc["batch_size"]
        opt = torch.optim.AdamW(self.model.parameters(), lr=rc["lr"], weight_decay=0.01,
                                fused=self.device == "cuda")
        steps = epochs * ((len(a) + bs - 1) // bs)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = torch.amp.GradScaler("cuda", enabled=self.device == "cuda")
        loss_fn = torch.nn.BCEWithLogitsLoss()
        g = torch.Generator().manual_seed(seed)
        yt = torch.tensor(y, dtype=torch.float32)
        hist, best, best_state = [], -1.0, None
        for ep in range(epochs):
            self.model.train()
            perm = torch.randperm(len(a), generator=g).tolist()
            t0, tot = time.time(), 0.0
            for s in range(0, len(perm), bs):
                idx = perm[s:s + bs]
                enc = self._batch([a[i] for i in idx], [b[i] for i in idx])
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                    logits = self.model(**enc).logits.squeeze(-1)
                loss = loss_fn(logits.float(), yt[idx].to(self.device))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += loss.item() * len(idx)
            acc = val_fn() if val_fn else float("nan")
            hist.append({"epoch": ep + 1, "loss": round(tot / len(a), 4), "val_acc": round(acc, 4),
                         "seconds": round(time.time() - t0, 1)})
            log.info("epoch %s", hist[-1])
            if val_fn is None or acc > best:
                best = acc
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        if best_state is not None:
            self.model.load_state_dict(best_state)
        best_epoch = max(hist, key=lambda h: (h["val_acc"] if val_fn else h["epoch"], -h["epoch"]))["epoch"]
        return {"history": hist, "best_epoch": best_epoch}


# ---------------------------------------------------------------- training data
def icd_children_map(codes: set[str]) -> dict[str, list[str]]:
    from termmap.retrieval.engine import _parent

    ch: dict[str, list[str]] = {}
    for c in sorted(codes):
        ch.setdefault(_parent(c), []).append(c)
    return ch


def build_training_pairs(rows: pd.DataFrame, cands: pd.DataFrame, ctext: CandidateText, system: str,
                         n_neg: int, ctx_dropout: float, ctx_chars: int, seed: int,
                         icd_children: dict[str, list[str]] | None = None) -> tuple[list[str], list[str], np.ndarray]:
    """1 positive + n_neg hard negatives per row from its Stage 2 candidates (gold injected if missing).

    ICD: up to 5 dictionary siblings of the gold code are always among the negatives.
    """
    rng = np.random.default_rng(seed)
    by_row = {r: g.sort_values(["fused_rank", "code"]) for r, g in cands[cands.in_ce == 1].groupby("row")}
    A, B, Y = [], [], []
    for rid, r in rows.iterrows():
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
            B.append(ctext(c))
            Y.append(1.0 if c == gold else 0.0)
    return A, B, np.asarray(Y, dtype=np.float32)


def score_candidates(ce: CrossEncoder, rows: pd.DataFrame, cands: pd.DataFrame, ctext: CandidateText,
                     with_context: bool, ctx_chars: int) -> np.ndarray:
    """Cross-encoder logit for every in_ce candidate row of `cands` (NaN elsewhere)."""
    sel = cands.index[cands.in_ce == 1]
    q = {}
    for rid, r in rows.loc[cands.loc[sel, "row"].unique()].iterrows():
        ctx = context_window(r["term_norm"], r["chunk_text"], ctx_chars) if with_context else ""
        q[rid] = query_text(r["term_norm"], ctx)
    a = [q[x] for x in cands.loc[sel, "row"]]
    b = [ctext(c) for c in cands.loc[sel, "code"]]
    out = np.full(len(cands), np.nan, dtype=np.float32)
    out[cands.index.get_indexer(sel)] = ce.score(a, b)
    return out


def top1_accuracy(cands: pd.DataFrame, score_col: str, n_rows: int) -> float:
    """Share of rows whose highest-scoring candidate (ties by code) is gold."""
    c = cands[cands[score_col].notna()].sort_values(["row", score_col, "code"], ascending=[True, False, True])
    top = c.drop_duplicates("row")
    return float(top["gold"].sum()) / max(n_rows, 1)
