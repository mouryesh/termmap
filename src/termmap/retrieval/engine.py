"""Builds/loads the dictionary indexes and runs Stage 2 hybrid retrieval for one system."""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from termmap.dictionaries import Dictionary, file_hash, load_icd, load_snomed
from termmap.icd_knowledge import knowledge_terms
from termmap.io import ICD, SNOMED, SYSTEMS
from termmap.normalize import SpellCorrector, encoder_text, level1, strip_semantic_tag, tokens
from termmap.retrieval.bm25 import BM25Index
from termmap.retrieval.chargram import CharGramIndex
from termmap.retrieval.dense import DenseIndex, Encoder
from termmap.retrieval.fusion import Ranking, best_rank_merge, rrf
from termmap.retrieval.memory import MemoryIndex

log = logging.getLogger(__name__)
CHANNELS = ("bm25", "char", "dense", "dense2", "memory")


def description_table(d: Dictionary, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (code, normalized description); SNOMED semantic tags stripped for matching.

    `extra` (code, description, source) adds knowledge synonyms; the table then gets a `source`
    column ('dict' for supplied rows) and an extra row is dropped when the code already has that text.
    """
    text = d.rows["description"].map(level1)
    if d.system == SNOMED:
        text = text.map(strip_semantic_tag)
    t = pd.DataFrame({"code": d.rows["code"].values, "text": text.values})
    if extra is not None:
        x = pd.DataFrame({"code": extra["code"].values, "text": extra["description"].map(level1).values,
                          "source": extra["source"].values})
        t = pd.concat([t.assign(source="dict"), x], ignore_index=True)
        t = t[t["text"] != ""].drop_duplicates(["code", "text"]).sort_values(["code", "text"], kind="stable")
        return t.reset_index(drop=True)
    t = t[t["text"] != ""].drop_duplicates().sort_values(["code", "text"], kind="stable")
    return t.reset_index(drop=True)


def load_dictionaries(cfg: dict) -> dict[str, Dictionary]:
    art = cfg["paths"]["artifacts"]
    dc = cfg.get("dictionaries", {})
    return {ICD: load_icd(cfg["paths"]["icd_dict"], art, dc.get("icd_title", "first")),
            SNOMED: load_snomed(cfg["paths"]["snomed_dict"], art, dc.get("snomed_tag", "any"))}


def _stamp(cfg: dict) -> dict:
    st = {"icd": file_hash(cfg["paths"]["icd_dict"]), "snomed": file_hash(cfg["paths"]["snomed_dict"]),
          "dense_model": cfg["retrieval"]["dense_model"]}
    if cfg["retrieval"].get("dense_max_len", 32) != 32:  # non-default only, so v1 stamps stay valid
        st["dense_max_len"] = cfg["retrieval"]["dense_max_len"]
    if "dense2" in cfg["retrieval"].get("channels", ()):
        st["dense2_model"] = cfg["retrieval"]["dense2_model"]
    if cfg.get("dictionaries", {}).get("icd_synonyms"):
        st["icd_synonyms"] = list(cfg["dictionaries"]["icd_synonyms"])
        st["icd_index_xml"] = file_hash(cfg["paths"]["icd_index_xml"])
    return st


def _writable(path: Path) -> Path:
    """Remove a symlink at `path` before writing, so a workspace that links to another workspace's
    index (artifacts_v2 -> artifacts) never overwrites the linked files."""
    if path.is_symlink():
        path.unlink()
    return path


def build_index(cfg: dict, force: bool = False) -> None:
    """Build all dictionary indexes into artifacts/index, stamped with the dictionary file hashes."""
    out = Path(cfg["paths"]["artifacts"]) / "index"
    out.mkdir(parents=True, exist_ok=True)
    stamp = _stamp(cfg)
    manifest = out / "manifest.json"
    if not force and manifest.exists() and json.loads(manifest.read_text()).get("stamp") == stamp:
        log.info("index up to date (%s)", stamp)
        return
    dicts = load_dictionaries(cfg)
    model_dir = out / "sapbert"
    _writable(model_dir)
    enc = Encoder(cfg["retrieval"]["dense_model"], max_len=cfg["retrieval"].get("dense_max_len", 32))
    enc.save(model_dir)
    enc2 = None
    if "dense2" in cfg["retrieval"].get("channels", ()):  # item C4: MedCPT Query Encoder on both sides
        enc2 = Encoder(cfg["retrieval"]["dense2_model"], max_len=cfg["retrieval"].get("dense_max_len", 32))
        enc2.save(_writable(out / "dense2_model"))
    vocab: Counter = Counter()
    info = {}
    for sysname, d in dicts.items():
        t0 = time.time()
        syn = cfg.get("dictionaries", {}).get("icd_synonyms") if sysname == ICD else None
        extra = knowledge_terms(d.codes, syn, cfg["paths"].get("icd_index_xml")) if syn else None
        table = description_table(d, extra)
        if extra is not None:
            log.info("%s knowledge synonyms: %s", sysname, table["source"].value_counts().to_dict())
        table.to_parquet(_writable(out / f"{sysname}_table.parquet"), index=False)
        d.meta.to_parquet(_writable(out / f"{sysname}_meta.parquet"))  # codes + metadata; predict reads only this
        texts, codes = table["text"].tolist(), table["code"].tolist()
        for s in texts:
            vocab.update(tokens(s))
        joblib.dump(CharGramIndex(texts, codes), _writable(out / f"{sysname}_char.joblib"))
        joblib.dump(BM25Index(texts, codes), _writable(out / f"{sysname}_bm25.joblib"))
        emb = enc.encode([encoder_text(x) for x in texts], batch_size=512)
        np.save(_writable(out / f"{sysname}_dense.npy"), emb.astype(np.float16))
        if enc2 is not None:
            np.save(_writable(out / f"{sysname}_dense2.npy"),
                    enc2.encode([encoder_text(x) for x in texts], batch_size=512).astype(np.float16))
        info[sysname] = {"texts": len(texts), "codes": len(set(codes)), "seconds": round(time.time() - t0, 1)}
        log.info("%s index: %s", sysname, info[sysname])
    _writable(out / "vocab.json").write_text(json.dumps(dict(sorted(vocab.items()))))
    _writable(manifest).write_text(json.dumps({"stamp": stamp, "info": info}, indent=2))


@dataclass
class SystemIndex:
    system: str
    table: pd.DataFrame
    char: CharGramIndex
    bm25: BM25Index | None
    dense: DenseIndex
    meta: pd.DataFrame
    codes: set[str]
    dense2: DenseIndex | None = None
    children: dict[str, list[str]] = field(default_factory=dict)  # ICD: parent prefix -> codes


class Indexes:
    """All loaded dictionary indexes plus the shared encoder and spell corrector."""

    def __init__(self, cfg: dict, device: str | None = None):
        out = Path(cfg["paths"]["artifacts"]) / "index"
        manifest = json.loads((out / "manifest.json").read_text())
        dict_files = [Path(cfg["paths"]["icd_dict"]), Path(cfg["paths"]["snomed_dict"])]
        if cfg.get("dictionaries", {}).get("icd_synonyms"):
            dict_files.append(Path(cfg["paths"]["icd_index_xml"]))
        if all(f.exists() for f in dict_files):
            if manifest["stamp"] != _stamp(cfg):
                raise RuntimeError("dictionary files changed since the index was built: run `termmap build-index`")
        else:
            log.info("dictionary files not present; using the stamped index %s", manifest["stamp"])
        self.encoder = Encoder(str(out / "sapbert"), device=device, max_len=cfg["retrieval"].get("dense_max_len", 32))
        chans = cfg["retrieval"].get("channels", CHANNELS)
        self.encoder2 = Encoder(str(out / "dense2_model"), device=device, max_len=cfg["retrieval"].get("dense_max_len", 32)) \
            if "dense2" in chans else None
        self.spell = SpellCorrector(json.loads((out / "vocab.json").read_text()))
        self.sys: dict[str, SystemIndex] = {}
        use_bm25 = "bm25" in cfg["retrieval"].get("channels", CHANNELS)
        for s in SYSTEMS:
            table = pd.read_parquet(out / f"{s}_table.parquet")
            meta = pd.read_parquet(out / f"{s}_meta.parquet")
            emb = np.load(out / f"{s}_dense.npy")
            si = SystemIndex(s, table, joblib.load(out / f"{s}_char.joblib"), joblib.load(out / f"{s}_bm25.joblib") if use_bm25 else None,
                             DenseIndex(emb, table["code"].tolist()), meta, set(meta.index))
            del emb
            if self.encoder2 is not None:
                si.dense2 = DenseIndex(np.load(out / f"{s}_dense2.npy"), table["code"].tolist())
            if s == ICD:
                for c in sorted(si.codes):
                    si.children.setdefault(_parent(c), []).append(c)
            self.sys[s] = si


def _parent(code: str) -> str:
    p = code[:-1]
    return p[:-1] if p.endswith(".") else p


def icd_siblings(code: str, si: SystemIndex) -> list[str]:
    """Parent, 3-character category and same-parent siblings of an ICD code that exist in the dictionary."""
    out = []
    if len(code) > 3:
        par = _parent(code)
        if par in si.codes:
            out.append(par)
        if code[:3] in si.codes:
            out.append(code[:3])
        out.extend(si.children.get(par, []))
    out.extend(si.children.get(code, []))  # a parent's own children
    return out


@dataclass
class Candidates:
    """Stage 2 output for one term in one system."""

    fused: Ranking                                  # RRF-ordered (code, rrf score), top K
    ranks: dict[str, dict[str, int]]                # code -> channel -> rank (1-based)
    pool: list[str]                                 # fused top K + sibling expansion + embedded codes
    embedded: list[str] = field(default_factory=list)


class Retriever:
    """Stage 2 hybrid retrieval with per-query-string caching of the dictionary channels."""

    def __init__(self, idx: Indexes, cfg: dict):
        self.idx = idx
        self.k = cfg["retrieval"]["rrf_k"]
        self.top_k = cfg["retrieval"]["top_k"]
        self.sib_top = cfg["retrieval"]["sibling_expand_top"]
        self.max_words = cfg["retrieval"]["long_term_words"]
        self.channels = tuple(cfg["retrieval"].get("channels", CHANNELS))
        # per-system {system: {channel: w}} or one shared {channel: w}; missing channel -> weight 1
        self.weights = dict(cfg["retrieval"].get("channel_weights", {}))
        self._dict_cache: dict[tuple[str, str], dict[str, Ranking]] = {}
        self._emb_cache: dict[str, np.ndarray] = {}
        self._emb2_cache: dict[str, np.ndarray] = {}

    def embed(self, strings: list[str], second: bool = False) -> np.ndarray:
        """SapBERT (or, with second=True, dense2) embeddings of query strings (first clause for very
        long strings), cached."""
        cache, enc = (self._emb2_cache, self.idx.encoder2) if second else (self._emb_cache, self.idx.encoder)
        new = sorted({s for s in strings if s not in cache})
        if new:
            cache.update(zip(new, enc.encode([encoder_text(s, self.max_words) for s in new])))
        return np.stack([cache[s] for s in strings]) if strings else np.zeros((0, 768), np.float32)

    def prime(self, system: str, strings: list[str]) -> None:
        """Run the three dictionary channels for all uncached query strings in one batch."""
        si = self.idx.sys[system]
        new = sorted({s for s in strings if (system, s) not in self._dict_cache})
        if not new:
            return
        t0 = time.time()
        b = si.bm25.search(new) if si.bm25 is not None else [[] for _ in new]
        c = si.char.search(new)
        d = si.dense.search(self.embed(new))
        d2 = si.dense2.search(self.embed(new, second=True)) if si.dense2 is not None else [[] for _ in new]
        for s, rb, rc, rd, rd2 in zip(new, b, c, d, d2):
            self._dict_cache[(system, s)] = {"bm25": rb, "char": rc, "dense": rd, "dense2": rd2}
        log.info("%s: dictionary channels for %d strings in %.1fs", system, len(new), time.time() - t0)

    def channel_rankings(self, system: str, variants: list[dict[str, str]],
                         memory: MemoryIndex | None, use_variants: tuple[str, ...] = ("orig", "spell", "abbr"),
                         channels: tuple[str, ...] | None = None) -> list[dict[str, Ranking]]:
        """Per term: each channel's ranking, merged across query variants by best rank."""
        channels = channels or self.channels
        strings = [v[n] for v in variants for n in use_variants if n in v and v[n]]
        self.prime(system, strings)
        mem: dict[str, Ranking] = {}
        if memory is not None and "memory" in channels:
            uniq = sorted(set(strings))
            fused, _, _ = memory.search(uniq, self.embed(uniq))
            mem = dict(zip(uniq, fused))
        out = []
        for v in variants:
            qs = [v[n] for n in use_variants if n in v and v[n]]  # an empty term retrieves nothing
            per = {}
            for ch in channels:
                if ch == "memory":
                    per[ch] = best_rank_merge([mem[q] for q in qs]) if mem else []
                else:
                    per[ch] = best_rank_merge([self._dict_cache[(system, q)][ch] for q in qs])
            out.append(per)
        return out

    def fuse(self, system: str, per: dict[str, Ranking], embedded: list[str] | None = None,
             top_k: int | None = None) -> Candidates:
        """RRF over channels, top K, ICD sibling expansion, embedded codes appended."""
        top_k = top_k or self.top_k
        chans = list(per)
        w = self.weights.get(system, self.weights)
        fused = rrf([per[ch] for ch in chans], self.k, [float(w.get(ch, 1.0)) for ch in chans])[:top_k]
        ranks = {}
        for ch, r in per.items():
            for i, (code, _) in enumerate(r, start=1):
                ranks.setdefault(code, {})[ch] = i
        pool = [c for c, _ in fused]
        seen = set(pool)
        if system == ICD:
            for c, _ in fused[: self.sib_top]:
                for s in icd_siblings(c, self.idx.sys[ICD]):
                    if s not in seen:
                        pool.append(s); seen.add(s)
        for c in embedded or []:
            if c not in seen:
                pool.append(c); seen.add(c)
        return Candidates(fused, ranks, pool, list(embedded or []))
