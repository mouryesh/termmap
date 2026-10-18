"""Train-time artifact assembly and the end-to-end predict path.

retrieval -> cross-encoder -> LambdaRank -> ICD hierarchy / SNOMED argmax -> output guard.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from termmap.build_dataset import build_rows, candidate_frame
from termmap.coupling import CodeEmbeddings, CooccurrenceTable, couple
from termmap.hierarchy import choose_argmax, choose_icd, group_softmax
from termmap.io import ICD, SNOMED, SYSTEMS, load_inference_input
from termmap.normalize import embedded_codes, level1, mine_abbreviations, query_variants
from termmap.retrieval.engine import Indexes, Retriever
from termmap.retrieval.memory import MemoryIndex

log = logging.getLogger(__name__)


def model_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["artifacts"]) / "model"


def fit_final(cfg: dict, idx: Indexes, retr: Retriever) -> None:
    """Abbreviation table and training memory from ALL training rows (hyperparameters fixed in CV)."""
    out = model_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    df = build_rows(cfg, {s: idx.sys[s].codes for s in SYSTEMS})
    abbr = mine_abbreviations(df["term_norm"].tolist(), df["display_norm"].tolist())
    (out / "abbreviations.json").write_text(json.dumps(abbr, indent=0, sort_keys=True))
    for s in SYSTEMS:
        rows = df[df.coding_system == s]
        texts = rows["term_norm"].tolist() + rows["display_norm"].tolist()
        codes = rows["code"].tolist() * 2
        keep = [i for i, t in enumerate(texts) if t]
        texts, codes = [texts[i] for i in keep], [codes[i] for i in keep]
        joblib.dump({"texts": texts, "codes": codes, "emb": retr.embed(texts).astype(np.float16)},
                    out / f"{s}_memory.joblib")
        rows[["term_norm", "display_norm", "code"]].to_parquet(out / f"{s}_train_stats.parquet", index=False)
        fallback = rows["code"].value_counts().sort_index().sort_values(ascending=False, kind="stable").index[0]
        (out / f"{s}_fallback.txt").write_text(fallback)
    log.info("final artifacts written to %s (%d abbreviations)", out, len(abbr))


class Predictor:
    """Loads indexes and final artifacts once; predicts both systems for every input row."""

    def __init__(self, cfg: dict, device: str | None = None):
        from termmap.rerank.cross_encoder import CandidateText, CrossEncoder
        from termmap.rerank.features import FeatureBuilder

        self.cfg = cfg
        if device is None and os.environ.get("TERMMAP_USE_GPU", "1") == "0":
            device = "cpu"
        self.idx = Indexes(cfg, device=device)
        self.retr = Retriever(self.idx, cfg)
        md = model_dir(cfg)
        art = Path(cfg["paths"]["artifacts"])
        self.abbr = json.loads((md / "abbreviations.json").read_text())
        hp = md / "hierarchy.json"
        self.calib = json.loads(hp.read_text()) if hp.exists() else {
            ICD: {"T": 1.0, "margin": cfg["hierarchy"]["category_margin"]}, SNOMED: {"T": 1.0}}
        self.coupling = json.loads((md / "coupling.json").read_text())
        self.conf_thr = {k: (v["threshold"] if v["threshold"] is not None else float("inf"))
                         for k, v in json.loads((md / "confidence.json").read_text()).items()}
        pairs = pd.read_parquet(md / "cooc_pairs.parquet")
        self.cooc = CooccurrenceTable(list(zip(pairs["icd"], pairs["snomed"])))
        self.memory, self.fallback, self.ce, self.ctext, self.fb, self.ltr, self.stats = {}, {}, {}, {}, {}, {}, {}
        metas = {}
        for s in SYSTEMS:
            m = joblib.load(md / f"{s}_memory.joblib")
            self.memory[s] = MemoryIndex(m["texts"], m["codes"], m["emb"].astype(np.float32), self.retr.k)
            self.fallback[s] = (md / f"{s}_fallback.txt").read_text().strip()
            table = self.idx.sys[s].table
            meta = metas[s] = pd.read_parquet(art / "index" / f"{s}_meta.parquet")
            self.ctext[s] = CandidateText(s, table, meta)
            self.fb[s] = FeatureBuilder(s, table, meta, self.ctext[s])
            if (md / f"ce_{s}" / "config.json").exists() and (md / f"ltr_{s}.joblib").exists():
                self.ce[s] = CrossEncoder(str(md / f"ce_{s}"), device=self.idx.encoder.device,
                                          max_len=cfg["rerank"]["max_len"])
                self.ltr[s] = joblib.load(md / f"ltr_{s}.joblib")
            else:  # fallback ranker without cross-encoder features, with its own calibration
                noce = json.loads((md / "noce.json").read_text())[s]
                log.warning("%s: no final cross-encoder; using the no-cross-encoder ranker", s)
                self.ce[s] = None
                self.ltr[s] = joblib.load(md / f"ltr_{s}_noce.joblib")
                self.calib[s] = {**self.calib[s], "T": noce["T"]}
                self.conf_thr[s] = noce["threshold"] if noce["threshold"] is not None else float("inf")
            self.stats[s] = pd.read_parquet(md / f"{s}_train_stats.parquet")
        self.code_emb = CodeEmbeddings(self.idx.encoder, metas)

    def _rank_system(self, s: str, rows: pd.DataFrame, cands: list) -> pd.DataFrame:
        """Cross-encoder + LambdaRank scores and calibrated probabilities for every pool candidate."""
        from termmap.rerank.cross_encoder import score_candidates

        rc = self.cfg["rerank"]
        cf = candidate_frame(s, rows.index.tolist(), [None] * len(rows), cands, rc["ce_top"])
        if cf.empty:
            return cf
        if self.ce[s] is not None:
            cf["ce_ctx"] = score_candidates(self.ce[s], rows, cf, self.ctext[s], True, rc["context_chars"])
        feats = self.fb[s].build(cf, rows, self.stats[s], "ce_ctx" if self.ce[s] is not None else None)
        booster = self.ltr[s]
        cf["ltr"] = booster.predict(feats[booster.feature_name()])
        cf["prob"] = group_softmax(cf, "ltr", self.calib[s]["T"])
        return cf

    def _choose(self, s: str, g, cand) -> dict:
        """Independent choice for one term in one system (hierarchy for ICD), with fallbacks."""
        if g is not None and len(g):
            gc, gp = g["code"].tolist(), g["prob"].to_numpy()
            order = np.lexsort((np.array(gc), -gp))
            out = {"top": [(gc[i], float(gp[i])) for i in order[: self.coupling["top_n"]]],
                   "top5": "|".join(f"{gc[i]}:{gp[i]:.3f}" for i in order[:5]), "flag": "", "cat": "",
                   "probs": dict(zip(gc, gp.tolist())), "argmax": gc[order[0]],
                   "margin": float(gp[order[0]] - (gp[order[1]] if len(order) > 1 else 0.0))}
            if s == ICD:
                code, info = choose_icd(gc, gp, self.calib[ICD]["margin"])
                out["cat"] = f"{info['top_category']}:{info['category_mass']}|kept={info['kept']}"
                out["flag"] = "category_override" if info["changed_from_argmax"] else ""
            else:
                code = out["argmax"]
            out["code"] = code
            return out
        code, flag = (cand.fused[0][0], "fallback_fused") if cand.fused else (self.fallback[s], "fallback_default")
        return {"code": code, "top": [], "top5": "", "flag": flag, "cat": "", "probs": {code: 0.0},
                "argmax": code, "margin": 0.0}

    def predict_frame(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (predictions, debug). Every output code is validated against its dictionary."""
        rows = pd.DataFrame({"term_norm": df["entity"].map(level1).values,
                             "chunk_text": df["chunk_text"].values}, index=range(len(df)))
        variants = [query_variants(t, self.idx.spell, self.abbr) for t in rows["term_norm"]]
        choice: dict[str, list[dict]] = {}
        for s in SYSTEMS:
            per_rows = self.retr.channel_rankings(s, variants, self.memory[s])
            cands = []
            for raw, per in zip(df["entity"], per_rows):
                ei, es = embedded_codes(raw, self.idx.sys[ICD].codes, self.idx.sys[SNOMED].codes)
                cands.append(self.retr.fuse(s, per, ei if s == ICD else es))
            try:
                cf = self._rank_system(s, rows, cands)
                by_row = {r: g for r, g in cf.groupby("row")} if not cf.empty else {}
            except Exception:  # never lose a row: fall back to fused retrieval for the whole system
                log.exception("%s reranking failed; falling back to fused retrieval", s)
                by_row = {}
            choice[s] = [self._choose(s, by_row.get(r), cand) for r, cand in enumerate(cands)]
        # Stage 5: SNOMED evidence may move the ICD pick (icd_only coupling); SNOMED stays independent
        lam = self.coupling["lambda"]
        self.code_emb.prime([(ICD, c) for ch in choice[ICD] for c, _ in ch["top"]] +
                            [(SNOMED, c) for ch in choice[SNOMED] for c, _ in ch["top"]])
        fits, coocs = [], []
        for ci, cs in zip(choice[ICD], choice[SNOMED]):
            fit = co = 0.0
            if ci["top"] and cs["top"] and lam > 0:
                ic, _, _, _ = couple(ci["top"], cs["top"], lam, self.cooc, self.code_emb)
                if ic != ci["argmax"]:
                    ci["code"], ci["flag"] = ic, (ci["flag"] + "|" if ci["flag"] else "") + "coupling_override"
            if ci["top"] and cs["top"]:
                co = self.cooc(ci["code"], cs["code"])
                fit = 0.5 * co + 0.5 * self.code_emb.cosine(ci["code"], cs["code"])
            fits.append(round(fit, 4)); coocs.append(round(co, 4))
        pred = df[["entity", "chunk_text"]].copy()
        debug = df[["entity", "chunk_text"]].copy()
        col = {ICD: "icd10cm_code", SNOMED: "snomed_code"}
        for s in SYSTEMS:
            p = "icd" if s == ICD else "snomed"
            codes, probs = [], []
            for ch in choice[s]:
                code = ch["code"]
                if code not in self.idx.sys[s].codes:
                    code, ch["flag"] = self.fallback[s], "fallback_invalid"
                codes.append(code)
                probs.append(round(float(ch["probs"].get(code, 0.0)), 6))
            assert all(c in self.idx.sys[s].codes for c in codes), "output guard: invalid code"
            pred[col[s]] = codes
            debug[f"{p}_code"] = codes
            debug[f"{p}_prob"] = probs
            debug[f"{p}_margin"] = [round(ch["margin"], 6) for ch in choice[s]]
            debug[f"{p}_low_confidence"] = [int(x < self.conf_thr[s]) for x in probs]
            debug[f"{p}_flags"] = [ch["flag"] for ch in choice[s]]
            debug[f"{p}_top5"] = [ch["top5"] for ch in choice[s]]
            if s == ICD:
                debug["icd_category_step"] = [ch["cat"] for ch in choice[s]]
        debug["pair_fit"] = fits
        debug["pair_cooccurrence"] = coocs
        debug["pair_agreement"] = [int(c > 0) for c in coocs]
        return pred, debug


def predict_file(cfg: dict, input_path: str, output_path: str) -> dict:
    """The single submission command: read input, predict both codes, write predictions + debug."""
    t0 = time.time()
    df = load_inference_input(input_path)
    p = Predictor(cfg)
    t1 = time.time()
    pred, debug = p.predict_frame(df)
    t2 = time.time()
    assert len(pred) == len(df) and not pred.isna().any().any()
    assert (pred["icd10cm_code"] != "").all() and (pred["snomed_code"] != "").all()
    from termmap.closed_world import check
    cw = check(pred, df, p.idx.sys[ICD].codes, p.idx.sys[SNOMED].codes)
    pred.to_csv(output_path, index=False)
    dbg = Path(output_path).with_name(Path(output_path).stem + "_debug.csv")
    debug.to_csv(dbg, index=False)
    stats = {"rows": len(df), "load_s": round(t1 - t0, 1), "predict_s": round(t2 - t1, 1),
             "mean_ms_per_term_batch": round(1000 * (t2 - t1) / max(len(df), 1), 2), "closed_world": cw}
    log.info("predict: %s", stats)
    return stats
