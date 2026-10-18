"""TermMap demo: type a clinical term (and optional chart context), see both codes, top-5, confidence, flags.

Run: .venv/bin/streamlit run app/demo.py   (uses the saved artifacts; no network calls)
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from termmap.config import load_config, set_seeds

EXAMPLES = {
    "— pick an example —": ("", ""),
    "Abbreviation collision: 'ms' (neuro context)": ("ms", "neuro: relapsing remitting ms on ocrelizumab, new optic neuritis"),
    "Abbreviation collision: 'severe ms' (cardiac context)": ("severe ms", "echo: severe mitral stenosis, mva 0.9 cm2"),
    "OCR-mangled term": ("pnuemonia rll", "cxr: right lower lobe consolidation c/w pnuemonia"),
    "OCR-mangled term (no context)": ("diabetis melitus type 2", ""),
    "Specificity: laterality + acuity": ("acute on chronic systolic heart failure", "admitted for acute on chronic systolic chf, ef 25%"),
    "Specificity: side named, code may stay unspecified": ("left knee pain", "pt reports left knee pain x 3 weeks after fall"),
    "No context at all": ("htn", ""),
}


@st.cache_resource(show_spinner="Loading indexes and models (about 30 s)…")
def predictor():
    from termmap.pipeline import Predictor

    cfg = load_config()
    set_seeds(cfg["seed"])
    return Predictor(cfg)


def top5_table(p, system: str, top5: str) -> pd.DataFrame:
    rows = []
    for item in filter(None, top5.split("|")):
        code, prob = item.rsplit(":", 1)
        rows.append({"code": code, "probability": float(prob), "description": p.ctext[system](code)})
    return pd.DataFrame(rows)


def show_system(p, d: pd.Series, system: str, label: str) -> None:
    k = "icd" if system == "ICD10CM" else "snomed"
    code, prob = d[f"{k}_code"], float(d[f"{k}_prob"])
    low = int(d[f"{k}_low_confidence"])
    st.subheader(label)
    st.markdown(f"**{code}** — {p.ctext[system](code)}")
    c1, c2, c3 = st.columns(3)
    c1.metric("probability", f"{prob:.2f}")
    c2.metric("margin to #2", f"{float(d[f'{k}_margin']):.2f}")
    c3.metric("confidence", "LOW — review" if low else "high")
    if d[f"{k}_flags"]:
        st.caption(f"flags: {d[f'{k}_flags']}")
    if system == "ICD10CM" and d.get("icd_category_step"):
        st.caption(f"category step: {d['icd_category_step']}")
    st.dataframe(top5_table(p, system, d[f"{k}_top5"]), hide_index=True, use_container_width=True)


st.set_page_config(page_title="TermMap", layout="wide")
st.title("TermMap — clinical term → ICD-10-CM + SNOMED CT")
tab1, tab2 = st.tabs(["Single term", "Batch confidence / error view"])

with tab1:
    ex = st.selectbox("Examples", list(EXAMPLES))
    t0, c0 = EXAMPLES[ex]
    term = st.text_input("Clinical term (entity)", value=t0)
    ctx = st.text_area("Chart context (chunk_text, optional)", value=c0, height=80)
    if st.button("Map term", type="primary") and term.strip():
        p = predictor()
        _, debug = p.predict_frame(pd.DataFrame({"entity": [term], "chunk_text": [ctx]}))
        d = debug.iloc[0]
        left, right = st.columns(2)
        with left:
            show_system(p, d, "ICD10CM", "ICD-10-CM")
        with right:
            show_system(p, d, "SNOMEDCT_US", "SNOMED CT")
        st.caption(f"pair agreement (codes co-occur in training): {int(d['pair_agreement'])} · "
                   f"pair fit {float(d['pair_fit']):.2f}")

with tab2:
    st.write("Upload a `predictions_debug.csv` produced by `termmap predict` to triage rows by confidence.")
    up = st.file_uploader("predictions_debug.csv", type="csv")
    if up is not None:
        dbg = pd.read_csv(up, dtype=str, keep_default_na=False)
        for c in ("icd_prob", "snomed_prob"):
            dbg[c] = dbg[c].astype(float)
        st.metric("rows", len(dbg))
        c1, c2 = st.columns(2)
        c1.metric("ICD low-confidence share", f"{dbg['icd_low_confidence'].astype(int).mean():.0%}")
        c2.metric("SNOMED low-confidence share", f"{dbg['snomed_low_confidence'].astype(int).mean():.0%}")
        sort = st.radio("Sort by", ["icd_prob", "snomed_prob"], horizontal=True)
        st.dataframe(dbg.sort_values(sort)[["entity", "icd_code", "icd_prob", "icd_flags", "snomed_code",
                                            "snomed_prob", "snomed_flags", "pair_agreement"]],
                     hide_index=True, use_container_width=True)
