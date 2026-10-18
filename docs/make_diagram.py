"""Draws docs/architecture.png (pipeline overview used by the README and the PDF)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis("off")

def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=color, ec="#333", lw=1))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=9.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.5, body, ha="center", va="top", fontsize=7.6, linespacing=1.35)

def arrow(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#333"))

box(0.1, 4.5, 2.1, 1.5, "Input", "entity + chunk_text\n(chunk may be empty)", "#eef2f7")
box(2.5, 4.5, 2.6, 1.5, "Stage 0 · normalize", "NFKC, lowercase, strip <br>,\nheaders, (hcc) · variants:\nspell (symspell), abbreviation\n(mined per fold), embedded code", "#e8f4ea")
box(5.4, 4.5, 2.6, 1.5, "Stage 1 · exact tags", "exact_dict · exact_memory ·\nembedded_code\n(features only, no shortcut)", "#e8f4ea")
box(8.3, 4.1, 2.6, 1.9, "Stage 2 · retrieval", "char 3-4-gram TF-IDF\nSapBERT dense (FAISS exact)\ntraining memory (char+dense)\nRRF k=60 → top-80\n+ ICD siblings", "#fff4e0")
box(8.3, 1.9, 2.6, 1.9, "Stage 3A · cross-encoder", "MedCPT fine-tuned per system\nterm [SEP] ±100 chars [ENT]\nvs code | synonyms | category\nscores fused top-20", "#fde8e8")
box(5.4, 1.9, 2.6, 1.9, "Stage 3B · LambdaRank", "LightGBM per system\nCE score/rank/gap, channel\nranks, exact flags, train freq,\nleaf/depth/tag, qualifiers", "#fde8e8")
box(2.5, 1.9, 2.6, 1.9, "Stage 4-5", "ICD: temperature softmax →\n3-char category → code\nICD-only coupling with\nSNOMED top-5 (λ=2)\nconfidence threshold", "#ece8fb")
box(0.1, 1.9, 2.1, 1.9, "Output guard", "codes validated vs\ndictionaries; fallback\nfused top-1 → most\nfrequent code", "#eef2f7")
box(0.1, 0.1, 10.8, 1.4, "Outputs", "predictions.csv (entity, chunk_text, icd10cm_code, snomed_code) · predictions_debug.csv (prob, margin, low_confidence, flags,\ntop-5, category step, pair fit)   ·   deterministic: fixed seeds, tie-break by code string, byte-identical reruns   ·   CPU or GPU, no network at predict", "#f6f6f6")
arrow(2.2, 5.25, 2.5, 5.25); arrow(5.1, 5.25, 5.4, 5.25); arrow(8.0, 5.25, 8.3, 5.25)
arrow(9.6, 4.1, 9.6, 3.8); arrow(8.3, 2.85, 8.0, 2.85); arrow(5.4, 2.85, 5.1, 2.85); arrow(2.5, 2.85, 2.2, 2.85); arrow(1.15, 1.9, 1.15, 1.5)
ax.text(5.5, 6.12, "TermMap pipeline (ICD-10-CM and SNOMED CT lanes run in parallel from Stage 2)", ha="center", fontsize=11, weight="bold")
fig.savefig("docs/architecture.png", dpi=170, bbox_inches="tight")
