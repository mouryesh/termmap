# TermMap

Maps a clinical term extracted from a chart (`entity`) plus the chart text around it (`chunk_text`) to one
**ICD-10-CM** code and one **SNOMED CT** concept from the supplied dictionaries. Fully local: no LLM or
third-party API sees the data.

## Run

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Linux/macOS: .venv/bin/pip
.venv/Scripts/pip install -e . --no-deps
cp .env.example .env                              # optional; no secrets needed
# put the 3 competition files in data/ (never committed)
termmap predict --input eval_input.csv --output predictions.csv
```

Input: CSV with `entity` (alias `clinical_entity`) and optional `chunk_text`; any other column is ignored
(`subconcept` is never read). Output: `entity, chunk_text, icd10cm_code, snomed_code`, one row per input row in
the same order, plus `predictions_debug.csv` (probabilities, margin, low-confidence flag, top-5, flags).
Every output passes a closed-world check (`src/termmap/closed_world.py`): same row count/order, no blanks,
every code present in its dictionary, ICD format valid.

`eval_input.csv` for a self-check: `python scripts/make_eval_input.py` (unique `entity, chunk_text` of the
training file, all label columns removed, 7,522 rows). It is training data, so scores on it are optimistic.

## Pipeline

1. **Normalise** (NFKC, lowercase, strip table headers / `<br>`), spelling variant (SymSpell, dictionary vocab),
   abbreviation variant (table mined from training pairs), codes written in the term.
2. **Retrieve** per system: char 3-4-gram TF-IDF + SapBERT dense (FAISS exact IP) over every dictionary
   description, plus a training-memory index (char + dense over training terms/displays) → reciprocal rank
   fusion (k = 60) → top-80 pool (+ ICD siblings, embedded codes). BM25 is built but off (lowered recall in CV).
3. **Cross-encoder**: MedCPT cross-encoder fine-tuned per system (1 positive + 15 hard negatives, context window
   with `[ENT]` marks, 50% context dropout) scores the fused top-20.
4. **LightGBM LambdaRank** over CE score/rank/gap, channel ranks, RRF, exact-match flags, training frequency,
   code metadata (leaf, depth, chapter / semantic tag), qualifier lexicon.
5. **ICD hierarchy** (3-char category mass, margin 0.05) and **ICD-only coupling** with the SNOMED top-5
   (co-occurrence + SapBERT cosine, λ = 2 chosen by nested CV); SNOMED is never changed by coupling.
6. **Calibrated confidence**: temperature-scaled LambdaRank probability; `low_confidence` flag in the debug file.

## Results (3-fold GroupKFold by chunk, out-of-fold, entity + chunk_text)

Full tables and the error analysis are in `reports/validation.md`, `reports/error_analysis_noce.md` and `reports/report.pdf`. These quote short training terms, so per the data rules they are **submitted directly and not committed** (regenerate: `python scripts/error_report.py noce`). Headline: lower of the two systems.

| out-of-fold top-1 | ICD10CM | SNOMEDCT_US | min |
|---|---|---|---|
| **submitted build: ranker without cross-encoder** (re-measured) | **55.9** | **44.5** | **44.5** |
| v1 with cross-encoder (previously measured; weights lost, not rebuilt in time) | 61.5 | 50.0 | 50.0 |
| pool ceiling (gold in candidate pool) | 92.6 | 86.1 | |

Submitted build: seen codes 63.6 / 54.6, unseen (long tail) 34.6 / 34.4; ICD same 3-char category 66.8, same chapter
79.6; SNOMED same semantic tag 66.6; accuracy of the most confident 25%: ICD 89.8, SNOMED 72.3.

**Why no cross-encoder in the submission.** The development laptop and the trained cross-encoders were lost on
18 Oct. On the replacement laptop (power/thermal-capped GPU) one ICD cross-encoder epoch took 36 min, so the
final models could not be retrained before the deadline. The pipeline uses the cross-encoder automatically for any
system whose `artifacts/model/ce_<system>/` exists (run `python scripts/rebuild_final.py ce <system>` then
`python scripts/rebuild_final.py ltr`); otherwise it uses the no-cross-encoder ranker (`ltr_<system>_noce.joblib`,
own temperature and threshold in `noce.json`).

## Model weights (Hugging Face)

| what | where |
|---|---|
| trained TermMap artifacts: LambdaRank boosters (`ltr_*_noce.joblib`), calibration (`noce.json`, `hierarchy.json`, `coupling.json`), training memory, abbreviation table, dictionary indexes (char TF-IDF, BM25, SapBERT embeddings + encoder copy) | **https://huggingface.co/Vorun/termmap-weights** (`model/`; `index/` only if uploaded, else `termmap build-index`) |
| SapBERT dense encoder (pre-trained, used frozen) | https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext (rev `090663c`) |
| MedCPT cross-encoder (pre-trained base; fine-tuned only in v1, not in this build) | https://huggingface.co/ncbi/MedCPT-Cross-Encoder (rev `71caf65`) |

Pull everything for evaluation, then predict:

```bash
hf auth login          # only if the weights repo is private; use your own token, never commit it
python scripts/hf_weights.py download
termmap build-index    # only if index/ was not downloaded (about 30 min on an 8 GB laptop GPU)
termmap predict --input eval.csv --output predictions.csv
```

No fine-tuned cross-encoder weights exist for this build (see "Why no cross-encoder"). No API keys are in this
repository: `.env.example` lists variables only; Hugging Face tokens come from `hf auth login` or `HF_TOKEN`.

## Models, data, licences

| item | version | licence |
|---|---|---|
| `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | HF revision `090663c` | Apache-2.0 (HF model card) |
| `ncbi/MedCPT-Cross-Encoder` | HF revision `71caf65` | "other" on HF: NCBI public-domain notice (US Government work) |
| LightGBM | 4.5.0 | MIT |
| competition data (training CSV, ICD-10-CM and SNOMED CT US dictionaries) | as supplied | proprietary / competition terms; SNOMED CT under the SNOMED/UMLS licence |

Licences are as stated on the model cards; verify before any redistribution.

**Data handling.** The data never leaves private machines. It is not in this repository (`data/` is git-ignored)
and was never sent to any LLM or hosted inference API. Disclosure: during development the SNOMED cross-encoder
cross-validation was run in a **private** Kaggle notebook on a **private** Kaggle dataset (account owner's
only); nothing was made public. The final models in this submission were rebuilt locally.

## Reproducibility

- Seed 42 everywhere (Python, NumPy, torch, LightGBM `deterministic`, fixed thread count); torch deterministic
  kernels, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Predict twice on the same device gives byte-identical files
  (checked with SHA-256). GPU runs the encoders in fp16, CPU in fp32: predictions can differ between devices.
- Hardware for the final build: RTX 3070 Ti Laptop (8 GB, 55 W cap), 16 logical cores, 15 GB RAM, Windows 11,
  Python 3.12.10, torch 2.5.1+cu124. v1 development: RTX 3050 6 GB laptop + one private Kaggle T4 notebook.
- Regenerate all artifacts from scratch: `termmap build-index` then `termmap train` (cross-validation + final
  models; ≈1.5 h on an 8 GB GPU). Recovery path used tonight (reuses the saved out-of-fold tables, see
  `RECOVERY_LOG.md`): `python scripts/rebuild_final.py index|memory|ltr_noce|cooc` (and `ce <system>` + `ltr` when GPU time allows).
- Large artifacts (indexes, cross-encoder weights, boosters) are not in git; they are regenerated by the above.
- Tests: `pytest` (closed-world check, coupling regression, end-to-end format + determinism).

## Limitations

- Training-data agreement caps exact match: identical normalised terms get the same SNOMED code in only 76.4%
  of row pairs; many errors are near-twin concepts.
- Codes never seen in training are much harder (see the seen/unseen table).
- No IS-A relationship file was supplied, so SNOMED partial credit is approximated by the semantic tag.
- The 98%-precision confidence threshold covers few rows; use the coverage curve for triage.
- Stacked out-of-fold features carry mild second-order optimism (documented in the validation report).
