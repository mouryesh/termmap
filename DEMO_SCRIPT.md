# TermMap demo script (2-3 minutes)

Setup before recording: `demo_inputs.csv` (repo root, git-ignored: training rows chosen for illustration only)
and a terminal in the repo with the venv installed. Everything runs offline from `artifacts/`.

## 0:00-0:20 Problem
"Clinical charts mention conditions as free text: abbreviations, typos, OCR noise, table cells. TermMap maps
each extracted term, with its surrounding chart text, to one ICD-10-CM code and one SNOMED CT concept from the
supplied dictionaries, and says how sure it is."

## 0:20-1:00 One command and the output
```bash
.venv/bin/termmap predict --input demo_inputs.csv --output demo_predictions.csv
```
```bash
.venv/bin/python -c "import pandas as pd; print(pd.read_csv('demo_predictions.csv', keep_default_na=False)[['entity', 'icd10cm_code', 'snomed_code']].to_string())"
```
Show: one row per input row, same order, both codes always filled and checked against the dictionaries.
Then open `demo_predictions_debug.csv`: probability, top1-top2 margin, `low_confidence`, flags, top-5.

## 1:00-2:00 Three cases from the debug output
Each row of `demo_inputs.csv` is one case; point at these debug columns: `icd_top5` / `snomed_top5`
(candidates with probabilities), `icd_prob`, `icd_margin`, `icd_flags`, `icd_category_step`.
1. **Abbreviations** (row 1, `acute hypox resp fx`; row 4, `afl`): mined abbreviation expansions and the training
   memory turn shorthand into acute respiratory failure with hypoxia / atrial flutter.
2. **Typo** (row 2, `accupunture`): the spelling variant ("acupuncture") and the character channel still retrieve it.
3. **Specificity / qualifier** (row 3, `... with hypoxia and hypercapnea`): the ICD pick is the "with hypoxia and
   hypercapnia" code, not the unspecified one; show `icd_top5`, the margin and the category step.

Say on camera: these rows come from the training data and are shown only to illustrate the mechanics; the
model has seen them, so they are not evidence of accuracy (the out-of-fold numbers are).

## 2:00-2:30 Error analysis and honest ceiling
Show the failure-mode table in `reports/validation.md` and say:
"Out-of-fold, min(ICD, SNOMED) is 50.0% (ICD 61.5, SNOMED 50.0). The candidate pool contains the gold code 92.6% / 86.1%
of the time; the rest is retrieval misses, codes never seen in training, near-twin concepts and coder conventions.
Even identical terms get the same SNOMED code in only 76% of row pairs. When we answer only the most confident
quarter of rows, accuracy is 89.6% (ICD) and 85.3% (SNOMED); the rest is flagged for human review."

## Interactive alternative
```bash
.venv/bin/streamlit run app/demo.py
```
