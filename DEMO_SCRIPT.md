# TermMap demo script (about 2 min 40 s, about 330 words spoken)

Before recording: `python scripts/pick_demo_rows.py` → `demo_inputs.csv` (8 rows chosen **by rule** from the
out-of-fold results, first match in file order per case; see `demo_cases.csv` for each row's reason). Both files are
git-ignored because they hold training text. Then pre-run predict once so the indexes are warm in the OS cache.

| # | case (rule; the term itself is in `demo_cases.csv`) |
|---|---|
| 1 | easy hit, ICD |
| 2 | easy hit, SNOMED |
| 3 | abbreviation, correct |
| 4 | abbreviation, wrong |
| 5 | near-twin error |
| 6 | abstained (low confidence) |
| 7 | long-tail code, correct (SNOMED) |
| 8 | long-tail code, correct (ICD) |

**0:00–0:20 Problem.** "Charts mention conditions as free text: abbreviations, typos, table cells. TermMap maps each
term, with its surrounding chart text, to one ICD-10-CM code and one SNOMED CT concept from the supplied
dictionaries, fully offline, and says how sure it is."

**0:20–0:50 One command.**
```bash
termmap predict --input demo_inputs.csv --output demo_predictions.csv
```
"One row out per row in, same order, both codes always filled, and every code checked against its dictionary
before the file is written."

**0:50–1:50 Cases (open `demo_predictions_debug.csv`: `*_prob`, `*_top5`, `*_low_confidence`).**
Rows 1–2: "plain hits with high probability." Row 3: "the abbreviation is expanded by a table mined from training
pairs." Row 4: "another abbreviation shows the limit: the ranker picks a related but wrong concept." Row 5:
"a near-twin: right family, wrong concept; over a third of SNOMED in-pool errors look like this." Row 6: "low
probability, so it is flagged for human review instead of trusted." Rows 7–8: "codes never seen in training,
still found through the dictionary indexes."

**1:50–2:40 Honest numbers.** "Out-of-fold over all 9,644 training rows, the submitted build gets ICD 55.9 and
SNOMED 44.5 percent exact. Answering only the most confident quarter, accuracy is 90 percent for ICD and 72 for
SNOMED. The gold code is in our candidate pool 93 and 86 percent of the time, and identical terms receive the
same SNOMED code in only 76 percent of pairs, so the training labels themselves cap exact match. Our earlier
cross-encoder build reached 61.5 and 50.0; those weights were lost with a laptop the day before the deadline,
and we report the build we could rebuild and verify."
