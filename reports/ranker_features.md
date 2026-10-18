# Track F: LightGBM ranker feature rebuild (18 Oct 2026)

**Decision (F8): not adopted.** The submission keeps the v1 ranker. Tagged follow-up: `ranker-v2-experiment`.

Protocol: v1 saved pools and out-of-fold cross-encoder scores (`artifacts/cv/candidates_ce_*.parquet`), the same 3 folds
(chunk groups), the same ranker settings and early-stopping split as v1. Feature statistics come from training folds only.
Top-1 counts every row (gold outside the pool = wrong). Interval: paired bootstrap 95% over rows (2,000 resamples, seed 42).
Noise floor: one standard error is about 1.0 point (ICD, 2,304 rows) and 0.6 (SNOMED, 7,340 rows).

Groups (in `FeatureBuilder(groups=...)`, computable at predict time): **string** (F2: rapidfuzz ratio / token-set / partial vs
preferred name, best token-set over all descriptions, token recall/precision, containment, length gap); **term** (F3: tokens,
chars, digit, short, vowel-less, chunk length, table chunk, term in chunk, qualifier classes); **structure** (F5: description
count, name length, other/NEC/NOS flags, ICD 7th character, unspecified x qualifiers, over-coding x digit); **pool** (F4 subset:
CE softmax share, CE mass and size of the ICD category / SNOMED tag in the pool).

| system | variant | top-1 | diff vs baseline | 95% CI | per-fold diff |
|---|---|---|---|---|---|
| ICD10CM | baseline | 60.68 |  |  |  |
| ICD10CM | +string | 60.55 | -0.13 | [-1.0, 0.74] | [-1.11, 0.26, 0.54] |
| ICD10CM | +term | 60.29 | -0.39 | [-1.13, 0.35] | [-0.49, 0.4, -1.08] |
| ICD10CM | +structure | 60.46 | -0.22 | [-1.09, 0.69] | [0.49, 0.67, -1.89] |
| ICD10CM | +pool | 60.72 | 0.04 | [-0.78, 0.83] | [-0.25, 1.33, -0.95] |
| ICD10CM | all | 60.59 | -0.09 | [-1.0, 0.78] | [-0.62, 1.07, -0.68] |
| SNOMEDCT_US | baseline | 49.97 |  |  |  |
| SNOMEDCT_US | +string | 49.96 | -0.01 | [-0.34, 0.31] | [-0.04, 0.2, -0.2] |
| SNOMEDCT_US | +term | 49.85 | -0.12 | [-0.44, 0.19] | [-0.04, 0.0, -0.32] |
| SNOMEDCT_US | +structure | 50.12 | 0.15 | [-0.18, 0.5] | [0.33, -0.04, 0.17] |
| SNOMEDCT_US | +pool | 50.07 | 0.1 | [-0.33, 0.52] | [0.29, 0.24, -0.24] |
| SNOMEDCT_US | all | 50.26 | 0.29 | [-0.16, 0.76] | [0.63, 0.24, 0.0] |

Objective / regularisation (all groups), chosen on the inner split only (inner top-1 = early-stopping split inside the
training folds); the out-of-fold column is shown for the record, not used to choose.

| system | setting | inner top-1 | out-of-fold top-1 |
|---|---|---|---|
| ICD10CM | v1 | 65.58 | 60.59 |
| ICD10CM | lr_at1_t10 | 65.34 | 58.68 |
| ICD10CM | xendcg | 64.9 | 60.07 |
| ICD10CM | leaves15 | 64.68 | 60.37 |
| ICD10CM | leaves7_mcs50_l2 | 64.48 | 58.85 |
| ICD10CM | ff05 (chosen) | 66.01 | 60.63 |
| SNOMEDCT_US | v1 | 52.14 | 50.26 |
| SNOMEDCT_US | lr_at1_t10 (chosen) | 52.48 | 50.3 |
| SNOMEDCT_US | xendcg | 51.26 | 49.74 |
| SNOMEDCT_US | leaves15 | 51.32 | 49.81 |
| SNOMEDCT_US | leaves7_mcs50_l2 | 51.87 | 49.54 |
| SNOMEDCT_US | ff05 | 52.21 | 50.3 |

Result: the chosen settings give ICD 60.63 (baseline 60.68) and SNOMED 50.30 (baseline 49.97): min(ICD, SNOMED) +0.33,
below the +1.0 adoption threshold. No group's interval excludes zero. The ranker is saturated on string/term/structure
signals the cross-encoder and channel ranks already carry; the remaining errors are retrieval misses, unseen codes and
near-twin / coder-convention choices that these features cannot separate.

Not run (time-boxed): F1 raw channel scores (needs pool rescoring), F4 near-twin features, F6 context-in-chunk, seed bagging.
