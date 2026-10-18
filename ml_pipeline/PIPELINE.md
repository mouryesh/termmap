# TermMap ML pipeline ledger

Milestone mapping (build prompt §10 → ml-pipeline steps):
M0 → steps 1–3 (Gate A) · M1 → steps 4–9 (retrieval = baseline; Gate B before any model training) ·
M2–M5 → steps 10–12 (Gate C) · M7 → steps 13–14, 16 · step 15 only if asked.
"Test set" = the held-out CV folds and held-out-code split; the hidden HiLabs eval file is never seen.

| # | Step | Status | Result |
|---|---|---|---|
| 1 | Data inspection | done | 9,646 rows × 6 cols; 7,522 term instances; no missing values; 2 invalid ICD rows (L6.0); no leakage suspects |
| 2 | EDA | done | §4 facts reproduced (reports/data_audit.json); 7 figures below |
| 3 | Problem contract | approved-gate | Gate A approved 2026-10-17 |
| 4 | Data cleaning | done | 2 rows with invalid code L6.0 dropped + logged (D4); nothing else removed; subconcept dropped at load |
| 5 | Data engineering | done | long rows per system (ICD 2,304 / SNOMED 7,340); Level-1 normalized term + display; dictionary tables (ICD 159k texts/98k codes, SNOMED 985k-row file → deduped texts) |
| 6 | Split | done | GroupKFold(5) on in-memory canonical chunk (folds 1,929 rows each); long-tail = 20% of codes held out (ICD 424 / SNOMED 1,469 rows) |
| 7 | Feature engineering | done (M1 part) | Level-2 variants: spell (symspell, dictionary vocab), abbreviation table mined per training fold, embedded codes, long-term clause |
| 8 | Preprocessing | done | indexes built from dictionaries only, hash-stamped (artifacts/index) |
| 9 | Baseline | done | char TF-IDF top-1 20.9/30.2; memory NN 48.4/29.0; fused RRF top-1 39.0/35.1, R@80 88.9/86.0 (ICD/SNOMED, 5-fold CV) |
| 10 | Model training | done | retrieval -> MedCPT CE (top-20) -> LambdaRank -> ICD hierarchy -> ICD-only coupling |
| 11 | Tuning | done | K=80; CE top-20; LTR ndcg@10 stop; margin 0.05; T 0.889; coupling lambda 2.0 (inner CV) |
| 12 | Evaluation | done | final 3-fold OOF: ICD 61.5, SNOMED 50.0 (min 50.0); ECE .042/.077; 98%-precision coverage 9.5%/0.3%; GPU 225 ms, CPU 1.16 s per term |
| 13 | Error analysis | todo | |
| 14 | Final test | todo | |
| 15 | Deployment | todo | |
| 16 | Monitoring plan | todo | |

## Step 3 — problem contract
- **Target:** for each (entity, chunk_text) instance, one ICD-10-CM code and one SNOMED CT concept id from the supplied dictionaries.
- **Unit / population:** one clinical term extracted from a chart, any subconcept (subconcept never used).
- **Information at prediction time:** `entity` and `chunk_text` only (chunk may be empty). No coding_system, display, subconcept, IDs.
- **Objective / metric:** exact-match accuracy per system; headline = min(ICD, SNOMED). Secondary: recall@10/50, long-tail accuracy, ICD category accuracy, calibration, accuracy vs coverage, latency.
- **Validation:** 5-fold GroupKFold on in-memory canonical chunk text; held-out 20% of gold codes for long-tail.
- **Constraints:** deterministic, one command, CPU predict from artifacts, no network at predict, no data leaves the machine.
- **Model families:** retrieval (BM25, char TF-IDF, SapBERT, training memory) + cross-encoder + LambdaRank (§5 of build prompt). Not a flat classifier: 466/867 ICD and 2,964/4,335 SNOMED codes are singletons.

## Model rationale (proposed for Gate B)
Model rationale:
- traits: multiclass over ~98k ICD / ~350k SNOMED codes, 2,304 ICD + 7,340 SNOMED labelled rows, 466/867 ICD and 2,964/4,335 SNOMED gold codes are singletons, groups = shared chunk text, no datetime
- baseline: char TF-IDF top-1 (20.9 / 30.2) and nearest training term (48.4 / 29.0); fused RRF retrieval top-1 39.0 / 35.1
- candidates: MedCPT cross-encoder fine-tuned per system on retrieval hard negatives (M2), then LightGBM LambdaRank over cross-encoder + retrieval + qualifier features (M3)
- ruled out: flat softmax classifier over seen codes (ceiling 79.8% ICD / 59.6% SNOMED, singletons); random split (shared chunks leak); fine-tuning SapBERT (D10 trigger met, SNOMED R@50 82.3% < 90%, but not affordable before the deadline)
- metric: exact-match accuracy per system, headline min(ICD, SNOMED); recall@K as the reranker ceiling; long-tail accuracy on held-out codes

## Gate approvals
- Gate A: approved 2026-10-17
- Gate B: approved 2026-10-17
- Gate C: approved 2026-10-18
- Override: decision D14 - user approved ICD-only coupling (SNOMED stays independent) 2026-10-18 - reason: joint coupling netted zero on paired rows (ICD +0.3, SNOMED -0.4); ICD-only +0.7 ICD
- Override: M6 skipped - user approved skipping optional LLM milestone 2026-10-18 - reason: deadline
- Override: M2 scope - user approved cross-encoder on fused top-20 for both systems 2026-10-18 - reason: CPU predict 3.5 s/term; OOF top-1 60.7/50.0 vs 60.2/50.3
- Override: decision D9 - user approved dropping BM25 from fusion 2026-10-17 - reason: M1 ablation, removing BM25 raised R@50 ICD 83.1->89.2, SNOMED 82.3->84.6
- Override: M2 scope - user approved 3 folds, 2 epochs, cross-encoder scores fused top-30 + ICD siblings 2026-10-17 - reason: RTX 3050 GPU time before 18 Oct deadline

## Figures
- Figure 01_missingness.png: No column has missing values.
- Figure 02_target_balance.png: code=185389009 is 0.3% of 9,646 rows, so a majority-class dummy scores 0.3%; the minority share is 0.0%.
- Figure 02_distributions.png: Histograms of 2 numeric features; 2 are strongly skewed (term_words, chunk_chars), which matters for scaling and outliers.
- Figure 02_correlations.png: Strongest correlation with code: chunk_chars (0.04); anything above 0.95 is a leakage suspect.
- Figure 02_code_long_tail.png: Gold codes follow a long tail: 466 ICD and 2964 SNOMED codes appear exactly once. A classifier over seen codes cannot reach the brief's bar, so retrieval over the whole dictionary is required.
- Figure 02_tags_and_chapters.png: SNOMED gold codes span many tags (top: finding 1895, procedure 1828), so the tag is a ranking feature, not a filter. ICD is dominated by chapter R (690 rows): symptoms coded as symptoms.
- Figure 02_term_length.png: Terms are short (median 2 words), so abbreviations and spelling noise carry most of the difficulty; chunk context is the main extra signal.
- Figure 04_recall_at_k.png: Fused retrieval puts the gold code in the top 80 for 88.9% of ICD and 86.0% of SNOMED rows, but first only 39.0% / 35.1%. The candidates are there; the reranker (M2-M3) must pick the right one, and these recall values are its ceiling.
- Figure 12_stage_accuracy.png: Each stage adds accuracy: ICD 40.2 -> 45.3 -> 60.2%, SNOMED 33.5 -> 44.4 -> 50.3%. The dashed line is the most any reranker could reach with these candidates, so the remaining gap splits into ranking errors (below the line) and retrieval misses (above it).
- Figure 12_feature_importance.png: Cross-encoder features dominate both rankers (top: ce_rank / ce_rank); retrieval ranks and code metadata (is_leaf, chapter, semantic tag, training frequency) come next. The qualifier features carry little gain, so specificity errors are not yet being fixed by them.
- Figure 12_coverage_curve.png: Confidence ranks rows well: the most confident 25% are right 90% (ICD) / 85% (SNOMED) of the time versus 61% / 50% overall, but 98% precision is only reached on 9.5% / 0.3% of rows, so nearly every row carries the low-confidence flag.
- Figure 13_error_tags.png: Largest error sources: ICD 'under-specific' and SNOMED 'not retrieved'. Retrieval misses (1017 SNOMED rows) and abbreviations/unseen codes dominate SNOMED; ICD errors are mostly specificity (under/over-specific) and abbreviations, which better qualifier handling or abbreviation expansion could address.
