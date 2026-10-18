# TermMap audit

Protocol for every number below: the v1 3-fold out-of-fold split (`fold` column of `artifacts/cv/rows.parquet`),
all rows (2,304 ICD / 7,340 SNOMED). v1 artifacts (`artifacts/`, `artifacts/model_backup_m5/`) and the v1 predict
path are untouched: every v2 change sits behind a flag whose default reproduces v1, and v2 runs with
`--config configs/v2.yaml` into `artifacts_v2/`.

## Phase A — flagged problems

| item | confirmed in | fix | evidence / measurement |
|---|---|---|---|
| A1 CE training length | `rerank/cross_encoder.py:train_pairs` capped at `epochs: 2`; final epochs = mode of CV best epochs; val accuracy still rising at the cap (ICD fold 0 37.8 → 45.9) | `epochs: 5`, `patience: 1` early stopping on the 10% validation groups, best checkpoint kept; final model trained for the median CV best epoch (`final_epochs: median`); histories (incl. final) saved to `artifacts*/cv/ce_*_history.json` | needs Phase C retrain (Kaggle, C6) |
| A2 CE loss | pointwise `BCEWithLogitsLoss` over shuffled pairs | `loss: listwise`: softmax cross-entropy (InfoNCE) over each term's group (1 positive + 15 negatives), whole groups per batch, masked padding for short groups | Morand et al. 2026, *Building Better Encoder-only Cross-Encoders*, arXiv 2603.03010: comparison-based objectives (listwise InfoNCE, pairwise MarginMSE) beat pointwise BCE on all 9 backbones; objective change ≈ one backbone size tier. Phase C retrain |
| A3 hard negatives | `build_training_pairs` used fused-retrieval order inside the CE set only | `negatives: ltr`: gold siblings (ICD, up to 5, always) then the v1 out-of-fold LambdaRank order over the whole pool, then fused order | same paper: negative quality matters as much as the loss. Phase C retrain |
| A4 CE scoring set | `in_ce` = fused top-20 + embedded; ICD siblings never scored | `ce_top: {ICD10CM: 30, SNOMEDCT_US: 40}`, `ce_siblings: true` (all siblings expanded from the fused top-20, ICD) via `build_dataset.ce_scope` | Phase C re-score; ICD "under-specific" was the top ICD error tag (182) |
| A5 ICD preferred title | `dictionaries.load_icd` took the first row per code | `icd_title: official`: the row matching the official CMS long description (simple-icd-10-cm 1.5.0, MIT), else the closest by rapidfuzz token_sort_ratio; other rows stay synonyms | title equals the gold display for **799 / 897** gold codes vs 561 before; 20,222 of 98,182 titles change. CE text and coupling embeddings pick it up in Phase C |
| A6 SNOMED tag parsing | any trailing "(…)" became the tag: 639 distinct "tags" incl. "acquired immunodeficiency syndrome", "left ventricle", "1", "AJCC" | `snomed_tag: official`: tag accepted only if it is an official semantic tag (SNOMED CT Editorial Guide, *Semantic Tag* table, docs.snomed.org) | 55 distinct tags; 1,893 concepts corrected; 5 left untagged. Using the tag-stripped FSN as title was tested and **rejected** (matches gold display 3,593 / 4,335 vs 4,049 for description order 1) |
| A7 qualifier lexicon | v1 words only; regex missed `3b`, `ii` after `ckd`; no contrast / s/p / h/o / r/o | `features.qualifiers_v2` (`qualifier_version: 2`): wo, w/o, w/, w, with/without/with-and-without contrast; s/p, h/o, r/o; bilat, b/l; lt/rt, l/r only before a body-site word; acute on chronic; stage/type/grade with letter suffix or roman numeral, `ckd 3a` → stage 3a; new classes contrast / history / certainty | LambdaRank CV on v1 CE scores: v2 qualifiers alone ICD 60.0 / SNOMED 49.9 (v1 60.7 / 50.0) — no gain until siblings get CE scores (A4) |
| A8 coupling grid | λ grid {0,.25,.5,1,2}; λ=2 at the edge | `coupling.lambdas: [0, 0.5, 1, 2, 4, 8]` | ICD-only coupling (nested): folds chose 4 / 8 / 2; paired ICD 61.1 → 62.3; all ICD rows 60.5 → **61.6** (v1 final 61.5). λ=8 chosen once: still the edge on one fold |
| A9 BM25 | dropped because equal-weight RRF diluted recall | weighted RRF (`fusion.rrf(..., weights)`, `retrieval.channel_weights`), weights grid-searched per system on out-of-fold recall@80 | out-of-fold R@10/30/80, ICD: equal weights with BM25 66.9/78.8/88.7, v1 (no BM25) 75.0/85.6/90.7, tuned **75.9/86.5/91.1** (BM25 weight 0, dense 1.5, memory 1.5); SNOMED: 68.4/78.2/85.7, 71.0/80.8/86.1, tuned **71.0/80.3/86.6** (BM25 0.75, dense 2.0, memory 1.5). Grid of 90 weightings per system, chosen on the same OOF rows (mild selection optimism) |
| A10 determinism | `set_seeds` (Python/NumPy/torch seeds, `use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, cuDNN deterministic, benchmark off) is called in every torch entry point: `cli.main`, `app/demo.py`, `kaggle/train_snomed_ce.py`; device was chosen by an env var | `predict.device: cuda` pinned in config (env `TERMMAP_USE_GPU=0` still forces CPU; missing CUDA falls back to CPU with a warning). GPU runs the encoders in fp16 and CPU in fp32, so GPU and CPU predictions can differ; byte-identical reruns hold per device | `tests/test_e2e.py` (CPU) passes |
| A11 leakage | traced every fold-dependent artifact | none needed for v1 | memory (`evaluate.memory_for` on training folds), abbreviations (`mine_abbreviations` on training folds), co-occurrence (inner and outer folds in `evaluate_m5`), training frequency and exact-memory (`FeatureBuilder.build(train_rows=other folds)`), CE and LambdaRank fold models, temperatures, λ and margin are all fitted without the held-out fold. Second-order effects, disclosed: stacked OOF features for a training fold come from models that saw the held-out fold; A3 negatives come from v1 OOF LambdaRank ranks (same effect); A9 weights are chosen on the same OOF rows that report recall. Phase C rules: LLM kNN examples and fine-tuned encoders are per fold |

## Phase B — line-by-line audit (every module and function)

Status key: **correct**; **partial** (incomplete); **wrong** (bug); **weak** (works, but a better standard method exists).
"v2 flag" = fixed behind a `configs/v2.yaml` switch, v1 default unchanged (v1 predictions verified byte-identical
after every Phase B change on a 40-row CPU run). Edge-case probe (CPU predict on: empty term, whitespace term,
"???", a 500-word term, embedded "E11.9", "ø fever", duplicate rows, 5,000-char chunk, the literal "NA", a
UTF-8-BOM CSV): 11/11 rows output, no nulls, all codes valid, empty-like terms flagged `fallback_default`.

### src/termmap/io.py
| function | what it does | status | issue | evidence | fix |
|---|---|---|---|---|---|
| load_train | read training CSV as strings, check columns | correct | — | `keep_default_na=False` keeps "NA"/"None" terms as text | — |
| pair_instances | long → one row per (entity, chunk) | correct | used by audit only | — | — |
| group_key | in-memory CV group: lowercase alnum chunk; empty chunk → singleton | correct | — | spacing variants collapse 6,414 → 5,281 (data audit) | — |
| load_inference_input | read eval CSV, `clinical_entity` alias, optional chunk | correct | BOM suspected | probe: UTF-8-BOM CSV reads correctly (pandas strips the BOM) | — |

### src/termmap/normalize.py
| function | what it does | status | issue | evidence | fix |
|---|---|---|---|---|---|
| level1 | NFKC, lowercase, strip `<br>`/headers/(hcc), tidy punctuation | weak | trailing "+"/"%" stripped ("edema 2+" → "edema 2") | regex `_EDGE` | documented; changing it alters every index (rebuild) — left for a later release |
| tokens | alnum tokens | correct | — | — | — |
| strip_semantic_tag | drop trailing "(tag)" | correct | — | — | — |
| SpellCorrector.__init__/correct_word/variant | unique closest dictionary word, d≤1 (≤2 for 8+ letters), none on ties | correct | measured no effect (A-ablation, M1) | SymSpell, github.com/wolfgarbe/SymSpell | — |
| mine_abbreviations | short form → expansion from training pairs | weak | one pair could fix an expansion ("abd" → "abdominal binder device") | data inspection | v2 flag `abbr_min_support: 2` for whole-display expansions; initials matches still need one; test |
| abbreviation_variant | expand known short forms | correct | — | — | — |
| embedded_codes | codes written in the term | weak | missed undotted ICD codes ("E119") | regex | v2 flag `undotted_codes`; test |
| encoder_text | first clause for >32-word terms | correct | — | probe: 500-word term handled | — |
| query_variants | orig/spell/abbr variants | correct | — | — | — |

### src/termmap/dictionaries.py
| function | what it does | status | issue | evidence | fix |
|---|---|---|---|---|---|
| file_hash / _cached | SHA-256 stamp; parquet cache keyed by hash | correct | — | — | — |
| _load_icd_raw / _load_snomed_raw | read sheet 0 as strings | correct | — | codes kept as strings (leading zeros) | — |
| Dictionary / __contains__ | container | correct | — | — | — |
| official_icd_titles | official CMS description row per code (A5) | correct | — | 799/897 gold match | Phase A |
| load_icd | ICD rows, ranges dropped, metadata | correct (v2 flag) | first-row title (A5) | — | Phase A |
| load_snomed | SNOMED rows, tag, title | correct (v2 flag) | junk tags (A6) | — | Phase A |

### src/termmap/retrieval/
| function | what it does | status | issue | evidence | fix |
|---|---|---|---|---|---|
| fusion.collapse / top_n | description hits → best score per code, ties by code | correct | — | — | — |
| fusion.best_rank_merge | merge variant rankings by best rank | correct | — | — | — |
| fusion.rrf | (weighted) RRF, ties by code | correct | — | Cormack, Clarke & Büttcher, SIGIR 2009, doi.org/10.1145/1571941.1572114 (k=60) | Phase A weights |
| chargram.CharGramIndex | char_wb 3-4-gram TF-IDF cosine | correct | — | scispaCy candidate generation, Neumann et al. 2019, arxiv.org/abs/1902.07669 | — |
| bm25.BM25Index | Lucene BM25 (k1 1.5, b 0.75) | correct | — | bm25s, Lù 2024, arxiv.org/abs/2407.03618 | — |
| dense.Encoder | SapBERT CLS, L2-normalized, fp16 on GPU | weak | 32-token cap truncates long ICD descriptions (7th-character qualifiers live at the end) | SapBERT card uses CLS and max_length 25 for short names (huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) | v2 flag `dense_max_len: 64` (re-embed in Phase C on Kaggle); the index stamp records it so stale embeddings are refused |
| dense.DenseIndex | FAISS IndexFlatIP exact search | correct | — | — | — |
| memory.MemoryIndex | char + dense over training terms/displays, internal RRF | correct | — | built per training fold (A11) | — |
| engine.description_table | (code, normalized text) rows, tags stripped | correct | — | — | — |
| engine.load_dictionaries | loaders with v2 modes | correct | — | — | — |
| engine._stamp | dictionary hashes + model | partial | did not record the dense length | — | records `dense_max_len` when non-default; test |
| engine.build_index | build all indexes | wrong (v2 only) | in `artifacts_v2/index` the files are symlinks to v1; writing would overwrite v1 | `ls -la artifacts_v2/index` | `_writable` unlinks a symlink before every write; test |
| engine.Indexes | load indexes; artifacts-only predict | correct | fixed in M7 review | test_e2e artifacts-only | — |
| engine._parent / icd_siblings | ICD parent and siblings | correct | — | — | — |
| engine.Retriever.embed/prime | cached SapBERT and dictionary channels | correct (FYI) | batching depends on the whole input, so float sums may differ in the last bit between a row predicted alone and in a batch; reruns of one file are identical | test_e2e | documented |
| engine.Retriever.channel_rankings | per-channel rankings, best rank over variants, empty term retrieves nothing | correct | — | probe | — |
| engine.Retriever.fuse | weighted RRF, top-K, sibling expansion, embedded codes | correct | — | — | — |

### src/termmap/build_dataset.py
| function | what it does | status | issue | evidence | fix |
|---|---|---|---|---|---|
| build_rows | rows, invalid codes dropped (L6.0 ×2), GroupKFold | correct | — | GroupKFold, scikit-learn docs (deterministic, no shuffle) | — |
| heldout_codes | 20% codes for long tail | correct | — | — | — |
| ce_scope | per-system CE scope (A4) | correct | — | — | — |
| candidate_frame | flatten candidates, in_ce | correct | — | — | — |
| oof_candidates | out-of-fold pools, memory/abbr per fold | correct | — | A11 | — |

### src/termmap/rerank/
| function | what it does | status | issue | evidence | fix |
|---|---|---|---|---|---|
| cross_encoder.context_window / query_text | ±100 chars with [ENT] marks | correct | — | — | — |
| CandidateText | preferred | synonyms | category / tag | correct | picks up A5/A6 metadata | — | — |
| CrossEncoder.__init__/save/_batch/score | MedCPT CE, [ENT] tokens, fp16 scoring | correct | — | MedCPT, Jin et al. 2023, arxiv.org/abs/2307.00589 | — |
| CrossEncoder.train_pairs | fine-tuning | weak → fixed (v2) | BCE, 2-epoch cap (A1/A2) | arXiv 2603.03010 | Phase A |
| icd_children_map | parent → codes | correct | — | — | — |
| build_training_pairs | positives + hard negatives | weak → fixed (v2) | fused-order negatives (A3) | arXiv 2603.03010 | Phase A |
| score_candidates / top1_accuracy | score in_ce set; top-1 with code tie-break | correct | — | test_m2 | — |
| features.qualifiers / qualifiers_v2 / qualifier_features | qualifier sets and mismatch features | weak → fixed (v2) | shorthand, stage suffixes (A7), negation unmodelled (16 training rows, 7 negated gold displays) | data check | A7 + negation class in v2; tests |
| features.FeatureBuilder (+_code_q, build, categorical) | LambdaRank features, fold statistics | correct | — | test_m3 | — |
| features.feature_columns | feature list | correct | — | — | — |
| ltr.ranker_params/_sorted/_dataset/fit/predict | LightGBM LambdaRank, deterministic | correct | — | LightGBM docs, lightgbm.readthedocs.io/en/latest/Parameters.html (`deterministic`, `force_row_wise`) | — |
| ltr.top1_from_scores / importances / split_groups | metrics, gains, group split | correct | — | — | — |
| train_ce._val_split / run | CE fold + final training, checkpoints | correct (v2 flags) | — | coverage-checked checkpoints | Phase A |
| train_ltr.oof_features / cv_ranker / run | OOF LambdaRank | correct | — | — | — |

### src/termmap/ (stages 4–5, orchestration)
| function | what it does | status | issue | evidence | fix |
|---|---|---|---|---|---|
| hierarchy.group_softmax / nll / fit_temperature | temperature scaling by grid search on NLL | correct | had no tests | Guo et al. 2017, arxiv.org/abs/1706.04599 | tests added |
| hierarchy.choose_icd / choose_argmax | category-then-code, code tie-break | correct | had no tests | — | tests added |
| coupling.CooccurrenceTable / CodeEmbeddings / couple | pair re-scoring | correct | had no tests | — | tests added |
| coupling.coupled_icd | ICD-only coupling decision | new | predict and evaluation duplicated this rule | grep | single helper used by both; mutation check: flipping its condition fails the suite |
| confidence.threshold_for_precision / coverage_curve / expected_calibration_error | confidence metrics | correct | had no tests | ECE as in Guo et al. 2017 | tests added |
| evaluate.memory_for / retrieval_records / summarize / run_retrieval_cv | retrieval CV | correct | — | — | — |
| evaluate._icd_decisions / evaluate_hierarchy | nested hierarchy evaluation | correct | — | — | — |
| evaluate._top / evaluate_m5 | nested coupling + confidence | correct | duplicated coupling rule | — | uses `coupled_icd` |
| evaluate.tune_rrf_weights | A9 grid search | weak | weights chosen on the reporting rows | — | disclosed (small grid of 90) |
| pipeline.model_dir / fit_final | final abbreviations, memory, statistics | correct | — | — | — |
| pipeline.Predictor (+_rank_system, _choose, predict_frame) | full predict path, output guard | correct | — | probe 11/11 | uses `coupled_icd` |
| pipeline.predict_file | the one command | correct | — | test_e2e | — |
| config.load_config / set_seeds | config + determinism | correct | `PYTHONHASHSEED` set at runtime only affects child processes | Python docs | comment corrected; reruns verified identical |
| train._save_m5 / train / retrieval_cv | orchestration | correct | — | reproduced identical numbers (M7) | — |
| cli.main | commands | correct | — | — | — |
| error_analysis._title / tag_errors | rule-based error tags | weak | heuristic rules, some debatable tags | — | disclosed in report |
| audit.pct / run | M0 data audit | correct | reads `subconcept` for descriptive statistics only (allowed for reports) | — | — |

### app/, kaggle/, tests/
| function | what it does | status | issue | evidence | fix |
|---|---|---|---|---|---|
| app/demo.py predictor / top5_table / show_system | Streamlit demo | correct | — | AppTest run, no exceptions (M7) | Phase D adds LLM option probabilities |
| kaggle/make_bundle.py | private-dataset bundle | partial (for v2) | SNOMED cross-encoder only | — | generalised per Phase C job (K1–K6) |
| kaggle/train_snomed_ce.py find | Kaggle job script | weak | Kaggle's preinstalled torch is not pinned (only transformers) | script | Phase C notebooks log torch/CUDA versions and pin transformers; numbers labelled with hardware |
| tests/test_m0..m3, test_e2e | unit + end-to-end determinism | correct | hierarchy, coupling, confidence untested | — | `tests/test_v2_phase_b.py` |
| tests/test_v2_phase_a | Phase A fixes | correct | — | — | — |
| tests/test_v2_phase_b | Phase B fixes + new coverage | new | — | 10 tests | — |

### Design choices checked against primary sources
| choice | source | verdict |
|---|---|---|
| RRF k = 60 | Cormack et al., SIGIR 2009 | standard; weights added (A9) |
| char 3-4-gram TF-IDF candidates | Neumann et al. 2019 (scispaCy) | standard for biomedical entity linking candidate generation |
| SapBERT CLS embeddings | Liu et al., NAACL 2021, aclanthology.org/2021.naacl-main.334; model card | matches the recipe; fine-tuning on in-domain pairs (C3) is the standard upgrade |
| cross-encoder reranking with listwise loss + hard negatives | Morand et al. 2026, arXiv 2603.03010 | adopted (A2/A3) |
| LambdaRank stacker | LightGBM docs | standard; deterministic mode used |
| temperature scaling + ECE | Guo et al., ICML 2017 | standard |
| GroupKFold by chunk | scikit-learn docs | correct guard against same-chunk leakage |
