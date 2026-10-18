# TermMap submission status (18 Oct 2026, IST)

## §1 Inventory (15:19)
- [x] Git repo: yes, branch `master`, last commit d5c2ea2. **No remote configured**: a private remote is needed for S8 (needs the user).
- [x] Root files: README.md, .env.example, .gitignore, requirements.txt present. No LICENSE file (licence notes go in README).
- [x] v1 final model intact in `artifacts/model/` (ce_ICD10CM, ce_SNOMEDCT_US, ltr_*.joblib, *_memory.joblib, *_train_stats.parquet, abbreviations.json, hierarchy.json, coupling.json, confidence.json, cooc_pairs.parquet); index in `artifacts/index/`; backup in `artifacts/model_backup_m5/`.
- [x] Evaluation input: no organiser test file in the repo. Built `data/eval_input.csv` = unique (entity, chunk_text) of the training CSV, labels removed: 7,522 rows (9,646 training rows).
- [x] Hardware: RTX 3050 6 GB laptop GPU, 16 cores, 14 GB RAM (heavy jobs must run one at a time; two at once were OOM-killed at 14:35).
- [x] Kaggle: K1 (Qwen3-8B rewrite) re-pushed after a GPU OOM at batch 32, QUEUED; K2 (SapBERT fine-tune) QUEUED. Both out of scope tonight (§2b gate 4); no results.
- [x] v2 state: C1 ICD synonyms index built (ICD R@10/30/80 75.8/86.4/91.0 -> 80.3/89.3/92.75, out-of-fold, `artifacts_v2/cv/c1_retrieval_*.json`). C4 MedCPT `dense2` index built 15:06-15:14. `c5_tune` (dense2 weight tuning) was stopped at 15:16 before finishing; to be rerun for the Track V pool gate.

## Track S
- [x] S1 `git ls-files`: no training text, chunk text or dictionary tracked (tests/fixtures/sample_input.csv is synthetic; its only overlap with training is two empty chunk strings).
- [x] S1 secrets scan of tree and full history: no keys/tokens, no personal paths or emails.
- [x] S1 `.env.example` lists every variable the code reads (TERMMAP_USE_GPU, TERMMAP_THREADS; HF_TOKEN/HF_HOME optional).
- [x] S1 requirements pinned (Python 3.11.15); added missing `simple-icd-10-cm==1.5.0`. Clean-venv install: not tested.
- [x] S2 15:40 **v1 predict crashed on the full input** (KeyError in the coupling debug column: the ICD hierarchy pick can sit outside the primed top-5; latent since the first commit, never hit by the fixtures). Fixed in `CodeEmbeddings.cosine` (embeds missing codes on demand; cache only, codes unchanged) + regression test. Rerun started 15:42.
- [ ] S3 determinism (2 runs, hashes), format checks, latency
- [ ] S4 validation.md, S5 README, S6 PDF, S7 demo script, S8 freeze/tag/push

## Track F (15:22-15:30) - decision: NOT adopted
- [x] F0 baseline reproduced: ICD 60.68, SNOMED 49.97 (v1 ranker, out-of-fold).
- [x] F2 string, F3 term, F5 structure, F4 pool (CE share, category/tag mass) groups: all within noise (best SNOMED all-groups +0.29, CI [-0.16, 0.76]; ICD all-groups -0.09).
- [x] Objective/regularisation chosen on the inner split: ICD 60.63, SNOMED 50.30 -> min +0.33 < +1.0. Not adopted. `reports/ranker_features.md`.
- [ ] F1 raw channel scores, F4 near-twin, F6, seed bagging: not run (time-box).
## Track S docs
- [x] S4 `reports/validation.md` submission section (ceilings verified: 76.4% SNOMED pair agreement over 101 rows; near-twin 37.4% / 26.9% at char-cosine >= 0.5; 85% / 74% quoted from the brief, not re-verified).
- [x] S5 README: pipeline in 6 lines, ceilings, selective accuracy, data handling + Kaggle disclosure, limitations, experiments. Cost per term pending S3.
