# Recovery log (18 Oct 2026, IST)

The original laptop (RTX 3050 6 GB) and its repo, weights and indexes were lost. This repo was rebuilt on a new
laptop (RTX 3070 Ti Laptop 8 GB, 16 logical cores, Windows 11, Python 3.12.10).

## R0 source choice
- Source: `termmap_restore_bundle.tar.gz` (v1 `src/termmap/`, 26 modules, coupling crash fix + regression test).
- Kaggle `termmap-k2` (newer source snapshot) and `snomed_ce_outputs.zip`: **not checked**: no `kaggle.json`
  on this laptop at recovery time. The bundle is the v1 predict path that all reported numbers come from, so
  it was used as-is. v2 flags described in `AUDIT.md` are not in this source.
- Dictionary files are byte-identical to the v1 build: the SHA-256 stamps in the v1 index manifest
  (`artifacts/index/manifest_v1.json`) equal the rebuilt ones (icd 60a91dffd515b77c, snomed 7cd4ea3c94bcdeff).
- Lost and recreated: `pyproject.toml` (CLI entry point), `.gitignore`, `.env.example`, README, tests beyond the
  coupling regression.

## R2 rebuild
- The laptop has a CUDA GPU, so per plan everything was rebuilt locally; Kaggle was not used tonight.
- Cross-validation was NOT recomputed: the surviving out-of-fold candidates and cross-encoder scores
  (`artifacts/cv/candidates*_*.parquet`) are reused. Only final artifacts were rebuilt with
  `scripts/rebuild_final.py` (same functions and seed as `termmap train`). Planned: indexes, final cross-encoders on all
  rows (2 epochs = CV best epoch in every fold), memory/abbreviations/statistics, LambdaRank boosters
  (refit, which also re-measures LambdaRank out-of-fold top-1), co-occurrence pairs. See "R2 outcome" for what ran.
- `hierarchy.json`, `coupling.json`, `confidence.json` are the v1 files (fitted on out-of-fold scores).

## R2 outcome (what actually happened)
- Index build: ICD 74 s; SNOMED 1,808 s (GPU at 88 °C, 55 W cap; v1 on an RTX 3050 took 222 s).
- ICD cross-encoder final fit: epoch 1 took 2,185 s (loss 0.2235). Stopped at 20:33 during epoch 2: using it would
  have needed an ICD ranker refit plus two more full predict runs (about 25 min each) before 21:30. No final
  cross-encoder exists in this build.
- Fallback adopted (planned in the brief): LambdaRank without cross-encoder features for both systems
  (`scripts/rebuild_final.py ltr_noce`). Out-of-fold ICD 55.5 (55.9 with hierarchy), SNOMED 44.5. Identical to v1
  (55.5 / 44.5) with the same best iterations per fold.
- Memory/abbreviations (442)/statistics rebuilt; co-occurrence pairs 2,122.

## R3
- `predictions.csv`: 7,522 rows, closed-world check passed. Load 400 s, predict 995 s (132 ms/term batch, GPU for
  SapBERT queries, 15 GB RAM laptop). Tagged `v1-fallback`.
