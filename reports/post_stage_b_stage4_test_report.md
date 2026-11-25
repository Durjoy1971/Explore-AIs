# Post-Stage B — Stage 4 Regression & Verification Test Report

**Date:** 2026-11-24  
**Environment:** Local Linux CPU (`Python 3.12.3`, `numpy 2.5.5`, `scipy 1.18.1`, `pandas 3.1.1`)  
**Scope:** Mocked local unit/integration tests and static integrity checks only. Zero model training executed.

---

## 1. Exact Test Commands & Observed Results

| # | Check Scope | Exact Command | Outcome | Key Observed Result |
|---:|---|---|---|---:|
| **1** | **Full `unittest` suite (`63` retained + `18` prior Stage 4 + `5` remediation)** | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 19 .venv/bin/python -m unittest discover -s tests -v` | **Exit `0` (`OK`)** | **`Ran 86 tests in 109.188s`** (`0` failures, `0` errors, `0` skipped) |
| **2** | **Core dataset/masking script suite** | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 19 .venv/bin/python tests/test_pipeline.py` | **Exit `0` (`OK`)** | `All tests passed!` |
| **3** | **Frozen `v1` bundle verification** | `OMP_NUM_THREADS=1 nice -n 19 .venv/bin/python scripts/prepare_post_stage_b_bundle.py --verify-only --bundle-dir data/processed/post_stage_b_bundle` | **Exit `0` (`OK`)** | `13/13` protected inputs; manifest `2e2e76e2...`; `15` cohort pairs checked |
| **4** | **Frozen `v2` bundle verification** | `OMP_NUM_THREADS=1 nice -n 19 .venv/bin/python scripts/prepare_post_stage_b_bundle.py --verify-only --bundle-dir data/processed/post_stage_b_bundle_v2` | **Exit `0` (`OK`)** | `13/13` protected inputs; manifest `308d1c88...`; `0` normalization mismatches |
| **5** | **Protected `282`-file + mask verification** | `OMP_NUM_THREADS=1 nice -n 19 .venv/bin/python scripts/verify_stage4_integrity.py --verify-only --skip-tfidf` | **Exit `0` (`OK`)** | `282/282` protected files (`240` scientific + `42` metadata); `4/4` frozen test masks matched |
| **6** | **Isolated Portable Colab Kit CLI export & independent verification** | `prepare_stage4_colab_kit.py --kit-dir $TMP_KIT` + `(cd /tmp && PYTHONPATH=$TMP_KIT/src python -c "verify_portable_colab_kit($TMP_KIT)")` | **Exit `0` (`OK`)** | `post_stage_b_stage4_kit_v2`, `33` runtime files, `8/8` `v2` cross-checked, `verifies_full_repository_282=false` |
| **7** | **Statistical artifact immutability** | `sha256sum reports/stage4_statistical_validation_results.json` | **Unchanged** | `bde8085b77066cbf449a1b17d0a1afa77f2dcbcb4ea1a82f7d0dde68f6fe504f` |
| **8** | **Stray CPU process check** | `pgrep -af python \|\| echo NO_STRAY_PYTHON` | **Clean** | Zero stray training/simulation processes |

---

## 2. Stage 4 Regression Test Coverage (`18` New Mocked Tests)

- [test_stage4_option_a_and_cohorts.py](file:///home/bs01415/Documents/Programming/ExploringAIs/tests/test_stage4_option_a_and_cohorts.py) (`4` tests):
  - Option A `160`-unit factorial config & rejection of `brand_masked` / `brand_stripped` and non-standard factors.
  - Exact `A/B/C/D` cohort membership, B class weights, `0` synthetic validation rows, exact optimizer step budgets (`300 / 300 / 333 / 363`), `gradient_accumulation_steps=2` step-budget invariance, and `0/280` representation collisions.
  - Exact reproducibility and SHA256 parity across all 4 frozen test masks (`1687`, `1698`, `1693`, `1675`).
  - `t4_benchmark` config `5`-unit schedule & fail-closed rejection of `run_heldout_eval=True`.
- [test_stage4_ensemble_and_stats.py](file:///home/bs01415/Documents/Programming/ExploringAIs/tests/test_stage4_ensemble_and_stats.py) (`7` tests):
  - Incomplete fold / revision / config-hash / row-order / unnormalized-probability rejection before ensemble aggregation.
  - Validation calibration against `265` Normal rows (`<= 2` FPs) & complete split export schema verification.
  - Analytical `(TP, E)` Macro-F1 gradient vs. finite differences (`eps = 1e-6`, `rtol < 1e-8`).
  - Exact parity between literal `B_ci = 2,000, B_pval = 50,000` bootstrap and short-circuit check.
  - Multiplicity separation (`Holm-12`, separate `Holm-24`, exploratory `C vs D` Holm-4), fail-closed default `CONFIRMATORY_DECLARATIONS_ENABLED == False`, and degenerate/zero-disagreement/missing-class safety.
- [test_stage4_checkpointing_and_kit.py](file:///home/bs01415/Documents/Programming/ExploringAIs/tests/test_stage4_checkpointing_and_kit.py) (`7` tests):
  - Four staging states (`VALID_RECOVERABLE`, `INCOMPLETE_UPLOAD`, `CORRUPT_OR_UNVERIFIABLE`, `ALREADY_SUPERSEDED`), interrupted upload/event/pointer recovery, Phase 1–3 ordering, and post-`UNIT_COMMITTED` pruning under `keep_model_weights="none"`.
  - Corrupted manifest & stale `config_hash` fail-closed `RecoveryError` preserving all evidence untouched.
  - Single-writer lease enforcement, `282`-file baseline & `240`-file portable check, Stage 4 notebook generator parity + double authorization gates (`AUTHORIZE_T4_BENCHMARK`, `AUTHORIZE_FULL_TRAINING`), end-to-end mocked `retraining_registry` + `step_budget` persistence, and Stage 4 Colab kit export/verification.

- [test_stage4_remediation.py](file:///home/bs01415/Documents/Programming/ExploringAIs/tests/test_stage4_remediation.py) (`5` tests):
  - Isolated portable Colab kit export & independent verification (from temp dir outside repo, rejecting missing/corrupt/unexpected/incompatible artifacts).
  - Mocked Colab dependency preflight (`torch >= 2.6`, `transformers >= 4.57.1,<5`, `normalizer` pinned commit, `torch==<ver>` constraints + restart instruction) & synchronized notebook Cell 4/5 verification.
  - Zero `fcntl`/`_run_lock` verification + all 7 single-writer lease / `SIGKILL` subprocess / stale takeover / conflict / corrupt release / unit-preserving recovery tests.
  - Cross-attempt `best_model_checkpoint` local relocation (leaving Drive staging immutable), missing best checkpoint fail-closed `RecoveryError`, retention limit preservation, partial `shutil.rmtree` failure handling, and post-`UNIT_COMMITTED` pruning.
  - Statistical governance bypass guards (`compare_models_cluster_bootstrap`, `holm_reject_family`, `evaluate_stage4_hypothesis_families`, `build_ensemble_evaluation_summary`), standalone `exports/retraining_metadata.json` parity, Markdown/CSV/JSON status + step-budget exports, and `7,114`-row dev-only T4 benchmark fixture validity.

---

## 3. Checks **`NOT VERIFIED locally`** (Colab Benchmark Required)

Local `.venv` lacks `torch`, `transformers`, `datasets`, `accelerate`, `sentencepiece`, `normalizer`, CUDA GPU, and mounted Google Drive FUSE. Therefore the following items are **NOT VERIFIED locally** and remain gated on the separately authorized 5-unit T4 Colab benchmark:

1. End-to-end `hf_train_and_predict_fold` training & `AutoTokenizer` / `normalizer` compatibility across all 4 backbones.
2. Allowed-view tokenizer-level (`input_ids`) collision checks on the 32 Stage 3 candidate pairs and `raw`/`canon_masked`.
3. Empirical NVIDIA T4 peak GPU VRAM (`xlm-roberta-base` and `muril` headroom at `bs=32` vs. `bs=16 + accum=2`).
4. Measured wall-clock breakdown, Google Drive FUSE readback latency, `O_EXCL`/`flock` behavior, Drive Trash 30-day quota retention, and actual Colab compute-unit (`CU/hr`) burn rate.
