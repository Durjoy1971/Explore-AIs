# Post-Stage B — Stage 4 Engineering Implementation Report

**Date:** 2026-11-24  
**Repository / Branch / Commit:** `ExploringAIs` · `main` · `4d341183cb7a6fce0ed773ff33da2a3b757d6f94`  
**Scope:** Authorized Stage 4 Category-C engineering implementation only. Zero model training, zero Colab execution, zero Google Drive writes/deletions, zero modifications to frozen datasets/bundles (`282/282` Category A untouched). Full 160-unit training remains strictly **`NO-GO`**.

---

## 1. Option A Experimental Grid & Representation Controls

Implemented in [post_stage_b_runner.py](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py) and frozen in [post_stage_b_stage4_option_a_config.json](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/configs/post_stage_b_stage4_option_a_config.json):

- **Exact `160`-Unit Grid (`confirmatory_mode: true`):**
  - Experiments: `("A", "B", "C", "D")`
  - Backbones (`4`): `csebuetnlp/banglabert`, `google/muril-base-cased`, `xlm-roberta-base`, `distilbert-base-multilingual-cased`
  - Allowed text views (`2`): `canon_masked` (primary), `raw` (reference)
  - Folds (`5`): `(0, 1, 2, 3, 4)`
  - Schedule: `epochs = 3`, `batch_size = 32`, `gradient_accumulation_steps = 1`, `eval_batch_size = 64`, `seed = 42`
  - `4 × 4 × 2 × 5 = 160` training units.
- **Confirmatory Guard:** `PostStageBConfig.__post_init__` explicitly rejects `brand_masked`, `brand_stripped`, partial grids, or altered seeds/epochs whenever `confirmatory_mode=True`.
- **Verified Cohort Memberships (via `build_experiment_fold_data`):**
  - **A (`exp_a_unbalanced_real`):** `3,999` real rows (`3,199`/`3,200` train + `800`/`799` val), unweighted loss.
  - **B (`exp_b_original_only_balanced`):** Identical ordered real training IDs to Arm A per fold, fold-specific inverse-frequency `class_weight` training loss only.
  - **C (`exp_c_clean_stage_b`):** `4,344` pool (`3,199`/`3,200` real train + `345` Batches 1–7 Promotional added strictly to training folds; `0` synthetic validation rows).
  - **D (`exp_d_full_stage_b`):** `4,644` pool (`3,199`/`3,200` real train + `300` Pilot + `345` Batches 1–7 added strictly to training folds; `0` synthetic validation rows).
- **Equal-Epoch Optimizer-Step Budget Disclosure (Binding Requirement #1):**
  `compute_optimizer_step_budget(...)` computes and persists per-unit step budgets into `split_meta`, unit `metrics.json`, standalone `exports/retraining_metadata.json`, `aggregated_results.csv`/`.json`, and the generated `exports/post_stage_b_report.md` table + disclosure section (`effective_batch_size = batch_size × gradient_accumulation_steps = 32`):

| Arm | Train Rows (Folds 0–3 / Fold 4) | Micro-Batches / Epoch | Optimizer Steps / Epoch | Total Optimizer Steps (3 Epochs) | Delta vs. Arm A |
|---|---:|---:|---:|---:|---:|
| **A** (Real Unbalanced) | `3,199` / `3,200` | `100` / `100` | `100` | `300` | Baseline (`1.0000×`) |
| **B** (Real Class-Weighted) | `3,199` / `3,200` | `100` / `100` | `100` | `300` | `+0` (`1.0000×`) |
| **C** (Real + B1–B7) | `3,544` / `3,545` | `111` / `111` | `111` | `333` | `+33` (`1.1100×`, `+11.0%`) |
| **D** (Real + Pilot + B1–B7) | `3,844` / `3,845` | `121` / `121` | `121` | `363` | `+63` (`1.2100×`, `+21.0%`) |

*(Total gradient updates across 160 Option A units at `bs=32, accum=1`: `40×300 + 40×300 + 40×333 + 40×363 = 51,840` steps.)*

---

## 2. Five-Fold Ensemble Evaluation & Frozen Threshold Calibration

Implemented in [post_stage_b_ensemble.py](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_ensemble.py) and integrated into `post_stage_b_runner._aggregate_verified`:

- **Pre-Aggregation Fold Validation (`validate_fold_predictions_for_ensemble`):** Blocks incomplete ensembles (`set(folds) != {0,1,2,3,4}`), misaligned `row_id`/`label_idx`/`family_id` vectors, non-finite or unnormalized probabilities, or mismatched `model_revision`, `tokenizer_revision`, `config_hash`, or `dataset_manifest_sha256`.
- **Probability-Averaged Ensembling & Calibration:** Computes `float64` arithmetic mean across all 5 fold probability matrices (`mean /= row_sum`) and calibrates the Smishing threshold strictly on `smishdetect_validation` (`265` Normal rows $\Rightarrow$ integer budget $\le 2$ Normal false positives under `max_fpr = 0.01`, matching `<= 0.7547%`).
- **Frozen Transfer & Exports:** Applies the exact frozen `float64` validation threshold unchanged to `smishdetect_test` cohort masks and `banglaphish_ood` (overall + `SMS` + `other`), exporting: `threshold`, `calibration_achieved_fpr_pct`, `achieved_fpr_pct`, `normal_false_positives`, `normal_denominator`, `promotional_false_positives_at_threshold`, `promotional_denominator`, `promotional_fpr_at_threshold_pct`, `smishing_recall_at_frozen_threshold_pct`, `smishing_recall_argmax_pct`, `macro_f1_pct` (with 95% family-clustered CI), `accuracy_pct`, `per_class`, `roc_auc_smishing_vs_rest`, `pr_auc_smishing_vs_rest`, `confusion_matrix_argmax`, `constraint_met`, and an explicit caveat that satisfying development `<= 1%` Normal FPR never guarantees `<= 1%` FPR on test or OOD splits. Test and OOD scores remain strictly excluded from model selection.

---

## 3. Reproducible Test Evaluation Cohort Masks

Implemented in [post_stage_b_cohorts.py](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_cohorts.py) and frozen to [stage4_frozen_test_masks.json](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/stage4_frozen_test_masks.json) prior to any model training or test inspection:

| Cohort Mask Key | Role | Rows | Families | Normal | Promotional | Smishing | Excluded Count |
|---|---|---:|---:|---:|---:|---:|---:|
| `primary_family_decontam_1687` | Primary confirmatory | `1,687` | `1,299` | `581` | `384` | `722` | `11` rows (`5` families, all Smishing) |
| `supplementary_full_1698` | Supplementary benchmark | `1,698` | `1,304` | `581` | `384` | `733` | `0` |
| `sensitivity_row_only_1693` | Row-only sensitivity | `1,693` | `1,302` | `581` | `384` | `728` | `5` ambiguous rows |
| `sensitivity_template_stress_1675` | Template-stress sensitivity | `1,675` | `1,292` | `580` | `384` | `711` | `23` rows (`12` families) |

All four masks apply identically across every compared model and verify independently against the 12,466-row char-ngram (`3–5`) TF-IDF `>= 0.85` screen. Threshold calibration is verified independent of test exclusions because `smishdetect_validation` calibration uses its `265` Normal rows (`validation_normal_count == 265`). **Documented Limitation:** post-hoc test evaluation masking cannot undo any indirect influence of train–CV or train–validation near-duplicates on epoch checkpoint selection during training.

---

## 4. Colab Local-SSD Checkpointing, Immutable Drive Staging & 3-Phase Recovery

Implemented in [post_stage_b_checkpointing.py](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_checkpointing.py) and integrated into `post_stage_b_runner.py`:

- **Local SSD Training + UUID Staging:** Epoch checkpoints train on Colab local SSD (`cfg.local_checkpoint_root = "/content/tmp_stage4_ckpt"`) and sync to immutable UUID-disambiguated staging directories (`staging/<uid>/attempt_<n>_<uuid8>/checkpoint-<step>/`) with `stage4_checkpoint_manifest.json` per-file SHA256 digests, full readback verification, append-only commit events (`events/event_*.json` with embedded `event_sha256`), and versioned pointer files (`pointers/<uid>/ptr_*.json` with embedded `pointer_sha256`). Complete removal of `fcntl.flock` / `.runner.lock` from `post_stage_b_runner.py`; zero reliance on Google Drive FUSE `os.replace`, `fsync`, `O_EXCL`, or `flock` guarantees (and no distributed mutual-exclusion claim is made for Drive FUSE).
- **Operational Single-Writer Lease & Stale-Takeover Protocol:** `register_single_writer_lease` / `release_single_writer_lease` / `inspect_writer_leases` / `verify_writer_lease_still_sole_active` write immutable lease, release, and owner-confirmed takeover records (`leases/lease_*.json`, `release_*.json`, `takeover_*.json` with `lease_sha256`). Unresolved active leases or interrupted/corrupt lease records fail closed by default (lease age alone is never treated as proof of termination); recovery after abrupt `SIGKILL`/preemption requires explicit `allow_stale_takeover=True` with non-empty `takeover_reason`, preserving old leases and corrupt records for audit.
- **Four Staging States, Cross-Attempt Best-Checkpoint Relocation & Three Ordered Recovery Phases (Epoch-Level Only):**
  1. **Phase 1 (`scan_unit_staging`):** Read-only classification into `VALID_RECOVERABLE`, `INCOMPLETE_UPLOAD`, `CORRUPT_OR_UNVERIFIABLE`, or `ALREADY_SUPERSEDED` (resilient to interrupted tail event/pointer JSONs; honors `checkpoint_retention_limit` while always preserving at least the newest valid candidate plus any distinct pinned `best_model_checkpoint`; flags missing required best checkpoints or stale `config_hash` / `dataset_manifest_sha256`).
  2. **Phase 2 (`recover_latest_valid_checkpoint`):** Restores the newest `VALID_RECOVERABLE` checkpoint (and any distinct pinned `best_model_checkpoint`) onto local SSD with SHA256/byte readback checks, relocating `best_model_checkpoint` **strictly inside the restored local copy** of `trainer_state.json` so `load_best_model_at_end=True` works without modifying immutable Drive staging. Fails closed (`RecoveryError`) without deleting evidence if required best checkpoints are missing or if `CORRUPT_OR_UNVERIFIABLE` exists without a valid fallback. Granularity is strictly **epoch-level** (`recovery_granularity = "epoch_level"`).
  3. **Phase 3 (`prune_superseded_staging`):** Executes inter-epoch in `Stage4CheckpointCallback.on_save`, post-restore, and post-`UNIT_COMMITTED`; prunes only verified `ALREADY_SUPERSEDED`, capturing per-directory `OSError`s in `prune_errors` without deleting `CORRUPT_OR_UNVERIFIABLE`, `INCOMPLETE_UPLOAD`, or active `VALID_RECOVERABLE` checkpoints.

---

## 5. Model Artifact Retention Policy & Deterministic Retraining Plan

Configured via `PostStageBConfig.keep_model_weights` (`"none"` default, `"best_fold0"`, or `"all"`):

- **Implications of `keep_model_weights="none"`:** Saves all per-fold predictions (`preds.npz`), row IDs, labels, per-unit metrics (`metrics.json`), ensemble metrics, `config_hash`, `dataset_manifest_sha256`, standalone `exports/retraining_metadata.json`, and embedded `retraining_registry` inside `aggregated_results.json`, allowing full metric/cohort/statistical recomputation while pruning heavy multi-GB training checkpoints post-`UNIT_COMMITTED`. However, scoring brand-new unseen texts outside the saved cohorts later requires retraining from the logged deterministic recipe (`model_name`, `model_revision`, `tokenizer_revision`, `text_view`, `fold`, `seed`, optimizer hyperparameters, `class_weights`, and actual step budget).
- **Anti-Leakage Rule:** Weight retention is configured strictly before execution in `PostStageBConfig` and is **never** selected dynamically using held-out test or OOD performance. Historical artifacts and checkpoints are never deleted.

---

## 6. Versioned Stage 4 Colab Execution Kit (Prepared Only, Not Executed)

Prepared and verified via tests without executing any Colab notebook or writing to Google Drive:
- [post_stage_b_stage4_option_a_config.json](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/configs/post_stage_b_stage4_option_a_config.json) (160-unit confirmatory Option A)
- [post_stage_b_stage4_t4_benchmark_config.json](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json) (5-unit T4 calibration, `run_heldout_eval: false`)
- [post_stage_b_colab_deps.py](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_colab_deps.py) (version-aware Colab dependency preflight, preserving `torch==<ver>` + restart instruction + post-restart import checks)
- [generate_post_stage_b_stage4_notebook.py](file:///home/bs01415/Documents/Programming/ExploringAIs/scripts/generate_post_stage_b_stage4_notebook.py) $\rightarrow$ [post_stage_b_stage4_experiments.ipynb](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/post_stage_b_stage4_experiments.ipynb) (`AUTHORIZE_T4_BENCHMARK = False`, `AUTHORIZE_FULL_TRAINING = False`, `ALLOW_STALE_WRITER_TAKEOVER = False`)
- [prepare_stage4_colab_kit.py](file:///home/bs01415/Documents/Programming/ExploringAIs/scripts/prepare_stage4_colab_kit.py) (`post_stage_b_stage4_kit_v2`: self-contained 33-file runtime kit with `verify_portable_colab_kit` contract + `reports/{protected_baseline_282,stage4_frozen_test_masks,stage4_statistical_validation_results}.json`)
