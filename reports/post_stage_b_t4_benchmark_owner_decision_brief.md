# Post-Stage B — Post-Stage 4 Owner Decision Brief: Five-Unit Tesla T4 Benchmark

**Date:** 2026-11-24  
**Repository:** `/home/bs01415/Documents/Programming/ExploringAIs` (`HEAD`: `4d341183cb7a6fce0ed773ff33da2a3b757d6f94`)  
**Prepared Baseline Reference:** [`reports/post_stage_b_stage4_final_independent_acceptance_review.md`](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/post_stage_b_stage4_final_independent_acceptance_review.md)  
**Benchmark Config:** [`experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json) (SHA256 `03a5f10d433f657af82d1af401774fd40d9d36209d35e13a14bc934086416180`)  
**Benchmark Notebook:** [`experiments/colab/post_stage_b_stage4_experiments.ipynb`](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/post_stage_b_stage4_experiments.ipynb) (SHA256 `53b35f508f9a4c783ba951e409c0cf324b9992732f629b65ac06eb587f50a219`)  
**Task Scope Note:** Planning and audit brief only. Zero source files modified, zero Colab/Drive operations performed, zero tests/training executed.

---

## 1. Executive Summary

Stage 4 local engineering, CPU compliance (`CPUQuota=300%` + `taskset -c 0-2`, `<= 8.61%` host CPU across `90` tests), scientific governance, and 282-file protected integrity have received **`CONDITIONALLY ACCEPTED`** status in [`reports/post_stage_b_stage4_final_independent_acceptance_review.md`](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/post_stage_b_stage4_final_independent_acceptance_review.md). All remaining unverified conditions reside strictly in the live Google Colab T4 + Google Drive + HuggingFace Hub runtime.

The next proposed milestone is the **five-unit Tesla T4 calibration benchmark** (`post_stage_b_stage4_t4_benchmark_v1`), designed exclusively on Fold 0 development data and a dev-only length-matched inference fixture (**`0` held-out test or OOD rows scored**). Its sole purpose is to verify live HuggingFace tokenization (`canon_masked` + `normalizer`), Colab `pip`/CUDA dependency installation, Drive FUSE sync/Trash behavior, peak T4 VRAM across all four backbones, per-step latency (`WeightedTrainer` and unweighted `Trainer`), and Colab Compute Unit (`CU/hr`) burn rate before making any budget or scientific go/no-go decision on the 160-unit Option A grid.

**Recommended Owner Decisions:**
- **Permission A (Five-Unit T4 Benchmark Execution):** **`CONDITIONAL GO`** (pending preflight checklist & Permission B).
- **Permission B (Isolated Google Drive Writes to Fresh `MyDrive/bangla_smish_post_stage_b_stage4_v3/`):** **`CONDITIONAL GO`** (co-required when `USE_GOOGLE_DRIVE = True`).
- **Permission C (Additional Code Modifications Before Benchmark):** **`NO PRE-BENCHMARK EDITS REQUIRED (Option C0 recommended)`** (0 local blockers; optional enhancements listed separately as Option C1).
- **Permission D (Full 160-Unit Option A Training):** **`NO-GO`** (strictly unauthorized until empirical T4 measurements are returned and separately approved).

> [!NOTE]
> **Audit Transparency Erratum on Prior Acceptance Review:** Lines `77–79` of [`reports/post_stage_b_stage4_final_independent_acceptance_review.md`](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/post_stage_b_stage4_final_independent_acceptance_review.md#L77-L79) contained a prose transcription typo naming `csebuetnlp/banglishbert` and `sagorsarker/bangla-bert-base` next to the MuRIL/XLM-R/DistilBERT commit hashes. Direct inspection confirms that [`src/bangla_smish/post_stage_b_tokenizer_gate.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_tokenizer_gate.py#L18-L23), [`post_stage_b_stage4_option_a_config.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/configs/post_stage_b_stage4_option_a_config.json#L5-L16), and [`post_stage_b_stage4_t4_benchmark_config.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json#L5-L16) all consistently freeze the exact four Option A backbones listed in Section 2 below.

---

## 2. Exact Benchmark Scope

Verified directly from [`experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json#L1-L48), [`data/processed/post_stage_b_bundle_v2/dataset_manifest.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/post_stage_b_bundle_v2/dataset_manifest.json#L311-L469), and [`src/bangla_smish/post_stage_b_runner.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L467-L506):

### 2.1 Configuration Identity
- `run_name`: `"post_stage_b_stage4_t4_benchmark_v1"`
- `run_mode`: `"t4_benchmark"`
- `experiments`: `["B", "D"]`
- `text_views`: `["canon_masked"]` (1 view only)
- `folds`: `[0]` (Fold 0 only)
- `confirmatory_mode`: `false`
- `run_heldout_eval`: `false` (fail-closed against held-out test/OOD scoring)
- `benchmark_workload_fixture_eval`: `true`
- `run_adversarial_eval`: `true`
- `run_counterfactual_eval`: `true`
- `keep_model_weights`: `"none"` (predictions, IDs, labels, metrics, step budgets, and `retraining_metadata.json` retained; multi-GB weights pruned post-commit)
- `checkpoint_retention_limit`: `1`
- `local_checkpoint_root`: `"/content/tmp_stage4_ckpt"`

### 2.2 Exact Five Units & Model Architectures

| Unit # | Exact `unit_id` | Experiment Arm & Trainer | Model Identifier & Architecture | Pinned Tokenizer Revision (40-hex SHA) | Text View | Fold | Train Rows (Real / Synth) | Optimizer Steps (`3` Epochs) |
|---:|---|---|---|---|---|---:|---:|---:|
| **1** | `exp_D__csebuetnlp__banglabert__canon_masked__fold_0` | **D** (`exp_d_full_stage_b`, unweighted `Trainer`) | `csebuetnlp/banglabert` (ELECTRA-base discriminator, Bangla WordPiece + `normalizer`) | `9ce791f330578f50da6bc52b54205166fb5d1c8c` | `canon_masked` | `0` | `3,844` (`3,199` real + `300` Pilot + `345` B1–B7) | `121 × 3 = 363` |
| **2** | `exp_D__google__muril-base-cased__canon_masked__fold_0` | **D** (unweighted `Trainer`) | `google/muril-base-cased` (BERT-base cased multilingual Indian encoder, 197k vocab) | `afd9f36c7923d54e97903922ff1b260d091d202f` | `canon_masked` | `0` | `3,844` (`3,199` + `645` synth) | `121 × 3 = 363` |
| **3** | `exp_D__xlm-roberta-base__canon_masked__fold_0` | **D** (unweighted `Trainer`) | `xlm-roberta-base` (RoBERTa-base multilingual encoder, 250k SentencePiece vocab; largest backbone) | `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` | `canon_masked` | `0` | `3,844` (`3,199` + `645` synth) | `121 × 3 = 363` |
| **4** | `exp_D__distilbert-base-multilingual-cased__canon_masked__fold_0` | **D** (unweighted `Trainer`) | `distilbert-base-multilingual-cased` (6-layer distilled mBERT encoder, 119k vocab) | `45c032ab32cc946ad88a166f7cb282f58c753c2e` | `canon_masked` | `0` | `3,844` (`3,199` + `645` synth) | `121 × 3 = 363` |
| **5** | `exp_B__xlm-roberta-base__canon_masked__fold_0` | **B** (`exp_b_original_only_balanced`, `WeightedTrainer` with `class_weight`) | `xlm-roberta-base` (RoBERTa-base multilingual encoder, 250k vocab) | `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` | `canon_masked` | `0` | `3,199` (`3,199` real + `0` synth) | `100 × 3 = 300` |

- **Total Benchmark Optimizer Steps:** `(4 × 363) + 300 = 1,752` optimizer steps (`3.38%` of the `51,840` optimizer steps in the 160-unit Option A grid).
- **Note on Exact Arithmetic:** For Arm D (`3,844` rows), `ceil(3,844 / 32) = 121` micro-batches/epoch (120 full batches of 32 + 1 tail batch of 4). For Arm B (`3,199` rows), `ceil(3,199 / 32) = 100` micro-batches/epoch (99 full batches of 32 + 1 tail batch of 31).

### 2.3 Dataset, Fold & Evaluation Row Counts

- **Source Bundle:** `post_stage_b_v2` ([`data/processed/post_stage_b_bundle_v2`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/post_stage_b_bundle_v2/dataset_manifest.json)) — read-only, SHA256-verified before execution (`smoke_max_*_samples: null`).
- **Per-Unit Fold 0 Evaluation During Training (`Trainer` `eval_strategy = "epoch"`):**
  - `val_df` (Fold 0 CV validation): `800` rows (`302` Normal, `189` Promotional, `309` Smishing) × `3` epochs = `2,400` eval inferences/unit (`12,000` across 5 units).
- **Per-Unit Post-Training Inference (`10,205` texts/unit; `51,025` total across 5 units):**
  1. `val_probs` (`val_df` Fold 0): `800` rows (`text_canon_masked`).
  2. `val_adv_probs` (`val_df` Fold 0 adversarial): `800` rows (`text_adv_canon_masked`).
  3. `smishdetect_validation_probs` (dev validation set): `708` rows (`265` Normal, `178` Promotional, `265` Smishing).
  4. Dev-only length-matched workload fixture (`_build_t4_benchmark_workload_fixture`, timed and discarded): `7,114` synthetic-length dev texts (`1,698` single dev texts + `5,416` deterministically pair-concatenated dev texts matching `smishdetect_test` + `banglaphish_ood` counts/length profile; **`0` held-out test or OOD rows**).
  5. Fold 0 counterfactual probe (`counterfactual_smish_probe` filtered to Fold 0): `261` Fold-0 Smishing parents × `3` text variants (`orig`, `contact_swap`, `contact_brand_swap`) = `783` texts.
- **Held-Out Test & OOD Evaluation Count:** **`0` rows** (`eval_smishdetect_test` `1,698` rows and `eval_banglaphish_ood` `5,416` rows are neither scored nor loaded into `Trainer.predict`).

### 2.4 Epochs & Optimizer Hyperparameters
- `epochs`: `3`
- `batch_size`: `32`, `gradient_accumulation_steps`: `1` (effective batch size `32`), `eval_batch_size`: `64`
- `learning_rate`: `3e-5`, `weight_decay`: `0.01`, `warmup_ratio`: `0.1`, optimizer: HF `Trainer` default `AdamW` (`adamw_torch`)
- `max_length`: `128`, `seed`: `42` (`unit seed = 42 + fold 0 = 42`)
- `fp16`: `"auto"` (resolves on Tesla T4 Compute Capability 7.5 to `fp16=True, bf16=False`, requiring `scaler.pt` in checkpoints)
- Checkpoint / selection policy: `save_strategy="epoch"`, `eval_strategy="epoch"`, `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, `greater_is_better=False`

---

## 3. Remaining Technical Conditions (Blockers vs. Non-Blocking Observations)

The table below distinguishes genuine **runtime preflight gates** (which must execute and pass inside Colab before any training begins) from **non-blocking observations** and **future pre-Option-A considerations**:

| Condition / Item | Classification | Impact on 5-Unit T4 Benchmark | Required Owner / Runtime Action |
|---|---|---|---|
| **1. Live HuggingFace Tokenizer Revision Verification** | **Runtime Preflight Gate** *(Not a pre-authorization blocker)* | Cannot be verified offline; Colab Cell 5 calls `run_allowed_view_tokenizer_collision_gate` to download all 4 pinned revisions and test `input_ids` with `normalizer` before Cell 6 is unlocked. | Must pass cleanly in Notebook Cell 5. Also note: [`post_stage_b_runner.py:575`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L575) pins `revision=pinned_tok_rev` on `AutoTokenizer`, whereas `AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)` loads default `main` and logs `model.config._commit_hash` afterward (non-blocking for T4 calibration; decide before Option A whether to pin model weight commits too). |
| **2. `128` `canon_masked` Checks vs. `256` `raw + canon_masked` Checks** | **Non-Blocking for T4 Benchmark** *(Coverage gap for `raw` until Option A or Permission C)* | [`experiments/colab/post_stage_b_stage4_experiments.ipynb:173`](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/post_stage_b_stage4_experiments.ipynb#L173) passes `allowed_views=tuple(v for v in cfg.text_views if v in {'raw', 'canon_masked'})`. Because `t4_benchmark` uses `canon_masked` only, Cell 5 runs `4 × 1 × 32 = 128` checks (`100%` of T4 active views), leaving `raw` (`128` additional checks) untested live on HF Hub until either Permission C changes Cell 5 or `option_a_config` runs Cell 5. | Accept `128`-check scope for T4 (`Option C0`), OR authorize Permission `C1` to check all `256` pairs during Cell 5. |
| **3. `_commit_hash` Fallback Issue** | **Non-Blocking Observation** *(Low risk)* | In [`post_stage_b_tokenizer_gate.py:141-145`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_tokenizer_gate.py#L141-L145), `actual_rev` falls back to `expected_rev` if an HF tokenizer object omits `init_kwargs["_commit_hash"]` / `_commit_hash`. `AutoTokenizer.from_pretrained(model_name, revision=expected_rev)` (`L70`) still requests the exact commit from HF Hub. | Audit `tokenizer_collision_gate.json` + `retraining_metadata.json` after benchmark; optionally harden under Permission C before Option A. |
| **4. Non-Default Checkpoint Path Edge Case** | **Non-Blocking Observation** *(Unreachable in T4 & Option A)* | In [`post_stage_b_runner.py:1094-1098`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L1094-L1098), `_read_committed_unit` resolves `trainer_checkpoint` against `artifacts_dir / "checkpoints" / uid` rather than `local_checkpoint_root`, ONLY if `commit` contains `"trainer_checkpoint"`, which only occurs when `keep_model_weights != "none"` (`L1386-L1409`). Both `t4_benchmark` and `option_a` enforce `keep_model_weights = "none"`. | None required for T4 (`keep_model_weights = "none"`). Fix under Permission C only if `"best_fold0"` or `"all"` weight retention is later enabled alongside `local_checkpoint_root`. |
| **5. Google Drive FUSE Lease Limitations** | **Operational Constraint** *(Non-blocking with single-session rule)* | Drive FUSE lacks true distributed `flock`/`O_EXCL` across separate Colab VMs. Append-only leases block accidental multi-writer overlap and corrupt files, but cannot physically block an operator who manually puts a live session's `writer_id` into `EXPECTED_STALE_WRITER_IDS`. Drive FUSE sync/Trash behavior is empirically unverified until T4 benchmark. | Strictly enforce single active Colab runtime; never enable `ALLOW_STALE_WRITER_TAKEOVER` unless the prior runtime is confirmed terminated; record `drive_sync_seconds` & Trash behavior. |
| **6. Live Colab Dependency Install & CUDA PyTorch Preservation** | **Runtime Preflight Gate** *(Not a pre-authorization blocker)* | [`post_stage_b_colab_deps.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_colab_deps.py#L37-L210) pins `torch==<installed_version>` via constraints file to preserve Colab's CUDA wheel, installs compatible HuggingFace packages + pinned `normalizer`, and asserts `torch.cuda.is_available()` and callable `normalizer.normalize`. May require one runtime restart after pip install. | Execute Cell 4; restart runtime if prompted and re-run Cells 1–4 before Cell 5. |

**Distinction Summary:** Zero code defects block starting the 5-unit T4 benchmark today. Items **(1)** and **(6)** are the intended Colab runtime preflight gates executed automatically in Cells 4–5 before Cell 6 can train. Items **(2)**, **(3)**, and **(4)** are non-blocking observations, and Item **(5)** is an operational discipline + Drive measurement goal of the benchmark itself.

---

## 4. Resource Estimates

Strict honesty policy: no unmeasured GPU, timing, permanent Drive growth, or Compute Unit figures are fabricated below.

| Resource Category | Known / Measured Value | Unknown / Unverified Until T4 Benchmark | Measurement Telemetry Captured by Runner |
|---|---|---|---|
| **Target GPU** | NVIDIA Tesla T4 (`15.0 GB` VRAM, FP16 via `fp16="auto"`) | — | Cell 2 `nvidia-smi` + `environment_snapshot.json` |
| **Peak GPU VRAM** | Bounded by `batch_size=32, max_length=128, fp16=True` across ~110M–278M param backbones (highest risk: `xlm-roberta-base` Unit 3 & Unit 5 `WeightedTrainer`) | **`UNKNOWN (empirically unverified on T4)`** | `peak_cuda_allocated_bytes`, `peak_cuda_reserved_bytes` per unit in `metrics.json` |
| **Wall-Clock Execution Time** | Workload: `1,752` optimizer steps + `12,000` epoch-eval inferences + `51,025` post-train inferences | **`UNKNOWN (empirically unverified on T4)`** | `init_seconds`, `tokenization_seconds`, `optimizer_step_ms`, `eval_seconds`, `inference_seconds`, `local_checkpoint_seconds`, `drive_sync_seconds`, `train_seconds` |
| **Staged Portable Kit Size** | **`~15 MB` total** on local disk (`14 MB` `post_stage_b_bundle_v2` + ~`0.6 MB` `src/bangla_smish/*.py` + ~`0.16 MB` configs/notebook/reports/`pyproject.toml`, excl. `__pycache__`) | — | Verified locally via `du -ch` |
| **Permanent Drive Artifact Storage (`keep_model_weights="none"`)** | Prunes heavy checkpoints post-commit; retains `preds.npz`, JSON metadata/leases/events/pointers, logs, CSV/MD exports | **`UNKNOWN exact byte total (expected low tens of MBs, empirically unverified)`** | Measured via Drive quota / `du` after Cell 7 |
| **Transient Peak Local SSD (`/content`) & Drive Staging (+ Trash)** | Holds `checkpoint_retention_limit=1`–`2` active Trainer checkpoints (weights + FP32 AdamW states + scaler) mid-unit | **`UNKNOWN exact GB peak per backbone on Drive/SSD & Drive Trash retention`** | Monitored in `/content` & Drive quota during/after benchmark |
| **Colab Compute Unit (CU) Consumption** | Colab T4 hourly burn depends on Colab session rate × measured wall-clock time | **`UNKNOWN (empirically unverified until T4 benchmark)`** | Record Colab UI CU balance before Cell 1 & after Cell 7; used to compute $\hat{\text{CU}}_{160}^{\text{safe}}$ |

---

## 5. Google Drive Permissions & Execution Safety

### 5.1 Google Drive OAuth Scope & Exact Written Directories
- **OAuth Scope Reality:** `google.colab.drive.mount('/content/drive')` grants full read/write access to the owner's `MyDrive` (Colab cannot technically scope OAuth to a single folder). Containment relies strictly on notebook/runner configuration targeting an isolated brand-new workspace folder.
- **Exact Drive Directories Created / Written (and ONLY these):**
  - Fresh workspace root (staged manually beforehand): `/content/drive/MyDrive/bangla_smish_post_stage_b_stage4_v3/`
  - Benchmark artifact root (written during Colab run): `/content/drive/MyDrive/bangla_smish_post_stage_b_stage4_v3/artifacts/t4_benchmark/<ARTIFACT_RUN_ID>/` (recommended ID: `stage4-t4-bench-01`):
    - `leases/` (`lease_*.json`, `release_*.json`, `takeover_*.json`)
    - `run_state.json`
    - `logs/runner.log`
    - `staging/<uid>/attempt_<n>_<uuid8>/checkpoint-<step>/` *(transient epoch staging; pruned post-`UNIT_COMMITTED`)*
    - `events/` (`event_*.json`)
    - `pointers/<uid>/` (`ptr_*.json`)
    - `units/<uid>/attempts/<n>/` (`preds.npz`, `metrics.json`)
    - `exports/` (`tokenizer_collision_gate.json`, `environment_snapshot.json`, `active_config.json`, `aggregated_results.json`, `aggregated_results.csv`, `retraining_metadata.json`, `post_stage_b_report.md`, `history/<timestamp>/*`)
  - Ephemeral Colab local SSD (never synced to Drive): `/content/tmp_stage4_ckpt/post_stage_b_stage4_t4_benchmark_v1/<uid>/<attempt>/`.
  - **Zero edits** to any historical Stage 1–3, `v1`/`v2`, or unrelated personal Drive directories.

### 5.2 Writer Lease, Recovery, Resume, Integrity & Rollback
- **Writer Lease Behavior:** [`register_single_writer_lease`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_checkpointing.py#L247-L306) writes an immutable lease file before any unit runs and [`verify_writer_lease_still_sole_active`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_checkpointing.py) re-checks ownership before every unit (`finally` writes immutable release file). Any active or corrupt lease record fails closed unless `ALLOW_STALE_WRITER_TAKEOVER=True`, non-empty `STALE_TAKEOVER_REASON`, and exact matching `EXPECTED_STALE_WRITER_IDS` / `EXPECTED_CORRUPT_LEASE_FILES` are provided.
- **Checkpoint Recovery & Resume/Interruption:** Granularity is strictly **epoch-level** (`recovery_granularity = "epoch_level"`). Each saved epoch on local SSD receives an atomic SHA256 commit (`post_stage_b_checkpoint_commit.json`, including `scaler.pt` under FP16) -> UUID Drive upload -> readback SHA256 verification -> append-only commit event & pointer -> Phase 3 prune of verified `ALREADY_SUPERSEDED` epochs. On resume after preemption, Phases 1–2 scan Drive staging, restore the newest `VALID_RECOVERABLE` epoch (and any distinct pinned `best_model_checkpoint`) to local SSD, relocate `best_model_checkpoint` strictly inside the restored local `trainer_state.json`, refresh local `_CHECKPOINT_COMMIT` (`L685-L694, L709, L804`), and resume `Trainer`. Partial intra-epoch steps are discarded cleanly. Already-committed units (`UNIT_COMMITTED` + verified `preds.npz` / `metrics.json`) are skipped via `RESUME=True` after `run_state.json` identity verification.
- **Artifact Integrity Verification:** Cell 5 validates all `34` portable kit files, `8/8` `v2` baseline hashes, live re-derivation of `4` frozen test masks, and rogue-file rejection before Cell 6 starts. Every committed unit verifies `metrics.json` + `preds.npz` SHA256, finite probability sums (`1.0`), shape, row-ID/label ordering, and any `checkpoint_prune_errors`.
- **Rollback & Cleanup Procedures:**
  - Local Git workspace is untouched by Colab runs.
  - Normal benchmark completion: keep `.../artifacts/t4_benchmark/<ARTIFACT_RUN_ID>/` intact for audit and check/empty Google Drive Trash if pruned staging folders consume quota.
  - Abort / full rollback: terminate Colab runtime, inspect `leases/` and `logs/runner.log` if needed, and delete ONLY `.../artifacts/t4_benchmark/<ARTIFACT_RUN_ID>/` (or the entire `MyDrive/bangla_smish_post_stage_b_stage4_v3/` workspace folder) + empty Drive Trash.

---

## 6. Scientific Safety & Risk Assessment

### 6.1 Scientific Safety Confirmations (All Enforced in Code & Verified)
- **Frozen datasets untouched:** `post_stage_b_v2` bundle & 282-file baseline are read-only inputs verified by SHA256 (`post_stage_b_runner.py:1123-1124` rejects overlapping bundle and artifact paths).
- **Confirmatory declarations disabled:** `CONFIRMATORY_DECLARATIONS_ENABLED = False`, `STATISTICAL_VALIDATION_STATUS = "FAIL_OR_INCONCLUSIVE_CONFIRMATORY_DISABLED"`, [`reports/stage4_statistical_validation_results.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/stage4_statistical_validation_results.json) untouched (`bde8085b77066cbf449a1b17d0a1afa77f2dcbcb4ea1a82f7d0dde68f6fe504f`).
- **Zero held-out / OOD exposure:** `run_heldout_eval = false` (`post_stage_b_runner.py:181-182` fail-closed); `load_best_model_at_end=True` selects strictly on Fold 0 CV `eval_loss`; inference benchmark uses dev-only workload fixture (`0` test/OOD rows).
- **No unauthorized views:** `canon_masked` only (`brand_masked` and `brand_stripped` excluded).
- **No synthetic generation:** Permanently stopped; zero generation paths exist in Stage 4 runner/notebook.

### 6.2 Risk Register

| Risk | Likelihood / Impact | Built-in Safeguard & Mitigation | Escalation / Stop Trigger |
|---|---|---|---|
| **T4 VRAM OOM on `xlm-roberta-base` or `muril-base-cased` (`bs=32`)** | Medium / Medium | FP16 enabled (`fp16="auto"`); `max_length=128`; `eval_batch_size=64` | **STOP** benchmark -> request Permission C to set `batch_size=16, gradient_accumulation_steps=2` |
| **Colab `pip`/CUDA wheel conflict or post-install crash** | Medium / Low-Medium | `torch==<ver>` constraint file + CUDA/import verification in Cell 4 | **STOP** at Cell 4; restart runtime once; if still failing, **STOP** |
| **Live HF 32-pair tokenizer collision or `normalizer` mismatch** | Low / Critical | Cell 5 fail-closed gate blocks Cell 6 + saves `tokenizer_collision_gate.json` | **STOP** immediately at Cell 5; inspect collision report |
| **Drive FUSE latency stall / Drive Trash quota exhaustion** | Medium / Medium | Local SSD training; `checkpoint_retention_limit=1`; post-commit pruning | If Drive sync fails or quota fills, **STOP**, inspect `runner.log`, check Drive Trash |
| **Colab session preemption / timeout mid-unit** | Medium / Low | Epoch-level recovery + `run_state.json` unit skip (`RESUME=True`) | Inspect `leases/`; only if runtime is confirmed dead, authorize takeover (`ALLOW_STALE_WRITER_TAKEOVER`) |
| **Accidental second Colab tab running concurrently** | Low / High | Single-writer lease fails closed at unit boundaries | Never open 2 tabs against same `ARTIFACT_RUN_ID` |
| **`raw` view (128 checks) unverified live during T4-only Cell 5** | Certain under `C0` / Low | Non-blocking for T4 (`canon_masked` only); checked when Option A runs or via optional `C1` patch | Choose `C0` (defer `raw` gate) or `C1` (check all 256 in Cell 5) |

---

## 7. Preflight Checklist (Mandatory Sequence Before Training Starts)

1. `[ ]` Obtain explicit owner sign-off on **Permission A** and **Permission B** (and decision **C0** vs **C1**); verify **Permission D** remains **`NO-GO`**.
2. `[ ]` Locally stage a fresh kit with `scripts/prepare_stage4_colab_kit.py --kit-dir <fresh_local_dir>` and confirm `34/34` files + `4/4` masks verify locally.
3. `[ ]` Upload the staged kit to brand-new Drive folder `MyDrive/bangla_smish_post_stage_b_stage4_v3/` (confirm destination did not previously exist).
4. `[ ]` Record pre-run Colab Compute Units (CU) and free Google Drive GB quota.
5. `[ ]` Open a **single** Colab T4 GPU runtime; configure Cell 1: `RUN_MODE='t4_benchmark'`, `AUTHORIZE_T4_BENCHMARK=True`, `AUTHORIZE_FULL_TRAINING=False`, `USE_GOOGLE_DRIVE=True`, `DRIVE_WORKSPACE_REL='bangla_smish_post_stage_b_stage4_v3'`, `ARTIFACT_RUN_ID='stage4-t4-bench-01'`, `RESUME=True`, `ALLOW_STALE_WRITER_TAKEOVER=False`.
6. `[ ]` Run Cell 2: confirm Tesla T4 (`~15 GB` VRAM), `torch.cuda.is_available() == True`, sufficient free `/content` SSD.
7. `[ ]` Run Cell 3: mount Drive and verify `WORKSPACE_ROOT`, `SRC_DIR`, `BUNDLE_DIR`, and `CONFIG_PATH`.
8. `[ ]` Run Cell 4: install/verify dependencies preserving CUDA PyTorch + `normalizer`; restart Colab runtime if prompted and re-run Cells 1–4.
9. `[ ]` Run Cell 5: confirm `verify_portable_colab_kit` (`34` files, `4` masks), `verify_manifest`, and `run_allowed_view_tokenizer_collision_gate` (`0` collisions) all pass and `CONFIRMATORY_DECLARATIONS_ENABLED = False`.
10. `[ ]` Only after Cells 1–5 succeed, run Cell 6 (`5` units) and Cell 7; record post-run Colab CU balance, VRAM peaks, timing breakdown, and Drive/Trash usage.

---

## 8. Explicit GO / NO-GO Recommendation

| Permission | Decision | Rationale |
|---|---|---|
| **A. Five-Unit T4 Benchmark Execution** | **`CONDITIONAL GO`** | Safe, dev-only (`0` held-out/OOD exposure), `3.38%` of full-grid steps, essential to measure VRAM, latency, Drive FUSE, and `CU/hr` before any Option A commitment. Requires Cell 4–5 runtime gates to pass. |
| **B. Google Drive Artifact Writes (`..._stage4_v3` only)** | **`CONDITIONAL GO`** | Required if `USE_GOOGLE_DRIVE = True`; strictly isolated to brand-new folder `MyDrive/bangla_smish_post_stage_b_stage4_v3/` with append-only audit trails and zero writes to historical workspaces. |
| **C. Additional Code Modifications Before Benchmark** | **`C0: NONE REQUIRED` (Recommended)** | Zero blockers prevent starting the 5-unit benchmark. Optional **C1** patches (checking all 256 pairs in Cell 5, pinning `revision=` on `AutoModelForSequenceClassification`, failing closed on missing `_commit_hash`) are clean if owner prefers hardening before benchmarking. |
| **D. Full 160-Unit Option A Training** | **`NO-GO`** | Remains strictly blocked pending empirical T4 VRAM, runtime, Drive stability, and 200-CU budget review + separate owner authorization. |

---

## 9. Owner Approval Checklist (Strictly Separated Permissions)

*Approving any single permission below does NOT authorize any other permission.*

### Permission A — Five-Unit T4 Benchmark Execution (`post_stage_b_stage4_t4_benchmark_v1`)
- `[ ]` **APPROVE** (`t4_benchmark` only, Fold `0`, `canon_masked` only, `5` units, zero held-out/OOD evaluation, fresh `ARTIFACT_RUN_ID`)
- `[ ]` **REJECT** / `[ ]` **DEFER**

### Permission B — Scoped Google Drive Artifact Writes (`MyDrive/bangla_smish_post_stage_b_stage4_v3/` ONLY)
- `[ ]` **APPROVE** (Fresh isolated Stage 4 Drive workspace + `artifacts/t4_benchmark/<ARTIFACT_RUN_ID>/` only; acknowledges Colab OAuth mounts full `MyDrive`; zero edits to existing Drive workspaces)
- `[ ]` **REJECT** / `[ ]` **DEFER** *(If rejected, Colab T4 run must set `USE_GOOGLE_DRIVE = False` or not run)*

### Permission C — Additional Code Modifications (Before or During Benchmark Recovery)
- `[ ]` **OPTION C0 (Recommended): NO CODE MODIFICATIONS BEFORE BENCHMARK**
- `[ ]` **OPTION C1: APPROVE SPECIFIC PRE-BENCHMARK MODIFICATIONS ONLY:**
  - `[ ]` Modify Notebook Cell 5 to run all `256` checks (`raw` + `canon_masked`) even during `t4_benchmark`
  - `[ ]` Pin `revision=` on `AutoModelForSequenceClassification.from_pretrained(...)` in `post_stage_b_runner.py:575`
  - `[ ]` Fail closed if HF tokenizer omits `_commit_hash` in `post_stage_b_tokenizer_gate.py:141-145`
- `[ ]` **REJECT ANY CODE MODIFICATIONS**

### Permission D — Full 160-Unit Option A Training (`experiments_abcd`)
- **`[X]` REMAINS STRICTLY `NO-GO` (NOT REQUESTED & NOT AUTHORIZED HERE)**
- Future authorization requires a separate decision brief reviewing completed T4 benchmark telemetry (`VRAM`, `CU/hr`, `drive_sync_seconds`, extrapolated $\hat{\text{CU}}_{160}^{\text{safe}}$) and statistical governance status.
