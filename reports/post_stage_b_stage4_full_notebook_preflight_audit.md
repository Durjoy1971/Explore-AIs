# Stage 4 Full Notebook Line-by-Line Preflight Audit & v7 Independent-Review Remediation Report

- **Repository:** `/home/bs01415/Documents/Programming/ExploringAIs`
- **Canonical Notebook:** [`experiments/colab/post_stage_b_stage4_experiments.ipynb`](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/post_stage_b_stage4_experiments.ipynb) (`8` cells total: Cell 0 markdown + Cells 1–7 code)
- **Canonical Generator:** [`scripts/generate_post_stage_b_stage4_notebook.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/scripts/generate_post_stage_b_stage4_notebook.py)
- **Latest Verified Portable Kit:** `bangla_smish_post_stage_b_stage4_v7` ([`stage4_colab_kit_upload/bangla_smish_post_stage_b_stage4_v7/`](file:///home/bs01415/Documents/Programming/ExploringAIs/stage4_colab_kit_upload/bangla_smish_post_stage_b_stage4_v7))
- **Preserved Historical Kits (Untouched & Independently Re-Verified):** `v3`, `v4`, `v5`, `v6` under `stage4_colab_kit_upload/`
- **Execution Boundary Disclosure:** All validation in this report used **mocked CPU unit tests only** under a hard kernel-enforced `<=25%` host CPU cap (`3/12` CPUs). **Mocked tests do NOT validate real HuggingFace `Trainer`, real HuggingFace Hub downloads, real `normalizer`, real Tesla T4 CUDA memory/FP16 behavior, or real Google Drive FUSE durability/latency.** No Colab execution, GPU benchmarking, model training, synthetic generation, held-out/OOD evaluation, Google Drive writes, or Git mutations were performed.

---

## Executive Summary

Across the Stage 4 Cell 5/6 audit (`v4`), full notebook line-by-line audit (`v5`), v6 independent-review remediation (`v6`), and v7 targeted remediation (`v7`), all identified deterministic integration, pre-lease compatibility, hardware-binding, crash-reconciliation, step-count validation, atomic-write, package-fingerprint, retention, and benchmark-scope defects in [`post_stage_b_checkpointing.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_checkpointing.py), [`post_stage_b_runner.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py), and [`generate_post_stage_b_stage4_notebook.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/scripts/generate_post_stage_b_stage4_notebook.py) have been fixed and verified without modifying any Category A file, frozen dataset, frozen configuration, or evaluation mask:

| Issue | Root Cause | Resolution (v4 → v7) |
| :--- | :--- | :--- |
| **Cell 5 / Cell 6 artifact init conflict** | Cell 5 wrote `ARTIFACTS_DIR/exports/tokenizer_collision_gate.json` before Cell 6 initialized `run_state.json`, triggering `_read_state()` rejection | Cell 5 writes preflight report to `tempfile.gettempdir()`, captures `TOK_GATE_REPORT`, passes `tokenizer_gate_report` explicitly to Cell 6; runner validates fail-closed before `run_state.json` init and persists atomically post-init (`_read_state()` preserved untouched) |
| **Cross-`ARTIFACT_RUN_ID` local SSD contamination** | `local_ckpt_base` used only `cfg.run_name`, allowing a fresh `ARTIFACT_RUN_ID` on the same Colab VM to silently resume another run's local checkpoint | `_local_checkpoint_base(adir, cfg)` scopes by `<adir.name>_<sha256(adir)[:8]>` and `_next_attempt` scans it |
| **Unbounded local SSD checkpoint growth** | Committed units pruned Drive staging only, leaving ~37 GB (5-unit T4) / ~1.2 TB (160-unit Option A) on `/content/tmp_stage4_ckpt` | Post-commit local SSD unit directory cleanup runs after durable `run_state.json` commit when unit weights are not retained |
| **Cell 6 stale-state / skipped-cell bypass & Cell 0/1/3/7 gaps** | Re-running Cell 1 with new `RUN_MODE`/`ARTIFACT_RUN_ID` and jumping to Cell 6 reused stale globals; Cell 0 lacked working-copy & restart notices | Cell 1 resets preflight markers & validates `DRIVE_WORKSPACE_REL`; Cell 3 uses `Path('/content') / DRIVE_WORKSPACE_REL`; Cell 5 binds `PORTABLE_KIT_VERIFIED` + `VERIFIED_*`; Cell 6 enforces exact binding; Cell 0 documents working-copy, Cell 4 restart, and T4 hardware/fail-closed rules |
| **A.1 Model revision pinning** | `AutoModelForSequenceClassification.from_pretrained` did not pass `revision=...` or verify `model.config._commit_hash` | Both `AutoTokenizer` and `AutoModelForSequenceClassification` pass `revision=pinned_rev`, fail closed if `_commit_hash` is missing or mismatched, bind `pinned_model_revisions` in `_run_identity`, and enforce pinned `model_revision`/`tokenizer_revision` in `_validate_unit` on resume |
| **A.2 / v7-F4 Expected vs. actual Trainer step integrity & terminology** | Earlier fallback substituted expected steps for missing/`0` actual steps; v6 allowed mismatched positive `actual_global_step != expected` | Removed all fallbacks; verified exact `transformers==4.57.1` `src/transformers/trainer.py:2747-2754` behavior (`state.global_step` increments per gradient-accumulation update step even when FP16 `GradScaler` skips a weight update on overflow); real `hf_train_and_predict_fold` now fails closed on missing, `<=0`, or `!= expected_optimizer_steps`; distinguishes expected Trainer update-step budget, actual `global_step`, and uninstrumented `successful_optimizer_weight_updates=None` |
| **A.3 / v7-F3 Idempotent completion-event reconciliation & pruning recovery** | Crash between durable `run_state.json` commit and `UNIT_COMMITTED` emission or post-commit pruning left missing events or unpruned checkpoints on resume | Added `ensure_unit_completion_event()` (accepts exact 1 matching valid event, creates when 0 exist, rejects duplicate matching events, rejects mismatched `config_hash`/`dataset_manifest_sha256`/`metrics_sha256`/`preds_sha256`) and `_reconcile_committed_unit_post_commit()` called both post-commit and on resume skip without ever rewriting `metrics.json` or invalidating authoritative `run_state.json` commits |
| **A.4 CUDA cleanup on exceptions** | `del trainer, model; gc.collect(); torch.cuda.empty_cache()` ran only on the normal return path | Wrapped model/trainer lifecycle in `hf_train_and_predict_fold` inside `try ... finally` so cleanup runs on both normal return and every exception |
| **A.5 / v7-F6 Complete package, environment & hardware fingerprint** | v6 used `getattr(mod, "__version__", "installed")` (emitting `"installed"` for `normalizer`), omitted `tokenizers`/`packaging`/`huggingface_hub`/`protobuf`/normalizer transitive deps, and omitted GPU/driver identity | Upgraded `_IDENTITY_PACKAGE_DISTS` (19 runtime-critical distributions including `tokenizers`, `packaging`, `huggingface_hub`, `protobuf`, `normalizer`, `unicodedata2`, `ftfy`, `emoji`) resolved via `importlib.metadata.version` first with zero `"installed"` fallback (`"unknown_version"` fails closed); bound complete `hardware_fingerprint` (including exact `nvidia_driver_version`) + `environment_fingerprint` into `_run_identity` (`schema_version: 4`, Decision 3 STRICT resume match) |
| **A.6 Weight-retention compatibility** | `_read_committed_unit` hardcoded `artifacts_dir/checkpoints/<uid>`, breaking when `local_checkpoint_root` was set with `best_fold0`/`all` | `_read_committed_unit` resolves `_local_checkpoint_base(artifacts_dir, cfg) / uid`; `_execute_units` applies `keep_this_unit = _should_keep_weights(cfg.keep_model_weights, fold)` consistently across `none`, `best_fold0`, and `all` |
| **B.7 / v7-F1 Pre-lease, pre-state synthetic workload compatibility guard** | v6 checked `SYNTHETIC_WORKLOAD_FIXTURE_ENABLED` inside `hf_train_and_predict_fold` after GPU training & inference | Moved compatibility check into `_validate_pre_run_compatibility()` at the top of `_run_locked()` BEFORE `adir.mkdir`, lease acquisition, `run_state.json`, model loading, or training (retaining defense-in-depth inside `hf_train_and_predict_fold`). Under **Decision 2 (Option A)**, frozen `t4_benchmark` config stays `benchmark_workload_fixture_eval=true` with `SYNTHETIC_WORKLOAD_FIXTURE_ENABLED=False` (fail-closed) |
| **v7-F2 Strict Tesla T4 hardware verification** | Cell 2 and runner checked CUDA availability only without enforcing single-GPU Tesla T4 hardware | Both Notebook Cell 2 and `_validate_pre_run_compatibility()` enforce `cuda_available=True`, `gpu_count == 1`, `"tesla t4"` in GPU name, `compute_capability == "7.5"`, and verified non-empty `nvidia_driver_version` for `t4_benchmark`; `gpu_smoke_test` behavior remains unchanged |
| **v7-F5 Hardened atomic writes & POSIX directory `fsync`** | `atomic_write_text` and `_write_prediction_archive` used PID-only temp names without orphan cleanup or parent directory `fsync` | Upgraded both to PID+UUID temp filenames, `except BaseException:` orphan cleanup, `flush()` + `os.fsync()`, `os.replace()`, and `_fsync_dir_best_effort()` (tolerating FUSE `EINVAL`/`EOPNOTSUPP`/`EPERM`/`EBADF` while honestly disclosing local-SSD vs. Google Drive FUSE durability limits) |
| **B.8 Held-out / OOD access guard** | `_run_locked` and `aggregate_post_stage_b_results` unconditionally loaded test/OOD parquets when `run_heldout_eval=False` | Replaced with `_load_configured_eval_sets(bdir, cfg)` (zero runner reads of test/OOD parquets when `run_heldout_eval=False`) + explicit fail-closed guard inside `hf_train_and_predict_fold` |
| **B.9 Counterfactual evaluation disclosure (Decision 4 — PRESERVE)** | `t4_benchmark` runs `run_counterfactual_eval=true` on Fold 0 Smishing probe where `contact_swap` is a `98.59%` no-op on `canon_masked` | Preserved untouched per **Decision 4**; continued documenting `98.59%` `contact_swap` no-op on `canon_masked` and `~765` Fold 0 forward passes/unit in Section 4 |

---

## 1. Complete Cell-by-Cell Review (All 8 Cells)

The canonical notebook [`experiments/colab/post_stage_b_stage4_experiments.ipynb`](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/post_stage_b_stage4_experiments.ipynb) contains **8 cells** (`s4md0001` through `s4out008`).

### Cell 0 (`id: s4md0001`, Markdown)
1. **Purpose:** Governance header stating working-copy requirement (canonical notebook immutability), expected Cell 4 first-run restart, permanent Stage B synthetic stop, T4 benchmark & Option A authorization gates, 32-pair tokenizer collision gate, statistical governance (`CONFIRMATORY_DECLARATIONS_ENABLED = False`), and checkpoint/lease guarantees.
2. **Inputs/Prerequisites:** None.
3. **State created:** None.
4. **Functions/modules invoked:** None.
5. **Files read:** None.
6. **Files written:** None.
7. **Modifies `WORKSPACE_ROOT`/`ARTIFACTS_DIR`:** No (when executed from an external working copy).
8. **Expected output:** Rendered markdown governance notice.
9. **Exceptions/failures:** Operational failure only if executed directly from `WORKSPACE_ROOT` (Colab autosave mutates canonical SHA256 → caught fail-closed in Cell 5).
10. **Safe to rerun:** Yes.
11. **Safe after restart:** Yes.
12. **Interaction with subsequent cells:** Operational guidance.

### Cell 1 (`id: s4cfg002`, Code — Runtime Configuration & Explicit Authorization Gates)
1. **Purpose:** Sets 12 form parameters (`RUN_MODE`, `AUTHORIZE_T4_BENCHMARK`, `AUTHORIZE_FULL_TRAINING`, `REQUIRE_GPU`, `USE_GOOGLE_DRIVE`, `DRIVE_WORKSPACE_REL = 'bangla_smish_post_stage_b_stage4_v7'`, `ARTIFACT_RUN_ID = 'stage4-v7-smoke-01'`, `RESUME`, `ALLOW_STALE_WRITER_TAKEOVER`, `EXPECTED_STALE_WRITER_IDS`, `EXPECTED_CORRUPT_LEASE_FILES`, `STALE_TAKEOVER_REASON`), defines & runs `validate_stage4_launch_settings()`, and resets all preflight verification markers.
2. **Inputs/Prerequisites:** Working copy outside `WORKSPACE_ROOT`; owner-authorized parameter values.
3. **State created:** All 12 config globals, `validate_stage4_launch_settings`, and reset preflight markers (`PORTABLE_KIT_VERIFIED=False`, `VERIFIED_WORKSPACE_ROOT=None`, `VERIFIED_RUN_MODE=None`, `VERIFIED_ARTIFACT_RUN_ID=None`, `TOKENIZER_COLLISION_GATE_PASSED=False`, `TOK_GATE_REPORT=None`, `TOK_GATE_PREFLIGHT_PATH=None`).
4. **Functions/modules invoked:** `re`, `pathlib.Path`, `validate_stage4_launch_settings()`.
5. **Files read / written:** None / None.
6. **Modifies `WORKSPACE_ROOT`/`ARTIFACTS_DIR`:** No.
7. **Expected output:** `Configured Stage 4 RUN_MODE=... ARTIFACT_RUN_ID=...`.
8. **Exceptions/failures:** `ValueError` on invalid `RUN_MODE`, empty/absolute/`..` `DRIVE_WORKSPACE_REL`, invalid `ARTIFACT_RUN_ID`, or incomplete takeover args; `RuntimeError` on unauthorized `t4_benchmark` or `experiments_abcd`.
9. **Safe to rerun:** Yes (intentionally invalidates stale Cell 5 state so Cells 3 and 5 must be re-run before Cell 6).
10. **Safe after restart:** Yes.
11. **Interaction with subsequent cells:** Feeds `RUN_MODE` and `REQUIRE_GPU` (Cells 2, 4), path/mode variables (Cells 3, 5, 6), and `validate_stage4_launch_settings` + takeover params (Cell 6).

### Cell 2 (`id: s4gpu003`, Code — Verify Colab Runtime, GPU & Local SSD)
1. **Purpose:** Inspects Python, platform, `nvidia-smi` (GPU name, VRAM, exact `driver_version`), `torch.__version__`, CUDA availability (`REQUIRE_GPU`), GPU count, compute capability (`torch.cuda.get_device_capability(0)`), and `/content` free SSD space. When `RUN_MODE == 't4_benchmark'`, enforces strictly single-GPU Tesla T4 hardware (`gpu_count == 1`, `'tesla t4'` in GPU name, `compute_capability == '7.5'`, non-empty `nvidia_driver_version`) while preserving normal `gpu_smoke_test` behavior unchanged.
2. **Inputs/Prerequisites:** Cell 1 (`RUN_MODE`, `REQUIRE_GPU`).
3. **State created:** `IN_COLAB`, `nvidia_driver_version`, `cuda_ok`, `gpu_count`, `gpu_name`, `compute_capability`, `props`, `HARDWARE_PREFLIGHT`, `ssd_path`, `disk`.
4. **Functions/modules invoked:** `platform`, `shutil`, `subprocess`, `sys`, `Path`, `torch`.
5. **Files read / written:** `/content` `statvfs` / None.
6. **Modifies `WORKSPACE_ROOT`/`ARTIFACTS_DIR`:** No.
7. **Expected output:** Runtime, GPU count/name/CC/VRAM, NVIDIA driver version, Tesla T4 confirmation (when `RUN_MODE == 't4_benchmark'`), and free SSD summary.
8. **Exceptions/failures:** `NameError` if Cell 1 skipped; `RuntimeError` if `REQUIRE_GPU=True` and CUDA absent, or if `RUN_MODE == 't4_benchmark'` and hardware is not a single verified Tesla T4 (`7.5`) with exact driver version.
9. **Safe to rerun / after restart:** Yes / Yes (after Cell 1).
10. **Interaction with subsequent cells:** Pre-imports `torch` (preserved via Cell 4 `-c constraints.txt`) and records `HARDWARE_PREFLIGHT` matching the runner's `collect_hardware_fingerprint()`.

### Cell 3 (`id: s4mnt004`, Code — Mount Fresh Stage 4 Google Drive Kit & Resolve Paths)
1. **Purpose:** Mounts Drive (`force_remount=False`) if `USE_GOOGLE_DRIVE`, resolves `WORKSPACE_ROOT`, `SRC_DIR`, `BUNDLE_DIR`, `CONFIG_DIR`, `ARTIFACTS_DIR`, `CFG_MAP`, `CONFIG_PATH`, checks required kit paths exist, and inserts `SRC_DIR` into `sys.path`.
2. **Inputs/Prerequisites:** Cell 1 (`USE_GOOGLE_DRIVE`, `DRIVE_WORKSPACE_REL`, `RUN_MODE`, `ARTIFACT_RUN_ID`); staged kit at `WORKSPACE_ROOT`.
3. **State created:** Path globals + `sys.path` entry.
4. **Functions/modules invoked:** `google.colab.drive.mount` (if `USE_GOOGLE_DRIVE`), `Path.is_dir`, `Path.is_file`.
5. **Files read / written:** `WORKSPACE_ROOT` structure / None.
6. **Modifies `WORKSPACE_ROOT`/`ARTIFACTS_DIR`:** No.
7. **Expected output:** Clean mount and path resolution.
8. **Exceptions/failures:** `FileNotFoundError` if kit incomplete; Drive auth error; `KeyError` on unknown `RUN_MODE`.
9. **Safe to rerun / after restart:** Yes / Yes (after Cell 1).
10. **Interaction with subsequent cells:** Supplies all workspace paths and `sys.path` to Cells 4–7.

### Cell 4 (`id: s4dep005`, Code — Preflight, Install & Verify Compatible Dependencies)
1. **Purpose:** Executes [`post_stage_b_colab_deps.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_colab_deps.py) via `runpy.run_path` (without importing `bangla_smish` first) and runs `preflight_colab_dependencies(cfg_models, install_missing=True, require_cuda=REQUIRE_GPU)`.
2. **Inputs/Prerequisites:** Cell 1 (`REQUIRE_GPU`), Cell 3 (`SRC_DIR`, `CONFIG_PATH`).
3. **State created:** `deps_mod`, `cfg_models`, `dep_status`.
4. **Functions/modules invoked:** `runpy.run_path`, `json.loads`, `preflight_colab_dependencies`.
5. **Files read / written:** `post_stage_b_colab_deps.py`, `post_stage_b_tokenization.py`, `CONFIG_PATH`, `importlib.metadata` / temp pip constraints + `site-packages` if installing.
6. **Modifies `WORKSPACE_ROOT`/`ARTIFACTS_DIR`:** No.
7. **Expected output:** First run on fresh Colab VM raises intentional `RuntimeError("... RESTART REQUIRED ...")` after constrained install; post-restart rerun prints `status: "verified"`.
8. **Exceptions/failures:** `RuntimeError` for `torch < 2.6`, intentional `RESTART REQUIRED`, import failure, CUDA absent, or missing `normalizer.normalize`.
9. **Safe to rerun / after restart:** Yes / Yes (designed for restart-and-rerun).
10. **Interaction with subsequent cells:** Ensures verified package stack before Cell 5 imports `bangla_smish.*` and HF tokenizers.

### Cell 5 (`id: s4ver006`, Code — Verify Portable Kit, Frozen Masks, Config & 32-Pair Tokenizer Gate)
1. **Purpose:** Verifies portable kit (`34` files, `4` live-rederived masks), verifies v2 dataset manifest & checksums, loads & seeds `cfg`, runs 32-pair allowed-view tokenizer collision gate writing ONLY to `tempfile.gettempdir()`, captures `TOK_GATE_REPORT`, and binds `PORTABLE_KIT_VERIFIED` + `VERIFIED_*`.
2. **Inputs/Prerequisites:** Cells 1, 3, 4.
3. **State created:** `PORTABLE_KIT_VERIFIED`, `VERIFIED_WORKSPACE_ROOT`, `VERIFIED_RUN_MODE`, `VERIFIED_ARTIFACT_RUN_ID`, `TOKENIZER_COLLISION_GATE_PASSED`, `TOK_GATE_REPORT`, `TOK_GATE_PREFLIGHT_PATH`, `kit_summary`, `manifest`, `cfg`.
4. **Functions/modules invoked:** `verify_portable_colab_kit`, `verify_manifest`, `sha256_file`, `load_config`, `seed_everything`, `run_allowed_view_tokenizer_collision_gate`, `collect_environment_info`.
5. **Files read:** `stage4_kit_manifest.json`, 34 runtime kit files, v2 bundle parquets, `CONFIG_PATH`, pinned HF tokenizers.
6. **Files written:** System temp preflight JSON (`Path(tempfile.gettempdir()) / f'stage4_tokenizer_collision_gate_{RUN_MODE}_{ARTIFACT_RUN_ID}_{cfg.config_hash()[:12]}.json'`), Python `__pycache__/*.pyc` (ignored by kit verifier), `~/.cache/huggingface`.
7. **Modifies `WORKSPACE_ROOT`/`ARTIFACTS_DIR`:** `ARTIFACTS_DIR`: **No.** `WORKSPACE_ROOT`: ignored `__pycache__` only.
8. **Expected output:** `34` files verified, `4` masks verified, `Tokenizer collision gate PASSED: 128 checks, collisions= 0` (`t4_benchmark`), statistical governance status, config hash, environment snapshot.
9. **Exceptions/failures:** Kit corruption/tampering, manifest mismatch, 32-pair missing rows, revision mismatch, or any tokenizer collision (`>0`).
10. **Safe to rerun / after restart:** Yes / Yes.
11. **Interaction with subsequent cells:** Supplies verified `cfg`, `TOK_GATE_REPORT`, and binding flags to Cell 6.

### Cell 6 (`id: s4run007`, Code — Execute Stage 4 Pipeline)
1. **Purpose:** Re-validates launch settings, verifies all prerequisite globals & exact Cell 5 binding (`PORTABLE_KIT_VERIFIED`, `VERIFIED_*`, `ARTIFACTS_DIR`, `CONFIG_PATH`, `cfg.run_mode`), verifies `TOKENIZER_COLLISION_GATE_PASSED` + `isinstance(TOK_GATE_REPORT, dict)`, and runs `run_post_stage_b_pipeline(..., tokenizer_gate_report=globals().get('TOK_GATE_REPORT'))`.
2. **Inputs/Prerequisites:** Cells 1, 3, 4, 5 executed cleanly for current `RUN_MODE` and `ARTIFACT_RUN_ID`.
3. **State created:** `stale_ids`, `corrupt_files`, `summary`.
4. **Functions/modules invoked:** `validate_stage4_launch_settings`, `run_post_stage_b_pipeline` and full runner lifecycle.
5. **Files read / written:** Configured eval/train parquets only (zero test/OOD reads when `run_heldout_eval=False`), `ARTIFACTS_DIR/*` (leases, `run_state.json`, `exports/*`, `logs/*`, `staging/*`, `events/*`, `pointers/*`, `units/*`), run-isolated local SSD checkpoints under `<local_checkpoint_root>/<cfg.run_name>/<adir.name>_<sha[:8]>/`.
6. **Modifies `WORKSPACE_ROOT`/`ARTIFACTS_DIR`:** Writes inside `ARTIFACTS_DIR` (under `WORKSPACE_ROOT/artifacts/`, ignored by kit verifier); never modifies any verified kit file.
7. **Expected output:** Unit execution/skip logs + `Completed units total: N / M`. *(Note: Under Decision 2 Option A, the frozen `t4_benchmark` config with `benchmark_workload_fixture_eval=True` and `SYNTHETIC_WORKLOAD_FIXTURE_ENABLED=False` intentionally fails closed BEFORE acquiring a lease, creating `ARTIFACTS_DIR`, initializing `run_state.json`, or training.)*
8. **Exceptions/failures:** Pre-lease synthetic workload compatibility guard, strict Tesla T4 hardware verification failure, missing/unversioned required runtime package or pinned `normalizer` commit mismatch, stale/missing preflight state, invalid/missing gate report, unidentified pre-existing `ARTIFACTS_DIR` (`_read_state`), unresolved lease/corrupt lease, incompatible identity/environment/NVIDIA driver version on resume, pinned model/tokenizer revision mismatch, `actual_global_step <= 0` or `!= expected_optimizer_steps`, duplicate or hash-mismatched `UNIT_COMMITTED` event, CUDA OOM / network errors.
9. **Safe to rerun / after restart:** Yes when `RESUME=True` (skips verified committed units; preserves `exports/tokenizer_collision_gate.json`).
10. **Interaction with subsequent cells:** Produces `ARTIFACTS_DIR/exports/*` for Cell 7.

### Cell 7 (`id: s4out008`, Code — Inspect Exported Telemetry, Optimizer Steps, Retraining Metadata & Results)
1. **Purpose:** Read-only summary of `ARTIFACTS_DIR/exports/` printing filenames, `tokenizer_collision_gate.json` pass status, `retraining_metadata.json` unit count & statistical status, and `aggregated_results.csv`.
2. **Inputs/Prerequisites:** Cell 3 (`ARTIFACTS_DIR`).
3. **State created:** `exports_dir`, `tok_gate_path`, `tgate`, `retrain_path`, `rmeta`, `csv_path`.
4. **Functions/modules invoked:** `json.loads`, `pandas.read_csv`, `Path` read methods.
5. **Files read / written:** `ARTIFACTS_DIR/exports/*` / None.
6. **Modifies `WORKSPACE_ROOT`/`ARTIFACTS_DIR`:** No.
7. **Expected output:** Export list, tokenizer gate status, retraining metadata summary, and results CSV (or clean notice if `exports/` does not yet exist).
8. **Safe to rerun / after restart:** Yes / Yes.

---

## 2. Crash-Safe Validation, State-Commit, Idempotent Event Reconciliation & Step-Count Semantics

In [`src/bangla_smish/post_stage_b_runner.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py) and [`src/bangla_smish/post_stage_b_checkpointing.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_checkpointing.py), `_run_locked` and `_execute_units` enforce the following strict fail-closed sequence:

0. **Pre-lease, pre-state compatibility validation (`_validate_pre_run_compatibility`):** Before creating `ARTIFACTS_DIR` or acquiring any writer lease, rejects `t4_benchmark` + `benchmark_workload_fixture_eval=True` (`SYNTHETIC_WORKLOAD_FIXTURE_ENABLED=False`), enforces strict single-GPU Tesla T4 (`7.5`) + verified `nvidia_driver_version` for `t4_benchmark`, and validates all required runtime distributions (including `tokenizers` and pinned `normalizer_commit`) for real HF runs.
1. **Train & infer** (`runner_fn`) with pinned tokenizer + model revisions and `try/finally` CUDA cleanup.
2. **Step-count semantics & strict validation (`transformers==4.57.1` verified):**
   - Verified directly against `transformers==4.57.1` `src/transformers/trainer.py` (lines 2747–2754): `self.state.global_step += 1` executes unconditionally outside `if not self.accelerator.optimizer_step_was_skipped:`. Under FP16 `GradScaler` overflow, `lr_scheduler.step()` and optimizer weight updates are skipped while `state.global_step` still increments.
   - Runner strictly distinguishes: (a) **expected Trainer update-step budget** (`expected_optimizer_steps` / `expected_trainer_update_steps`), (b) **actual Trainer global-step count** (`actual_global_step` = `Trainer.state.global_step`), and (c) **successful optimizer weight updates** (`successful_optimizer_weight_updates = None`, explicitly uninstrumented; never conflated with `actual_global_step`).
   - Real `hf_train_and_predict_fold` fails closed if `actual_global_step` is missing, `<= 0`, or `!= expected_optimizer_steps`.
3. **Hardened atomic writes (`atomic_write_text`, `_write_prediction_archive`):** PID+UUID temporary filenames, `except BaseException:` orphan cleanup, `flush()` + `os.fsync()`, `os.replace()`, and `_fsync_dir_best_effort(path.parent)` (persisting POSIX local-SSD directory entries while cleanly tolerating FUSE `EINVAL`/`EOPNOTSUPP`/`EPERM`/`EBADF` without overstating Google Drive FUSE durability).
4. **Build `commit` metadata & validate unit (`_validate_unit`) BEFORE any durable commit, event emission, or pruning:** Checks unit identity, split metadata, pinned `model_revision`/`tokenizer_revision`, metrics, low-FPR invariants, `best_model` hashes, and prediction archive integrity. Any failure marks attempt `"failed"` and leaves all Drive staging and local SSD recovery checkpoints 100% intact.
5. **Durable authoritative commit in `run_state.json`:** `attempt.status = "completed"`, `state["committed_units"][uid] = commit`, `atomic_write_json(state_path, state)`. `metrics.json` is strictly immutable after this point.
6. **Idempotent post-commit reconciliation (`_reconcile_committed_unit_post_commit` & `ensure_unit_completion_event`):**
   - Runs immediately after durable `run_state.json` commit AND on resume inside the verified `committed is not None` skip branch (outside the commit-invalidation `try/except`, so event-write failures or event-integrity errors NEVER invalidate or modify an authoritative `run_state.json` commit and NEVER rewrite `metrics.json`).
   - `ensure_unit_completion_event`: creates `UNIT_COMMITTED` when 0 matching valid events exist; accepts exact 1 matching valid event idempotently (`created=False`); rejects duplicate matching events (`>1`); rejects any mismatch in `config_hash`, `dataset_manifest_sha256`, `metrics_sha256`, or `preds_sha256`.
   - Post-commit pruning (ONLY when `not keep_this_unit`): scans and prunes `ALREADY_SUPERSEDED` Drive staging + local SSD checkpoints; if a transient FUSE `OSError` occurs, records `post_commit_prune_errors` in `run_state.json` and reconciles cleanly on resume (clearing `post_commit_prune_errors` and setting `training_checkpoints_pruned_post_commit = True`).

---

## 3. Weight-Retention Compatibility (`none`, `best_fold0`, `all`)

- `_read_committed_unit` ([`post_stage_b_runner.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L1406-L1422)) now accepts `cfg` and verifies `commit["trainer_checkpoint"]` against `_local_checkpoint_base(artifacts_dir, cfg) / uid` (matching `_execute_units` for both `local_checkpoint_root=None` and `local_checkpoint_root!=None`).
- All three retention modes (`none`, `best_fold0`, `all`) were verified in mocked multi-fold resume + `aggregate_post_stage_b_results` tests with `local_checkpoint_root` set.
- Current frozen configs remain **untouched** (`keep_model_weights: "none"`).
- **Important operational note:** If an owner later authorizes `best_fold0` or `all` together with ephemeral Colab `/content/tmp_stage4_ckpt`, local SSD checkpoints vanish on VM reset even though `unit_dir/best_model` is durable on Drive; in that future scenario, `local_checkpoint_root` should be set to `None` or `trainer_checkpoint` storage relocated to durable storage.

---

## 4. Benchmark-Scope Investigation & Disclosures (Items B.7, B.8, B.9)

### B.7 Synthetic Inference Fixture (`_build_t4_benchmark_workload_fixture`) — Owner Decision 2: Option A
- **Finding:** [`post_stage_b_runner.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py) contains `_build_t4_benchmark_workload_fixture` which constructs `5,416` concatenated two-message strings (`f"{pool[(i * 7) % n_pool]} {pool[(i * 13 + 3) % n_pool]}"`) when `benchmark_workload_fixture_eval=True`, conflicting with the permanent synthetic-generation stop.
- **v7 Pre-Lease Guard & Disposition (Decision 2 — Option A):** `SYNTHETIC_WORKLOAD_FIXTURE_ENABLED: bool = False` remains set, frozen `post_stage_b_stage4_t4_benchmark_config.json` remains untouched (`benchmark_workload_fixture_eval: true`), and `_validate_pre_run_compatibility()` in `_run_locked()` now rejects the incompatible configuration **before** creating `ARTIFACTS_DIR`, acquiring a writer lease, initializing `run_state.json`, loading model weights, or running GPU training (retaining defense-in-depth inside `hf_train_and_predict_fold`). No replacement workload has been implemented or authorized in v7.

### B.8 Exact Dataset & Held-Out/OOD Access Audit (`run_heldout_eval=False`)
`run_heldout_eval=False` must **not** be misrepresented as meaning held-out/OOD files are never opened anywhere in the workflow. Exact file-by-file access across the entire `t4_benchmark` pipeline:

| Stage / Function | `real_train_pool` & Stage B synthetic | `eval_smishdetect_validation` (dev) | `counterfactual_smish_probe` (CV real) | `eval_smishdetect_test` (held-out) | `eval_banglaphish_ood` (OOD) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cell 5: `verify_portable_colab_kit`** | SHA256/bytes only | SHA256/bytes only | SHA256/bytes only | SHA256/bytes + **read for 4 frozen mask re-derivation** (no model scoring) | SHA256/bytes only |
| **Cell 5 & 6: `verify_manifest`** | SHA256 + Parquet schema/row-count | SHA256 + Parquet schema/row-count | SHA256 + Parquet schema/row-count | SHA256 + Parquet schema/row-count (integrity only) | SHA256 + Parquet schema/row-count (integrity only) |
| **Cell 5: `run_allowed_view_tokenizer_collision_gate`** | Read `row_id`, `text_raw`, `text_canon_masked` for 32 pairs | Same (32-pair lookup only) | Not read | Read `row_id`, `text_raw`, `text_canon_masked` **strictly for documented 32 pairs** (tokenizer ID comparison only; no model weights/scoring) | Same (32-pair lookup only) |
| **Cell 6: `_load_configured_eval_sets` (`run_heldout_eval=False`)** | Loaded per fold (`build_experiment_fold_data`) | Loaded if `benchmark_workload_fixture_eval=True` | Loaded if `run_counterfactual_eval=True` | **NEVER READ** (verified by `pd.read_parquet` spy test) | **NEVER READ** (verified by `pd.read_parquet` spy test) |
| **Cell 6: Model training / inference (`t4_benchmark`)** | Train on fold train; score CV `val_df` + `val_adv` | Scored if dev benchmark enabled | Scored on Fold 0 subset (`~255` rows × 3 variants) | **NEVER LOADED OR SCORED** (blocked fail-closed in `hf_train_and_predict_fold`) | **NEVER LOADED OR SCORED** (blocked fail-closed in `hf_train_and_predict_fold`) |

### B.9 Explicit Counterfactual Evaluation Disclosure — Owner Decision 4: PRESERVE
- `post_stage_b_stage4_t4_benchmark_config.json` keeps `"run_counterfactual_eval": true` completely untouched per **Owner Decision 4 (PRESERVE)**.
- `counterfactual_smish_probe.parquet` (`1,279` rows) is derived strictly from `real_train_pool.parquet` Smishing rows containing URLs or phone numbers. `select_counterfactual_fold` selects only the active CV validation fold (Fold 0 during `t4_benchmark`: `~255` rows, disjoint from training by both `row_id` and `family_id`, zero held-out/OOD exposure) and evaluates 3 variants (`orig`, `contact_swap`, `contact_brand_swap` = `~765` forward passes/unit).
- **Documented `canon_masked` behavior:** Because `canon_masked` already replaces URLs with `<URL>` and phones with `<PHONE>`, `contact_swap_text_canon_masked` is identical to `orig_text_canon_masked` on `98.59%` (`1,261/1,279`) of rows, while `contact_brand_swap_text_canon_masked` modifies real BD brand strings on `238` brand-bearing rows.

---

## 5. Regression Test Evidence & CPU-Capped Verification Commands (v7)

All tests below are **CPU-only mocked unit tests** executed under `systemd-run --user --scope -p CPUQuota=300% taskset -c 0-2 .venv/bin/python scripts/run_stage4_capped.py -- ...` (`3/12` host CPUs = `25%` ceiling, verified parent/child affinity `[0, 1, 2]`, `cgroup_quota_cpus = 3.0`). **None of these tests execute real HuggingFace `Trainer` training or real Tesla T4 GPU CUDA kernels.**

| Suite / Check | Exact Verification Scope (wrapped with `run_stage4_capped.py`) | Result | Effective Host CPU % |
| :--- | :--- | :--- | ---: |
| Stage `v7` portable kit | `prepare_stage4_colab_kit.py --kit-dir stage4_colab_kit_upload/bangla_smish_post_stage_b_stage4_v7` | Exit `0` (`34/34` files, `4/4` masks) | `9.11%` (combined capped run) |
| Independent verification of `v3`, `v4`, `v5`, `v6`, `v7` kits | `verify_portable_colab_kit` on all 5 kits | Exit `0` on all 5 kits (`34/34` files, `4/4` masks each; `v3`–`v6` untouched) | `9.11%` (combined capped run) |
| Generator / Notebook / `v7` Kit Parity | `nbformat.validate` + `build_stage4_notebook() == nb` + `34/34` file SHA256 parity | **Exact match** (`8` cells, `34/34` files; only `post_stage_b_checkpointing.py`, `post_stage_b_runner.py`, and notebook changed vs `v6`) | `9.11%` (combined capped run) |
| Focused Stage 4 tests (Scenarios A–L + v6 + v7 Findings 1–6) | `unittest -v tests/test_stage4_final_remediation.py` | **14/14 OK** (`9.338s`) | `9.11%` (standalone `9.51%`) |
| Full `unittest` regression suite | `unittest discover -s tests -p "test_*.py" -v` | **100/100 OK** (`120.752s`) | `9.11%` (combined capped run) |
| Core pipeline sanity test | `tests/test_pipeline.py` | **All tests passed!** | `9.11%` (combined capped run) |
| Protected 282-file & Frozen Data Integrity | `scripts/verify_stage4_integrity.py --verify-only` | **282/282** protected files, **13/13** inputs, v1 & v2 bundles (`0` normalization mismatches), **4/4** frozen masks | `9.11%` (combined capped run) |

### Stage 4 v7 Mocked Test Coverage Added in [`tests/test_stage4_final_remediation.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/tests/test_stage4_final_remediation.py)
- `test_v7_pre_lease_synthetic_workload_guard_frozen_t4_config_untouched`: verifies unmodified frozen `post_stage_b_stage4_t4_benchmark_config.json` fails closed BEFORE lease acquisition, `artifacts_dir` creation, `run_state.json`, model init, or training (`0` calls) with frozen config SHA256 unchanged.
- `test_v7_strict_t4_hardware_verification_notebook_runner_and_driver_resume`: verifies runner pre-lease hardware checks (missing CUDA, multi-GPU, non-T4 `NVIDIA L4`, wrong compute capability `8.0`, missing driver), Decision 3 strict `nvidia_driver_version` resume mismatch (rejects resume without mutating `run_state.json`), and verbatim Notebook Cell 2 T4 enforcement vs. preserved `gpu_smoke_test` behavior on non-T4 GPUs.
- `test_v7_idempotent_completion_event_reconciliation_and_fault_injection`: verifies `ensure_unit_completion_event` (0→1 create, 1→1 reuse, `>1` duplicate fail-closed, 4 hash-mismatch fail-closed cases) and all 5 pipeline interruption/reconciliation fault-injection scenarios without retraining or rewriting `metrics.json`.
- `test_v7_strict_trainer_global_step_validation_and_terminology`: verifies `transformers==4.57.1` terminology disclosure and fail-closed rejection when real HF `actual_global_step` (`2` or `4`) differs from expected Trainer update-step budget (`3`), while preserving `successful_optimizer_weight_updates=None`.
- `test_v7_hardened_atomic_writes_posix_fsync_fuse_tolerance_and_orphan_cleanup`: verifies POSIX file + parent-directory `fsync`, FUSE `EINVAL`/`EOPNOTSUPP` tolerance, `EIO` propagation, and mid-write temp-file cleanup leaving target intact.
- `test_v7_package_fingerprint_completeness_tokenizers_and_no_installed_fallback`: verifies all 19 runtime-critical distributions (including `tokenizers`, `packaging`, `huggingface_hub`, `protobuf`, `normalizer`, `unicodedata2`, `ftfy`, `emoji`), `importlib.metadata` version resolution with zero `"installed"` fallback, and fail-closed pre-lease rejection on unresolved `tokenizers` or wrong `normalizer_commit`.

---

## 6. Portable-Kit Manifest & Changed-File Inventory (v7)

### Portable-Kit Version Lineage (All Five Independently Verified; Historical Kits v3–v6 Untouched)
- **`v3`** (untouched): `manifest_sha256 = 2d720c4e835525f0825fd811c06b867b7ddc8e2ceb850a23fb63494719607bc2`
- **`v4`** (untouched): `manifest_sha256 = 8f1b44e828b08afd68b82891d7f2ea7b0fa074d82784d0fd63bcf71c422c0c07`
- **`v5`** (untouched): `manifest_sha256 = dfdbd65a310ae68c75947df3fc12b0bf9defba08145e1472de82f9d6b82bc419`
- **`v6`** (untouched): `manifest_sha256 = e0f9757a4dab63e7964762f0033577a6fa3826b771c81e0f1353db3b7fd45f0d`
- **`v7`** (latest verified): `manifest_sha256 = df8eb6e2b0862f8ae642b96c7402b7209559c6a078bbd4c17f0823faf2c3ab19` (`stage4_kit_manifest.json` file SHA256: `38d6a9efb951d570fd2e40643dcd8bb253e215dcd5316a8ae6b0e59308ace0b8`)
  - `src/bangla_smish/post_stage_b_checkpointing.py` SHA256: `b4014a7c26cc6cde98e5f84c736da70adba0576fe2018d5e848109e02b931201`
  - `src/bangla_smish/post_stage_b_runner.py` SHA256: `815bc31fe64bd5071573cfdae1bb46bd7f28c83242d121c32e4c79403865aa60`
  - `experiments/colab/post_stage_b_stage4_experiments.ipynb` SHA256: `8b52b0ba06ebb21259f207cfed3a9bf57919b4fb78cff7a25ae3d2d251bfec41`
  - Only `post_stage_b_checkpointing.py`, `post_stage_b_runner.py`, and `post_stage_b_stage4_experiments.ipynb` changed vs `v6`; all 31 other runtime files (configs, datasets, frozen masks, statistical validation JSON, other modules) remain bit-for-bit identical.

### Complete Modified / Added File Inventory (v7)
| Action | Path | Notes |
| :--- | :--- | :--- |
| Modified | [`src/bangla_smish/post_stage_b_checkpointing.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_checkpointing.py) | Added fail-closed `ensure_unit_completion_event()` |
| Modified | [`src/bangla_smish/post_stage_b_runner.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py) | Findings 1–6 + Requirements A–D (`_validate_pre_run_compatibility`, T4 hardware/driver fingerprint & strict resume binding, `_reconcile_committed_unit_post_commit`, strict expected-vs-actual step check & `transformers==4.57.1` disclosure, hardened PID+UUID atomic writes + best-effort dir `fsync`, 19-dist `importlib.metadata` package fingerprint including `tokenizers`) |
| Modified | [`scripts/generate_post_stage_b_stage4_notebook.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/scripts/generate_post_stage_b_stage4_notebook.py) | Cell 0/1/2 updates (`v7` defaults, strict single-GPU Tesla T4 + CC `7.5` + `nvidia_driver_version` gate for `t4_benchmark`, `gpu_smoke_test` unchanged) |
| Regenerated | [`experiments/colab/post_stage_b_stage4_experiments.ipynb`](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/post_stage_b_stage4_experiments.ipynb) | Canonical output of `generate_post_stage_b_stage4_notebook.py` |
| Modified | [`tests/test_stage4_final_remediation.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/tests/test_stage4_final_remediation.py) | Updated v7 defaults & added 6 dedicated v7 test groups (`14/14` passing) |
| Created | [`stage4_colab_kit_upload/bangla_smish_post_stage_b_stage4_v7/`](file:///home/bs01415/Documents/Programming/ExploringAIs/stage4_colab_kit_upload/bangla_smish_post_stage_b_stage4_v7) | New versioned portable Colab kit (`v3`, `v4`, `v5`, `v6` untouched) |
| Updated | [`reports/post_stage_b_stage4_full_notebook_preflight_audit.md`](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/post_stage_b_stage4_full_notebook_preflight_audit.md) | This full audit & v7 remediation report |
| Updated | [`CHECKPOINT.md`](file:///home/bs01415/Documents/Programming/ExploringAIs/CHECKPOINT.md) | Prepended Stage 4 `v7` checkpoint entry |

---

## 7. Remaining Risks & Unresolved Live GPU / Colab Limitations (Independent-Review Gate)

**Status:** **STOPPED AT v7 INDEPENDENT-REVIEW GATE.** No T4 benchmark or 160-unit Option A training is authorized.

1. **Intentional Fail-Closed T4 Benchmark Configuration (Decision 2 — Option A):**
   Under Decision 2 Option A, `post_stage_b_stage4_t4_benchmark_config.json` still has `benchmark_workload_fixture_eval: true` while `SYNTHETIC_WORKLOAD_FIXTURE_ENABLED = False`. Attempting to run `t4_benchmark` on Colab will fail closed BEFORE acquiring a lease, initializing `run_state.json`, or training until a future owner authorization approves an unmodified dev-data-only benchmarking alternative.
2. **Unverified Live Colab / Tesla T4 GPU Environment (Mocked CPU Tests Cannot Verify These):**
   - Real HuggingFace Hub downloads of all 4 pinned commit SHAs (`9ce791f...`, `afd9f36...`, `e73636d...`, `45c032a...`) confirming both `AutoTokenizer` and `AutoModelForSequenceClassification` expose `_commit_hash` matching the pinned commit.
   - Real `transformers>=4.57.1,<5` `Trainer` FP16 execution on Tesla T4 (confirming `scaler.pt` presence on `on_save`, `actual_global_step == expected_optimizer_steps`, T4 15 GB VRAM headroom, and whether FP16 `GradScaler` overflow skips occur — note that `successful_optimizer_weight_updates` is currently `None`/uninstrumented).
   - Real Colab `nvidia-smi` output format & driver stability across Colab runtime reconnections (under Decision 3 STRICT binding, any Colab host driver upgrade between sessions will reject resume and require a fresh `ARTIFACT_RUN_ID`).
   - Real Google Drive FUSE durability, readback latency, rename/fsync error codes (`EINVAL`/`EOPNOTSUPP`), and `post_stage_b_colab_deps.py` install/restart flow with live PyTorch CUDA + `tokenizers` + `csebuetnlp/normalizer`.
   - `post_stage_b_tokenizer_gate.py#L141-L145` still retains `or expected_rev` fallback if an HF tokenizer omits `_commit_hash` (strictly closed inside `hf_train_and_predict_fold`, which fails closed if `_commit_hash` is absent).
