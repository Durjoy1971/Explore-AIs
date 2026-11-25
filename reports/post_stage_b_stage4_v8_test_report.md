# Post-Stage B Stage 4 v8 Test & CPU-Cap Verification Report

**Date:** 2026-11-25  
**Host Environment:** Linux, Python `3.12.3` (`.venv`), `12` logical CPUs, hard kernel cap `<= 25%` (`systemd-run --user --scope -p CPUQuota=300% taskset -c 0-2` + `scripts/run_stage4_capped.py`).

---

## 1. Exact Verification Commands & Measured Results

All verification commands executed sequentially under the hard `<= 25%` host CPU cap (`cgroup_quota_cpus = 3.0`, `max_allowed_cpus = 3`, `parent_affinity = [0, 1, 2]`, `child_affinity = [0, 1, 2]`):

| # | Command (wrapped in `systemd-run --user --scope -p CPUQuota=300% taskset -c 0-2 .venv/bin/python scripts/run_stage4_capped.py -- ...`) | Exit Code | Result | Wall (s) | Effective Host CPU % | Throttled Delta |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| 1 | `.venv/bin/python -m unittest tests/test_stage4_v8_remediation.py tests/test_stage4_final_remediation.py -v` | `0` | **26/26 OK** (`10.182s`) | `11.530` | `9.59%` | `0` |
| 2 | `.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v` | `0` | **112/112 OK** (`118.954s`) | `120.714` | `9.30%` | `6` |
| 3 | `.venv/bin/python tests/test_pipeline.py` | `0` | **PASS** | `1.205` | `11.96%` | `1` |
| 4 | `.venv/bin/python scripts/verify_stage4_integrity.py --verify-only` | `0` | **PASS** (`282/282` Cat A, `13/13` inputs, `v1`/`v2`, `4/4` masks) | `19.025` | `8.30%` | `0` |
| 5 | Portable kit verifier (`v3`–`v8`) + `v8` `34/34` repo-to-kit SHA256 parity | `0` | **PASS** (`6/6` kits, `34/34` parity, `0` mismatches) | `1.684` | `10.94%` | `0` |

Every run stayed strictly below the `25.0%` host CPU ceiling (`8.30%–11.96%`).

---

## 2. Stage 4 v8 Scenario Coverage (`tests/test_stage4_v8_remediation.py`)

| Scenario | Test Method | Evidence Type | Key Assertions Verified |
| :--- | :--- | :--- | :--- |
| **A (P0)** | `test_scenario_a_tokenizer_revision_verification_and_gate_binding` | Real `huggingface_hub.try_to_load_from_cache` + Mock tokenizer (NO `_commit_hash`) | Passes on snapshot `vocab_file` and real temp HF cache; fails closed on missing cache, mismatched SHA, non-snapshot local file, redirected `name_or_path`, tampered `bundle_manifest_sha256`, or mismatched `tokenizer_package_versions`. |
| **B (P0)** | `test_scenario_b_dependency_closure_real_dist_info_no_unicodedata2` | Real `importlib.metadata.PathDistribution` on temp `.dist-info` dirs | Zero `unicodedata2` anywhere; Cell 4 `preflight_colab_dependencies` & Cell 6 `_validate_required_packages_fingerprint` pass on exact 24-dist closure; removing `wcwidth`/`regex`/`tokenizers` fails closed in both Cell 4 and Cell 6. |
| **C (HIGH)** | `test_scenario_c_attempt2_checkpoint_survives_historical_attempt1_commit_event` | Real filesystem checkpointing | Historical Attempt 1 `UNIT_COMMITTED` event does not supersede or prune Attempt 2 checkpoint; Attempt 3 restores Attempt 2 cleanly. |
| **D (HIGH)** | `test_scenario_d_no_silent_local_fallback_when_drive_staging_configured` | CPU-mocked `Trainer` + real filesystem | Unstaged local Attempt 1 checkpoint is NOT silently resumed when Drive staging is configured and empty. |
| **E & F (HIGH)** | `test_scenarios_e_and_f_cross_vm_best_restore_and_fp16_scaler_validation` | Real filesystem checkpointing | `_valid_checkpoint`/`_find_latest_trainer_checkpoint` enforce `require_scaler=True`; VM wipe + restore recovers both latest and best checkpoints with valid `_verify_trainer_checkpoint` on `restored_<hex6>_checkpoint-*`. |
| **G (HIGH)** | `test_scenario_g_lease_safety_live_pid_toctou_and_conflict_preservation` | Real OS PID (`os.getppid()`) + filesystem | Live same-host PID blocks stale takeover; non-empty `expected_stale_writer_ids` with 0 active leases fails closed; TOCTOU race and mid-run conflict abort before commit and preserve both active lease files (`finally:` does not auto-release). |
| **H (HIGH)** | `test_scenario_h_resume_timing_accuracy` | CPU-mocked `Trainer` + real `stage4_timing.json` | Resuming at step `10` and running `5` steps separates `attempt_train_seconds`, `prior_resumed_train_seconds=20.0`, `cumulative_train_seconds>=20.03`, `steps_executed_this_attempt=5`, `attempt_optimizer_step_ms`, and `optimizer_step_ms`. |
| **I (MED)** | `test_scenario_i_persistence_fuse_enosys_orphan_tmp_backup_and_upload_safety` | Real POSIX + injected `errno.ENOSYS` | Tolerates `ENOSYS` on file/dir fsync; cleans orphan `.run_state.json.tmp.*` while still rejecting stray `exports/`; creates `run_state.backup.json`; classifies out-of-order upload without `CHECKPOINT_SYNCED` as `INCOMPLETE_UPLOAD`; refuses to prune last valid checkpoint before replacement `CHECKPOINT_SYNCED`. |
| **J (MED)** | `test_scenario_j_identity_completeness_strict_config_and_bootstrap_seed` | Real bundle & metrics | `schema_version == 5` hashes all 18 kit modules; rejects traversal `run_name`, bool/negative `seed`, bool `folds`, and altered `confirmatory_mode` hyperparams/flags; verifies `seed` propagation in `evaluate_predictions` and `paired_family_accuracy_delta`. |
| **K (MED)** | `test_scenario_k_multi_gpu_rejected_all_modes_and_smoke_pkg_check` | CPU-mocked hardware/env | Rejects `gpu_count=2` before lease/state in `gpu_smoke_test`, `t4_benchmark`, and `experiments_abcd`; rejects missing packages in `gpu_smoke_test` real-HF mode before lease. |
| **L (MED)** | `test_scenario_l_offline_cpu_aggregation_and_durable_kept_checkpoint` | CPU-mocked GPU -> real offline CPU | Durable `keep_model_weights="all"` checkpoint persists in `artifacts_dir/checkpoints/` after ephemeral local SSD wipe; offline CPU `aggregate_post_stage_b_results(..., offline_cpu_ok=True)` succeeds while still rejecting config/source drift. |
| **M** | `test_scenario_m_frozen_t4_config_untouched_and_kit_os_metadata` | Real frozen config & kit verifier | Frozen T4 config remains `benchmark_workload_fixture_eval=True`, fail-closed in both pipeline and direct `hf_train_and_predict_fold`; kit verifier ignores `Thumbs.db` and `._*`. |

---

## 3. Scope Limitations (Honest Boundary)

- **112/112 local tests PASS** covers all local deterministic logic, real `importlib.metadata` and `huggingface_hub` cache resolution, real POSIX filesystem atomicity, and real `v2` bundle/mask/kit integrity.
- **NOT verified locally:** Real `torch` CUDA kernels, real `transformers==4.57.1` model weight download and `Trainer` FP16 `GradScaler` behavior, live Google Drive FUSE mount latency, and real HuggingFace Hub 32-pair tokenization on the 4 backbones.
