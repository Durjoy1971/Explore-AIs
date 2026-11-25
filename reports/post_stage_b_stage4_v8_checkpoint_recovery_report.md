# Post-Stage B Stage 4 v8 Checkpoint Recovery, Lease & Persistence Report

**Date:** 2026-11-25  
**Modules Hardened:** [`src/bangla_smish/post_stage_b_checkpointing.py`](../src/bangla_smish/post_stage_b_checkpointing.py), [`src/bangla_smish/post_stage_b_runner.py`](../src/bangla_smish/post_stage_b_runner.py), [`src/bangla_smish/post_stage_b_cohorts.py`](../src/bangla_smish/post_stage_b_cohorts.py), [`src/bangla_smish/post_stage_b_tokenizer_gate.py`](../src/bangla_smish/post_stage_b_tokenizer_gate.py).

---

## 1. Checkpoint Recovery & Retry Integrity (HIGH)

1. **Attempt 2+ Checkpoint Survival After Invalidated Attempt 1 `UNIT_COMMITTED`:**
   - `_read_unit_commit_events` and `scan_unit_staging` now accept `active_attempt` and `trust_commit_events`.
   - During active training (`on_save`) and resume restoration, `hf_train_and_predict_fold` calls `scan_unit_staging(..., active_attempt=int(attempt or 1), trust_commit_events=False)`. Historical `UNIT_COMMITTED` events from an earlier invalidated attempt (`attempt < active_attempt`) can no longer classify Attempt 2+ checkpoints as `ALREADY_SUPERSEDED` or cause `prune_superseded_staging` to delete live recovery checkpoints.
   - Post-commit cleanup (`_reconcile_committed_unit_post_commit`) passes `unit_already_committed=True`, `active_attempt=int(commit["attempt"])`, and `trust_commit_events=True` only AFTER durable `run_state.json` commit.
2. **Elimination of Silent Local-Only Fallback After Failed Drive Staging:**
   - When `staging_unit_dir is not None`, `hf_train_and_predict_fold` recovers strictly from verified Drive staging and NEVER silently falls back to `_find_latest_trainer_checkpoint` on local SSD.
   - Local-only `_find_latest_trainer_checkpoint(..., require_scaler=use_fp16)` is used strictly when `staging_unit_dir is None`.
3. **Cross-VM Best-Checkpoint Restoration & Durable Retention:**
   - During `Stage4CheckpointCallback.on_save`, if `state.best_model_checkpoint` references an earlier local checkpoint not yet present as `VALID_RECOVERABLE` in Drive staging, that best checkpoint is synced to staging BEFORE syncing the latest epoch checkpoint.
   - `_verify_trainer_checkpoint` now accepts relocated `restored_<hex6>_checkpoint-<step>` directory names.
   - When `keep_model_weights != "none"` and `local_checkpoint_root` is set, `_persist_durable_trainer_checkpoint` copies the best trainer checkpoint atomically into `artifacts_dir/checkpoints/<uid>/<attempt>/` with SHA256 verification and tags `"storage": "artifacts_dir"` so cross-VM resume and offline CPU aggregation succeed after ephemeral `/content/tmp_stage4_ckpt` disappears.
4. **FP16 `scaler.pt` Validation:**
   - `_valid_checkpoint`, `_find_latest_trainer_checkpoint`, `_verify_trainer_checkpoint`, and `_read_committed_unit` now enforce `require_scaler=True` when FP16 is active (`use_fp16=True`), ensuring corrupt/missing `scaler.pt` checkpoints are rejected consistently on both local SSD and Drive staging.

---

## 2. Single-Writer Lease Safety (HIGH)

1. **Same-Host Live PID Guard:** `register_single_writer_lease` checks `_pid_is_alive_on_local_host(pid)` (`os.kill(pid, 0)`) when `hostname == socket.gethostname()` and `pid != os.getpid()`, failing closed even if `allow_stale_takeover=True` was passed.
2. **Empty-Active Takeover Guard:** Passing non-empty `expected_stale_writer_ids` when zero active leases exist now fails closed instead of silently ignoring mismatched operator intent.
3. **Post-Write TOCTOU Guard:** Immediately after writing `lease_*.json`, `register_single_writer_lease` calls `verify_writer_lease_still_sole_active(artifacts_dir, lease_token)` without releasing the lease on conflict, preserving both conflicting lease files on disk.
4. **Mid-Run Ownership Checks & Conflict-Preserving `finally:`:**
   - `verify_writer_lease_still_sole_active` now runs before unit start, inside `Stage4CheckpointCallback.on_save` (via `lease_check_fn`), after training before artifact writes, immediately before durable `run_state.json` commit, inside `_reconcile_committed_unit_post_commit`, and before `_aggregate_verified`.
   - `_run_locked`'s `finally:` block inspects leases first and ONLY calls `release_single_writer_lease` if our `writer_id` is STILL the sole active writer with zero unacknowledged corrupt files — preventing conflict evidence from being erased on supersession and preventing Drive disconnect `OSError`s from masking the primary exception.

---

## 3. Resume Timing Integrity (HIGH)

- Each epoch checkpoint now embeds `stage4_timing.json` (`STAGE4_TIMING_METADATA`, SHA256-protected inside both `_CHECKPOINT_COMMIT` and `stage4_checkpoint_manifest.json`) recording cumulative elapsed train/pure-opt/eval/checkpoint/sync seconds up to `state.global_step`.
- On resume, `hf_train_and_predict_fold` loads `resumed_from_step` and prior cumulative seconds from the restored checkpoint and records:
  - `attempt_train_seconds`
  - `prior_resumed_train_seconds`
  - `cumulative_train_seconds` (also exported as `train_seconds`)
  - `resumed_from_step` and `steps_executed_this_attempt`
  - `attempt_optimizer_step_ms` (attempt pure-opt time divided ONLY by `steps_executed_this_attempt`)
  - `optimizer_step_ms` (cumulative pure-opt time divided by `actual_global_step`, or `attempt_optimizer_step_ms` if resuming a legacy checkpoint without `stage4_timing.json`)

---

## 4. Persistence, FUSE Atomicity & Out-of-Order Upload Safety (MEDIUM)

- **FUSE `ENOSYS` Tolerance:** Added `errno.ENOSYS` alongside `EINVAL`/`EOPNOTSUPP`/`EPERM`/`EBADF` in `_FUSE_FSYNC_IGNORED_ERRNOS` and `_fsync_fd_best_effort` for both file and directory `fsync` across `atomic_write_text`, `_write_prediction_archive`, `_atomic_write_text_local`, `_atomic_write_gate_json`, `_copy_file_verified_best_effort_fsync`, and `export_frozen_test_masks` while still propagating real `EIO`/`ENOSPC` failures fail-closed.
- **Orphan `.tmp` Safety:** `_checkpoint_file_hashes` and `_collect_local_ckpt_files` ignore orphan `.tmp.<pid>.<hex8>` files; `_read_state` removes ONLY exact orphan `.run_state.json.tmp.<pid>.<hex8>` files in uninitialized directories while removing dead `.runner.lock` whitelisting and preserving strict rejection of `exports/` or any stray file.
- **`run_state.backup.json` & Durable `metrics.json` Readback:** `_write_run_state` writes the prior valid `run_state.json` to `run_state.backup.json` before atomic replacement and verifies JSON readback; `_execute_units` reads back `metrics.json` from disk and verifies equality before `_validate_unit` and commit SHA256 recording.
- **Out-of-Order Async Drive Upload & Last-Checkpoint Safety:** `scan_unit_staging` requires a weight file in every manifest, classifies manifest-before-payload missing/short files without a `CHECKPOINT_SYNCED` event as `INCOMPLETE_UPLOAD` (not `CORRUPT_OR_UNVERIFIABLE`), and refuses to supersede older valid checkpoints until the newer replacement has a verified `CHECKPOINT_SYNCED` event.
