# Stage 4 — Five-Unit T4 Benchmark Local Kit Preparation & Colab Execution Handoff

**Execution Status:** **LOCALLY STAGED & VERIFIED — BLOCKED ON REMOTE COLAB / GOOGLE DRIVE EXECUTION**  
**Honest Environment Disclosure:** This local Linux workstation has **no GPU** (`nvidia-smi: command not found`) and **no Google Colab or Google Drive CLI/API access** (`rclone`/`gdrive`/`colab` unavailable). In accordance with strict restrictions (no fake execution, no local CPU training substitute, no unauthorized source edits), this agent did not run Colab/Drive remotely and did not train models on CPU.

---

## 1. Completed Local Preflight Preparation (Hardware 25% CPU Cap Enforced)

Executed under `systemd-run --user --scope -p CPUQuota=300% taskset -c 0-2` wrapped by [`scripts/run_stage4_capped.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/scripts/run_stage4_capped.py):
- **CPU Enforcement Telemetry:** `cpu.max=300000 100000` (`3.0` CPUs / `12` host logical CPUs), `parent_affinity=[0, 1, 2]`, `child_affinity=[0, 1, 2]`, `cgroup_nr_throttled_delta=0`, `effective_host_cpu_pct=10.27`, `returncode=0`.
- **Fresh Staged 34-File Portable Colab Kit:** [`stage4_colab_kit_upload/bangla_smish_post_stage_b_stage4_v3/`](file:///home/bs01415/Documents/Programming/ExploringAIs/stage4_colab_kit_upload/bangla_smish_post_stage_b_stage4_v3) (`15M` total size on disk)
  - `stage4_kit_manifest.json` SHA256: `b27dfbd5db27b4312e76fc90e3a265ad899cd484fe75ce934be35378c49fd258`
  - `prepare_stage4_colab_kit.py --verify-only`: `verified_runtime_file_count = 34`, `cross_checked_baseline_v2_files = 8`, `reverified_frozen_masks = 4`, `verifies_full_repository_282 = false`, `statistical_validation_verdict = FAIL_OR_INCONCLUSIVE_CONFIRMATORY_DISABLED`
  - Config & Notebook Digests: `t4_benchmark_config = 03a5f10d433f657af82d1af401774fd40d9d36209d35e13a14bc934086416180`, `stage4_notebook = 53b35f508f9a4c783ba951e409c0cf324b9992732f629b65ac06eb587f50a219`, `bundle_manifest_sha256 = 308d1c889840c1ec04e1cd6f6751cccfca0e559f2e12c61e18ff2e54d6fa2922`
- **Full Repository Integrity Verification (`verify_stage4_integrity.py --verify-only --skip-tfidf`):**
  - `protected_input_count: 13`, `protected_baseline_282.verified_count: 282`, `v1` & `v2` bundles intact (`0` normalization mismatches), `4/4` frozen test masks verified.

---

## 2. Exact Owner Google Colab T4 Runbook (Five Authorized Units Only)

1. **Upload Isolated Kit to Drive:** Upload [`stage4_colab_kit_upload/bangla_smish_post_stage_b_stage4_v3`](file:///home/bs01415/Documents/Programming/ExploringAIs/stage4_colab_kit_upload/bangla_smish_post_stage_b_stage4_v3) cleanly to `/MyDrive/bangla_smish_post_stage_b_stage4_v3` (verify this folder did not exist previously; never overwrite historical folders).
2. **Capture Starting Balances:** Record your starting Google Colab Compute Units (CU) and free Google Drive GB storage before launching.
3. **Open Notebook on a Single Tesla T4 GPU Runtime:** Open `MyDrive/bangla_smish_post_stage_b_stage4_v3/experiments/colab/post_stage_b_stage4_experiments.ipynb`.
4. **Set Exact Cell 1 Parameters:**
   ```python
   RUN_MODE = 't4_benchmark'
   AUTHORIZE_T4_BENCHMARK = True
   AUTHORIZE_FULL_TRAINING = False
   REQUIRE_GPU = True
   USE_GOOGLE_DRIVE = True
   DRIVE_WORKSPACE_REL = 'bangla_smish_post_stage_b_stage4_v3'
   ARTIFACT_RUN_ID = 'stage4-t4-bench-20261124-01'
   RESUME = True
   ALLOW_STALE_WRITER_TAKEOVER = False
   EXPECTED_STALE_WRITER_IDS = ''
   EXPECTED_CORRUPT_LEASE_FILES = ''
   STALE_TAKEOVER_REASON = ''
   ```
5. **Run Preflight Cells Strictly Before Training (Cells 1 -> 2 -> 3 -> 4 -> 5):**
   - **Cell 2:** Confirm Tesla T4 GPU (~`15 GB` VRAM), `torch.cuda.is_available() == True`, and free `/content` SSD.
   - **Cell 3:** Mount Drive and verify `WORKSPACE_ROOT`, `BUNDLE_DIR`, and `post_stage_b_stage4_t4_benchmark_config.json`.
   - **Cell 4:** Install/verify dependencies preserving CUDA PyTorch + `normalizer`. If prompted to restart runtime, restart once and re-run Cells `1 -> 4`.
   - **Cell 5:** Confirm `verify_portable_colab_kit`, `verify_manifest`, and `run_allowed_view_tokenizer_collision_gate` (`canon_masked` across all 4 pinned HF commits) pass with `0` collisions. **STOP immediately before Cell 6 if any check fails.**
6. **Execute Five Authorized Units & Inspect Outputs (Cells 6 -> 7):**
   - Cell 6 executes ONLY the 5 authorized Fold-0 `canon_masked` units (`D` × 4 backbones at `363` steps + `B` `xlm-roberta-base` at `300` steps = `1,752` expected steps; `run_heldout_eval=False`; `CONFIRMATORY_DECLARATIONS_ENABLED=False`).
   - If any unit fails, stop and preserve all logs, lease records, and staging/checkpoint files without deleting evidence.
7. **Post-Run Collection:** Record ending Colab CU balance and Drive quota / Drive Trash usage; download `MyDrive/bangla_smish_post_stage_b_stage4_v3/artifacts/t4_benchmark/stage4-t4-bench-20261124-01/` (`exports/`, `logs/runner.log`, `run_state.json`, `leases/`, `events/`, `pointers/`, `units/*/attempts/*/metrics.json`) into the local workspace so an independent post-benchmark audit report covering all 5 units can be produced.

---

## 3. Restrictions Compliance Attestation
- Zero modifications to source code, frozen configurations, datasets, or the 282 protected files.
- Zero Git commits, pushes, merges, resets, or discards.
- Zero synthetic generation, zero held-out/OOD evaluation, and full 160-unit training remains strictly **`NO-GO`**.
