# Post-Stage B — Stage 4 Final Remediation Report

**Date:** 2026-11-24  
**Repository / Branch / Base Commit:** `ExploringAIs` · `main` · `4d341183cb7a6fce0ed773ff33da2a3b757d6f94`  
**Scope:** Category-C resolution of every remaining finding from the Stage 4 Independent Acceptance Review (`reports/post_stage_b_stage4_independent_acceptance_review.md`), verified locally under a kernel-enforced `<= 25%` CPU limit. Zero dataset edits, zero Category A modifications (`282/282` preserved), zero statistical simulation reruns, zero model training, zero Colab execution, zero Google Drive operations, zero Git history mutations.  
**Statistical Governance Status:** **`FAIL_OR_INCONCLUSIVE_CONFIRMATORY_DISABLED`** (`CONFIRMATORY_DECLARATIONS_ENABLED = False`; `reports/stage4_statistical_validation_results.json` preserved byte-for-byte with SHA256 `bde8085b77066cbf449a1b17d0a1afa77f2dcbcb4ea1a82f7d0dde68f6fe504f`).  
**Stop Condition:** Stopped cleanly at the Stage 4 independent-review gate. Five-unit T4 benchmark still requires separate explicit owner authorization; full 160-unit training remains strictly **`NO-GO`**.

---

## 1. Finding-by-Finding Resolution Summary

| Finding | Implemented Resolution | Local Verification Status |
|---|---|---|
| **1. CPU Compliance (`<= 25%` Hard Limit)** | Created [run_stage4_capped.py](file:///home/bs01415/Documents/Programming/ExploringAIs/scripts/run_stage4_capped.py) failing closed before any tests run unless `len(sched_getaffinity) <= 3` (25% of 12 CPUs), cgroup v2 `cpu.max <= 3.0` CPUs (`300000 100000`), and probed child subprocess inherits exact `[0, 1, 2]` affinity; records `cpu.stat` `usage_usec` & `rusage`. | **ENFORCED & VERIFIED** (`CPUQuota=300%` + `taskset -c 0-2`: `8.24%` targeted, `8.33%` full suite) |
| **2. Allowed-View 32-Pair Tokenizer Gate** | Pinned all 4 HF tokenizer commits in `option_a` & `t4_benchmark` configs and `PostStageBConfig.tokenizer_revisions` ([post_stage_b_runner.py#L183-L192](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L183-L192)); created [post_stage_b_tokenizer_gate.py](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_tokenizer_gate.py) freezing exact Stage 3 pairs `P01..P32`; integrated fail-closed gate into Notebook Cells 5 & 6. | **MOCKED GATE PASSED (`256/256`); LIVE COLAB TOKENIZERS `NOT VERIFIED`** |
| **3A/B/C. Strict Writer Lease Takeover & Audit** | `register_single_writer_lease` ([post_stage_b_checkpointing.py#L193-L315](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_checkpointing.py#L193-L315)) now unconditionally requires exact non-empty `expected_stale_writer_ids` even for a single stale lease AND `expected_corrupt_files` for any corrupt lease/release files; Notebook Cell 1 enforces both; all lease/takeover/corrupt files remain append-only. | **VERIFIED LOCALLY** *(live Drive FUSE remains Colab-only)* |
| **3D. Local `_CHECKPOINT_COMMIT` Refresh** | `_refresh_local_ckpt_commit_file` ([post_stage_b_checkpointing.py#L666-L673](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_checkpointing.py#L666-L673), [#L781](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_checkpointing.py#L781)) refreshes `post_stage_b_checkpoint_commit.json` strictly inside the restored local SSD directory after relocating `trainer_state.json`, keeping Drive staging untouched. | **VERIFIED LOCALLY** (`_valid_checkpoint` passes post-relocation) |
| **3E/F. Surface `prune_errors` & Retention** | `hf_train_and_predict_fold` & `_execute_units` log `prune_errors` warnings (`on_save`, `post_restore`, `post_commit`) and persist `checkpoint_prune_errors` into unit `metrics.json`; retention never removes the sole recoverable checkpoint or distinct pinned best checkpoint. | **VERIFIED LOCALLY** |
| **4A/B. Portable Kit OS Metadata & Mask Re-Derivation** | `_enumerate_kit_files` ignores only `.ipynb_checkpoints/`, `.DS_Store`, and `desktop.ini` (still rejecting rogue files); `verify_portable_colab_kit` now imports `build_smishdetect_test_cohort_masks` to **re-derive and verify all 4 frozen test masks live** from the staged parquet + enforces `tokenizer_revisions` across configs ([post_stage_b_integrity.py#L384-L645](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_integrity.py#L384-L645)). Now verifies `34` runtime files cleanly. | **VERIFIED IN ISOLATED SUBPROCESS** *(tampered masks & rogue files rejected)* |
| **4C/D. Transitive Deps & `pyproject.toml` `[gpu]`** | Added `packaging>=23.0`, `huggingface_hub>=0.24,<2`, `protobuf>=3.20`, documented `BANGLABERT_NORMALIZER_TRANSITIVE_DEPS`, verified callable `normalizer.normalize`, and reconciled `pyproject.toml` `[gpu]` extras (`torch>=2.6`, `transformers>=4.57.1,<5`, `safetensors>=0.4`, `sentencepiece>=0.2,<1`, `protobuf>=3.20`, `packaging>=23.0`). | **MOCKED & STATICALLY VERIFIED** *(live Colab pip install `NOT VERIFIED`)* |

---

## 2. Actual Kernel CPU Enforcement & Telemetry Evidence

Both targeted and full verification runs were launched inside a dedicated systemd user scope with cgroup v2 `CPUQuota=300%` (`3.0 / 12` logical CPUs = `25.0%` hard host capacity ceiling) and `taskset -c 0-2`, wrapped by [run_stage4_capped.py](file:///home/bs01415/Documents/Programming/ExploringAIs/scripts/run_stage4_capped.py) which failed closed unless parent affinity, child subprocess affinity inheritance, and numeric `cpu.max` were verified prior to starting:

1. **Pre-Test Hard Cap Probe (`CAP_PROBE_OK [0, 1, 2]`):**
   - `cgroup_cpu_max`: `/sys/fs/cgroup/user.slice/user-1001.slice/user@1001.service/app.slice/run-p264647-i249542.scope/cpu.max=300000 100000` (`cgroup_quota_cpus = 3.0`, `host_logical_cpus = 12`)
   - `parent_affinity`: `[0, 1, 2]`, `child_affinity`: `[0, 1, 2]`
   - `effective_host_cpu_pct`: `8.61%`
2. **Targeted Stage 4 Suite (`27 tests in 10.303s`, `OK`):**
   - `cgroup_cpu_max`: `.../run-p264646-i279032.scope/cpu.max=300000 100000`
   - `parent_affinity`: `[0, 1, 2]`, `child_affinity`: `[0, 1, 2]`
   - `wall_seconds`: `11.883`, `cgroup_usage_seconds`: `11.755`, **`effective_host_cpu_pct`: `8.24%`** (`<= 25%`)
3. **Complete Regression, Pipeline, Bundles, 282-File & Isolated Kit Suite:**
   - `cgroup_cpu_max`: `.../run-p265279-i249604.scope/cpu.max=300000 100000`
   - `parent_affinity`: `[0, 1, 2]`, `child_affinity`: `[0, 1, 2]`
   - `wall_seconds`: `134.2`, `cgroup_usage_seconds`: `134.122`, `rusage_children_cpu_seconds`: `134.121`, **`effective_host_cpu_pct`: `8.33%`** (`<= 25%`, `returncode = 0`)

---

## 3. Allowed-View 32-Pair Tokenizer Gate Results & Explicit `NOT VERIFIED` Boundaries

- **Implemented Gate ([post_stage_b_tokenizer_gate.py](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_tokenizer_gate.py)):** Resolves the exact `32` Stage 3 pairs (`P01..P32`) directly from `post_stage_b_bundle_v2` preserving original text (refusing to invent replacements on missing pairs), enforces `allowed_views <= ("raw", "canon_masked")`, enforces pinned 40-hex `tokenizer_revisions` (`9ce791f...`, `afd9f36...`, `e73636d...`, `45c032a...`), writes JSON evidence (`tokenizer_revisions`, `config_hash`, `pair_ids`, per-pair results), and fails closed in Notebook Cells 5 & 6 (`TOKENIZER_COLLISION_GATE_PASSED`).
- **Mocked Verification:** Passed `256 / 256` checks (`4` backbones $\times$ `2` views $\times$ `32` pairs) with mocked tokenizers, rejecting colliding inputs, wrong revisions, missing pairs, and `brand_masked`.
- **Live HuggingFace Tokenizer Execution Status:** **`NOT VERIFIED` locally** because `transformers` and `normalizer` are not installed locally and Colab execution is prohibited here. Actual execution on live pinned HuggingFace snapshots + `normalizer` remains gated on Colab Cell 5.

---

## 4. Checkpoint, Lease & Portable Kit Verification Evidence

- **Lease & Checkpoint Hardening:** Passed strict single-lease takeover rejection (`expected_stale_writer_ids` mandatory even for 1 active lease), corrupt lease/release `expected_corrupt_files` enforcement, append-only preservation, `_refresh_local_ckpt_commit_file` (`_valid_checkpoint` passes on relocated local restore while Drive staging remains untouched), `checkpoint_prune_errors` persistence into `metrics.json`, and retention safety (`limit=0, 1`).
- **Portable Kit CLI Verification (`post_stage_b_stage4_kit_v2`, `34` files):** Exported to an isolated `/tmp` kit directory and verified cleanly from `/tmp` with zero access to the repository root: `verified_runtime_file_count = 34`, `cross_checked_baseline_v2_files = 8`, `reverified_frozen_masks = 4`, `verifies_full_repository_282 = false`, `statistical_validation_verdict = "FAIL_OR_INCONCLUSIVE_CONFIRMATORY_DISABLED"`, `stage4_notebook_sha256 = 53b35f508f9a4c783ba951e409c0cf324b9992732f629b65ac06eb587f50a219`. Ignored `.DS_Store`, `desktop.ini`, and `.ipynb_checkpoints/` while still rejecting rogue files and live-mask tampering.

---

## 5. Exact Verification Commands, Outcomes & Protected Integrity

Executed under `systemd-run --user --scope -p CPUQuota=300% -- taskset -c 0-2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python scripts/run_stage4_capped.py -- ...`:
1. Targeted Stage 4 tests (`test_stage4_{final_remediation,remediation,checkpointing_and_kit,option_a_and_cohorts,ensemble_and_stats}.py`): **Exit `0` (`Ran 27 tests in 10.303s`, `OK`)**.
2. Complete repository `unittest` suite (`discover -s tests -v`): **Exit `0` (`Ran 90 tests in 118.437s`, `OK`)** (`63` retained + `18` Stage 4 + `5` prior remediation + `4` final remediation suites).
3. `tests/test_pipeline.py`: **Exit `0` (`All tests passed!`)**.
4. `scripts/prepare_post_stage_b_bundle.py --verify-only` on `v1` (`2e2e76e2...`) & `v2` (`308d1c88...`): **Exit `0`**.
5. `scripts/verify_stage4_integrity.py --verify-only --skip-tfidf`: **Exit `0` (`282/282` Category A protected files, `13/13` canonical inputs, `4/4` frozen test masks)**.
6. Isolated `/tmp` portable kit CLI export & verification (`prepare_stage4_colab_kit.py` + `verify_portable_colab_kit`): **Exit `0` (`34` runtime files, `4` re-derived masks)**.
7. `sha256sum reports/stage4_statistical_validation_results.json`: **`bde8085b77066cbf449a1b17d0a1afa77f2dcbcb4ea1a82f7d0dde68f6fe504f`** (untouched verbatim).

---

## 6. Modified & Added File Inventory (Stage 4 Final Remediation Pass)

- **Added Files (`4`):** `scripts/run_stage4_capped.py`, `src/bangla_smish/post_stage_b_tokenizer_gate.py`, `tests/test_stage4_final_remediation.py`, `reports/post_stage_b_stage4_final_remediation_report.md`.
- **Modified Category-C Files (`10`):** `pyproject.toml`, `experiments/colab/configs/post_stage_b_stage4_{option_a,t4_benchmark}_config.json`, `scripts/generate_post_stage_b_stage4_notebook.py`, `experiments/colab/post_stage_b_stage4_experiments.ipynb`, `src/bangla_smish/{post_stage_b_checkpointing,post_stage_b_colab_deps,post_stage_b_integrity,post_stage_b_runner}.py`, `tests/test_stage4_remediation.py`, `CHECKPOINT.md` *(plus ephemeral `response.md`)*. Zero Category A files modified.

---

## 7. Remaining Colab-Only Requirements (`NOT VERIFIED` Locally)

1. Live Colab GPU pip resolution & installation of `transformers>=4.57.1,<5`, `sentencepiece>=0.2,<1`, `protobuf>=3.20`, `csebuetnlp/normalizer@d405944...` + transitive packages (`unicodedata2`, `ftfy`, `emoji`, `urllib3`) preserving Colab CUDA `torch>=2.6`.
2. Live execution of Cell 5's `run_allowed_view_tokenizer_collision_gate` using actual pinned HuggingFace tokenizers (`9ce791f...`, `afd9f36...`, `e73636d...`, `45c032a...`) across all `32` Stage 3 pairs.
3. Live NVIDIA T4 VRAM (`xlm-roberta-base`, `muril` at `bs=32`), real HuggingFace `WeightedTrainer` + `Stage4CheckpointCallback` resume, live Google Drive FUSE staging/cleanup/Trash behavior, and empirical `CU/hr` burn rate (all gated behind separate owner authorization for the 5-unit T4 benchmark; full 160-unit training strictly **`NO-GO`**).
