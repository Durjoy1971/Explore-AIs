# Post-Stage B Stage 4 v8 Comprehensive Remediation Report

**Date:** 2026-11-25  
**Scope:** Local-only Stage 4 v8 remediation addressing all P0, High, and Medium findings from [`reports/post_stage_b_stage4_v7_independent_audit.md`](post_stage_b_stage4_v7_independent_audit.md).  
**Execution Status:** **STOPPED AT STAGE 4 v8 INDEPENDENT-REVIEW GATE.** Zero Colab execution, zero GPU execution, zero model training, zero synthetic generation, zero held-out/OOD model evaluation, zero Google Drive writes/deletions, zero Git commits, zero Category A edits (`282/282` preserved), zero frozen config/dataset/mask edits, and zero historical kit (`v3`–`v7`) edits.

---

## 1. Executive Summary

The read-only Stage 4 v7 independent audit returned **`FAIL`** after discovering two deterministic live-Colab P0 showstoppers (`transformers==4.57.1` popping `_commit_hash` from loaded tokenizers and an erroneous `unicodedata2` requirement that broke BanglaBERT startup), plus High/Medium defects in checkpoint recovery, lease safety, resume timing, FUSE persistence, identity hashing, config validation, multi-GPU rejection, and offline CPU aggregation.

Stage 4 v8 resolves all verified P0, High, and Medium findings across nine runtime source modules, the notebook generator/canonical notebook, `pyproject.toml`, and three test suites, stages a new versioned local portable Colab kit (`stage4_colab_kit_upload/bangla_smish_post_stage_b_stage4_v8/`), and verifies all changes locally under the kernel-enforced `<= 25%` host CPU cap (`effective_host_cpu_pct = 8.30%–11.96%`).

---

## 2. Finding-to-Code-to-Test Mapping

| Severity | v7 Audit Finding | Implemented Code Changes | Verified Tests |
| :--- | :--- | :--- | :--- |
| **P0-1** | `transformers==4.57.1` pops `_commit_hash` (`tokenization_utils_base.py:1930, 2121`); Cell 6 failed deterministically while Cell 5 silently fell back via `or expected_rev`. | [`src/bangla_smish/post_stage_b_tokenization.py`](../src/bangla_smish/post_stage_b_tokenization.py) (`extract_commit_hash_from_path`, `resolve_hf_cached_tokenizer_files`, `resolve_and_verify_tokenizer_revision`); [`src/bangla_smish/post_stage_b_tokenizer_gate.py`](../src/bangla_smish/post_stage_b_tokenizer_gate.py) (removed `or expected_rev`, unified fail-closed verifier, bound `bundle_manifest_sha256` & `tokenizer_package_versions`); [`src/bangla_smish/post_stage_b_runner.py`](../src/bangla_smish/post_stage_b_runner.py) (`hf_train_and_predict_fold`, `_validate_tokenizer_gate_report`). | `test_stage4_v8_remediation.py::test_scenario_a_*` (includes real `huggingface_hub.try_to_load_from_cache` on temp HF cache); `test_stage4_final_remediation.py::test_32_pair_*`. |
| **P0-2** | Bogus `unicodedata2` requirement in `_BANGLABERT_REQUIRED_PACKAGE_DISTS` & `BANGLABERT_NORMALIZER_TRANSITIVE_DEPS`; missing upstream `transformers==4.57.1` & `normalizer` transitive pins (`tokenizers`, `regex`, `wcwidth`, `pyyaml`, `requests`, `tqdm`, `filelock`, `huggingface_hub>=0.34.0,<1.0`, `safetensors>=0.4.3`). | [`src/bangla_smish/post_stage_b_colab_deps.py`](../src/bangla_smish/post_stage_b_colab_deps.py), [`pyproject.toml`](../pyproject.toml), [`src/bangla_smish/post_stage_b_runner.py`](../src/bangla_smish/post_stage_b_runner.py) (`_IDENTITY_PACKAGE_DISTS` 24 dists, `_CORE_REQUIRED_PACKAGE_DISTS` 20 dists, `_BANGLABERT_REQUIRED_PACKAGE_DISTS` 4 dists, zero `unicodedata2`). | `test_stage4_v8_remediation.py::test_scenario_b_*` (real `importlib.metadata.PathDistribution` over on-disk `.dist-info` closure). |
| **HIGH-1** | Attempt 2+ checkpoints after an invalidated Attempt 1 `UNIT_COMMITTED` event were misclassified as `ALREADY_SUPERSEDED` and pruned during `on_save`. | [`src/bangla_smish/post_stage_b_checkpointing.py`](../src/bangla_smish/post_stage_b_checkpointing.py) (`_read_unit_commit_events`, `scan_unit_staging` with `active_attempt` & `trust_commit_events`); [`src/bangla_smish/post_stage_b_runner.py`](../src/bangla_smish/post_stage_b_runner.py) (passes `active_attempt` and `trust_commit_events=False` during active training/restore). | `test_stage4_v8_remediation.py::test_scenario_c_*`. |
| **HIGH-2** | Silent fallback to unstaged local SSD checkpoint after failed Drive staging; cross-VM best-checkpoint restoration inconsistency. | [`src/bangla_smish/post_stage_b_runner.py`](../src/bangla_smish/post_stage_b_runner.py) (disables local fallback when `staging_unit_dir is not None`; stages earlier local `best_model_checkpoint` during `on_save` if absent from staging; `_verify_trainer_checkpoint` accepts `restored_<hex6>_checkpoint-<step>`). | `test_stage4_v8_remediation.py::test_scenario_d_*`, `test_scenarios_e_and_f_*`. |
| **HIGH-3** | `_valid_checkpoint` / `_find_latest_trainer_checkpoint` omitted `scaler.pt` check for FP16. | [`src/bangla_smish/post_stage_b_runner.py`](../src/bangla_smish/post_stage_b_runner.py) (`require_scaler` propagated through `_valid_checkpoint`, `_find_latest_trainer_checkpoint`, `_verify_trainer_checkpoint`, `_read_committed_unit`). | `test_stage4_v8_remediation.py::test_scenarios_e_and_f_*`. |
| **HIGH-4** | Lease acquisition TOCTOU race, same-host live PID takeover, missing mid-run lease checks (`on_save`/pre-commit/pre-prune/pre-export), and `finally:` auto-releasing conflicted leases. | [`src/bangla_smish/post_stage_b_checkpointing.py`](../src/bangla_smish/post_stage_b_checkpointing.py) (`_pid_is_alive_on_local_host`, empty-active `expected_stale_writer_ids` guard, post-write `verify_writer_lease_still_sole_active`); [`src/bangla_smish/post_stage_b_runner.py`](../src/bangla_smish/post_stage_b_runner.py) (`lease_check_fn`, pre-write/pre-commit/pre-reconcile/pre-export lease checks, conflict-preserving `finally:`). | `test_stage4_v8_remediation.py::test_scenario_g_*`. |
| **HIGH-5** | Resumed units divided single-attempt wall time by cumulative multi-epoch `global_step`, corrupting `train_seconds` and `optimizer_step_ms`. | [`src/bangla_smish/post_stage_b_checkpointing.py`](../src/bangla_smish/post_stage_b_checkpointing.py) (`STAGE4_TIMING_METADATA = "stage4_timing.json"`); [`src/bangla_smish/post_stage_b_runner.py`](../src/bangla_smish/post_stage_b_runner.py) (records `attempt_train_seconds`, `prior_resumed_train_seconds`, `cumulative_train_seconds`, `resumed_from_step`, `steps_executed_this_attempt`, `attempt_optimizer_step_ms`, `optimizer_step_ms`). | `test_stage4_v8_remediation.py::test_scenario_h_*`. |
| **MED-1** | FUSE `ENOSYS` on file/dir `fsync`, non-atomic helper writes, orphan `.tmp` handling, missing `metrics.json` readback, no `run_state.backup.json`, out-of-order upload misclassified as corrupt, and unsafe pruning before `CHECKPOINT_SYNCED`. | [`src/bangla_smish/post_stage_b_runner.py`](../src/bangla_smish/post_stage_b_runner.py), [`src/bangla_smish/post_stage_b_checkpointing.py`](../src/bangla_smish/post_stage_b_checkpointing.py), [`src/bangla_smish/post_stage_b_cohorts.py`](../src/bangla_smish/post_stage_b_cohorts.py), [`src/bangla_smish/post_stage_b_tokenizer_gate.py`](../src/bangla_smish/post_stage_b_tokenizer_gate.py). | `test_stage4_v8_remediation.py::test_scenario_i_*`. |
| **MED-2** | Partial 13/18 `source_sha256` hashing, weak `run_name`/`seed`/`folds`/`confirmatory_mode` validation, hardcoded bootstrap seed `42`, missing multi-GPU guard in non-benchmark modes, missing `gpu_smoke_test` package check, broken offline CPU aggregation, non-durable `keep_model_weights` checkpoints, and kit ignoring `Thumbs.db`/`._*`. | [`src/bangla_smish/post_stage_b_runner.py`](../src/bangla_smish/post_stage_b_runner.py) (`schema_version: 5`, 18-module `source_sha256`, strict config validation, multi-GPU rejection in all modes, `gpu_smoke_test` package check, `_persist_durable_trainer_checkpoint`, `offline_cpu_ok=True`), [`src/bangla_smish/metrics.py`](../src/bangla_smish/metrics.py) (`seed` parameter), [`src/bangla_smish/post_stage_b_integrity.py`](../src/bangla_smish/post_stage_b_integrity.py) (`Thumbs.db`, `._*`), [`scripts/generate_post_stage_b_stage4_notebook.py`](../scripts/generate_post_stage_b_stage4_notebook.py). | `test_stage4_v8_remediation.py::test_scenario_j_*`, `test_scenario_k_*`, `test_scenario_l_*`, `test_scenario_m_*`. |

---

## 3. Modified & Added Files Inventory (Exact SHA256)

| File | Status | SHA256 |
| :--- | :--- | :--- |
| [`pyproject.toml`](../pyproject.toml) | Modified | `7a464e74ed3a8199f3cb520aaa8c428462a6655ba7ff099959af775e07e413b0` |
| [`src/bangla_smish/metrics.py`](../src/bangla_smish/metrics.py) | Modified | `a42f4f6cbc9c2224ca4903b793ee74302e3b2b9874e707f7c6e6cda77793f586` |
| [`src/bangla_smish/post_stage_b_checkpointing.py`](../src/bangla_smish/post_stage_b_checkpointing.py) | Modified | `13b09b5320b146a4f52d27551952be121e6821180524158becee451449129666` |
| [`src/bangla_smish/post_stage_b_cohorts.py`](../src/bangla_smish/post_stage_b_cohorts.py) | Modified | `fe78d861e160bb17f42371ab463d323375dc84a8f536a92a3df1065ae18573cc` |
| [`src/bangla_smish/post_stage_b_colab_deps.py`](../src/bangla_smish/post_stage_b_colab_deps.py) | Modified | `3cc998c97cd2254b81c5ee5601f028b24854d947da83c8da5f4db26f1d6fce5b` |
| [`src/bangla_smish/post_stage_b_integrity.py`](../src/bangla_smish/post_stage_b_integrity.py) | Modified | `d7b702847df8c97350d2f9e26a6a1af8191ee206dc727712e0382f21d47bee35` |
| [`src/bangla_smish/post_stage_b_runner.py`](../src/bangla_smish/post_stage_b_runner.py) | Modified | `824b59400ca390f5d444dfea744a90223a0ae36dfee55716b4200146f0e38f00` |
| [`src/bangla_smish/post_stage_b_tokenization.py`](../src/bangla_smish/post_stage_b_tokenization.py) | Modified | `15cf98febea6e899f55a24217d43dd928a11d626fbf8301f1b1c3b916e8ef7f1` |
| [`src/bangla_smish/post_stage_b_tokenizer_gate.py`](../src/bangla_smish/post_stage_b_tokenizer_gate.py) | Modified | `ef37f03d04d62a36218bbc8cef169138fd5918c3d95a4221a58797c228516f13` |
| [`scripts/generate_post_stage_b_stage4_notebook.py`](../scripts/generate_post_stage_b_stage4_notebook.py) | Modified | `94181cc9159a3992e226bf78272aafc8fc1c28b170dc5681aa1f85d21e46ed6c` |
| [`experiments/colab/post_stage_b_stage4_experiments.ipynb`](../experiments/colab/post_stage_b_stage4_experiments.ipynb) | Regenerated | `70f8f15f858746539eb6935565611310c49bd71a4651624883d68d74f37b75fa` |
| [`tests/test_stage4_remediation.py`](../tests/test_stage4_remediation.py) | Modified | `5b8ef9c78b7169e92ed833ca12f69ee48a4ff72cbee617c0826052a840cc670d` |
| [`tests/test_stage4_final_remediation.py`](../tests/test_stage4_final_remediation.py) | Modified | `1b7222ff76e4510b4dca8ef8f87bf8f0d098442992f94813f5df12a76ed31ef0` |
| [`tests/test_stage4_v8_remediation.py`](../tests/test_stage4_v8_remediation.py) | Added | `cff8295c77792dd27c59cba9a217ce857dad39e8c0188defb18beda7dcfd8907` |
| `stage4_colab_kit_upload/bangla_smish_post_stage_b_stage4_v8/stage4_kit_manifest.json` | Added (`v8` kit) | `9933a1907cdd05e648c10955eda5c327d9dc3d4056506078e514faf82f35043c` |

Frozen configs confirmed **100% untouched**:
- `gpu_smoke_test_config.json`: `f60c0abe78c9c657bff471dccc124d8cc3ec1d369cb1487a75455ee319ecd41b`
- `post_stage_b_stage4_t4_benchmark_config.json`: `03a5f10d433f657af82d1af401774fd40d9d36209d35e13a14bc934086416180` (`benchmark_workload_fixture_eval: true`, fail-closed)
- `post_stage_b_stage4_option_a_config.json`: `518ec0cf56d52619fe5f24da7ef18a4b077466d38058c4d2538cd605c6910ecc`

---

## 4. Real-Library Local Evidence vs. CPU-Mocked Boundary

- **Verified with REAL local libraries and frozen data:** `importlib.metadata.PathDistribution` over real `.dist-info` closures, `huggingface_hub.try_to_load_from_cache` over real temporary HF cache snapshots, POSIX file/directory `fsync` and atomic `os.replace`, real `v2` bundle & 32-pair lookup, `4/4` frozen test mask re-derivation, `282/282` Category A baseline integrity, and `v3`–`v8` portable kit verification.
- **CPU-mocked ONLY (NOT verified on live Colab/GPU/Drive):** `torch`, `transformers==4.57.1` `AutoTokenizer`/`AutoModelForSequenceClassification`/`Trainer`, `datasets`, `normalizer`, CUDA Tesla T4 VRAM/compute units, and live Google Drive FUSE mount latency/reconnection.
