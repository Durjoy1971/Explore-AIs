# Post-Stage B — Stage 4 Remediation Independent Acceptance Review

**Date:** 2026-11-24  
**Repository / Branch / Base Commit:** `ExploringAIs` · `main` · `4d341183cb7a6fce0ed773ff33da2a3b757d6f94`  
**Audit Scope:** Strict read-only inspection of actual source code, generated Colab notebook, configs, tests, JSON artifacts, cgroup/CPU environment, and Git working tree. Zero existing project files modified; zero tests, statistical simulations, model training, Colab notebooks, Drive operations, or Git history mutations executed.  
**GPU Authorization Status:** **No GPU authorization is requested or assumed.** The 5-unit T4 benchmark is **NOT authorized yet**; full 160-unit Option A training remains strictly **`NO-GO`**.

---

## 1. Executive Verdict

**Overall Stage 4 Remediation Verdict:** **`CONDITIONALLY ACCEPTED`**

Independent inspection of the code confirms that all four prior blocking defects (**4.1–4.4**) and all three confirmatory governance gaps (**A–C**) were substantively fixed in Category-C source code, backed by real isolated-subprocess and `SIGKILL` tests, without touching any of the `282` Category A files or the failed statistical validation artifact (`reports/stage4_statistical_validation_results.json`, SHA256 `bde8085b...`). Acceptance is **conditional** rather than unconditional because: (1) hard `<= 25%` CPU-cap enforcement during test execution is **`NOT VERIFIED`** on this host (cgroup `cpu.max = max 100000`, `cpulimit` absent), (2) live Colab GPU, real `torch`/`transformers`/`normalizer` resolution, Google Drive FUSE `os.replace`/`fsync`/TOCTOU behavior, and the **32-pair allowed-view tokenizer `input_ids` collision check** remain **`NOT VERIFIED`** locally, and (3) minor defensive edge cases remain documented in Section 7.

| Review Track | Scope | Verdict | Primary Basis |
|---|---|---|---|
| **Review A.1** | Portable Colab Kit (`33` files) | **`CONDITIONALLY ACCEPTED`** | Strict isolated `post_stage_b_stage4_kit_v2` manifest + Cell 5 `verify_portable_colab_kit` verified; unexpected Drive dotfile risk remains. |
| **Review A.2** | Colab Dependency Preflight | **`CONDITIONALLY ACCEPTED`** | Ordered Cell 4 preflight preserves `torch==<ver>`, requires `>=2.6`, `transformers>=4.57.1,<5`, pinned `normalizer`; `protobuf`/transitive wheel resolution unverified on Colab. |
| **Review A.3** | Writer Lease & Drive FUSE | **`CONDITIONALLY ACCEPTED`** | Zero `fcntl.flock`/`.runner.lock`; owner takeover & audit trail verified cleanly; Drive FUSE TOCTOU & `os.replace` on `run_state.json` cannot be established locally. |
| **Review A.4** | Checkpoint Recovery & Retention | **`CONDITIONALLY ACCEPTED`** | Local `trainer_state.json` relocation keeps Drive staging immutable; independent best/latest verification; `keep_model_weights != "none"` `_CHECKPOINT_COMMIT` caveat & unlogged `prune_errors`. |
| **Review B** | Statistical Governance | **`ACCEPTED`** | `CONFIRMATORY_DECLARATIONS_ENABLED = False` un-bypassable across all helpers, ensembles, CSV, JSON, and Markdown exports; simulation helper isolated; failed JSON unchanged. |
| **Review C** | Scientific Reproducibility | **`CONDITIONALLY ACCEPTED`** | Frozen `1,687`/`1,698`/`1,693`/`1,675` masks, 5-fold alignment, `265`-Normal calibration, step budgets, `retraining_metadata.json`, and Option A view guards verified; **32-pair tokenizer `input_ids` check is `NOT VERIFIED`**. |
| **Review D** | CPU Compliance (`<= 25%` cap) | **`NOT VERIFIED`** | Host has 12 cores (`taskset 0-11`), `cpu.max = max 100000`, `no_cpulimit`, zero CPU telemetry logged; `OMP_NUM_THREADS=1 nice -n 19` does not enforce a 25% ceiling. |
| **Review E** | Protected Integrity (`282/282`) | **`ACCEPTED`** | Exact SHA256 verification of `protected_baseline_282.json`, `v1`/`v2` manifests, `13` canonical inputs, `4` masks, failed stats JSON; `0` Category A files touched. |
| **Review F** | T4 Benchmark Readiness | **`CONDITIONALLY ACCEPTED`** | Code technically ready for owner decision on a fresh `ARTIFACT_RUN_ID`; live Colab execution, Drive FUSE, and T4 VRAM/CU unverified; **5-unit benchmark NOT authorized yet; 160-unit grid strictly `NO-GO`**. |

---

## 2. Independent Assessment of the Four Engineering Fixes (Review A)

### 2.1 Review A.1 — Portable Colab Kit (`CONDITIONALLY ACCEPTED`)
- **Contract Separation ([post_stage_b_integrity.py#L264-L625](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_integrity.py#L264-L625)):** Full repository integrity (`verify_protected_baseline_manifest`, `282/282`) is untouched, while portable kit integrity (`post_stage_b_stage4_kit_v2`) sets `verifies_full_repository_282 = False` (`L501, L542`), embeds a canonical `manifest_sha256` (`L528-L549`), cross-checks all `8` `post_stage_b_bundle_v2/*` files against `protected_baseline_282.json` (`L589-L599`), checks `eval_smishdetect_test.parquet` against `stage4_frozen_test_masks.json` (`L601-L603`), checks `stage4_statistical_validation_results.json` verdict parity (`L605-L607`), checks runner flags (`confirmatory_mode=True` on Option A, `run_heldout_eval=False` on T4 benchmark at `L609-L612`), and rejects missing, corrupted (size/SHA256), unexpected extra, and incompatible files (`L568-L587`).
- **Notebook Cell 5 Sync ([post_stage_b_stage4_experiments.ipynb#L148-L166](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/post_stage_b_stage4_experiments.ipynb#L148-L166)):** Actual generated notebook Cell 5 calls `verify_portable_colab_kit(WORKSPACE_ROOT)` and never invokes `verify_protected_baseline_manifest`.
- **Complete Verified Inventory & SHA256 Values of All `33` Staged Portable Kit Files:**

| # | Category | Relative Path | Bytes | SHA256 |
|---:|---|---|---:|---|
| 1 | `v2_bundle` | `data/processed/post_stage_b_bundle_v2/counterfactual_smish_probe.parquet` | 1,185,746 | `9f2546e04f31368014a686605a843009a59ecb15284080f13391f87264a16c83` |
| 2 | `v2_bundle` | `data/processed/post_stage_b_bundle_v2/dataset_manifest.json` | 15,534 | `308d1c889840c1ec04e1cd6f6751cccfca0e559f2e12c61e18ff2e54d6fa2922` |
| 3 | `v2_bundle` | `data/processed/post_stage_b_bundle_v2/eval_banglaphish_ood.parquet` | 7,279,129 | `d08fd5cf7b45e89de7f9647c1e677e31a6120e40b39f520c0833eccafd73b12d` |
| 4 | `v2_bundle` | `data/processed/post_stage_b_bundle_v2/eval_smishdetect_test.parquet` | 1,354,154 | `80fb4d18d567add3d1854e0cad8a462fe10a9a84bb43fc8292aefcde7a2983ac` |
| 5 | `v2_bundle` | `data/processed/post_stage_b_bundle_v2/eval_smishdetect_validation.parquet` | 623,026 | `783cb23ec3920a3c5395fac83a8360af96b624e1413916f4c5ccf516d6b181d0` |
| 6 | `v2_bundle` | `data/processed/post_stage_b_bundle_v2/real_train_pool.parquet` | 3,404,813 | `b42029c001e9277f875a91b8b5e17d6cbd594a8fdafb3095ab847ac70f361324` |
| 7 | `v2_bundle` | `data/processed/post_stage_b_bundle_v2/synthetic_batches_b1_b7.parquet` | 291,112 | `62a099d761b0ce79293e4c215ed4bce4d18faf7bb9accab859a0ca19fd925abc` |
| 8 | `v2_bundle` | `data/processed/post_stage_b_bundle_v2/synthetic_pilot_300.parquet` | 284,460 | `62e95b7f037b3c899573ed416ea315941210420a32591237c409b5c941b8d54a` |
| 9 | `runner_config` | `experiments/colab/configs/gpu_smoke_test_config.json` | 660 | `f60c0abe78c9c657bff471dccc124d8cc3ec1d369cb1487a75455ee319ecd41b` |
| 10 | `runner_config` | `experiments/colab/configs/post_stage_b_stage4_option_a_config.json` | 1,023 | `a12f68cc3e804c4277b6a36186b89c276cd59308ea3255fa42bd400e4772041c` |
| 11 | `runner_config` | `experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json` | 1,275 | `dc038248eef2fde4a08cfe42893255c33b334b01e7c64de258a268930d1f9132` |
| 12 | `notebook` | `experiments/colab/post_stage_b_stage4_experiments.ipynb` | 10,898 | `0a1bd98bc94efa0cf5257f996524d4778b438a31d15e1accccccf416609bf515` |
| 13 | `dependency_spec` | `pyproject.toml` | 606 | `ea5c8cccdad08fbba8e1ce7b1f5def7a5278ca1abafdf2e83ddd29d2789646e7` |
| 14 | `governance_or_baseline` | `reports/protected_baseline_282.json` | 117,762 | `f0b80b1e29bd3d3002e6455672c43f73d01a87dd927ea0b55e693cb6f3945ce1` |
| 15 | `governance_or_baseline` | `reports/stage4_frozen_test_masks.json` | 6,347 | `db2cde6a5aa47a0050162cc97447de83eeb45a63819ee506ec4ca8d95e11c9cc` |
| 16 | `governance_or_baseline` | `reports/stage4_statistical_validation_results.json` | 14,554 | `bde8085b77066cbf449a1b17d0a1afa77f2dcbcb4ea1a82f7d0dde68f6fe504f` |
| 17 | `source_module` | `src/bangla_smish/__init__.py` | 270 | `57d0d5ec15b7f5f46281eb3a6aaa69713feef9c9f415d4d7dc76d9ee660af654` |
| 18 | `source_module` | `src/bangla_smish/audit.py` | 6,771 | `5ec0feb30e147a7aea63c531e507d6616232206bea46c9ea45573a94f667e9c0` |
| 19 | `source_module` | `src/bangla_smish/codemix_diversity.py` | 28,085 | `a0a62179be6b08a1adad27747dda406b1d79786858a86d6efe6b2984b183b8c2` |
| 20 | `source_module` | `src/bangla_smish/curation.py` | 43,966 | `461f5244615a191dc43fbd994ab81c0c28035ac4a813ed711f0e82c55557e9d2` |
| 21 | `source_module` | `src/bangla_smish/metrics.py` | 11,203 | `c50d3f0f53967921c9585055b68a4b1b2b2ea573329c247627456a38495f506f` |
| 22 | `source_module` | `src/bangla_smish/post_stage_b_checkpointing.py` | 37,794 | `b183ce9a805b5de6e67dca8a878bd9b054cb56d9220ddda8ed60f9d7ba4381bd` |
| 23 | `source_module` | `src/bangla_smish/post_stage_b_cohorts.py` | 11,257 | `d6aa505003f484e402280341e9f6a24b8d70e0cd86ec1c2c52f10dcf0a6274ca` |
| 24 | `source_module` | `src/bangla_smish/post_stage_b_colab_deps.py` | 7,457 | `3de8dcd3387e85b2defaf4f582ec0f400144ef2c125f26e333635e9ae8dd94a2` |
| 25 | `source_module` | `src/bangla_smish/post_stage_b_data.py` | 40,059 | `43276121a483264b9d1b48144bf69e8ea4f857adb8e796cff7845bd8f78f7e28` |
| 26 | `source_module` | `src/bangla_smish/post_stage_b_ensemble.py` | 17,207 | `89b600bf3217cf0354e2e4f8412e2d7567874fdc356d1b5d1924b487034db075` |
| 27 | `source_module` | `src/bangla_smish/post_stage_b_integrity.py` | 30,344 | `0627ca914fc9b832dc8577fe9c304f1fd6c5281925dfe9b25995d47d1ae43118` |
| 28 | `source_module` | `src/bangla_smish/post_stage_b_model_view_audit.py` | 2,814 | `34a2036d6cb45af2bb32fdcbcadca6d689a86ecead8d5530911589eac63a38b9` |
| 29 | `source_module` | `src/bangla_smish/post_stage_b_runner.py` | 83,180 | `6311103a5b37492c7ff801bb53728677fa170d95194c66d1ba1bbbc17401af1d` |
| 30 | `source_module` | `src/bangla_smish/post_stage_b_stats.py` | 24,887 | `21d6cea31a93298d5dbfdd9f7f3a4a650df15dc1222cbc7adf82a6666bcaef24` |
| 31 | `source_module` | `src/bangla_smish/post_stage_b_tokenization.py` | 834 | `45a036407e456f75174c5abcbc75663807f861ffebe2b52796aab50a75bc3439` |
| 32 | `source_module` | `src/bangla_smish/schema.py` | 38,400 | `39edd69c897ff37d7150f830b9708089c90dc3863eba0cb428b8bb03a49fdb98` |
| 33 | `source_module` | `src/bangla_smish/splits.py` | 3,045 | `fce582b7f4bcb0e4e0afe47333f0a51b69ab370da2733ecc56e742b06e8d6072` |

- **Remaining Kit Caveats:** `_enumerate_kit_files` ([post_stage_b_integrity.py#L408-L423](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_integrity.py#L408-L423)) excludes only `artifacts/`, `__pycache__/`, `*.pyc`, and `stage4_kit_manifest.json`; OS/Colab metadata files such as `.ipynb_checkpoints/`, `.DS_Store`, or `desktop.ini` in Drive will trigger an `UNEXPECTED` halt. Also, `verify_portable_colab_kit` checks `stage4_frozen_test_masks.json`'s file digest and `bundle_test_parquet_sha256` without re-deriving the 4 mask SHA256s live.

### 2.2 Review A.2 — Dependencies (`CONDITIONALLY ACCEPTED`)
- **Ordering, CUDA Torch Preservation & BanglaBERT Normalizer:** Notebook Cell 4 ([post_stage_b_stage4_experiments.ipynb#L121-L140](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/post_stage_b_stage4_experiments.ipynb#L121-L140)) executes BEFORE Cell 5 imports `bangla_smish`. `preflight_colab_dependencies` ([post_stage_b_colab_deps.py#L132-L205](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_colab_deps.py#L132-L205)) checks `torch >= 2.6`, preserves Colab CUDA PyTorch via temporary `-c constraints.txt` (`torch==<ver>`), requires `transformers>=4.57.1,<5`, `safetensors>=0.4`, `sentencepiece>=0.2,<1`, pins `normalizer @ git+https://github.com/csebuetnlp/normalizer@d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9` via `direct_url.json`, fails closed with `RESTART REQUIRED` on installation, and verifies top-level imports + `torch.cuda.is_available()`.
- **Transitive Dependency & Packaging Observations:**
  - `packaging` and `huggingface_hub` are imported directly in code (`post_stage_b_colab_deps.py:18`, `pyproject.toml:12`) and `protobuf` is commonly required by SentencePiece slow tokenizers (`muril-base-cased`/`xlm-roberta-base`), yet none of the three are explicitly listed in `STAGE4_BASE_PACKAGES` ([L37-L48](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_colab_deps.py#L37-L48)). They normally arrive preinstalled on Colab or transitively, but are **`NOT VERIFIED`** on live Colab.
  - `csebuetnlp/normalizer`'s transitive dependencies (`unicodedata2`, `ftfy`, `emoji`, etc.) and live pip dependency resolution against Colab `torch`/`torchvision`/`torchaudio` constraints remain **`NOT VERIFIED`** locally.
  - `pyproject.toml:19-26` `[gpu]` optional dependencies still list `torch>=2.2`, `transformers>=4.44` (cosmetic drift; Colab notebook uses `post_stage_b_colab_deps.py`, not `pyproject.toml [gpu]`).

### 2.3 Review A.3 — Writer Lease & Drive FUSE (`CONDITIONALLY ACCEPTED`)
- **Zero `fcntl.flock`:** Verified `0` `fcntl` imports, `0` `_run_lock` functions, and `0` `.runner.lock` creations in `post_stage_b_runner.py` (only read-only skip of legacy `.runner.lock` at `L868`).
- **Lease & Takeover Mechanics ([post_stage_b_checkpointing.py#L124-L325](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_checkpointing.py#L124-L325)):** Immutable `lease_*`, `release_*`, and `takeover_*` JSON records embed `lease_sha256`. Active leases, conflicting active writers, and corrupt/interrupted records all fail closed (`L226-L243`); age alone never triggers takeover; `allow_stale_takeover=True` requires non-empty `takeover_reason` (`L247`) and preserves all old records.
- **Unestablished Drive FUSE Guarantees & Minor Gap:**
  1. **True Distributed Mutex Impossible on Drive FUSE:** Google Drive FUSE lacks cross-VM atomic file locks and exhibits propagation lag; if two Colab runtimes are accidentally launched simultaneously, a TOCTOU window exists before `lease_*.json` syncs across VMs. `verify_writer_lease_still_sole_active` (`L312-L325`) detects conflicts only after propagation.
  2. **Single-Stale `expected_stale_writer_ids` Optional (`L249-L259`):** When only `1` active stale lease exists, `expected_stale_writer_ids=None` is permitted (only `> 1` conflicting writers mandate explicit IDs), so an operator leaving `EXPECTED_STALE_WRITER_IDS` blank in Cell 1 could supersede a single newly started lease rather than confirming its exact ID.
  3. **`os.fsync` / `os.replace` on `run_state.json` & `preds.npz` (`post_stage_b_runner.py#L204-L218, #L950-L954`):** While UUID checkpoint staging avoids `os.replace`/`fsync`, `run_state.json`, `metrics.json`, `preds.npz`, and `exports/*` still call `os.replace` / `os.fsync`, whose Drive FUSE durability under mid-write preemptions is **`NOT VERIFIED`** locally.

### 2.4 Review A.4 — Checkpoint Recovery & Retention (`CONDITIONALLY ACCEPTED`)
- **Verified Code Behavior ([post_stage_b_checkpointing.py#L486-L833](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_checkpointing.py#L486-L833)):** `scan_unit_staging` clamps `keep_n = max(1, int(checkpoint_retention_limit))` (`L622`) so the sole recoverable checkpoint is never superseded; distinct pinned `best_model_checkpoint` is retained alongside `keep_n` newest checkpoints (`L631-L645`); `recover_latest_valid_checkpoint` fails closed (`RecoveryError`, `L728-L732`) if a required best checkpoint is missing/corrupt, verifies per-file SHA256/byte readback for both best and latest checkpoints, and rewrites **only the restored local** `trainer_state.json` (`L768-L771`), leaving Drive staging untouched. `prune_superseded_staging` (`L787-L833`) catches `OSError` into `prune_errors` without deleting `VALID_RECOVERABLE`, `INCOMPLETE_UPLOAD`, or `CORRUPT_OR_UNVERIFIABLE` directories.
- **Minor Caveats Found:**
  1. `recover_latest_valid_checkpoint` (`L768-L771`) updates `restored_latest_dir / "trainer_state.json"` without refreshing local `_CHECKPOINT_COMMIT`. Harmless for default `keep_model_weights="none"`, but if `keep_model_weights != "none"` were used on a resumed attempt whose best checkpoint was `restored_latest_dir`, `_verify_trainer_checkpoint` ([post_stage_b_runner.py#L1367](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L1367)) would fail `_valid_checkpoint` on that modified local copy.
  2. `Stage4CheckpointCallback.on_save` ([post_stage_b_runner.py#L628](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L628)) ignores the `prune_superseded_staging` return dictionary rather than logging any non-empty `prune_errors`.

---

## 3. Statistical Governance Verification (Review B — `ACCEPTED`)

Inspected every code path referencing `CONFIRMATORY_DECLARATIONS_ENABLED`, `confirmatory_allowed`, `holm_reject_family`, `simulate_holm_reject_family`, and `confirmatory_status`:
- [post_stage_b_stats.py#L38-L39](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_stats.py#L38-L39): `CONFIRMATORY_DECLARATIONS_ENABLED: bool = False`, `STATISTICAL_VALIDATION_STATUS = "FAIL_OR_INCONCLUSIVE_CONFIRMATORY_DISABLED"`.
- `compare_models_cluster_bootstrap` ([L343-L355](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_stats.py#L343-L355)): `confirmatory_allowed` is `bool(method == "fixed_budget" and CONFIRMATORY_DECLARATIONS_ENABLED)` (always `False`), separated from `method_confirmatory_eligible`, paired with `p_value_status = "UNVALIDATED_DIAGNOSTIC_ONLY"`.
- `holm_reject_family` ([L456-L484](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_stats.py#L456-L484)): defaults to `confirmatory_allowed: bool = False` and intersects with `CONFIRMATORY_DECLARATIONS_ENABLED`, recording `governance_bypass_blocked = True` if a caller passes `True`.
- `simulate_holm_reject_family` ([L432-L453](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_stats.py#L432-L453)): isolated purely for Monte Carlo calibration (`validate_stage4_fixed_bootstrap.py:309-310, 434-435`), tagging `simulation_only = True, production_confirmatory_claim = False`.
- `evaluate_stage4_hypothesis_families` ([L487-L565](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_stats.py#L487-L565)) and `build_ensemble_evaluation_summary` ([post_stage_b_ensemble.py#L232, #L384-L393](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_ensemble.py#L384-L393)): enforce the global gate even against explicit caller override (`confirmatory_declarations_enabled=True`).
- Runner exports ([post_stage_b_runner.py#L1456-L1460, #L1578-L1691](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L1578-L1691)): emit `FAIL_OR_INCONCLUSIVE_CONFIRMATORY_DISABLED` and `confirmatory_declarations_enabled: False` into `aggregated_results.json`, every row of `aggregated_results.csv`, the governance header + table of `post_stage_b_report.md`, and standalone `exports/retraining_metadata.json`.
- Original failed evidence preserved byte-for-byte (`reports/stage4_statistical_validation_results.json` SHA256 `bde8085b77066cbf449a1b17d0a1afa77f2dcbcb4ea1a82f7d0dde68f6fe504f`); zero simulation reruns occurred.

---

## 4. Scientific Reproducibility Verification (Review C — `CONDITIONALLY ACCEPTED`)

- **Confirmed Accepted Controls:**
  - Frozen cohort masks in `reports/stage4_frozen_test_masks.json` (`db2cde6a...`): `primary_family_decontam_1687` (`mask_sha256 = 86527c48...`), `supplementary_full_1698` (`e73e775f...`), `sensitivity_row_only_1693` (`a70fe670...`), `sensitivity_template_stress_1675` (`0b2e9996...`). Applied identically across arms in `build_ensemble_evaluation_summary` ([post_stage_b_runner.py#L1565-L1576](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L1565-L1576)).
  - 5-fold ensemble validation & alignment ([post_stage_b_ensemble.py#L31-L102](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_ensemble.py#L31-L102)), `265`-Normal `smishdetect_validation` threshold calibration (`<= 2` FPs, zero held-out/OOD tuning), `confirmatory_mode=True` blocking `brand_masked`/`brand_stripped` ([post_stage_b_runner.py#L165-L178](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L165-L178)), equal-epoch optimizer-step reporting (`300 / 300 / 333 / 363`), and complete standalone `exports/retraining_metadata.json` (`L1581-L1593, L1689`).
- **Critical Unresolved Scientific Item (`NOT VERIFIED`):**
  - The **32-pair allowed-view (`raw`, `canon_masked`) tokenizer `input_ids` collision check** across all 4 HuggingFace backbones (`csebuetnlp/banglabert`, `google/muril-base-cased`, `xlm-roberta-base`, `distilbert-base-multilingual-cased`) has **NOT** been executed locally (no `transformers`/`normalizer`) and is **NOT yet wired into an automated pre-training Colab gate cell** in [post_stage_b_stage4_experiments.ipynb](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/post_stage_b_stage4_experiments.ipynb).

---

## 5. CPU Compliance Assessment (Review D — `NOT VERIFIED`)

- **Inspected Host & Cgroup Resource Controls:**
  - `nproc` = **`12`** CPUs; `taskset -pc $$` = **`0-11`** (unrestricted CPU affinity).
  - `/proc/self/cgroup` = `0::/user.slice/user-1001.slice/user@1001.service/app.slice/...`; cgroup v2 `cpu.max` across the hierarchy is **`max 100000`** (no hard CFS bandwidth quota).
  - `which cpulimit` = **`no_cpulimit`** (not installed).
  - Test logs contain no recorded per-process `%CPU` or cgroup `cpu.stat` measurements.
- **Finding:** `OMP_NUM_THREADS=1 ... nice -n 19` restricts BLAS internal threading and lowers scheduling priority under contention, but does **not** enforce or prove a hard `<= 25%` system CPU utilization ceiling on a multi-worker or multi-process run on an otherwise idle 12-core host. Therefore, hard `25%` CPU-cap compliance is **`NOT VERIFIED`**.

---

## 6. Protected Baseline Integrity (Review E — `ACCEPTED`)

- **Verified Disk SHA256s:**
  - `reports/stage4_statistical_validation_results.json`: `bde8085b77066cbf449a1b17d0a1afa77f2dcbcb4ea1a82f7d0dde68f6fe504f`
  - `reports/protected_baseline_282.json`: `f0b80b1e29bd3d3002e6455672c43f73d01a87dd927ea0b55e693cb6f3945ce1`
  - `reports/stage4_frozen_test_masks.json`: `db2cde6a5aa47a0050162cc97447de83eeb45a63819ee506ec4ca8d95e11c9cc`
  - `data/processed/post_stage_b_bundle/dataset_manifest.json` (`v1`): `2e2e76e296873ffc843629ccae560dd8df6597debf93cfe2ab522af49dcd7a9b`
  - `data/processed/post_stage_b_bundle_v2/dataset_manifest.json` (`v2`): `308d1c889840c1ec04e1cd6f6751cccfca0e559f2e12c61e18ff2e54d6fa2922`
- **Git Working-Tree Verification (`git status --short` & `git diff --stat`):** Only Category-C/B files modified or added (`M CHECKPOINT.md [+36]`, `M src/bangla_smish/post_stage_b_runner.py [+635/-78]`, plus untracked Stage 4 code/tests/configs/reports). Zero Category A (`282` files), dataset, bundle, Google Drive, Colab, model training, or Git history modifications occurred.

---

## 7. Exact Unresolved Issues

1. **CPU Hard-Cap Telemetry (`NOT VERIFIED`):** No hard cgroup/affinity/cpulimit `25%` cap or `%CPU` log proves compliance (Review D).
2. **32-Pair Allowed-View Tokenizer `input_ids` Collision Check (`NOT VERIFIED`):** Not executed locally and not yet automated inside a Colab pre-training gate cell (Review C).
3. **Colab Drive Hidden Metadata Sensitivity (Minor):** `_enumerate_kit_files` does not ignore `.ipynb_checkpoints/`, `.DS_Store`, or `desktop.ini`, which can appear inside mounted Google Drive folders and trigger an `UNEXPECTED` error in Cell 5 (Review A.1).
4. **Mask Re-Derivation Check Absent Inside Portable Kit Check (Minor):** `verify_portable_colab_kit` checks `stage4_frozen_test_masks.json` digest and parquet digest parity without re-running `build_smishdetect_test_cohort_masks` on Colab (Review A.1).
5. **Unpinned Indirect/Tokenizer Helpers (Minor):** `packaging`, `huggingface_hub`, `protobuf`, and `csebuetnlp/normalizer`'s transitive dependencies (`unicodedata2`, `ftfy`, `emoji`) remain unverified on live Colab pip (`pyproject.toml` `[gpu]` also shows older minimum pins) (Review A.2).
6. **Single-Stale Takeover ID Confirmation Optional (Minor):** `register_single_writer_lease` only mandates `expected_stale_writer_ids` when `> 1` active lease exists, allowing `allow_stale_takeover=True` with empty `EXPECTED_STALE_WRITER_IDS` for single-lease takeovers (Review A.3).
7. **Google Drive FUSE Runtime Unverified (`NOT VERIFIED`):** Simultaneous-launch TOCTOU window, `os.replace`/`os.fsync` on `run_state.json`/`preds.npz`, `shutil.rmtree` cleanup, and 30-day Drive Trash quota remain unverified locally (Review A.3).
8. **Resume `_CHECKPOINT_COMMIT` & Silent `prune_errors` (Minor):** Local `trainer_state.json` best-path relocation does not refresh local `_CHECKPOINT_COMMIT` (would affect `keep_model_weights != "none"` if later used on a resumed unit), and `Stage4CheckpointCallback.on_save` ignores returned `prune_errors` (Review A.4).

---

## 8. Final Readiness Matrix (Review F — `CONDITIONALLY ACCEPTED`)

| Capability / Component | Locally Verified | Mocked / Subprocess Verified | Colab-Only (`NOT VERIFIED`) | Requires Separate Owner Authorization |
|---|---|---|---|---|
| `282`-File Protected Inventory, `13` Inputs, `v1`/`v2` Bundles, `4` Frozen Masks | **Yes** (`282/282`, `13/13`, `4/4`) | — | — | Required for any change (strictly read-only) |
| Statistical Governance & Fail-Closed `CONFIRMATORY_DECLARATIONS_ENABLED=False` | **Yes** (`0.0620` preserved) | **Yes** (bypass guards & exports) | — | New owner plan needed to redesign/re-validate |
| Portable Colab Kit (`post_stage_b_stage4_kit_v2`, `33` files) & Integrity Contract | — | **Yes** (isolated subprocess outside repo) | Drive FUSE dotfile handling | — |
| Dependency Preflight (`torch >= 2.6`, `transformers >= 4.57.1,<5`, `normalizer`) | — | **Yes** (mocked dists, imports, pip constraint) | Live Colab pip, CUDA, `normalizer` transitive deps | — |
| Safe Lease Protocol & Phase 1–3 Epoch Checkpoint Relocation / Retention | — | **Yes** (`SIGKILL` + temp filesystem) | Live Drive FUSE, HF `Trainer` resume, Trash quota | Owner stale takeover if `AUTHORIZE`d run preempted |
| 5-Unit T4 Hardware Benchmark Execution (`run_heldout_eval=false`, `7,114` fixture) | Fixture construction only | Config & schema only | Live T4 VRAM, `WeightedTrainer`, `CU/hr`, 32-pair tokenizer check | **YES — NOT AUTHORIZED YET (`AUTHORIZE_T4_BENCHMARK=False`)** |
| Full `160`-Unit Option A Experiment Grid | Config guard only | 5-fold ensemble & step budgets | Full GPU execution | **STRICTLY `NO-GO`** |

---

## 9. Source File Path & Line Number Evidence Index

- `src/bangla_smish/post_stage_b_integrity.py`: `#L24`, `#L113-L352` (full 282 baseline), `#L355-L625` (`post_stage_b_stage4_kit_v2`, `_enumerate_kit_files`, `build_portable_colab_kit_manifest`, `verify_portable_colab_kit`).
- `src/bangla_smish/post_stage_b_colab_deps.py`: `#L35-L63`, `#L81-L205` (`preflight_colab_dependencies`).
- `src/bangla_smish/post_stage_b_checkpointing.py`: `#L124-L325` (`inspect_writer_leases`, `register_single_writer_lease`, `release_single_writer_lease`, `verify_writer_lease_still_sole_active`), `#L365-L833` (`scan_unit_staging`, `recover_latest_valid_checkpoint`, `prune_superseded_staging`).
- `src/bangla_smish/post_stage_b_stats.py`: `#L38-L39`, `#L343-L355`, `#L432-L565` (`simulate_holm_reject_family`, `holm_reject_family`, `evaluate_stage4_hypothesis_families`).
- `src/bangla_smish/post_stage_b_ensemble.py`: `#L13`, `#L31-L102`, `#L232`, `#L384-L393`.
- `src/bangla_smish/post_stage_b_runner.py`: `#L165-L178` (`confirmatory_mode`), `#L457-L496` (`_build_t4_benchmark_workload_fixture`), `#L591-L660` (`Stage4CheckpointCallback`, resume wiring), `#L860-L870`, `#L1080-L1176` (`_run_locked`, lease acquire/release), `#L1355-L1365`, `#L1386-L1408` (`aggregate_post_stage_b_results`), `#L1456-L1460`, `#L1565-L1691` (exports & `retraining_metadata.json`).
- `experiments/colab/post_stage_b_stage4_experiments.ipynb`: `#L28-L56` (Cell 1 gates), `#L121-L140` (Cell 4 deps), `#L148-L166` (Cell 5 kit check), `#L175-L191` (Cell 6 runner).
- `experiments/colab/configs/post_stage_b_stage4_option_a_config.json`: `#L1-L36`; `post_stage_b_stage4_t4_benchmark_config.json`: `#L1-L42`.
- `scripts/validate_stage4_fixed_bootstrap.py`: `#L56`, `#L309-L310`, `#L434-L435`.
