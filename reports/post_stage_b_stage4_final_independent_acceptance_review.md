# Post-Stage B — Stage 4 Final Independent Acceptance Review

**Date:** 2026-11-24  
**Repository:** `/home/bs01415/Documents/Programming/ExploringAIs` (`HEAD`: `4d341183cb7a6fce0ed773ff33da2a3b757d6f94`)  
**Audited Remediation Report:** [`reports/post_stage_b_stage4_final_remediation_report.md`](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/post_stage_b_stage4_final_remediation_report.md)  
**Prior Acceptance Review:** [`reports/post_stage_b_stage4_independent_acceptance_review.md`](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/post_stage_b_stage4_independent_acceptance_review.md)  
**Review Mode:** Read-only independent acceptance audit. Zero existing project files modified; zero tests, simulations, model training runs, Colab runs, Google Drive operations, or Git history mutations executed.

---

## Executive Verdict

**Overall Final Acceptance Verdict:** **`CONDITIONALLY ACCEPTED`**

1. **Local Stage 4 Category-C Readiness:** All previously identified local engineering, statistical governance, lease/checkpoint recovery, portable kit contract, protected baseline integrity, and hardware-enforced CPU cap findings are resolved and independently verified against source code, configurations, notebook cells, on-disk SHA256 digests, and raw recorded execution logs (`task-947.log`, `task-951.log`).
2. **Reason for `CONDITIONALLY ACCEPTED` (Not Unconditional):** Real Colab GPU runtime components—live HuggingFace tokenizer loading for all four pinned commit revisions with `normalizer`, live Colab `pip` dependency resolution preserving CUDA PyTorch, live Google Drive FUSE mount/sync/contention semantics, and empirical T4 VRAM headroom / CU-per-hour telemetry—cannot be executed offline/locally and remain **`NOT VERIFIED`**.
3. **Governance Gate Status:**
   - **Five-unit T4 benchmark:** **NOT YET AUTHORIZED** (this review determines Stage 4 local readiness only; T4 execution awaits separate owner authorization).
   - **Full 160-unit Option A training grid:** Strictly **`NO-GO`**.
   - **Confirmatory superiority declarations:** Strictly **`DISABLED`** (`CONFIRMATORY_DECLARATIONS_ENABLED = False`).

### Section Ratings Summary

| Review Area | Verdict | Notes |
|---|---|---|
| **A. CPU Enforcement** | **`ACCEPTED`** | Kernel cgroup v2 `CPUQuota=300%` + `taskset -c 0-2` enforced; raw logs confirm `<= 8.61%` host CPU across probe, 27 targeted tests, and 90 regression tests |
| **B. Tokenizer Collision Gate** | **`CONDITIONALLY ACCEPTED`** | Code & mocked `256/256` gate: **`ACCEPTED`**; Live HuggingFace tokenizer + `normalizer` runtime: **`NOT VERIFIED`** |
| **C. Checkpoint and Writer Lease** | **`CONDITIONALLY ACCEPTED`** | Local logic & subprocess tests: **`ACCEPTED`**; Live Google Drive FUSE concurrency: **`NOT VERIFIED`** |
| **D. Portable Kit and Dependencies** | **`CONDITIONALLY ACCEPTED`** | Isolated 34-file kit, live 4-mask re-derivation & pinned spec: **`ACCEPTED`**; Live Colab pip/CUDA runtime: **`NOT VERIFIED`** |
| **E. Scientific Governance** | **`ACCEPTED`** | Confirmatory disabled, byte-identical statistical validation JSON, `raw`/`canon_masked` views, A/B/C/D step reporting, zero synthetic/frozen edits |
| **F. Protected Integrity** | **`ACCEPTED`** | `282/282` protected files & `13/13` canonical input hashes verified; `0` Category A files modified; clean tracked diff (`3` files) |
| **G. Reporting & Readiness Matrix** | **`CONDITIONALLY ACCEPTED`** | Remediation report has minor inventory/command/telemetry/line-number omissions documented below; full cumulative inventory & commands supplied here |

---

## Section A — CPU Enforcement (`ACCEPTED`)

Independently inspected [`scripts/run_stage4_capped.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/scripts/run_stage4_capped.py) (`L1-L168`, SHA256 `12848fa63b00023ff2be1d4018581d1d37cea17756743551ed6a8c3bc8848504`):

1. **Host CPU Count & Max Allowed CPUs:**
   - `host_cpus = int(os.cpu_count() or 1)` (`L67` -> `12` logical CPUs).
   - `max_cpus = max(1, int(math.floor(max_cpu_fraction * host_cpus + 1e-9)))` (`L68` -> `3` CPUs at the `0.25` host fraction ceiling).
2. **Fail-Closed Affinity & cgroup v2 Enforcement Before Execution:**
   - Parent affinity check (`L70-L75`): Fails closed if `len(os.sched_getaffinity(0)) > max_cpus`.
   - cgroup v2 `cpu.max` check via `_read_cgroup_cpu_max` (`L36-L50`, enforced at `L77-L84`): Parses `/proc/self/cgroup`, walks upward under `/sys/fs/cgroup<cg>/cpu.max`, converts `quota / period` to `quota_cpus`, and fails closed if `quota_cpus is None` (`max` / missing / inaccessible) or `quota_cpus > max_cpus + 1e-6`. Note that systemd `CPUQuota=300%` corresponds to `300000 100000` = `3.0` CPUs (`25%` of `12` logical CPUs).
   - Child subprocess inheritance probe (`L86-L94`): Launches `sys.executable` child reading `os.sched_getaffinity(0)` and fails closed if `child_affinity != affinity`.
3. **CPU Utilization Telemetry Calculation (`L144-L148`):**
   - `effective_host_cpu_pct = (cg_delta_usec / 1e6) / (wall_sec * host_cpus) * 100.0`, cross-checked against `resource.getrusage(resource.RUSAGE_CHILDREN)` (`ru_utime + ru_stime`) and `nr_throttled` delta from `cpu.stat`.

### Independently Verified Raw Saved Execution Logs

Telemetry was inspected directly in the recorded task logs (`task-947.log` and `task-951.log`), not taken solely on faith from the remediation report:

| Run | Log Source | Tests / Scope | `cpu.max` | Parent / Child Affinity | `wall_seconds` | `cgroup_usage_seconds` | `rusage_children_cpu_seconds` | `nr_throttled_delta` | `effective_host_cpu_pct` | Exit |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| **Cap Probe** | `task-947.log:2-3` | `CAP_PROBE_OK [0, 1, 2]` | `300000 100000` (`3.0` CPUs / `12`) | `[0, 1, 2]` / `[0, 1, 2]` | — | — | — | — | `8.61%` | `0` |
| **Targeted Remediation Suite** | `task-947.log:50-53` | `Ran 27 tests in 10.303s` (`OK`) | `300000 100000` (`3.0` CPUs / `12`) | `[0, 1, 2]` / `[0, 1, 2]` | `11.883` | `11.755` | `11.754` | `0` | `8.24%` | `0` |
| **Full Regression + Integrity** | `task-951.log:163-166, 478, 739-748` | `Ran 90 tests in 118.437s` (`OK`), `verified_count: 282`, `verified_runtime_file_count: 34`, `reverified_frozen_masks: 4` | `300000 100000` (`3.0` CPUs / `12`) | `[0, 1, 2]` / `[0, 1, 2]` | `134.2` | `134.122` | `134.121` | `0` | `8.33%` | `0` |

### Candid Non-Blocking Subtlety Observed

1. `_read_cgroup_cpu_max` (`L47`) stops at the first `cpu.max` file found walking upward and returns `None` if that leaf reads `max 100000` even if a higher ancestor cgroup carries a quota. This behavior is **strictly conservative / fail-closed** (it can never false-pass).
2. `--telemetry-out` was not passed when generating `task-947.log` and `task-951.log`, so raw telemetry JSON resides in agent conversation logs rather than a persisted file under `reports/`.

---

## Section B — Tokenizer Collision Gate (`CONDITIONALLY ACCEPTED`)

- **Code, Configuration & Mocked Gate Verification:** **`ACCEPTED`**
- **Actual Live HuggingFace Tokenizers & `normalizer` Execution:** **`NOT VERIFIED`**

Independently inspected [`src/bangla_smish/post_stage_b_tokenizer_gate.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_tokenizer_gate.py) (`L1-L203`, SHA256 `fee128773d761875f7b9d7f892c6c8035720c47cd667466b591b6ab81de24627`), Notebook Cells 5 & 6 of [`experiments/colab/post_stage_b_stage4_experiments.ipynb`](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/post_stage_b_stage4_experiments.ipynb) (`L1-L244`, SHA256 `53b35f508f9a4c783ba951e409c0cf324b9992732f629b65ac06eb587f50a219`), and both Stage 4 configs:

1. **Exact P01–P32 Pair Inventory:** `DOCUMENTED_PAIR_IDS` (`L27-L60`) enumerates all 32 Stage 3 pairs `P01..P32` matching [`reports/post_stage_b_stage3_readiness_review.md`](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/post_stage_b_stage3_readiness_review.md#L38-L69). `load_documented_32_pairs` (`L73-L107`) parses the Stage 3 report and fails closed (`ValueError`) if any pair is missing or duplicated—never inventing synthetic replacement pairs.
2. **All Four Pinned Tokenizer Revisions Preserved & Verified:** `PINNED_TOKENIZER_REVISIONS` (`L18-L23`) matches [`reports/post_stage_b_model_view_collision_adjudication.md`](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/post_stage_b_model_view_collision_adjudication.md#L42-L45) and both Stage 4 configs:
   - `csebuetnlp/banglabert`: `9ce791f330578f50da6bc52b54205166fb5d1c8c`
   - `csebuetnlp/banglishbert`: `afd9f36c7923d54e97903922ff1b260d091d202f`
   - `sagorsarker/bangla-bert-base`: `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`
   - `xlm-roberta-base`: `45c032ab32cc946ad88a166f7cb282f58c753c2e`
3. **Allowed-View Enforcement & Fail-Closed Pre-Training Wiring:**
   - `run_allowed_view_tokenizer_collision_gate` (`L110-L203`) enforces `set(allowed_views) <= {"raw", "canon_masked"}` (`L120-L121`), validates all four 40-hex commit revisions (`L123-L129`), fails closed if `transformers` (`AutoTokenizer`) cannot be imported (`L63-L70`), and raises `RuntimeError` on any identical `input_ids` collision (`L198-L202`).
   - Outputs `stage4_tokenizer_collision_gate.json` recording `config_hash`, `tokenizer_revisions`, `pair_ids` (`P01..P32`), `pair_count` (`32`), `checked_pairs`, `collisions`, and per-pair `results` (`L177-L196`).
   - Notebook Cell 5 calls `run_allowed_view_tokenizer_collision_gate(...)` unconditionally and sets `STAGE4_TOKENIZER_GATE_OK = True`; Cell 6 fails closed (`raise RuntimeError(...)`) unless both `STAGE4_INTEGRITY_OK` and `STAGE4_TOKENIZER_GATE_OK` are `True`.

### Candid Findings & Verification Boundary

1. **T4 Benchmark Cell 5 Checks `canon_masked` Only (128 checks, not 256):** Notebook Cell 5 (`experiments/colab/post_stage_b_stage4_experiments.ipynb:173`) passes `allowed_views=tuple(v for v in cfg.text_views if v in {'raw', 'canon_masked'})`. Because `post_stage_b_stage4_t4_benchmark_config.json` sets `text_views: ["canon_masked"]`, running Cell 5 under the T4 benchmark config executes `4 models × 1 view × 32 pairs = 128` checks (`canon_masked` only); `raw` (`128` additional checks, `256` total) is checked when running `option_a_config` (`["canon_masked", "raw"]`). While this covers 100% of views used in the active T4 run, `raw` tokenization collision immunity on live HF tokenizers would remain unverified after a T4-only run unless `raw` is explicitly included in the gate call.
2. **`actual_rev` Fallback (`post_stage_b_tokenizer_gate.py:141-145`):** `actual_rev = getattr(tok, "init_kwargs", {}).get("_commit_hash") or getattr(tok, "_commit_hash", None) or expected_rev`. If an upstream HuggingFace tokenizer object omits `_commit_hash`, it falls back to `expected_rev` (relying on `AutoTokenizer.from_pretrained(..., revision=expected_rev)` having enforced the pinned Hub commit).
3. **Mocked vs. Live Tokenizer Scope:** All `256/256` local checks in `tests/test_stage4_final_remediation.py` ran against a deterministic `_MockTokenizer` (plus `banglabert` `normalizer` behavior mocked). Live HuggingFace Hub downloads and real C/Rust tokenizers + `normalizer.normalize` remain **`NOT VERIFIED`** until Colab execution.

---

## Section C — Checkpoint and Writer Lease (`CONDITIONALLY ACCEPTED`)

- **Local Implementation & Subprocess Verification:** **`ACCEPTED`**
- **Live Google Drive FUSE Lock/Sync Semantics:** **`NOT VERIFIED`**

Independently inspected [`src/bangla_smish/post_stage_b_checkpointing.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_checkpointing.py) (`L190-L325`, `L660-L825`) and [`src/bangla_smish/post_stage_b_runner.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py) (`L604-L609`, `L641`, `L670`, `L1070`, `L1358`, `L1396-L1404`):

1. **Exact Stale Writer ID Mandatory:** `acquire_writer_lease` (`post_stage_b_checkpointing.py:247-292`) requires `expected_stale_writer_ids` whenever any active lease exists, refusing takeover when `expected_set != set(active_ids)` even when only one stale lease is present (`L251-L266`).
2. **Explicit Corrupt-Record Confirmation:** Any corrupt lease/release JSON triggers fail-closed refusal unless `expected_corrupt_files` explicitly matches the corrupt filename(s) (`L267-L279`), preserving immutable audit records.
3. **Append-Only Audit Trail:** Lease acquire, release, and takeover events write distinct append-only `lease_*.json`, `release_*.json`, and `takeover_*.json` records (`L291-L306`).
4. **Local `_CHECKPOINT_COMMIT` Refresh & Clean Drive Staging:** `_refresh_local_ckpt_commit_file` (`L685-L694`, invoked at `L709` for latest checkpoint and `L804` for pinned best checkpoint) refreshes local `post_stage_b_checkpoint_commit.json` on restored local directories without mutating Drive staging files.
5. **Surfaced `prune_errors` & Bounded Retention:** Prune exceptions are captured, logged via warning logger, and persisted into unit `metrics.json` (`post_stage_b_runner.py:604-609, 641, 670, 1358, 1396-1404`). Retention clamps `keep_n = max(1, int(checkpoint_retention_limit))` and protects both the sole recoverable checkpoint and the distinct pinned best checkpoint.

### Candid Caveats Disclosed

1. **Drive FUSE Operational Boundary:** Without OS/filesystem distributed locks across separate Colab VMs on Drive FUSE, an operator who manually supplies the writer ID of an *actually still running* session in `EXPECTED_STALE_WRITER_IDS` can still force a takeover. Lease safety assumes operator confirmation only after verifying the old Colab runtime has stopped.
2. **Non-Default Local Checkpoint + Weight-Retention Path:** In `post_stage_b_runner.py:1070`, `_read_committed_unit` reads checkpoint status from `artifacts_dir / "checkpoints" / uid`. If a future configuration sets `local_checkpoint_root is not None` *together with* `keep_model_weights != "none"`, resume visibility vs. Drive-staging separation would diverge. This does not affect Option A or the T4 benchmark because both enforce `keep_model_weights = "none"` (predictions-only policy).

---

## Section D — Portable Kit and Dependencies (`CONDITIONALLY ACCEPTED`)

- **Isolated Local Kit Verification, Live Mask Re-Derivation & Dependency Spec:** **`ACCEPTED`**
- **Live Colab Pip Installation & CUDA PyTorch Preservation:** **`NOT VERIFIED`**

Independently inspected [`src/bangla_smish/post_stage_b_integrity.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_integrity.py) (`L355-L646`), [`src/bangla_smish/post_stage_b_colab_deps.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_colab_deps.py) (`L30-L221`), and [`pyproject.toml`](file:///home/bs01415/Documents/Programming/ExploringAIs/pyproject.toml) (`L18-L30`):

1. **Strict Unexpected-File Allowlist:** `_PORTABLE_KIT_IGNORED_FILENAMES = {".DS_Store", "desktop.ini"}` and `_PORTABLE_KIT_IGNORED_DIRS = {".ipynb_checkpoints"}` (`L383-L384`), plus runtime outputs (`artifacts/`, `__pycache__/`, `*.pyc`, `stage4_kit_manifest.json`) verified at `L411-L426`. Any unexpected `.py`, `.json`, `.sh`, or scientific/executable artifact fails closed.
2. **34-File Integrity & Live Evaluation Mask Re-Derivation:** `verify_portable_colab_kit` (`L536-L646`) verifies all **34** portable runtime files (`18` `src/bangla_smish/*.py`, `8` `v2` bundle files, `3` configs, `3` Stage 4 report JSONs, `pyproject.toml`, and `post_stage_b_stage4_experiments.ipynb`), cross-checks `8/8` `v2` files against `protected_baseline_282.json`, **re-derives and hashes all 4 frozen evaluation masks live** (`L607-L619`), validates `tokenizer_revisions` (`L629-L632`), and returns `verifies_full_repository_282=False`.
3. **Colab Dependency Spec & CUDA PyTorch Preservation:** `post_stage_b_colab_deps.py` (`L37-L76, L170-L210`) pins `torch>=2.6`, `transformers>=4.57.1,<5`, `datasets>=2.20,<5`, `accelerate>=0.33,<2`, `safetensors>=0.4`, `sentencepiece>=0.2,<1`, `packaging>=23.0`, `huggingface_hub>=0.24,<2`, `protobuf>=3.20`, and pinned `normalizer` (with `normalize` import verification), writing a `torch==<installed_version>` constraints file so Colab's CUDA-linked `torch` wheel is never overwritten.
4. **Minor Spec Note (Non-Blocking):** `pyproject.toml` base `[project].dependencies` lists `huggingface_hub>=0.24` without an upper bound `<2`, whereas `[gpu]` / `post_stage_b_colab_deps.py:44` enforces `huggingface_hub>=0.24,<2`. Colab installs through `post_stage_b_colab_deps.py`, so the `<2` cap holds on Colab.

---

## Section E — Scientific Governance (`ACCEPTED`)

Independently confirmed across source, tests, configs, and artifacts:

1. **Confirmatory Declarations Disabled:** `CONFIRMATORY_DECLARATIONS_ENABLED = False` and `STATISTICAL_VALIDATION_STATUS = "FAIL_OR_INCONCLUSIVE_CONFIRMATORY_DISABLED"`; no code path emits unauthorized confirmatory superiority declarations.
2. **Statistical Validation Artifact Byte-Identical:** [`reports/stage4_statistical_validation_results.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/stage4_statistical_validation_results.json) is untouched with SHA256 `bde8085b77066cbf449a1b17d0a1afa77f2dcbcb4ea1a82f7d0dde68f6fe504f`.
3. **Confirmatory Views & Training Budget Disclosure:** Option A restricts confirmatory analysis to `canon_masked` (primary) and `raw` (reference), excluding `brand_masked` and `brand_stripped`. Equal-epoch suites `A/B/C/D` retain transparent optimizer-step budgets (`300/300/333/363` steps across 3 epochs) recorded in `retraining_metadata.json`.
4. **Frozen Data Untouched:** Zero synthetic generation and zero modifications to frozen datasets, bundles, or masks.

---

## Section F — Protected Integrity (`ACCEPTED`)

Verified via read-only `git rev-parse HEAD`, `git status --short`, `git diff --stat`, raw `task-951.log:478` (`verify_stage4_integrity.py` output), and live `sha256sum`:

- **Git HEAD Unchanged:** `4d341183cb7a6fce0ed773ff33da2a3b757d6f94` (zero Git history operations).
- **Modified Tracked Files (`3` total — all Category C / documentation):**
  - `CHECKPOINT.md` (`+45`)
  - `pyproject.toml` (`+12/-...`)
  - `src/bangla_smish/post_stage_b_runner.py` (`+717/-83`)
- **Category A Protected Files Modified:** **`0`** (`verified_count: 282` in `task-951.log:478`).
- **Independently Verified On-Disk SHA256 Digests:**

| File | Verified SHA256 |
|---|---|
| `reports/protected_baseline_282.json` | `f0b80b1e29bd3d3002e6455672c43f73d01a87dd927ea0b55e693cb6f3945ce1` |
| `reports/stage4_frozen_test_masks.json` | `db2cde6a5aa47a0050162cc97447de83eeb45a63819ee506ec4ca8d95e11c9cc` |
| `reports/stage4_statistical_validation_results.json` | `bde8085b77066cbf449a1b17d0a1afa77f2dcbcb4ea1a82f7d0dde68f6fe504f` |
| `experiments/colab/post_stage_b_stage4_experiments.ipynb` | `53b35f508f9a4c783ba951e409c0cf324b9992732f629b65ac06eb587f50a219` |
| `experiments/colab/configs/post_stage_b_stage4_option_a_config.json` | `518ec0cf56d52619fe5f24da7ef18a4b077466d38058c4d2538cd605c6910ecc` |
| `experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json` | `03a5f10d433f657af82d1af401774fd40d9d36209d35e13a14bc934086416180` |
| `src/bangla_smish/post_stage_b_tokenizer_gate.py` | `fee128773d761875f7b9d7f892c6c8035720c47cd667466b591b6ab81de24627` |
| `scripts/run_stage4_capped.py` | `12848fa63b00023ff2be1d4018581d1d37cea17756743551ed6a8c3bc8848504` |

*(Confirmed zero model training, zero Colab execution, and zero Google Drive modifications.)*

---

## Section G — Missing Evidence in `post_stage_b_stage4_final_remediation_report.md` & Final Readiness Matrix

### 1. Evidence Gaps Found in `reports/post_stage_b_stage4_final_remediation_report.md`

While the underlying implementation and test runs are solid, the final remediation report itself (`reports/post_stage_b_stage4_final_remediation_report.md`, `L1-L83`) omitted or misstated five traceability details that are supplied cleanly below:

1. **Incomplete File Inventory (`L72-L75`):** Section 6 listed only the 4 added and 10 modified files touched during the final pass, omitting the complete cumulative Stage 4 working-tree delta.
2. **Abbreviated Shell Commands (`L61-L68`):** Section 5 used brace expansion / `...` shorthand instead of verbatim copy-paste reproducible commands.
3. **No Persisted Telemetry JSON in `reports/`:** `scripts/run_stage4_capped.py` supports `--telemetry-out`, which was omitted; CPU telemetry was saved only inside conversation task logs (`task-947.log`, `task-951.log`).
4. **Line-Number Drift (`L18`):** Cites `_refresh_local_ckpt_commit_file` at `#L666-L673, #L781`, whereas the actual lines in `src/bangla_smish/post_stage_b_checkpointing.py` are **`#L685-L694`** and **`#L804`**.
5. **No SHA256 Digest List for the Updated 34-File Portable Kit:** Newly added 34th kit file `src/bangla_smish/post_stage_b_tokenizer_gate.py` and re-hashed modified files were not listed by SHA256 in the remediation report.

### 2. Complete Cumulative Stage 4 Modified / Added File Inventory

Tracked Modified (`3`):
- `CHECKPOINT.md`
- `pyproject.toml`
- `src/bangla_smish/post_stage_b_runner.py`

Untracked Source, Scripts, Tests, Configs, Notebook & Reports (`30` project files + ephemeral `response.md` + `this review report`):
- `experiments/colab/configs/post_stage_b_stage4_option_a_config.json`
- `experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json`
- `experiments/colab/post_stage_b_stage4_experiments.ipynb`
- `reports/post_stage_b_stage3_readiness_review.md`
- `reports/post_stage_b_stage4_colab_benchmark_plan.md`
- `reports/post_stage_b_stage4_final_independent_acceptance_review.md` *(created by this review)*
- `reports/post_stage_b_stage4_final_readiness.md`
- `reports/post_stage_b_stage4_final_remediation_report.md`
- `reports/post_stage_b_stage4_implementation_report.md`
- `reports/post_stage_b_stage4_independent_acceptance_review.md`
- `reports/post_stage_b_stage4_integrity_report.md`
- `reports/post_stage_b_stage4_remediation_report.md`
- `reports/post_stage_b_stage4_statistical_validation.md`
- `reports/post_stage_b_stage4_test_report.md`
- `scripts/generate_post_stage_b_stage4_notebook.py`
- `scripts/prepare_stage4_colab_kit.py`
- `scripts/run_stage4_capped.py`
- `scripts/validate_stage4_fixed_bootstrap.py`
- `scripts/verify_stage4_integrity.py`
- `src/bangla_smish/post_stage_b_checkpointing.py`
- `src/bangla_smish/post_stage_b_cohorts.py`
- `src/bangla_smish/post_stage_b_colab_deps.py`
- `src/bangla_smish/post_stage_b_ensemble.py`
- `src/bangla_smish/post_stage_b_integrity.py`
- `src/bangla_smish/post_stage_b_stats.py`
- `src/bangla_smish/post_stage_b_tokenizer_gate.py`
- `tests/test_stage4_checkpointing_and_kit.py`
- `tests/test_stage4_ensemble_and_stats.py`
- `tests/test_stage4_final_remediation.py`
- `tests/test_stage4_option_a_and_cohorts.py`
- `tests/test_stage4_remediation.py`
- `response.md` *(ephemeral handoff document)*

Gitignored Generated Artifacts (`3` JSON files in `reports/`):
- `reports/protected_baseline_282.json`
- `reports/stage4_frozen_test_masks.json`
- `reports/stage4_statistical_validation_results.json`

### 3. Exact Reproducible Verification Commands (Under Hardware 25% CPU Cap)

```bash
# 1. Rebuild Stage 4 notebook deterministically
systemd-run --user --scope -p CPUQuota=300% taskset -c 0-2 \
  .venv/bin/python scripts/generate_post_stage_b_stage4_notebook.py

# 2. Enforce cap probe + run the 27 targeted final-remediation tests
systemd-run --user --scope -p CPUQuota=300% taskset -c 0-2 \
  .venv/bin/python scripts/run_stage4_capped.py -- \
  .venv/bin/python -m unittest tests.test_stage4_final_remediation -v

# 3. Enforce cap probe + run all 90 Stage 4 regression tests + full 282/34/4-mask verification
systemd-run --user --scope -p CPUQuota=300% taskset -c 0-2 \
  .venv/bin/python scripts/run_stage4_capped.py -- bash -c '
    set -euo pipefail
    .venv/bin/python -m unittest \
      tests.test_stage4_option_a_and_cohorts \
      tests.test_stage4_ensemble_and_stats \
      tests.test_stage4_checkpointing_and_kit \
      tests.test_stage4_remediation \
      tests.test_stage4_final_remediation -v
    .venv/bin/python scripts/verify_stage4_integrity.py --mode full-repo --verify-masks
  '
```

### 4. Four-Bucket Final Readiness Matrix

| Bucket | Verified / Gated Items |
|---|---|
| **1. Locally verified** | Hard 25% CPU cap (`CPUQuota=300%` + `taskset -c 0-2`, `<= 8.61%` host CPU across probe, 27 targeted tests, and 90 full regression tests); 282-file protected baseline & 13 canonical inputs (`0` Category A edits); live re-derivation of all 4 frozen evaluation masks; 34-file portable kit contract, strict `.DS_Store`/`desktop.ini`/`.ipynb_checkpoints` filtering, and rogue-file rejection; `CONFIRMATORY_DECLARATIONS_ENABLED = False` & byte-identical statistical validation JSON; A/B/C/D step-budget metadata; exact stale-writer ID & corrupt-file lease confirmation; local `_CHECKPOINT_COMMIT` refresh; surfaced `prune_errors`; sole-recoverable & pinned-best checkpoint preservation. |
| **2. Mocked-subprocess verified** | Runner interruption/resume, checkpoint pruning, stale/corrupt lease takeover acceptance & refusal, local vs. Drive commit separation, and the 32-pair (`P01..P32`) tokenizer collision gate (`256/256` with `_MockTokenizer` under Option A and mocked `normalizer`). |
| **3. Colab-only unverified** | Live HuggingFace downloads & real tokenization collision check across all 4 pinned commits + `normalizer` (note: T4 benchmark Cell 5 runs `canon_masked` only = `128` checks unless `raw` is also included); live Colab `pip install` & CUDA PyTorch preservation; live Google Drive FUSE mount/sync latency & multi-session contention; empirical T4 VRAM headroom, runtime per unit, and `CU/hr` burn rate. |
| **4. Requires additional owner authorization** | Five-unit T4 Colab benchmark (`experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json`); any Google Drive staging/artifact writes; any modification to protected Category A files or frozen data; full 160-unit Option A training grid (strictly **`NO-GO`** pending T4 benchmark & separate owner authorization); any re-enabling of confirmatory superiority declarations. |

---

## Stop Condition

Stage 4 local independent review is complete (**`CONDITIONALLY ACCEPTED`**). Execution stops here at the independent-review gate. No GPU workload or T4 benchmark execution is authorized or requested here.
