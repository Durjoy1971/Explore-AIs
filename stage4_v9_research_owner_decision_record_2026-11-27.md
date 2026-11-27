# Stage 4 v9 — Consolidated Research Owner Decision Record (Gate A-1 Reconciliation)

> **STATUS: DRAFT — NOT SIGNED — NO AUTHORIZATION GRANTED.**  
> **Gate A-1 remains NOT AUTHORIZED. Gate A-2 remains UNAUTHORIZED. Stage 4 v9 remains NOT FINALIZED.**  
> This document is an **unsigned planning and authorization-record draft** prepared for Research Owner review. It records **zero** Research Owner approvals and infers **zero** authorization. All described protections, compatibility behaviors, and Option S2 controls are **proposed specifications/static analyses—not yet implemented or executed**. No source code, configuration, test, dataset, frozen artifact, historical kit (`v3–v8`), or `response-local.md` was modified; no tests, runtime probes, package installations, key generation, signing, GPU workloads, commits, or pushes were executed for this audit revision. Every unresolved choice is explicitly labeled **`OWNER DECISION REQUIRED`**.

---

## 0. Document Provenance & Preserved Baseline

| Field | Verified Value |
|---|---|
| **Document Date** | `2026-11-27` |
| **Document Path** | `stage4_v9_research_owner_decision_record_2026-11-27.md` *(accompanying record; preserves `response-local.md` unmodified under `A6`)* |
| **Repository & Branch** | `https://github.com/Durjoy1971/Explore-AIs` (`Durjoy1971/Explore-AIs`), branch `main` |
| **Primary Specification (Untouched)** | `response-local.md` |
| **Verified `response-local.md` SHA-256** | `a62ce62add5a39acd7218f5d1a12f92a2154853dd87c0feeb7d3c7d81146e1ff` (`178,154` bytes) |
| **Reviewed Code Baseline Commit (`main`)** | `720e43bac9a58e5acf9d1b197f9d96085c1b24ea` *(README-only commit over baseline `5e1da6b62823fd41feb5d9e7aa7952d591a426b0` cited in `response-local.md`; kept strictly separate from subsequent decision-record commits—all code, configs, tests, datasets, and historical kits remain identical to `720e43b`)* |
| **Initial Decision-Record Introduction Commit (`main`)** | `a53ab9f4e75c47f8fc76b014f7c8979bb6e603e5` *(verified diff `720e43b..a53ab9f`: adds **only** `stage4_v9_research_owner_decision_record_2026-11-27.md`, `+366` lines; blob `7e65c846272c318ae1dec0dbf8364dad91115844`, verified SHA-256 `d4b461d972e836e16c5434523afab7d2b32e6be712f678762ca44f0c0a8ffcb4`; zero changes to code, configs, tests, datasets, or `response-local.md`)* |
| **Subsequent Committed Update / Latest `main` HEAD** | `5ee875cee301a67828cd0a451790350065156ea6` *(verified diff `a53ab9f..5ee875c`: modifies **only** `stage4_v9_research_owner_decision_record_2026-11-27.md`, `+151/-115` lines; zero changes to code, configs, tests, datasets, or `response-local.md`)* |
| **Currently Committed Decision-Record Revision (at `5ee875c`)** | Git blob `6b24dac2d6212bc913158f2f6a88e6e1bf72e950` *(verified file SHA-256 at `5ee875c`: `27704ef441e328ef0833ef0624cb882e902b00ab026c78a68c526c0e2f59e06e`)* |
| **Current Working Tree State** | Targeted post-`5ee875c` correction edit on top of committed blob `6b24dac2d6212bc913158f2f6a88e6e1bf72e950` *(uncommitted locally; final committed/signed SHA-256 and Git commit/blob SHA are recorded externally upon Owner commit and signing)* |
| **Authorization Status** | **`DRAFT — NOT SIGNED — NO AUTHORIZATION GRANTED.`** |
| **`CONFIRMATORY_DECLARATIONS_ENABLED`** | **`False`** *(unchanged throughout Stage 4 v9)* |

**Reconciliation Purpose:** This audited draft resolves all authorization, technical, and trust-boundary issues identified across both read-only Gate A-1 audits:
1. Exact provenance separating the reviewed code baseline (`720e43b`) from the initial decision-record introduction commit (`a53ab9f`, blob `7e65c846...`) and the subsequent committed update at `main` HEAD (`5ee875c`, blob `6b24dac2...`, SHA-256 `27704ef4...`), while preserving unmodified `response-local.md` (`a62ce62a...`).
2. One unambiguous, fail-closed authorization-recording procedure binding `D01–D18`, `A1–A5`, `A6`, and Section 12 with zero pre-filled Owner approvals.
3. Explicit per-file `A1` authorization (zero implicit file permissions) and static source analysis of Option B vs. Option A compatibility across all 9 test files (explicitly distinguishing module-level `__getattr__` [PEP 562] compatibility from ordinary function-local lazy imports replacing the two eager top-level `src/` imports).
4. Accurate I/O-gating specification for optional `src/bangla_smish/post_stage_b_model_view_audit.py` (`include_heldout=False`).
5. Explicit separation in `A5` between cryptographic/structural integrity + frozen-mask digest re-derivation (`--verify-only --skip-tfidf`) vs. corpus-wide TF-IDF near-duplicate screening (data-quality analysis), while keeping `D17` closed (`DEFER`).
6. Explicit `A3` complete execution trust boundary covering source, tests, scripts, configurations, interpreter, and dependencies (addressing the `0775` group-writable repository ancestors and `.venv`, preventing silent `A5` execution from an unprotected repository tree, disclosing the same-UID limitation of `0700` permissions, requiring recursive directory/regular-file/symlink checks with fail-closed enforcement under Mode 2, and pinning `scripts/run_stage4_capped.py` by SHA-256 without adding new source files to `A1`).
7. Removal of premature-verification claims so proposed designs and static analyses are never presented as implemented or tested.

---

## 1. Corrected `D01–D18` Atomic Decision Matrix & Authorization-Recording Procedure

> **STATUS: DRAFT — NOT SIGNED — NO AUTHORIZATION GRANTED.**

### 1.1 Unambiguous Fail-Closed Authorization-Recording Procedure (`response-local.md:625-629, 685`)
1. **Advisory recommendations confer zero authority.** The `Advisory Draft State` column below and the advisory notes in Sections 2–8 are planning recommendations only; they do **not** record any Research Owner decision.
2. **Every decision requires exactly one canonical Owner state.** To record decisions, the Research Owner MUST complete **all three** synchronized locations:
   - **Location A (Section 1 Matrix `D01–D18`):** Write **exactly one** canonical state (`APPROVE`, `REJECT`, or `DEFER`) in the blank **`Owner State (OWNER DECISION REQUIRED)`** column for every row `D01–D18`. Never place scope text in the `Owner State` slot.
   - **Location B (Sections 2–6 Gate A Permissions `A1–A5` & unresolved choices):** Write **exactly one** canonical state (`APPROVE`, `REJECT`, or `DEFER`) in the blank **`Owner State (OWNER DECISION REQUIRED)`** field for each of `A1`, `A2`, `A3`, `A4`, and `A5`, and explicitly select each unresolved choice (`Option B` vs. `Option A`; `INCLUDE` vs. `EXCLUDE` `post_stage_b_model_view_audit.py`; `A5` `--skip-tfidf` vs. separate `A5-TFIDF` sub-authorization; `A3` execution trust-boundary mode).
   - **Location C (Section 12 Copy-Ready Declaration):** Complete every `[OWNER DECISION REQUIRED]` slot so that all `D01–D18` states, `A1–A5` states, resolved scope selections, and the exact `D10` statement match Locations A and B **verbatim**, then date and sign.
3. **Fail-Closed Rule:** Any blank `Owner State` field, any unselected `OWNER DECISION REQUIRED` option, any duplicate/conflicting state in one row, or any mismatch between Section 1 (`D01–D18`), Sections 2–6 (`A1–A5`), and Section 12 grants **zero authority** (`Gate A-1` remains unauthorized) until resolved explicitly in writing by the Research Owner.

### 1.2 `D01–D18` Decision Matrix

| ID | Decision Topic | Advisory Draft State *(Non-Binding)* | Normalized Scope, Evidence, and Conditions | Owner State (`APPROVE / REJECT / DEFER` — `OWNER DECISION REQUIRED`) |
|---|---|---|---|---|
| **D01** | **Specification sign-off** | `APPROVE` | Accept saved specification `response-local.md` (`sha256: a62ce62add5a39acd7218f5d1a12f92a2154853dd87c0feeb7d3c7d81146e1ff`): Option S2, dev-vs-eval separation, always-inert Candidate Release Notebook, Trusted Control Notebook, Verified Python Execution Child, Option D launcher design, `-I -S -B -X pycache_prefix=... -c ...` startup contract, and proposed acceptance criteria. **Conveys zero engineering authority.** | `__________` |
| **D02** | **Engineering platform selection** | `APPROVE` | Select local self-managed Ubuntu workstation `bs-support` (`Ubuntu 26.04.1 LTS`, `Linux 7.0.0-43-generic x86_64`) as the engineering planning and local CPU validation platform. Actual engineering work and checks are strictly bounded by `D10` (`A1–A6`). | `__________` |
| **D03** | **Parent runtime provenance** | `DEFER` | Deferred pending production/research target selection (`D12`) and Gate A-2 controller work. Zero parent/controller/kernel startup provenance approved in Gate A-1. | `__________` |
| **D04** | **Python/native runtime trust profile** | `DEFER` | Deferred pending deployment target selection (`D12`) and Gate A-2 runtime-binding work. Recording `/usr/bin/python3.12` and `/usr/bin/openssl` digests in `A3` does **not** approve a `D04` runtime trust profile. | `__________` |
| **D05** | **Interpreter execution binding** | `APPROVE` | Select independently protected absolute-path execution binding as the architectural default design (optional executable-FD binding not selected). Actual interpreter/native closure acceptance remains deferred under `D04`. | `__________` |
| **D06** | **Trusted bootstrap and launcher identity** | `DEFER` | Deferred until Gate A-2 produces concrete controller, fixed `-c` loader, and launcher artifacts with independently verifiable out-of-band digests/pins (Option D design is accepted conceptually under `D01` only). | `__________` |
| **D07** | **Ed25519 signing and release-key custody** | `APPROVE` | Retain offline Research Owner Ed25519 private-key custody and anti-rollback policy **only**. **Conveys zero authority to generate keys or sign any release or fixture.** | `__________` |
| **D08** | **Section B.3 strict filesystem profile** | `APPROVE` | Accept Section B.3 strict atomic descriptor-based filesystem architecture as the design default **only**. **Section B.3 implementation (`openat2`, `statx`, `OPENAT2_REGULAR`, `fstatfs`, ABI dispatch, Tests 8–28) is explicitly DEFERRED to Gate A-2.** | `__________` |
| **D09** | **Filesystem or syscall exceptions** | `DEFER` | **Zero** filesystem, mount-transition, `OPENAT2_REGULAR` fallback, or Option A-only seccomp exceptions granted. Strict baseline remains in force. | `__________` |
| **D10** | **Gate A engineering authorization** | **`APPROVE`** | **`Gate A-1 only, subject to A1–A6.`** Grants zero authority for Gate A-2 items (full B.3 syscalls, Trusted Control Notebook / Option D launcher, `verify_stage4_release_attestation.py`, notebook regeneration, fixture signing), dependency installs, real release signing, live Colab/target probes, deployment acceptance, GPU runs, Track A/B, held-out evaluation, or confirmatory research. *(D10 approval alone grants no authority without explicit `A1–A5` Owner states and `A6` exclusions below.)* | `__________` |
| **D11** | **Dependency provisioning** | `DEFER` | Deferred (`0` `pip`/`uv` package installations in Gate A-1; declarative dependency specs only under `A1`). | `__________` |
| **D12** | **Runtime/deployment acceptance** | `DEFER` | Deferred. Hosted-Colab runtime provenance and execution binding remain unresolved; `A3` (`bs-support`) conveys zero `D12` deployment authority. | `__________` |
| **D13** | **GPU smoke-test authorization (Gate B)** | `DEFER` | Deferred. Creating and locally validating `gpu_smoke_test_config_v2.json` under `A1`/`A2` conveys zero authority to run GPU smoke tests. | `__________` |
| **D14** | **T4 benchmark authorization (Gate C)** | `DEFER` | Deferred. Creating and locally validating `post_stage_b_stage4_t4_benchmark_config_v2.json` under `A1`/`A2` conveys zero authority to run T4 benchmarks. | `__________` |
| **D15** | **Track A authorization** | `DEFER` | Deferred pending separate signed statistical preregistration (Wilson UCB policy, Holm-24 `D vs C`, CR2 vs. `G/(G-1)`, and fresh owner-committed seeds). Zero edits to `post_stage_b_stats.py` or `validate_stage4_fixed_bootstrap.py`. | `__________` |
| **D16** | **Track B authorization** | `DEFER` | Deferred pending target FPR estimand selection and disjoint external Normal cohort sampling protocol. | `__________` |
| **D17** | **Held-out evaluation authorization** | `DEFER` | Closed (`NO-GO`). `A2` mocked tests and `A5` read-only integrity checks convey zero authority for trained-model inference, real test/OOD predictions, predictive metrics, threshold tuning, or held-out exports. | `__________` |
| **D18** | **Confirmatory research authorization** | `DEFER` | Closed. `CONFIRMATORY_DECLARATIONS_ENABLED` remains `False` throughout Stage 4 v9 and cannot be changed under v9 without a separate explicitly authorized specification revision (`response-local.md:664`). | `__________` |

**Normalized `D10` Wording Rule (Required Identical Phrasing If `D10` Is Approved by the Owner):**
- **Decision Matrix (`D10` row):** `Owner State` = `APPROVE` | `Scope` = `Gate A-1 only, subject to A1–A6.`
- **General `APPROVE` List (Section 12):** `D10 = APPROVE. Scope is Gate A-1 only, subject to A1–A6.`
- **Dedicated `Gate A` Field (Section 12):** `D10 = APPROVE. Scope is Gate A-1 only, subject to A1–A6.`

---

## 2. Exact `A1` Authorized-File Evaluation & Scope (`OWNER DECISION REQUIRED`)

> **Advisory Draft State:** `APPROVE` *(strictly limited to the exact file list selected and signed by the Research Owner below — **no file is authorized implicitly**).*  
> **Owner State (`APPROVE / REJECT / DEFER` — `OWNER DECISION REQUIRED`):** `__________`

### 2.1 Existing Source & Configuration Files Evaluated

| File Path | Why Modification Is Needed (Proposed Gate A-1 Scope) | Gate A-1 Task Requiring It | Essential or Optional | Conflicts With `D03/D04/D06/D08/D09/D11`? | Gate A-1 Status |
|---|---|---|---|---|---|
| `src/bangla_smish/post_stage_b_data.py` | Split `DEV_ONLY_BUNDLE_FILENAMES` (`5` parquets) vs. `FULL_CONFIRMATORY_BUNDLE_FILENAMES` (`7` parquets, keeping `BUNDLE_FILENAMES` alias); add `dev_only_manifest.json` builder & fail-closed non-override `verify_dev_only_manifest()`; externalize `banglaphish2026_dedup.parquet` (`16aaa636...`) digest and held-out `EXPECTED_COHORT_COUNTS` into `reports/stage4_confirmatory_metadata_registry.json` (exposed via module-level `__getattr__` [PEP 562] under Option B so zero static held-out literals remain in `src/`). | Option S2 inventory split, strict `dev_only_manifest.json` non-override, zero static held-out literals in `src/`. | **Essential** | **None.** Pure data/manifest logic; uses interim Option S2 `os.scandir()` (labeled non-B.3) when `require_physical_absence=True`; zero syscall probes, zero installs. | **Included in Core `A1`** |
| `src/bangla_smish/post_stage_b_cohorts.py` | Externalize held-out cohort literals (`EXPECTED_SD_TEST_ROWS`, `EXPECTED_SD_TEST_FAMILIES`, `AMBIGUOUS_TEST_ROW_IDS`, `AMBIGUOUS_TEST_FAMILY_IDS`, `TEMPLATE_STRESS_TEST_FAMILY_IDS`, `COHORT_SPECS`) into `reports/stage4_confirmatory_metadata_registry.json` (exposed via module-level `__getattr__` [PEP 562] under Option B, failing closed when the registry is absent in `v9_dev_only`). | Option S2 zero static held-out literals in shared `src/`. | **Essential** | **None.** Pure metadata externalization; designed to fail closed in `v9_dev_only` while preserving local-repo `verify_stage4_integrity.py`. | **Included in Core `A1`** |
| `src/bangla_smish/post_stage_b_tokenizer_gate.py` | Externalize the `18` test-touching candidate pairs into `reports/stage4_confirmatory_metadata_registry.json` (exposed via module-level `__getattr__` [PEP 562] for `STAGE3_32_CANDIDATE_PAIRS` under Option B); keep `DEV_ONLY_14_PAIR_IDS` (`14` pairs / `24` unique dev `row_id`s) and `DEV_ONLY_TOKENIZER_PARQUETS`; add `load_dev_only_14_pairs(bundle_dir, require_physical_absence=...)` reading only dev parquets; add `pair_scope: "dev_only_14" \| "all_32"` to `run_allowed_view_tokenizer_collision_gate()`; remove redundant `eval_banglaphish_ood.parquet` read from `load_documented_32_pairs()`. | Option S2 dev-only `14`-pair/`24`-row loader & `112`-check gate schema. | **Essential** | **Compatible with `D11 = DEFER`:** Code supports real HF tokenizers later; Gate A-1 executes mocked tokenizers only (`112` planned reported separately from mocked locally executed). Zero conflicts with `D03/D04/D06/D08/D09`. | **Included in Core `A1`** |
| `src/bangla_smish/post_stage_b_runner.py` | Enforce `run_mode in {"gpu_smoke_test", "t4_benchmark"} => run_heldout_eval is False` in `PostStageBConfig.__post_init__()`; add `benchmark_dev_timing_eval: bool = False` and `708`-row validation timing-only `t4_benchmark_v2` harness contract (dynamic buckets `[1–32],[33–64],[65–96],[97–128]`, fixed-128, `R=5` raw arrays + summary stats + `rows_per_sec`/`sec_per_1000_rows`, keeping `SYNTHETIC_WORKLOAD_FIXTURE_ENABLED = False`); **remove eager top-level `STAGE3_32_CANDIDATE_PAIRS` import (`line 83`)** and replace it with an ordinary function-local lazy import (not PEP 562) executed only for `pair_scope="all_32"` so importing the runner inside `v9_dev_only` succeeds without the confirmatory registry; externalize `1698`/`5416` literals (including `line 2692` and `_build_t4_benchmark_workload_fixture()`); add `bundle_profile="dev_only"` strict `dev_only_manifest.json` + `dev_only_14` tokenizer-gate validation; add stratified 3-class fallback in `_split_for_unit` for `smoke_max_*`; tag OOF `mcnemar_vs_exp_a` diagnostic-only (`CONFIRMATORY_DECLARATIONS_ENABLED` stays `False`). | `v2` smoke & T4 config validation, Option S2 dev-only pipeline binding, removing static held-out literals and eager confirmatory imports from `src/`. | **Essential** | **None.** Zero edits to statistical gate (`post_stage_b_stats.py`); zero GPU runs (`D13/D14 = DEFER`); zero installs (`D11 = DEFER`); no launcher/B.3 syscalls (`D03/D04/D06/D08/D09`). | **Included in Core `A1`** |
| `src/bangla_smish/post_stage_b_integrity.py` | **Remove eager top-level `PROTECTED_INPUT_DIGESTS` import (`line 17`)** and replace it with an ordinary function-local lazy import (not PEP 562) inside `build_protected_baseline_manifest()` (`line 176`) so importing `post_stage_b_integrity` inside `v9_dev_only` succeeds without the confirmatory registry; add disposable dev-only kit manifest builder and interim Option S2 `os.scandir()` physical-absence verifier (`verify_v9_dev_only_kit()`) emitting mandatory `b3_syscall_hardening_implemented = False` and `release_eligible = False`; preserve `build_portable_colab_kit_manifest()` and `verify_portable_colab_kit()` backward-compatible with untouched `34`-file `v3–v8` kits (and supporting `35`-file newly exported full kits containing `reports/stage4_confirmatory_metadata_registry.json`). | Option S2 dev-only staging/absence verification + preserving `v3–v8` historical verification under `A5` + removing eager confirmatory imports. | **Essential** | **Reconciled with `D08`/`D09`:** Explicitly labeled interim Option S2 inventory/absence enforcement only (`b3_syscall_hardening_implemented = False`, `release_eligible = False`); zero `openat2`/`statx`/`OPENAT2_REGULAR`/`fstatfs` syscall probes or B.3 compliance claims. | **Included in Core `A1`** |
| `scripts/prepare_stage4_colab_kit.py` | Add `--profile dev_only` (disposable pre-B.3 unsigned Option S2 dev-only export path) staging only the `5` dev parquets, `dev_only_manifest.json`, `18` `src/` modules, `v2` dev configs, `pyproject.toml`, slim `reports/stage4_dev_statistical_governance.json`, and unsigned `stage4_kit_manifest.json` (omitting unregenerated v8 `.ipynb`, attestations/signatures, held-out files, and `tests/`); keep existing full-kit export path intact (copying `reports/stage4_confirmatory_metadata_registry.json` into newly exported full kits under Option B so isolated full-kit verification works). | Real dev-only kit exporter verification required by `A2`. | **Essential** | **Reconciled with `A4`/`D06`/`D07`/`D08`:** Produces unsigned, pre-B.3, non-release-eligible disposable artifacts only; zero keygen, zero `.sig`, zero attestation script, zero B.3 syscalls. | **Included in Core `A1`** |
| `src/bangla_smish/post_stage_b_colab_deps.py` | Pin `ftfy==6.0.3`, `emoji==1.4.2`, and `wcwidth>=0.2.13,<1` in `required_packages_for_models()` (currently unpinned strings at `line 99`) so all `24` runtime distributions match `pyproject.toml [gpu]` and `response-local.md:205` verbatim (retaining `transformers>=4.57.1,<5`). | `24`-distribution dependency-declaration sync. | **Essential** | **Compatible with `D11 = DEFER`:** Declarative strings only; `0` `pip`/`uv` installs executed locally. | **Included in Core `A1`** |
| `pyproject.toml` | Sync `[project.optional-dependencies].gpu` with all `24` verified runtime distributions (adding `normalizer @ git+https://github.com/csebuetnlp/normalizer@d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9`, `ftfy==6.0.3`, `emoji==1.4.2`, `wcwidth>=0.2.13,<1`, `numpy>=1.26`, `pandas>=2.1`, `scipy>=1.11`, `scikit-learn>=1.4`, `pyarrow>=15.0`, `requests>=2.31`, while retaining `torch>=2.6` and **`transformers>=4.57.1,<5`**). | `24`-distribution dependency-declaration sync preserving `test_stage4_final_remediation.py:2085`. | **Essential** | **Compatible with `D11 = DEFER`:** Declarative metadata only; zero package installs or lockfile mutations. | **Included in Core `A1`** |
| `src/bangla_smish/post_stage_b_model_view_audit.py` | **I/O-gating guard (NOT merely an in-memory DataFrame guard):** Currently `audit_model_view_overlaps()` (`line 38`) unconditionally invokes `load_heldout_eval_sets(bundle_dir)` (`post_stage_b_data.py:850-858`), which reads `eval_smishdetect_test.parquet` and `eval_banglaphish_ood.parquet` from disk. Adding `include_heldout: bool = True` (preserving legacy default for `test_stage4_option_a_and_cohorts.py:101`) MUST enforce that `include_heldout=False` **never** invokes `load_heldout_eval_sets()`, **never** opens/reads/probes `eval_smishdetect_test.parquet` or `eval_banglaphish_ood.parquet`, still audits CV and counterfactual (and permitted dev-validation) overlaps using dev-only parquets only, and has explicit I/O-sentinel tests in `tests/test_stage4_v9_remediation.py`. | Defense-in-depth I/O isolation for direct callers of `audit_model_view_overlaps()` on dev-only bundles (`STAGE4_V9_SESSION_HANDOFF.md:45`; module is included in the `18`-file staged `src/` tree). | **Optional** | **None.** Purely prevents held-out parquet I/O when `include_heldout=False`; zero conflicts with `D03/D04/D06/D08/D09/D11`. | **`OWNER DECISION REQUIRED`** *(Recommended: **INCLUDE**)* |

### 2.2 Exact New Configuration & Report Files Permitted Under `A1`

| Exact New Path | Purpose | Essential or Optional | Conflicts With `D03/D04/D06/D08/D09/D11`? |
|---|---|---|---|
| `experiments/colab/configs/gpu_smoke_test_config_v2.json` | Matrix M1 dev-only smoke config (`run_name="post_stage_b_gpu_smoke_test_v3"`, `experiments=["B","D"]`, 4 pinned backbones with 40-hex `tokenizer_revisions`, `text_views=["canon_masked"]`, `folds=[0]`, `run_heldout_eval=false`, `confirmatory_mode=false`, `benchmark_workload_fixture_eval=false`, `benchmark_dev_timing_eval=false`, `smoke_max_*=64/32/32`). Preserves frozen v1 `gpu_smoke_test_config.json` untouched. | **Essential** | **None.** Config file only; `D13` (GPU smoke execution) remains `DEFER`. |
| `experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config_v2.json` | `708`-row timing-only dev config (`run_mode="t4_benchmark"`, `run_heldout_eval=false`, `benchmark_workload_fixture_eval=false`, `benchmark_dev_timing_eval=true`, `confirmatory_mode=false`, 4 pinned backbones, zero `1698`/`5416`/`7114` literals). Preserves frozen v1 `post_stage_b_stage4_t4_benchmark_config.json` untouched. | **Essential** | **None.** Config file only; `D14` (T4 benchmark execution) remains `DEFER`. |
| `reports/stage4_confirmatory_metadata_registry.json` | Externalized repository-only / `full_confirmatory`-only registry holding held-out and source-OOD static literals removed from shared `src/` (`banglaphish2026_dedup.parquet` digest, held-out cohort counts/row IDs/family IDs/specs, `18` test-touching Stage 3 candidate pairs, legacy fixture row targets). **Physically excluded from `v9_dev_only` kits.** | **Essential** | **None.** Strictly outside Category A `282/282` protected files and excluded from dev-only exports. |
| `reports/stage4_dev_statistical_governance.json` | Slim dev-only statistical governance summary staged inside `v9_dev_only` (recording `confirmatory_declarations_enabled: false` and `statistical_validation_status: "FAIL_OR_INCONCLUSIVE_CONFIRMATORY_DISABLED"` with **zero** held-out counts, `emp_1687` strings, test row IDs, or OOD hashes). Preserves `reports/stage4_statistical_validation_results.json` untouched. | **Essential** | **None.** Strictly outside Category A `282/282`; `D15`/`D18` remain `DEFER`. |

### 2.3 Exact Test Files Permitted Under `A1` (`OWNER DECISION REQUIRED` — See Section 8)

- **Under Recommended Option B (Minimal Scoped Compatibility — 4 test files total):**
  1. `tests/test_stage4_v9_remediation.py` *(CREATE NEW — Essential; includes explicit I/O-sentinel test for `audit_model_view_overlaps(include_heldout=False)` if `post_stage_b_model_view_audit.py` is included)*
  2. `tests/test_post_stage_b_pipeline.py` *(MODIFY lines `206, 229` — Essential: v1 smoke config fails closed on `run_heldout_eval=True`)*
  3. `tests/test_stage4_v8_remediation.py` *(MODIFY line `1027` — Essential: v1 smoke config fails closed on `run_heldout_eval=True`)*
  4. `tests/test_stage4_final_remediation.py` *(MODIFY lines `495, 851` ONLY — Essential when newly exported full kits include `reports/stage4_confirmatory_metadata_registry.json` [`34 -> 35` runtime files] while untouched `34`-file `v3–v8` kits remain accepted)*
- **Under Alternative Option A (Full Legacy Test Rewrite — 9 test files total):**
  All 9 files listed in Section 8 (`test_stage4_v9_remediation.py` + all 8 existing Stage 4 / Post-Stage B test files).

**Explicitly Prohibited Under `A1` (Zero Edits Permitted):**
- `scripts/run_stage4_capped.py` *(pinned unmodified for `A3` trust boundary; see Section 4.2)*
- `src/bangla_smish/post_stage_b_stats.py`, `scripts/validate_stage4_fixed_bootstrap.py` (`D15 = DEFER`)
- `scripts/verify_stage4_release_attestation.py` (Gate A-2; `A4/D06 = DEFER`)
- `scripts/generate_post_stage_b_stage4_notebook.py`, `experiments/colab/post_stage_b_stage4_experiments.ipynb` (Gate A-2; `D03/D04/D06 = DEFER`)
- `response-local.md` and every `A6` protected artifact.

---

## 3. `A2` Approved Local Test Execution

> **Advisory Draft State:** `APPROVE`  
> **Owner State (`APPROVE / REJECT / DEFER` — `OWNER DECISION REQUIRED`):** `__________`

All `A2` executions MUST run locally on `A3` (`bs-support`, CPU-only) under the verified `A3` trust boundary (Section 4.2) and wrapped by the pinned copy of `scripts/run_stage4_capped.py` (`<= 25%` host CPU: `systemd-run --user --scope -p CPUQuota=300% -- taskset -c 0-2 ...`):

1. **Full local `unittest` suite (CPU + mocked Colab-only dependencies):**
   ```text
   .venv/bin/python -m unittest discover -s tests -v
   ```
   Covers existing regression suites plus new `tests/test_stage4_v9_remediation.py` (Option S2 I/O/syscall sentinels blocking forbidden held-out filenames across dev codepaths—and across `audit_model_view_overlaps(include_heldout=False)` if approved—`load_dev_only_14_pairs()` resolving exact `14` pairs / `24` unique `row_id`s, strict `dev_only_manifest.json` non-override rejection when only `dataset_manifest.json` exists, mocked `run_allowed_view_tokenizer_collision_gate(pair_scope="dev_only_14")`, mocked `gpu_smoke_test_v2` & `t4_benchmark_v2`, `v1` smoke fail-closed check, `24`-distribution sync check, confirmatory registry fail-closed absence check, and interim Option S2 `os.scandir()` physical-absence / symlink / nested-forbidden / unexpected-parquet / `.pyc` rejection tests).
2. **Direct standalone pipeline test invocation:**
   ```text
   .venv/bin/python tests/test_pipeline.py
   ```
   Required because `tests/test_pipeline.py` defines standalone functions invoked only via `__main__` (not discovered by `unittest discover`).
3. **Read-only frozen bundle `--verify-only` checks (`v1` and `v2`):**
   ```text
   .venv/bin/python scripts/prepare_post_stage_b_bundle.py --verify-only
   .venv/bin/python scripts/prepare_post_stage_b_bundle.py --bundle-dir data/processed/post_stage_b_bundle_v2 --verify-only
   ```
   *(Read-only integrity checks cross-referenced under `A5`; zero model evaluation.)*
4. **Real disposable dev-only kit export & non-parquet metadata scan:**
   Execute `scripts/prepare_stage4_colab_kit.py --profile dev_only` into a fresh private `0700` directory under `/home/bs01415/.stage4_v9_work/` or an owner-`0700` `/tmp/` subdirectory, verify exact `5` allowed dev parquets, verify `b3_syscall_hardening_implemented == False` and `release_eligible == False`, verify clean `import` of `post_stage_b_runner` and `post_stage_b_integrity` inside the staged kit without `stage4_confirmatory_metadata_registry.json`, and scan every non-parquet file in the exported tree to confirm `0` held-out/source-OOD SHA-256s, test row/family IDs, counts (`1698`, `5416`, `7114`, `1687`, `1693`, `1675`), or held-out class/language breakdowns.

**Mandatory `A2` Finish Line & Honesty Reporting Rules:**
- **Zero live target probes** (no Google Colab session, no remote VM, no `openat2`/`statx`/`OPENAT2_REGULAR` kernel probes, no GPU workloads).
- **Mandatory separation in all reports:**
  - Report **`112` planned real HF tokenizer comparisons** (`14 pairs × 4 models × 2 views`) separately from **mocked locally executed** checks (`0` real HF comparisons executed in Gate A-1 because `D11 = DEFER`).
  - Report **mocked unit-test passes**, **real fail-closed refusals**, and **real non-mocked local verifications** in separate sections.
  - Never present mocked test outputs as empirical research or model-evaluation results.

---

## 4. `A3` Approved Validation Environment, Host Identity, & Execution Trust Boundary

> **Advisory Draft State:** `APPROVE`  
> **Owner State (`APPROVE / REJECT / DEFER` — `OWNER DECISION REQUIRED`):** `__________`

### 4.1 Observed Engineering Host Inventory

| Attribute | Observed `A3` Engineering Host Value |
|---|---|
| **Host & OS** | Local self-managed workstation `bs-support` — `Ubuntu 26.04.1 LTS`, `Linux 7.0.0-43-generic x86_64` |
| **Administrative Owner & Group** | `bs01415` (`uid=1001`, `gid=1001`; `/etc/group` and `/etc/passwd` currently list `bs01415` as the sole member of group `bs01415` [GID `1001`]; POSIX ACLs not audited) |
| **CPU / RAM / GPU Limits** | `12` logical CPUs strictly capped `<= 25%` (`<= 3` CPUs via pinned `run_stage4_capped.py`), `30 GiB RAM`, **no NVIDIA GPU** |
| **Repository & Observed Ancestor Modes** | `/` (`0755 root:root`), `/home` (`0755 root:root`), `/home/bs01415` (`0750 bs01415:bs01415`), `/home/bs01415/Documents` (`0755 bs01415:bs01415`), `/home/bs01415/Documents/Programming` (**`0775 bs01415:bs01415` — group-writable**), `/home/bs01415/Documents/Programming/ExploringAIs` (**`0775 bs01415:bs01415` — group-writable**), `.venv` (**`0775 bs01415:bs01415` — group-writable**), `scripts/` (`0755`), `scripts/run_stage4_capped.py` (`0644`), `src/` (`0755`), `tests/` (`0755`). |
| **Pinned CPU-Cap Wrapper Digest** | `scripts/run_stage4_capped.py` — `sha256: 12848fa63b00023ff2be1d4018581d1d37cea17756743551ed6a8c3bc8848504` *(must NOT be modified under `A1`)* |
| **Disposable Staging Paths** | Fresh owner-only `0700` directories strictly under `/home/bs01415/.stage4_v9_work/` *(requires `/home/bs01415` to remain single-user / no foreign group members or `0700`)* or an owner-`0700` subdirectory under sticky `01777 root:root` `/tmp/`. |
| **Observed Host Binary Inventory** | `.venv/bin/python -> /usr/bin/python3.12` (`CPython 3.12.3`, `sha256: e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f`); `/usr/bin/openssl` (`3.5.5`, `sha256: 1d72cbc5bf62be3f3dbd7626c295b63c5fdc1dd8439f323ac47998d8a062b4c9`). |
| **Permitted Use** | Exclusively `A1`, `A2`, and `A5` Gate A-1 local CPU engineering tasks. Excludes hosted Colab, remote hosts, GPU execution, and live target acceptance checks (`response-local.md:681`). |

### 4.2 `A3` Execution Trust-Boundary Finding & Required Controls (`OWNER DECISION REQUIRED`)
- **Audit Finding:** Because `/home/bs01415/Documents/Programming`, `/home/bs01415/Documents/Programming/ExploringAIs`, and `.venv/` are **`0775` (group-writable)** and `/home/bs01415` (`0750`) permits group traversal, the entire local execution closure—`scripts/run_stage4_capped.py` (`0644`), `src/`, `tests/`, `scripts/`, `experiments/colab/configs/`, `pyproject.toml`, and `.venv/` (interpreter symlink and installed site-packages)—is **NOT independently protected against modification or directory-rename substitution by another member of group `bs01415`** (and `scripts/run_stage4_capped.py` itself checks CPU affinity and cgroup `cpu.max` only—it performs no self-digest, ancestor-permission, or dependency-closure checks). Although `/etc/passwd` and `/etc/group` currently list `bs01415` as the sole member of group `1001`, group membership is mutable host state, POSIX ACLs are unaudited, and neither can substitute for verified execution-closure protection.
- **Required Controls Before Executing Any `A2` or `A5` Command (Zero New `A1` Source Files Added; Zero Permission Changes Performed Prior to Authorization):**
  1. **Pinned CPU-cap wrapper verification (required in both modes):** Verify immediately before every `A2`/`A5` invocation that the `scripts/run_stage4_capped.py` copy being invoked has exact SHA-256 **`12848fa63b00023ff2be1d4018581d1d37cea17756743551ed6a8c3bc8848504`** (staged as an owner-only `0400`/`0600` file inside the private `0700` execution closure under Mode 1, or verified in place after recursive closure verification under Mode 2).
  2. **Execution trust-boundary mode (`OWNER DECISION REQUIRED` — select one mode before signing):**
     - **`A3-Mode 1` (Recommended — Private Owner-Only Complete Execution Closure):**
       - **Complete execution closure:** Stage and verify **all executable and configuration inputs** inside a fresh owner-only `0700` directory beneath sticky `01777 root:root` `/tmp/` (e.g., `/tmp/stage4_v9_exec_<id>/`) or beneath `/home/bs01415/.stage4_v9_work/` (only if `/home/bs01415` is verified single-member or `0700`): **source (`src/`), tests (`tests/`), scripts (`scripts/`, including the `0400`/`0600` pinned `run_stage4_capped.py` copy verified against `12848fa63b00023ff2be1d4018581d1d37cea17756743551ed6a8c3bc8848504`), configurations (`experiments/colab/configs/`, `pyproject.toml`), interpreter, and dependencies**.
       - **Addressing the existing group-writable `.venv` (`0775`):** Mode 1 MUST NOT silently import or execute dependencies from the unprotected `0775` `.venv/`. Before running any `A2` or `A5` command, either:
         - **(1a) Independently protect the dependency environment** (verify that `/usr/bin/python3.12` [`sha256: e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f`], its system ancestor chain, and the entire `.venv/` tree—recursively across all directories, regular files, and symlink targets—carry zero group/world write bits and no untrusted symlink targets; note this does not constitute `D04` runtime approval, which remains `DEFER`), **OR**
         - **(1b) Include a verified private `0700` execution copy of the dependency environment** inside the private `0700` execution closure (e.g., an owner-only `0700` copy of the venv/site-packages bound to root-owned `0755` `/usr/bin/python3.12` verified against `e50d468e...`, with `PYTHONPATH` / `sys.path` / `VIRTUAL_ENV` pinned exclusively to the private closure and zero imports from the `0775` repository tree or `0775` `.venv/`).
       - **No silent `A5` execution from an unprotected repository tree:** All `A5` code (`scripts/verify_stage4_integrity.py`, `scripts/prepare_post_stage_b_bundle.py`, `src/bangla_smish/*`, pinned `run_stage4_capped.py`, interpreter, and site-packages) MUST execute exclusively from the verified private `0700` execution closure—never importing scripts, `src/`, or `.venv/` from the unprotected `0775` repository tree. Under Mode 1, the repository path is passed to `A5` **strictly as an inert read-only data target** (for hashing the `282` Category A files, verifying `v1`/`v2` bundles, and checking untouched `v3–v8` kits under an explicit single-operator no-concurrent-writer disclosure); if read-only data-target provenance must also be hardened against group-member tampering, apply Mode 2 hardening to the repository tree first.
       - **Explicit same-UID limitation of `0700` permissions:** POSIX `0700` isolates the private execution closure against other unprivileged UIDs and group members (`uid != os.geteuid()`), **NOT** against same-UID (`bs01415`, `uid=1001`) processes or privileged `root` (`uid=0`), and standard POSIX mode checks do not inspect extended POSIX ACLs. Mode 1 therefore explicitly retains the engineering-host assumption of a trusted single operator (`bs01415`) with zero untrusted same-UID concurrent writers (consistent with `D03`/`D04`/`D12 = DEFER`).
     - **`A3-Mode 2` (Alternative — Recursive In-Place Closure Hardening & Fail-Closed Verification):**
       - **Recursive checks of directories AND regular files:** After the Research Owner hardens host permissions outside Git (zero permission changes occur before authorization), recursively verify **every directory AND every regular file** across the **complete execution closure**—all ancestor directories from `/`, the repository tree (`src/`, `tests/`, `scripts/`, `experiments/colab/configs/`, `pyproject.toml`), `.venv/` and all installed site-packages/native extensions, and the interpreter (`/usr/bin/python3.12`) plus its resolved paths.
       - **Group/world write permissions & symlink targets throughout the execution closure:** Check `(st_mode & 0o022) == 0` on all non-sticky directories and regular files, resolve **every symlink target** throughout the execution closure (including `.venv/bin/python -> /usr/bin/python3.12`, `.venv/bin/*`, and any symlinks inside `.venv/` or the repository) and recursively verify each target and its ancestor chain against group/world write permissions, verify `scripts/run_stage4_capped.py` SHA-256 equals **`12848fa63b00023ff2be1d4018581d1d37cea17756743551ed6a8c3bc8848504`**, confirm sole membership of group `bs01415` (`1001`), and explicitly disclose the same-UID (`bs01415`) / `root` / unaudited-POSIX-ACL limitation.
       - **Fail-closed dependency rule:** If **any** directory, regular file, symlink target, ancestor, or executable dependency (including any file inside `.venv/` or `site-packages/`) remains group-writable, world-writable, or otherwise unprotected, **fail closed and refuse all `A2` and `A5` execution immediately**.
  3. **Universal Fail-Closed Rule:** Under either mode, if the complete execution closure (including `.venv`/dependencies and `A5` code-vs-data isolation), exact pinned wrapper SHA-256 (`12848fa63b00023ff2be1d4018581d1d37cea17756743551ed6a8c3bc8848504`), single-operator no-concurrent-writer condition, or `run_stage4_capped.py`'s `<= 25%` affinity + cgroup `cpu.max` gate cannot be verified, **fail closed and halt all `A2`/`A5` execution immediately.**

### 4.3 Critical Clarification — Host Digests vs. `D04`, `D07`, and `A4`
The SHA-256 digests recorded above for `/usr/bin/python3.12` and `/usr/bin/openssl` **identify the engineering-host inputs on `bs-support` only**. Recording them in `A3`:
- **DOES NOT** constitute or imply **`D04`** Python/native/OpenSSL runtime trust-profile approval (`D04` remains **`DEFER`**).
- **DOES NOT** constitute or imply **`D07`** real release signing or key-generation authorization (`D07` is policy-only).
- **DOES NOT** constitute or imply **`A4`** disposable fixture-signing authorization (`A4` remains **`DEFER`**; `/usr/bin/openssl` MUST NOT be invoked for key generation or signing during Gate A-1).

---

## 5. `A4` Fixture Signing Restrictions

> **Advisory Draft State:** **`DEFER`**  
> **Owner State (`APPROVE / REJECT / DEFER` — `OWNER DECISION REQUIRED`):** `__________`

- Because **`A4 = DEFER`** (`response-local.md:670, 677`) and `D07` authorizes offline custody/anti-rollback policy only:
  - **All** disposable fixture signing is **prohibited** in Gate A-1.
  - **All** real release signing is **prohibited** in Gate A-1.
  - **All** Ed25519 key generation (`openssl genpkey`, `pkeyutl -sign`, `.sig` generation) is **prohibited** in Gate A-1.
  - `stage4_v9_dev_only_release_attestation.json` / `.sig`, `scripts/verify_stage4_release_attestation.py`, and Tests 31/37 real-signature verification paths are deferred to Gate A-2.

---

## 6. `A5` Integrity-Only Permissions & `D17` Separation (`OWNER DECISION REQUIRED`)

> **Advisory Draft State:** `APPROVE` *(strictly scoped to the Owner-selected `A5` integrity boundary below)*  
> **Owner State (`APPROVE / REJECT / DEFER` — `OWNER DECISION REQUIRED`):** `__________`

### 6.1 Distinct Operations Performed by `scripts/verify_stage4_integrity.py`
Direct inspection of `scripts/verify_stage4_integrity.py:32-63` and `src/bangla_smish/post_stage_b_cohorts.py:97-239` distinguishes **three separate read-only operations** (strictly `--verify-only`; `--write-manifests` is **prohibited** under `A6`):

1. **Operation 1 — Cryptographic SHA-256 & Structural Integrity Verification:**
   - Hashes `282/282` Category A files against `reports/protected_baseline_282.json` and `13/13` protected inputs (including `data/processed/banglaphish2026_dedup.parquet`, `eval_smishdetect_test.parquet`, `eval_banglaphish_ood.parquet`), and runs `verify_manifest()` + `audit_post_stage_b_bundle()` on `v1`/`v2` bundles.
2. **Operation 2 — Frozen-Mask Digest Re-Derivation:**
   - Reads `eval_smishdetect_test.parquet` and `eval_smishdetect_validation.parquet` via `build_smishdetect_test_cohort_masks(v2_dir, verify_tfidf_reproduction=False)` to recompute the `4` frozen cohort mask hashes (`1687/1698/1693/1675`) against `reports/stage4_frozen_test_masks.json`.
3. **Operation 3 — Cross-Boundary TF-IDF Near-Duplicate Screening (Data-Quality Analysis, NOT a Checksum):**
   - When `--skip-tfidf` is **omitted**, `build_smishdetect_test_cohort_masks(..., verify_tfidf_reproduction=True)` invokes `_reproduce_cross_boundary_test_hits(v2_dir)` (`post_stage_b_cohorts.py:97-145`), which fits character-ngram `(3, 5)` `TfidfVectorizer` and computes pairwise cosine similarity `>= 0.85` across `12,466` rows from `6` parquets (`real_train_pool`, `synthetic_pilot_300`, `synthetic_batches_b1_b7`, `eval_smishdetect_validation`, `eval_smishdetect_test`, `eval_banglaphish_ood`). **This is a read-only corpus data-quality analysis, not merely a checksum operation.**

### 6.2 `A5` Scope Selection (`OWNER DECISION REQUIRED`)

- **Core `A5` Scope — Recommended Default (`--skip-tfidf` Integrity-Only):**
  1. `.venv/bin/python scripts/verify_stage4_integrity.py --verify-only --skip-tfidf` *(Operations 1 & 2 ONLY: SHA-256 hashing, structural bundle parity, and frozen-mask digest re-derivation; excludes Operation 3 TF-IDF screening).*
  2. `.venv/bin/python scripts/prepare_post_stage_b_bundle.py --verify-only` on `post_stage_b_bundle` (`v1`) and `post_stage_b_bundle_v2` (`v2`) *(Operation 1 bundle integrity audit cross-referenced with `A2`).*
  3. Read-only `verify_portable_colab_kit()` checks on untouched `stage4_colab_kit_upload/bangla_smish_post_stage_b_stage4_v{3..8}/` *(SHA-256 checks + Operation 2 mask hash re-derivation with `verify_tfidf_reproduction=False`).*
  4. Existing local mocked-probability regression tests under `A2` (`test_post_stage_b_pipeline.py`, `test_stage4_option_a_and_cohorts.py`, `test_stage4_ensemble_and_stats.py`, `test_stage4_remediation.py`, `test_stage4_final_remediation.py`, `test_stage4_v8_remediation.py`), which read frozen parquet schemas/rows strictly with **synthetic/Dirichlet/mocked probability arrays** to verify software math/wiring.
- **Optional Separate Sub-Authorization — `A5-TFIDF` (`OWNER DECISION REQUIRED`; Advisory Default: `DEFER` / `EXCLUDE`):**
  - Separate, explicitly authorized read-only corpus data-quality reproduction running `scripts/verify_stage4_integrity.py --verify-only` **without** `--skip-tfidf` (adding Operation 3 `12,466`-row TF-IDF near-duplicate screening across the 6 parquets). Must NEVER be conflated with cryptographic checksums or held-out model evaluation.
  - **`A5-TFIDF` Owner State (`APPROVE / REJECT / DEFER` — `OWNER DECISION REQUIRED`):** `__________`

### 6.3 Explicit Boundary With `D17 = DEFER` (Held-Out Evaluation)
Data-integrity checks (`A5`), optional TF-IDF data-quality reproduction (`A5-TFIDF`), and mocked-probability unit tests (`A2`) are **NOT actual model evaluation**. `D17` remains **`DEFER` (`NO-GO`)**. Strictly **prohibited** in Gate A-1:
- Trained-model inference on `eval_smishdetect_test.parquet` or `eval_banglaphish_ood.parquet`.
- Real test/OOD predictions (`preds.npz` or in-memory model logits/probabilities).
- Predictive performance reporting (F1, recall, FPR, ECE, ROC/PR-AUC) from real models on held-out/OOD splits.
- Threshold selection or tuning using held-out/OOD splits.
- Held-out/OOD exports or reuse of any `A5` read inside dev-only codepaths.
- Mixing mocked test outputs with scientific/research results.

---

## 7. `A6` Binding Exclusions & Temporary Gate A-1 Development-Kit Restrictions

> **Binding Status:** Applies unconditionally to all Gate A-1 work (`A1–A5`).

### 7.1 Untouched Protected Artifacts (`A6`)
Preserve byte-identical and unmodified:
- `DataSet.csv`, `documents.md`, `data/**` (including `data/processed/post_stage_b_bundle/`, `data/processed/post_stage_b_bundle_v2/`, and `dataset_manifest.json` — **never** write `dev_only_manifest.json` into `data/**`)
- `reports/protected_baseline_282.json`, `reports/stage4_frozen_test_masks.json`, `reports/stage4_statistical_validation_results.json`
- Frozen `v1` configs: `experiments/colab/configs/gpu_smoke_test_config.json`, `post_stage_b_stage4_t4_benchmark_config.json`, `post_stage_b_stage4_option_a_config.json`, `post_stage_b_abcd_config.json`
- Historical portable kits: `stage4_colab_kit_upload/bangla_smish_post_stage_b_stage4_v{3..8}/`
- Saved specification: `response-local.md` (`sha256: a62ce62add5a39acd7218f5d1a12f92a2154853dd87c0feeb7d3c7d81146e1ff`)

### 7.2 Excluded Activities (`A6`)
Excluded from Gate A-1: Gate A-2 implementation (Section B.3 raw syscalls, Trusted Control Notebook / Option D fixed `-c` launcher, `scripts/verify_stage4_release_attestation.py`, notebook regeneration, Ed25519 fixture signing), dependency installs (`pip`/`uv`), real Ed25519 release signing, live Colab or target-runtime probes (any future CPU-only Colab diagnostic probe requires separate Owner authorization, uses disposable temporary files with no project data or Drive mount, and bills per the Owner's verified Colab tier), deployment acceptance (`D12`), GPU smoke test (`D13`), T4 benchmark (`D14`), Track A/B (`D15`/`D16`), synthetic generation (permanently stopped), held-out/OOD evaluation (`D17`), and enabling `CONFIRMATORY_DECLARATIONS_ENABLED` (stays `False`, `D18`).

### 7.3 Temporary Gate A-1 Development-Kit Restrictions
Any kit produced by `scripts/prepare_stage4_colab_kit.py --profile dev_only` during Gate A-1 is strictly classified as:
- **Unsigned**
- **Pre-B.3**
- **Non-release-eligible**
- **Disposable local development artifact only** (written exclusively to a private `0700` directory under `/home/bs01415/.stage4_v9_work/` or `/tmp/`, never uploaded as a signed release and never modifying `stage4_colab_kit_upload/v3–v8`).

**Mandatory Machine-Readable Flags in Gate A-1 Dev-Only Manifest & Verifier Output:**
```python
b3_syscall_hardening_implemented = False
release_eligible = False
```

**Mandatory Contents & Exclusions of the Temporary Gate A-1 Dev-Only Export:**
- **MUST contain ONLY:** the exact `5` `DEV_ONLY_BUNDLE_FILENAMES` parquets (`real_train_pool.parquet`, `synthetic_pilot_300.parquet`, `synthetic_batches_b1_b7.parquet`, `eval_smishdetect_validation.parquet`, `counterfactual_smish_probe.parquet`), `dev_only_manifest.json`, the `18` `src/bangla_smish/*.py` runtime modules, `experiments/colab/configs/gpu_smoke_test_config_v2.json`, `experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config_v2.json`, `pyproject.toml`, `reports/stage4_dev_statistical_governance.json`, and unsigned `stage4_kit_manifest.json` (`kit_profile="dev_only"`, `b3_syscall_hardening_implemented=False`, `release_eligible=False`).
- **MUST exclude:** the unregenerated v8 notebook (`experiments/colab/post_stage_b_stage4_experiments.ipynb`), release attestations and `.sig` files, `eval_smishdetect_test.parquet`, `eval_banglaphish_ood.parquet`, `dataset_manifest.json`, `reports/protected_baseline_282.json`, `reports/stage4_frozen_test_masks.json`, `reports/stage4_confirmatory_metadata_registry.json`, `reports/stage4_statistical_validation_results.json`, frozen `v1` configs, and all `tests/test_*.py` files.
- **MUST preserve** historical kits `v3–v8` unchanged and **NEVER claim Section B.3 atomic security compliance**.

### 7.4 Why Option S2 `os.scandir()` Is NOT Equivalent to Section B.3
Option S2 `os.scandir()` physical-absence verification provides **inventory hygiene and accidental-leakage detection only**. It does **not** supply Section B.3's descriptor-based security architecture (`response-local.md:54-176`) for five distinct technical reasons:
1. **No atomic kernel mount-boundary prevention (`openat2` with `RESOLVE_NO_XDEV | RESOLVE_NO_SYMLINKS`):** Userspace `os.scandir()` cannot atomically prevent kernel path resolution from crossing a mount point or a **same-device bind mount** (which inherits an identical `st_dev`).
2. **No descriptor-based mount-ID verification (`statx(..., AT_EMPTY_PATH, STATX_MNT_ID_UNIQUE | STATX_MNT_ID)`):** `os.scandir()` cannot pin a session-wide mount-ID representation, detect same-device bind mounts over `kit_dir` (eliminating staging-root self-trust), or verify every child directory and file descriptor against the anchor mount ID.
3. **No component-by-component trusted ancestor walk:** `os.scandir()` inspects descendants inside `kit_dir` only; it does not walk component-by-component from `/` or a pre-opened trusted anchor FD verifying every ancestor's `fstat()` UID, non-sticky group/world-write bits, and sticky-parent ownership rules against directory-rename substitution.
4. **No kernel file-type enforcement (`OPENAT2_REGULAR`), ABI validation, `fstatfs()` backing-mount policy, or same-FD payload binding:** Between `os.scandir()` inspection and pathname `open()`, a concurrent same-mount writer can substitute a symlink, FIFO (`S_ISFIFO`), Unix socket (`S_ISSOCK`), or device node (`S_ISCHR`/`S_ISBLK`), causing `open(2)` side effects before post-open `fstat()` runs. Option S2 `os.scandir()` also omits ABI/syscall-table resolution, `fstatfs()` backing-filesystem policy checks, and strict pre-payload attestation ordering.
5. **No cryptographic provenance or immutability:** Without Ed25519 detached signatures (`A4`/`D06` deferred) and an independently protected runtime/storage boundary, directory enumeration cannot authenticate file origin or freeze bytes against same-UID or privileged mutation.

---

## 8. Option A vs. Option B — Legacy Test Compatibility Analysis & Recommendation (`OWNER DECISION REQUIRED`)

> **Epistemic Boundary:** Everything in Section 8 is based on **static source and manifest inspection at code baseline `720e43b`**. **Zero tests have been implemented or executed.** Runtime pass/fail behavior remains unverified until `A2` runs after Owner sign-off.

### 8.1 Side-by-Side Static Inspection of All 9 Test Files

| Test File | Static Source Findings (Baseline `720e43b`) | Option A *(Broad Legacy Rewrite)* | Option B *(Recommended: Scoped Compatibility)* |
|---|---|---|---|
| `tests/test_stage4_v9_remediation.py` | Does not exist yet. Proposed new Gate A-1 Option S2, `14`-pair/`24`-row, strict `dev_only_manifest.json` non-override, real disposable dev-only export, `v2` config, `24`-dist sync, and `os.scandir()` suite (plus `audit_model_view_overlaps(include_heldout=False)` I/O sentinel if approved). | **CREATE NEW** | **CREATE NEW** *(also tests strict `bundle_profile="dev_only"` non-override, missing-registry fail-closed behavior, and clean `v9_dev_only` imports of `post_stage_b_runner` / `post_stage_b_integrity`)* |
| `tests/test_post_stage_b_pipeline.py` | Lines `206, 229` call `load_config("gpu_smoke_test_config.json")` (`run_heldout_eval: true`), which is designed to fail closed under the proposed `PostStageBConfig.__post_init__()` guard. | **MODIFY** *(switch to `v2` + assert `v1` fails closed)* | **MODIFY** *(lines `206, 229`: switch to `v2` + assert `v1` fails closed)* |
| `tests/test_stage4_v8_remediation.py` | Line `1027` loads `gpu_smoke_test_config.json` (`run_heldout_eval: true`). Lines `273–308, 740, 924, 1028–1031, 1115` invoke `gpu_smoke_test`/`t4_benchmark` against `data/processed/post_stage_b_bundle_v2`. | **MODIFY** *(line `1027` + rewrite 5+ bundle/gate/T4 tests)* | **MODIFY** *(line `1027` ONLY — switch `v1` smoke fixture to `v2`)* |
| `tests/test_stage4_final_remediation.py` | Lines `495, 851` assert `verified_runtime_file_count == 34` on newly exported full kits. Lines `71, 161, 284–307, 604–776, 1113–1135, 1388, 1434–1438, 1878` import `STAGE3_32_CANDIDATE_PAIRS` and exercise the **unregenerated frozen v8 notebook** (Cells 1–7) on `v1` `t4_benchmark` (`benchmark_workload_fixture_eval=False`, `benchmark_dev_timing_eval=False`, `all_32` gate). Line `1085` patches `_build_t4_benchmark_workload_fixture`. Line `2085` asserts `"transformers>=4.57.1,<5"` in `pyproject.toml`. | **MODIFY EXTENSIVELY** *(risks gutting frozen v8 notebook lifecycle coverage before Gate A-2 replaces the notebook)* | **MODIFY (2 LINES ONLY)** *(lines `495, 851`: `34 -> 35` for newly exported full kits containing `stage4_confirmatory_metadata_registry.json`; frozen v8 notebook lifecycle tests, `STAGE3_32_CANDIDATE_PAIRS` via module-level `__getattr__` [PEP 562], line `1085` patch target, and `transformers>=4.57.1,<5` stay intact)* |
| `tests/test_post_stage_b_preprocessing.py` | Line `102` reads raw JSON `gpu_smoke_test_config.json["models"]` (never calls `load_config()`) for the legacy Phase-2 notebook Cell 4 (`distilbert` only; switching to `v2` adds BanglaBERT and breaks its `installed_distribution` mock unless rewritten). | **MODIFY** *(switch to `v2` + rewrite mock)* | **NO CHANGE NEEDED** |
| `tests/test_post_stage_b_runner.py` | Line `36` runs `gpu_smoke_test` directly against `data/processed/post_stage_b_bundle` (`v1`, which has `dataset_manifest.json` and cannot hold `dev_only_manifest.json` under `A6`). | **MODIFY** *(stage temp dev-only bundle)* | **NO CHANGE NEEDED** |
| `tests/test_stage4_checkpointing_and_kit.py` | Line `265` runs `gpu_smoke_test` against `data/processed/post_stage_b_bundle_v2`. Line `224` tests `export_stage4_colab_kit()` / `verify_portable_colab_kit()`. | **MODIFY** *(stage temp dev-only bundle)* | **NO CHANGE NEEDED** |
| `tests/test_stage4_remediation.py` | Lines `50, 631–635` import `_build_t4_benchmark_workload_fixture` asserting `(1698, 5416)`/`7114`. Lines `110–128` run `verify_portable_colab_kit(staged)` in an **isolated child process outside the repo** (`cwd=self.root, PYTHONPATH=staged/src`). Line `596` runs `gpu_smoke_test` on `post_stage_b_bundle_v2`. | **MODIFY** *(rewrite fixture test, temp dev-only bundle, and isolated subprocess check)* | **NO CHANGE NEEDED** *(registry-backed lookup removes `1698`/`5416` literals from `src/`; exporting `stage4_confirmatory_metadata_registry.json` into newly exported full kits keeps the isolated child process self-contained)* |
| `tests/test_stage4_option_a_and_cohorts.py` | Lines `16–20, 133–134` import `AMBIGUOUS_TEST_FAMILY_IDS` and `AMBIGUOUS_TEST_ROW_IDS` from `post_stage_b_cohorts.py`. Line `101` calls `audit_model_view_overlaps(self.v2_bundle, views=OPTION_A_ALLOWED_VIEWS)` expecting `280` checks. | **MODIFY** *(read from registry JSON directly)* | **NO CHANGE NEEDED** *(module-level `__getattr__` [PEP 562] in `post_stage_b_cohorts.py` serves them from `reports/stage4_confirmatory_metadata_registry.json` in the full repo and fails closed when absent in `v9_dev_only`; `include_heldout=True` default preserves line `101`)* |

### 8.2 Static Source Compatibility Analysis of All 7 Dimensions Under Recommended Option B *(Pre-Implementation — Zero Tests Executed)*
1. **Historical `v3–v8` kit compatibility:** Direct manifest inspection confirms all six historical kits (`stage4_colab_kit_upload/bangla_smish_post_stage_b_stage4_v{3..8}/stage4_kit_manifest.json`) use `kit_version = "post_stage_b_stage4_kit_v2"` with **exact `34` runtime files** and no confirmatory registry file. Under Option B, `verify_portable_colab_kit()` is designed to accept both untouched `34`-file `v3–v8` kits (resolving `reports/stage4_confirmatory_metadata_registry.json` from the local repo root during `A5` checks) and newly exported `35`-file full kits (self-contained for `test_stage4_remediation.py:110-128`'s isolated child process).
2. **Existing `v1`/`v2` bundle preservation:** `data/processed/post_stage_b_bundle/` (`v1`) and `data/processed/post_stage_b_bundle_v2/` (`v2`) remain untouched under `A6`; `dev_only_manifest.json` is never written into `data/**`. Both bundles are designed to pass `prepare_post_stage_b_bundle.py --verify-only` and `verify_stage4_integrity.py --verify-only --skip-tfidf` unchanged (to be confirmed by `A2`/`A5` execution after sign-off).
3. **Module-level `__getattr__` (PEP 562) vs. ordinary function-local lazy imports (two distinct mechanisms):** Option B uses two separate Python mechanisms that must not be conflated:
   - **Module-level `__getattr__` compatibility (PEP 562):** Implemented in `post_stage_b_data.py`, `post_stage_b_cohorts.py`, and `post_stage_b_tokenizer_gate.py` to serve externalized confirmatory-only module attributes (`AMBIGUOUS_TEST_*`, `TEMPLATE_STRESS_TEST_FAMILY_IDS`, `COHORT_SPECS`, `STAGE3_32_CANDIDATE_PAIRS`, full `PROTECTED_INPUT_DIGESTS`) from `reports/stage4_confirmatory_metadata_registry.json` on demand in the full repository while failing closed when the registry is absent in `v9_dev_only`. This keeps shared `src/` free of static held-out/source-OOD literals and preserves repo-local test imports without modifying `test_stage4_option_a_and_cohorts.py` or `test_stage4_remediation.py`.
   - **Ordinary function-local lazy imports (not PEP 562):** Static grep confirmed two eager top-level `src/` imports (`post_stage_b_runner.py:83` importing `STAGE3_32_CANDIDATE_PAIRS` and `post_stage_b_integrity.py:17` importing `PROTECTED_INPUT_DIGESTS`). Removing both eager top-level imports and replacing them with ordinary function-local imports (inside the `pair_scope="all_32"` branch of `post_stage_b_runner.py` and inside `build_protected_baseline_manifest()` at `post_stage_b_integrity.py:176`) prevents module import time from invoking the PEP 562 `__getattr__` handlers, allowing `v9_dev_only` kits to import `post_stage_b_runner` and `post_stage_b_integrity` cleanly while still failing closed on any confirmatory-symbol access.
4. **Strict `dev_only_manifest.json` enforcement:** By design, `verify_dev_only_manifest()` is unconditionally fail-closed non-override (`dataset_manifest.json` is never accepted as a fallback). In `run_post_stage_b_pipeline()`, strict `dev_only_manifest.json` non-override is enforced whenever `bundle_profile="dev_only"`, whenever `dev_only_manifest.json` is present, or when running `v2` dev-only configs in dev-only mode; legacy `gpu_smoke_test` unit tests calling against `data/processed/post_stage_b_bundle[_v2]` with `run_heldout_eval=False` continue to run without mutating `data/**`.
5. **`dev_only_14` tokenizer validation:** By design, `load_dev_only_14_pairs()` reads dev parquets only (`14` pairs, `24` unique rows, `0` test/OOD opens). `_validate_tokenizer_gate_report()` enforces `pair_scope == "dev_only_14"` in `dev_only` mode and `pair_scope == "all_32"` in `confirmatory_mode=True`, while keeping frozen-v8-notebook calls in `test_stage4_final_remediation.py` operational until Gate A-2 replaces the notebook.
6. **T4 benchmark lifecycle compatibility (`benchmark_dev_timing_eval`):** `SYNTHETIC_WORKLOAD_FIXTURE_ENABLED = False` remains unconditionally fail-closed for `benchmark_workload_fixture_eval=True`. `t4_benchmark_v2` requires `benchmark_dev_timing_eval=True` for real HF execution and `dev_only` mode, while mocked lifecycle tests on `replace(v1_t4, benchmark_workload_fixture_eval=False)` in `test_stage4_final_remediation.py` are expected by static analysis to remain passing without mutating the frozen v8 notebook (to be confirmed by `A2` execution after sign-off).
7. **Preservation of `transformers>=4.57.1,<5`:** Retained verbatim in both `pyproject.toml [gpu]` and `src/bangla_smish/post_stage_b_colab_deps.py` (matching `response-local.md:205` and `test_stage4_final_remediation.py:2085`), rather than narrowing to `transformers==4.57.1`.

### 8.3 Recommendation (`OWNER DECISION REQUIRED`)
- **Recommended:** **Option B** (`4` authorized test files: `test_stage4_v9_remediation.py` [new], `test_post_stage_b_pipeline.py`, `test_stage4_v8_remediation.py`, and 2 lines in `test_stage4_final_remediation.py`).
- **Why Option B is supported by static inspection:** It preserves fail-closed Option S2 isolation in all proposed dev-only codepaths and exported disposable dev-only kits (combining module-level `__getattr__` [PEP 562] compatibility for externalized confirmatory symbols with ordinary function-local lazy imports replacing the two eager top-level imports at `post_stage_b_runner.py:83` and `post_stage_b_integrity.py:17`) without mutating `data/**`, without weakening historical `v3–v8` or frozen-v8-notebook regression coverage prior to Gate A-2, and with the smallest possible diff outside new v9 code.
- **No silent edits:** Under either option, **zero** test files outside the Owner-approved `A1` list may be modified.

---

## 9. Exact Gate A-1 Permitted Operations *(Only After Owner Completes & Signs)*

1. Implement only the Owner-approved `A1` file list (core 8 files + `post_stage_b_model_view_audit.py` iff Owner selects `INCLUDE` + 4 new config/report files + Owner-selected Option B or Option A test files).
2. Establish the Owner-selected `A3` execution trust boundary (verifying pinned `scripts/run_stage4_capped.py` SHA-256 `12848fa63b00023ff2be1d4018581d1d37cea17756743551ed6a8c3bc8848504` and `<= 25%` CPU cap before any test execution; fail closed if unmet).
3. Execute only the `A2` local CPU checks (`unittest discover`, direct `tests/test_pipeline.py`, `v1`/`v2` bundle `--verify-only`, and disposable `scripts/prepare_stage4_colab_kit.py --profile dev_only` export into a private `0700` dir on `A3`) with `b3_syscall_hardening_implemented = False` and `release_eligible = False`.
4. Execute only the Owner-approved `A5` read-only integrity checks (`scripts/verify_stage4_integrity.py --verify-only --skip-tfidf` and untouched `v3–v8` kit checks, plus `A5-TFIDF` iff explicitly approved separately) on `A3` under the pinned `run_stage4_capped.py` wrapper.
5. Deliver a Gate A-1 completion report strictly separating `112` *planned* vs. *mocked locally executed* tokenizer comparisons, and separating mocked passes, real fail-closed refusals, and real non-mocked verifications.

---

## 10. Prohibited & Blocked Operations

### Prohibited Operations (Forbidden Under `A1`/`A4`/`A5`/`A6` & `D07`/`D15`/`D17`/`D18`)
- Modifying any `A6` protected artifact (`DataSet.csv`, `documents.md`, `data/**`, `dataset_manifest.json`, `reports/protected_baseline_282.json`, `reports/stage4_frozen_test_masks.json`, `reports/stage4_statistical_validation_results.json`, frozen `v1` configs, `v3–v8` kits, `response-local.md`).
- Modifying any file omitted from the Owner-signed `A1` list (including `scripts/run_stage4_capped.py`, `post_stage_b_stats.py`, `validate_stage4_fixed_bootstrap.py`, `generate_post_stage_b_stage4_notebook.py`, `post_stage_b_stage4_experiments.ipynb`, `verify_stage4_release_attestation.py`, or unlisted test files).
- Running `verify_stage4_integrity.py --write-manifests`, or running TF-IDF near-duplicate screening without explicit `A5-TFIDF` approval.
- Installing any package (`pip`/`uv`), generating keys, signing fixtures or releases, invoking `/usr/bin/openssl` for signing/keygen, running raw B.3 syscall probes (`openat2`/`statx`/`OPENAT2_REGULAR`), running live Colab/target probes, running trained-model inference or predictive metrics on held-out/OOD files, tuning thresholds on held-out data, generating synthetic data, or flipping `CONFIRMATORY_DECLARATIONS_ENABLED` from `False`.
- Labeling any Gate A-1 disposable dev-only export as signed, B.3-compliant, or release-eligible.

### Blocked Operations (Cannot Legally or Technically Proceed Because Required Decisions Are `DEFER`)
- **Real `112`-check HF tokenizer execution:** Blocked by `D11 = DEFER` and absence of `torch`/`transformers`/`tokenizers`/`sentencepiece`/`normalizer` in `.venv`.
- **Signed `v9_dev_only` release kit, `stage4_v9_dev_only_release_attestation.json`/`.sig`, `scripts/verify_stage4_release_attestation.py`:** Blocked by `A4 = DEFER`, `D06 = DEFER`, and `D07` policy-only restriction.
- **Section B.3 raw-syscall implementation & Tests 8–28 (`openat2`/`statx`/`OPENAT2_REGULAR`/`fstatfs`/ABI dispatch):** Blocked by `D08` implementation deferral to Gate A-2, `D09 = DEFER`, and `bs-support`'s `Linux 7.0.0-43-generic` (`< 7.2` upstream `OPENAT2_REGULAR`).
- **Trusted Control Notebook, Option D fixed `-c` launcher, `-I -S -B -X pycache_prefix`, ELF/OpenSSL execution binding, notebook regeneration, Tests 30–41:** Blocked by `D03`, `D04`, `D06`, `A4 = DEFER`.
- **Live Colab feasibility / target-runtime validation:** Blocked by `A2`/`A3`, `response-local.md:681`, and `D12 = DEFER`.
- **GPU smoke (`D13`), T4 benchmark (`D14`), Track A (`D15`), Track B (`D16`), held-out eval (`D17`), confirmatory research (`D18`):** All independently blocked (`DEFER`).

---

## 11. Remaining Research Owner Decisions (`OWNER DECISION REQUIRED`)

### 11.1 Immediate Pre-Signing Gate A-1 Choices (`OWNER DECISION REQUIRED`)
1. **Individual `D01–D18` & `A1–A5` Canonical States (`OWNER DECISION REQUIRED`):** Enter exactly one state (`APPROVE`, `REJECT`, or `DEFER`) per row in Section 1 and Sections 2–6, and mirror them verbatim in Section 12.
2. **Legacy Test Compatibility Strategy (`OWNER DECISION REQUIRED`):** Select **`Option B` (Recommended — 4 test files)** or **`Option A` (9 test files)** to lock the exact `A1` test-file list.
3. **Optional Model-View I/O Guard (`OWNER DECISION REQUIRED`):** Decide whether to **`INCLUDE` (Recommended)** or **`EXCLUDE`** `src/bangla_smish/post_stage_b_model_view_audit.py` (`include_heldout=False` I/O-gating guard + sentinel tests) in `A1`.
4. **`A5` Scope Boundary (`OWNER DECISION REQUIRED`):** Confirm core `A5` uses **`--verify-only --skip-tfidf` (Recommended)**, and record `A5-TFIDF` (`APPROVE / REJECT / DEFER`; Recommended: `DEFER`) separately for the `12,466`-row TF-IDF data-quality screen.
5. **`A3` Execution Trust-Boundary Mode (`OWNER DECISION REQUIRED`):** Select **`A3-Mode 1` (Recommended — private `0700` complete execution closure covering source, tests, scripts, configs, interpreter, and dependencies with independently protected or privately copied `.venv`, no silent `A5` execution from the `0775` repo tree, disclosed same-UID `0700` limitation, and pinned `run_stage4_capped.py` SHA-256)** or **`A3-Mode 2` (Owner in-place hardening + recursive directory and regular-file group/world-write and symlink-target checks across the complete execution closure, fail-closed on any unprotected executable dependency, disclosed same-UID limitation, and pinned wrapper SHA-256)**.
6. **Final Written Authorization (`OWNER DECISION REQUIRED`):** Complete and sign Section 12. Until every field is completed consistently and signed, **Gate A-1 remains NOT AUTHORIZED**.

### 11.2 Deferred Post-Gate A-1 Decisions (`OWNER DECISION REQUIRED` Later — Closed Now)
- **Hosted Colab vs. Self-Managed GPU Path (`D03`, `D04`, `D06`, `D09`, `D12`):** Whether to authorize a separate CPU-only hosted-Colab diagnostic probe (disposable temporary files, no project data or Drive mount, billing per verified Colab tier), adopt a Path 3 managed-runtime trust disclosure ("VM private to the account"), or provision a self-managed T4 Ubuntu VM before scoping Gate A-2 (`A4`, B.3 syscalls, launcher, control notebook).
- **Dependency Provisioning & GPU Gates (`D11`, `D13`, `D14`):** Authorize package installs/lock, Gate B Matrix M1 smoke run, and Gate C `708`-row timing-only T4 benchmark separately.
- **Statistical Governance — Track A (`D15`) & Track B (`D16`):** Retain v1 Wilson cutoffs (`k <= 280` at `N=5,000`) vs. approve proposed UCB additions (`k <= 272`); reclassify exploratory `D vs C` to non-gating telemetry; select CR2 vs. `G/(G-1)` upfront and commit fresh preregistered seeds (replacing unverified placeholders `2026112601`/`2026112602`); select Track B FPR estimand and independent Cohort A/B sampling protocol.
- **Held-Out Evaluation & Confirmatory Claims (`D17`, `D18`):** Independently closed (`NO-GO`); `CONFIRMATORY_DECLARATIONS_ENABLED` stays `False`.

---

## 12. Copy-Ready Unsigned Research Owner Declaration

> **STATUS: DRAFT — NOT SIGNED — NO AUTHORIZATION GRANTED.**  
> *(Per Section 1.1, this template records **zero** Owner decisions until the Research Owner fills all `OWNER DECISION REQUIRED` placeholders consistently with Sections 1–6 and signs. Any blank field, unselected option, or mismatch grants **zero authority**. Do not modify `response-local.md`.)*

> I, **[Research Owner — OWNER DECISION REQUIRED]**, record this accompanying dated Decision Record (`stage4_v9_research_owner_decision_record_2026-11-27.md`, dated **[YYYY-MM-DD — OWNER DECISION REQUIRED]**, final signed revision/SHA-256 **[RECORDED AT SIGNING — OWNER DECISION REQUIRED]**) against unmodified **`response-local.md` (`sha256: a62ce62add5a39acd7218f5d1a12f92a2154853dd87c0feeb7d3c7d81146e1ff`)**, reviewed code baseline `Durjoy1971/Explore-AIs` `main` commit `720e43bac9a58e5acf9d1b197f9d96085c1b24ea` (README-only commit over `5e1da6b6`), initial decision-record introduction commit `a53ab9f4e75c47f8fc76b014f7c8979bb6e603e5` (blob `7e65c846272c318ae1dec0dbf8364dad91115844`, `sha256: d4b461d972e836e16c5434523afab7d2b32e6be712f678762ca44f0c0a8ffcb4`), and subsequent committed decision-record update at `main` HEAD `5ee875cee301a67828cd0a451790350065156ea6` (blob `6b24dac2d6212bc913158f2f6a88e6e1bf72e950`, `sha256: 27704ef441e328ef0833ef0624cb882e902b00ab026c78a68c526c0e2f59e06e`).
>
> **APPROVE** — `[OWNER DECISION REQUIRED — enter exact IDs matching Section 1 matrix; advisory draft: D01 (specification sign-off on response-local.md a62ce62a...; zero engineering authority), D02 (self-managed Ubuntu workstation bs-support as engineering platform), D05 (independently protected absolute-path binding as architectural default; actual runtime deferred to D04), D07 (offline Owner Ed25519 custody and anti-rollback policy only; zero key generation or signing authorized), D08 (Section B.3 strict atomic design default only; implementation deferred to Gate A-2), D10 = APPROVE. Scope is Gate A-1 only, subject to A1–A6.]`
>
> **REJECT** — `[OWNER DECISION REQUIRED — enter exact IDs matching Section 1 matrix, or None]`
>
> **DEFER** — `[OWNER DECISION REQUIRED — enter exact IDs matching Section 1 matrix; advisory draft: D03, D04, D06, D09, D11, D12, D13, D14, D15, D16, D17, D18]`
>
> **Engineering environment** — `bs-support` (`Ubuntu 26.04.1 LTS`, `Linux 7.0.0-43-generic x86_64`, admin `bs01415`, capped `<= 25%` / `3` CPUs via pinned `scripts/run_stage4_capped.py` `sha256: 12848fa63b00023ff2be1d4018581d1d37cea17756743551ed6a8c3bc8848504`, no GPU, local `ext4`, private `0700` disposable staging under `/home/bs01415/.stage4_v9_work/` or owner-`0700` `/tmp/` subdir; `A3` complete execution-closure trust-boundary mode per Section 4.2: **`[A3-Mode 1 (Recommended: private 0700 complete execution closure + protected/private .venv + no silent A5 repo execution + disclosed same-UID 0700 limitation + pinned wrapper SHA-256) / A3-Mode 2 (Owner in-place hardening + recursive dir/regular-file group/world-write and symlink-target checks across full closure + fail-closed on any unprotected dependency + disclosed same-UID limitation + pinned wrapper SHA-256) — OWNER DECISION REQUIRED]`**; observed host inventory `.venv/bin/python -> /usr/bin/python3.12` `sha256: e50d468e...` and `/usr/bin/openssl` `sha256: 1d72cbc5...` identifying engineering-host inputs only, conveying zero `D04`, `D07`, or `A4` authority); permitted for Gate A-1 `A1`, `A2`, and `A5` checks only.
>
> **Production/research execution environment** — No approval (`DEFER` under `D12`; hosted-Colab provenance unresolved).
>
> **GPU execution environment** — No approval (`DEFER` under `D13` and `D14`).
>
> **Gate A** — `[OWNER DECISION REQUIRED — must match Section 1 D10 row and general decision lists verbatim; if approving Gate A-1, enter exactly: D10 = APPROVE. Scope is Gate A-1 only, subject to A1–A6.]` D10 approval alone grants no authority. Recorded Gate A-1 permissions matching Sections 2–8 verbatim:
> - **A1 (`[APPROVE / REJECT / DEFER — OWNER DECISION REQUIRED]`)**: Exact authorized file list only — core 8 source/config files (`scripts/prepare_stage4_colab_kit.py`, `src/bangla_smish/post_stage_b_integrity.py`, `src/bangla_smish/post_stage_b_data.py`, `src/bangla_smish/post_stage_b_cohorts.py`, `src/bangla_smish/post_stage_b_tokenizer_gate.py`, `src/bangla_smish/post_stage_b_runner.py`, `src/bangla_smish/post_stage_b_colab_deps.py`, `pyproject.toml`), **`[INCLUDE / EXCLUDE src/bangla_smish/post_stage_b_model_view_audit.py (include_heldout=False I/O guard + sentinel tests) — OWNER DECISION REQUIRED]`**, exact 4 new config/report files (`experiments/colab/configs/gpu_smoke_test_config_v2.json`, `experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config_v2.json`, `reports/stage4_confirmatory_metadata_registry.json`, `reports/stage4_dev_statistical_governance.json`), and exact test-file set under **`[Option B (Recommended: 4 test files) / Option A (9 test files) — OWNER DECISION REQUIRED]`**. Retain `transformers>=4.57.1,<5`. No implicit file edits; zero edits to `scripts/run_stage4_capped.py`, `post_stage_b_stats.py`, `validate_stage4_fixed_bootstrap.py`, `generate_post_stage_b_stage4_notebook.py`, `post_stage_b_stage4_experiments.ipynb`, or `verify_stage4_release_attestation.py`.
> - **A2 (`[APPROVE / REJECT / DEFER — OWNER DECISION REQUIRED]`)**: Local CPU-only checks on `A3` under the selected `A3` trust-boundary mode and pinned `scripts/run_stage4_capped.py` (`<= 25%` CPU): `unittest discover -s tests -v`, direct `.venv/bin/python tests/test_pipeline.py`, `scripts/prepare_post_stage_b_bundle.py --verify-only` (`v1` & `v2`), and disposable `scripts/prepare_stage4_colab_kit.py --profile dev_only` export into a private `0700` dir verifying exact `5` dev parquets, `b3_syscall_hardening_implemented = False`, `release_eligible = False`, exclusion of the unregenerated v8 notebook and attestations/signatures, and `0` held-out metadata across all non-parquet files. Report `112` planned real HF comparisons separately from mocked locally executed checks; zero live Colab/target probes.
> - **A3 (`[APPROVE / REJECT / DEFER — OWNER DECISION REQUIRED]`)**: Local workstation `bs-support` (`Ubuntu 26.04.1 LTS`, `<= 3` CPUs, no GPU, private `0700` staging, pinned `run_stage4_capped.py` SHA-256 `12848fa6...`, selected `A3-Mode 1` or `A3-Mode 2` complete execution-closure fail-closed trust boundary per Section 4.2) for `A1`, `A2`, and `A5` only.
> - **A4 (`[DEFER — OWNER DECISION REQUIRED]`)**: All fixture signing, real release signing, and Ed25519 key generation prohibited in Gate A-1.
> - **A5 (`[APPROVE / REJECT / DEFER — OWNER DECISION REQUIRED]`)**: Read-only integrity verification only — **`scripts/verify_stage4_integrity.py --verify-only --skip-tfidf`** (SHA-256 + structural integrity + frozen-mask digest re-derivation), `scripts/prepare_post_stage_b_bundle.py --verify-only`, untouched `v3–v8` kit checks, and mocked-probability unit-test reads under `A2`; separate optional TF-IDF near-duplicate data-quality screen **`A5-TFIDF = [APPROVE / REJECT / DEFER (Recommended: DEFER) — OWNER DECISION REQUIRED]`**. Strictly non-evaluative: `D17` remains `DEFER`; zero trained-model inference, real test/OOD predictions, predictive metrics, threshold tuning, held-out exports, or dev-only runtime reuse.
> - **A6 (Binding Exclusions)**: Preserve all protected artifacts (`DataSet.csv`, `documents.md`, `data/**`, `dataset_manifest.json`, `reports/protected_baseline_282.json`, `reports/stage4_frozen_test_masks.json`, `reports/stage4_statistical_validation_results.json`, frozen `v1` configs, `v3–v8` kits, `response-local.md`). Disposable Gate A-1 dev-only exports are unsigned, pre-B.3, non-release-eligible development artifacts (`b3_syscall_hardening_implemented = False`, `release_eligible = False`) and never claim Section B.3 compliance. Exclude all Gate A-2 items, dependency installs, signing/keygen, live probes, deployment, GPU runs, Track A/B, synthetic generation, held-out eval, and flipping `CONFIRMATORY_DECLARATIONS_ENABLED` (`False`).
>
> My approvals apply only to the named decisions and scopes. Omitted, blank, or inconsistent decisions grant no authority. Specification sign-off does not authorize implementation, and engineering, deployment, provisioning, GPU, statistical, held-out, and confirmatory decisions are independent.
>
> **Research Owner signature** — `[UNSIGNED — OWNER DECISION REQUIRED — NO AUTHORIZATION GRANTED]`

---

**Document Footer:** `DRAFT — NOT SIGNED — NO AUTHORIZATION GRANTED.` Gate A-1 remains **NOT AUTHORIZED** and **NOT STARTED**. Gate A-2 remains **UNAUTHORIZED**.
