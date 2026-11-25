# Explore-AIs — Stage 4 v9 Remediation Plan & Three-Way Alignment (`response-another.md`)

> **Communication Note to ChatGPT (from Gemini & Research Owner):**
> Per the Research Owner's explicit instruction, [`response.md`](file:///home/bs01415/Documents/Programming/ExploringAIs/response.md) is preserved completely untouched as the committed Stage 4 v8 Independent Read-Only Audit Report (`HEAD 1e00018e9f2457d40c4d163e95da984690c1e510`). This companion file, [`response-another.md`](file:///home/bs01415/Documents/Programming/ExploringAIs/response-another.md), is the communication channel containing the complete proposed contents of `stage4_v9_implementation_plan.md`, repository-grounded answers to all 12 technical questions, the Three-Way Alignment Report, authorization gates, and explicit design questions for ChatGPT and the Research Owner before any Stage 4 v9 code or configuration changes are authorized.
>
> Zero source code, configurations, datasets, evaluation masks, historical portable kits (`v3`–`v8`), or `response.md` were modified; zero dependencies were installed; zero training or GPU benchmarks were executed; and zero held-out test/OOD rows were read.

---

## Questions for ChatGPT & Research Owner Alignment

Before authorizing Stage 4 v9 implementation (Gate A), the following three concrete technical conflicts uncovered by repository inspection require explicit alignment between the Research Owner and ChatGPT:

1. **Q-Align-1 — Cell 5 32-Pair Tokenizer Gate vs. Strict Zero Test-Parquet Access in Smoke Testing:**
   - **Verified repo fact:** [`load_documented_32_pairs()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_tokenizer_gate.py#L89-L123) reads `eval_smishdetect_test.parquet` (because `14` `real_train <-> test` pairs + `4` `test <-> val` pairs = `18` of the `32` Stage 3 pairs have one endpoint in `eval_smishdetect_test.parquet`) **and** unconditionally reads `eval_banglaphish_ood.parquet` (even though `0/32` pairs come from OOD). Furthermore, [`verify_portable_colab_kit()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_integrity.py#L604-L620) in Cell 5 calls [`build_smishdetect_test_cohort_masks()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_cohorts.py#L153-L239), which reads `eval_smishdetect_test.parquet` to re-derive the 4 frozen masks.
   - **Conflict:** You cannot simultaneously (a) forbid `eval_smishdetect_test.parquet` from ever being opened anywhere in Cell 5 + Cell 6 during smoke testing AND (b) run the full 32-pair tokenizer collision gate + portable kit mask re-verification in Cell 5.
   - **Question for ChatGPT:** Do you agree with **Option G1** — remove the unused `eval_banglaphish_ood.parquet` read from `load_documented_32_pairs()` immediately, allow Cell 5's read-only structural/ID-blind boolean checks (`verify_portable_colab_kit` mask hashes + 32-pair `input_ids_a == input_ids_b`), and enforce **strict zero test/OOD file opens, reads, predictions, metrics, or exports** across Cell 6 (`run_post_stage_b_pipeline`, `_load_configured_eval_sets`, `hf_train_and_predict_fold`, `aggregate_post_stage_b_results`) when `run_heldout_eval=False`? Or do you require a `14`-pair dev-only subset (`8` CV + `6` train↔val) in `gpu_smoke_test` so `load_documented_32_pairs()` never opens `eval_smishdetect_test.parquet` during smoke testing?

2. **Q-Align-2 — Smoke-Test Model & Experiment Matrix (`gpu_smoke_test_config.json`):**
   - **Verified repo fact:** Current [`gpu_smoke_test_config.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/configs/gpu_smoke_test_config.json#L4-L24) runs `["A", "D"]` × `["distilbert-base-multilingual-cased"]` × `["raw"]` × `[0]`. That skips `WeightedTrainer` (only used by Experiment `B`), skips `csebuetnlp/banglabert` + `normalizer`, skips `canon_masked`, and leaves `resolve_and_verify_tokenizer_revision()` untested on `xlm-roberta-base`, `google/muril-base-cased`, and `csebuetnlp/banglabert` in live `transformers==4.57.1`.
   - **Question for ChatGPT:** Between:
     - **Matrix M1 (Recommended by Gemini):** `experiments: ["B", "D"]`, `models: [all 4 pinned backbones]`, `text_views: ["canon_masked"]`, `folds: [0]` (`8` tiny capped units: `64` train / `32` val / `32` eval, `1` epoch, `batch_size=16`, ~`4` steps/unit = `32` total steps), which exercises `WeightedTrainer`, `Trainer`, `normalizer`, and live HF `4.57.1` snapshot-path resolution for **all 4** backbones; vs.
     - **Matrix M2:** `experiments: ["B", "D"]`, `models: ["csebuetnlp/banglabert", "xlm-roberta-base"]`, `text_views: ["canon_masked"]`, `folds: [0]` (`4` tiny units, leaving `muril-base-cased` and `distilbert` live `4.57.1` snapshot paths until `t4_benchmark`);
     which matrix do you recommend for v9? (Also confirm whether `gpu_smoke_test_config.json` should be updated in-place for v9 or preserved frozen alongside a new `gpu_smoke_test_config_v2.json`.)

3. **Q-Align-3 — Statistical Governance Path (`syn_stress::S1_both_zero` `62/1,000 = 0.0620 > 0.0600`):**
   - **Verified repo fact:** `CONFIRMATORY_DECLARATIONS_ENABLED = False` fails closed because `syn_stress::S1_both_zero` Recall Type-I error was `62/1,000 = 0.0620 > 0.0600` (`N_outer = 1,000`, `MC SE = 0.0076`). Meanwhile, `emp_1687::S1_both_zero` Recall (`0.0540`), Joint IUT (`0.0080`/`0.0090`), and all `Holm-12` primary FWER checks (`0.0530`, `0.0460`, `0.0510`) passed.
   - **Gemini's proposal:** Keep `CONFIRMATORY_DECLARATIONS_ENABLED = False` unconditionally throughout Stage 4 v9, T4 benchmark, and the 160-unit Option A execution (treat Option A results strictly as descriptive deltas + family-clustered 95% CIs + diagnostic p-values). Separately, **prior to or independent of** any future confirmatory manuscript claims, define a **pre-registered, single-shot recalibration protocol** ($N_{\text{outer}} \ge 5{,}000$, `MC SE <= 0.0031`, fixed seed, testing whether the `0.0620` spike was `N=1,000` MC noise or genuine ratio-estimator anti-conservatism on heavy-tail $m_g \in \{1..8\}$ Smishing clusters, plus a separate investigation of validation-threshold uncertainty).
   - **Question for ChatGPT:** Do you agree with keeping Stage 4 Option A strictly exploratory/descriptive under `CONFIRMATORY_DECLARATIONS_ENABLED = False` while documenting (without pre-committing to pass) a pre-registered $N_{\text{outer}} \ge 5{,}000$ recalibration protocol—or do you advise keeping Stage 4 permanently non-confirmatory regardless of simulation size due to `n_normal = 265` threshold calibration variance and single-seed conditioning?

---

# Part I — Complete Proposed Contents of `stage4_v9_implementation_plan.md`

## 1. Executive Summary

Stage 4 v8 achieved strong local engineering invariants (`112/112` mocked unit tests, `282/282` Category A baseline integrity, `4/4` frozen masks, `34/34` `v8` kit parity), yet failed independent audit (**FAIL — Not Ready for Live GPU Milestone**) due to four P0 defects:
1. **P0-1:** [`gpu_smoke_test_config.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/configs/gpu_smoke_test_config.json) enables `run_heldout_eval: true`, which reads, scores, and exports `32` rows each from `eval_smishdetect_test.parquet` and `eval_banglaphish_ood.parquet`, while omitting `csebuetnlp/banglabert` and `tokenizer_revisions` (bypassing Cell 5's 32-pair tokenizer collision gate and `normalizer` verification).
2. **P0-2:** Live Google Colab, CUDA PyTorch `>= 2.6`, `transformers==4.57.1`, `csebuetnlp/normalizer@d405944d...`, and Google Drive FUSE remain 100% unverified because the local `.venv` lacks all GPU/HF/normalizer libraries.
3. **P0-3:** Frozen [`post_stage_b_stage4_t4_benchmark_config.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json) sets `"benchmark_workload_fixture_eval": true` while [`SYNTHETIC_WORKLOAD_FIXTURE_ENABLED = False`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L92), blocking `t4_benchmark` pre-lease. Moreover, implementing Option B1 requires **runner-code changes** (not just JSON changes) because [`_load_configured_eval_sets()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L1491-L1500), [`hf_train_and_predict_fold()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L1219-L1231), and [`_prediction_frames()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L1756-L1757) currently gate `smishdetect_validation` (`708` rows—misstated as `757` in [`post_stage_b_stage4_v8_readiness_report.md`](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/post_stage_b_stage4_v8_readiness_report.md#L41)) behind `benchmark_workload_fixture_eval=True`.
4. **P0-4:** [`CONFIRMATORY_DECLARATIONS_ENABLED = False`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_stats.py#L38) because `syn_stress::S1_both_zero` Recall Type-I error (`62/1,000 = 0.0620`) exceeded `0.0600`. This governance lock must remain enabled-off (`False`) in v9.

Stage 4 v9 defines the smallest, defense-in-depth engineering and governance remediation required to unblock a **development-only live Colab smoke test**, followed by an **Option B1 development-only T4 benchmark**, while preserving all frozen baselines, datasets, masks, historical kits (`v3`–`v8`), and statistical safeguards unchanged.

---

## 2. Current Repository Baseline

Verified read-only at `main` HEAD `1e00018e9f2457d40c4d163e95da984690c1e510`:

- **Protected Datasets & Bundles:**
  - `data/processed/post_stage_b_bundle_v2/dataset_manifest.json` (`manifest_version: "post_stage_b_v2"`, `synthetic_generation_status: "PERMANENTLY_STOPPED"`):
    - `real_train_pool.parquet`: `3,999` rows (CV folds `0`–`4`, `~800` rows/fold; `799` in Fold `4`).
    - `synthetic_pilot_300.parquet`: `300` rows.
    - `synthetic_batches_b1_b7.parquet`: `345` rows.
    - `eval_smishdetect_validation.parquet`: **`708` rows** (`265` Normal, `178` Promotional, `265` Smishing; `606` families).
    - `eval_smishdetect_test.parquet`: `1,698` rows (`581` Normal, `384` Promotional, `733` Smishing; `1,304` families).
    - `eval_banglaphish_ood.parquet`: `5,416` rows (`2,336` Normal, `80` Promotional, `3,000` Smishing).
    - `counterfactual_smish_probe.parquet`: `1,279` rows (`orig`, `contact_swap`, `contact_brand_swap` across 4 views).
  - `282/282` Category A files indexed in `reports/protected_baseline_282.json`.
  - `4/4` frozen test cohort masks in `reports/stage4_frozen_test_masks.json` (`1,687`, `1,698`, `1,693`, `1,675`).
- **Historical Portable Kits:** `v3`, `v4`, `v5`, `v6`, `v7`, `v8` present under `stage4_colab_kit_upload/bangla_smish_post_stage_b_stage4_v{3..8}/`.
- **Current Configurations (`experiments/colab/configs/`):**
  - `gpu_smoke_test_config.json` (v8: `run_heldout_eval: true`, `distilbert` only, `raw` only, no `tokenizer_revisions`).
  - `post_stage_b_stage4_t4_benchmark_config.json` (v1: `benchmark_workload_fixture_eval: true`, fail-closed).
  - `post_stage_b_stage4_option_a_config.json` (v1: `160` units, `confirmatory_mode: true`, `run_heldout_eval: true`).
- **Statistical Validation Artifact:** `reports/stage4_statistical_validation_results.json` (`verdict: "FAIL_OR_INCONCLUSIVE_CONFIRMATORY_DISABLED"`).
- **Local Test & Package Baseline:** `pyproject.toml [gpu]` omits `normalizer`, `ftfy`, `emoji`, `wcwidth`; `.venv` contains CPU data/test deps only (zero `torch`, `transformers`, `tokenizers`, `datasets`, `accelerate`, `safetensors`, `sentencepiece`, `normalizer`).

---

## 3. Audit Finding → Remediation Mapping

| Finding | Severity | Exact Root Cause in v8 | v9 Remediation | Verification Mechanism |
|---|---|---|---|---|
| **P0-1: Held-out/OOD exposure in smoke test** | **P0** | `gpu_smoke_test_config.json` sets `run_heldout_eval: true`; `PostStageBConfig.__post_init__` (`post_stage_b_runner.py:300`) only blocks `run_heldout_eval=True` for `t4_benchmark`; `audit_model_view_overlaps` (`L2107`) reads held-out/OOD if `experiments_abcd` has `run_heldout_eval=False`. | Set `run_heldout_eval: false` in smoke config; block `run_heldout_eval=True` for `gpu_smoke_test` in `PostStageBConfig.__post_init__`; add runtime allowlist + export guards across loader/trainer/frames/metrics/archive/aggregator. | Sentinel `pd.read_parquet` & `Path.open` instrumentation tests proving zero test/OOD reads, predictions, metrics, or exports when `run_heldout_eval=False`. |
| **HIGH-1 / P0-1b: Smoke test skips BanglaBERT, `normalizer`, `canon_masked`, & 32-pair gate** | **P0** | `gpu_smoke_test_config.json` specifies `distilbert` + `raw` + no `tokenizer_revisions`; Cell 5 (`generate_post_stage_b_stage4_notebook.py:234`) exempts `gpu_smoke_test`; `load_documented_32_pairs` needlessly reads `eval_banglaphish_ood.parquet`. | Pin all 4 `tokenizer_revisions` in smoke config; include `csebuetnlp/banglabert` + `canon_masked` + `WeightedTrainer` (`B` + `D`); enforce `tokenizer_revisions` in Colab Cell 5 & `_validate_pre_run_compatibility`; remove `eval_banglaphish_ood.parquet` from `load_documented_32_pairs`. | Unit test verifying Cell 5 32-pair gate runs on `gpu_smoke_test`, checks `normalizer@d405944d...`, and rejects missing `tokenizer_revisions`; live Colab smoke test. |
| **P0-2: Live Colab/HF/CUDA/FUSE unverified** | **P0** | Local `.venv` lacks GPU/HF packages; all HF/GPU/Drive tests are CPU-mocked. | Add explicit live-Colab smoke protocol + diagnostic logging in `resolve_and_verify_tokenizer_revision()` (`candidate_paths` vs `cached_paths`) + `GradScaler` overflow diagnostic counter if accessible. | Separate live-Colab acceptance checklist (never conflated with CPU-mocked pass). |
| **MED-3: `pyproject.toml [gpu]` missing BanglaBERT deps** | **Med** | `pyproject.toml:19-37` omits `normalizer`, `ftfy`, `emoji`, `wcwidth`. | Add `normalizer @ git+https://github.com/csebuetnlp/normalizer@d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9`, `ftfy`, `emoji`, `wcwidth` to `pyproject.toml [project.optional-dependencies] gpu` (matching `post_stage_b_colab_deps.py:92-101`). | Automated parity test asserting `pyproject.toml [gpu]` covers all 24 distributions in `post_stage_b_colab_deps.py` & `post_stage_b_runner.py` (excluding `unicodedata2`). |
| **P0-3: T4 benchmark fail-closed on v1 config** | **P0** | `post_stage_b_stage4_t4_benchmark_config.json` sets `benchmark_workload_fixture_eval: true` with `SYNTHETIC_WORKLOAD_FIXTURE_ENABLED = False`; runner gates `smishdetect_validation` inference behind `benchmark_workload_fixture_eval=True`. | Preserve v1 T4 config untouched; create `post_stage_b_stage4_t4_benchmark_config_v2.json` (`benchmark_workload_fixture_eval: false`); update runner (`L1219`, `L1496`, `L1756`), integrity (`post_stage_b_integrity.py`), and notebook `CFG_MAP` to execute Option B1 on unmodified dev splits (`val`, `val_adv`, `smishdetect_validation` 708 rows, `cf` Fold 0) + analytical OOD bound report. | Tests proving v1 remains fail-closed & untouched, v2 executes Option B1 with 0 synthetic rows & 0 test/OOD reads, and emits `t4_benchmark_calibration_estimate.json` distinguishing measured vs analytical OOD estimates. |
| **P0-4: Statistical confirmatory gate disabled** | **P0** | `syn_stress::S1_both_zero` Recall `62/1,000 = 0.0620 > 0.0600`; `CONFIRMATORY_DECLARATIONS_ENABLED = False`. | **No threshold/code relaxation.** Keep `CONFIRMATORY_DECLARATIONS_ENABLED = False`; ensure all Option A & benchmark exports clearly label `UNVALIDATED_DIAGNOSTIC_ONLY` and separate exploratory C vs D / descriptive CIs from confirmatory claims. | Existing governance bypass tests + explicit export assertions. |

---

## 4. Exact Files to Create or Modify

### Preserved Untouched (Zero Edits)
- `data/**` (all raw, curated, generated, `post_stage_b_bundle`, `post_stage_b_bundle_v2` — `282/282` Category A protected files)
- `DataSet.csv`
- `reports/protected_baseline_282.json`
- `reports/stage4_frozen_test_masks.json`
- `reports/stage4_statistical_validation_results.json`
- `experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json` (frozen v1 T4 config)
- `experiments/colab/configs/post_stage_b_stage4_option_a_config.json` (frozen 160-unit Option A config)
- `experiments/colab/configs/post_stage_b_abcd_config.json`
- `stage4_colab_kit_upload/bangla_smish_post_stage_b_stage4_v3` through `v8`
- `response.md` (v8 audit report)

### Modified Files
1. [`pyproject.toml`](file:///home/bs01415/Documents/Programming/ExploringAIs/pyproject.toml) — Sync `[project.optional-dependencies] gpu` with `normalizer@d405944d...`, `ftfy`, `emoji`, `wcwidth`.
2. [`experiments/colab/configs/gpu_smoke_test_config.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/configs/gpu_smoke_test_config.json) *(or new `gpu_smoke_test_config_v2.json` if Owner wants v1 smoke JSON kept immutable; see Section 5 & Q-Align-2)* — Remediate smoke test: `run_heldout_eval: false`, `canon_masked`, `csebuetnlp/banglabert` (+ pinned `tokenizer_revisions`), `["B", "D"]`.
3. [`src/bangla_smish/post_stage_b_runner.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py) — Fail-closed `gpu_smoke_test` held-out guard in `PostStageBConfig.__post_init__`; execution-layer dataset allowlists; `tokenizer_revisions` check for `hf_train_and_predict_fold` in `gpu_smoke_test`; Option B1 dev-only `smishdetect_validation` (`708` rows) loading/inference/reporting + analytical OOD extrapolation artifact; guard `audit_model_view_overlaps` against `run_heldout_eval=False`.
4. [`src/bangla_smish/post_stage_b_tokenization.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_tokenization.py) — Add optional `return_source` / structured provenance logging helper so live Colab smoke run explicitly records whether Step 2 (`candidate_paths` snapshot dir) vs Step 4 (`cached_paths`) verified each backbone.
5. [`src/bangla_smish/post_stage_b_tokenizer_gate.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_tokenizer_gate.py) — Remove unused `eval_banglaphish_ood.parquet` from `load_documented_32_pairs()`; optionally support `pair_scope="all_32" | "dev_only_14"` depending on Owner/ChatGPT resolution of Q-Align-1.
6. [`src/bangla_smish/post_stage_b_integrity.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_integrity.py) — Register `post_stage_b_stage4_t4_benchmark_config_v2.json` in `STAGE4_KIT_REQUIRED_CONFIGS` (keeping v1 in `STAGE4_KIT_REQUIRED_CONFIGS` as well so historical reference & v1 fail-closed tests inside kit remain intact), add `t4_benchmark_v2_config_sha256` to kit manifest, and validate both `v1` and `v2` T4 config safety flags in `verify_portable_colab_kit()`.
7. [`scripts/generate_post_stage_b_stage4_notebook.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/scripts/generate_post_stage_b_stage4_notebook.py) — Bump defaults to `bangla_smish_post_stage_b_stage4_v9` / `stage4-v9-smoke-01`; point `CFG_MAP['t4_benchmark']` to `post_stage_b_stage4_t4_benchmark_config_v2.json`; require `cfg.tokenizer_revisions` in Cell 5 for ALL run modes including `gpu_smoke_test`; update Cell 0 markdown text for v9 Option B1 & smoke test.
8. [`experiments/colab/post_stage_b_stage4_experiments.ipynb`](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/post_stage_b_stage4_experiments.ipynb) — Regenerated deterministically by `scripts/generate_post_stage_b_stage4_notebook.py`.
9. Existing tests needing alignment with remediated smoke config / v2 T4 config: [`tests/test_post_stage_b_pipeline.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/tests/test_post_stage_b_pipeline.py), [`tests/test_stage4_checkpointing_and_kit.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/tests/test_stage4_checkpointing_and_kit.py), [`tests/test_stage4_v8_remediation.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/tests/test_stage4_v8_remediation.py).

### New Files
1. [`experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config_v2.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config_v2.json) — Option B1 dev-only T4 benchmark config (`benchmark_workload_fixture_eval: false`).
2. [`tests/test_stage4_v9_remediation.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/tests/test_stage4_v9_remediation.py) — Complete v9 regression matrix (Section 11).
3. `stage4_colab_kit_upload/bangla_smish_post_stage_b_stage4_v9/` — Newly staged portable kit + `stage4_kit_manifest.json` (only generated during authorized release step).
4. `reports/post_stage_b_stage4_v9_*` — Implementation, test, integrity, and readiness reports (during authorized release step).

---

## 5. Proposed Configuration Values

### A. Proposed `gpu_smoke_test_config.json` (Pending Q-Align-2 on 4 Backbones vs 2 Backbones)

Recommended configuration (`Matrix M1`, 8 tiny units = `~32` total steps across `4` backbones × `2` experiments):

```json
{
  "run_name": "post_stage_b_gpu_smoke_test_v3",
  "run_mode": "gpu_smoke_test",
  "experiments": ["B", "D"],
  "models": [
    "csebuetnlp/banglabert",
    "google/muril-base-cased",
    "xlm-roberta-base",
    "distilbert-base-multilingual-cased"
  ],
  "tokenizer_revisions": {
    "csebuetnlp/banglabert": "9ce791f330578f50da6bc52b54205166fb5d1c8c",
    "google/muril-base-cased": "afd9f36c7923d54e97903922ff1b260d091d202f",
    "xlm-roberta-base": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
    "distilbert-base-multilingual-cased": "45c032ab32cc946ad88a166f7cb282f58c753c2e"
  },
  "text_views": ["canon_masked"],
  "folds": [0],
  "epochs": 1,
  "batch_size": 16,
  "gradient_accumulation_steps": 1,
  "eval_batch_size": 32,
  "learning_rate": 3e-5,
  "weight_decay": 0.01,
  "warmup_ratio": 0.1,
  "max_length": 128,
  "seed": 42,
  "fp16": "auto",
  "experiment_b_balance_mode": "class_weight",
  "run_heldout_eval": false,
  "run_adversarial_eval": true,
  "run_counterfactual_eval": true,
  "keep_model_weights": "none",
  "checkpoint_retention_limit": 1,
  "local_checkpoint_root": "/content/tmp_stage4_ckpt",
  "confirmatory_mode": false,
  "benchmark_workload_fixture_eval": false,
  "smoke_max_train_samples": 64,
  "smoke_max_val_samples": 32,
  "smoke_max_eval_samples": 32
}
```

**Justification against existing implementation:**
- `experiments: ["B", "D"]`: `B` exercises `_weighted_trainer_class` (`WeightedTrainer`, `post_stage_b_runner.py:710-724,1137`) + `training_class_weights` recalibration under smoke caps (`L1724-1728`); `D` exercises standard HuggingFace `Trainer` + synthetic pool loading (`real_train_pool` + `synthetic_pilot_300` + `synthetic_batches_b1_b7`).
- `models`: All 4 Option A backbones (`OPTION_A_BACKBONES`) are pinned to the exact 40-hex SHAs from `PINNED_TOKENIZER_REVISIONS` (`post_stage_b_tokenizer_gate.py:34-39`), ensuring Cell 5's `run_allowed_view_tokenizer_collision_gate()`, Cell 6's `resolve_and_verify_tokenizer_revision()`, and `AutoModelForSequenceClassification` `model.config._commit_hash` verification are empirically tested on **all 4** models under live `transformers==4.57.1` before T4 benchmarking. (`xlm-roberta-base` also exercises SentencePiece + heaviest VRAM; `csebuetnlp/banglabert` exercises `normalizer@d405944d...`).
- `text_views: ["canon_masked"]`: Exercises the primary Option A decontaminated view (`text_canon_masked` + `text_adv_canon_masked` + `counterfactual_smish_probe` `canon_masked` columns) and runs the 32-pair collision gate (`32 × 4 × 1 = 128` checks) on `canon_masked`.
- `smoke_max_train_samples: 64`, `batch_size: 16`, `epochs: 1`: Exactly `4` gradient steps/unit (`32` steps total across 8 units) — completes in ~2–4 minutes on a single Colab T4 while still triggering epoch-end `eval_strategy="epoch"`, `save_strategy="epoch"`, `Stage4CheckpointCallback.on_save`, FP16 `scaler.pt` hashing, Drive staging sync, and resume/cleanup.
- `run_heldout_eval: false`: Zero held-out/OOD scoring or exports.
- `local_checkpoint_root: "/content/tmp_stage4_ckpt"` + `checkpoint_retention_limit: 1`: Exercises the exact Local-SSD + Drive staging path used in T4 benchmark and Option A.

*(Note: If the Research Owner selects Matrix M2 — `["csebuetnlp/banglabert", "xlm-roberta-base"]` only — `models` and `tokenizer_revisions` shrink to those 2 entries, giving 4 units / 16 steps.)*

### B. Proposed `post_stage_b_stage4_t4_benchmark_config_v2.json` (Option B1)

```json
{
  "run_name": "post_stage_b_stage4_t4_benchmark_v2",
  "run_mode": "t4_benchmark",
  "experiments": ["B", "D"],
  "models": [
    "csebuetnlp/banglabert",
    "google/muril-base-cased",
    "xlm-roberta-base",
    "distilbert-base-multilingual-cased"
  ],
  "tokenizer_revisions": {
    "csebuetnlp/banglabert": "9ce791f330578f50da6bc52b54205166fb5d1c8c",
    "google/muril-base-cased": "afd9f36c7923d54e97903922ff1b260d091d202f",
    "xlm-roberta-base": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
    "distilbert-base-multilingual-cased": "45c032ab32cc946ad88a166f7cb282f58c753c2e"
  },
  "text_views": ["canon_masked"],
  "folds": [0],
  "epochs": 3,
  "batch_size": 32,
  "gradient_accumulation_steps": 1,
  "eval_batch_size": 64,
  "learning_rate": 3e-5,
  "weight_decay": 0.01,
  "warmup_ratio": 0.1,
  "max_length": 128,
  "seed": 42,
  "fp16": "auto",
  "experiment_b_balance_mode": "class_weight",
  "run_heldout_eval": false,
  "run_adversarial_eval": true,
  "run_counterfactual_eval": true,
  "keep_model_weights": "none",
  "checkpoint_retention_limit": 1,
  "local_checkpoint_root": "/content/tmp_stage4_ckpt",
  "confirmatory_mode": false,
  "benchmark_workload_fixture_eval": false,
  "unit_schedule": [
    ["D", "csebuetnlp/banglabert", "canon_masked", 0],
    ["D", "google/muril-base-cased", "canon_masked", 0],
    ["D", "xlm-roberta-base", "canon_masked", 0],
    ["D", "distilbert-base-multilingual-cased", "canon_masked", 0],
    ["B", "xlm-roberta-base", "canon_masked", 0]
  ],
  "smoke_max_train_samples": null,
  "smoke_max_val_samples": null,
  "smoke_max_eval_samples": null
}
```

**Why `v1` must remain untouched:** `tests/test_stage4_final_remediation.py:1376-1428` and `tests/test_stage4_v8_remediation.py:1146-1166` explicitly assert that `post_stage_b_stage4_t4_benchmark_config.json` keeps `"benchmark_workload_fixture_eval": true` and fails closed pre-lease.

---

## 6. Detailed Code-Change Plan

### 6.1 `src/bangla_smish/post_stage_b_runner.py`
1. **Configuration-level fail-closed guard (`PostStageBConfig.__post_init__`, lines 300–311):**
   - **Why:** Currently `L300` only forbids `run_heldout_eval=True` for `t4_benchmark`.
   - **Implementation:** Change `L300` to:
     ```python
     if self.run_mode in {"t4_benchmark", "gpu_smoke_test"} and self.run_heldout_eval:
         raise ValueError(
             f"{self.run_mode} must set run_heldout_eval=False to prevent held-out/OOD exposure"
         )
     ```
   - **What could go wrong (Critical Repo Check):** Existing tests (`tests/test_post_stage_b_pipeline.py:229`, `tests/test_stage4_v8_remediation.py:1027`, `tests/test_post_stage_b_preprocessing.py:102`) load `gpu_smoke_test_config.json`, AND 10+ unit tests (`test_post_stage_b_runner.py:40`, `test_stage4_v8_remediation.py:251,410,509,...`) construct `PostStageBConfig(run_mode="gpu_smoke_test", ...)` **without** `tokenizer_revisions`. Therefore:
     - **Do NOT** force `tokenizer_revisions` inside `PostStageBConfig.__post_init__` for mock/synthetic `gpu_smoke_test` configs unless all unit test fixtures are updated; **instead**, enforce `tokenizer_revisions` for `gpu_smoke_test` inside `_validate_pre_run_compatibility()` whenever `runner_fn is hf_train_and_predict_fold` AND inside Colab Notebook Cell 5 (`generate_post_stage_b_stage4_notebook.py:234`). That keeps synthetic CPU unit tests clean while making real HuggingFace/GPU smoke execution fail-closed if `tokenizer_revisions` is omitted or incomplete!

2. **Execution-layer dataset allowlists & held-out/OOD isolation (`_load_configured_eval_sets`, `_validate_eval_sets_allowlist`, `_prediction_frames`, `_compute_unit_metrics`, `_write_prediction_archive`, `_run_locked`):**
   - **Why:** Relying on `run_heldout_eval=False` in one `if` branch is insufficient defense-in-depth.
   - **Implementation:**
     - Define explicit module constants:
       - `FORBIDDEN_NON_HELDOUT_KEYS: frozenset[str] = frozenset({"smishdetect_test", "banglaphish_ood"})`
       - `ALLOWED_DEV_EVAL_KEYS: frozenset[str] = frozenset({"smishdetect_validation", "counterfactual_smish_probe"})`
     - Add `_assert_eval_sets_allowed(eval_sets: dict[str, Any], cfg: PostStageBConfig, *, context: str) -> None` which unconditionally raises `RuntimeError` if `not cfg.run_heldout_eval` and any key in `FORBIDDEN_NON_HELDOUT_KEYS` (or any unknown key not in `ALLOWED_DEV_EVAL_KEYS`) is present. Call it inside `_load_configured_eval_sets()`, `hf_train_and_predict_fold()`, `_prediction_frames()`, `_compute_unit_metrics()`, and `_write_prediction_archive()`.
     - In `_write_prediction_archive()`, add an explicit key-prefix check rejecting `smishdetect_test_*` or `banglaphish_ood_*` arrays whenever `not cfg.run_heldout_eval` (pass `cfg` or enforce via `_prediction_frames`).
     - In `_run_locked()` (`L2106`), guard `audit_model_view_overlaps(...)`: if `cfg.run_mode == "experiments_abcd"` and `not cfg.run_heldout_eval`, fail closed or restrict overlap audit to non-heldout splits — because `audit_model_view_overlaps()` (`post_stage_b_model_view_audit.py:38`) unconditionally calls `load_heldout_eval_sets(bundle_dir)`! (Specifically, if `not cfg.run_heldout_eval`, `experiments_abcd` should never call `load_heldout_eval_sets`.)

3. **Option B1 T4 Benchmark Execution & Dev-Only Inference (`_load_configured_eval_sets`, `hf_train_and_predict_fold`, `_prediction_frames`, `_compute_unit_metrics`, `_aggregate_verified`):**
   - **Why:** Currently `smishdetect_validation` (`708` rows) is only loaded/scored in `t4_benchmark` when `cfg.benchmark_workload_fixture_eval` is `True` (`L1219, L1496, L1756`). Setting `benchmark_workload_fixture_eval: false` without updating the runner would silently skip `smishdetect_validation` timing!
   - **Implementation:**
     - In `_load_configured_eval_sets()` (`L1496`): Load `eval_sets["smishdetect_validation"] = pd.read_parquet(bundle_dir / "eval_smishdetect_validation.parquet")` whenever `cfg.run_mode == "t4_benchmark"` (regardless of `benchmark_workload_fixture_eval`), and keep `benchmark_workload_fixture_eval=True` blocked fail-closed when `SYNTHETIC_WORKLOAD_FIXTURE_ENABLED=False`.
     - In `hf_train_and_predict_fold()`:
       - Keep the pre-init fail-closed check on `cfg.benchmark_workload_fixture_eval and not SYNTHETIC_WORKLOAD_FIXTURE_ENABLED` (`L884-893`).
       - Replace the `elif cfg.run_mode == "t4_benchmark" and cfg.benchmark_workload_fixture_eval:` block (`L1219-1231`) with a clean `elif cfg.run_mode == "t4_benchmark":` Option B1 path that runs `smishdetect_validation` (`708` unmodified dev rows) with separate high-resolution timing (`sd_val_tok_seconds`, `sd_val_predict_seconds`, `dev_inference_rows_total`, `dev_inference_seconds`, `dev_eval_padded_tokens_total` if inspectable from collator/dataset lengths) and zero synthetic fixture generation.
     - In `_prediction_frames()` (`L1756`): Always include `frames["smishdetect_validation"] = eval_sets["smishdetect_validation"].iloc[:cap]` when `cfg.run_mode == "t4_benchmark"`.
     - **Structured Benchmark Calibration Report (`_aggregate_verified` / `_write_summary`):** When `cfg.run_mode == "t4_benchmark"`, emit `exports/t4_benchmark_calibration_report.json` containing:
       - Measured training telemetry per unit (`peak_cuda_allocated_bytes`, `peak_cuda_reserved_bytes`, `train_seconds`, `optimizer_step_ms`, `drive_sync_seconds`).
       - Measured unmodified dev inference telemetry (`val` 800 + `val_adv` 800 + `smishdetect_validation` 708 + Fold-0 `cf` ~`255×3`, total dev rows, `measured_dev_throughput_rows_per_sec`, `measured_sd_val_throughput_rows_per_sec`).
       - Clearly separated **Analytical OOD Inference Estimate** (`analytical_estimate_only: true`, `ood_rows_target: 5416`, `sd_test_rows_target: 1698`, `safety_factor_range: [1.5, 2.0]`, explicit formula and assumptions/limitations).
   - **What could go wrong:** Existing tests in `tests/test_stage4_final_remediation.py:1113,1137` construct `replace(t4_cfg, benchmark_workload_fixture_eval=False)` and assert `seen_eval_keys == [{"counterfactual_smish_probe"}]`! Those test assertions must be updated to expect `{"smishdetect_validation", "counterfactual_smish_probe"}` because Option B1 explicitly evaluates `smishdetect_validation` (`708` rows) when `benchmark_workload_fixture_eval=False`.

4. **Strict `hf_train_and_predict_fold` Revision Pinning (`L914-946, L973-991`):**
   - In `_validate_pre_run_compatibility()` (`L2048`): when `cfg.run_mode == "gpu_smoke_test"` and `runner_fn is hf_train_and_predict_fold`, require `cfg.tokenizer_revisions` to map every model in `cfg.models` to a valid 40-hex SHA.
   - Inside `hf_train_and_predict_fold()`: when `cfg.run_mode in {"gpu_smoke_test", "t4_benchmark", "experiments_abcd"}` (i.e., always for real HF runs), fail closed if `not pinned_rev` rather than silently calling unpinned `AutoTokenizer.from_pretrained(model_name)`.

### 6.2 `src/bangla_smish/post_stage_b_tokenization.py`
- **Why:** Audit finding MED-2 (`response.md:157-163`) noted that if a fast tokenizer in `transformers==4.57.1` leaves `_TOKENIZER_PATH_KEYS` empty and relies on Step 4 (`cached_paths`), Step 4 checks cache existence rather than direct in-memory path provenance.
- **Implementation:** Add `resolve_and_verify_tokenizer_revision_with_evidence(...) -> dict[str, Any]` (keeping `resolve_and_verify_tokenizer_revision(...) -> str` backward-compatible) returning `{"revision": expected, "verification_source": "explicit_commit_hash" | "snapshot_path" | "hf_cache_lookup", "candidate_paths_count": len(candidate_paths), "cached_paths_count": len(cached_paths)}` and log this in `hf_train_and_predict_fold` (`run_out["tokenizer_verification_source"]`) so the live Colab smoke test empirically proves whether Step 2 (`snapshot_path`) fired for all 4 backbones. (Remember: keep top-level imports stdlib-only per module docstring `L3-4`!)

### 6.3 `src/bangla_smish/post_stage_b_tokenizer_gate.py`
- **Why:** `load_documented_32_pairs()` (`L92-99`) currently reads `eval_banglaphish_ood.parquet` unconditionally despite `0/32` pairs coming from OOD (`post_stage_b_stage3_readiness_review.md:13,25-28`).
- **Implementation:**
  - Remove `"eval_banglaphish_ood.parquet"` from `parquets` in `load_documented_32_pairs()`.
  - Depending on Q-Align-1 resolution, if a dev-only smoke gate is chosen for `gpu_smoke_test`, support `pair_scope="dev_only_14"` (excluding `eval_smishdetect_test.parquet` and checking the 14 pairs whose both `row_id`s reside in `real_train_pool` + `eval_smishdetect_validation`); otherwise (recommended Option G1), run all 32 pairs in Cell 5 as an ID-blind structural check (`["row_id", text_col]` only, no labels/predictions/metrics) while keeping Cell 6 strictly test/OOD-free.

### 6.4 `src/bangla_smish/post_stage_b_integrity.py` & `scripts/generate_post_stage_b_stage4_notebook.py`
- **Why:** `STAGE4_KIT_REQUIRED_CONFIGS` (`L385-389`), `build_portable_colab_kit_manifest` (`L510`), and `verify_portable_colab_kit` (`L626-632`) hardcode v1 T4 config only; `generate_post_stage_b_stage4_notebook.py` hardcodes `v8` workspace names, v1 T4 config in `CFG_MAP`, and skips `tokenizer_revisions` enforcement when `RUN_MODE == 'gpu_smoke_test'` (`L234`).
- **Implementation:**
  - Add `"experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config_v2.json"` to `STAGE4_KIT_REQUIRED_CONFIGS` (keeping v1 in the kit so existing v1 fail-closed tests pass unmodified).
  - Update `verify_portable_colab_kit()` to verify `run_heldout_eval is False` and pinned `tokenizer_revisions` for **both** `t4_benchmark` (v1 & v2), `gpu_smoke_test` (`run_heldout_eval is False` + pinned `tokenizer_revisions`), and `option_a` (`confirmatory_mode is True` + pinned `tokenizer_revisions`), plus `benchmark_workload_fixture_eval is False` on `t4_benchmark_v2`.
  - Update `generate_post_stage_b_stage4_notebook.py`:
    - `DRIVE_WORKSPACE_REL = 'bangla_smish_post_stage_b_stage4_v9'`
    - `ARTIFACT_RUN_ID = 'stage4-v9-smoke-01'`
    - `CFG_MAP['t4_benchmark'] = 'post_stage_b_stage4_t4_benchmark_config_v2.json'`
    - In Cell 5 (`L214-235`): **Always** require `cfg.tokenizer_revisions` and run `run_allowed_view_tokenizer_collision_gate` (removing `elif RUN_MODE != 'gpu_smoke_test':` bypass!).
    - In Cell 6 (`L266`): Fail closed if `not TOKENIZER_COLLISION_GATE_PASSED` across **all** modes.

---

## 7. Dependency Remediation

- **Current gap (`pyproject.toml:19-37`):** `[project.optional-dependencies] gpu` defines 20 packages (17 listed + `peft`, `bitsandbytes`, `onnxruntime` — wait: note `numpy`, `pandas`, `scipy`, `scikit-learn`, `pyarrow`, `requests` are in `dependencies` `[L6-16]` and `torch` + 13 others + `peft`/`bitsandbytes`/`onnxruntime` are in `[gpu]`) and **omits** `normalizer`, `ftfy`, `emoji`, `wcwidth`. Meanwhile [`post_stage_b_colab_deps.py:92-109`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_colab_deps.py#L92-L109) and [`post_stage_b_runner.py:96-150`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L96-L150) already align on the exact 24-distribution closure (20 core + 4 BanglaBERT, zero `unicodedata2`).
- **Exact `pyproject.toml [gpu]` additions (preserving approved normalizer Git commit and avoiding upgrades):**
  ```toml
  "normalizer @ git+https://github.com/csebuetnlp/normalizer@d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9",
  "ftfy==6.0.3",
  "emoji==1.4.2",
  "wcwidth>=0.2.5",
  ```
  - *Check for dependency conflicts before editing:* Upstream `csebuetnlp/normalizer@d405944d...` `setup.py` pins `emoji==1.4.2` and `ftfy==6.0.3` and `regex` (`regex!=2019.12.17` in `STAGE4_BASE_PACKAGES`); `ftfy==6.0.3` requires `wcwidth` (any version, e.g. `>=0.2.5`, compatible with `ipython`/`prompt_toolkit` in Colab). Using `emoji==1.4.2` and `ftfy==6.0.3` matches `normalizer`'s exact upstream `setup.py` pins without forcing unnecessary upgrades. Wait: check whether Colab pre-installs a newer `emoji` or `wcwidth` — `pip` will cleanly install `emoji==1.4.2` and `ftfy==6.0.3` as `normalizer`'s own `install_requires` already demands them; `transformers==4.57.1` does not depend on `emoji` or `ftfy`.
- **Preserving Colab's CUDA PyTorch:**
  - Never pass `torch` in `STAGE4_BASE_PACKAGES` (`post_stage_b_colab_deps.py:37-57`).
  - Keep `preflight_colab_dependencies()` writing `torch=={torch_ver}` to `constraints.txt` (`L204`) and invoking `pip install -q -c constraints.txt`, followed by mandatory `RESTART REQUIRED` (`L207-210`) and `torch.cuda.is_available()` check (`L221-225`).
- **Live Colab verification procedure (during authorized smoke test):**
  1. Run Cell 4 (`preflight_colab_dependencies(cfg_models, install_missing=True, require_cuda=True)`).
  2. Observe `RESTART REQUIRED` on fresh VM -> click `Runtime -> Restart session` in the same GPU runtime -> re-run Cells 1–4.
  3. Confirm Cell 4 prints `"status": "verified"`, `torch` remains Colab's CUDA build (e.g., `2.x+cu12x`), `cuda_available: true`, `banglabert_normalizer_commit: "d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9"`, and `direct_url.json` `"url": "https://github.com/csebuetnlp/normalizer"` matched `check_requirement_status()`.

---

## 8. GPU Smoke-Test Design

- **Objective:** Verify end-to-end live Colab integration (CUDA PyTorch preservation, `normalizer@d405944d...`, `transformers==4.57.1` snapshot pinning, Cell 5 tokenizer collision gate, `Trainer` + `WeightedTrainer`, FP16 `scaler.pt`, Local-SSD + Drive FUSE staging, and `metrics.json`/`preds.npz`/`exports` integrity) with **zero** held-out test/OOD exposure.
- **Workload:** `experiments: ["B", "D"]`, `text_views: ["canon_masked"]`, `folds: [0]`, `epochs: 1`, `batch_size: 16`, `smoke_max_train_samples: 64`, `smoke_max_val_samples: 32`, `smoke_max_eval_samples: 32`, `models`: all 4 pinned backbones (`Matrix M1`, 8 units / 32 total steps; or 2 backbones `Matrix M2`, 4 units / 16 steps pending Q-Align-2).
- **Development-only evaluation splits exercised:**
  - Fold-0 `val_df` (`32` rows)
  - Fold-0 `val_adv` (`32` rows, `text_adv_canon_masked`)
  - Fold-0 `counterfactual_smish_probe` (`32` rows × 3 variants: `orig`, `contact_swap`, `contact_brand_swap`)
  - *(Optional/Excluded in smoke)* `smishdetect_validation` is **not** needed in `gpu_smoke_test` (reserved for `t4_benchmark` and `experiments_abcd`), keeping smoke eval minimal (`32` rows/split).
- **Multi-layer Held-Out/OOD Safeguards:**
  1. `PostStageBConfig.__post_init__`: rejects `run_heldout_eval=True` when `run_mode == "gpu_smoke_test"`.
  2. `_load_configured_eval_sets()`: never calls `load_heldout_eval_sets()`; reads only `counterfactual_smish_probe.parquet` (and `eval_smishdetect_validation.parquet` in `t4_benchmark` only).
  3. `hf_train_and_predict_fold()` & `_prediction_frames()` & `_compute_unit_metrics()` & `_write_prediction_archive()`: all assert `FORBIDDEN_NON_HELDOUT_KEYS` are absent when `not cfg.run_heldout_eval`.
  4. `_aggregate_verified()` (`L2686`): `complete_heldout_groups` requires `cfg.run_heldout_eval` (`False` -> `ensemble_5fold_evaluation: null`).
  5. Export audit: `metrics.json`, `preds.npz`, `aggregated_results.json`, `aggregated_results.csv`, and `post_stage_b_report.md` contain `None` / absent keys for `smishdetect_test` and `banglaphish_ood`.

---

## 9. T4 B1 Benchmark Design

- **Why Configuration-Only Changes Are Insufficient:** In v8 (`post_stage_b_runner.py:1219,1496,1756`), `smishdetect_validation` (`708` rows) is loaded, predicted, and recorded in `t4_benchmark` **only** when `cfg.benchmark_workload_fixture_eval` is `True`. Simply setting `benchmark_workload_fixture_eval: false` in `v2` JSON without modifying the runner would silently drop `smishdetect_validation` from benchmark telemetry.
- **Exact Unmodified Development Splits Measured per T4 Unit (`5` units: `D × 4 backbones + B × XLM-R` on `canon_masked`, Fold 0, `3` epochs, `batch_size=32`, `eval_batch_size=64`, `max_length=128`):**
  1. **Training (`100%` of peak VRAM & `>90%` of wall time):**
     - `Exp D` Fold 0: `3,844` rows (`3,199` real + `645` synthetic) → `363` steps (`121` steps/epoch × 3 epochs).
     - `Exp B` Fold 0 (`xlm-roberta-base`): `3,199` real rows (`WeightedTrainer`) → `300` steps (`100` steps/epoch × 3 epochs).
  2. **Unmodified Dev Inference (`eval_batch_size=64`, `no_grad`):**
     - Fold-0 `val_df`: `800` rows (`real_train_pool.parquet` Fold 0).
     - Fold-0 `val_adv`: `800` rows (`text_adv_canon_masked`).
     - `eval_smishdetect_validation.parquet`: **`708` rows** (`dataset_manifest.json:182`; correcting the `757` typo in `v8_readiness_report.md:41`).
     - Fold-0 `counterfactual_smish_probe.parquet`: Fold-0 subset (`~255`–`261` rows × 3 variants = `~765`–`783` sequences).
     - Total unmodified dev inference sequences per unit: `~3,073`–`3,091` sequences (zero synthetic strings, zero held-out/OOD rows).
- **Measurements Collected:**
  - `peak_cuda_allocated_bytes`, `peak_cuda_reserved_bytes` (`torch.cuda.max_memory_allocated/reserved()`), `fp16` status.
  - `train_seconds`, `attempt_optimizer_step_ms`, `eval_seconds`, `local_checkpoint_seconds`, `drive_sync_seconds`, `init_seconds`, `tokenization_seconds`.
  - Separate `sd_val_inference_seconds` and `dev_inference_total_seconds` (`rows/s`).
  - `actual_global_step` (`363` for D, `300` for B) vs `expected_optimizer_steps`.
- **Analytical OOD Inference Estimation & Safety Factor Justification (`1.5×–2.0×`):**
  - Target future held-out/OOD inference volume per Option A unit: `smishdetect_test` (`1,698` rows) + `banglaphish_ood` (`5,416` rows) = `7,114` rows (`eval_batch_size=64`, `max_length=128`).
  - **Scientific basis & limits of `1.5×–2.0×`:**
    1. All Option A models enforce `max_length = 128` (`post_stage_b_runner.py:959`) with `DataCollatorWithPadding` (`L1143`).
    2. `banglaphish_ood` has longer messages (`~165` chars mean vs `~80–84` chars in dev splits, per `response.md:230-231` without touching OOD data), so dynamic batch padding in dev yields shorter average padded sequence length $\bar{L}_{\text{dev}} < \bar{L}_{\text{OOD}} \le 128$.
    3. At $L \le 128$, BERT/RoBERTa inference time is dominated by linear projections $O(B \cdot L_{\text{batch}})$ plus fixed Python/DataLoader/tokenization overhead. Thus, even if $\bar{L}_{\text{OOD}}$ saturates near the `128`-token cap while $\bar{L}_{\text{dev}} \approx 64$–$85$, the GPU compute ratio is bounded by $128 / \bar{L}_{\text{dev}} \approx 1.5\times\text{–}2.0\times$ (plus CPU `normalizer` / subword tokenization scaling linearly with character length).
    4. **Crucial limitation:** Without measuring `padding="max_length"` or reading OOD token lengths, `1.5×–2.0×` is an **analytical bounding assumption**, NOT an empirical OOD benchmark. **Optional low-cost enhancement:** During B1 dev inference, log the empirical mean padded token length $\bar{L}_{\text{dev}}$ from the tokenizer outputs on `smishdetect_validation` (`708` rows) so the upper-bound multiplier `128 / mean_dev_tokens` is computed analytically from real dev telemetry, and also note that inference is `<10%` of total 3-epoch unit wall-time (so even a `2.0×` OOD inference variance shifts total 160-unit Option A runtime by `<5%`).
  - **Strict Three-Way Distinction in All B1 Reports:**
    1. *Measured Dev Inference* (`val` + `val_adv` + `smishdetect_validation` + `cf`),
    2. *Analytical OOD Inference Estimate* (`analytical_estimate_only: true`, never reported as a measured OOD benchmark),
    3. *Future Live OOD Evaluation* (deferred strictly to authorized Option A `experiments_abcd`).

---

## 10. Statistical Governance Assessment (`P0-4`)

- **Observed Failure (`reports/stage4_statistical_validation_results.json:324-348,663-667`):**
  - `syn_stress::S1_both_zero` (`N_outer = 1,000`, `B_pval = 50,000`, `B_ci = 2,000`):
    - Macro-F1 Type-I error: `51/1,000 = 0.0510` (Wilson 95% CI `[0.0390, 0.0664]`) — **PASS** (`<= 0.060`).
    - **Recall Type-I error: `62/1,000 = 0.0620`** (`MC SE = 0.007626`, Wilson 95% CI `[0.048664, 0.078688]`) — **FAIL** (`> 0.0600`).
    - Joint IUT Type-I error: `9/1,000 = 0.0090` — **PASS**.
    - `syn_stress_holm12_correlated` Joint FWER: `51/1,000 = 0.0510`, Endpoint FWER: `45/1,000 = 0.0450` — **PASS**.
- **Underlying Metric & Denominator Inspection (`post_stage_b_stats.py:115-120,149-164,189-203`):**
  - The Recall delta statistic is $\hat{\Delta}_{\text{rec}} = \frac{\sum_{g=1}^G d_g}{\sum_{g=1}^G n_{2,g}}$, where $d_g = \sum_{i \in g, y_i = 2} (h_{M,i} - h_{A,i})$ and $n_{2,g} = \sum_{i \in g} \mathbb{I}(y_i = 2)$ (`smish_idx = 2`).
  - In `syn_stress` (`validate_stage4_fixed_bootstrap.py:534-543`), `G = 1,299` clusters have class prior `pi_s = [0.35, 0.25, 0.40]` (~`520` Smishing clusters, while the remaining ~`779` clusters contribute $(0, 0)$ to `(d_g, n_{2,g})`), heavy-tail cluster sizes $m_g \in \{1, 2, 4, 8\}$ (`p = [0.75, 0.14, 0.07, 0.04]`, `max/min = 8×`), and strong asymmetric intra-cluster correlation (`rho_a = 0.20`, `rho_m = 0.65`, `rho_am_fam = 0.40`).
  - Because `R_raw` is a **cluster-ratio estimator** restricted to the ~40% Smishing subset with heavy-tail cluster weights $\{1, 2, 4, 8\}$, the effective number of Smishing clusters $G_{\text{eff}}$ is substantially smaller than $520$, making the studentized cluster-bootstrap t-statistic $\frac{\hat{\Delta}^*_b - \hat{\Delta}_{\text{obs}}}{\widehat{\text{SE}}^*_b}$ potentially slightly anti-conservative ($\sim 0.055\text{–}0.062$) OR reflecting finite-$N_{\text{outer}}=1{,}000$ Monte Carlo sampling variance ($P(K \ge 62 \mid p=0.05) = 0.0511$; $P(K \ge 62 \mid p=0.06) = 0.4135$).
- **Prespecified Protocol Check:** `N_OUTER_BOUNDARY = 1,000` and `MAX_EMPIRICAL_TYPE1 = 0.060` were fixed in code prior to execution (`validate_stage4_fixed_bootstrap.py:64-75`) and `CONFIRMATORY_DECLARATIONS_ENABLED` switched fail-closed to `False` automatically without post-hoc simulation extension or threshold editing.
- **Governance Decision for Stage 4 v9:**
  1. **Keep `CONFIRMATORY_DECLARATIONS_ENABLED = False`** in `post_stage_b_stats.py:38` and `STATISTICAL_VALIDATION_STATUS = "FAIL_OR_INCONCLUSIVE_CONFIRMATORY_DISABLED"`.
  2. **Do not** relax `MAX_EMPIRICAL_TYPE1`, do not remove `syn_stress`, and do not make confirmatory superiority declarations from Option A.
  3. Frame all Option A results strictly as **exploratory / descriptive**: point estimates, influence-function SEs, `B_ci = 2,000` family-clustered 95% CIs, and nominal p-values tagged `"p_value_status": "UNVALIDATED_DIAGNOSTIC_ONLY"`.
  4. **Recommended Additional Diagnostics (Zero Threshold Mutation):**
     - Report exact `k_recall / N`, `actual_fpr` on `smishdetect_validation` (`265` Normal rows, Clopper-Pearson 95% UCB disclosure), `ROC-AUC`, `PR-AUC`, `Macro-F1`, and confusion matrices on `primary_family_decontam_1687` & `sensitivity_template_stress_1675`.
     - **Does this block the next engineering/GPU milestones (`gpu_smoke_test`, `t4_benchmark`, or descriptive `experiments_abcd`)?**
       - It does **NOT** block `gpu_smoke_test` or `t4_benchmark` (which test engineering and resource calibration).
       - It **DOES** permanently block **confirmatory superiority claims** until/unless a separate **pre-registered validation study** ($N_{\text{outer}} \ge 5{,}000$, `MC SE <= 0.0031`, testing both `S1_both_zero` and whether exact `n_normal = 265` validation-threshold estimation variance—not modeled in the conditional `eval_smishdetect_validation` simulation—causes additional Type-I inflation) is authorized and passes without post-hoc tuning. Even then, remember `INFERENCE_CONDITIONING_STATEMENT` (`post_stage_b_stats.py:41-44`): single-seed (`seed=42`) comparisons are conditional on the trained models and frozen threshold, so multi-seed replication ($\ge 3$ seeds on top arms) is scientifically required for generalizable claims.

---

## 11. Regression Test Matrix (`tests/test_stage4_v9_remediation.py` + Existing Suites)

| # | Test Name | Target Module / Function | Expected Result | Scope | Mocks? | Evidence Produced |
|---|---|---|---|---|---|---|
| 1 | `test_smoke_mode_rejects_heldout_eval_config` | `post_stage_b_runner.PostStageBConfig` | `ValueError` when `run_mode="gpu_smoke_test"` and `run_heldout_eval=True` | Unit | No | Fail-closed config rejection proof |
| 2 | `test_smoke_execution_never_opens_test_or_ood_parquets` | `run_post_stage_b_pipeline`, `_load_configured_eval_sets`, `aggregate_post_stage_b_results` | `0` calls opening/reading `eval_smishdetect_test.parquet` or `eval_banglaphish_ood.parquet` in Cell 6 runner/aggregator | Integration | Mock `train_eval_fn` + sentinel `pd.read_parquet` / `Path.open` | Recorded list of opened paths (`0` test/OOD hits) |
| 3 | `test_smoke_exports_contain_zero_test_or_ood_metrics_or_preds` | `_prediction_frames`, `_write_prediction_archive`, `_compute_unit_metrics`, `_write_summary` | `preds.npz` has no `smishdetect_test_*`/`banglaphish_ood_*`; `metrics.json` & `aggregated_results.{json,csv}` have `None`/absent test/OOD fields | Integration | Mock `train_eval_fn` | Inspected `preds.npz` keys, `metrics.json`, `aggregated_results.json` |
| 4 | `test_unknown_or_forbidden_eval_datasets_rejected_fail_closed` | `hf_train_and_predict_fold`, `_assert_eval_sets_allowed`, `_prediction_frames`, `_compute_unit_metrics` | `RuntimeError` if `smishdetect_test`, `banglaphish_ood`, or unknown key passed when `run_heldout_eval=False` | Unit | Mock HF modules | Defense-in-depth rejection at every layer |
| 5 | `test_smoke_requires_pinned_tokenizer_revisions_for_real_hf` | `_validate_pre_run_compatibility`, `hf_train_and_predict_fold`, notebook Cell 5 | `RuntimeError`/`ValueError` when `run_mode="gpu_smoke_test"` lacks `tokenizer_revisions` for all `cfg.models` | Unit | Mock HF + `env_fp` | Pre-lease rejection before `mkdir`/training |
| 6 | `test_banglabert_normalizer_verification_and_exact_git_commit` | `post_stage_b_colab_deps.check_requirement_status`, `prepare_tokenizer_texts`, `collect_stable_environment_fingerprint` | Accepts `d405944d...`; rejects wrong commit, wrong `direct_url`, or missing `normalizer.normalize` | Unit | Temp `.dist-info` (`PathDistribution`) + mock `normalizer` | Real `importlib.metadata` check output |
| 7 | `test_32_pair_tokenizer_collision_gate_views_and_count` | `post_stage_b_tokenizer_gate.run_allowed_view_tokenizer_collision_gate` | Executes exact `32` pairs (`P01`–`P32`) across configured models × allowed views; `0` reads of `eval_banglaphish_ood.parquet`; rejects incomplete pairs or `collisions > 0` | Integration | Mock tokenizer + sentinel `pd.read_parquet` | `tokenizer_collision_gate.json` (`n_pairs=32`, exact `total_checks`) |
| 8 | `test_smoke_exercises_weighted_trainer_and_standard_trainer` | `run_post_stage_b_pipeline`, `hf_train_and_predict_fold` | Smoke config runs `B` (`WeightedTrainer`) and `D` (`Trainer`) on `canon_masked` with `csebuetnlp/banglabert` | Integration | Mock HF `Trainer` | Verified `class_weights` passed to `B` and `None` to `D` |
| 9 | `test_t4_b1_v2_no_synthetic_text_no_test_ood_access` | `run_post_stage_b_pipeline`, `_build_t4_benchmark_workload_fixture` | `v2` config passes pre-lease on mock T4; `_build_t4_benchmark_workload_fixture` never called; `0` test/OOD reads | Integration | Mock T4 HW + mock `train_eval_fn` + sentinel `read_parquet` | Spy call counts (`fixture=0`, `test/ood=0`) |
| 10 | `test_t4_b1_v2_measures_smishdetect_validation_and_emits_analytical_ood` | `_load_configured_eval_sets`, `_prediction_frames`, `_aggregate_verified` | Loads `smishdetect_validation` (`708` rows); computes dev throughput; writes `t4_benchmark_calibration_report.json` with `analytical_estimate_only=True` | Integration | Mock T4 HW + mock `train_eval_fn` | Verified `smishdetect_validation` `n_samples=708` + calibration JSON |
| 11 | `test_frozen_configs_baseline_masks_and_historical_kits_unchanged` | `verify_all_stage4_integrity`, `verify_portable_colab_kit`, SHA256 checks | Frozen v1 T4 config, Option A config, `282/282` Category A, `4/4` masks, `v3`–`v8` kits byte-identical | Integration | No (real files) | SHA256 digest comparisons & integrity JSON |
| 12 | `test_pyproject_gpu_dependency_parity_with_colab_deps` | `pyproject.toml`, `post_stage_b_colab_deps.py`, `post_stage_b_runner.py` | `[gpu]` covers all 24 distributions (zero `unicodedata2`) and exact `normalizer@d405944d...` | Unit | No (`tomllib`) | Exact set equality check |
| 13 | `test_portable_v9_kit_export_and_manifest_verification` | `prepare_stage4_colab_kit.export_stage4_colab_kit`, `verify_portable_colab_kit` | Exports isolated `v9` kit, verifies `35/35` files (with `v2` T4 config), `4/4` masks, and tamper rejection | Integration | No (temp dir) | `stage4_kit_manifest.json` verification output |
| 14 | `test_live_colab_smoke_validation` *(Manual / Live Gate)* | Colab Notebook Cells 1–7 (`RUN_MODE='gpu_smoke_test'`) | Real CUDA `torch>=2.6` preserved; `pip` installs `normalizer@d405944d...`; Cell 5 32-pair gate passes on real HF `4.57.1`; Step 2 snapshot provenance logged; Drive staging + resume succeed; `0` test/OOD metrics | **Live GPU** | **NO (Real Colab/HF/Drive)** | Live `run_state.json`, `tokenizer_collision_gate.json`, `aggregated_results.json`, `runner.log` |

**Explicit Boundary:** Tests 1–13 run locally with CPU mocks for `torch`/`transformers`/`Trainer`/T4 hardware and prove **software contract & data-isolation invariants only**. Only Test 14 proves live Colab CUDA, `transformers==4.57.1`, `normalizer`, and Drive FUSE compatibility.

---

## 12. Research Integrity Safeguards

To prove historical artifacts remain byte-identical before and after v9 changes:
1. Capture pre-edit SHA256 digests of:
   - `experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json` (`03a5f10d433f657af82d1af401774fd40d9d36209d35e13a14bc934086416180`)
   - `experiments/colab/configs/post_stage_b_stage4_option_a_config.json` (`518ec0cf56d52619fe5f24da7ef18a4b077466d38058c4d2538cd605c6910ecc`)
   - `reports/stage4_statistical_validation_results.json` (`bde8085b77066cbf449a1b17d0a1afa77f2dcbcb4ea1a82f7d0dde68f6fe504f`)
   - `response.md` (at `HEAD 1e00018e...`)
   - All `stage4_kit_manifest.json` digests for `v3`–`v8` (`b27dfbd5...`, `59bc766e...`, `db7e7eeac...`, `528f0d9ab...`, `38d6a9ef...`, `9933a190...`).
2. After v9 edits, run:
   - `.venv/bin/python -B scripts/verify_stage4_integrity.py --verify-only --skip-tfidf` → must pass `282/282` Category A, `13/13` protected inputs, `v1`/`v2` bundles, `4/4` masks.
   - `verify_portable_colab_kit` on each of `v3`–`v8` → all `6/6` must pass with unchanged `manifest_sha256`.
   - `git diff` inspection confirming zero modifications to `data/`, `reports/protected_baseline_282.json`, `reports/stage4_frozen_test_masks.json`, `reports/stage4_statistical_validation_results.json`, `post_stage_b_stage4_option_a_config.json`, `post_stage_b_stage4_t4_benchmark_config.json`, `response.md`, and `v3`–`v8`.

---

## 13. Portable Kit Release Process (Stage 4 v9)

Execute only **after** Gate A owner authorization:
1. Update `pyproject.toml` `[gpu]` dependencies.
2. Create `experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config_v2.json` and remediate `gpu_smoke_test_config.json`.
3. Implement code changes in `post_stage_b_runner.py`, `post_stage_b_tokenization.py`, `post_stage_b_tokenizer_gate.py`, and `post_stage_b_integrity.py`.
4. Update `scripts/generate_post_stage_b_stage4_notebook.py` (`v9`, `t4_benchmark_v2`, mandatory Cell 5 tokenizer gate across all modes).
5. Regenerate `experiments/colab/post_stage_b_stage4_experiments.ipynb` via `scripts/generate_post_stage_b_stage4_notebook.py` and verify parity.
6. Implement `tests/test_stage4_v9_remediation.py` and align impacted existing tests.
7. Run full regression suite under CPU cap (`scripts/run_stage4_capped.py -- .venv/bin/python -B -m unittest discover -s tests -p "test_*.py" -v` and `tests/test_pipeline.py`).
8. Run `scripts/verify_stage4_integrity.py --verify-only` (including `--skip-tfidf` and full check) + verify historical kits `v3`–`v8`.
9. Stage `stage4_colab_kit_upload/bangla_smish_post_stage_b_stage4_v9` via `scripts/prepare_stage4_colab_kit.py`.
10. Verify `v9` kit standalone via `scripts/prepare_stage4_colab_kit.py --verify-only` and check 100% repo-to-kit SHA256 parity across all staged files (`35/35` expected).
11. Write v9 remediation, test, dependency, integrity, and readiness reports under `reports/`.
12. Conduct an independent read-only acceptance review of `v9` before requesting Gate B live GPU smoke authorization.

---

## 14. Acceptance Criteria

- **For v9 Local Engineering Acceptance (Pre-Gate B):**
  - `100%` of `unittest` suite (all existing `112` + new `v9` tests) passes under `run_stage4_capped.py`.
  - `282/282` Category A, `13/13` protected inputs, `4/4` masks, `6/6` historical kits (`v3`–`v8`), frozen v1 T4 & Option A configs unchanged.
  - `v9` portable kit passes `verify_portable_colab_kit` with `35/35` exact repo-to-kit SHA256 parity.
  - Notebook generator matches `post_stage_b_stage4_experiments.ipynb` byte-for-byte.
  - `CONFIRMATORY_DECLARATIONS_ENABLED` remains `False`.
- **For Live GPU Smoke Acceptance (Gate B Outcome -> Pre-Gate C Benchmark):**
  - Clean Cell 4 install preserving CUDA `torch >= 2.6` and `normalizer@d405944d...`.
  - Clean Cell 5 32-pair tokenizer collision gate (`0` collisions) + `Stage 4 v9` kit verification on Drive.
  - Clean completion of all smoke units with `verification_source == "snapshot_path"` (or documented HF cache provenance) for all target models, `WeightedTrainer` + `Trainer` exercised, FP16 `scaler.pt` verified, `0` test/OOD rows in `preds.npz`/`metrics.json`/`exports/`.

---

## 15. Estimated Implementation Complexity

- **Scope:** Moderate-low, well-localized across 4 `src/` modules, 1 notebook generator + regenerated `.ipynb`, 2 configs (1 modified, 1 new), `pyproject.toml`, and 1 new test file (+ minor updates to 3 existing test assertions).
- **Risk level:** Low locally (everything testable via existing mock harness), **Moderate** at the live Colab/HF `4.57.1` boundary (since `candidate_paths` behavior on all 4 backbones and `pip` resolution can only be observed in Colab).

---

## 16. Remaining Risks

1. **Live Colab `transformers==4.57.1` Fast-Tokenizer `init_kwargs` Behavior:** Until live smoke runs, we do not know whether all 4 backbones populate `_TOKENIZER_PATH_KEYS` (Step 2) or fall back to Step 4 cache lookup.
2. **Drive FUSE Eventual Consistency & Trash Quota:** Repeated interrupted uploads or pruned checkpoints in Drive Trash remain unmeasured until Colab runs.
3. **OOD Sequence-Length Multiplier Uncertainty (`1.5×–2.0×`):** Analytical estimate without touching OOD cannot account for unexpected subword fragmentation spikes (`banglaphish_ood` Unicode/emoji mix).
4. **Scientific Confounders Remaining for Option A (Non-engineering):**
   - `n_normal = 265` threshold calibration variance (95% Clopper-Pearson UCB `1.12%`–`2.36%`).
   - Equal-epoch step imbalance (`300 / 300 / 333 / 363`).
   - Residual `8` CV↔CV and `6` train↔val template pairs.
   - `0.9560` TF-IDF synthetic detectability on `B1–B7` and `0/8` diversity gates on `pilot_300`.
   - `CONFIRMATORY_DECLARATIONS_ENABLED = False` (`syn_stress::S1_both_zero` `62/1,000 = 0.0620`).

---

## 17. Explicit Authorization Gates

- **Gate A:** Research Owner review & approval of this plan (plus resolution of Q-Align-1, Q-Align-2, Q-Align-3) before modifying any repo files or staging `v9`.
- **Gate B:** Independent read-only acceptance of staged `v9` kit before authorizing a live Colab `gpu_smoke_test` run (`RUN_MODE = 'gpu_smoke_test'`).
- **Gate C-Bench:** Clean live `v9` `gpu_smoke_test` pass before authorizing `AUTHORIZE_T4_BENCHMARK = True` (`5`-unit Option B1 T4 calibration).
- **Gate C-Full:** Empirical T4 benchmark telemetry + CU budget approval + explicit governance sign-off (`CONFIRMATORY_DECLARATIONS_ENABLED = False`) before any `AUTHORIZE_FULL_TRAINING = True` (`160`-unit Option A).

---

# Part II — Answers to the 12 Independent Technical Questions

1. **Is the existing `PostStageBConfig` the correct enforcement boundary, or should additional restrictions be implemented at the evaluation loader and export layers?**
   - **No, `PostStageBConfig` alone is NOT sufficient.** [`PostStageBConfig.__post_init__`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L300-L301) is the *first* fail-closed boundary (and currently misses `gpu_smoke_test`), while [`PostStageBConfig`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L185) even defaults `run_heldout_eval: bool = True`! Defense-in-depth requires enforcement across **all 5 downstream boundaries**:
     1. Config init (`PostStageBConfig.__post_init__`: reject `run_heldout_eval=True` for `gpu_smoke_test` and `t4_benchmark`),
     2. Eval loader ([`_load_configured_eval_sets()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L1491-L1500)),
     3. Trainer entry ([`hf_train_and_predict_fold()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L907-L912), which currently checks `forbidden` only if passed in `heldout_sets`),
     4. Prediction/metric/archive builders ([`_prediction_frames()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L1746-L1761), [`_compute_unit_metrics()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L1340-L1360), [`_write_prediction_archive()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L1764-L1783)),
     5. Aggregation/export ([`_aggregate_verified()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L2686)).

2. **Can any other code path indirectly load held-out/OOD data even when `run_heldout_eval=False`?**
   - **YES — three indirect paths exist in the current repository!**
     1. [`load_documented_32_pairs()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_tokenizer_gate.py#L89-L123) (called by Cell 5 `run_allowed_view_tokenizer_collision_gate`) reads **both** `eval_smishdetect_test.parquet` (because `18/32` pairs have a test `row_id`) **and** `eval_banglaphish_ood.parquet` (`0/32` pairs reside there — completely unnecessary read!).
     2. [`verify_portable_colab_kit()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_integrity.py#L604-L620) (called by Cell 5 on every run mode) invokes [`build_smishdetect_test_cohort_masks()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_cohorts.py#L160-L161), which calls `pd.read_parquet(bdir / "eval_smishdetect_test.parquet")` to re-derive and verify the 4 frozen masks.
     3. [`_run_locked()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L2106-L2107) calls [`audit_model_view_overlaps()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_model_view_audit.py#L30-L38) whenever `cfg.run_mode == "experiments_abcd"`, and `audit_model_view_overlaps()` unconditionally calls `load_heldout_eval_sets(bundle_dir)` even if someone constructed an `experiments_abcd` config with `run_heldout_eval=False`!

3. **Does the existing 32-pair tokenizer collision gate actually validate the intended tokenizer views and candidate pairs?**
   - **Yes, architecturally it does — EXCEPT it is bypassed in `gpu_smoke_test_config.json`.**
   - In [`post_stage_b_tokenizer_gate.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_tokenizer_gate.py#L26,L43-L76,L183-L246), `ALLOWED_COLLISION_GATE_VIEWS = ("raw", "canon_masked")` and `STAGE3_32_CANDIDATE_PAIRS` encodes the exact `32` pairs (`P01`–`P32`) matching [`post_stage_b_stage3_readiness_review.md`](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/post_stage_b_stage3_readiness_review.md#L38-L69). It applies `prepare_tokenizer_texts(..., model_name)` (invoking `normalizer.normalize()` for BanglaBERT) and compares `tok(..., truncation=True, max_length=int(cfg.max_length), add_special_tokens=True)["input_ids"]`.
   - **However:** In [`generate_post_stage_b_stage4_notebook.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/scripts/generate_post_stage_b_stage4_notebook.py#L214-L235), Cell 5 gates execution on `if cfg.tokenizer_revisions:` and exempts `gpu_smoke_test` (`L234`), and `_resolve_and_validate_tokenizer_gate()` (`post_stage_b_runner.py:1657`) sets `gate_required = bool(cfg.tokenizer_revisions) or cfg.confirmatory_mode or cfg.run_mode == "t4_benchmark"`. Because `gpu_smoke_test_config.json` omitted `tokenizer_revisions`, the 32-pair gate was completely skipped!

4. **Is pinning the HuggingFace model revision sufficient, or must tokenizer configuration and normalizer revisions be independently pinned?**
   - **Pinning the model revision alone is NOT sufficient; tokenizer and normalizer revisions must be independently pinned and verified.**
   - Reasons grounded in code:
     1. In [`hf_train_and_predict_fold()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L915-L926,L974-L990), `AutoTokenizer.from_pretrained(model_name, revision=pinned_rev)` and `AutoModelForSequenceClassification.from_pretrained(...)` are separate calls; `transformers==4.57.1` pops `_commit_hash` on tokenizers (`post_stage_b_tokenization.py:123-124`), so [`resolve_and_verify_tokenizer_revision()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_tokenization.py#L112-L220) must independently verify the tokenizer's resolved snapshot commit SHA.
     2. `csebuetnlp/banglabert` preprocessing relies on an external GitHub package (`csebuetnlp/normalizer`), NOT files inside the HF model repo. [`BANGLABERT_NORMALIZER_COMMIT`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_tokenization.py#L15) (`d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9`) is verified independently via `direct_url.json` in [`check_requirement_status()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_colab_deps.py#L131-L148) and [`_validate_required_packages_fingerprint()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L552-L556).
     3. Furthermore, `TOKENIZER_GATE_PACKAGE_DISTS` (`transformers`, `tokenizers`, `sentencepiece`, `huggingface_hub`) are bound into `tokenizer_collision_gate.json` and checked against live runtime versions (`post_stage_b_runner.py:1561-1578`) because Rust/SentencePiece library upgrades can alter tokenization even at the same HF commit.

5. **Can the existing Trainer implementations be exercised meaningfully with a minimal smoke workload without weakening the real training path?**
   - **Yes.** [`hf_train_and_predict_fold()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L861-L1312) uses the exact same `TrainingArguments`, `DataCollatorWithPadding`, `Trainer` / `_weighted_trainer_class` (`L710-L724,1137`), `Stage4CheckpointCallback` (`L1040-L1135`), `_resolve_precision("auto")` (`L759-L775`), `_checkpoint_file_hashes(..., require_scaler=use_fp16)` (`L652-L669`), `sync_local_checkpoint_to_staging`, and `recover_latest_valid_checkpoint` regardless of sample cap. With `smoke_max_train_samples=64`, `batch_size=16`, `epochs=1` (`4` steps) and `experiments=["B", "D"]`, **every** Trainer code path (`WeightedTrainer` + standard `Trainer`, FP16 scaler, epoch eval, epoch save, staging sync, `preds.npz` + `metrics.json` commit) runs identically to production, only on capped `train_df`/`val_df`/`cf_df` slices (`L1712-1713`) and `1` epoch.

6. **Is Option B1 scientifically and computationally valid with development-only measurements?**
   - **Yes for GPU/VRAM/training-time/Drive calibration; conditionally valid (analytical lower/upper bound only) for OOD inference latency.**
   - Computationally: In `t4_benchmark`, `3` epochs of training at `batch_size=32` (`363` steps for D, `300` steps for B) generate **100% of peak VRAM** (activations + gradients + FP16 master/AdamW optimizer states exceed `eval_batch_size=64` `torch.no_grad()` inference memory) and **>90% of unit wall-clock time**. Measuring `val` (`800`), `val_adv` (`800`), `smishdetect_validation` (`708`), and Fold-0 `cf` (`~255×3`) gives genuine unmodified dev GPU throughput without touching test/OOD data or generating synthetic text.
   - Scientifically: It is valid **only** if dev measurements and analytical OOD extrapolations are strictly separated in reporting, never conflated as an empirical OOD benchmark.

7. **Is the 1.5–2.0 OOD inference safety factor defensible, or should we use a different estimation methodology?**
   - **It is a reasonable heuristic upper-bound range for `max_length=128`, NOT an empirically calibrated constant—and can be strengthened without touching OOD data.**
   - Under [`hf_train_and_predict_fold()`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L957-L963,L1143), tokenization uses `truncation=True, max_length=128` (`cfg.max_length=128`) and `DataCollatorWithPadding` pads each batch of `64` to the longest sequence in that batch. Because `banglaphish_ood` averages `~165` chars vs `~80–84` chars in dev, dynamic padding makes dev batches shorter than OOD batches.
   - **Improved methodology (zero OOD access):** During B1 dev inference on `smishdetect_validation` (`708` rows), record the empirical mean padded batch token length $\bar{L}_{\text{dev}}$ (available from the tokenized dataset) and separate `tok_seconds` vs `predict_seconds`. Then compute the deterministic worst-case compute scaling factor $S_{\max} = 128 / \bar{L}_{\text{dev}}$ (which typically falls near `1.5×–2.0×`) alongside the `1.5×–2.0×` band, and explicitly disclose subword-length/CPU tokenization uncertainty.

8. **Does the current benchmark runner require architectural changes rather than a configuration-only fix?**
   - **Yes — architectural runner, integrity, and notebook changes are mandatory.**
   - Verified lines in [`post_stage_b_runner.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L1219,L1496,L1756): `_load_configured_eval_sets()`, `hf_train_and_predict_fold()`, and `_prediction_frames()` all gate `smishdetect_validation` behind `if cfg.run_mode == "t4_benchmark" and cfg.benchmark_workload_fixture_eval:`. Setting `benchmark_workload_fixture_eval: false` in a v2 JSON without changing those 3 sites silently skips `smishdetect_validation` (`708` rows). Additionally, [`post_stage_b_integrity.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_integrity.py#L388,L510,L626) and [`generate_post_stage_b_stage4_notebook.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/scripts/generate_post_stage_b_stage4_notebook.py#L159) hardcode the v1 benchmark config filename.

9. **Is the current statistical confirmatory restriction sufficient, or are additional safeguards needed?**
   - **Structurally sufficient to block premature confirmatory declarations, but two reporting/diagnostic safeguards should be noted:**
     1. [`CONFIRMATORY_DECLARATIONS_ENABLED = False`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_stats.py#L38) + `post_stage_b_ensemble.py:384` + `holm_reject_family:466` fail closed even if a caller passes `confirmatory_allowed=True`.
     2. **Additional safeguards needed:** (a) In `post_stage_b_runner.py:2664-2680`, `mcnemar_vs_exp_a` still emits Holm-adjusted p-values on OOF accuracy (`L2679`) without tagging them `"p_value_status": "UNVALIDATED_DIAGNOSTIC_ONLY"` inside each entry; adding that tag prevents misreading OOF McNemar as confirmatory. (b) Ensure Option A reports do not treat `confirmatory_mode: true` (`post_stage_b_stage4_option_a_config.json:36`) as authorizing claims; keep the explicit distinction (`confirmatory_mode` = 160-unit grid guard only). (c) Ensure `n_normal = 265` threshold uncertainty (Clopper-Pearson 95% UCB up to `2.36%` at `k=2`) is always co-reported with `ROC-AUC`, `PR-AUC`, and `Macro-F1`.

10. **Are there any additional P0 risks that the independent audit did not identify?**
    - **Not new live-GPU P0s beyond P0-1..P0-4, but three critical implementation/governance bugs discovered during this planning inspection that would have broken v9 if overlooked:**
      1. **`verify_portable_colab_kit()` & `load_documented_32_pairs()` indirect test/OOD reads (See Q2):** `load_documented_32_pairs()` unconditionally reads `eval_banglaphish_ood.parquet` (`0/32` pairs), and both `load_documented_32_pairs()` and `verify_portable_colab_kit()` read `eval_smishdetect_test.parquet` during Cell 5 even in `gpu_smoke_test` / `t4_benchmark`.
      2. **`audit_model_view_overlaps()` bypass in `experiments_abcd` (`post_stage_b_runner.py:2106-2107`):** Calls `load_heldout_eval_sets()` unconditionally regardless of `cfg.run_heldout_eval`.
      3. **`v8_readiness_report.md` row-count discrepancy (`757` vs `708`):** `reports/post_stage_b_stage4_v8_readiness_report.md:41,47` cites `757` rows for `eval_smishdetect_validation.parquet`, whereas `dataset_manifest.json:182` verifies `708` rows. Planning or asserting `757` in v9 code/tests would fail!
      4. **Test-fixture breakage risk on `PostStageBConfig.__post_init__`:** Requiring `tokenizer_revisions` for `gpu_smoke_test` inside `__post_init__` would break 10+ existing mock unit tests; enforcing it in `_validate_pre_run_compatibility()` (for `hf_train_and_predict_fold`) + Colab Cell 5 avoids false-positive breakage while remaining fail-closed for real HF runs.

11. **What is the smallest set of changes that would make v9 genuinely ready for an authorized live GPU smoke test?**
    - Minimal smoke-unblocking set:
      1. Remediate `gpu_smoke_test_config.json` (`run_heldout_eval: false`, `canon_masked`, pinned `tokenizer_revisions`, `csebuetnlp/banglabert` + `WeightedTrainer` `B` & `D`).
      2. Fail-closed `gpu_smoke_test` guard on `run_heldout_eval=True` in `PostStageBConfig.__post_init__` + execution-layer allowlist checks in `post_stage_b_runner.py` + `_validate_pre_run_compatibility` requiring `tokenizer_revisions` for real HF smoke runs.
      3. Remove `eval_banglaphish_ood.parquet` read from `load_documented_32_pairs()`.
      4. Add Option B1 `v2` T4 config + runner `smishdetect_validation` (`708`) dev-benchmark support + `post_stage_b_integrity.py` registration (so the v9 kit is complete and doesn't need an immediate v10 for T4 benchmarking).
      5. Update `pyproject.toml [gpu]`, regenerate `post_stage_b_stage4_experiments.ipynb` (requiring Cell 5 tokenizer gate in all modes), pass full regression tests (`test_stage4_v9_remediation.py`), and stage `v9` kit.

12. **What evidence would you require before recommending the 160-unit experiment?**
    1. **Clean v9 Local Acceptance:** 100% unit/integration tests passing under `run_stage4_capped.py`, `282/282` integrity, `v3`–`v8` untouched, `v9` kit `35/35` parity.
    2. **Clean Live Colab `gpu_smoke_test`:** Empirical proof of CUDA `torch >= 2.6` preservation, `normalizer@d405944d...` VCS metadata match, Cell 5 32-pair gate `0` collisions, `resolve_and_verify_tokenizer_revision()` Step 2 `snapshot_path` confirmation on target models, `WeightedTrainer` + `Trainer` + FP16 `scaler.pt` + Drive staging/resume working, and `0` test/OOD outputs.
    3. **Clean Live 5-Unit `t4_benchmark` (v2 Option B1):** Measured peak CUDA VRAM for `xlm-roberta-base` & `muril-base-cased` safely below T4 `15 GB` (`< 12 GB` target), acceptable FP16 `GradScaler` behavior (`actual_global_step == expected_optimizer_steps`), measured `drive_sync_seconds`, Drive Trash quota headroom for 160 units, and projected Colab Compute Unit (`CU`) cost within Owner budget for `51,840` total training steps (`160` units × avg `324` steps/unit vs `1,752` benchmark steps = `~29.6×` training step ratio).
    4. **Explicit Owner Scientific Governance Sign-off:** Written confirmation that Option A (`160` units) runs under `CONFIRMATORY_DECLARATIONS_ENABLED = False` (descriptive effects + 95% CIs + diagnostic p-values only) and acknowledging the documented confounders (equal-epoch `300/300/333/363` step differences, `n_normal = 265` threshold uncertainty, `0.9560` synthetic detectability, `8` CV + `6` train↔val template pairs).

---

# Part III — Three-Way Alignment Report

| Topic | ChatGPT's Position | Gemini's Independent Position | Evidence / Reasoning | Final Recommendation |
|---|---|---|---|---|
| **1. Held-out data protection** | **AGREEMENT / PARTIAL:** Enforce restrictions at config & execution layers, not JSON flag alone; test zero access via sentinel readers. | **PARTIAL AGREEMENT (with critical nuance):** Config + `_load_configured_eval_sets` + `hf_train_and_predict_fold` + `_prediction_frames` + `_compute_unit_metrics` + `_write_prediction_archive` + `audit_model_view_overlaps` must all enforce allowlists. **Crucial repo nuance:** Cell 5 (`load_documented_32_pairs` & `verify_portable_colab_kit`) currently reads `eval_smishdetect_test.parquet` (and `load_documented_32_pairs` needlessly reads `eval_banglaphish_ood.parquet`). | `post_stage_b_runner.py:300,907,1491,1746,2106`; `post_stage_b_tokenizer_gate.py:92-99`; `post_stage_b_integrity.py:604-609`. | Enforce strict zero test/OOD access across Cell 6 runner/metrics/exports (`run_heldout_eval=False`), drop OOD read from `load_documented_32_pairs()`, and resolve Q-Align-1 (Cell 5 32-pair vs 14-pair smoke gate) with Owner & ChatGPT. |
| **2. Smoke-test architecture** | Minimal representative workload exercising `csebuetnlp/banglabert`, `canon_masked`, `normalizer`, pinned revisions, 32-pair gate, and Trainers. | **PARTIAL AGREEMENT (Expanded Scope):** Adding `banglabert` + `canon_masked` is essential, AND changing `experiments` from `["A", "D"]` to `["B", "D"]` is necessary to exercise `WeightedTrainer`; also recommend all 4 pinned backbones (`Matrix M1`, 8 tiny units = 32 steps) so live HF `4.57.1` snapshot resolution is verified on all 4 models before `t4_benchmark`. | `gpu_smoke_test_config.json:4-24`; `post_stage_b_runner.py:710-724,1137`; `generate_post_stage_b_stage4_notebook.py:214-235`. | Adopt `["B", "D"]` × `["canon_masked"]` × `[0]` (`64/32/32` caps, `epochs=1`) with all 4 backbones (or 2 backbones per Q-Align-2), and require `tokenizer_revisions` in Cell 5 & `_validate_pre_run_compatibility` for real HF smoke runs. |
| **3. Tokenizer & normalizer verification** | Pin tokenizer & normalizer revisions; run 32-pair collision gate. | **AGREEMENT + ADDITIONAL FINDING:** Agree fully; additionally log Step 2 (`snapshot_path`) vs Step 4 (`hf_cache_lookup`) provenance in `resolve_and_verify_tokenizer_revision()` (addressing `MED-2`) and bind `tokenizer_gate` package versions. | `post_stage_b_tokenization.py:112-233`; `post_stage_b_colab_deps.py:131-148`; `response.md:157-163`. | Pin all 4 `tokenizer_revisions` + `normalizer@d405944d...`; add provenance source logging for live Colab verification. |
| **4. Dependency management** | Sync `pyproject.toml [gpu]` with `normalizer`, `ftfy`, `emoji`, `wcwidth`; preserve Colab CUDA PyTorch. | **AGREEMENT:** Exact 24-distribution alignment (`pyproject.toml [gpu]` matches `post_stage_b_colab_deps.py` & `post_stage_b_runner.py`, zero `unicodedata2`); pin `ftfy==6.0.3`, `emoji==1.4.2`, `wcwidth>=0.2.5`, `normalizer@d405944d...`. | `pyproject.toml:19-37`; `post_stage_b_colab_deps.py:92-109,202-206`; `post_stage_b_stage4_v8_dependency_audit.md:22-39`. | Sync `pyproject.toml [gpu]`, add automated 24-dist parity test, and preserve `constraints.txt` `torch=={torch_ver}` Colab install flow. |
| **5. T4 B1 benchmark** | Preserve v1 config; add `v2` config as genuine dev-data benchmark (not config-only edit). | **AGREEMENT + ADDITIONAL FINDING:** Runner code at `post_stage_b_runner.py:1219,1496,1756` currently hides `smishdetect_validation` behind `benchmark_workload_fixture_eval=True`, AND `v8_readiness_report.md:41` misstates `708` rows as `757`. Code changes in runner, integrity, and notebook `CFG_MAP` are mandatory. | `post_stage_b_runner.py:1219,1496,1756`; `dataset_manifest.json:182`; `post_stage_b_integrity.py:388,626`. | Preserve v1 untouched; create `t4_benchmark_config_v2.json`; modify runner to evaluate unmodified `val` (`800`), `val_adv` (`800`), `smishdetect_validation` (`708`), `cf` (`~255×3`). |
| **6. OOD inference estimation** | Do not treat `1.5×–2.0×` as an automatic constant or actual OOD benchmark; justify via `max_length`, batch size, padding, architecture, VRAM. | **AGREEMENT (Stronger Method):** `1.5×–2.0×` is a heuristic bound, not measured truth. Improve it cleanly (zero OOD reads) by logging mean dev padded batch length $\bar{L}_{\text{dev}}$ on `smishdetect_validation` so $128 / \bar{L}_{\text{dev}}$ bounds GPU compute scaling analytically, and always separate Measured Dev vs Analytical OOD Estimate vs Future OOD Eval. | `post_stage_b_runner.py:959,1143`; `response.md:230-231`. | Emit structured `t4_benchmark_calibration_report.json` separating measured dev throughput, $\bar{L}_{\text{dev}}$-grounded analytical OOD estimate (`1.5×–2.0×` / $128/\bar{L}_{\text{dev}}$), and explicit limitations. |
| **7. Statistical confirmatory declarations** | Keep `CONFIRMATORY_DECLARATIONS_ENABLED = False`; investigate `syn_stress::S1_both_zero` (`0.0620 > 0.0600`) without relaxing gates. | **AGREEMENT + ADDITIONAL FINDING:** Keep `False` in v9 and Option A; `S1_both_zero` Recall is a ratio estimator over ~520 heavy-tail (`1..8`) Smishing clusters (`MC SE=0.0076` at `N=1,000`). Also tag OOF `mcnemar_vs_exp_a` Holm p-values (`L2679`) as `"UNVALIDATED_DIAGNOSTIC_ONLY"` and document $N_{\text{outer}} \ge 5{,}000$ recalibration as a separate pre-registered gate (Q-Align-3). | `post_stage_b_stats.py:38,149-164`; `validate_stage4_fixed_bootstrap.py:534-543`; `stage4_statistical_validation_results.json:324-348`; `post_stage_b_runner.py:2679`. | Keep `CONFIRMATORY_DECLARATIONS_ENABLED = False`; tag all p-values diagnostic; defer any $N_{\text{outer}} \ge 5{,}000$ recalibration to a separate pre-registered track. |
| **8. Portable kit versioning** | Preserve v8 (and v3–v7) as immutable; release v9 independently verifiable. | **AGREEMENT:** Stage `bangla_smish_post_stage_b_stage4_v9`, preserve `v3`–`v8` untouched, update `post_stage_b_integrity.py` to support both v1 and v2 T4 configs. | `post_stage_b_integrity.py:385-646`; `prepare_stage4_colab_kit.py`. | Stage fresh `v9` kit (`35/35` files) and verify `v3`–`v8` SHA256s remain unchanged. |
| **9. GPU milestone acceptance** | Strictly distinguish CPU-mocked verification from real Colab/GPU validation. | **AGREEMENT:** Local `.venv` tests prove software/data-isolation contracts only; live Colab smoke must verify CUDA `torch`, `normalizer`, HF `4.57.1` snapshots, FP16 `GradScaler`, and Drive FUSE before T4 benchmark. | `response.md:56-68`; `tests/test_stage4_v8_remediation.py:1-10`. | Require Gate B (`gpu_smoke_test`) live evidence before Gate C-Bench (`t4_benchmark`). |
| **10. 160-unit experiment authorization** | Require independent acceptance + T4 telemetry before authorizing 160 units. | **AGREEMENT:** Blocked until Gate B + Gate C-Bench pass, CU budget (`~29.6×` step ratio vs benchmark) is approved, and Owner signs off on exploratory framing (`CONFIRMATORY_DECLARATIONS_ENABLED = False`) and documented confounders. | `post_stage_b_stage4_option_a_config.json`; `response.md:106-140,284,295`. | Keep `experiments_abcd` strictly **`NO-GO`** until all 4 prerequisite evidence gates are satisfied. |

---

# Part IV — Final Decision & Authorization Gates

### Gate A — v9 Implementation
- **Gemini Recommendation:** **CONDITIONALLY READY FOR OWNER AUTHORIZATION** once Q-Align-1 (Cell 5 32-pair vs 14-pair smoke access) and Q-Align-2 (4-backbone vs 2-backbone smoke matrix, plus in-place vs v2 smoke config) are resolved by the Owner and ChatGPT. No source/config edits have been made yet.

### Gate B — Live GPU Smoke Test
- **Required Evidence Before Authorization:**
  1. `100%` passing local regression suite (`112` existing + all new v9 tests in `test_stage4_v9_remediation.py`) under `run_stage4_capped.py`.
  2. Verified `0` reads/writes/metrics/exports of `smishdetect_test` / `banglaphish_ood` in Cell 6 sentinel tests.
  3. `282/282` Category A, `4/4` masks, `6/6` historical kits (`v3`–`v8`), frozen v1 T4 & Option A configs byte-identical.
  4. `v9` portable kit staged and verified (`35/35` repo-to-kit parity) + independent read-only review of `v9`.

### Gate C — Full 160-Unit Experiment (`experiments_abcd`)
- **Required Additional Evidence Before Recommending Execution:**
  1. **Live Gate B Pass:** Verified live Colab `gpu_smoke_test` logs showing CUDA `torch >= 2.6` preserved, `normalizer@d405944d...` exact commit, Cell 5 gate `0` collisions, Step 2 `snapshot_path` verification on target models, FP16 `scaler.pt` + Drive FUSE sync/resume working, `0` held-out outputs.
  2. **Live T4 Benchmark Pass (v2 Option B1):** Empirical Tesla T4 (`7.5`) telemetry for peak VRAM (`xlm-roberta-base`, `muril-base-cased`), `actual_global_step == expected_optimizer_steps`, FP16 stability, `drive_sync_seconds`, Drive Trash headroom, dev throughput + analytical OOD bound, and Colab `CU/hr` burn rate projecting 160-unit cost (`51,840` steps vs `1,752` benchmark steps).
  3. **Scientific & Governance Sign-off:** Explicit Owner authorization of exploratory framing (`CONFIRMATORY_DECLARATIONS_ENABLED = False`) and acceptance of documented Option A scope/confounders (`equal-epoch` steps `300/300/333/363`, `n_normal = 265` threshold uncertainty, `0.9560` TF-IDF detectability, `8` CV + `6` train↔val pairs).
