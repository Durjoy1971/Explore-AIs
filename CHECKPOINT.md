# Research Checkpoint — Bangla & Code-Mixed Smishing Detection

**Saved:** 2026-11-17 (19:45 +06:00)  
**Status:** Paused after completing **Stage A (`curation_v1` & `curation_v2`)**, **Stage B 300-row Pilot + Independent Review**, and **Pipeline Hardening (Dynamic Brand Masking, Currency-vs-Shortcode Fix, CodeMix/Entity Diversity Gates, Class-Matched & Stratum-Controlled Synthetic-Origin Diagnostics, Read-Only Dry-Run)**. **Stopped before generating the remaining `345` `Promotional` samples**, awaiting explicit user approval.

---

## 1. Completed Work & Verified Findings

### Environment & Workspace
* Python 3.12 virtual environment `.venv/` configured via `uv` (`pyproject.toml`).
* VS Code (`Google.colab` + `ms-toolsai.jupyter`) installed by user; **200 Google Colab compute units** available for Phases 2–5 GPU runs.
* Unit & regression suite (`.venv/bin/python -m unittest discover -s tests -v`): **26/26 passed** (`67.641s`); `.venv/bin/python tests/test_pipeline.py`: **All tests passed!**.

### Phase 1 & Phase 2–3 (CPU) — Baseline Audit
* **Phase 1 (`reports/phase1_dataset_audit.md`, `reports/phase1_audit.json`):** Audited `legacy_2024` (`DataSet.csv`), `bangalabarta_2025` (1,205 unique; 100% overlap in `smishdetect_2026`), `smishdetect_2026` (6,998 unique), and `banglaphish_2026` (5,416 unique).
* **Phase 2–3 CPU Baselines (`reports/phase2_3_cpu_baselines.md`, `reports/phase2_3_cpu_baselines.json`):** `StratifiedGroupKFold` (`cos >= 0.85`), shortcut masking/ZWJ drops, cross-script LODO collapse, and zero-shot OOD collapse (`BanglaPhish-2026`).

### Stage A — Consolidated Curated Corpora (`curation_v1` & `curation_v2`, Untouched)
* **`data/processed/curation_v1/` (`reports/dataset_consolidation_v1.md`):** Initial Stage A consolidation (`existing_unique=11,947`, `eligible_train=4,125`, `quarantined=968`, `provenance=18,357`).
* **`data/processed/curation_v2/` (`reports/stage_a_focused_audit_v2.md`, `reports/stage_b_coverage_and_pilot_plan.md`):** Focused audit reclassified mislabeled `Banglish`/`English`/`CodeMix` rows and quarantined ambiguous items (`existing_unique=11,821`, `quarantined=1,094`, `provenance=18,357`, `audit_decisions=398`).
  * **`eligible_train.parquet` (`N = 3,999`, `3,803` families):** `Normal = 1,509`, `Promotional = 942`, `Smishing = 1,548` (strictly isolated from `smishdetect_validation=708`, `smishdetect_test=1,698`, `banglaphish_ood=5,416`).
  * **Stage B Quotas (`T = 1,548`):** `Normal = +39`, `Promotional = +606`, `Smishing = 0` (`Total = 645`).

### Stage B — 300-Row Pilot & Pipeline Hardening (`curation_v2`)
* **300-Row Pilot Artifacts (`data/generated/curation_v2/`, `reports/stage_b_pilot_v2.md` — Preserved Untouched):**
  * `candidates.jsonl` (`312` rows) → **`accepted.jsonl` (`300` rows, `96.15%`, all independently reviewed)** + `rejected.jsonl` (`12` rows flagged for naturalness/grammar/typo/lexical issues across `review_shard_1.jsonl`, `review_shard_2.jsonl`, `review_replacements.jsonl`).
  * Exact pilot counts: `Normal = 39` (`Bengali 10, Banglish 8, English 7, CodeMix 14`), `Promotional = 261` (`Bengali 65, Banglish 115, English 21, CodeMix 60`), `Smishing = 0`.
  * **Remaining deficit:** `Promotional = 345` (`Normal = 0`, `Smishing = 0`) — **NOT generated yet**.
* **Pipeline Hardening Implemented & Verified (`--dry-run` on existing 300 accepted rows):**
  1. **Regex & Currency Fixes (`src/bangla_smish/schema.py`, `src/bangla_smish/stage_b_pilot.py`):** `_URL_RE` catches bare/subdomains (`t.me`, `wa.me`); `_PHONE_RE` catches hyphenated `01700-0000XX`, spaced/hyphenated `+880`/`+৮৮০`, Bengali digits, and `16xxx`/`১৬xxx` shortcodes while excluding prefix/suffix currency amounts (`BDT 16000`, `16000 taka`, `১৬০০০ টাকা`, `16000/-`).
  2. **Dynamic Brand Masking (`codemix_diversity.py`, `schema.py`, `stage_b_pilot.py`, `validate_stage_b_pilot.py`):** `collect_brand_aliases` + metadata extraction (`brand`, `entity_name`, `brand_names`, `brand_aliases`, `entities`) + `--brand-registry <json>` + dynamic fallback (`extract_dynamic_brands`) + inflection/spacing matching across `raw`, `canon_masked`, `brand_masked` (`<BRAND>`), and `brand_stripped` (`""`).
  3. **CodeMix & Entity Diversity Gates (`src/bangla_smish/codemix_diversity.py`):** `mode="audit"` vs `mode="new_batch"` (`hook_plus_pure_en == 0.0`, `bn_script_en >= 0.80`, `intra_sentential_mix >= 0.60`, `sent1_excl_rate <= 0.35`, `sent1_quest_rate <= 0.05`, `max_brand_count <= 5`, `brandless_rate >= 0.30`, `ussd_1234_share <= 0.25`).
  4. **Extended Diagnostics (`stage_b_pilot.py`, `validate_stage_b_pilot.py`):** 4-view 3-class TF-IDF+LR probes, counterfactual URL/phone/brand swap probes (`-10.40 pp` / `-12.82 pp` raw flip → `0.00 pp` masked), and class-matched + stratum-controlled `(language, domain, length_band)` `is_synthetic` probes (`Normal vs Normal` & `Promotional vs Promotional`).

---

## 2. Current File Inventory

* **Core package (`src/bangla_smish/`):** `__init__.py`, `schema.py`, `audit.py`, `splits.py`, `metrics.py`, `curation.py`, `curation_v2.py`, `stage_b_pilot.py`, `codemix_diversity.py`
* **Scripts (`scripts/`):** `phase1_fetch_and_audit.py`, `phase2_cpu_baselines.py`, `generate_colab_notebook.py`, `curate_stage_a.py`, `curate_stage_a_v2.py`, `validate_stage_b_pilot.py`
* **Tests (`tests/`):** `test_pipeline.py`, `test_curation.py`, `test_curation_v2.py`, `test_codemix_diversity.py`, `test_schema_masking.py`, `test_stage_b_pilot.py`
* **Data (`data/`):**
  * `data/raw/` (`bangalabarta/`, `smishdetect_hf/`, `banglaphish2026/`, gitignored) + `DataSet.csv`
  * `data/processed/` (`unified_3class_dedup.parquet`, `banglaphish2026_dedup.parquet`, `curation_v1/`, `curation_v2/`)
  * `data/generated/curation_v2/` (`raw_batch_a.jsonl`, `raw_batch_b.jsonl`, `raw_batch_c_replacements.jsonl`, `candidates.jsonl`, `accepted.jsonl`, `rejected.jsonl`, `review_shard_1.jsonl`, `review_shard_2.jsonl`, `review_replacements.jsonl`, `pilot_validation_summary.json`)
* **Reports (`reports/`):** `phase1_*`, `phase2_3_*`, `dataset_consolidation_v1.md`, `stage_a_focused_audit_v2.md`, `stage_b_coverage_and_pilot_plan.md`, `stage_b_pilot_v2.md`
* **Notebooks (`notebooks/`):** `colab_phase2_5_gpu_pipeline.ipynb`

---

## 3. Exact Resume Plan (From Any Laptop)

1. **Sync & Verify Environment:**
   * Pull repo (`git pull`) and verify `.venv` (`uv sync` if needed; check `data/raw/` is present if re-running Stage A).
   * Run test suite: `.venv/bin/python -m unittest discover -s tests -v && .venv/bin/python tests/test_pipeline.py`.
   * Optional read-only dry-run check on current 300-row pilot (writes only to a non-production `--out-dir`):
     `.venv/bin/python scripts/validate_stage_b_pilot.py --dry-run --out-dir /tmp/stage_b_dry_run`
2. **Next Authorized Task — Generate Remaining `345` `Promotional` Rows (ONLY after explicit user approval):**
   * Derive strictly from `data/processed/curation_v2/eligible_train.parquet` (`0` held-out/OOD content inspected).
   * Target language mix for the `345` Promotional rows (check `reports/stage_b_coverage_and_pilot_plan.md` & `codemix_diversity.py` targets: `Banglish +131`, `Bengali +93`, `CodeMix +93`, `English +28` for equal `387/lang`, or confirm allocation with user).
   * Enforce `--diversity-mode new_batch`: `0%` `[hook !/?] + [pure EN]` on CodeMix, `>=80%` Bengali-script CodeMix, `>=60%` intra-sentential lexical mixing, `>=30%` brandless promos, `max_brand_count <= 5` with brand/alias metadata (`brand_names`/`brand_aliases`) + JSON registry, non-`*1234#` USSD codes, `sent1!` `<=35%`, `sent1?` `<=5%`.
   * Run mandatory row-by-row independent review + replacement of flagged rows, then run full validation (`validate_stage_b_pilot.py`) and compare class-matched `is_synthetic` AUC reductions (`raw`, `canon_masked`, `brand_masked`, `brand_stripped`).
3. **Post-Stage B — Baseline vs. Balanced Evaluation & Colab GPU Pipeline (Phases 2–5):**
   * Run fold-safe 5-fold CV + `smishdetect_test` + `BanglaPhish-2026` OOD comparisons (`unchanged baseline` vs `balanced original-only` vs `original + Stage B synthetic`) across raw, canonical-masked, brand-masked, and brand-stripped views.
   * Proceed to Colab GPU suite (`notebooks/colab_phase2_5_gpu_pipeline.ipynb`: `BanglaBERT`, `MuRIL`, `XLM-R`, `Distil-mBERT`, `MobileBERT`) and paper writeup.
