# Research Checkpoint — Bangla & Code-Mixed Smishing Detection

**Saved:** 2026-11-16  
**Status:** Paused after completing **Phase 1 (Dataset Verification)** and **Phase 2–3 (CPU Baselines & Shortcut Audit)**; `Google.colab` VS Code extension is installed and ready for GPU execution.

---

## 1. Completed Work & Verified Findings

### Environment & Workspace
* Python 3.12 virtual environment `.venv/` configured via `uv` (`pyproject.toml`).
* VS Code (`Google.colab` + `ms-toolsai.jupyter`) installed by user; **200 Google Colab compute units** available for Phases 2–5 GPU runs.
* Unit tests in `tests/test_pipeline.py` passed (`All tests passed!`).

### Phase 1 — Dataset Verification (`reports/phase1_dataset_audit.md`, `reports/phase1_audit.json`)
Fetched and audited 3 public 2025–2026 corpora alongside the repository's legacy 2024 `DataSet.csv`:
1. **`legacy_2024` (`DataSet.csv`, UNDOCUMENTED license):** 2,580 rows (2,083 unique), 19.26% exact duplicates, 1 conflicting label, 89.15% surface-only 5-fold accuracy (`Length-Only Risk AUC = 0.8738`). Retained as legacy reference only; excluded from primary training.
2. **`bangalabarta_2025` (Mendeley `10.17632/jfkfbw3gzh.3`, CC BY-NC-SA 4.0):** 2,772 rows (only **1,205 unique** — **56.53% exact duplicates**, 1 label conflict), 78.09% surface-only accuracy (`Length-Only AUC = 0.8453`).
3. **`smishdetect_2026` (HF `shariul-islam/bengali-sms-smishing-dataset`, MIT):** 7,005 rows (6,998 unique), 3-class (`Normal`, `Promotional`, `Smishing`) across `Bengali`, `Banglish`, `English`, `CodeMix`. **Contains 100.0% (1,205/1,205) of `bangalabarta_2025`'s unique texts.**
4. **`banglaphish_2026` (GitHub/Zenodo `10.5281/zenodo.20501760`, CC BY-NC 4.0):** 6,000 synthetic rows (5,416 unique, 20.81% near-dup reduction at `cos >= 0.85`), 95.44% surface-only accuracy (`Length-Only AUC = 0.9375`; Normal median 100 chars vs. Smishing median 216 chars; `[REDACTED_*]` tag shortcut).

### Phase 2–3 (CPU) — Fair Evaluation & Shortcut-Stress (`reports/phase2_3_cpu_baselines.md`, `reports/phase2_3_cpu_baselines.json`)
Built leakage-free `StratifiedGroupKFold` (`dup_cluster_id`, `cos >= 0.85`) datasets:
* `data/processed/unified_3class_dedup.parquet` (**6,997** unique non-conflicting rows, 6,493 clusters)
* `data/processed/banglaphish2026_dedup.parquet`

Key TF-IDF (word 1–2 + char_wb 3–5) + LogisticRegression results:
* **In-distribution is saturated:** `smishdetect_original_test_split` = **98.77% Macro-F1**; deduplicated 5-fold CV = **98.65% Macro-F1** [98.38, 98.92].
* **Shortcut masking & adversarial evasion hurt significantly:** `<URL>/<PHONE>/<USSD>` masking drops Macro-F1 to **98.03%** (`McNemar p = 0.0`); ZWJ/homoglyph perturbations drop Macro-F1 to **97.67%** (`McNemar p = 0.0`, Smishing Recall 98.75% → 96.79%).
* **Cross-script LODO collapse:** Train `BangalaBarta` (Bangla) → Test non-overlapping `SmishDetect` (multilingual/Banglish, `N=5,793`): Macro-F1 = **76.32%** (raw) / **67.61%** (masked), Recall @ ≤1% FPR = **63.34%** / **45.49%**.
* **Zero-shot OOD collapse:** Train unified 3-class → Test `BanglaPhish-2026` SMS (`N=1,540`): Macro-F1 = **53.12%**, Normal→Smish FPR = **24.36%**, Recall @ ≤1% FPR = **6.0%**.

---

## 2. Current File Inventory

* **Core package (`src/bangla_smish/`):** `schema.py`, `audit.py`, `splits.py`, `metrics.py`, `__init__.py`
* **Scripts (`scripts/`):** `phase1_fetch_and_audit.py`, `phase2_cpu_baselines.py`, `generate_colab_notebook.py`
* **Notebooks (`notebooks/`):** `colab_phase2_5_gpu_pipeline.ipynb`
* **Processed data (`data/processed/`):** `unified_3class_dedup.parquet`, `banglaphish2026_dedup.parquet`
* **Raw data (`data/raw/`, gitignored):** `bangalabarta/`, `smishdetect_hf/`, `banglaphish2026/`
* **Reports (`reports/`):** `phase1_dataset_audit.md`, `phase1_audit.json`, `phase2_3_cpu_baselines.md`, `phase2_3_cpu_baselines.json`
* **Tests:** `tests/test_pipeline.py`

---

## 3. Exact Resume Plan (Next Session)

1. **Choose Colab sync mode** (tell the assistant at start of next session):
   * **Option A:** Google Drive only (`MyDrive/ExploringAIs`)
   * **Option B:** GitHub repo (code) + Google Drive (outputs/checkpoints)
2. **Automate Colab runner (optional upgrade before running):**
   * Package Phases 2–5 GPU code into a resumable `scripts/run_colab_suite.py` and shrink `notebooks/colab_phase2_5_gpu_pipeline.ipynb` to a 3-cell launcher (Mount/Sync → Run `--resume` → Save outputs).
3. **Run Colab GPU Pipeline (~35–50 / 200 compute units):**
   * Connect VS Code notebook kernel via **Select Kernel → Colab → Auto Connect** (T4 or L4 GPU).
   * Run Fold-0 pilot first (~5–10 min), then full 5-fold CV (`BanglaBERT`, `MuRIL`, `XLM-R`, `Distil-mBERT`, `MobileBERT`) + LODO OOD on `BanglaPhish-2026` SMS + ONNX INT8 latency/RAM benchmark (Phase 4) + 4-bit selective SLM explanation samples (Phase 5).
4. **Local Post-Processing & Paper Writing (Phase 5):**
   * Sync `reports_gpu/` back to workspace → run McNemar & bootstrap CIs, ablations, error analysis, Android profiling, and draft the paper.
