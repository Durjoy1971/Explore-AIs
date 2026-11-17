# Shortcut-Resistant, On-Device Bangla & Code-Mixed Smishing Detection

Research workspace modernized from the legacy 2024 prototype ([documents.md](documents.md), [DataSet.csv](DataSet.csv)) to investigate **H1 (Shortcut & Leakage Hypothesis)** and **H2 (On-Device Privacy-Preserving Detection + Selective SLM Explanations)** against 2025–2026 baselines.

## Repository Layout

- `DataSet.csv`, `documents.md`: Original 2024 artifacts (preserved unmodified; audited in Phase 1).
- `src/bangla_smish/`:
  - `schema.py`: Unified 3-class (`Normal`, `Promotional`, `Smishing`) loaders, normalization, surface-feature extraction, and `<URL>/<PHONE>/<USSD>` shortcut masking.
  - `audit.py`: Duplicate, predefined split-leakage, cross-dataset overlap, near-duplicate (`cos >= 0.85`), and surface-only shortcut diagnostics.
  - `splits.py`: Leakage-free `StratifiedGroupKFold` cluster splits and adversarial ZWJ/homoglyph & length-matched stress sets.
  - `metrics.py`: Macro-F1 (95% bootstrap CI), Smishing Recall @ <=1% Normal-as-Smishing FPR, ECE, and McNemar significance tests.
- `scripts/`:
  - `phase1_fetch_and_audit.py`: Produces `reports/phase1_dataset_audit.md` and `reports/phase1_audit.json`.
  - `phase2_cpu_baselines.py`: Builds deduplicated `data/processed/*.parquet` and runs Phase 2–3 CPU baselines + shortcut-stress evaluations into `reports/phase2_3_cpu_baselines.md`.
  - `generate_colab_notebook.py`: Generates `notebooks/colab_phase2_5_gpu_pipeline.ipynb`.
- `notebooks/colab_phase2_5_gpu_pipeline.ipynb`: Google Colab GPU pipeline (~35–50 compute units out of 200) for BanglaBERT, MuRIL, XLM-R, Distil-mBERT, MobileBERT, INT8 ONNX quantization, CPU latency profiling, and 4-bit selective SLM explanations.

## Quickstart

### Local (CPU — Phases 1, 2 & 3)
```bash
uv venv .venv --python 3.12
uv pip install -r <(uv pip compile pyproject.toml)
.venv/bin/python scripts/phase1_fetch_and_audit.py
.venv/bin/python scripts/phase2_cpu_baselines.py
```

### Google Colab (GPU — Phases 2–5, ~35–50 / 200 units)
1. Upload `data/processed/unified_3class_dedup.parquet` and `data/processed/banglaphish2026_dedup.parquet`.
2. Open `notebooks/colab_phase2_5_gpu_pipeline.ipynb` in Colab (T4 or L4 GPU).
3. Run the fold-0 pilot first (~5–10 min), then set `folds_to_run=(0,1,2,3,4)` for full 5-fold paper tables.
