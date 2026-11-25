# Post-Stage B Stage 4 v8 Readiness Report & Benchmark Workload Decision Analysis

**Date:** 2026-11-25  
**Current Gate Status:** **HOLD AT STAGE 4 v8 INDEPENDENT-REVIEW GATE.**  
- **Full 160-Unit Option A Training (`experiments_abcd`):** **`NO-GO`** (blocked until independent review of v8, owner decision on T4 benchmark workload config, and completed T4 benchmark calibration).  
- **5-Unit T4 Benchmark (`t4_benchmark`):** **FAIL-CLOSED UNDER FROZEN CONFIG** (`benchmark_workload_fixture_eval: true` with `SYNTHETIC_WORKLOAD_FIXTURE_ENABLED = False`). Requires owner decision (Section 2 below) plus separate explicit execution authorization.  
- **Capped GPU Smoke Test (`gpu_smoke_test`):** **READY FOR INDEPENDENT REVIEW** (does not invoke synthetic workload fixture; requires separate owner authorization before any Colab execution).  
- **Statistical Governance:** `CONFIRMATORY_DECLARATIONS_ENABLED = False` (`FAIL_OR_INCONCLUSIVE_CONFIRMATORY_DISABLED`).

---

## 1. Local Readiness Summary (Stage 4 v8)

All verified P0, High, and Medium findings from the Stage 4 v7 independent audit have been remediated and verified locally under the hard `<= 25%` host CPU cap (`112/112` `unittest`, `tests/test_pipeline.py`, `282/282` Category A integrity, `v3`–`v8` kits verified, `34/34` `v8` parity).

### Remaining Unverified Items (Strictly Live Colab / GPU / Drive)
The following CANNOT be verified on this local CPU-only workstation and remain **`NOT VERIFIED`** until an owner-authorized Colab execution:
1. Live `transformers==4.57.1` `AutoTokenizer.from_pretrained` / `AutoModelForSequenceClassification.from_pretrained` download and snapshot commit resolution for all 4 pinned HuggingFace backbones (`csebuetnlp/banglabert`, `google/muril-base-cased`, `xlm-roberta-base`, `distilbert-base-multilingual-cased`).
2. Live 32-pair allowed-view tokenizer collision gate (`256/256` checks on real HF tokenizers + real `csebuetnlp/normalizer@d405944d...`).
3. Live `pip install` dependency resolution preserving Colab's pre-installed CUDA `torch >= 2.6`.
4. Real Tesla T4 VRAM peak usage, FP16 `GradScaler` overflow rate, actual wall-clock time per epoch/unit, and Colab Compute Unit (CU) burn rate.
5. Real Google Drive FUSE mount behavior (latency, reconnect readback, Trash quota accounting after checkpoint pruning).

---

## 2. Benchmark Workload Configuration Decision Analysis (Mandatory Section 4)

### Current Frozen State (Preserved Untouched)
- File: [`experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json`](../experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json) (SHA256 `03a5f10d433f657af82d1af401774fd40d9d36209d35e13a14bc934086416180`)
- Setting: `"benchmark_workload_fixture_eval": true`, `"run_heldout_eval": false`, `"run_adversarial_eval": true`, `"run_counterfactual_eval": true`
- Code guard: `SYNTHETIC_WORKLOAD_FIXTURE_ENABLED = False` (enforced pre-lease in `_validate_pre_run_compatibility` AND pre-init in `hf_train_and_predict_fold`).
- **Effect:** Executing the frozen v1 T4 benchmark config currently fails closed BEFORE acquiring a lease, creating `run_state.json`, loading weights, or running GPU training. Synthetic text concatenation (`_build_t4_benchmark_workload_fixture`) is **NOT** enabled and the frozen config was **NOT** modified.

### Why the Conflict Exists
`_build_t4_benchmark_workload_fixture` originally paired/concatenated two development-pool messages (`f"{pool[i]} {pool[j]}"`) to construct `5,416` long synthetic strings (~165 chars mean) approximating `eval_banglaphish_ood.parquet` length without touching held-out/OOD data. Under the permanent Stage B stop on all synthetic text generation, that string concatenation is blocked fail-closed.

### Unmodified Dev-Data-Only Benchmarking Alternatives (Zero Synthetic Text, Zero Held-Out/OOD Exposure)

Available unmodified development-only splits in `post_stage_b_bundle_v2`:
- `fold_0` validation split: `800` unmodified real rows (`real_train_pool.parquet`)
- `eval_smishdetect_validation.parquet`: `757` unmodified real rows
- `counterfactual_smish_probe.parquet` (Fold 0 OOF subset): `~255` rows × 3 variants (`~765` inference calls)
- Adversarial `val` split: `800` rows

| Option | Design | Held-Out / OOD Rows Used | Synthetic Text Generated | Inference Workload Covered | Resource-Measurement Implications | Config Change Required? |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **Option B1 (Recommended — Pure Unmodified Dev Benchmark)** | Authorize a new versioned config `post_stage_b_stage4_t4_benchmark_config_v2.json` with `"benchmark_workload_fixture_eval": false`, adding an unmodified `smishdetect_validation` (`757` rows) inference pass when `run_mode == "t4_benchmark"`. Extrapolate `smishdetect_test` (`1,698` rows) + `banglaphish_ood` (`5,416` rows) inference time analytically from measured per-row ms/seq at `max_length=128` (`batch_size=64`) with a **1.5×–2.0× safety margin** on OOD inference. | `0` | `0` | Training (`3,844`/`3,199` rows × 3 epochs) + `val` (`800`) + `val_adv` (`800`) + `smishdetect_validation` (`757`) + `counterfactual` (`~765`). | Training dominates >90% of T4 unit wall-clock time and 100% of peak VRAM (`batch_size=32` forward+backward+AdamW states vs. `eval_batch_size=64` no-grad forward). Measured `val` + `smishdetect_validation` inference throughput (`rows/s`) gives an honest lower-bound OOD inference estimate without any synthetic text; analytical extrapolation explicitly accounts for longer OOD sequence lengths up to `max_length=128`. | Yes — owner must authorize a **new versioned v2 benchmark config** (keeping v1 frozen). |
| **Option B2 (Repeated Unmodified Dev-Row Inference Pass)** | Authorize a new v2 benchmark config with a dedicated `unmodified_dev_inference_repeat_rows = 7114` mode that cycles **unmodified** existing dev rows (`smishdetect_validation` + `fold_0 val`, selecting longest real dev rows up to `max_length=128` WITHOUT concatenating or altering text) to execute `7,114` exact forward passes. | `0` | `0` | Full training + `7,114` unmodified-row forward passes (same batch count as `1,698 + 5,416`). | Directly measures real GPU/DataLoader/tokenization overhead for `7,114` inference rows without creating any new text strings; however, mean real dev sequence length (~80–84 chars) is still shorter than `banglaphish_ood` (~165 chars), so dynamic padding makes batches slightly faster than OOD unless padded to `max_length=128` (which gives a strict **upper bound** on OOD inference latency!). | Yes — owner must authorize a **new versioned v2 benchmark config** + small runner flag. |
| **Option B3 (Keep Current Frozen Config Blocked)** | Make no changes; keep `benchmark_workload_fixture_eval: true` and `SYNTHETIC_WORKLOAD_FIXTURE_ENABLED = False`. | `0` | `0` | None (`t4_benchmark` remains fail-closed). | `t4_benchmark` cannot execute until Option B1 or B2 is approved; `gpu_smoke_test` remains usable for live Colab smoke verification (tokenizer gate, dependency install, CUDA PyTorch, Drive staging) if separately authorized. | None. |

### Required Owner Decisions Before Next Step
1. **Independent Review of Stage 4 v8:** Audit the v8 codebase, `v8` portable kit, and 6 v8 reports.
2. **Benchmark Workload Decision:** Select **Option B1** (recommended: unmodified dev splits + analytical OOD inference bound), **Option B2** (repeated unmodified dev rows, optionally fixed-padded to `max_length=128` for a strict upper bound), or **Option B3** (keep `t4_benchmark` blocked).
3. **Separate Execution Authorization:** If v8 passes independent review, decide whether to first authorize a minimal live Colab `gpu_smoke_test` (using `v8` kit) to verify real Colab `transformers==4.57.1` tokenizer snapshot pinning, `normalizer` dependency closure, and Drive FUSE before running any 5-unit T4 benchmark.
