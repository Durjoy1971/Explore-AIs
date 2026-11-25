# Post-Stage B — Stage 4 Five-Unit T4 Colab Benchmark Plan (Prepared Only, Not Authorized)

**Date:** 2026-11-24  
**Benchmark Config:** [post_stage_b_stage4_t4_benchmark_config.json](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json)  
**Benchmark Notebook:** [post_stage_b_stage4_experiments.ipynb](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/post_stage_b_stage4_experiments.ipynb)  
**Authorization Status:** **PREPARED ONLY — NOT AUTHORIZED TO EXECUTE.** Separate explicit owner authorization on a fresh `ARTIFACT_RUN_ID` is mandatory after Stage 4 review.

---

## 1. Benchmark Unit Schedule & Anti-Leakage Controls

- **`unit_schedule` (`5` Units Total, `1.25×` all-backbone workload):**
  1. `exp_D__csebuetnlp__banglabert__canon_masked__fold_0` (`3,844` train rows, `363` steps)
  2. `exp_D__google__muril-base-cased__canon_masked__fold_0` (`3,844` train rows, `363` steps)
  3. `exp_D__xlm-roberta-base__canon_masked__fold_0` (`3,844` train rows, `363` steps)
  4. `exp_D__distilbert-base-multilingual-cased__canon_masked__fold_0` (`3,844` train rows, `363` steps)
  5. `exp_B__xlm-roberta-base__canon_masked__fold_0` (`3,199` weighted real train rows, `300` steps — validates `WeightedTrainer` on largest backbone)
- **Strict Anti-Leakage Guard & Length-Matched Dev Fixture:** Config enforces `run_mode = "t4_benchmark"`, `run_heldout_eval = false`, and `benchmark_workload_fixture_eval = true`. Held-out `smishdetect_test` and `banglaphish_ood` are **never scored or exposed** during hardware benchmarking or batch-size tuning. Inference throughput is measured on `smishdetect_validation` (`708` rows) + `_build_t4_benchmark_workload_fixture(...)` (`7,114` dev-only rows: `1,698` single dev texts matching test row count + `5,416` deterministically pair-concatenated dev texts matching OOD row count and ~165-char / ~26-word length profile).
- **Fixture Limitations & Uncertainty:** Although synthetic pair concatenation aligns sequence length and token-truncation load toward `banglaphish_ood` (`max_length = 128`), dev SMS texts cannot replicate exact OOD subword vocabulary fragmentation; runtime extrapolation must apply a conservative `20–30%` safety margin and treat throughput numbers as estimates subject to inference-distribution uncertainty.
- **Fresh Run Identity:** Must use a dedicated `ARTIFACT_RUN_ID` (e.g. `stage4-t4-bench-01`) inside fresh Drive workspace `bangla_smish_post_stage_b_stage4_v3` (`AUTHORIZE_T4_BENCHMARK = True`, `AUTHORIZE_FULL_TRAINING = False`).

---

## 2. Required Measurements (`hf_train_and_predict_fold` Telemetry + Manual/Colab Log)

1. Peak GPU VRAM (`peak_cuda_allocated_bytes`, `peak_cuda_reserved_bytes`) vs. `15 GB` T4 ceiling; test `batch_size=16, gradient_accumulation_steps=2` fallback if `xlm-roberta-base` or `muril` nears OOM.
2. Effective batch size (`32`) and `gradient_accumulation_steps` (`1`, or `2` if halved).
3. Optimizer-step latency (`optimizer_step_ms`, both unweighted `Trainer` and `WeightedTrainer`).
4. Epoch-end CV validation time (`eval_seconds`).
5. CPU tokenization time (`tokenization_seconds`).
6. Inference latency (`inference_seconds` over dev validation + workload fixture).
7. Local SSD checkpoint commit time (`local_checkpoint_seconds`).
8. Google Drive UUID staging + readback sync time (`drive_sync_seconds`) + Drive Trash behavior verification.
9. Initialization & model download/load overhead (`init_seconds`).
10. Colab Compute-Unit (`CU/hr`) consumption before vs. after benchmark in Colab UI.
11. Live mid-run interruption + Phase 1–3 epoch-level checkpoint recovery behavior and allowed-view 32-pair tokenizer collision check.

---

## 3. Full `160`-Unit Extrapolation Formula & Budget Gate

For each backbone $b \in \{1..4\}$, let $\bar{T}_{\text{base},b}$ be fixed per-unit time (`init + tok + inf + ckpt + sync`), $t_{\text{opt},b}$ be measured `optimizer_step_ms`, and measured step counts over `5` folds $\times$ `2` views be $S_A = 300$, $S_B = 300$, $S_C = 333$, $S_D = 363$ (`(300 + 300 + 333 + 363) * 2 * 5 = 12,960` steps per backbone across `40` units):
$$\hat{T}_{160} = \sum_{b=1}^{4} \Big[ 40\,\bar{T}_{\text{base},b} + 12{,}960 \cdot t_{\text{opt},b}^{\text{unweighted}} \text{ (adjusted for B weighted overhead)} \Big] + T_{\text{restart}},$$
$$\hat{\text{CU}}_{160}^{\text{safe}} = (1 + m_{\text{safety}}) \cdot \frac{\hat{T}_{160}}{3600} \cdot R_{\text{CU/hr}}, \qquad m_{\text{safety}} \in [0.20,\, 0.30].$$

- **Budget Honesty Gate:** Whether Colab's `200`-CU budget suffices for all `160` units (and multi-seed follow-up) is **UNVERIFIED before empirical measurement** on T4. Full training must NOT start until measured $\hat{\text{CU}}_{160}^{\text{safe}}$ and VRAM/Drive gates are reviewed and separately approved.
