# Post-Stage B — Stage 4 Five-Unit T4 Benchmark Final Preflight Audit

**Date:** 2026-11-24  
**Repository:** `/home/bs01415/Documents/Programming/ExploringAIs` (`HEAD`: `4d341183cb7a6fce0ed773ff33da2a3b757d6f94`)  
**Audited Config:** [`experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json) (SHA256 `03a5f10d433f657af82d1af401774fd40d9d36209d35e13a14bc934086416180`)  
**Audited Bundle:** [`data/processed/post_stage_b_bundle_v2/dataset_manifest.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/post_stage_b_bundle_v2/dataset_manifest.json) (`manifest_version = "post_stage_b_v2"`)  
**Audit Mode:** Strictly read-only inspection of source code, Parquet cohorts, upstream curation provenance (`curation_v2`), configuration files, and notebook cells. Zero project files modified; zero unit tests executed; zero Colab, Drive, model-training, synthetic-generation, or Git history actions performed.

---

## Executive Audit Verdict

| Preflight Audit Target | Verdict | Verified Summary |
|---|---|---|
| **1. 7,114-Row Dev-Only Workload Fixture** | **`PASS` — Zero Held-Out / OOD Contribution Proven** | Constructed deterministically from **`1,508`** non-empty `text_canon_masked` dev rows (`708` `eval_smishdetect_validation` + `800` Fold-0 `real_train_pool` CV validation) across `1,367` families. Zero (`0`) row IDs, family IDs, exact texts, or upstream `provenance.parquet` links originate from `eval_smishdetect_test` (`1,698`) or `eval_banglaphish_ood` (`5,416`). Output probabilities are discarded immediately after timing (`_ = _predict_probs(fixture_texts)`). |
| **2. 783 Counterfactual Texts (`261 × 3`)** | **`PASS` — Zero Pilot Training Membership Proven** | **`0 / 261`** parent rows come from `synthetic_pilot_300.parquet`. The matching number `261` is a **pure numerical coincidence** (`synthetic_pilot_300` has `261` synthetic *Promotional* rows + `39` Normal, `0` Smishing; Fold 0 of `counterfactual_smish_probe` has **`261` real *Smishing*** rows across `239` families drawn strictly from Fold-0 CV validation `val_df`, `0` overlap with Fold-0 train, Pilot, B1–B7, `smishdetect_validation`, `smishdetect_test`, or `banglaphish_ood`). Classified as **leakage-free, real, out-of-fold Fold-0 CV validation Smishing probes**. |
| **3. Five-Unit Config & Safety Restrictions** | **`PASS` — Fail-Closed Invariants Verified** | Exact 5 units verified (`D` × 4 backbones at `363` steps + `B` × `xlm-roberta-base` at `300` steps = `1,752` steps; Fold `0`, `canon_masked` only); `run_heldout_eval = false`; `confirmatory_mode = false`; `CONFIRMATORY_DECLARATIONS_ENABLED = False`; `keep_model_weights = "none"`; Notebook Cells 1, 5, and 6 enforce explicit authorization and preflight gates. |

---

## 1. Provenance Proof — 7,114-Row Dev-Only Workload Fixture

### 1.1 Exact Source Code & Call Path

1. **Fixture Construction Helper — [`src/bangla_smish/post_stage_b_runner.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L467-L506) (`_build_t4_benchmark_workload_fixture`):**
   - `L473-L475`: Filters non-empty strings from `val_df[text_col]` (`fold_val_texts`) and `sd_val_df[text_col]` (`sd_val_texts`), then concatenates `pool = sd_val_texts + fold_val_texts`.
   - `L480`: Builds `short_texts = [pool[i % len(pool)] for i in range(1698)]` (`1,698` rows mirroring `eval_smishdetect_test` cardinality).
   - `L483`: Builds `long_texts = [f"{pool[(i * 7) % n_pool]} {pool[(i * 13 + 3) % n_pool]}" for i in range(5416)]` (`5,416` rows mirroring `eval_banglaphish_ood` cardinality and length profile).
   - `L484`: Concatenates `fixture_texts = short_texts + long_texts` (`1,698 + 5,416 = 7,114`).
2. **Invocation & Non-Storage Call Site — [`src/bangla_smish/post_stage_b_runner.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L714-L721):**
   - `L714`: Executes ONLY when `cfg.run_mode == "t4_benchmark" and cfg.benchmark_workload_fixture_eval` (`cfg.run_heldout_eval` is `False`).
   - `L716-L717`: Predicts and saves `"smishdetect_validation_probs"` (`708` rows).
   - `L718-L721`: Calls `_build_t4_benchmark_workload_fixture(val_df, heldout_sets["smishdetect_validation"], text_col)` and runs `_ = _predict_probs(fixture_texts)` strictly inside the `inference_seconds` timer (`L691-L736`). Predicted probabilities on the 7,114 fixture texts are **immediately discarded** (never written to `preds.npz`, `metrics.json`, or used in threshold calibration / checkpoint selection).
3. **Input Split Loaders — [`src/bangla_smish/post_stage_b_data.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_data.py):**
   - `val_df`: Loaded identically for both Arm `D` and Arm `B` Fold `0` via `build_experiment_fold_data` (`L785-L847`, `va_real = real_df[real_df["curation_cv_fold"] == 0]` at `L807`, asserted `0` synthetic rows at `L821-L822` and `0` row/family overlap with `train_df` at `L825-L828`).
   - `sd_val_df`: Loaded via `load_heldout_eval_sets(bdir)["smishdetect_validation"]` (`L850-L858`), built exclusively from `existing_unique.parquet` rows with `eval_group == "smishdetect_validation"` (`L537-L542`).

### 1.2 Row-Level & Upstream Dataset Provenance of the `1,508`-Row Source Pool

Under `post_stage_b_stage4_t4_benchmark_config.json` (`text_views = ["canon_masked"]`, `folds = [0]`), `text_col = "text_canon_masked"` yields **`0` empty strings dropped**, so the source pool has exact size **`1,508` rows (`1,367` unique `family_id`s)**:

| Source Component | Bundle Table | Rows (`is_synthetic`) | Unique Families | Label Breakdown | Language Breakdown | Upstream `existing_unique.parquet` (`primary_dataset` / `original_split`) | Upstream `provenance.parquet` Raw Links (Files & Counts) | Links to `test` or `banglaphish` |
|---|---|---:|---:|---|---|---|---|---:|
| **Pool Part A (`sd_val`)** | `eval_smishdetect_validation.parquet` (`curation_cv_fold=-1`, `source_type="SMS"`) | `708` (`False`: `708`) | `606` | `Normal: 265`, `Smishing: 265`, `Promotional: 178` | `Banglish: 184`, `CodeMix: 181`, `Bengali: 179`, `English: 164` | `708/708` exact `canonical_id` + `family_id` match; `primary_dataset=smishdetect_2026 (708)`; `original_split`: `validation (610 direct)`, `train (98 family_overlap moved to val)`; `eligible_for_training=False (708)`; `contributing_datasets`: `smishdetect_2026 (591)`, `bangalabarta_2025,legacy_2024,smishdetect_2026 (107)`, `bangalabarta_2025,smishdetect_2026 (10)` | **`1,253` raw provenance links:** `validation-00000-of-00001.parquet` (`611`), `BangalaBarta bangla_spam_sms smishing.csv` (`319`), `DataSet.csv` (`178`), `train-00000-of-00001.parquet` (`145`). Dispositions: `canonical_kept: 708`, `exact_dup_merged: 499`, `near_dup_variant_collapsed: 46` | **`0`** |
| **Pool Part B (`val_df` Fold 0)** | `real_train_pool.parquet` (`curation_cv_fold=0`, `source_type="SMS"`) | `800` (`False`: `800`) | `761` | `Smishing: 309`, `Normal: 302`, `Promotional: 189` | `English: 234`, `Banglish: 207`, `Bengali: 191`, `CodeMix: 168` | `800/800` exact `canonical_id` + `family_id` match (`val_D == val_B` identical); `primary_dataset=smishdetect_2026 (800)`; `original_split=train (800)`; `eval_group=train_pool (800)`; `benchmark_membership=primary_train_eligible (800)`; `contributing_datasets`: `smishdetect_2026 (680)`, `bangalabarta_2025,legacy_2024,smishdetect_2026 (117)`, `bangalabarta_2025,smishdetect_2026 (3)` | **`1,309` raw provenance links:** `train-00000-of-00001.parquet` (`829`), `BangalaBarta bangla_spam_sms smishing.csv` (`297`), `DataSet.csv` (`183`). Dispositions: `canonical_kept: 800`, `exact_dup_merged: 481`, `near_dup_variant_collapsed: 28` | **`0`** |
| **Total Fixture Pool (`pool`)** | Disjoint Union (`sd_val + val_fold0`) | **`1,508`** (`100%` real) | **`1,367`** (`606 + 761`, zero cross-split family overlap) | `Smishing: 574`, `Normal: 567`, `Promotional: 367` | Mixed 4 canonical languages | `1,508/1,508` traced to `smishdetect_2026` + deduplicated `bangalabarta_2025` / `legacy_2024` train/validation sources | `2,562` raw provenance links total (`0` synthetic, `0` test, `0` OOD) | **`0`** |

**Cryptographic Pool & Fixture Fingerprints (Reproducible):**
- Sorted `1,508` pool `row_id`s SHA256 (`"\n".join(sorted(used_rids))`): `9c897995551f5782f7fea199e0b6282b4918befb2975631ec78f1ae20dabd7c0`
- Sorted `1,367` pool `family_id`s SHA256 (`"\n".join(sorted(used_fids))`): `24a3b3c8d71f8842852c7f36c089a35332496abd27c20d1ced78fd5a007dd0a7`
- Ordered `7,114` `text_canon_masked` fixture strings SHA256 (`"\n".join(fixture_texts)`): `fbd6bd00a61ad3f1bcbcb819bd30d0c597940a854940810f5c2b08ad492184bc`

### 1.3 Deterministic Transformations, Modulo Structure & Length Comparison

1. **`short_texts` (`1,698` rows):** `pool[i % 1508]` for `i = 0..1697`.
   - Exercises all `1,508` pool rows once plus repeats the first `1,698 - 1,508 = 190` rows from `sd_val[0:190]` a second time (`1,318` once, `190` twice).
   - Slot cohort distribution: `smishdetect_validation: 898`, `real_train: 800`. Slot labels: `Smishing: 638`, `Normal: 632`, `Promotional: 428`.
   - Unique `text_canon_masked` strings in `short_texts`: `1,507` (exactly two `eval_smishdetect_validation` sibling Smishing rows in `fam_a988df2728`, `cur_30e7f36224fb` and `cur_9a8039e30251`, differ only in their URL so their canonical `<URL>`-masked text is identical).
2. **`long_texts` (`5,416` rows):** `f"{pool[(i * 7) % 1508]} {pool[(i * 13 + 3) % 1508]}"` for `i = 0..5415`.
   - Cohort slot distribution: left operand (`real_train: 2,857`, `smishdetect_validation: 2,559`), right operand (`real_train: 2,831`, `smishdetect_validation: 2,585`).
   - Label slot distribution: left (`Smishing: 2,063`, `Normal: 2,053`, `Promotional: 1,300`), right (`Normal: 1,867`, `Promotional: 1,775`, `Smishing: 1,774`).
   - **Exact Modulo Arithmetic Property (audited finding, non-leakage):** Because `1,508 = 4 × 13 × 29`, `gcd(7, 1508) = 1` (all `1,508` rows cycle as left operands), whereas **`gcd(13, 1508) = 13`** (`1508 / 13 = 116`), so right operands cycle across `116` pool indices (`3 mod 13` congruence class). Therefore `long_texts` contains `1,508` unique `(left, right)` pairs (`892` occurring `4×` and `616` occurring `3×`), with **`0` self-pairs** (`left == right`) and **`0` same-family pairs**. Combined across `7,114` slots, there are `3,015` unique strings (`1,507` short + `1,508` long).
   - **Length Profile vs. True Held-Out + OOD (`7,114` rows, `text_canon_masked`):**
     - Fixture (`7,114`): `char_len_mean = 145.58`, `p50 = 130.0`, `p95 = 308.70` (`word_len_mean = 23.2`, `p50 = 21.0`, `p95 = 48.0`).
     - Real `eval_smishdetect_test` (`79.21` mean) + `eval_banglaphish_ood` (`165.11` mean) combined (`7,114`): `char_len_mean = 144.61`, `p50 = 116.0`, `p95 = 283.0`. (Fixture is length-matched and slightly conservative in `p50`/`p95`.)

### 1.4 Exhaustive Overlap Proof Against Every Split in `post_stage_b_bundle_v2`

| Compared Bundle Split / Dataset | Rows | `row_id` Overlap w/ Fixture Pool | `family_id` Overlap w/ Fixture Pool | Pool `text_canon_masked` Overlap | 7,114 Fixture `text_canon_masked` Exact Overlap |
|---|---:|---:|---:|---:|---:|
| **`eval_smishdetect_test.parquet`** *(Held-Out Test)* | `1,698` | **`0`** | **`0`** | **`0`** | **`0`** |
| **`eval_banglaphish_ood.parquet`** *(Held-Out OOD)* | `5,416` | **`0`** | **`0`** | **`0`** | **`0`** |
| **`synthetic_pilot_300.parquet`** | `300` | **`0`** | **`0`** | **`0`** | **`0`** |
| **`synthetic_batches_b1_b7.parquet`** | `345` | **`0`** | **`0`** | **`0`** | **`0`** |
| **`train_D_fold0`** *(Arm D Fold 0 Training)* | `3,844` | **`0`** | **`0`** | **`0`** | — |
| **`train_B_fold0`** *(Arm B Fold 0 Training)* | `3,199` | **`0`** | **`0`** | **`0`** | — |
| **`val_fold0`** *(Fold 0 CV Validation)* | `800` | `800` | `761` | `800` | `800` |
| **`real_train_pool.parquet`** *(Full Pool, Folds 0–4)* | `3,999` | `800` *(Fold 0 `val_df` only)* | `761` | `800` | `800` |
| **`eval_smishdetect_validation.parquet`** | `708` | `708` | `606` | `707` | `707` |
| **`counterfactual_smish_probe.parquet`** *(All Folds)* | `1,279` | `261` *(Fold 0 `val_df` Smishing parents)* | `239` | N/A | N/A |

**Stage 3 Cross-Boundary Near-Duplicate Disclosure (`14` of the `32` TF-IDF `>= 0.85` pairs touch the `1,508`-row dev pool):**  
Zero rows from `eval_smishdetect_test` enter the fixture, and all fixture probabilities are discarded (`_ = _predict_probs`). For completeness, `14` Stage 3 candidate pairs have one member in `sd_val` or `val_fold0` (`P04`, `P05`, `P06`, `P07`, `P10`, `P13`, `P18`, `P20`, `P22`, `P23`, `P24`, `P26`, `P31`, `P32`); of these, **`5` (`P10`, `P20`, `P26`, `P31`, `P32`)** have an upstream SmishDetect TF-IDF `>= 0.85` partner in `eval_smishdetect_test` (`cur_c00c6e3fa781`, `cur_ede29997f5d4`, `cur_b6041ad053cb`, `cur_9b7b07f54ebb`, `cur_a8aaa428cc24`), and **`9`** link to other `real_train_pool` folds. Because the T4 benchmark never scores `eval_smishdetect_test` (`run_heldout_eval = false`) and never saves fixture predictions, this causes zero leakage or evaluation contamination.

---

## 2. Provenance & Classification — All 783 Counterfactual Texts (`261 × 3`)

### 2.1 Exact Source Code & Construction Pipeline

1. **Counterfactual Probe Table Builder — [`src/bangla_smish/post_stage_b_data.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_data.py#L326-L372) (`_build_counterfactual_probe_df`):**
   - Called strictly on `real_train` (`eligible_train.parquet`) at `post_stage_b_data.py:561` — **never reads `synthetic_pilot_300` or `synthetic_batches_b1_b7`**.
   - Selection filter (`L332-L335`): Selects rows in `real_train_pool.parquet` where `label_3class == "Smishing"` AND (`_SCHEMA_URL_RE` or `_SCHEMA_PHONE_RE` matches `text`), producing `1,279` rows partitioned across all 5 CV folds (`{0: 261, 1: 256, 2: 252, 3: 261, 4: 249}`).
   - Deterministic 3-variant perturbation per parent (`L340-L369`):
     - `orig_text_<view>`: Original Smishing message masked for `<view>` (`canon_masked` in T4).
     - `contact_swap_text_<view>`: Replaces URLs with `"https://example.com/offer"` and phones with `"01700-000011"`, then masks for `<view>`.
     - `contact_brand_swap_text_<view>`: Additionally replaces `REAL_BD_BRANDS` matches with `"PurbachalBank"` on `contact_swap`, then masks for `<view>`.
2. **Fold Selection & Anti-Leakage Verification — [`src/bangla_smish/post_stage_b_data.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_data.py#L384-L408) (`select_counterfactual_fold`):**
   - Invoked by `_split_for_unit` in [`src/bangla_smish/post_stage_b_runner.py:907-910`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L907-L910).
   - Filters `probe["curation_cv_fold"] == 0` (`261` rows × `3` columns = **`783` texts**; verified identical across Exp `D` and Exp `B` Fold `0`).
   - Enforces fail-closed assertions (`L396-L407`) verifying **zero** `row_id` or `family_id` overlap against `train_df` and exact parent equality against Fold-0 `val_df`. Scored at [`post_stage_b_runner.py:725-735`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L725-L735).

### 2.2 Exact Source Row IDs, Family IDs, Labels & Upstream Provenance (`261` Parents)

- **Parent Cardinality:** **`261` unique `row_id`s**, **`239` unique `family_id`s** (`curation_cv_fold = 0` for `261/261`).
- **Cryptographic Digests (Reproducible):**
  - Sorted `261` `row_id`s SHA256 (`"\n".join(sorted(cf_f0["row_id"]))`): `106d955952c1d24a99248cc067ff511f19599190621cf7aad5bec015e4d5fb15`
  - Sorted `239` `family_id`s SHA256 (`"\n".join(sorted(set(cf_f0["family_id"])))`): `e9b5ea10ee406c477e8f072d6d9795411ff42a0e3dd18892bf91b4b5860a2f93`
- **First 10 Verified Fold-0 Parents (`row_id`, `family_id`, `language`):**
  1. `cur_00136d3def1f` | `fam_13716b63f2` | `Bengali`
  2. `cur_009a93cbcae5` | `fam_4529bb3f16` | `Bengali`
  3. `cur_0128ac36dc3b` | `fam_a0af283224` | `English`
  4. `cur_016733fcebdf` | `fam_fd4f46e8ea` | `CodeMix`
  5. `cur_02349676f7f2` | `fam_57cddb80b1` | `English`
  6. `cur_02a0fde5fe37` | `fam_c95382bd43` | `English`
  7. `cur_03f6b0152595` | `fam_ddbdc8f062` | `English`
  8. `cur_04627d68a34b` | `fam_49726cbc74` | `English`
  9. `cur_0586ba0ece0f` | `fam_2d9d6f6c4a` | `Bengali`
  10. `cur_0631e41ea805` | `fam_589e5a6445` | `Bengali`
- **Labels & Languages:** **`261 / 261` (`100%`) `Smishing`** (`label_idx = 2`); Languages: `English: 81`, `Banglish: 64`, `CodeMix: 60`, `Bengali: 56`.
- **Upstream `existing_unique.parquet` & `provenance.parquet` Evidence:**
  - `261/261` match `real_train_pool.parquet` (`curation_cv_fold = 0`, `is_synthetic = False`) and `existing_unique.parquet` (`primary_dataset = smishdetect_2026`, `original_split = train`, `eval_group = train_pool`, `benchmark_membership = primary_train_eligible`, `eligible_for_training = True`).
  - `provenance.parquet` (`415` raw links across `261` IDs): `data/raw/smishdetect_hf/data/train-00000-of-00001.parquet` (`272`), `data/raw/bangalabarta/BangalaBarta bangla_spam_sms smishing.csv` (`90`), `DataSet.csv` (`53`); dispositions: `canonical_kept: 261`, `exact_dup_merged: 143`, `near_dup_variant_collapsed: 11`. Zero links from `test` or `banglaphish`.

### 2.3 Direct Resolution: Do the 261 Source Messages Belong to the Pilot Training Cohort?

**NO — `0 / 261` belong to `synthetic_pilot_300.parquet`.**

The shared integer `261` is a **pure coincidence across disjoint cohorts and opposite classes**:
- **`synthetic_pilot_300.parquet`** (`300` rows total): `Promotional: 261`, `Normal: 39`, **`Smishing: 0`**, `is_synthetic = True` (`300/300`).
- **`counterfactual_smish_probe.parquet` Fold 0** (`261` rows × `3` = `783` texts): **`Smishing: 261`**, `Promotional: 0`, `Normal: 0`, `is_synthetic = False` (`261/261`).

Exact cross-cohort overlap audit of the `261` Fold-0 counterfactual parents (`1,279` total also checked against Pilot = `0/0`):

| Cohort Compared Against 261 CF Parents | `row_id` Overlap | `family_id` Overlap | Interpretation |
|---|---:|---:|---|
| **`synthetic_pilot_300.parquet`** | **`0`** | **`0`** | Completely disjoint (`0/300` Pilot; `0/1,279` across all folds too) |
| **`synthetic_batches_b1_b7.parquet`** | **`0`** | **`0`** | Completely disjoint |
| **`train_D_fold0`** (`3,844` rows) | **`0`** | **`0`** | Zero Fold-0 training leakage |
| **`train_B_fold0`** (`3,199` rows) | **`0`** | **`0`** | Zero Fold-0 training leakage |
| **`val_fold0`** (`800` Fold-0 CV val rows) | **`261`** | **`239`** | `100%` of parents come strictly from Fold-0 CV validation |
| **`real_train_pool.parquet`** (`3,999` Folds 0–4) | `261` *(Fold 0 only)* | `239` | Validation parents in Fold 0; real training rows only in Folds 1–4 |
| **`eval_smishdetect_validation.parquet`** | **`0`** | **`0`** | Disjoint |
| **`eval_smishdetect_test.parquet`** | **`0`** | **`0`** | Disjoint |
| **`eval_banglaphish_ood.parquet`** | **`0`** | **`0`** | Disjoint |

### 2.4 Classification & `canon_masked` Interpretability Finding

- **Probe Classification:** **Leakage-free, real, out-of-fold Fold-0 CV validation Smishing counterfactual probes** (NOT synthetic Pilot rows; NOT Fold-0 training rows; NOT held-out/OOD rows). *(Note: Across all 5 folds in full Option A CV, the 261 Fold-0 rows are out-of-fold validation probes in Fold 0 and training rows in Folds 1–4. Also note that `4` parents touch Stage 3 TF-IDF `>= 0.85` pairs: `P04: cur_0667dc3a1230`, `P07: cur_902a0a1a8ac5`, `P13: cur_be7f24b65438` with Folds 2–3 partners, and `P31: cur_e9011c46d295` with a `smishdetect_test` partner—non-blocking because test splits are disabled in T4 benchmark.)*
- **Important `canon_masked` Diagnostic Behavior:**
  - On `raw`: `orig == contact_swap` is `0/261` (`100%` altered) and `contact_swap == contact_brand_swap` is `217/261` (`44` brand swaps).
  - On `canon_masked` (used by T4 benchmark): **`orig == contact_swap` for `257 / 261` (`98.5%`)** because canonical masking already converts URLs to `<URL>` and phones to `<PHONE>` (only `4` Banglish rows—`cur_13ddd42a95e2`, `cur_1b5e53eb1152`, `cur_8305bc8d2e0d`, `cur_c7208fe25839`—differ between `orig` and `contact_swap`), while **`contact_swap == contact_brand_swap` is `217 / 261` (`44/261` differ due to `REAL_BD_BRANDS -> PurbachalBank`)**. Thus, on `canon_masked`, these 783 texts validly measure inference runtime (`783` texts) and canonical brand-swap robustness (`44` active rows), while `orig` vs. `contact_swap` is structurally invariant by design under canonical contact masking.

---

## 3. Five-Unit Configuration & Benchmark Safety Verification

Verified directly against [`experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json#L1-L48) (`30` config keys, SHA256 `03a5f10d433f657af82d1af401774fd40d9d36209d35e13a14bc934086416180`), [`src/bangla_smish/post_stage_b_runner.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L67-L192), [`src/bangla_smish/post_stage_b_stats.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_stats.py), and [`experiments/colab/post_stage_b_stage4_experiments.ipynb`](file:///home/bs01415/Documents/Programming/ExploringAIs/experiments/colab/post_stage_b_stage4_experiments.ipynb#L27-L212):

1. **Exact Five Units (`unit_schedule`, `L38-L44`):**
   - `["D", "csebuetnlp/banglabert", "canon_masked", 0]` (`3,844` train rows -> `363` steps)
   - `["D", "google/muril-base-cased", "canon_masked", 0]` (`3,844` train rows -> `363` steps)
   - `["D", "xlm-roberta-base", "canon_masked", 0]` (`3,844` train rows -> `363` steps)
   - `["D", "distilbert-base-multilingual-cased", "canon_masked", 0]` (`3,844` train rows -> `363` steps)
   - `["B", "xlm-roberta-base", "canon_masked", 0]` (`3,199` real rows, `class_weight` -> `300` steps)
   - Total: **`5` units**, **`1,752` optimizer steps** (`epochs=3, batch_size=32, gradient_accumulation_steps=1, eval_batch_size=64, lr=3e-5, weight_decay=0.01, warmup_ratio=0.1, max_length=128, seed=42, fp16="auto"`).
2. **All Four Pinned Tokenizer Revisions (`L11-L16`) Match `PINNED_TOKENIZER_REVISIONS` (`post_stage_b_tokenizer_gate.py:18-23`):**
   - `csebuetnlp/banglabert`: `9ce791f330578f50da6bc52b54205166fb5d1c8c`
   - `google/muril-base-cased`: `afd9f36c7923d54e97903922ff1b260d091d202f`
   - `xlm-roberta-base`: `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`
   - `distilbert-base-multilingual-cased`: `45c032ab32cc946ad88a166f7cb282f58c753c2e`
3. **Fail-Closed Safety & Governance Restrictions Verified:**
   - `run_heldout_eval: false` (`L30`; `PostStageBConfig.__post_init__` at `post_stage_b_runner.py:181-182` raises `ValueError` if enabled in `t4_benchmark`).
   - `confirmatory_mode: false` (`L36`), `CONFIRMATORY_DECLARATIONS_ENABLED = False`, `STATISTICAL_VALIDATION_STATUS = "FAIL_OR_INCONCLUSIVE_CONFIRMATORY_DISABLED"`, [`reports/stage4_statistical_validation_results.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/stage4_statistical_validation_results.json) untouched (`bde8085b77066cbf449a1b17d0a1afa77f2dcbcb4ea1a82f7d0dde68f6fe504f`).
   - `text_views: ["canon_masked"]` (`L17`); zero `brand_masked` / `brand_stripped` / `raw`.
   - `keep_model_weights: "none"` (`L33`), `checkpoint_retention_limit: 1` (`L34`), `local_checkpoint_root: "/content/tmp_stage4_ckpt"` (`L35`).
   - Runner blocks overlapping bundle/artifact dirs (`post_stage_b_runner.py:1123-1124`).
   - Notebook defaults fail-closed (`AUTHORIZE_T4_BENCHMARK = False`, `AUTHORIZE_FULL_TRAINING = False`, `ALLOW_STALE_WRITER_TAKEOVER = False` at `L30-L56`), Cell 5 gates on `verify_portable_colab_kit` + `verify_manifest` + `run_allowed_view_tokenizer_collision_gate` (`L160-L179`), and Cell 6 verifies `TOKENIZER_COLLISION_GATE_PASSED` (`L196-L197`).

---

## 4. Exact Reproducible Read-Only Verification Commands

```bash
sha256sum \
  experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json \
  reports/stage4_statistical_validation_results.json \
  experiments/colab/post_stage_b_stage4_experiments.ipynb

.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))
from bangla_smish.post_stage_b_data import build_experiment_fold_data, load_heldout_eval_sets, select_counterfactual_fold
from bangla_smish.post_stage_b_runner import _build_t4_benchmark_workload_fixture, load_config

bdir = ROOT / "data/processed/post_stage_b_bundle_v2"
cfg = load_config(ROOT / "experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json")
heldout = load_heldout_eval_sets(bdir)
train_d, val_d, _ = build_experiment_fold_data(bdir, "D", 0, seed=cfg.seed)
sd_val = heldout["smishdetect_validation"]

fixture_texts, fixture_meta = _build_t4_benchmark_workload_fixture(val_d, sd_val, "text_canon_masked")
pool_df = pd.concat([sd_val, val_d], ignore_index=True)
pool_rids = set(pool_df["row_id"])
pool_fids = set(pool_df["family_id"])
cf_f0 = select_counterfactual_fold(heldout["counterfactual_smish_probe"], train_d, val_d, 0)
pilot = pd.read_parquet(bdir / "synthetic_pilot_300.parquet")

assert len(pool_rids) == 1508 and len(pool_fids) == 1367
assert len(fixture_texts) == 7114 and fixture_meta["heldout_or_ood_rows_used"] == 0
assert pool_rids.isdisjoint(heldout["smishdetect_test"]["row_id"])
assert pool_rids.isdisjoint(heldout["banglaphish_ood"]["row_id"])
assert len(cf_f0) == 261 and set(cf_f0["row_id"]).issubset(set(val_d["row_id"]))
assert set(cf_f0["row_id"]).isdisjoint(pilot["row_id"])
assert set(cf_f0["family_id"]).isdisjoint(pilot["family_id"])
assert hashlib.sha256("\n".join(fixture_texts).encode()).hexdigest() == "fbd6bd00a61ad3f1bcbcb819bd30d0c597940a854940810f5c2b08ad492184bc"
assert hashlib.sha256("\n".join(sorted(cf_f0["row_id"])).encode()).hexdigest() == "106d955952c1d24a99248cc067ff511f19599190621cf7aad5bec015e4d5fb15"
print("ALL PREFLIGHT PROVENANCE ASSERTIONS PASSED")
PY
```
