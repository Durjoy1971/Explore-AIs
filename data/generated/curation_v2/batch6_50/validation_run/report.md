# Stage B Pilot Screening & Validation Report (`curation_v2`)

Generated reproducibly by `scripts/validate_stage_b_pilot.py` (`src/bangla_smish/stage_b_pilot.py`). Screening enforces schema, `length_band` consistency (`short(<50)`, `medium(50-120)`, `long(>120)`), `is_synthetic=True`, exact & char-ngram cosine (`>= 0.85` raw & slot-template) isolation against `curation_v2/existing_unique.parquet` (including held-out/OOD) and `quarantined.parquet`, intra-pilot template diversity, fictional entities/reserved domains, label/language rules, and independent review log ingestion.

## 1. Candidate Screening & Independent Review Summary

- **Total candidates screened (`candidates.jsonl`):** **61**
- **Accepted (`accepted.jsonl`):** **50** (81.97%)
  - Confirmed by **both** automated screening and independent review (`review_status="accepted"`): **50**
  - Passed automated screening, pending independent review (`review_status="pending_independent_review"`): **0**
- **Rejected (`rejected.jsonl`):** **11**
- **Exact pilot quota match (`300` rows):** `False`

### Accepted Pilot Distribution (`Label × Language`)

| Class | Bengali | Banglish | English | CodeMix | Total Accepted | Target Pilot Quota |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 0 | 0 | 0 | 0 | **0** | 39 |
| `Promotional` | 14 | 19 | 4 | 13 | **50** | 261 |
| `Smishing` | 0 | 0 | 0 | 0 | **0** | 0 |

### Rejection Breakdown

| Rejection Reason | Count |
| :--- | ---: |
| `independent_review:brand_detection_mismatch` | 1 |
| `independent_review:cross_batch_scenario_clone` | 7 |
| `independent_review:metadata_mismatch` | 1 |
| `independent_review:near_duplicate_slot_template` | 6 |
| `independent_review:unnatural_phrasing` | 2 |

- **Rejected IDs (11):** `syn_b6_0002`, `syn_b6_0005`, `syn_b6_0008`, `syn_b6_0011`, `syn_b6_0018`, `syn_b6_0029`, `syn_b6_0030`, `syn_b6_0031`, `syn_b6_0044`, `syn_b6_r_0001`, `syn_b6_r_0009`

## 2. Domain, Length & Cue Coverage of Accepted Pilot (`N = 50`)

### Domain Breakdown (`Domain × Label`)

| Domain | Normal | Promotional | Total |
| :--- | ---: | ---: | ---: |
| `banking_mfs` | 0 | 5 | **5** |
| `civic_utility_health` | 0 | 1 | **1** |
| `delivery_parcel` | 0 | 3 | **3** |
| `employment` | 0 | 1 | **1** |
| `personal_or_general` | 0 | 11 | **11** |
| `shopping_ecommerce` | 0 | 14 | **14** |
| `telecom` | 0 | 15 | **15** |
| **Total** | **0** | **50** | **50** |

### Length Band & Surface Cue Coverage

| Class | `short(<50)` | `medium(50-120)` | `long(>120)` | Has Link | Has Urgency |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 0 | 0 | 0 | 0 | 0 |
| `Promotional` | 1 | 41 | 8 | 15 | 6 |

## 3. Surface-Only & Length-Only Shortcut Diagnostics (Before vs. After Pilot)

| Metric (`StratifiedGroupKFold`, grouped by `family_id`) | `eligible_train` Alone | `eligible_train + accepted` | Delta |
| :--- | ---: | ---: | ---: |
| Row count (`N`) | 3999 | 4049 | +50 |
| Surface-only 5-fold Accuracy | 73.14% | 72.49% | -0.65% |
| Surface-only 5-fold Macro-F1 | 62.86% | 62.81% | -0.05% |
| Length-only Smishing ROC-AUC | 0.7693 | 0.7678 | -0.0015 |
| Length-only Promo vs. Normal ROC-AUC | 0.7414 | 0.7412 | -0.0002 |

### Per-Class Surface Cue Rates (`Before -> After Pilot`)

| Class | Count (`Before -> After`) | Median Chars (`Before -> After`) | % Has URL (`Before -> After`) | % Has Phone (`Before -> After`) |
| :--- | :--- | :--- | :--- | :--- |
| `Normal` | 1509 -> 1509 | 47.0 -> 47.0 | 0.73% -> 0.73% | 0.73% -> 0.73% |
| `Promotional` | 942 -> 992 | 74.0 -> 74.0 | 29.83% -> 29.84% | 8.92% -> 8.87% |
| `Smishing` | 1548 -> 1548 | 99.0 -> 99.0 | 59.43% -> 59.43% | 27.45% -> 27.45% |

### 3b. 4-View 3-Class TF-IDF+LR Probes (`Before -> After Pilot`)

| Text View | Macro-F1 (`Before -> After`) | Accuracy (`Before -> After`) | Smishing Recall (`Before -> After`) | Recall @ <=1% FPR (`Before -> After`) |
| :--- | :--- | :--- | :--- | :--- |
| `raw` | 98.01% -> 98.07% | 98.1% -> 98.17% | 98.77% -> 98.9% | 98.51% -> 98.71% |
| `canon_masked` | 97.86% -> 97.74% | 97.95% -> 97.83% | 98.45% -> 98.51% | 97.8% -> 97.48% |
| `brand_masked` | 97.76% -> 97.68% | 97.85% -> 97.75% | 98.39% -> 98.45% | 97.8% -> 97.61% |
| `brand_stripped` | 97.71% -> 97.68% | 97.8% -> 97.78% | 98.39% -> 98.51% | 97.8% -> 97.87% |

### 3c. Synthetic-Origin (`is_synthetic`) Separability Probes

| Comparison | Strata / N | Feature View | ROC-AUC | Accuracy | Macro-F1 | Top Positive Synthetic Features |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| `all_vs_real` | 50 syn vs 3999 real | `raw` | 0.9700 | 99.28% | 79.96% | `word__example` (+4.37), `word__chhar` (+3.61), `word__20` (+3.10), `word__15` (+2.90) |
| `all_vs_real` | 50 syn vs 3999 real | `canon_masked` | 0.9713 | 99.14% | 77.05% | `word__chhar` (+3.94), `word__20` (+3.25), `char__ ও ` (+3.16), `word__15` (+3.13) |
| `all_vs_real` | 50 syn vs 3999 real | `brand_masked` | 0.9717 | 99.16% | 77.99% | `word__chhar` (+3.88), `word__20` (+3.23), `char__ ও ` (+3.21), `word__15` (+3.15) |
| `all_vs_real` | 50 syn vs 3999 real | `brand_stripped` | 0.9695 | 99.18% | 77.79% | `word__chhar` (+3.94), `word__20` (+3.34), `char__ ও ` (+3.27), `word__15` (+3.22) |
| `all_vs_real` | 50 syn vs 3999 real | `surface_8feat` | 0.5387 | 67.0% | 41.52% | `word_len` (+0.76), `has_ussd` (+0.27), `has_url` (+0.18), `has_bn` (+0.08) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `raw` | 0.9966 | 99.06% | 90.94% | `word__example` (+3.07), `word__chhar` (+2.70), `word__20` (+2.21), `word__15` (+2.15) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `canon_masked` | 0.9971 | 98.75% | 88.05% | `word__chhar` (+2.93), `word__ussd` (+2.52), `word__20` (+2.29), `word__15` (+2.26) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `brand_masked` | 0.9972 | 98.87% | 89.48% | `word__chhar` (+2.91), `word__ussd` (+2.54), `word__ar` (+2.24), `word__15` (+2.24) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `brand_stripped` | 0.9970 | 98.81% | 88.77% | `word__chhar` (+2.93), `word__ussd` (+2.58), `word__20` (+2.32), `word__15` (+2.31) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `surface_8feat` | 0.7390 | 80.04% | 52.25% | `word_len` (+1.29), `has_ussd` (+0.21), `has_bn` (+0.03), `has_redacted_tag` (+0.00) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `raw` | 0.9567 | 97.48% | 83.11% | `word__example` (+2.87), `word__ar` (+2.19), `word__chhar` (+2.12), `char__ ও ` (+2.00) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `canon_masked` | 0.9361 | 97.18% | 81.32% | `word__ar` (+2.44), `word__chhar` (+2.37), `char__ ও ` (+2.31), `word__15` (+2.12) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `brand_masked` | 0.9306 | 97.18% | 81.32% | `word__ar` (+2.53), `char__ ও ` (+2.40), `word__chhar` (+2.37), `word__15` (+2.21) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `brand_stripped` | 0.9301 | 96.98% | 79.98% | `word__ar` (+2.51), `char__ ও ` (+2.41), `word__chhar` (+2.35), `word__15` (+2.20) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `surface_8feat` | 0.4906 | 59.88% | 41.31% | `word_len` (+0.97), `has_url` (+0.23), `has_bn` (+0.06), `has_phone` (+0.01) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 29 strata (50/50 syn, 818/942 real) | `raw` | 0.9475 | 97.24% | 83.89% | `word__example` (+1.02), `word__20` (+0.69), `word__ar` (+0.61), `word__chhar` (+0.59) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 29 strata (50/50 syn, 818/942 real) | `canon_masked` | 0.9095 | 96.08% | 77.72% | `word__20` (+0.74), `word__ar` (+0.65), `word__chhar` (+0.64), `word__15` (+0.61) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 29 strata (50/50 syn, 818/942 real) | `brand_masked` | 0.9021 | 95.85% | 77.48% | `word__20` (+0.78), `word__ar` (+0.66), `word__chhar` (+0.64), `word__15` (+0.63) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 29 strata (50/50 syn, 818/942 real) | `brand_stripped` | 0.9001 | 96.08% | 78.24% | `word__20` (+0.78), `word__ar` (+0.66), `word__chhar` (+0.63), `word__15` (+0.63) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 29 strata (50/50 syn, 818/942 real) | `surface_8feat` | 0.4928 | 52.19% | 38.37% | `word_len` (+0.35), `has_url` (+0.29), `has_phone` (+0.10), `has_bn` (+0.01) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (50/50 syn, 890/942 real) | `raw` | 0.9351 | 97.23% | 82.17% | `word__example` (+0.43), `word__20` (+0.39), `word__20 off` (+0.35), `word__body` (+0.27) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (50/50 syn, 890/942 real) | `canon_masked` | 0.9017 | 96.17% | 73.29% | `word__20` (+0.40), `word__20 off` (+0.34), `word__chhar` (+0.28), `word__body` (+0.26) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (50/50 syn, 890/942 real) | `brand_masked` | 0.8901 | 96.17% | 74.0% | `word__20` (+0.40), `word__20 off` (+0.34), `word__chhar` (+0.28), `word__body` (+0.26) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (50/50 syn, 890/942 real) | `brand_stripped` | 0.8885 | 96.06% | 73.63% | `word__20` (+0.40), `word__20 off` (+0.34), `word__chhar` (+0.28), `word__body` (+0.26) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (50/50 syn, 890/942 real) | `surface_8feat` | 0.4782 | 47.87% | 36.62% | `has_url` (+0.18), `word_len` (+0.10), `has_bn` (+0.07), `has_phone` (+0.00) |

### 3d. Counterfactual URL/Phone/Brand Probes (`n = 1279` Real Smishing Rows)

| Text View | Baseline Smish % | Contact Swap Smish % (`Delta pp`, `Delta P`) | Contact+Brand Swap Smish % (`Delta pp`, `Delta P`) |
| :--- | ---: | :--- | :--- |
| `raw` | 100.0% | 98.36% (-1.64 pp, -0.0796) | 98.44% (-1.56 pp, -0.0824) |
| `canon_masked` | 100.0% | 100.0% (+0.00 pp, +0.0001) | 100.0% (+0.00 pp, -0.0012) |
| `brand_masked` | 100.0% | 100.0% (+0.00 pp, +0.0001) | 100.0% (+0.00 pp, +0.0001) |
| `brand_stripped` | 100.0% | 100.0% (+0.00 pp, +0.0001) | 100.0% (+0.00 pp, +0.0001) |

### 3e. CodeMix Structural Diversity & Entity-Frequency Gates (`mode=new_batch`)

| Gate / Metric | Observed Value | Threshold | Gate Passed |
| :--- | :--- | :--- | :--- |
| `brandless_rate` | `0.62` | `>= 0.3` | `True` |
| `codemix_bn_script_en` | `1.0` | `>= 0.8` | `True` |
| `codemix_hook_plus_pure_en` | `0.0` | `== 0.0` | `True` |
| `codemix_intra_sentential_mix` | `1.0` | `>= 0.6` | `True` |
| `max_brand_count` | `3` | `<= 5` | `True` |
| `promo_sent1_excl_rate` | `0.08` | `<= 0.35` | `True` |
| `promo_sent1_quest_rate` | `0.0` | `<= 0.05` | `True` |
| `ussd_1234_share` | `0.0` | `<= 0.25` | `True` |
- **All diversity gates passed:** `True`

## 4. Updated Class Counts & Remaining Stage B Deficits

Target `T = 1548`:

| Class | `curation_v2` Original Eligible | Accepted Pilot | Combined Total | Target (`T`) | Remaining Stage B Deficit |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 1509 | 0 | 1509 | 1548 | **39** |
| `Promotional` | 942 | 50 | 992 | 1548 | **556** |
| `Smishing` | 1548 | 0 | 1548 | 1548 | **0** |
| **Total** | **3999** | **50** | **4049** | — | **595** |

## 5. Contamination, Review Status & Synthetic-Origin Cue / Limitation Notes

- **Anti-leakage & isolation:** Zero exact or near-duplicate (`cos >= 0.85` raw or slot-template) overlap with `curation_v2/existing_unique.parquet` (including `smishdetect_validation`, `smishdetect_test`, and `banglaphish_ood`) or `curation_v2/quarantined.parquet`.
- **Cross-lingual family grouping:** Accepted rows span **50** `family_id` groups (including **0** linked cross-lingual parallel family groups so `StratifiedGroupKFold` keeps parallel realizations in the same fold).
- **Independent review status:** **50** accepted rows are confirmed by both automated screening and independent review (`review_shard_1.jsonl`, `review_shard_2.jsonl`, `review_replacements.jsonl`); **11** rows flagged by independent review are isolated in `rejected.jsonl`; **0** rows are `pending_independent_review`.
- **Synthetic-origin cue & limitation notes:**
  - Reserved domains (`example.com`, `example.org`, `example.net`, `example.edu.bd`), fictional phone numbers (`01700-0000XX` / `+88017000000XX`), and recurring fictional brand names appear only in pilot `Normal` and `Promotional` rows (`0` in `eligible_train` `Smishing`), creating potential lexical synthetic-origin cues not tested by `compute_surface_diagnostics` (which only measures length, script, URL/phone/USSD presence).
  - As noted during independent review, `CodeMix` promotional rows frequently share a two-sentence `[Bengali/Banglish hook] + [English clause]` discourse pattern; downstream experiments must evaluate on real held-out and `BanglaPhish-2026` OOD sets and never treat synthetic-origin patterns as real-world prevalence.
- **Remaining deficit:** Remaining `Promotional` deficit (`345` rows) has **not** been generated pending final pilot sign-off.
