# Stage B Pilot Screening & Validation Report (`curation_v2`)

Generated reproducibly by `scripts/validate_stage_b_pilot.py` (`src/bangla_smish/stage_b_pilot.py`). Screening enforces schema, `length_band` consistency (`short(<50)`, `medium(50-120)`, `long(>120)`), `is_synthetic=True`, exact & char-ngram cosine (`>= 0.85` raw & slot-template) isolation against `curation_v2/existing_unique.parquet` (including held-out/OOD) and `quarantined.parquet`, intra-pilot template diversity, fictional entities/reserved domains, label/language rules, and independent review log ingestion.

## 1. Candidate Screening & Independent Review Summary

- **Total candidates screened (`candidates.jsonl`):** **63**
- **Accepted (`accepted.jsonl`):** **50** (79.37%)
  - Confirmed by **both** automated screening and independent review (`review_status="accepted"`): **50**
  - Passed automated screening, pending independent review (`review_status="pending_independent_review"`): **0**
- **Rejected (`rejected.jsonl`):** **13**
- **Exact pilot quota match (`300` rows):** `False`

### Accepted Pilot Distribution (`Label × Language`)

| Class | Bengali | Banglish | English | CodeMix | Total Accepted | Target Pilot Quota |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 0 | 0 | 0 | 0 | **0** | 39 |
| `Promotional` | 13 | 19 | 4 | 14 | **50** | 261 |
| `Smishing` | 0 | 0 | 0 | 0 | **0** | 0 |

### Rejection Breakdown

| Rejection Reason | Count |
| :--- | ---: |
| `independent_review:cross_batch_scenario_clone` | 2 |
| `independent_review:near_duplicate_paraphrase` | 1 |
| `independent_review:near_duplicate_slot_template` | 2 |
| `independent_review:shallow_template_clone` | 1 |
| `independent_review:unnatural_phrasing` | 11 |

- **Rejected IDs (13):** `syn_b4_0002`, `syn_b4_0010`, `syn_b4_0014`, `syn_b4_0019`, `syn_b4_0023`, `syn_b4_0037`, `syn_b4_0039`, `syn_b4_0044`, `syn_b4_0045`, `syn_b4_0049`, `syn_b4_r_0004`, `syn_b4_r_0008`, `syn_b4_r_0010`

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
| `Promotional` | 1 | 41 | 8 | 14 | 4 |

## 3. Surface-Only & Length-Only Shortcut Diagnostics (Before vs. After Pilot)

| Metric (`StratifiedGroupKFold`, grouped by `family_id`) | `eligible_train` Alone | `eligible_train + accepted` | Delta |
| :--- | ---: | ---: | ---: |
| Row count (`N`) | 3999 | 4049 | +50 |
| Surface-only 5-fold Accuracy | 73.14% | 72.49% | -0.65% |
| Surface-only 5-fold Macro-F1 | 62.86% | 63.02% | +0.16% |
| Length-only Smishing ROC-AUC | 0.7693 | 0.7642 | -0.0051 |
| Length-only Promo vs. Normal ROC-AUC | 0.7414 | 0.7453 | +0.0039 |

### Per-Class Surface Cue Rates (`Before -> After Pilot`)

| Class | Count (`Before -> After`) | Median Chars (`Before -> After`) | % Has URL (`Before -> After`) | % Has Phone (`Before -> After`) |
| :--- | :--- | :--- | :--- | :--- |
| `Normal` | 1509 -> 1509 | 47.0 -> 47.0 | 0.73% -> 0.73% | 0.73% -> 0.73% |
| `Promotional` | 942 -> 992 | 74.0 -> 75.0 | 29.83% -> 29.74% | 8.92% -> 8.87% |
| `Smishing` | 1548 -> 1548 | 99.0 -> 99.0 | 59.43% -> 59.43% | 27.45% -> 27.45% |

### 3b. 4-View 3-Class TF-IDF+LR Probes (`Before -> After Pilot`)

| Text View | Macro-F1 (`Before -> After`) | Accuracy (`Before -> After`) | Smishing Recall (`Before -> After`) | Recall @ <=1% FPR (`Before -> After`) |
| :--- | :--- | :--- | :--- | :--- |
| `raw` | 98.01% -> 98.09% | 98.1% -> 98.2% | 98.77% -> 99.03% | 98.51% -> 98.64% |
| `canon_masked` | 97.9% -> 97.75% | 98.0% -> 97.83% | 98.58% -> 98.51% | 97.67% -> 97.55% |
| `brand_masked` | 97.76% -> 97.68% | 97.85% -> 97.75% | 98.39% -> 98.32% | 97.67% -> 97.48% |
| `brand_stripped` | 97.81% -> 97.64% | 97.9% -> 97.73% | 98.45% -> 98.45% | 97.87% -> 97.35% |

### 3c. Synthetic-Origin (`is_synthetic`) Separability Probes

| Comparison | Strata / N | Feature View | ROC-AUC | Accuracy | Macro-F1 | Top Positive Synthetic Features |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| `all_vs_real` | 50 syn vs 3999 real | `raw` | 0.9959 | 99.38% | 83.18% | `word__example` (+4.16), `word__paben` (+3.58), `word__কছ` (+3.33), `word__ar` (+3.32) |
| `all_vs_real` | 50 syn vs 3999 real | `canon_masked` | 0.9822 | 99.11% | 73.31% | `word__কছ` (+3.79), `word__paben` (+3.59), `word__ar` (+3.47), `char__ ও ` (+2.71) |
| `all_vs_real` | 50 syn vs 3999 real | `brand_masked` | 0.9798 | 99.14% | 75.13% | `word__কছ` (+3.81), `word__paben` (+3.56), `word__ar` (+3.42), `char__ ও ` (+2.71) |
| `all_vs_real` | 50 syn vs 3999 real | `brand_stripped` | 0.9813 | 99.14% | 75.13% | `word__কছ` (+3.88), `word__ar` (+3.58), `word__paben` (+3.57), `char__ ও ` (+2.75) |
| `all_vs_real` | 50 syn vs 3999 real | `surface_8feat` | 0.5950 | 66.41% | 41.28% | `char_len` (+1.12), `has_ussd` (+0.19), `has_bn` (+0.10), `has_redacted_tag` (+0.00) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `raw` | 0.9993 | 99.37% | 94.28% | `word__example` (+2.87), `word__paben` (+2.65), `word__ar` (+2.32), `word__কছ` (+2.31) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `canon_masked` | 0.9976 | 99.19% | 92.49% | `word__discount` (+2.67), `word__paben` (+2.53), `word__কছ` (+2.51), `word__ar` (+2.30) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `brand_masked` | 0.9974 | 99.19% | 92.65% | `word__discount` (+2.66), `word__paben` (+2.52), `word__কছ` (+2.49), `word__ar` (+2.29) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `brand_stripped` | 0.9975 | 99.19% | 92.49% | `word__discount` (+2.73), `word__কছ` (+2.51), `word__paben` (+2.49), `word__ar` (+2.33) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `surface_8feat` | 0.7374 | 80.48% | 52.98% | `has_ussd` (+0.18), `has_redacted_tag` (+0.00), `has_bn` (-0.01), `char_len` (-0.12) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `raw` | 0.9875 | 97.88% | 86.16% | `word__example` (+2.71), `word__ar` (+2.68), `word__paben` (+2.38), `word__কছ` (+2.35) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `canon_masked` | 0.9524 | 97.58% | 83.58% | `word__ar` (+2.84), `word__কছ` (+2.61), `word__paben` (+2.45), `char__ ও ` (+1.93) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `brand_masked` | 0.9445 | 97.48% | 83.11% | `word__ar` (+2.88), `word__কছ` (+2.68), `word__paben` (+2.42), `char__ ও ` (+1.98) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `brand_stripped` | 0.9488 | 97.48% | 83.11% | `word__ar` (+2.90), `word__কছ` (+2.69), `word__paben` (+2.41), `char__ ও ` (+1.98) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `surface_8feat` | 0.6380 | 68.35% | 46.95% | `char_len` (+0.40), `word_len` (+0.27), `has_bn` (+0.08), `has_redacted_tag` (+0.00) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 24 strata (47/50 syn, 749/942 real) | `raw` | 0.9795 | 96.86% | 82.51% | `word__example` (+0.93), `word__https example` (+0.54), `word__কছ` (+0.54), `word__paben` (+0.51) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 24 strata (47/50 syn, 749/942 real) | `canon_masked` | 0.9358 | 96.36% | 80.21% | `word__কছ` (+0.61), `word__paben` (+0.50), `word__রথম` (+0.46), `word__25` (+0.45) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 24 strata (47/50 syn, 749/942 real) | `brand_masked` | 0.9280 | 95.98% | 78.94% | `word__কছ` (+0.62), `word__paben` (+0.49), `word__রথম` (+0.46), `word__25` (+0.45) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 24 strata (47/50 syn, 749/942 real) | `brand_stripped` | 0.9323 | 96.23% | 79.78% | `word__কছ` (+0.63), `word__paben` (+0.49), `word__রথম` (+0.46), `word__25` (+0.46) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 24 strata (47/50 syn, 749/942 real) | `surface_8feat` | 0.6075 | 69.6% | 48.43% | `char_len` (+0.32), `has_url` (+0.29), `has_bn` (+0.18), `has_phone` (+0.07) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (49/50 syn, 922/942 real) | `raw` | 0.9568 | 96.29% | 70.01% | `word__example` (+0.64), `word__https example` (+0.48), `word__org` (+0.38), `word__example org` (+0.38) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (49/50 syn, 922/942 real) | `canon_masked` | 0.9089 | 95.16% | 54.42% | `word__বল url` (+0.24), `word__শর বল` (+0.24), `word__করল` (+0.23), `word__at url` (+0.22) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (49/50 syn, 922/942 real) | `brand_masked` | 0.8996 | 94.95% | 54.16% | `word__শর বল` (+0.24), `word__বল url` (+0.24), `word__করল` (+0.23), `word__কছ` (+0.23) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (49/50 syn, 922/942 real) | `brand_stripped` | 0.9024 | 94.95% | 54.16% | `word__শর বল` (+0.24), `word__বল url` (+0.24), `word__করল` (+0.23), `word__কছ` (+0.23) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (49/50 syn, 922/942 real) | `surface_8feat` | 0.5045 | 65.19% | 44.49% | `has_url` (+0.66), `has_bn` (+0.15), `word_len` (+0.01), `has_redacted_tag` (+0.00) |

### 3d. Counterfactual URL/Phone/Brand Probes (`n = 1279` Real Smishing Rows)

| Text View | Baseline Smish % | Contact Swap Smish % (`Delta pp`, `Delta P`) | Contact+Brand Swap Smish % (`Delta pp`, `Delta P`) |
| :--- | ---: | :--- | :--- |
| `raw` | 100.0% | 98.44% (-1.56 pp, -0.0738) | 98.44% (-1.56 pp, -0.0766) |
| `canon_masked` | 100.0% | 100.0% (+0.00 pp, +0.0001) | 100.0% (+0.00 pp, -0.0012) |
| `brand_masked` | 100.0% | 100.0% (+0.00 pp, +0.0001) | 100.0% (+0.00 pp, +0.0001) |
| `brand_stripped` | 100.0% | 100.0% (+0.00 pp, +0.0001) | 100.0% (+0.00 pp, +0.0001) |

### 3e. CodeMix Structural Diversity & Entity-Frequency Gates (`mode=new_batch`)

| Gate / Metric | Observed Value | Threshold | Gate Passed |
| :--- | :--- | :--- | :--- |
| `brandless_rate` | `0.62` | `>= 0.3` | `True` |
| `codemix_bn_script_en` | `1.0` | `>= 0.8` | `True` |
| `codemix_hook_plus_pure_en` | `0.0` | `== 0.0` | `True` |
| `codemix_intra_sentential_mix` | `0.9286` | `>= 0.6` | `True` |
| `max_brand_count` | `3` | `<= 5` | `True` |
| `promo_sent1_excl_rate` | `0.02` | `<= 0.35` | `True` |
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
- **Independent review status:** **50** accepted rows are confirmed by both automated screening and independent review (`review_shard_1.jsonl`, `review_shard_2.jsonl`, `review_replacements.jsonl`); **13** rows flagged by independent review are isolated in `rejected.jsonl`; **0** rows are `pending_independent_review`.
- **Synthetic-origin cue & limitation notes:**
  - Reserved domains (`example.com`, `example.org`, `example.net`, `example.edu.bd`), fictional phone numbers (`01700-0000XX` / `+88017000000XX`), and recurring fictional brand names appear only in pilot `Normal` and `Promotional` rows (`0` in `eligible_train` `Smishing`), creating potential lexical synthetic-origin cues not tested by `compute_surface_diagnostics` (which only measures length, script, URL/phone/USSD presence).
  - As noted during independent review, `CodeMix` promotional rows frequently share a two-sentence `[Bengali/Banglish hook] + [English clause]` discourse pattern; downstream experiments must evaluate on real held-out and `BanglaPhish-2026` OOD sets and never treat synthetic-origin patterns as real-world prevalence.
- **Remaining deficit:** Remaining `Promotional` deficit (`345` rows) has **not** been generated pending final pilot sign-off.
