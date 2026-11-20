# Stage B Pilot Screening & Validation Report (`curation_v2`)

Generated reproducibly by `scripts/validate_stage_b_pilot.py` (`src/bangla_smish/stage_b_pilot.py`). Screening enforces schema, `length_band` consistency (`short(<50)`, `medium(50-120)`, `long(>120)`), `is_synthetic=True`, exact & char-ngram cosine (`>= 0.85` raw & slot-template) isolation against `curation_v2/existing_unique.parquet` (including held-out/OOD) and `quarantined.parquet`, intra-pilot template diversity, fictional entities/reserved domains, label/language rules, and independent review log ingestion.

## 1. Candidate Screening & Independent Review Summary

- **Total candidates screened (`candidates.jsonl`):** **51**
- **Accepted (`accepted.jsonl`):** **45** (88.24%)
  - Confirmed by **both** automated screening and independent review (`review_status="accepted"`): **45**
  - Passed automated screening, pending independent review (`review_status="pending_independent_review"`): **0**
- **Rejected (`rejected.jsonl`):** **6**
- **Exact pilot quota match (`300` rows):** `False`

### Accepted Pilot Distribution (`Label × Language`)

| Class | Bengali | Banglish | English | CodeMix | Total Accepted | Target Pilot Quota |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 0 | 0 | 0 | 0 | **0** | 39 |
| `Promotional` | 12 | 17 | 4 | 12 | **45** | 261 |
| `Smishing` | 0 | 0 | 0 | 0 | **0** | 0 |

### Rejection Breakdown

| Rejection Reason | Count |
| :--- | ---: |
| `independent_review:cross_batch_scenario_clone` | 3 |
| `independent_review:domain_mismatch` | 1 |
| `independent_review:intra_batch_scenario_clone` | 1 |
| `independent_review:unnatural_phrasing` | 2 |

- **Rejected IDs (6):** `syn_b7_0004`, `syn_b7_0020`, `syn_b7_0025`, `syn_b7_0026`, `syn_b7_0032`, `syn_b7_0040`

## 2. Domain, Length & Cue Coverage of Accepted Pilot (`N = 45`)

### Domain Breakdown (`Domain × Label`)

| Domain | Normal | Promotional | Total |
| :--- | ---: | ---: | ---: |
| `banking_mfs` | 0 | 4 | **4** |
| `civic_utility_health` | 0 | 1 | **1** |
| `delivery_parcel` | 0 | 3 | **3** |
| `education` | 0 | 1 | **1** |
| `personal_or_general` | 0 | 10 | **10** |
| `shopping_ecommerce` | 0 | 13 | **13** |
| `telecom` | 0 | 13 | **13** |
| **Total** | **0** | **45** | **45** |

### Length Band & Surface Cue Coverage

| Class | `short(<50)` | `medium(50-120)` | `long(>120)` | Has Link | Has Urgency |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 0 | 0 | 0 | 0 | 0 |
| `Promotional` | 1 | 37 | 7 | 13 | 6 |

## 3. Surface-Only & Length-Only Shortcut Diagnostics (Before vs. After Pilot)

| Metric (`StratifiedGroupKFold`, grouped by `family_id`) | `eligible_train` Alone | `eligible_train + accepted` | Delta |
| :--- | ---: | ---: | ---: |
| Row count (`N`) | 3999 | 4044 | +45 |
| Surface-only 5-fold Accuracy | 73.14% | 72.35% | -0.79% |
| Surface-only 5-fold Macro-F1 | 62.86% | 62.53% | -0.33% |
| Length-only Smishing ROC-AUC | 0.7693 | 0.768 | -0.0013 |
| Length-only Promo vs. Normal ROC-AUC | 0.7414 | 0.7411 | -0.0003 |

### Per-Class Surface Cue Rates (`Before -> After Pilot`)

| Class | Count (`Before -> After`) | Median Chars (`Before -> After`) | % Has URL (`Before -> After`) | % Has Phone (`Before -> After`) |
| :--- | :--- | :--- | :--- | :--- |
| `Normal` | 1509 -> 1509 | 47.0 -> 47.0 | 0.73% -> 0.73% | 0.73% -> 0.73% |
| `Promotional` | 942 -> 987 | 74.0 -> 74.0 | 29.83% -> 29.79% | 8.92% -> 8.92% |
| `Smishing` | 1548 -> 1548 | 99.0 -> 99.0 | 59.43% -> 59.43% | 27.45% -> 27.45% |

### 3b. 4-View 3-Class TF-IDF+LR Probes (`Before -> After Pilot`)

| Text View | Macro-F1 (`Before -> After`) | Accuracy (`Before -> After`) | Smishing Recall (`Before -> After`) | Recall @ <=1% FPR (`Before -> After`) |
| :--- | :--- | :--- | :--- | :--- |
| `raw` | 98.01% -> 98.15% | 98.1% -> 98.22% | 98.77% -> 98.77% | 98.51% -> 98.19% |
| `canon_masked` | 97.86% -> 97.58% | 97.95% -> 97.68% | 98.45% -> 98.51% | 97.8% -> 97.67% |
| `brand_masked` | 97.76% -> 97.61% | 97.85% -> 97.7% | 98.39% -> 98.45% | 97.8% -> 97.74% |
| `brand_stripped` | 97.71% -> 97.66% | 97.8% -> 97.75% | 98.39% -> 98.45% | 97.8% -> 97.67% |

### 3c. Synthetic-Origin (`is_synthetic`) Separability Probes

| Comparison | Strata / N | Feature View | ROC-AUC | Accuracy | Macro-F1 | Top Positive Synthetic Features |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| `all_vs_real` | 45 syn vs 3999 real | `raw` | 0.9850 | 99.51% | 85.59% | `word__example` (+4.36), `word__চলছ` (+3.43), `word__chhar` (+3.09), `word__paben` (+3.01) |
| `all_vs_real` | 45 syn vs 3999 real | `canon_masked` | 0.9855 | 99.28% | 77.51% | `word__চলছ` (+3.56), `word__paben` (+3.37), `char__ ও ` (+3.25), `word__20` (+3.21) |
| `all_vs_real` | 45 syn vs 3999 real | `brand_masked` | 0.9798 | 99.33% | 78.4% | `word__চলছ` (+3.65), `char__ ও ` (+3.37), `word__paben` (+3.28), `word__20` (+3.16) |
| `all_vs_real` | 45 syn vs 3999 real | `brand_stripped` | 0.9833 | 99.31% | 77.95% | `word__চলছ` (+3.62), `char__ ও ` (+3.38), `word__paben` (+3.28), `word__20` (+3.14) |
| `all_vs_real` | 45 syn vs 3999 real | `surface_8feat` | 0.4265 | 63.67% | 39.79% | `word_len` (+0.69), `has_ussd` (+0.26), `has_url` (+0.10), `has_bn` (+0.10) |
| `synth_vs_real_smishing` | 45 syn vs 1548 real | `raw` | 0.9983 | 98.87% | 87.55% | `word__example` (+2.98), `word__চলছ` (+2.46), `word__chhar` (+2.29), `word__paben` (+2.27) |
| `synth_vs_real_smishing` | 45 syn vs 1548 real | `canon_masked` | 0.9967 | 98.93% | 88.69% | `word__ussd` (+2.65), `word__paben` (+2.50), `word__চলছ` (+2.47), `word__20` (+2.47) |
| `synth_vs_real_smishing` | 45 syn vs 1548 real | `brand_masked` | 0.9967 | 99.18% | 91.77% | `word__ussd` (+2.65), `word__চলছ` (+2.51), `word__paben` (+2.51), `word__20` (+2.45) |
| `synth_vs_real_smishing` | 45 syn vs 1548 real | `brand_stripped` | 0.9967 | 99.0% | 89.49% | `word__ussd` (+2.77), `word__চলছ` (+2.65), `word__paben` (+2.56), `word__20` (+2.45) |
| `synth_vs_real_smishing` | 45 syn vs 1548 real | `surface_8feat` | 0.7958 | 81.23% | 52.84% | `word_len` (+1.89), `has_ussd` (+0.18), `has_bn` (+0.04), `has_redacted_tag` (+0.00) |
| `synthetic_promo_vs_real_promo` | 45 syn vs 942 real | `raw` | 0.9869 | 98.07% | 86.12% | `word__example` (+2.79), `word__চলছ` (+2.46), `char__ ও ` (+2.04), `word__paben` (+1.98) |
| `synthetic_promo_vs_real_promo` | 45 syn vs 942 real | `canon_masked` | 0.9766 | 97.47% | 80.69% | `word__চলছ` (+2.55), `char__ ও ` (+2.39), `word__paben` (+2.34), `word__20` (+2.02) |
| `synthetic_promo_vs_real_promo` | 45 syn vs 942 real | `brand_masked` | 0.9711 | 97.37% | 79.62% | `word__চলছ` (+2.59), `char__ ও ` (+2.47), `word__paben` (+2.33), `word__20` (+1.99) |
| `synthetic_promo_vs_real_promo` | 45 syn vs 942 real | `brand_stripped` | 0.9724 | 97.37% | 79.62% | `word__চলছ` (+2.59), `char__ ও ` (+2.47), `word__paben` (+2.30), `word__20` (+1.98) |
| `synthetic_promo_vs_real_promo` | 45 syn vs 942 real | `surface_8feat` | 0.4532 | 55.52% | 38.43% | `word_len` (+0.92), `has_url` (+0.16), `has_bn` (+0.09), `has_phone` (+0.03) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 27 strata (45/45 syn, 803/942 real) | `raw` | 0.9675 | 97.52% | 84.14% | `word__example` (+0.99), `word__চলছ` (+0.84), `char__ ও ` (+0.62), `word__chhar` (+0.57) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 27 strata (45/45 syn, 803/942 real) | `canon_masked` | 0.9530 | 96.23% | 76.16% | `word__চলছ` (+0.88), `char__ ও ` (+0.68), `word__20` (+0.61), `word__paben` (+0.60) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 27 strata (45/45 syn, 803/942 real) | `brand_masked` | 0.9445 | 96.11% | 75.75% | `word__চলছ` (+0.89), `char__ ও ` (+0.69), `word__20` (+0.61), `word__paben` (+0.60) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 27 strata (45/45 syn, 803/942 real) | `brand_stripped` | 0.9460 | 96.11% | 75.75% | `word__চলছ` (+0.89), `char__ ও ` (+0.69), `word__20` (+0.61), `word__paben` (+0.60) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 27 strata (45/45 syn, 803/942 real) | `surface_8feat` | 0.4656 | 50.71% | 37.31% | `word_len` (+0.45), `has_url` (+0.27), `has_phone` (+0.14), `has_bn` (+0.01) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (45/45 syn, 891/942 real) | `raw` | 0.9299 | 97.44% | 81.15% | `word__চলছ` (+0.49), `word__example` (+0.46), `word__১৫` (+0.31), `word__নত` (+0.29) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (45/45 syn, 891/942 real) | `canon_masked` | 0.8894 | 96.69% | 74.54% | `word__চলছ` (+0.50), `word__১৫` (+0.30), `word__নত` (+0.30), `char__ ও ` (+0.28) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (45/45 syn, 891/942 real) | `brand_masked` | 0.8774 | 96.69% | 74.54% | `word__চলছ` (+0.50), `word__১৫` (+0.30), `word__নত` (+0.29), `char__ ও ` (+0.28) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (45/45 syn, 891/942 real) | `brand_stripped` | 0.8767 | 96.69% | 74.54% | `word__চলছ` (+0.50), `word__১৫` (+0.30), `word__নত` (+0.29), `char__ ও ` (+0.28) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (45/45 syn, 891/942 real) | `surface_8feat` | 0.4279 | 52.46% | 37.92% | `has_url` (+0.21), `word_len` (+0.19), `has_phone` (+0.08), `has_bn` (+0.08) |

### 3d. Counterfactual URL/Phone/Brand Probes (`n = 1279` Real Smishing Rows)

| Text View | Baseline Smish % | Contact Swap Smish % (`Delta pp`, `Delta P`) | Contact+Brand Swap Smish % (`Delta pp`, `Delta P`) |
| :--- | ---: | :--- | :--- |
| `raw` | 100.0% | 98.44% (-1.56 pp, -0.0781) | 98.44% (-1.56 pp, -0.0807) |
| `canon_masked` | 100.0% | 100.0% (+0.00 pp, +0.0001) | 100.0% (+0.00 pp, -0.0011) |
| `brand_masked` | 100.0% | 100.0% (+0.00 pp, +0.0001) | 100.0% (+0.00 pp, +0.0001) |
| `brand_stripped` | 100.0% | 100.0% (+0.00 pp, +0.0001) | 100.0% (+0.00 pp, +0.0001) |

### 3e. CodeMix Structural Diversity & Entity-Frequency Gates (`mode=new_batch`)

| Gate / Metric | Observed Value | Threshold | Gate Passed |
| :--- | :--- | :--- | :--- |
| `brandless_rate` | `0.6444` | `>= 0.3` | `True` |
| `codemix_bn_script_en` | `1.0` | `>= 0.8` | `True` |
| `codemix_hook_plus_pure_en` | `0.0` | `== 0.0` | `True` |
| `codemix_intra_sentential_mix` | `1.0` | `>= 0.6` | `True` |
| `max_brand_count` | `3` | `<= 5` | `True` |
| `promo_sent1_excl_rate` | `0.0444` | `<= 0.35` | `True` |
| `promo_sent1_quest_rate` | `0.0` | `<= 0.05` | `True` |
| `ussd_1234_share` | `0.0` | `<= 0.25` | `True` |
- **All diversity gates passed:** `True`

## 4. Updated Class Counts & Remaining Stage B Deficits

Target `T = 1548`:

| Class | `curation_v2` Original Eligible | Accepted Pilot | Combined Total | Target (`T`) | Remaining Stage B Deficit |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 1509 | 0 | 1509 | 1548 | **39** |
| `Promotional` | 942 | 45 | 987 | 1548 | **561** |
| `Smishing` | 1548 | 0 | 1548 | 1548 | **0** |
| **Total** | **3999** | **45** | **4044** | — | **600** |

## 5. Contamination, Review Status & Synthetic-Origin Cue / Limitation Notes

- **Anti-leakage & isolation:** Zero exact or near-duplicate (`cos >= 0.85` raw or slot-template) overlap with `curation_v2/existing_unique.parquet` (including `smishdetect_validation`, `smishdetect_test`, and `banglaphish_ood`) or `curation_v2/quarantined.parquet`.
- **Cross-lingual family grouping:** Accepted rows span **45** `family_id` groups (including **0** linked cross-lingual parallel family groups so `StratifiedGroupKFold` keeps parallel realizations in the same fold).
- **Independent review status:** **45** accepted rows are confirmed by both automated screening and independent review (`review_shard_1.jsonl`, `review_shard_2.jsonl`, `review_replacements.jsonl`); **6** rows flagged by independent review are isolated in `rejected.jsonl`; **0** rows are `pending_independent_review`.
- **Synthetic-origin cue & limitation notes:**
  - Reserved domains (`example.com`, `example.org`, `example.net`, `example.edu.bd`), fictional phone numbers (`01700-0000XX` / `+88017000000XX`), and recurring fictional brand names appear only in pilot `Normal` and `Promotional` rows (`0` in `eligible_train` `Smishing`), creating potential lexical synthetic-origin cues not tested by `compute_surface_diagnostics` (which only measures length, script, URL/phone/USSD presence).
  - As noted during independent review, `CodeMix` promotional rows frequently share a two-sentence `[Bengali/Banglish hook] + [English clause]` discourse pattern; downstream experiments must evaluate on real held-out and `BanglaPhish-2026` OOD sets and never treat synthetic-origin patterns as real-world prevalence.
- **Remaining deficit:** Remaining `Promotional` deficit (`345` rows) has **not** been generated pending final pilot sign-off.
