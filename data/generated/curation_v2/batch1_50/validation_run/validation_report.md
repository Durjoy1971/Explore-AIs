# Stage B Pilot Screening & Validation Report (`curation_v2`)

Generated reproducibly by `scripts/validate_stage_b_pilot.py` (`src/bangla_smish/stage_b_pilot.py`). Screening enforces schema, `length_band` consistency (`short(<50)`, `medium(50-120)`, `long(>120)`), `is_synthetic=True`, exact & char-ngram cosine (`>= 0.85` raw & slot-template) isolation against `curation_v2/existing_unique.parquet` (including held-out/OOD) and `quarantined.parquet`, intra-pilot template diversity, fictional entities/reserved domains, label/language rules, and independent review log ingestion.

## 1. Candidate Screening & Independent Review Summary

- **Total candidates screened (`candidates.jsonl`):** **52**
- **Accepted (`accepted.jsonl`):** **50** (96.15%)
  - Confirmed by **both** automated screening and independent review (`review_status="accepted"`): **50**
  - Passed automated screening, pending independent review (`review_status="pending_independent_review"`): **0**
- **Rejected (`rejected.jsonl`):** **2**
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
| `independent_review:inconsistent_script_punct` | 1 |
| `independent_review:topic_overlap` | 1 |
| `independent_review:unnatural_phrasing` | 2 |

- **Rejected IDs (2):** `syn_b1_0020`, `syn_b1_0026`

## 2. Domain, Length & Cue Coverage of Accepted Pilot (`N = 50`)

### Domain Breakdown (`Domain × Label`)

| Domain | Normal | Promotional | Total |
| :--- | ---: | ---: | ---: |
| `banking_mfs` | 0 | 5 | **5** |
| `civic_utility_health` | 0 | 1 | **1** |
| `delivery_parcel` | 0 | 3 | **3** |
| `education` | 0 | 1 | **1** |
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
| Surface-only 5-fold Accuracy | 73.14% | 72.56% | -0.58% |
| Surface-only 5-fold Macro-F1 | 62.86% | 63.11% | +0.25% |
| Length-only Smishing ROC-AUC | 0.7693 | 0.7679 | -0.0014 |
| Length-only Promo vs. Normal ROC-AUC | 0.7414 | 0.7415 | +0.0001 |

### Per-Class Surface Cue Rates (`Before -> After Pilot`)

| Class | Count (`Before -> After`) | Median Chars (`Before -> After`) | % Has URL (`Before -> After`) | % Has Phone (`Before -> After`) |
| :--- | :--- | :--- | :--- | :--- |
| `Normal` | 1509 -> 1509 | 47.0 -> 47.0 | 0.73% -> 0.73% | 0.73% -> 0.73% |
| `Promotional` | 942 -> 992 | 74.0 -> 74.0 | 29.83% -> 29.74% | 8.92% -> 8.87% |
| `Smishing` | 1548 -> 1548 | 99.0 -> 99.0 | 59.43% -> 59.43% | 27.45% -> 27.45% |

### 3b. 4-View 3-Class TF-IDF+LR Probes (`Before -> After Pilot`)

| Text View | Macro-F1 (`Before -> After`) | Accuracy (`Before -> After`) | Smishing Recall (`Before -> After`) | Recall @ <=1% FPR (`Before -> After`) |
| :--- | :--- | :--- | :--- | :--- |
| `raw` | 98.01% -> 98.15% | 98.1% -> 98.25% | 98.77% -> 99.03% | 98.51% -> 98.64% |
| `canon_masked` | 97.9% -> 97.67% | 98.0% -> 97.75% | 98.58% -> 98.39% | 97.67% -> 97.74% |
| `brand_masked` | 97.76% -> 97.67% | 97.85% -> 97.73% | 98.39% -> 98.26% | 97.67% -> 97.42% |
| `brand_stripped` | 97.81% -> 97.64% | 97.9% -> 97.73% | 98.45% -> 98.39% | 97.87% -> 97.87% |

### 3c. Synthetic-Origin (`is_synthetic`) Separability Probes

| Comparison | Strata / N | Feature View | ROC-AUC | Accuracy | Macro-F1 | Top Positive Synthetic Features |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| `all_vs_real` | 50 syn vs 3999 real | `raw` | 0.9812 | 99.33% | 81.83% | `word__ar` (+3.88), `word__example` (+3.86), `word__chhar` (+2.81), `word__matro` (+2.47) |
| `all_vs_real` | 50 syn vs 3999 real | `canon_masked` | 0.9769 | 99.09% | 72.16% | `word__ar` (+4.24), `word__chhar` (+3.08), `char__ ও ` (+2.68), `word__সব` (+2.51) |
| `all_vs_real` | 50 syn vs 3999 real | `brand_masked` | 0.9713 | 98.81% | 69.7% | `word__brand` (+4.27), `word__ar` (+4.05), `word__chhar` (+3.00), `char__ ও ` (+2.75) |
| `all_vs_real` | 50 syn vs 3999 real | `brand_stripped` | 0.9635 | 99.09% | 72.96% | `word__ar` (+4.56), `word__chhar` (+3.65), `char__ ও ` (+3.07), `word__সব` (+2.68) |
| `all_vs_real` | 50 syn vs 3999 real | `surface_8feat` | 0.6025 | 66.21% | 41.46% | `char_len` (+2.06), `has_ussd` (+0.34), `has_bn` (+0.15), `has_redacted_tag` (+0.00) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `raw` | 0.9988 | 99.0% | 90.44% | `word__ar` (+2.97), `word__example` (+2.77), `word__২৫` (+2.38), `word__সব` (+2.32) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `canon_masked` | 0.9981 | 99.0% | 90.44% | `word__ar` (+3.16), `word__সব` (+2.47), `word__chhar` (+2.42), `word__২৫` (+2.23) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `brand_masked` | 0.9987 | 99.44% | 95.02% | `word__brand` (+3.78), `word__ar` (+2.90), `word__সব` (+2.55), `word__chhar` (+2.23) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `brand_stripped` | 0.9977 | 99.06% | 91.14% | `word__ar` (+3.35), `word__chhar` (+2.68), `word__সব` (+2.64), `word__২৫` (+2.19) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `surface_8feat` | 0.7782 | 83.1% | 54.86% | `char_len` (+0.68), `has_ussd` (+0.24), `has_bn` (+0.07), `has_redacted_tag` (+0.00) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `raw` | 0.9417 | 97.28% | 81.29% | `word__ar` (+3.16), `word__example` (+2.49), `word__matro` (+1.76), `char__ ও ` (+1.70) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `canon_masked` | 0.9338 | 96.47% | 74.44% | `word__ar` (+3.41), `char__ ও ` (+1.97), `word__chhar` (+1.81), `word__রব` (+1.79) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `brand_masked` | 0.9283 | 96.07% | 72.27% | `word__ar` (+3.41), `word__brand` (+2.92), `char__ ও ` (+2.12), `word__chhar` (+1.90) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `brand_stripped` | 0.9101 | 95.97% | 69.54% | `word__ar` (+3.66), `char__ ও ` (+2.34), `word__chhar` (+2.28), `word__সব` (+1.80) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `surface_8feat` | 0.4791 | 61.9% | 42.51% | `char_len` (+0.55), `has_bn` (+0.08), `has_ussd` (+0.06), `has_redacted_tag` (+0.00) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 23 strata (47/50 syn, 702/942 real) | `raw` | 0.9242 | 96.26% | 78.43% | `word__example` (+0.83), `word__ar` (+0.71), `word__https example` (+0.54), `word__example com` (+0.50) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 23 strata (47/50 syn, 702/942 real) | `canon_masked` | 0.8794 | 94.93% | 69.89% | `word__ar` (+0.77), `word__সব` (+0.54), `word__at` (+0.47), `word__রত` (+0.46) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 23 strata (47/50 syn, 702/942 real) | `brand_masked` | 0.8787 | 94.53% | 70.48% | `word__ar` (+0.79), `word__সব` (+0.59), `word__brand` (+0.57), `word__at` (+0.49) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 23 strata (47/50 syn, 702/942 real) | `brand_stripped` | 0.8396 | 94.53% | 67.97% | `word__ar` (+0.81), `word__সব` (+0.58), `word__at` (+0.49), `word__রত` (+0.45) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 23 strata (47/50 syn, 702/942 real) | `surface_8feat` | 0.4591 | 54.87% | 41.02% | `has_url` (+0.36), `word_len` (+0.09), `has_phone` (+0.05), `has_redacted_tag` (+0.00) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (49/50 syn, 922/942 real) | `raw` | 0.9171 | 96.4% | 71.29% | `word__example` (+0.57), `word__https example` (+0.42), `word__example com` (+0.30), `word__ar` (+0.29) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (49/50 syn, 922/942 real) | `canon_masked` | 0.9048 | 95.26% | 56.19% | `word__ar` (+0.33), `word__কছ` (+0.24), `word__at` (+0.23), `word__সব` (+0.23) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (49/50 syn, 922/942 real) | `brand_masked` | 0.9207 | 95.26% | 59.12% | `word__ar` (+0.33), `word__কছ` (+0.24), `word__at` (+0.23), `word__brand` (+0.23) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (49/50 syn, 922/942 real) | `brand_stripped` | 0.8818 | 95.06% | 54.28% | `word__ar` (+0.34), `word__at` (+0.24), `word__কছ` (+0.24), `word__সব` (+0.23) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (49/50 syn, 922/942 real) | `surface_8feat` | 0.4968 | 55.41% | 40.0% | `has_url` (+0.48), `has_phone` (+0.23), `has_bn` (+0.11), `has_redacted_tag` (+0.00) |

### 3d. Counterfactual URL/Phone/Brand Probes (`n = 1279` Real Smishing Rows)

| Text View | Baseline Smish % | Contact Swap Smish % (`Delta pp`, `Delta P`) | Contact+Brand Swap Smish % (`Delta pp`, `Delta P`) |
| :--- | ---: | :--- | :--- |
| `raw` | 100.0% | 97.73% (-2.27 pp, -0.0896) | 97.73% (-2.27 pp, -0.0930) |
| `canon_masked` | 100.0% | 100.0% (+0.00 pp, +0.0001) | 100.0% (+0.00 pp, -0.0014) |
| `brand_masked` | 100.0% | 100.0% (+0.00 pp, +0.0001) | 100.0% (+0.00 pp, +0.0001) |
| `brand_stripped` | 100.0% | 100.0% (+0.00 pp, +0.0001) | 100.0% (+0.00 pp, +0.0001) |

### 3e. CodeMix Structural Diversity & Entity-Frequency Gates (`mode=new_batch`)

| Gate / Metric | Observed Value | Threshold | Gate Passed |
| :--- | :--- | :--- | :--- |
| `brandless_rate` | `0.36` | `>= 0.3` | `True` |
| `codemix_bn_script_en` | `1.0` | `>= 0.8` | `True` |
| `codemix_hook_plus_pure_en` | `0.0` | `== 0.0` | `True` |
| `codemix_intra_sentential_mix` | `1.0` | `>= 0.6` | `True` |
| `max_brand_count` | `4` | `<= 5` | `True` |
| `promo_sent1_excl_rate` | `0.1` | `<= 0.35` | `True` |
| `promo_sent1_quest_rate` | `0.02` | `<= 0.05` | `True` |
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
- **Independent review status:** **50** accepted rows are confirmed by both automated screening and independent review (`review_shard_1.jsonl`, `review_shard_2.jsonl`, `review_replacements.jsonl`); **2** rows flagged by independent review are isolated in `rejected.jsonl`; **0** rows are `pending_independent_review`.
- **Synthetic-origin cue & limitation notes:**
  - Reserved domains (`example.com`, `example.org`, `example.net`, `example.edu.bd`), fictional phone numbers (`01700-0000XX` / `+88017000000XX`), and recurring fictional brand names appear only in pilot `Normal` and `Promotional` rows (`0` in `eligible_train` `Smishing`), creating potential lexical synthetic-origin cues not tested by `compute_surface_diagnostics` (which only measures length, script, URL/phone/USSD presence).
  - As noted during independent review, `CodeMix` promotional rows frequently share a two-sentence `[Bengali/Banglish hook] + [English clause]` discourse pattern; downstream experiments must evaluate on real held-out and `BanglaPhish-2026` OOD sets and never treat synthetic-origin patterns as real-world prevalence.
- **Remaining deficit:** Remaining `Promotional` deficit (`345` rows) has **not** been generated pending final pilot sign-off.
