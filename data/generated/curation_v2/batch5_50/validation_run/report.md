# Stage B Pilot Screening & Validation Report (`curation_v2`)

Generated reproducibly by `scripts/validate_stage_b_pilot.py` (`src/bangla_smish/stage_b_pilot.py`). Screening enforces schema, `length_band` consistency (`short(<50)`, `medium(50-120)`, `long(>120)`), `is_synthetic=True`, exact & char-ngram cosine (`>= 0.85` raw & slot-template) isolation against `curation_v2/existing_unique.parquet` (including held-out/OOD) and `quarantined.parquet`, intra-pilot template diversity, fictional entities/reserved domains, label/language rules, and independent review log ingestion.

## 1. Candidate Screening & Independent Review Summary

- **Total candidates screened (`candidates.jsonl`):** **58**
- **Accepted (`accepted.jsonl`):** **50** (86.21%)
  - Confirmed by **both** automated screening and independent review (`review_status="accepted"`): **50**
  - Passed automated screening, pending independent review (`review_status="pending_independent_review"`): **0**
- **Rejected (`rejected.jsonl`):** **8**
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
| `banglish_appears_pure_english` | 1 |
| `independent_review:cross_batch_scenario_clone` | 2 |
| `independent_review:domain_mismatch` | 1 |
| `independent_review:metadata_mismatch` | 3 |
| `independent_review:unnatural_phrasing` | 2 |
| `length_band_mismatch:long(>120)!=medium(50-120)` | 1 |
| `length_band_mismatch:medium(50-120)!=short(<50)` | 1 |

- **Rejected IDs (8):** `syn_b5_0001`, `syn_b5_0003`, `syn_b5_0012`, `syn_b5_0016`, `syn_b5_0028`, `syn_b5_0039`, `syn_b5_0047`, `syn_b5_0049`

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
| `Promotional` | 1 | 41 | 8 | 15 | 5 |

## 3. Surface-Only & Length-Only Shortcut Diagnostics (Before vs. After Pilot)

| Metric (`StratifiedGroupKFold`, grouped by `family_id`) | `eligible_train` Alone | `eligible_train + accepted` | Delta |
| :--- | ---: | ---: | ---: |
| Row count (`N`) | 3999 | 4049 | +50 |
| Surface-only 5-fold Accuracy | 73.14% | 72.56% | -0.58% |
| Surface-only 5-fold Macro-F1 | 62.86% | 62.97% | +0.11% |
| Length-only Smishing ROC-AUC | 0.7693 | 0.7677 | -0.0016 |
| Length-only Promo vs. Normal ROC-AUC | 0.7414 | 0.7413 | -0.0001 |

### Per-Class Surface Cue Rates (`Before -> After Pilot`)

| Class | Count (`Before -> After`) | Median Chars (`Before -> After`) | % Has URL (`Before -> After`) | % Has Phone (`Before -> After`) |
| :--- | :--- | :--- | :--- | :--- |
| `Normal` | 1509 -> 1509 | 47.0 -> 47.0 | 0.73% -> 0.73% | 0.73% -> 0.73% |
| `Promotional` | 942 -> 992 | 74.0 -> 74.0 | 29.83% -> 29.84% | 8.92% -> 8.87% |
| `Smishing` | 1548 -> 1548 | 99.0 -> 99.0 | 59.43% -> 59.43% | 27.45% -> 27.45% |

### 3b. 4-View 3-Class TF-IDF+LR Probes (`Before -> After Pilot`)

| Text View | Macro-F1 (`Before -> After`) | Accuracy (`Before -> After`) | Smishing Recall (`Before -> After`) | Recall @ <=1% FPR (`Before -> After`) |
| :--- | :--- | :--- | :--- | :--- |
| `raw` | 98.01% -> 98.15% | 98.1% -> 98.25% | 98.77% -> 98.97% | 98.51% -> 98.84% |
| `canon_masked` | 97.86% -> 97.77% | 97.95% -> 97.85% | 98.45% -> 98.51% | 97.8% -> 97.74% |
| `brand_masked` | 97.76% -> 97.66% | 97.85% -> 97.73% | 98.39% -> 98.39% | 97.8% -> 97.61% |
| `brand_stripped` | 97.71% -> 97.66% | 97.8% -> 97.75% | 98.39% -> 98.45% | 97.8% -> 97.74% |

### 3c. Synthetic-Origin (`is_synthetic`) Separability Probes

| Comparison | Strata / N | Feature View | ROC-AUC | Accuracy | Macro-F1 | Top Positive Synthetic Features |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| `all_vs_real` | 50 syn vs 3999 real | `raw` | 0.9921 | 99.51% | 87.38% | `word__example` (+4.62), `word__chhar` (+3.20), `word__20` (+2.73), `word__১৫` (+2.52) |
| `all_vs_real` | 50 syn vs 3999 real | `canon_masked` | 0.9815 | 99.26% | 78.98% | `word__chhar` (+3.44), `word__20` (+3.17), `word__১৫` (+2.75), `word__off` (+2.59) |
| `all_vs_real` | 50 syn vs 3999 real | `brand_masked` | 0.9783 | 99.18% | 78.37% | `word__chhar` (+3.42), `word__20` (+3.37), `word__১৫` (+2.72), `word__off` (+2.59) |
| `all_vs_real` | 50 syn vs 3999 real | `brand_stripped` | 0.9794 | 99.23% | 78.57% | `word__chhar` (+3.48), `word__20` (+3.44), `word__১৫` (+2.62), `word__off` (+2.62) |
| `all_vs_real` | 50 syn vs 3999 real | `surface_8feat` | 0.4965 | 62.93% | 39.61% | `word_len` (+0.67), `has_ussd` (+0.28), `has_bn` (+0.15), `has_url` (+0.14) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `raw` | 0.9980 | 99.25% | 92.99% | `word__example` (+3.27), `word__chhar` (+2.52), `word__20` (+2.27), `word__pack` (+2.22) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `canon_masked` | 0.9980 | 99.12% | 92.0% | `word__off` (+2.63), `word__chhar` (+2.62), `word__20` (+2.44), `word__১৫` (+2.28) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `brand_masked` | 0.9977 | 99.37% | 94.4% | `word__chhar` (+2.54), `word__off` (+2.50), `word__20` (+2.40), `word__১৫` (+2.33) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `brand_stripped` | 0.9980 | 99.19% | 92.65% | `word__chhar` (+2.64), `word__off` (+2.59), `word__20` (+2.59), `word__ussd` (+2.26) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `surface_8feat` | 0.8047 | 81.41% | 53.4% | `word_len` (+1.40), `has_ussd` (+0.23), `has_bn` (+0.14), `has_redacted_tag` (+0.00) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `raw` | 0.9702 | 98.08% | 87.77% | `word__example` (+3.06), `word__chhar` (+1.97), `word__20` (+1.77), `word__takay` (+1.54) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `canon_masked` | 0.9384 | 96.88% | 79.57% | `word__chhar` (+2.10), `word__20` (+2.04), `word__20 off` (+1.75), `word__যট` (+1.73) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `brand_masked` | 0.9299 | 96.88% | 79.06% | `word__20` (+2.22), `word__chhar` (+2.14), `word__20 off` (+1.75), `word__te` (+1.75) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `brand_stripped` | 0.9310 | 97.08% | 80.88% | `word__20` (+2.23), `word__chhar` (+2.15), `word__te` (+1.75), `word__20 off` (+1.75) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `surface_8feat` | 0.5032 | 54.44% | 38.75% | `word_len` (+0.97), `has_url` (+0.19), `has_bn` (+0.14), `has_phone` (+0.02) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 22 strata (47/50 syn, 688/942 real) | `raw` | 0.9495 | 96.87% | 82.98% | `word__example` (+0.74), `word__chhar` (+0.59), `word__কল` (+0.51), `word__চলছ` (+0.47) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 22 strata (47/50 syn, 688/942 real) | `canon_masked` | 0.9003 | 95.65% | 74.62% | `word__chhar` (+0.61), `word__কল` (+0.50), `word__20` (+0.47), `word__২০ চলছ` (+0.47) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 22 strata (47/50 syn, 688/942 real) | `brand_masked` | 0.8881 | 95.65% | 74.62% | `word__chhar` (+0.61), `word__কল` (+0.53), `word__20` (+0.51), `word__চলছ` (+0.49) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 22 strata (47/50 syn, 688/942 real) | `brand_stripped` | 0.8902 | 95.65% | 75.33% | `word__chhar` (+0.62), `word__কল` (+0.54), `word__20` (+0.52), `word__চলছ` (+0.50) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 22 strata (47/50 syn, 688/942 real) | `surface_8feat` | 0.4117 | 58.78% | 39.89% | `word_len` (+0.29), `has_phone` (+0.16), `has_bn` (+0.09), `has_url` (+0.07) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 9 strata (50/50 syn, 931/942 real) | `raw` | 0.9397 | 96.53% | 74.82% | `word__example` (+0.45), `word__৪০` (+0.35), `word__পর` (+0.29), `word__https example` (+0.27) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 9 strata (50/50 syn, 931/942 real) | `canon_masked` | 0.8840 | 94.9% | 59.62% | `word__৪০` (+0.34), `word__পর` (+0.29), `word__পর যন` (+0.26), `word__যন` (+0.25) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 9 strata (50/50 syn, 931/942 real) | `brand_masked` | 0.8756 | 95.01% | 61.02% | `word__৪০` (+0.34), `word__পর` (+0.29), `word__পর যন` (+0.26), `word__যন` (+0.25) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 9 strata (50/50 syn, 931/942 real) | `brand_stripped` | 0.8764 | 95.01% | 61.02% | `word__৪০` (+0.34), `word__পর` (+0.29), `word__পর যন` (+0.26), `word__যন` (+0.25) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 9 strata (50/50 syn, 931/942 real) | `surface_8feat` | 0.4048 | 54.43% | 38.44% | `word_len` (+0.17), `has_url` (+0.13), `has_bn` (+0.06), `has_phone` (+0.01) |

### 3d. Counterfactual URL/Phone/Brand Probes (`n = 1279` Real Smishing Rows)

| Text View | Baseline Smish % | Contact Swap Smish % (`Delta pp`, `Delta P`) | Contact+Brand Swap Smish % (`Delta pp`, `Delta P`) |
| :--- | ---: | :--- | :--- |
| `raw` | 100.0% | 98.2% (-1.80 pp, -0.0824) | 98.28% (-1.72 pp, -0.0853) |
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
- **Independent review status:** **50** accepted rows are confirmed by both automated screening and independent review (`review_shard_1.jsonl`, `review_shard_2.jsonl`, `review_replacements.jsonl`); **8** rows flagged by independent review are isolated in `rejected.jsonl`; **0** rows are `pending_independent_review`.
- **Synthetic-origin cue & limitation notes:**
  - Reserved domains (`example.com`, `example.org`, `example.net`, `example.edu.bd`), fictional phone numbers (`01700-0000XX` / `+88017000000XX`), and recurring fictional brand names appear only in pilot `Normal` and `Promotional` rows (`0` in `eligible_train` `Smishing`), creating potential lexical synthetic-origin cues not tested by `compute_surface_diagnostics` (which only measures length, script, URL/phone/USSD presence).
  - As noted during independent review, `CodeMix` promotional rows frequently share a two-sentence `[Bengali/Banglish hook] + [English clause]` discourse pattern; downstream experiments must evaluate on real held-out and `BanglaPhish-2026` OOD sets and never treat synthetic-origin patterns as real-world prevalence.
- **Remaining deficit:** Remaining `Promotional` deficit (`345` rows) has **not** been generated pending final pilot sign-off.
