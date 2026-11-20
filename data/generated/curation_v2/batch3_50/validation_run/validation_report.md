# Stage B Pilot Screening & Validation Report (`curation_v2`)

Generated reproducibly by `scripts/validate_stage_b_pilot.py` (`src/bangla_smish/stage_b_pilot.py`). Screening enforces schema, `length_band` consistency (`short(<50)`, `medium(50-120)`, `long(>120)`), `is_synthetic=True`, exact & char-ngram cosine (`>= 0.85` raw & slot-template) isolation against `curation_v2/existing_unique.parquet` (including held-out/OOD) and `quarantined.parquet`, intra-pilot template diversity, fictional entities/reserved domains, label/language rules, and independent review log ingestion.

## 1. Candidate Screening & Independent Review Summary

- **Total candidates screened (`candidates.jsonl`):** **68**
- **Accepted (`accepted.jsonl`):** **50** (73.53%)
  - Confirmed by **both** automated screening and independent review (`review_status="accepted"`): **50**
  - Passed automated screening, pending independent review (`review_status="pending_independent_review"`): **0**
- **Rejected (`rejected.jsonl`):** **18**
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
| `independent_review:cross_batch_scenario_clone` | 6 |
| `independent_review:domain_mismatch` | 1 |
| `independent_review:metadata_mismatch` | 1 |
| `independent_review:near_duplicate_paraphrase` | 1 |
| `independent_review:near_duplicate_slot_template` | 2 |
| `independent_review:shallow_template_clone` | 5 |
| `independent_review:unnatural_phrasing` | 10 |
| `independent_review:unrealistic_ussd_digits` | 2 |
| `independent_review:weak_promotional_cue` | 1 |

- **Rejected IDs (18):** `syn_b3_0002`, `syn_b3_0005`, `syn_b3_0006`, `syn_b3_0012`, `syn_b3_0018`, `syn_b3_0020`, `syn_b3_0021`, `syn_b3_0022`, `syn_b3_0023`, `syn_b3_0029`, `syn_b3_0035`, `syn_b3_0036`, `syn_b3_0037`, `syn_b3_0038`, `syn_b3_0046`, `syn_b3_0047`, `syn_b3_0049`, `syn_b3_r_0010`

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
| `Promotional` | 1 | 41 | 8 | 15 | 4 |

## 3. Surface-Only & Length-Only Shortcut Diagnostics (Before vs. After Pilot)

| Metric (`StratifiedGroupKFold`, grouped by `family_id`) | `eligible_train` Alone | `eligible_train + accepted` | Delta |
| :--- | ---: | ---: | ---: |
| Row count (`N`) | 3999 | 4049 | +50 |
| Surface-only 5-fold Accuracy | 73.14% | 72.51% | -0.63% |
| Surface-only 5-fold Macro-F1 | 62.86% | 63.04% | +0.18% |
| Length-only Smishing ROC-AUC | 0.7693 | 0.7675 | -0.0018 |
| Length-only Promo vs. Normal ROC-AUC | 0.7414 | 0.742 | +0.0006 |

### Per-Class Surface Cue Rates (`Before -> After Pilot`)

| Class | Count (`Before -> After`) | Median Chars (`Before -> After`) | % Has URL (`Before -> After`) | % Has Phone (`Before -> After`) |
| :--- | :--- | :--- | :--- | :--- |
| `Normal` | 1509 -> 1509 | 47.0 -> 47.0 | 0.73% -> 0.73% | 0.73% -> 0.73% |
| `Promotional` | 942 -> 992 | 74.0 -> 75.0 | 29.83% -> 29.84% | 8.92% -> 8.87% |
| `Smishing` | 1548 -> 1548 | 99.0 -> 99.0 | 59.43% -> 59.43% | 27.45% -> 27.45% |

### 3b. 4-View 3-Class TF-IDF+LR Probes (`Before -> After Pilot`)

| Text View | Macro-F1 (`Before -> After`) | Accuracy (`Before -> After`) | Smishing Recall (`Before -> After`) | Recall @ <=1% FPR (`Before -> After`) |
| :--- | :--- | :--- | :--- | :--- |
| `raw` | 98.01% -> 98.09% | 98.1% -> 98.2% | 98.77% -> 99.03% | 98.51% -> 98.77% |
| `canon_masked` | 97.9% -> 97.59% | 98.0% -> 97.68% | 98.58% -> 98.32% | 97.67% -> 97.35% |
| `brand_masked` | 97.76% -> 97.58% | 97.85% -> 97.65% | 98.39% -> 98.32% | 97.67% -> 97.42% |
| `brand_stripped` | 97.81% -> 97.48% | 97.9% -> 97.58% | 98.45% -> 98.32% | 97.87% -> 97.8% |

### 3c. Synthetic-Origin (`is_synthetic`) Separability Probes

| Comparison | Strata / N | Feature View | ROC-AUC | Accuracy | Macro-F1 | Top Positive Synthetic Features |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| `all_vs_real` | 50 syn vs 3999 real | `raw` | 0.9830 | 99.26% | 80.08% | `word__example` (+4.57), `word__kinle` (+2.98), `word__২৫` (+2.96), `word__ইন` (+2.88) |
| `all_vs_real` | 50 syn vs 3999 real | `canon_masked` | 0.9767 | 99.04% | 70.65% | `word__kinle` (+3.07), `word__২৫` (+3.02), `word__ইন` (+2.83), `word__কম` (+2.40) |
| `all_vs_real` | 50 syn vs 3999 real | `brand_masked` | 0.9757 | 98.99% | 70.03% | `word__২৫` (+3.07), `word__ইন` (+2.93), `word__kinle` (+2.92), `word__brand` (+2.62) |
| `all_vs_real` | 50 syn vs 3999 real | `brand_stripped` | 0.9709 | 99.04% | 71.5% | `word__kinle` (+3.18), `word__২৫` (+3.03), `word__ইন` (+2.86), `word__কম` (+2.49) |
| `all_vs_real` | 50 syn vs 3999 real | `surface_8feat` | 0.5941 | 64.49% | 40.55% | `char_len` (+1.38), `has_ussd` (+0.28), `has_bn` (+0.08), `has_redacted_tag` (+0.00) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `raw` | 0.9979 | 98.87% | 89.0% | `word__example` (+3.20), `word__২৫` (+2.43), `word__ইন` (+2.37), `word__kinle` (+2.23) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `canon_masked` | 0.9970 | 98.75% | 88.05% | `word__ইন` (+2.35), `word__kinle` (+2.23), `word__ussd` (+2.15), `word__২৫` (+2.08) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `brand_masked` | 0.9974 | 98.56% | 86.41% | `word__ইন` (+2.42), `word__brand` (+2.29), `word__ussd` (+2.19), `word__২৫` (+2.06) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `brand_stripped` | 0.9964 | 98.75% | 88.05% | `word__ইন` (+2.38), `word__kinle` (+2.27), `word__ussd` (+2.21), `word__২৫` (+2.10) |
| `synth_vs_real_smishing` | 50 syn vs 1548 real | `surface_8feat` | 0.7633 | 83.1% | 54.62% | `has_ussd` (+0.20), `has_bn` (+0.00), `has_redacted_tag` (+0.00), `word_len` (-0.10) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `raw` | 0.9566 | 97.18% | 80.85% | `word__example` (+2.89), `word__ইন` (+2.05), `word__kinle` (+1.97), `word__net` (+1.90) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `canon_masked` | 0.9358 | 96.77% | 76.31% | `word__২৫` (+2.05), `word__kinle` (+2.01), `word__ইন` (+1.99), `word__ei` (+1.55) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `brand_masked` | 0.9284 | 96.77% | 76.31% | `word__ইন` (+2.08), `word__kinle` (+2.04), `word__২৫` (+1.99), `word__ei` (+1.57) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `brand_stripped` | 0.9225 | 96.77% | 76.31% | `word__kinle` (+2.14), `word__ইন` (+2.05), `word__২৫` (+1.94), `word__ei` (+1.57) |
| `synthetic_promo_vs_real_promo` | 50 syn vs 942 real | `surface_8feat` | 0.4958 | 68.85% | 45.54% | `char_len` (+0.18), `has_url` (+0.06), `has_bn` (+0.04), `has_redacted_tag` (+0.00) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 27 strata (50/50 syn, 778/942 real) | `raw` | 0.9221 | 96.86% | 82.07% | `word__example` (+0.91), `word__example com` (+0.57), `word__ইন` (+0.56), `word__https example` (+0.50) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 27 strata (50/50 syn, 778/942 real) | `canon_masked` | 0.8821 | 94.44% | 67.47% | `word__২৫` (+0.64), `word__ইন` (+0.59), `word__prothom` (+0.42), `word__ei` (+0.40) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 27 strata (50/50 syn, 778/942 real) | `brand_masked` | 0.8598 | 94.08% | 65.78% | `word__২৫` (+0.64), `word__ইন` (+0.60), `word__prothom` (+0.46), `word__parcel` (+0.42) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 27 strata (50/50 syn, 778/942 real) | `brand_stripped` | 0.8600 | 94.2% | 66.05% | `word__২৫` (+0.64), `word__ইন` (+0.60), `word__prothom` (+0.46), `word__parcel` (+0.42) |
| `synthetic_promo_vs_real_promo_stratum_3way` | 27 strata (50/50 syn, 778/942 real) | `surface_8feat` | 0.4235 | 45.17% | 34.51% | `has_url` (+0.24), `has_phone` (+0.03), `has_redacted_tag` (+0.00), `char_len` (-0.05) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (50/50 syn, 891/942 real) | `raw` | 0.9392 | 96.39% | 73.31% | `word__19` (+0.46), `word__example` (+0.42), `word__২৫` (+0.38), `word__https example` (+0.30) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (50/50 syn, 891/942 real) | `canon_masked` | 0.8790 | 95.01% | 61.41% | `word__২৫` (+0.52), `word__ussd` (+0.26), `word__ইন` (+0.23), `word__indoor` (+0.20) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (50/50 syn, 891/942 real) | `brand_masked` | 0.8642 | 94.58% | 58.12% | `word__২৫` (+0.52), `word__ussd` (+0.26), `word__ইন` (+0.23), `word__indoor` (+0.20) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (50/50 syn, 891/942 real) | `brand_stripped` | 0.8690 | 94.69% | 59.56% | `word__২৫` (+0.52), `word__ussd` (+0.26), `word__ইন` (+0.23), `word__indoor` (+0.20) |
| `synthetic_promo_vs_real_promo_stratum_2way` | 8 strata (50/50 syn, 891/942 real) | `surface_8feat` | 0.4810 | 56.85% | 40.71% | `has_url` (+0.27), `has_ussd` (+0.11), `has_bn` (+0.09), `word_len` (+0.02) |

### 3d. Counterfactual URL/Phone/Brand Probes (`n = 1279` Real Smishing Rows)

| Text View | Baseline Smish % | Contact Swap Smish % (`Delta pp`, `Delta P`) | Contact+Brand Swap Smish % (`Delta pp`, `Delta P`) |
| :--- | ---: | :--- | :--- |
| `raw` | 100.0% | 96.72% (-3.28 pp, -0.0992) | 96.72% (-3.28 pp, -0.1022) |
| `canon_masked` | 100.0% | 100.0% (+0.00 pp, +0.0001) | 100.0% (+0.00 pp, -0.0012) |
| `brand_masked` | 100.0% | 100.0% (+0.00 pp, +0.0001) | 100.0% (+0.00 pp, +0.0001) |
| `brand_stripped` | 100.0% | 100.0% (+0.00 pp, +0.0001) | 100.0% (+0.00 pp, +0.0001) |

### 3e. CodeMix Structural Diversity & Entity-Frequency Gates (`mode=new_batch`)

| Gate / Metric | Observed Value | Threshold | Gate Passed |
| :--- | :--- | :--- | :--- |
| `brandless_rate` | `0.62` | `>= 0.3` | `True` |
| `codemix_bn_script_en` | `1.0` | `>= 0.8` | `True` |
| `codemix_hook_plus_pure_en` | `0.0` | `== 0.0` | `True` |
| `codemix_intra_sentential_mix` | `0.8462` | `>= 0.6` | `True` |
| `max_brand_count` | `3` | `<= 5` | `True` |
| `promo_sent1_excl_rate` | `0.0` | `<= 0.35` | `True` |
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
- **Independent review status:** **50** accepted rows are confirmed by both automated screening and independent review (`review_shard_1.jsonl`, `review_shard_2.jsonl`, `review_replacements.jsonl`); **18** rows flagged by independent review are isolated in `rejected.jsonl`; **0** rows are `pending_independent_review`.
- **Synthetic-origin cue & limitation notes:**
  - Reserved domains (`example.com`, `example.org`, `example.net`, `example.edu.bd`), fictional phone numbers (`01700-0000XX` / `+88017000000XX`), and recurring fictional brand names appear only in pilot `Normal` and `Promotional` rows (`0` in `eligible_train` `Smishing`), creating potential lexical synthetic-origin cues not tested by `compute_surface_diagnostics` (which only measures length, script, URL/phone/USSD presence).
  - As noted during independent review, `CodeMix` promotional rows frequently share a two-sentence `[Bengali/Banglish hook] + [English clause]` discourse pattern; downstream experiments must evaluate on real held-out and `BanglaPhish-2026` OOD sets and never treat synthetic-origin patterns as real-world prevalence.
- **Remaining deficit:** Remaining `Promotional` deficit (`345` rows) has **not** been generated pending final pilot sign-off.
