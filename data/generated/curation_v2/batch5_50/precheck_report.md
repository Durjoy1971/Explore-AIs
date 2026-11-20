# Stage B Pilot Screening & Validation Report (`curation_v2`)

Generated reproducibly by `scripts/validate_stage_b_pilot.py` (`src/bangla_smish/stage_b_pilot.py`). Screening enforces schema, `length_band` consistency (`short(<50)`, `medium(50-120)`, `long(>120)`), `is_synthetic=True`, exact & char-ngram cosine (`>= 0.85` raw & slot-template) isolation against `curation_v2/existing_unique.parquet` (including held-out/OOD) and `quarantined.parquet`, intra-pilot template diversity, fictional entities/reserved domains, label/language rules, and independent review log ingestion.

## 1. Candidate Screening & Independent Review Summary

- **Total candidates screened (`candidates.jsonl`):** **50**
- **Accepted (`accepted.jsonl`):** **47** (94.0%)
  - Confirmed by **both** automated screening and independent review (`review_status="accepted"`): **0**
  - Passed automated screening, pending independent review (`review_status="pending_independent_review"`): **47**
- **Rejected (`rejected.jsonl`):** **3**
- **Exact pilot quota match (`300` rows):** `False`

### Accepted Pilot Distribution (`Label × Language`)

| Class | Bengali | Banglish | English | CodeMix | Total Accepted | Target Pilot Quota |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 0 | 0 | 0 | 0 | **0** | 39 |
| `Promotional` | 13 | 16 | 4 | 14 | **47** | 261 |
| `Smishing` | 0 | 0 | 0 | 0 | **0** | 0 |

### Rejection Breakdown

| Rejection Reason | Count |
| :--- | ---: |
| `banglish_appears_pure_english` | 1 |
| `length_band_mismatch:long(>120)!=medium(50-120)` | 1 |
| `length_band_mismatch:medium(50-120)!=short(<50)` | 1 |

- **Rejected IDs (3):** `syn_b5_0039`, `syn_b5_0047`, `syn_b5_0049`

## 2. Domain, Length & Cue Coverage of Accepted Pilot (`N = 47`)

### Domain Breakdown (`Domain × Label`)

| Domain | Normal | Promotional | Total |
| :--- | ---: | ---: | ---: |
| `banking_mfs` | 0 | 4 | **4** |
| `civic_utility_health` | 0 | 1 | **1** |
| `delivery_parcel` | 0 | 3 | **3** |
| `education` | 0 | 1 | **1** |
| `personal_or_general` | 0 | 10 | **10** |
| `shopping_ecommerce` | 0 | 13 | **13** |
| `telecom` | 0 | 15 | **15** |
| **Total** | **0** | **47** | **47** |

### Length Band & Surface Cue Coverage

| Class | `short(<50)` | `medium(50-120)` | `long(>120)` | Has Link | Has Urgency |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 0 | 0 | 0 | 0 | 0 |
| `Promotional` | 1 | 39 | 7 | 13 | 5 |

## 3. Surface-Only & Length-Only Shortcut Diagnostics (Before vs. After Pilot)

| Metric (`StratifiedGroupKFold`, grouped by `family_id`) | `eligible_train` Alone | `eligible_train + accepted` | Delta |
| :--- | ---: | ---: | ---: |
| Row count (`N`) | 3999 | 4046 | +47 |
| Surface-only 5-fold Accuracy | 73.14% | 72.54% | -0.60% |
| Surface-only 5-fold Macro-F1 | 62.86% | 62.82% | -0.04% |
| Length-only Smishing ROC-AUC | 0.7693 | 0.7678 | -0.0015 |
| Length-only Promo vs. Normal ROC-AUC | 0.7414 | 0.7414 | +0.0000 |

### Per-Class Surface Cue Rates (`Before -> After Pilot`)

| Class | Count (`Before -> After`) | Median Chars (`Before -> After`) | % Has URL (`Before -> After`) | % Has Phone (`Before -> After`) |
| :--- | :--- | :--- | :--- | :--- |
| `Normal` | 1509 -> 1509 | 47.0 -> 47.0 | 0.73% -> 0.73% | 0.73% -> 0.73% |
| `Promotional` | 942 -> 989 | 74.0 -> 74.0 | 29.83% -> 29.83% | 8.92% -> 8.8% |
| `Smishing` | 1548 -> 1548 | 99.0 -> 99.0 | 59.43% -> 59.43% | 27.45% -> 27.45% |

### 3e. CodeMix Structural Diversity & Entity-Frequency Gates (`mode=new_batch`)

| Gate / Metric | Observed Value | Threshold | Gate Passed |
| :--- | :--- | :--- | :--- |
| `brandless_rate` | `0.6596` | `>= 0.3` | `True` |
| `codemix_bn_script_en` | `1.0` | `>= 0.8` | `True` |
| `codemix_hook_plus_pure_en` | `0.0` | `== 0.0` | `True` |
| `codemix_intra_sentential_mix` | `1.0` | `>= 0.6` | `True` |
| `max_brand_count` | `3` | `<= 5` | `True` |
| `promo_sent1_excl_rate` | `0.1064` | `<= 0.35` | `True` |
| `promo_sent1_quest_rate` | `0.0` | `<= 0.05` | `True` |
| `ussd_1234_share` | `0.0` | `<= 0.25` | `True` |
- **All diversity gates passed:** `True`

## 4. Updated Class Counts & Remaining Stage B Deficits

Target `T = 1548`:

| Class | `curation_v2` Original Eligible | Accepted Pilot | Combined Total | Target (`T`) | Remaining Stage B Deficit |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 1509 | 0 | 1509 | 1548 | **39** |
| `Promotional` | 942 | 47 | 989 | 1548 | **559** |
| `Smishing` | 1548 | 0 | 1548 | 1548 | **0** |
| **Total** | **3999** | **47** | **4046** | — | **598** |

## 5. Contamination, Review Status & Synthetic-Origin Cue / Limitation Notes

- **Anti-leakage & isolation:** Zero exact or near-duplicate (`cos >= 0.85` raw or slot-template) overlap with `curation_v2/existing_unique.parquet` (including `smishdetect_validation`, `smishdetect_test`, and `banglaphish_ood`) or `curation_v2/quarantined.parquet`.
- **Cross-lingual family grouping:** Accepted rows span **47** `family_id` groups (including **0** linked cross-lingual parallel family groups so `StratifiedGroupKFold` keeps parallel realizations in the same fold).
- **Independent review status:** **0** accepted rows are confirmed by both automated screening and independent review (`review_shard_1.jsonl`, `review_shard_2.jsonl`, `review_replacements.jsonl`); **3** rows flagged by independent review are isolated in `rejected.jsonl`; **47** rows are `pending_independent_review`.
- **Synthetic-origin cue & limitation notes:**
  - Reserved domains (`example.com`, `example.org`, `example.net`, `example.edu.bd`), fictional phone numbers (`01700-0000XX` / `+88017000000XX`), and recurring fictional brand names appear only in pilot `Normal` and `Promotional` rows (`0` in `eligible_train` `Smishing`), creating potential lexical synthetic-origin cues not tested by `compute_surface_diagnostics` (which only measures length, script, URL/phone/USSD presence).
  - As noted during independent review, `CodeMix` promotional rows frequently share a two-sentence `[Bengali/Banglish hook] + [English clause]` discourse pattern; downstream experiments must evaluate on real held-out and `BanglaPhish-2026` OOD sets and never treat synthetic-origin patterns as real-world prevalence.
- **Remaining deficit:** Remaining `Promotional` deficit (`345` rows) has **not** been generated pending final pilot sign-off.
