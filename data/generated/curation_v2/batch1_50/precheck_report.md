# Stage B Pilot Screening & Validation Report (`curation_v2`)

Generated reproducibly by `scripts/validate_stage_b_pilot.py` (`src/bangla_smish/stage_b_pilot.py`). Screening enforces schema, `length_band` consistency (`short(<50)`, `medium(50-120)`, `long(>120)`), `is_synthetic=True`, exact & char-ngram cosine (`>= 0.85` raw & slot-template) isolation against `curation_v2/existing_unique.parquet` (including held-out/OOD) and `quarantined.parquet`, intra-pilot template diversity, fictional entities/reserved domains, label/language rules, and independent review log ingestion.

## 1. Candidate Screening & Independent Review Summary

- **Total candidates screened (`candidates.jsonl`):** **52**
- **Accepted (`accepted.jsonl`):** **50** (96.15%)
  - Confirmed by **both** automated screening and independent review (`review_status="accepted"`): **48**
  - Passed automated screening, pending independent review (`review_status="pending_independent_review"`): **2**
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
- **Independent review status:** **48** accepted rows are confirmed by both automated screening and independent review (`review_shard_1.jsonl`, `review_shard_2.jsonl`, `review_replacements.jsonl`); **2** rows flagged by independent review are isolated in `rejected.jsonl`; **2** rows are `pending_independent_review`.
- **Synthetic-origin cue & limitation notes:**
  - Reserved domains (`example.com`, `example.org`, `example.net`, `example.edu.bd`), fictional phone numbers (`01700-0000XX` / `+88017000000XX`), and recurring fictional brand names appear only in pilot `Normal` and `Promotional` rows (`0` in `eligible_train` `Smishing`), creating potential lexical synthetic-origin cues not tested by `compute_surface_diagnostics` (which only measures length, script, URL/phone/USSD presence).
  - As noted during independent review, `CodeMix` promotional rows frequently share a two-sentence `[Bengali/Banglish hook] + [English clause]` discourse pattern; downstream experiments must evaluate on real held-out and `BanglaPhish-2026` OOD sets and never treat synthetic-origin patterns as real-world prevalence.
- **Remaining deficit:** Remaining `Promotional` deficit (`345` rows) has **not** been generated pending final pilot sign-off.
