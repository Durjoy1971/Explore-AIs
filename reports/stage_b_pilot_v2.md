# Stage B Pilot Screening & Validation Report (`curation_v2`)

Generated reproducibly by `scripts/validate_stage_b_pilot.py` (`src/bangla_smish/stage_b_pilot.py`). Screening enforces schema, `length_band` consistency (`short(<50)`, `medium(50-120)`, `long(>120)`), `is_synthetic=True`, exact & char-ngram cosine (`>= 0.85` raw & slot-template) isolation against `curation_v2/existing_unique.parquet` (including held-out/OOD) and `quarantined.parquet`, intra-pilot template diversity, fictional entities/reserved domains, label/language rules, and independent review log ingestion.

## 1. Candidate Screening & Independent Review Summary

- **Total candidates screened (`candidates.jsonl`):** **312**
- **Accepted (`accepted.jsonl`):** **300** (96.15%)
  - Confirmed by **both** automated screening and independent review (`review_status="accepted"`): **300**
  - Passed automated screening, pending independent review (`review_status="pending_independent_review"`): **0**
- **Rejected (`rejected.jsonl`):** **12**
- **Exact pilot quota match (`300` rows):** `True`

### Accepted Pilot Distribution (`Label × Language`)

| Class | Bengali | Banglish | English | CodeMix | Total Accepted | Target Pilot Quota |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 10 | 8 | 7 | 14 | **39** | 39 |
| `Promotional` | 65 | 115 | 21 | 60 | **261** | 261 |
| `Smishing` | 0 | 0 | 0 | 0 | **0** | 0 |

### Rejection Breakdown

| Rejection Reason | Count |
| :--- | ---: |
| `independent_review:grammar_error` | 2 |
| `independent_review:lexical_error` | 3 |
| `independent_review:typo` | 1 |
| `independent_review:unnatural_phrasing` | 12 |

- **Rejected IDs (12):** `syn_p2_a_0003`, `syn_p2_a_0013`, `syn_p2_a_0044`, `syn_p2_b_0006`, `syn_p2_b_0054`, `syn_p2_b_0060`, `syn_p2_b_0073`, `syn_p2_b_0083`, `syn_p2_b_0101`, `syn_p2_b_0104`, `syn_p2_b_0111`, `syn_p2_b_0113`

## 2. Domain, Length & Cue Coverage of Accepted Pilot (`N = 300`)

### Domain Breakdown (`Domain × Label`)

| Domain | Normal | Promotional | Total |
| :--- | ---: | ---: | ---: |
| `banking_mfs` | 12 | 43 | **55** |
| `civic_utility_health` | 7 | 14 | **21** |
| `delivery_parcel` | 7 | 32 | **39** |
| `education` | 5 | 29 | **34** |
| `employment` | 0 | 25 | **25** |
| `personal_or_general` | 0 | 13 | **13** |
| `shopping_ecommerce` | 1 | 58 | **59** |
| `telecom` | 7 | 47 | **54** |
| **Total** | **39** | **261** | **300** |

### Length Band & Surface Cue Coverage

| Class | `short(<50)` | `medium(50-120)` | `long(>120)` | Has Link | Has Urgency |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 0 | 16 | 23 | 17 | 15 |
| `Promotional` | 36 | 120 | 105 | 59 | 61 |

## 3. Surface-Only & Length-Only Shortcut Diagnostics (Before vs. After Pilot)

| Metric (`StratifiedGroupKFold`, grouped by `family_id`) | `eligible_train` Alone | `eligible_train + accepted` | Delta |
| :--- | ---: | ---: | ---: |
| Row count (`N`) | 3999 | 4299 | +300 |
| Surface-only 5-fold Accuracy | 71.37% | 67.53% | -3.84% |
| Surface-only 5-fold Macro-F1 | 60.71% | 61.42% | +0.71% |
| Length-only Smishing ROC-AUC | 0.7693 | 0.7341 | -0.0352 |
| Length-only Promo vs. Normal ROC-AUC | 0.7414 | 0.7422 | +0.0008 |

### Per-Class Surface Cue Rates (`Before -> After Pilot`)

| Class | Count (`Before -> After`) | Median Chars (`Before -> After`) | % Has URL (`Before -> After`) | % Has Phone (`Before -> After`) |
| :--- | :--- | :--- | :--- | :--- |
| `Normal` | 1509 -> 1548 | 47.0 -> 49.0 | 0.73% -> 1.81% | 0.73% -> 1.03% |
| `Promotional` | 942 -> 1203 | 74.0 -> 80.0 | 27.28% -> 26.27% | 7.22% -> 6.48% |
| `Smishing` | 1548 -> 1548 | 99.0 -> 99.0 | 45.48% -> 45.48% | 26.68% -> 26.68% |

## 4. Updated Class Counts & Remaining Stage B Deficits

Target `T = 1548`:

| Class | `curation_v2` Original Eligible | Accepted Pilot | Combined Total | Target (`T`) | Remaining Stage B Deficit |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 1509 | 39 | 1548 | 1548 | **0** |
| `Promotional` | 942 | 261 | 1203 | 1548 | **345** |
| `Smishing` | 1548 | 0 | 1548 | 1548 | **0** |
| **Total** | **3999** | **300** | **4299** | — | **345** |

## 5. Contamination, Review Status & Synthetic-Origin Cue / Limitation Notes

- **Anti-leakage & isolation:** Zero exact or near-duplicate (`cos >= 0.85` raw or slot-template) overlap with `curation_v2/existing_unique.parquet` (including `smishdetect_validation`, `smishdetect_test`, and `banglaphish_ood`) or `curation_v2/quarantined.parquet`.
- **Cross-lingual family grouping:** Accepted rows span **292** `family_id` groups (including **4** linked cross-lingual parallel family groups so `StratifiedGroupKFold` keeps parallel realizations in the same fold).
- **Independent review status:** **300** accepted rows are confirmed by both automated screening and independent review (`review_shard_1.jsonl`, `review_shard_2.jsonl`, `review_replacements.jsonl`); **12** rows flagged by independent review are isolated in `rejected.jsonl`; **0** rows are `pending_independent_review`.
- **Synthetic-origin cue & limitation notes:**
  - Reserved domains (`example.com`, `example.org`, `example.net`, `example.edu.bd`), fictional phone numbers (`01700-0000XX` / `+88017000000XX`), and recurring fictional brand names appear only in pilot `Normal` and `Promotional` rows (`0` in `eligible_train` `Smishing`), creating potential lexical synthetic-origin cues not tested by `compute_surface_diagnostics` (which only measures length, script, URL/phone/USSD presence).
  - As noted during independent review, `CodeMix` promotional rows frequently share a two-sentence `[Bengali/Banglish hook] + [English clause]` discourse pattern; downstream experiments must evaluate on real held-out and `BanglaPhish-2026` OOD sets and never treat synthetic-origin patterns as real-world prevalence.
- **Remaining deficit:** Remaining `Promotional` deficit (`345` rows) has **not** been generated pending final pilot sign-off.
