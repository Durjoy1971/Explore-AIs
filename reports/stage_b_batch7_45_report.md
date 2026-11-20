# Stage B Final Batch 7 Evaluation Report (`batch7_45`, N = 45 Promotional SMS)

**Date:** 2026-11-20  
**Cohort Directory:** [`data/generated/curation_v2/batch7_45/`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch7_45)  
**Validation Artifacts:** [`pilot_validation_summary.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch7_45/validation_run/pilot_validation_summary.json), [`batch7_45_validation_summary.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch7_45/batch7_45_validation_summary.json), [`combined_cohorts_diagnostics.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch7_45/combined_cohorts_diagnostics.json)  
**Status:** **Completed & Verified (`45/45` accepted, `0` pending independent review, `0` remaining Stage B deficit — ALL SYNTHETIC GENERATION STOPPED pending user review)**

---

## 1. Executive Summary

- **Final Allocation Achieved (`N = 45` Promotional):** `Banglish: 17`, `Bengali: 12`, `CodeMix: 12`, `English: 4` (`0` Normal, `0` Smishing).
- **Zero Remaining Deficits:** Combined with `eligible_train` (`3,999`), the `300`-row Pilot, and `Batches 1–6` (`300`), the training pool reaches exact three-class balance (`Normal = 1,548`, `Promotional = 1,548`, `Smishing = 1,548`, **Total = `4,644`**) and exact language balance inside Promotional (**`387` rows per language** across `Banglish`, `Bengali`, `CodeMix`, and `English`).
- **Two-Round Screening & Independent Review:** Screened **51** candidates (`45` initial + `6` replacements); accepted **45/51 (`88.24%`)** and isolated **6** rejected rows in [`rejected.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch7_45/validation_run/rejected.jsonl). `0` rows remain `pending_independent_review`.
- **Unified Brand Detection (`0` Mismatches):** Evaluated with unified [`schema.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/schema.py) (`DEFAULT_KNOWN_BRANDS` covering real BD + Pilot + Batches 1–6) merged with Batch 7's [`brand_registry.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch7_45/brand_registry.json) (`8` fresh fictional brands, `0` collisions). Verified **0/942** mismatches on `eligible_train` Promotional and **0/606** mismatches on `full_combined_645` Promotional across `detect_brand_mentions`, `mask_brands`, and brand stripping. Batch 7 achieved **`29/45` (`64.44%`)** naturally brandless rows and **`max_brand_count = 3`**. Note: Batch 7's 8 fictional brands are maintained in `batch7_45/brand_registry.json` during review (matching Batch 5/6 precedent) and have not yet been appended to `DEFAULT_KNOWN_BRANDS` pending user approval.
- **Realism, USSD & Length Alignment:** `8/8` diversity gates pass (`mode=new_batch`). Median length matches `eligible_train` Promotional identically (**`P50 = 74.0`** vs real `74.0`; `mean = 86.00 ± 33.02` vs real `89.20 ± 40.77`), `100%` of Batch 7 strata match real `eligible_train` Promotional 3-way `(language, domain, length_band)` strata (`45/45`), and USSD formatting mixes realistic ASCII (`4`: `*321*57#`, `*121*29#`, `*566*83#`, `*247*74#`) and Bengali digits (`2`: `*১২১*৩৯#`, `*৫৬৬*৭৪#`) with `0` `*1234#` codes.
- **Surface-AUC Orientation Verified:** `clf.classes_ == [0, 1]` and `predict_proba[:, 1]` = `P(is_synthetic_bin == 1)` in every CV fold. Sub-`0.50` Batch 7 surface-only AUCs (`0.4279–0.4656`) reflect finite-sample leave-one-fold-out anti-correlation at near-chance separability (`N_syn = 45`) rather than inverted labels.

---

## 2. Final Dataset Counts & Zero-Deficit Verification

| Class | `eligible_train` | `pilot_300` | `Batches 1–6` | `batch7_45` | `Combined Total` | Target (`T`) | Remaining Deficit |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 1,509 | 39 | 0 | 0 | **1,548** | 1,548 | **0** |
| `Promotional` | 942 | 261 | 300 | 45 | **1,548** | 1,548 | **0** |
| `Smishing` | 1,548 | 0 | 0 | 0 | **1,548** | 1,548 | **0** |
| **Total** | **3,999** | **300** | **300** | **45** | **4,644** | **4,644** | **0** |

### Promotional Language Parity Breakdown (`T_lang = 387`)

| Language | `eligible_train` Promo | `pilot_300` Promo | `Batches 1–6` Promo | `batch7_45` Promo | `Combined Promo` | Target | Remaining Deficit |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Banglish` | 141 | 115 | 114 | 17 | **387** | 387 | **0** |
| `Bengali` | 229 | 65 | 81 | 12 | **387** | 387 | **0** |
| `CodeMix` | 234 | 60 | 81 | 12 | **387** | 387 | **0** |
| `English` | 338 | 21 | 24 | 4 | **387** | 387 | **0** |
| **Total** | **942** | **261** | **300** | **45** | **1,548** | **1,548** | **0** |

Domain breakdown in `batch7_45` (`N = 45`): `telecom: 13`, `shopping_ecommerce: 13`, `personal_or_general: 10`, `banking_mfs: 4`, `delivery_parcel: 3`, `civic_utility_health: 1`, `education: 1`.

---

## 3. Protected-File Integrity Verification (`SHA256`)

All 12 protected datasets, splits, and earlier Stage B cohorts (`pilot_300` + `Batches 1–6`) remain byte-identical to baseline:

| Protected Artifact | `SHA256` Digest | Matches Baseline |
| :--- | :--- | :---: |
| [`eligible_train.parquet`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/curation_v2/eligible_train.parquet) | `d9af8bf952558b91b45db5b5f721da2735ec4b18547cdca59d86e21a8b4993e7` | `True` |
| [`existing_unique.parquet`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/curation_v2/existing_unique.parquet) | `acb40c57870b8987b72a920e914d9e81966201caa26867456437a389d88b3e4f` | `True` |
| [`provenance.parquet`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/curation_v2/provenance.parquet) | `9a0fc3f2de54a7db4f040983504f5907b592dec751db9cb1dfcc07d99c21d443` | `True` |
| [`quarantined.parquet`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/curation_v2/quarantined.parquet) | `c5cb0855abe2f2998e91d6affb8df7a2ae8984daeacb5373a63d4a1294b17cb4` | `True` |
| [`banglaphish2026_dedup.parquet`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/banglaphish2026_dedup.parquet) | `16aaa636aa8cf89cd07b8dd44887c11a7cc0398340c6698d36c2bbb91651c1f1` | `True` |
| [`data/generated/curation_v2/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/accepted.jsonl) | `23af149826dec2411d79b61abceda44c564387574c97d7d12468d7f5a101c5ac` | `True` |
| [`batch1_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch1_50/accepted.jsonl) | `f6bc19c3b0466ac3ce327a51ee88f7e42c49668246a345183ee96c482c507dbd` | `True` |
| [`batch2_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch2_50/accepted.jsonl) | `5f4cb5afb8aaaf0ce37ab4cedb64639df6fb462d5f0f872e8444c7e85a6d32a6` | `True` |
| [`batch3_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch3_50/accepted.jsonl) | `abd5c9d931af3a2f9e84ac21d8a006554ae11bc17059729af19a33efe38b60a1` | `True` |
| [`batch4_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch4_50/accepted.jsonl) | `a97f0a912bad36ced9c5f104f2a4596a8b27f875a074bff4bf487cbce9ecbd64` | `True` |
| [`batch5_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch5_50/accepted.jsonl) | `a0cc662f85097f40767ddc78c85982fd50fab8d71897d76ce553742e6d3f6da1` | `True` |
| [`batch6_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch6_50/accepted.jsonl) | `ba4b6329c6a52ffcd6c233a76173eeb5c2ef713be9cbfb1174680dcef2eec682` | `True` |
| **New Final Cohort:** [`batch7_45/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch7_45/accepted.jsonl) | `5e7d237b9b12c08b68ea909b15563a95f2352f33e9a66d19986682f074c38438` | `New (Verified)` |

---

## 4. Independent Row-by-Row Review & Replacements

Independent review was conducted row-by-row by subagent `independent_batch_reviewer` (`Batch 7 Reviewer`) against naturalness, language/domain/length consistency, USSD realism, brand detection, and cross-batch/intra-batch template & scenario overlap:

- **Round 1 ([`raw_batch7_45.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch7_45/raw_batch7_45.jsonl), `N = 45`):** **`39/45` accepted (`86.67%`)**, **`6/45` rejected**.
- **Round 2 ([`replacements.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch7_45/replacements.jsonl), `N = 6`):** **`6/6` accepted (`100.00%`)**.
- **Final Outcome:** **`45/51` accepted (`88.24%`)**, `0` `pending_independent_review`.

| Rejected ID | Stratum / Brand | Independent Reviewer Reason(s) | Replacement ID & Stratum | Replacement Outcome |
| :--- | :--- | :--- | :--- | :---: |
| `syn_b7_0004` | Banglish / `telecom` / `medium` / Brandless | `intra_batch_scenario_clone` (`syn_b7_0001`), `unnatural_phrasing` (verbless `'4GB net pack chalu'`) | `syn_b7_r_0001` (`medium(50-120)`, `60` chars, Brandless, `example.com/wk5`) | `accept` |
| `syn_b7_0020` | Banglish / `personal_or_general` / `long` / Brandless | `cross_batch_scenario_clone` (`syn_b3_r_0014` rooftop compost/pot combo) | `syn_b7_r_0002` (`long(>120)`, `172` chars, Brandless, pet grooming) | `accept` |
| `syn_b7_0025` | Banglish / `personal_or_general` / `medium` / Brandless | `unnatural_phrasing` (`'Dorkari pest control'`, `'shoptah-e'`) | `syn_b7_r_0003` (`medium(50-120)`, `101` chars, Brandless, `01700-000017`) | `accept` |
| `syn_b7_0026` | Banglish / `shopping_ecommerce` / `medium` / `ShatabdiBazar` | `cross_batch_scenario_clone` (rice/lentils/oil staple trio in `syn_b5_0045`, `syn_b1_0038`, `syn_p2_a_0050`) | `syn_b7_r_0004` (`medium(50-120)`, `95` chars, `ShatabdiBazar`, frozen snacks/dairy) | `accept` |
| `syn_b7_0032` | Banglish / `shopping_ecommerce` / `medium` / Brandless | `cross_batch_scenario_clone` (`syn_b3_0009` trolley bag + backpack) | `syn_b7_r_0005` (`medium(50-120)`, `74` chars, Brandless, wristwatch & belt) | `accept` |
| `syn_b7_0040` | CodeMix / `education` / `long` / `DhansiriBank` | `domain_mismatch` (bank student account belongs to `banking_mfs`, not `education`) | `syn_b7_r_0006` (`long(>120)`, `137` chars, Brandless, 2D/3D CAD & BIM course) | `accept` |

---

## 5. Deduplication, Family Isolation & Unified Brand Detection

### 5a. Deduplication vs. All Prior Corpora (`batch7_45`, `N = 45`)

- `unique_ids`: **45**, `unique_families`: **45** (`100%` unique), `b7_brand_collisions_with_prior`: **`[]`** (`0` collisions).
- Intra-batch (`N = 45`): `0` exact normalized hits, `0` near-dup raw (`max_cos_raw = 0.3634`), `0` near-dup slot (`max_cos_slot = 0.3763`).

| Reference Corpus | `n_ref` | Exact Hits | Near-Dup Raw (`>=0.85`) | `max_cos_raw` | Near-Dup Slot (`>=0.85`) | `max_cos_slot` | `family_overlap` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `existing_unique.parquet` | 11,821 | 0 | 0 | 0.4016 | 0 | 0.4202 | 0 |
| `quarantined.parquet` | 1,094 | 0 | 0 | 0.2954 | 0 | 0.3414 | — |
| `banglaphish2026_dedup.parquet` | 5,416 | 0 | 0 | 0.2462 | 0 | 0.2475 | — |
| `pilot_300` | 300 | 0 | 0 | 0.3269 | 0 | 0.3773 | 0 |
| `batch1_50` | 50 | 0 | 0 | 0.2887 | 0 | 0.4148 | 0 |
| `batch2_50` | 50 | 0 | 0 | 0.3356 | 0 | 0.3951 | 0 |
| `batch3_50` | 50 | 0 | 0 | 0.3912 | 0 | 0.4147 | 0 |
| `batch4_50` | 50 | 0 | 0 | 0.2571 | 0 | 0.3353 | 0 |
| `batch5_50` | 50 | 0 | 0 | 0.4140 | 0 | 0.4378 | 0 |
| `batch6_50` | 50 | 0 | 0 | 0.3765 | 0 | 0.4427 | 0 |

### 5b. Unified Brand Detection & Frequency Balance

- Consistency across `detect_brand_mentions`, `mask_brands` (`brand_masked`), and stripping (`brand_stripped`): **0/942** mismatches on `eligible_train` Promotional; **0/606** mismatches on `full_combined_645` Promotional.
- Batch 7 brand counts (`16/45` branded, **`29/45 = 64.44%` naturally brandless**): `ShuvroTel: 3`, `JolchobiTech: 3`, `AlponaMart: 3`, `BelaCourier: 2`, `KheyaPay: 2`, `ShoilpikStyle: 1`, `DhansiriBank: 1`, `ShatabdiBazar: 1` (`8` unique brands, `max_brand_count = 3`).

---

## 6. Promotional Distributions, Length Profile, USSD & Lexical Naturalness

### 6a. Summary Percentiles & Cue Rates (Promotional Only)

| Metric | `eligible_train` (`942`) | `pilot_300` (`261`) | `B1–B6` (`300`) | `batch7_45` (`45`) | `New B1–B7` (`345`) | `Full Combined` (`606`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Char Len `P25 / P50 / P75` | `66.0 / 74.0 / 99.0` | `91.0 / 116.0 / 127.0` | `68.75 / 79.5 / 99.0` | `64.0 / 74.0 / 95.0` | `68.0 / 78.0 / 99.0` | `70.25 / 94.0 / 122.75` |
| `P10 / P90 / Min–Max` | `58.0 / 140.0 / 42–448` | `45.0 / 140.0 / 36–190` | `63.0 / 143.0 / 40–173` | `57.6 / 134.6 / 47–174` | `62.0 / 142.6 / 40–174` | `59.0 / 141.0 / 36–190` |
| `Mean ± SD` | `89.20 ± 40.77` | `106.66 ± 32.86` | `89.45 ± 29.94` | `86.00 ± 33.02` | `89.00 ± 30.33` | `96.61 ± 32.61` |
| `short / medium / long` | `2.12% / 82.70% / 15.18%` (`20/779/143`) | `13.79% / 45.98% / 40.23%` (`36/120/105`) | `2.00% / 82.00% / 16.00%` (`6/246/48`) | `2.22% / 82.22% / 15.56%` (`1/37/7`) | `2.03% / 82.03% / 15.94%` (`7/283/55`) | `7.10% / 66.50% / 26.40%` (`43/403/160`) |
| `Brandless % (n)` | `69.75% (657)` | `1.92% (5)` | `57.33% (172)` | **`64.44% (29)`** | `58.26% (201)` | `33.99% (206)` |
| `Has URL % (n)` | `29.83% (281)` | `22.61% (59)` | `29.00% (87)` | `28.89% (13)` | `28.99% (100)` | `26.24% (159)` |
| `Has Phone % (n)` | `8.92% (84)` | `8.81% (23)` | `8.00% (24)` | `8.89% (4)` | `8.12% (28)` | `8.42% (51)` |
| `Has USSD % (n)` | `12.95% (122)` | `5.75% (15)` | `13.00% (39)` | `13.33% (6)` | `13.04% (45)` | `9.90% (60)` |
| `USSD (ASCII / BN / Mix)` | `90 / 32 / 0` | `15 / 0 / 0` | `35 / 4 / 0` | `4 / 2 / 0` | `39 / 6 / 0` | `54 / 6 / 0` |
| `Has Urgency % (n)` | `18.58% (175)` | `23.37% (61)` | `9.00% (27)` | `13.33% (6)` | `9.57% (33)` | `15.51% (94)` |

### 6b. Fine-Grained Character-Length Histogram (`Promotional`)

| Bin | `eligible_train` (`942`) | `batch7_45` (`45`) | `B1–B6` (`300`) | `New B1–B7` (`345`) | `pilot_300` (`261`) | `Full Combined` (`606`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `<50` | 20 (2.12%) | 1 (2.22%) | 6 (2.00%) | 7 (2.03%) | 36 (13.79%) | 43 (7.10%) |
| `50–59` | 91 (9.66%) | 4 (8.89%) | 10 (3.33%) | 14 (4.06%) | 5 (1.92%) | 19 (3.14%) |
| `60–69` | 236 (25.05%) | 12 (26.67%) | 63 (21.00%) | 75 (21.74%) | 3 (1.15%) | 78 (12.87%) |
| `70–79` | 205 (21.76%) | 9 (20.00%) | 71 (23.67%) | 80 (23.19%) | 2 (0.77%) | 82 (13.53%) |
| `80–89` | 70 (7.43%) | 4 (8.89%) | 41 (13.67%) | 45 (13.04%) | 12 (4.60%) | 57 (9.41%) |
| `90–99` | 94 (9.98%) | 4 (8.89%) | 37 (12.33%) | 41 (11.88%) | 23 (8.81%) | 64 (10.56%) |
| `100–120` | 83 (8.81%) | 4 (8.89%) | 24 (8.00%) | 28 (8.12%) | 75 (28.74%) | 103 (17.00%) |
| `121–150` | 65 (6.90%) | 3 (6.67%) | 27 (9.00%) | 30 (8.70%) | 92 (35.25%) | 122 (20.13%) |
| `>150` | 78 (8.28%) | 4 (8.89%) | 21 (7.00%) | 25 (7.25%) | 13 (4.98%) | 38 (6.27%) |

### 6c. USSD & Natural Phrasing Notes (`batch7_45`)

- **USSD Script Realism (`6` USSD rows):** `Bn+CM` (`n=4` USSD rows) has **2 ASCII** (`*321*57#`, `*247*74#`) and **2 Bengali-digit** (`*১২১*৩৯#`, `*৫৬৬*৭৪#`) (`50.0%` Bengali-digit, closely matching real `Bn+CM` patterns without artificial forced quotas). `Banglish` (`*121*29#`) and `English` (`*566*83#`) use `100%` ASCII. Dummy `*1234#` share is **`0.0%`**.
- **Lexical Frequencies (`Bn+CM n=24`, `Banglish n=17`):** `ও` (`18`, `75.0%`), `এবং` (`1`, `4.17%`), `পাবেন` (`6`, `25.0%`), `পাচ্ছেন` (`3`, `12.5%`), `থাকছে` (`1`, `4.17%`), `দেখুন` (`3`, `12.5%`), `উপলব্ধ` (`0`), `লাভ` (`0`); Banglish `ar` (`6`, `35.29%`), `o` (`7`, `41.18%`), `ebong` (`0`).

---

## 7. Eight Diversity Gates (Unified Detector)

| Gate | Threshold | `batch7_45` (`new_batch`) | `combined_new_345` (`new_batch`) | `pilot_300` (`audit`) | `full_combined_645` (`audit`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `brandless_rate` | `>= 0.30` | `0.6444` (`Pass`) | `0.5826` (`Pass`) | `0.0192` (`Fail`) | `0.3399` (`Pass`) |
| `codemix_bn_script_en` | `>= 0.80` | `1.0000` (`Pass`) | `1.0000` (`Pass`) | `0.5833` (`Fail`) | `0.8366` (`Pass`) |
| `codemix_hook_plus_pure_en` | `== 0.0` (`new_batch`) / `<= 0.15` (`audit`) | `0.0000` (`Pass`) | `0.0000` (`Pass`) | `0.8500` (`Fail`) | `0.3333` (`Fail`) |
| `codemix_intra_sentential_mix` | `>= 0.60` | `1.0000` (`Pass`) | `0.9462` (`Pass`) | `0.1333` (`Fail`) | `0.6275` (`Pass`) |
| `max_brand_count` | `<= 5` | `3` (`Pass`) | `4` (`Pass`) | `25` (`Fail`) | `25` (`Fail`) |
| `promo_sent1_excl_rate` | `<= 0.35` | `0.0444` (`Pass`) | `0.0580` (`Pass`) | `0.5632` (`Fail`) | `0.2756` (`Pass`) |
| `promo_sent1_quest_rate` | `<= 0.05` | `0.0000` (`Pass`) | `0.0058` (`Pass`) | `0.0920` (`Fail`) | `0.0429` (`Pass`) |
| `ussd_1234_share` | `<= 0.25` | `0.0000` (`Pass`) | `0.0000` (`Pass`) | `1.0000` (`Fail`) | `0.2500` (`Pass`) |
| **All 8 Gates Passed** | — | **`True (8/8)`** | **`True (8/8)`** | **`False (0/8)`** | **`False (6/8)`** |

Note: Addition of Batch 7 pushed `full_combined_645` over threshold on two additional cumulative audit gates (`codemix_intra_sentential_mix = 0.6275 >= 0.60` and `ussd_1234_share = 0.2500 <= 0.25`), leaving only the legacy `pilot_300` artifacts (`codemix_hook_plus_pure_en` and `max_brand_count`) failing in the 645-row cumulative union.

---

## 8. Five-Cohort Comparison, 4-View Probes & Counterfactual Sensitivity (`Seed = 42`)

### 8a. Surface Diagnostics & Length-Only Shortcut AUC (`StratifiedGroupKFold`, `n_splits=5`)

| Cohort | `N` (`elig + syn`) | Surf Acc | Surf Macro-F1 | Len Smish AUC | Len Promo vs Normal AUC |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `eligible_train` (Before) | 3,999 | 73.14% | 62.86% | 0.7693 | 0.7414 |
| `+ pilot_300` | 4,299 | 69.32% | 62.78% | 0.7341 | 0.7422 |
| `+ b1_to_b6_300` | 4,299 | 69.09% | 62.54% | 0.7579 | 0.7447 |
| `+ batch7_45` | 4,044 | 72.35% | 62.53% | 0.7680 | 0.7411 |
| `+ combined_new_345` (`B1–B7`) | 4,344 | 68.58% | 62.38% | 0.7577 | 0.7438 |
| `+ full_combined_645` (`All`) | 4,644 | 65.87% | 62.29% | 0.7269 | 0.7411 |

### 8b. 4-View 3-Class Performance (`Macro-F1 / Accuracy / Smishing Recall / Recall@<=1% FPR`)

| Cohort | `raw` | `canon_masked` | `brand_masked` | `brand_stripped` |
| :--- | :---: | :---: | :---: | :---: |
| `eligible_train` (Before) | `98.01% / 98.10% / 98.77% / 98.51%` | `97.86% / 97.95% / 98.45% / 97.80%` | `97.76% / 97.85% / 98.39% / 97.80%` | `97.71% / 97.80% / 98.39% / 97.80%` |
| `+ pilot_300` | `97.55% / 97.58% / 98.26% / 97.61%` | `97.26% / 97.28% / 98.06% / 95.35%` | `97.05% / 97.07% / 98.00% / 93.09%` | `97.24% / 97.26% / 97.87% / 91.93%` |
| `+ b1_to_b6_300` | `98.00% / 98.02% / 98.71% / 98.19%` | `97.78% / 97.81% / 98.51% / 97.93%` | `97.76% / 97.79% / 98.51% / 97.48%` | `97.73% / 97.77% / 98.51% / 97.74%` |
| `+ batch7_45` | `98.15% / 98.22% / 98.77% / 98.19%` | `97.58% / 97.68% / 98.51% / 97.67%` | `97.61% / 97.70% / 98.45% / 97.74%` | `97.66% / 97.75% / 98.45% / 97.67%` |
| `+ combined_new_345` | `97.93% / 97.95% / 98.71% / 98.39%` | `97.75% / 97.77% / 98.32% / 97.67%` | `97.78% / 97.79% / 98.26% / 97.42%` | `97.68% / 97.70% / 98.32% / 97.80%` |
| `+ full_combined_645` | `97.72% / 97.72% / 98.51% / 97.48%` | `97.31% / 97.31% / 97.93% / 95.03%` | `97.29% / 97.29% / 98.06% / 94.90%` | `97.18% / 97.18% / 97.74% / 94.25%` |

### 8c. Counterfactual URL/Phone/Brand Probes (`n = 1,279` Real Smishing Rows)

| Cohort | `raw` Contact Swap (`pp / dP`) | `raw` Contact+Brand Swap (`pp / dP`) | Masked Views (`canon / brand_masked / brand_stripped`) |
| :--- | :---: | :---: | :--- |
| `+ pilot_300` | `89.60% (-10.40 pp, -0.2109)` | `87.18% (-12.82 pp, -0.2449)` | `canon` CB: `-0.08 pp (-0.0193)`; `brand_masked/stripped`: `0.00 pp (+0.0001)` |
| `+ b1_to_b6_300` | `93.51% (-6.49 pp, -0.1456)` | `93.35% (-6.65 pp, -0.1502)` | `canon` CB: `0.00 pp (-0.0014)`; `brand_masked/stripped`: `0.00 pp (+0.0001)` |
| `+ batch7_45` | `98.44% (-1.56 pp, -0.0781)` | `98.44% (-1.56 pp, -0.0807)` | `canon` CB: `0.00 pp (-0.0011)`; `brand_masked/stripped`: `0.00 pp (+0.0001)` |
| `+ combined_new_345` | `94.06% (-5.94 pp, -0.1407)` | `93.98% (-6.02 pp, -0.1451)` | `canon` CB: `0.00 pp (-0.0014)`; `brand_masked/stripped`: `0.00 pp (+0.0001)` |
| `+ full_combined_645` | `88.19% (-11.81 pp, -0.2112)` | `86.08% (-13.92 pp, -0.2430)` | `canon` CB: `0.00 pp (-0.0179)`; `brand_masked/stripped`: `0.00 pp (+0.0001)` |

---

## 9. Four-View Synthetic-Origin Diagnostics, Surface-AUC Orientation & 10-Seed CIs

### 9a. Methodology & Non-Comparability Caveats
- **CV Protocol:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)` grouped by `family_id`, with TF-IDF + balanced `LogisticRegression` (or `StandardScaler` + balanced `LogisticRegression` on `surface_8feat`) refitted inside each fold; stratum-controlled probes restrict to shared strata and apply inverse-stratum-count sample weights.
- **`±` Definition:** Sample SD (`ddof=1`) of pooled OOF ROC-AUC across 10 random CV fold-assignment seeds (`42..51`), quantifying CV split-randomization variance on the fixed dataset (not SEM, not out-of-sample generalization error).
- **Bootstrap 95% CIs (`B = 1,000`):** Both row-level stratified percentile bootstrap and `family_id`-cluster stratified percentile bootstrap on 10-seed averaged OOF probabilities (`mean_oof`), plus seed-42 stratified bootstrap. OOF CIs reflect finite-sample score uncertainty conditional on the fixed pool.
- **Unequal `N_syn` Warning:** Cohorts vary in synthetic count (`45`, `261`, `300`, `345`, `606`) and stratum coverage (`pilot_300` 3-way match `71.26%`, `batch7_45` `100.0%`, `combined_new_345` `96.52%`); raw AUCs across unequal `N_syn` are not directly statistically comparable.

### 9b. Batch 7 Seed=42 Synthetic-Origin Probes (`45` Synthetic Promotional vs. Real Splits)

| Comparison | Coverage (`N`) | `raw` AUC | `canon_masked` AUC | `brand_masked` AUC | `brand_stripped` AUC | `surface_8feat` AUC |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `all_vs_real` | `45 syn vs 3999 real` | 0.9850 | 0.9855 | 0.9798 | 0.9833 | 0.4265 |
| `synth_vs_real_smishing` | `45 syn vs 1548 real` | 0.9983 | 0.9967 | 0.9967 | 0.9967 | 0.7958 |
| `synthetic_promo_vs_real_promo` | `45 syn vs 942 real` | 0.9869 | 0.9766 | 0.9711 | 0.9724 | 0.4532 |
| `synthetic_promo_vs_real_promo_stratum_3way` | `27 strata (45/45 syn, 803/942 real)` | 0.9675 | 0.9530 | 0.9445 | 0.9460 | 0.4656 |
| `synthetic_promo_vs_real_promo_stratum_2way` | `8 strata (45/45 syn, 891/942 real)` | 0.9299 | 0.8894 | 0.8774 | 0.8767 | 0.4279 |

### 9c. Surface-Only ROC-AUC Positive-Class Orientation Verification

Verified across all 5 cohorts and all 3 promo probes (`synthetic_promo_vs_real_promo`, `stratum_3way`, `stratum_2way`):
- `all_folds_column1_is_class1 == True` (`clf.classes_ == [0, 1]`, `predict_proba[:, 1]` = `P(is_synthetic_bin == 1)`).
- On `batch7_45`, weighted OOF `surface_8feat` ROC-AUC on `stratum_3way` is `0.5182` (weighted eval), `0.5229` in-sample full fit, and unweighted fit OOF AUC is `0.5350`, whereas unweighted OOF evaluation of weighted-fit probabilities yields `0.4656` (`oof_mean_p_syn` `0.4937` vs `0.5003`). Thus values below `0.50` reflect finite-sample leave-one-fold-out anti-correlation under inverse-stratum-count weighting near true chance (`0.50`), **not** inverted class labels.

### 9d. 10-Seed (`42..51`) & Cluster Bootstrap 95% CIs (`Synthetic Promo vs. Real Promo`)

| Cohort (`N_syn`, 3-way Cov) | Probe | View | `Seed 42` (`95% CI`) | `10-Seed Mean ± SD` `[min..max]` | `Mean-OOF AUC` (`Strat 95% CI` / `Cluster 95% CI`) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `batch7_45` (`45`, `100.0%`) | `unweighted` | `raw` | `0.9869` `[0.9759, 0.9957]` | `0.9774 ± 0.0066` `[0.9674..0.9869]` | `0.9814` (`[0.9644, 0.9944]` / `[0.9635, 0.9943]`) |
| `batch7_45` (`45`, `100.0%`) | `unweighted` | `brand_masked` | `0.9711` `[0.9506, 0.9883]` | `0.9569 ± 0.0099` `[0.9425..0.9720]` | `0.9652` (`[0.9398, 0.9858]` / `[0.9383, 0.9854]`) |
| `batch7_45` (`45`, `100.0%`) | `stratum_3way` | `brand_masked` | `0.9445` `[0.9123, 0.9704]` | `0.9327 ± 0.0092` `[0.9141..0.9445]` | `0.9445` (`[0.9085, 0.9714]` / `[0.9093, 0.9720]`) |
| `batch7_45` (`45`, `100.0%`) | `stratum_3way` | `surface_8feat` | `0.4656` `[0.3839, 0.5541]` | `0.4301 ± 0.0216` `[0.3968..0.4656]` | `0.4092` (`[0.3207, 0.4937]` / `[0.3291, 0.5006]`) |
| `batch7_45` (`45`, `100.0%`) | `stratum_2way` | `brand_masked` | `0.8774` `[0.8176, 0.9324]` | `0.8790 ± 0.0101` `[0.8668..0.8959]` | `0.8912` (`[0.8358, 0.9413]` / `[0.8278, 0.9435]`) |
| `combined_new_345` (`345`, `96.52%`) | `unweighted` | `brand_masked` | `0.9901` `[0.9857, 0.9939]` | `0.9892 ± 0.0008` `[0.9883..0.9911]` | `0.9909` (`[0.9867, 0.9944]` / `[0.9868, 0.9943]`) |
| `combined_new_345` (`345`, `96.52%`) | `stratum_3way` | `brand_masked` | `0.9550` `[0.9432, 0.9663]` | `0.9560 ± 0.0021` `[0.9526..0.9592]` | `0.9609` (`[0.9506, 0.9713]` / `[0.9499, 0.9704]`) |
| `combined_new_345` (`345`, `96.52%`) | `stratum_3way` | `surface_8feat` | `0.5062` `[0.4740, 0.5410]` | `0.5083 ± 0.0046` `[0.5008..0.5171]` | `0.5018` (`[0.4688, 0.5371]` / `[0.4640, 0.5368]`) |
| `combined_new_345` (`345`, `96.52%`) | `stratum_2way` | `brand_masked` | `0.9284` `[0.9122, 0.9428]` | `0.9265 ± 0.0032` `[0.9227..0.9331]` | `0.9393` (`[0.9252, 0.9517]` / `[0.9255, 0.9518]`) |
| `b1_to_b6_300` (`300`, `96.00%`) | `stratum_3way` | `brand_masked` | `0.9594` `[0.9477, 0.9703]` | `0.9567 ± 0.0027` `[0.9522..0.9594]` | `0.9622` (`[0.9510, 0.9729]` / `[0.9510, 0.9730]`) |
| `pilot_300` (`261`, `71.26%`) | `stratum_3way` | `brand_masked` | `0.9716` `[0.9615, 0.9808]` | `0.9688 ± 0.0025` `[0.9640..0.9717]` | `0.9733` (`[0.9634, 0.9822]` / `[0.9631, 0.9820]`) |
| `full_combined_645` (`606`, `85.48%`) | `stratum_3way` | `brand_masked` | `0.9495` `[0.9377, 0.9599]` | `0.9508 ± 0.0026` `[0.9479..0.9562]` | `0.9565` (`[0.9470, 0.9659]` / `[0.9471, 0.9655]`) |
| `full_combined_645` (`606`, `85.48%`) | `stratum_3way` | `surface_8feat` | `0.4719` `[0.4423, 0.5004]` | `0.4709 ± 0.0034` `[0.4655..0.4751]` | `0.4699` (`[0.4402, 0.4991]` / `[0.4405, 0.5015]`) |

---

## 10. Stop Confirmation

Batch 7 completes all Stage B synthetic Promotional deficits (`45/45` accepted, `Total = 4,644`, `1,548` per class, `387` Promotional per language). All synthetic generation is now **STOPPED** awaiting user review; see the synthesis in [`stage_b_completion_report.md`](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/stage_b_completion_report.md).
