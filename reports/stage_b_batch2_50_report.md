# Stage B Batch 2 (`50` Promotional Rows) Evaluation & Checkpoint Report (`curation_v2`)

**Date:** 2026-11-18  
**Batch Directory (Isolated):** `data/generated/curation_v2/batch2_50/`  
**Validation Summary JSON:** `data/generated/curation_v2/batch2_50/batch2_50_validation_summary.json`  
**Status:** **Evaluated & Passed All Automated + Independent Review Gates — Stopped before generating remaining `245` `Promotional` rows, awaiting explicit user approval.**

---

## 1. Integrity, Isolation & Deduplication Verification

1. **Protected Corpora, 300-Row Pilot & Batch 1 Untouched:**
   - `data/generated/curation_v2/accepted.jsonl` (`300` rows, SHA256 `23af149826dec2411d79b61abceda44c564387574c97d7d12468d7f5a101c5ac`) remains completely unmodified.
   - `data/generated/curation_v2/batch1_50/accepted.jsonl` (`50` rows, SHA256 `f6bc19c3b0466ac3ce327a51ee88f7e42c49668246a345183ee96c482c507dbd`) remains completely unmodified (`Batch 2` was **not** merged into the pilot or Batch 1 files).
   - `data/processed/curation_v2/eligible_train.parquet` (`3,999` rows, SHA256 `d9af8bf952558b91b45db5b5f721da2735ec4b18547cdca59d86e21a8b4993e7`), `existing_unique.parquet` (`11,821` rows, SHA256 `acb40c57870b8987b72a920e914d9e81966201caa26867456437a389d88b3e4f`), and `provenance.parquet` (`18,357` rows, SHA256 `9a0fc3f2de54a7db4f040983504f5907b592dec751db9cb1dfcc07d99c21d443`) remain completely unmodified.
   - Held-out evaluation splits (`smishdetect_validation=708`, `smishdetect_test=1,698`) and OOD benchmark (`banglaphish_ood=5,416`) were never inspected.

2. **Exact & Near-Duplicate (`cos >= 0.85`) Checks (Raw & Slot-Template):**

| Reference Corpus | Reference Rows | Exact Norm Hits | Near-Dup Raw (`cos >= 0.85`) | Max Raw Cos | Near-Dup Slot (`cos >= 0.85`) | Max Slot Cos | Family Overlap |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `curation_v2/existing_unique.parquet` (incl. held-out & OOD) | 11,821 | **0** | **0** | 0.3382 | **0** | 0.4851 | **0** |
| `curation_v2/quarantined.parquet` | 1,094 | **0** | **0** | 0.2243 | **0** | 0.3285 | — |
| `banglaphish2026_dedup.parquet` (OOD) | 5,416 | **0** | **0** | 0.2676 | **0** | 0.2712 | — |
| `curation_v2/accepted.jsonl` (300-row pilot) | 300 | **0** | **0** | 0.3343 | **0** | 0.4628 | **0** |
| `curation_v2/batch1_50/accepted.jsonl` (Batch 1) | 50 | **0** | **0** | 0.4390 | **0** | 0.5412 | **0** |
| Intra-batch (`batch2_50/accepted.jsonl`) | 50 | **0** | **0** | 0.3359 | **0** | 0.3911 | 50 unique `family_id`s |

- Automated screening (`scripts/validate_stage_b_pilot.py --diversity-mode new_batch`) matches `@reviewer`'s `data/generated/curation_v2/batch2_50/accepted.jsonl` 50/50 by ID and text (`matches_validation_run = True`).

---

## 2. Independent Review, Rejected & Replacement Samples

- **Total candidates reviewed (`raw_batch2_50.jsonl` + `replacements.jsonl`):** **53** (`reviewer_swarm`)
- **Initial batch acceptance:** **47 / 50** (`94.0%`); **overall acceptance across 53 candidates:** **50 / 53** (**94.34%**)
- **Final accepted (`batch2_50/accepted.jsonl`):** **50** (`review_status = "accepted"`, `0` `pending_independent_review`, `0` automated schema/endpoint/language/boundary rejections)

### Rejected & Replaced Rows (`3 / 53`)

1. **`syn_b2_0001`** (`Banglish | telecom | short(<50) | brandless | USSD`)
   - **Original text:** `55 takay 3GB night data 7 din, dial *121*34#.`
   - **Rejection reasons:** `independent_review:shallow_template_clone`
   - **Reviewer rationale:** Shallow slot-template clone of Batch 1 `syn_b1_0001` (`Matro 68 takay 4GB data 7 din, dial *121*45#.`, slot-template cosine `0.842`).
   - **Replacement (`syn_b2_r_0001`, accepted):** `Shob operator-e 48 poisha/min, dial *888*2#.` (`44` chars; distinct all-operator call-rate offer, non-`*1234#` USSD).
2. **`syn_b2_0042`** (`CodeMix | personal_or_general | medium(50-120) | brandless`)
   - **Original text:** `Bondhuder sathe evening coffee ebong burger platter-e get 20% discount on Friday.`
   - **Rejection reasons:** `independent_review:unnatural_phrasing`
   - **Reviewer rationale:** Switched abruptly from Banglish locative (`platter-e`) directly into an English imperative (`get 20% discount`), reading unnaturally.
   - **Replacement (`syn_b2_r_0002`, accepted):** `বিকেলের কফি ও লাইভ acoustic music-এর সঙ্গে snacks অর্ডারে থাকছে ২০% weekend ছাড়।` (`80` chars; natural intra-sentential Bengali-script CodeMix).
3. **`syn_b2_0047`** (`English | telecom | medium(50-120) | brandless | USSD`)
   - **Original text:** `Enjoy 15GB internet and 250 local minutes for 30 days at 279 TK. Dial *512*4# to activate.`
   - **Rejection reasons:** `independent_review:near_duplicate_slot_template`, `independent_review:shallow_template_clone`
   - **Reviewer rationale:** Direct slot-template clone of Batch 1 `syn_b1_0047` (`slot_template` cosine `0.868 >= 0.85`, raw cosine `0.717`).
   - **Replacement (`syn_b2_r_0003`, accepted):** `Activate our 3-day 5GB weekend streaming pack via *512*4# for only 59 TK.` (`73` chars; structurally distinct English weekend streaming USSD promo).

---

## 3. Distribution Comparison vs. `eligible_train`, Pilot (`261`), & Batch 1 (`50`)

| Dimension | `eligible_train` Promo (`N=942`) | Pilot `261` Promo (`N=261`) | Batch 1 (`N=50`) | **Batch 2 (`N=50`)** | **Combined B1+B2 (`N=100`)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Language** | En 338, CM 234, Bn 229, Bg 141 | Bg 115, Bn 65, CM 60, En 21 | Bg 19, CM 14, Bn 13, En 4 | **Bg 19, Bn 14, CM 13, En 4** | **Bg 38, Bn 27, CM 27, En 8** |
| **`short(<50)`** | 20 (2.1%) | 36 (13.8%) | 1 (2.0%) | **1 (2.0%)** | **2 (2.0%)** |
| **`medium(50-120)`** | 779 (82.7%) | 120 (46.0%) | 41 (82.0%) | **41 (82.0%)** | **82 (82.0%)** |
| **`long(>120)`** | 143 (15.2%) | 105 (40.2%) | 8 (16.0%) | **8 (16.0%)** | **16 (16.0%)** |
| **Median / Mean Chars** | `74.0` / `89.20` | `116.0` / `106.66` | `72.5` / `85.92` | **`74.5` / `86.40`** | **`74.0` / `86.16`** |
| **% Has URL** | 29.83% | 22.61% | 28.00% | **28.00%** (`14/50`) | **28.00%** (`28/100`) |
| **% Has Phone** | 8.92% | 8.81% | 8.00% | **8.00%** (`4/50`) | **8.00%** (`8/100`) |
| **% Has USSD** | 12.95% | 5.75% | 14.00% | **14.00%** (`7/50`) | **14.00%** (`14/100`) |
| **% Has Urgency** | 7.43% | 23.37% | 8.00% | **8.00%** (`4/50`) | **8.00%** (`8/100`) |
| **% Brandless (detected)** | 66.45% | 1.92% | 36.00% | **60.00%** (`30/50`) | **48.00%** (`48/100`) |

### Domain Breakdown (`Batch 2`, `Batch 1`, `Combined B1+B2`, vs `eligible_train` Promotional)

| Domain | `eligible_train` (`N=942`) | Pilot `261` | Batch 1 (`N=50`) | **Batch 2 (`N=50`)** | **Combined B1+B2 (`N=100`)** |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `telecom` | 281 (29.8%) | 47 | 15 | **15** | **30** |
| `shopping_ecommerce` | 270 (28.7%) | 58 | 14 | **14** | **28** |
| `personal_or_general` | 210 (22.3%) | 13 | 11 | **11** | **22** |
| `banking_mfs` | 89 (9.4%) | 43 | 5 | **5** | **10** |
| `delivery_parcel` | 66 (7.0%) | 32 | 3 | **3** | **6** |
| `civic_utility_health` | 15 (1.6%) | 14 | 1 | **1** | **2** |
| `education` | 4 (0.4%) | 29 | 1 | **0** | **1** |
| `employment` | 7 (0.7%) | 25 | 0 | **1** | **1** |

---

## 4. CodeMix Diversity, Punctuation & Brand Gates (`--diversity-mode new_batch`)

All **8 / 8** automated diversity gates passed on **both** `Batch 2` (`N=50`) and `Combined B1+B2` (`N=100`):

| Gate / Metric | Batch 2 (`N=50`) | Combined B1+B2 (`N=100`) | Threshold (`new_batch`) | Passed | Batch 1 (`N=50`) | Pilot `261` (`audit`) |
| :--- | ---: | ---: | :--- | :---: | ---: | ---: |
| `codemix_bn_script_en` | **1.0000** (`13/13`) | **1.0000** (`27/27`) | `>= 0.80` | **True** | `1.0000` | `0.5833` |
| `codemix_hook_plus_pure_en` | **0.0000** (`0/13`) | **0.0000** (`0/27`) | `== 0.00` | **True** | `0.0000` | `0.8500` |
| `codemix_intra_sentential_mix` | **0.8462** (`11/13`*) | **0.9259** (`25/27`*) | `>= 0.60` | **True** | `1.0000` | `0.1333` |
| `promo_sent1_excl_rate` | **0.0600** (`3/50`) | **0.0800** (`8/100`) | `<= 0.35` | **True** | `0.1000` | `0.5632` |
| `promo_sent1_quest_rate` | **0.0000** (`0/50`) | **0.0100** (`1/100`) | `<= 0.05` | **True** | `0.0200` | `0.0920` |
| `max_brand_count` | **3** | **4** (`ShaplaSeba`) | `<= 5` | **True** | `4` | `25` |
| `brandless_rate` | **0.6000** (`30/50`) | **0.4800** (`48/100`) | `>= 0.30` | **True** | `0.3600` | `0.0192` |
| `ussd_1234_share` | **0.0000** (`0/7`) | **0.0000** (`0/14`) | `<= 0.25` | **True** | `0.0000` | `1.0000` |

*\*Note: Batch 2 intentionally included 2 `en_lead_bn_body` CodeMix rows alongside 11 `intra_sentential_mix` rows for structural variety (`11 + 2 = 13`).*

- **Batch 2 fresh fictional brands (`8` brands, `20` branded rows, max count `3`):** `deshnet` (`3`), `nandanbazar` (`3`), `chaitimart` (`3`), `polashpay` (`3`), `ekusheytech` (`2`), `utshobfashion` (`2`), `turagbank` (`2`), `probashicourier` (`2`). Combined `B1+B2` spans **`21` unique fictional brands**, max `4` occurrences.
- **Batch 2 USSD codes (`7` rows):** `*888*2#`, `*566*21#`, `*789*12#`, `*321*9#`, `*654*7#`, `*121*67#`, `*512*4#` (`0` uses of `*1234#`).
- **Opener 3-gram uniqueness:** `max_opener_3gram_count = 1` across `Batch 2` (`2.0%`) and across `Combined B1+B2` (`1.0%` across all `100` rows!).

---

## 5. Surface, 4-View 3-Class, Counterfactual & Synthetic-Origin Diagnostics

### A. Surface-Only & 4-View 3-Class / Counterfactual Probes (`n = 1,279` Real Smishing Rows)

| Cohort | Surface Acc / MF1 | Length Smish AUC | `raw` MF1 / R@1%FPR / CF | `canon_masked` MF1 / R@1% / CF | `brand_masked` MF1 / R@1% / CF | `brand_stripped` MF1 / R@1% / CF |
| :--- | :--- | ---: | :--- | :--- | :--- | :--- |
| `eligible_train` (`3,999`) | 73.14% / 62.86% | 0.7693 | 98.01% / 98.51% / — | 97.90% / 97.67% / — | 97.76% / 97.67% / — | 97.81% / 97.87% / — |
| `+ Batch 2` (`4,049`) | 72.51% / 63.00% | 0.7679 | 97.96% / 98.58% / `-1.56pp` | 97.63% / 97.61% / **`0.00pp`** | 97.58% / 97.55% / **`0.00pp`** | 97.56% / 97.67% / **`0.00pp`** |
| `+ B1+B2` (`4,099`) | 71.85% / 62.93% | 0.7666 | 97.88% / 97.87% / `-3.21pp` | 97.41% / 96.83% / **`0.00pp`** | 97.57% / 96.51% / **`0.00pp`** | 97.43% / 96.77% / **`0.00pp`** |
| `+ All 400` (`4,399`) | 68.22% / 62.63% | 0.7330 | 97.51% / 97.35% / `-11.73/-14.31pp` | 97.24% / 95.80% / **`0.00pp`** | 97.16% / 95.67% / **`0.00pp`** | 97.09% / 94.96% / **`0.00pp`** |

### B. Synthetic-Origin (`is_synthetic`) 4-View Probes (`Promotional vs Promotional` & Overall)

| Probe & Control | Feature View | Pilot `300` (`261` Promo) | Batch 1 (`50`) | **Batch 2 (`50`)** | **Comb B1+B2 (`100`)** | **Comb All (`400`, `361` Promo)** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `all_vs_real` | `raw` | 0.9979 | 0.9812 | **0.9897** | **0.9949** | 0.9977 |
| (vs `3,999` real) | `canon_masked` | 0.9976 | 0.9769 | **0.9787** | **0.9921** | 0.9962 |
| | `brand_masked` | 0.9928 | 0.9713 | **0.9650** | **0.9894** | 0.9901 |
| | `brand_stripped` | 0.9789 | 0.9635 | **0.9629** | **0.9909** | 0.9834 |
| | `surface_8feat` | 0.7260 | 0.6025 | **0.5383** | **0.6048** | 0.6923 |
| `synthetic_promo_vs_real_promo` | `raw` | 0.9966 | 0.9417 | **0.9836** | **0.9882** | 0.9951 |
| (Uncontrolled vs `942` real Promo) | `canon_masked` | 0.9952 | 0.9338 | **0.9680** | **0.9820** | 0.9919 |
| | `brand_masked` | 0.9871 | 0.9283 | **0.9514** | **0.9759** | 0.9815 |
| | `brand_stripped` | 0.9728 | 0.9101 | **0.9502** | **0.9752** | 0.9772 |
| | `surface_8feat` | 0.7167 | 0.4791 | **0.4389** | **0.4999** | 0.6593 |
| `synthetic_promo_vs_real_promo_stratum_3way` | `raw` | 0.9904 | 0.9242 | **0.9512** | **0.9637** | 0.9747 |
| `(language, domain, length_band)` | `canon_masked` | 0.9848 | 0.8794 | **0.9211** | **0.9389** | 0.9587 |
| B2: `24` strata (`47/50` syn, `759/942` real) | `brand_masked` | 0.9735 | 0.8787 | **0.9010** | **0.9307** | 0.9509 |
| B1+B2: `26` strata (`94/100` syn, `769/942` real) | `brand_stripped` | 0.9483 | 0.8396 | **0.9057** | **0.9258** | 0.9262 |
| | `surface_8feat` | 0.5087 | 0.4591 | **0.4142** | **0.4792** | 0.4576 |
| `synthetic_promo_vs_real_promo_stratum_2way` | `raw` | 0.9803 | 0.9171 | **0.9298** | **0.9529** | 0.9618 |
| `(language, length_band)` | `canon_masked` | 0.9683 | 0.9048 | **0.8826** | **0.9164** | 0.9351 |
| B2: `7` strata (`49/50` syn, `882/942` real) | `brand_masked` | 0.9603 | 0.9207 | **0.8732** | **0.9171** | 0.9298 |
| B1+B2: `8` strata (`98/100` syn, `922/942` real) | `brand_stripped` | 0.9035 | 0.8818 | **0.8682** | **0.9063** | 0.8793 |
| | `surface_8feat` | 0.4595 | 0.4968 | **0.4576** | **0.5177** | 0.4461 |

Also `synth_vs_real_smishing` (`50/100/400` vs `1,548` real Smishing): Batch 2 `raw 0.9991, canon_masked 0.9982, brand_masked 0.9978, brand_stripped 0.9976, surface_8feat 0.7656`; Combined B1+B2 (`100`) `raw 0.9999, canon_masked 0.9996, brand_masked 0.9996, brand_stripped 0.9995, surface_8feat 0.7665`.

### C. Diagnostic Interpretation & Residual Lexical Notes

1. **Exact surface-feature parity achieved (`surface_8feat` Promo AUC = `0.4999` for Combined B1+B2):** Across `100` new Promotional rows, surface metadata (`char_len`, `word_len`, script, URL, phone, USSD) is completely indistinguishable from real `eligible_train` Promotional SMS (`median = 74.0` vs `74.0`, `surface_8feat` AUC `0.4389` on Batch 2, `0.4999` on Combined 100).
2. **Consistent reduction over Pilot (`261`) on 2-way `(language, length_band)` controlled views:** Batch 2 achieved the lowest 2-way controlled lexical separability yet across masked/stripped views (`canon_masked 0.8826`, `brand_masked 0.8732`, `brand_stripped 0.8682`, down from Pilot `0.9683 / 0.9603 / 0.9035`), benefiting from `60%` natural brandlessness and `<= 3` mentions per fictional brand.
3. **Why `3-way` `brand_stripped` was `0.9057` on Batch 2 (vs `0.8396` on Batch 1) and `0.9258` on Combined 100:** Inspecting top positive TF-IDF features on Batch 2 3-way `brand_stripped` (`word__ওয়` +0.577, `word__লব` +0.497, `word__parcel` +0.486, `word__কর url` +0.479) and Combined 100 (`word__ar` +0.467, `char__ ও ` +0.432, `word__কর url` +0.429, `word__ওয়` +0.392) shows that shifting Bengali/CodeMix phrasing uniformly toward `ও` (`word__ওয়`), `উপলব্ধ`/`লাভ` (`লব`), and `ভিজিট করুন <URL>` (`কর url`) created a different small-sample lexical signature in Batch 2, while `Combined 100` combines both sets' vocabularies against the same `769` real rows. For the remaining batches, mixing natural Bengali conjunctions (`আর`, `ও`, `এবং`, comma clauses) and varying URL call-to-action phrasing (`বিস্তারিত:`, `লগইন:`, bare URLs without repeating `ভিজিট করুন`) will distribute phrasing naturally without over-concentrating any single token.

---

## 6. Updated Stage B Deficit & Next Step (STOP & REPORT)

Target `T = 1,548`:

| Class | `eligible_train` | Pilot (`accepted.jsonl`) | Batch 1 (`batch1_50`) | **Batch 2 (`batch2_50`)** | Combined Total | Target (`T`) | **Remaining Deficit** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 1,509 | 39 | 0 | 0 | 1,548 | 1,548 | **0** |
| `Promotional` | 942 | 261 | 50 | **50** | 1,303 | 1,548 | **245** |
| `Smishing` | 1,548 | 0 | 0 | 0 | 1,548 | 1,548 | **0** |
| **Total** | **3,999** | **300** | **50** | **50** | **4,399** | — | **245** |

- **Exact remaining `Promotional` deficit by language (for equal `387/lang` across `1,548` Promotional rows):**
  - `Banglish`: `387 - (141 + 115 + 19 + 19) = 93`
  - `Bengali`: `387 - (229 + 65 + 13 + 14) = 66`
  - `CodeMix`: `387 - (234 + 60 + 14 + 13) = 66`
  - `English`: `387 - (338 + 21 + 4 + 4) = 20`
  - **Total remaining:** `93 + 66 + 66 + 20 = 245` *(Note: `295` remaining after Batch 1 minus `50` in Batch 2 = `245`, clarifying the earlier `195` sum typo while preserving exact per-language remainders `93/66/66/20`.)*
- **No remaining `245` rows have been generated.** Ready for user review and approval before proceeding with the next reviewable batch.
