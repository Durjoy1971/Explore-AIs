# Stage B Batch 5 (`50` Promotional Rows) Evaluation & Checkpoint Report (`curation_v2`)

**Date:** 2026-11-19  
**Batch Directory (Isolated):** [`data/generated/curation_v2/batch5_50/`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch5_50)  
**Validation Summary JSON:** [`batch5_50_validation_summary.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch5_50/batch5_50_validation_summary.json)  
**Multi-Cohort Diagnostics JSON:** [`combined_cohorts_diagnostics.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch5_50/combined_cohorts_diagnostics.json)  
**Status:** **Pre-Generation Fixes Verified; Batch 5 (`50` Promotional Rows) Independently Reviewed, Validated & Exported — Stopped before generating remaining `95` `Promotional` samples, awaiting explicit user approval.**

---

## 1. Pre-Generation Fixes & Distribution Alignment (Requirements 1–3)

### 1.1 Unified Brand Detection & Inflection Fix (`schema.py` & `codemix_diversity.py`)
- **Merged `DEFAULT_KNOWN_BRANDS` (`228` aliases across `152+` canonical entities):** Added `FICTIONAL_BATCH1_TO_4_BRANDS` (`37` brands from [`batch1_50/brand_registry.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch1_50/brand_registry.json)–[`batch4_50/brand_registry.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch4_50/brand_registry.json)) and unified `DEFAULT_CANONICAL_BRAND_ALIASES` in [`schema.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/schema.py) and [`codemix_diversity.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/codemix_diversity.py) (including previously omitted Pilot aliases such as `নীড়হাট` -> `neerhaat`, and real BD bilingual pairs such as `বিকাশ` -> `bkash`).
- **Fixed Bengali & Banglish inflection handling:**
  - Updated `compile_brand_regex` with NFC normalization (`unicodedata.normalize("NFC", ...)`) and expanded Bengali nominal/locative suffixes (`-ের`, `-এ`, `-এর`, `-ে`, `-তে`, `-কে`, `-র`, `-য়`, `-য়`, `-য়ে`, `-য়ে`, `-দের`), fixing `শাপলাসেবা` / `শাপলাসেবায়` (`syn_b1_0035`) and NFD/NFC `য়`/`য়` equivalence.
  - Added boundary-safe Banglish suffixes (`-er`, `-e`, `-te`, `-ke`, `-r`, `-y`, `-ye`, `-der` on hyphenated or `>=4`-char single-word brands, plus `'s`/`’s`), while preserving boundary safety so `'Hon'ble'` never matches `BL` and `'রবিবার'` never matches `রবি`.
  - Added generic promotional colon-header stopwords (`"weekend offer"`, `"flash offer"`, etc.) to `_COLON_OPENER_STOPWORDS` so `Weekend Offer:` (`syn_b4_0029`) is not misclassified as a brand.
- **Single detection & masking path:** Added `detect_brand_mentions` in [`schema.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/schema.py) and routed `codemix_diversity._extract_brands_in_text`, `mask_brands`, and `mask_text_views` through the same contact-masking (`mask_shortcuts`), NFC normalization, and compiled brand/base regexes.
  - Verified **0 mismatches** between `detect_brand_mentions(text)`, `"<BRAND>" in mask_text_views(text)["brand_masked"]`, and `brand_stripped != canon_masked` across all **`942`** real `eligible_train` Promotional rows (`brandless = 657/942 = 69.75%`) and all **`461`** Pilot + Batch 1–4 Promotional rows (`Pilot 5/261 = 1.92%`, `B1..B4 110/200 = 55.00%`, `Full Comb 115/461 = 24.95%`).
  - Added regression tests in [`tests/test_schema_masking.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/tests/test_schema_masking.py) and [`tests/test_codemix_diversity.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/tests/test_codemix_diversity.py) (updated Pilot baseline check in [`tests/test_stage_b_pilot.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/tests/test_stage_b_pilot.py#L378-L384) from `0.023` to `0.0192` now that `নীড়হাট` is unified).

### 1.2 Corrected USSD Review Policy (Requirement 2)
- Real `eligible_train` Promotional USSD SMS (`122 / 942 = 12.95%`) split as:
  - `Banglish` (`26/26`) and `English` (`32/32`): **100% ASCII**.
  - `Bengali` (`17` ASCII / `18` Bengali-digit) and `CodeMix` (`15` ASCII / `14` Bengali-digit): **`32/64 = 50.0%` Bengali-digit**.
- Updated generator & independent reviewer policy to accept **both** realistic ASCII and realistic Bengali-digit USSD codes (`*১২১*৪৬#`, `*৫৬৬*২৭#`), rejecting only unrealistic placeholder sequences (`*1234#`, `*১২৩৪#`, `*789*...#`), with Bengali-digit `*1234#` normalization added to `ussd_1234_share` in [`codemix_diversity.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/codemix_diversity.py).
- **Batch 5 USSD (`7 / 50 = 14.0%`, `0` `*1234#`):** `5` ASCII (`*247*38#`, `*121*72#`, `*566*64#`, `*321*45#`, `*247*81#`) + `2` Bengali-digit (`*১২১*৪৬#`, `*৫৬৬*২৭#` — `50%` of Bn+CM USSD rows).

### 1.3 Full Character-Length Distribution Comparison & Matching (Requirement 3)

| Cohort | `N` | Min | P10 | **P25** | **P50 (Median)** | **P75** | P90 | Max | **Mean** | **SD (`ddof=1`)** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `eligible_train` (`Promotional`) | 942 | 42 | 58.0 | **66.0** | **74.0** | **99.0** | 140.0 | 448 | **89.20** | **40.77** |
| `Batch 4` (`batch4_50`) | 50 | 43 | 82.9 | **87.5** | **95.5** | **107.0** | 150.3 | 167 | **102.96** | **25.51** |
| **`Batch 5` (`batch5_50`)** | **50** | **43** | **57.9** | **65.0** | **77.0** | **94.75** | **133.1** | **171** | **86.86** | **32.22** |
| `Combined New B1..B5` | 250 | 43 | 65.0 | 70.0 | 81.0 | 99.0 | 143.2 | 173 | 90.13 | 29.59 |

Fine-grained character-length bins (`right-exclusive`, matching `compute_length_band`):

| Cohort | `<50` | `50-59` | `60-69` | `70-79` | `80-89` | `90-99` | `100-120` | `121-150` | `>150` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `eligible_train` (`942`) | 20 (2.1%) | 91 (9.7%) | 236 (25.1%) | 205 (21.8%) | 70 (7.4%) | 94 (10.0%) | 83 (8.8%) | 65 (6.9%) | 78 (8.3%) |
| `Batch 4` (`50`) | 1 (2.0%) | 0 (0.0%) | 0 (0.0%) | 2 (4.0%) | 12 (24.0%) | 15 (30.0%) | 12 (24.0%) | 3 (6.0%) | 5 (10.0%) |
| **`Batch 5` (`50`)** | **1 (2.0%)** | **5 (10.0%)** | **12 (24.0%)** | **8 (16.0%)** | **6 (12.0%)** | **8 (16.0%)** | **2 (4.0%)** | **4 (8.0%)** | **4 (8.0%)** |

Batch 4 had `0/50` rows in `50-69` chars and `78%` concentrated in `80-120` chars (`median=95.5`); Batch 5 matches the full real shape (`P25=65.0 vs 66.0`, `P50=77.0 vs 74.0`, `P75=94.75 vs 99.0`, `mean=86.86 vs 89.20`) with natural concise SMS phrasing (`52.0%` in `50-79` vs `56.5%` in `eligible_train`).

---

## 2. Integrity, Isolation & Cross-Batch Deduplication Verification

1. **Protected Corpora, 300-Row Pilot & Batches 1–4 Untouched (SHA256 Verified):**
   - [`eligible_train.parquet`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/curation_v2/eligible_train.parquet): `d9af8bf952558b91b45db5b5f721da2735ec4b18547cdca59d86e21a8b4993e7` (`matches_baseline=True`)
   - [`existing_unique.parquet`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/curation_v2/existing_unique.parquet): `acb40c57870b8987b72a920e914d9e81966201caa26867456437a389d88b3e4f` (`matches_baseline=True`)
   - [`provenance.parquet`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/curation_v2/provenance.parquet): `9a0fc3f2de54a7db4f040983504f5907b592dec751db9cb1dfcc07d99c21d443` (`matches_baseline=True`)
   - [`quarantined.parquet`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/curation_v2/quarantined.parquet): `c5cb0855abe2f2998e91d6affb8df7a2ae8984daeacb5373a63d4a1294b17cb4` (`matches_baseline=True`)
   - [`banglaphish2026_dedup.parquet`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/banglaphish2026_dedup.parquet): `16aaa636aa8cf89cd07b8dd44887c11a7cc0398340c6698d36c2bbb91651c1f1` (`matches_baseline=True`)
   - [`accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/accepted.jsonl) (`Pilot 300`): `23af149826dec2411d79b61abceda44c564387574c97d7d12468d7f5a101c5ac` (`matches_baseline=True`)
   - [`batch1_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch1_50/accepted.jsonl): `f6bc19c3b0466ac3ce327a51ee88f7e42c49668246a345183ee96c482c507dbd` (`matches_baseline=True`)
   - [`batch2_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch2_50/accepted.jsonl): `5f4cb5afb8aaaf0ce37ab4cedb64639df6fb462d5f0f872e8444c7e85a6d32a6` (`matches_baseline=True`)
   - [`batch3_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch3_50/accepted.jsonl): `abd5c9d931af3a2f9e84ac21d8a006554ae11bc17059729af19a33efe38b60a1` (`matches_baseline=True`)
   - [`batch4_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch4_50/accepted.jsonl): `a97f0a912bad36ced9c5f104f2a4596a8b27f875a074bff4bf487cbce9ecbd64` (`matches_baseline=True`)

2. **Exact & Near-Duplicate (`cos >= 0.85`) Checks (Raw & Slot-Template) for Batch 5 (`N = 50`):**

| Reference Corpus | Reference Rows | Exact Norm Hits | Near-Dup Raw (`>=0.85`) | Max Raw Cos | Near-Dup Slot (`>=0.85`) | Max Slot Cos | Family Overlap |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `curation_v2/existing_unique.parquet` | 11,821 | **0** | **0** | 0.3924 | **0** | 0.4925 | **0** |
| `curation_v2/quarantined.parquet` | 1,094 | **0** | **0** | 0.3568 | **0** | 0.3590 | — |
| `banglaphish2026_dedup.parquet` (OOD) | 5,416 | **0** | **0** | 0.2253 | **0** | 0.2295 | — |
| `curation_v2/accepted.jsonl` (`Pilot`) | 300 | **0** | **0** | 0.3337 | **0** | 0.4514 | **0** |
| `batch1_50/accepted.jsonl` (`Batch 1`) | 50 | **0** | **0** | 0.3961 | **0** | 0.5176 | **0** |
| `batch2_50/accepted.jsonl` (`Batch 2`) | 50 | **0** | **0** | 0.3292 | **0** | 0.5195 | **0** |
| `batch3_50/accepted.jsonl` (`Batch 3`) | 50 | **0** | **0** | 0.3236 | **0** | 0.4513 | **0** |
| `batch4_50/accepted.jsonl` (`Batch 4`) | 50 | **0** | **0** | 0.3166 | **0** | 0.4925 | **0** |
| Intra-batch (`batch5_50/accepted.jsonl`) | 50 | **0** | **0** | 0.3927 | **0** | 0.4832 | 50 unique `family_id`s |

- Automated screening ([`scripts/validate_stage_b_pilot.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/scripts/validate_stage_b_pilot.py) `--diversity-mode new_batch`) matches [`batch5_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch5_50/accepted.jsonl) `50/50` (`matches_validation_run = True`).

---

## 3. Independent Row-by-Row Review, Rejected & Replacement Samples

- **Independent Reviewer:** `independent_reviewer_subagent` (`independent_batch5_reviewer`) auditing [`raw_batch5_50.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch5_50/raw_batch5_50.jsonl) and [`replacements.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch5_50/replacements.jsonl) across 2 rounds, logged in [`review_batch5_50.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch5_50/review_batch5_50.jsonl) (`58` review records).
- **Total candidates screened & reviewed:** **58** (`50` initial + `8` replacements)
  - **Round 1 (`raw_batch5_50.jsonl`, 50 rows):** Independent reviewer accepted **43 / 50 (86.0%)** and rejected **7 / 50** (`0001, 0003, 0012, 0016, 0028, 0047, 0049`); automated screening (`validate_stage_b_pilot.py`) additionally flagged `syn_b5_0039` (`banglish_appears_pure_english`), yielding **42 / 50 (84.0%)** combined Round 1 pass and **8** replacements.
  - **Round 2 (`replacements.jsonl`, `syn_b5_r_0001`..`syn_b5_r_0008`):** **8 / 8** accepted by both independent review and automated screening (**100.0%**).
- **Overall acceptance across 58 screened candidates:** **50 / 58 (86.21%)**; **final accepted (`batch5_50/accepted.jsonl`):** **50 / 50** (`review_status = "accepted"`, `0` `pending_independent_review`).

| Rejected ID | Stratum | Rejection Reasons | Accepted Replacement | Fix Applied |
| :--- | :--- | :--- | :--- | :--- |
| `syn_b5_0001` | Bg \| telecom \| med \| brandless | `independent_review:metadata_mismatch` | `syn_b5_r_0001` | Fixed `has_link=false -> true` for `example.net/sg` |
| `syn_b5_0003` | CM \| telecom \| med \| brandless | `independent_review:unnatural_phrasing` | `syn_b5_r_0002` | Fixed calque `সামাজিক প্যাক` -> `social media প্যাক` and added CTA `ডায়াল করুন *৫৬৬*২৭#` |
| `syn_b5_0012` | Bn \| personal \| long \| brandless | `independent_review:domain_mismatch` | `syn_b5_r_0003` | Replaced dental clinic offer (belongs in `civic_utility_health`) with home interior painting & waterproofing offer |
| `syn_b5_0016` | Bg \| delivery \| med \| `ToriCourier` | `cross_batch_scenario_clone`, `unnatural_phrasing` | `syn_b5_r_0004` | Replaced Sylhet/Chattogram clone (`syn_p2_b_0066`/`b4_0049`) with express home pickup discount |
| `syn_b5_0028` | Bg \| personal \| long \| brandless | `cross_batch_scenario_clone` | `syn_b5_r_0005` | Replaced Gazipur resort clone (`syn_b3_0015`) with weekend music hobby club batch discount |
| `syn_b5_0039` | Bg \| shopping \| med \| `BeliMart` | `banglish_appears_pure_english` (auto) | `syn_b5_r_0006` | Replaced `seasonal off, shop` with explicit Banglish `chhar paben, dekhun` |
| `syn_b5_0047` | Bg \| personal \| med \| `KanchanBazar` | `metadata_mismatch`, `length_band_mismatch` | `syn_b5_r_0007` | Expanded from `48` to `55` chars (`KanchanBazar-e mishti o doi gift box-e 12% chhar paben.`) |
| `syn_b5_0049` | Bg \| banking \| long \| `KarnaphuliBank` | `metadata_mismatch`, `length_band_mismatch` | `syn_b5_r_0008` | Expanded from `118` to `132` chars (`long(>120)`) and smoothed triple `-e` locatives |

---

## 4. Five-Cohort Distribution, Lexical & Unified Diversity Comparison (`Promotional`)

All **5 cohorts** (`Pilot 300`, `B1..B4 200`, `Batch 5 50`, `Combined New B1..B5 250`, `Full Combined 550`) are reported using the single unified brand detector in [`schema.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/schema.py) / [`codemix_diversity.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/codemix_diversity.py).

### 4.1 Distribution, Surface Cues & Lexical Comparison

| Dimension | `eligible_train` (`N=942`) | (1) Original Pilot (`N=261`) | (2) `B1..B4` (`N=200`) | **(3) Batch 5 (`N=50`)** | **(4) Comb New `B1..B5` (`N=250`)** | **(5) Full Comb (`N=511`)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Language** | En 338, CM 234, Bn 229, Bg 141 | Bg 115, Bn 65, CM 60, En 21 | Bg 76, Bn 54, CM 54, En 16 | **Bg 19, CM 14, Bn 13, En 4** | **Bg 95, CM 68, Bn 67, En 20** | **Bg 210, Bn 132, CM 128, En 41** |
| **`short(<50)`** | 20 (2.12%) | 36 (13.79%) | 4 (2.00%) | **1 (2.00%)** | **5 (2.00%)** | **41 (8.02%)** |
| **`medium(50-120)`** | 779 (82.70%) | 120 (45.98%) | 164 (82.00%) | **41 (82.00%)** | **205 (82.00%)** | **325 (63.60%)** |
| **`long(>120)`** | 143 (15.18%) | 105 (40.23%) | 32 (16.00%) | **8 (16.00%)** | **40 (16.00%)** | **145 (28.38%)** |
| **P25 / P50 / P75 Chars** | `66.0` / `74.0` / `99.0` | `91.0` / `116.0` / `127.0` | `71.0` / `82.0` / `99.25` | **`65.0` / `77.0` / `94.75`** | **`70.0` / `81.0` / `99.0`** | **`73.0` / `96.0` / `124.0`** |
| **Mean ± SD Chars** | `89.20 ± 40.77` | `106.66 ± 32.86` | `90.95 ± 28.93` | **`86.86 ± 32.22`** | **`90.13 ± 29.59`** | **`98.58 ± 32.35`** |
| **% Has URL** | 29.83% (`281`) | 22.61% (`59`) | 28.50% (`57`) | **30.00%** (`15`) | **28.80%** (`72`) | **25.64%** (`131`) |
| **% Has Phone** | 8.92% (`84`) | 8.81% (`23`) | 8.00% (`16`) | **8.00%** (`4`) | **8.00%** (`20`) | **8.41%** (`43`) |
| **% Has USSD (ASCII / Bn)** | 12.95% (`122`: `90`/`32`) | 5.75% (`15`: `15`/`0`) | 12.50% (`25`: `25`/`0`) | **14.00%** (`7`: **`5`/`2`**) | **12.80%** (`32`: `30`/`2`) | **9.20%** (`47`: `45`/`2`) |
| **% Has Urgency** | 18.58% (`175`)* | 23.37% (`61`) | 8.00% (`16`) | **10.00%** (`5`) | **8.40%** (`21`) | **16.05%** (`82`) |
| **% Brandless (Unified)** | 69.75% (`657`)** | 1.92% (`5`) | 55.00% (`110`) | **62.00%** (`31`) | **56.40%** (`141`) | **28.57%** (`146`) |

*\*Note: `eligible_train` `has_urgency` (`18.58%`) reflects our script's expanded urgency regex including `এখনই`; synthetic cohorts report audited row `has_urgency` metadata (`10.00%` in Batch 5).*  
*\*\*Note: Under unified `detect_brand_mentions` (running `mask_shortcuts` first so URLs like `mygp.li/...` are `<URL>` rather than prose brands), `eligible_train` Promo has `657/942 = 69.75%` brandless in prose, matching `brand_masked` and `brand_stripped` with `0` mismatches.*

### 4.2 Domain & Lexical Indicators Across All Five Cohorts

- **Batch 5 Domains (`N = 50`):** `telecom` `15 (30.0%)`, `shopping_ecommerce` `14 (28.0%)`, `personal_or_general` `11 (22.0%)`, `banking_mfs` `5 (10.0%)`, `delivery_parcel` `3 (6.0%)`, `civic_utility_health` `1 (2.0%)`, `education` `1 (2.0%)`.
- **Batch 5 `Bn+CM` (`N = 27`) Lexical Balance:** `' ও '` `7 (25.93%)` (down from `70.37%` in Batch 4), `' এবং '` `2 (7.41%)` (vs real `8.64%`), `' আর '` `3 (11.11%)`, `'থাকছে'` `2 (7.41%)`, `'পাবেন'` `5 (18.52%)`, `'পাচ্ছেন'` `4 (14.81%)`, `'দেখুন'` `5 (18.52%)`, `'ভিজিট করুন'` `1 (3.70%)`, `'বিস্তারিত'` `2 (7.41%)`, `'উপলব্ধ'`/`'লাভ'`/`'মিলবে'` `0`.
- **Batch 5 `Banglish` (`N = 19`) Conjunctions:** `' ar '` `5 (26.32%)` (down from `52.63%` in Batch 4), `' ebong '` `1 (5.26%)`, `' o '` `4 (21.05%)`.

### 4.3 Automated Diversity Gates (Unified Detector)

| Gate / Metric | Threshold | (1) Pilot (`300`, `audit`) | (2) `B1..B4` (`200`, `new_batch`) | **(3) Batch 5 (`50`, `new_batch`)** | **(4) Comb New (`250`, `new_batch`)** | **(5) Full Comb (`550`, `audit`)** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `codemix_bn_script_en` | `>= 0.80` | 0.5833 (`False`) | 1.0000 (`True`) | **1.0000** (`14/14`, **True**) | **1.0000** (`68/68`, **True**) | **0.8047 (`True`)** |
| `codemix_hook_plus_pure_en` | `== 0.0` / `<= 0.15` | 0.8500 (`False`) | 0.0000 (`True`) | **0.0000** (`0/14`, **True**) | **0.0000** (`0/68`, **True**) | 0.3984 (`False`) |
| `codemix_intra_sentential_mix` | `>= 0.60` | 0.1333 (`False`) | 0.9074 (`True`) | **1.0000** (`14/14`, **True**) | **0.9265** (`63/68`, **True**) | 0.5547 (`False`) |
| `promo_sent1_excl_rate` | `<= 0.35` | 0.5632 (`False`) | 0.0450 (`True`) | **0.1000** (`5/50`, **True**) | **0.0560** (`14/250`, **True**) | **0.3151 (`True`)** |
| `promo_sent1_quest_rate` | `<= 0.05` | 0.0920 (`False`) | 0.0050 (`True`) | **0.0200** (`1/50`, **True**) | **0.0080** (`2/250`, **True**) | 0.0509 (`False`) |
| `max_brand_count` | `<= 5` | 25 (`False`) | 4 (`True`) | **3** (**True**) | **4** (**True**) | 25 (`False`) |
| `brandless_rate` (unified) | `>= 0.30` | 0.0192 (`False`) | 0.5500 (`True`) | **0.6200** (`31/50`, **True**) | **0.5640** (`141/250`, **True**) | 0.2857 (`False`) |
| `ussd_1234_share` | `<= 0.25` | 1.0000 (`False`) | 0.0000 (`True`) | **0.0000** (`0/7`, **True**) | **0.0000** (`0/32`, **True**) | 0.3191 (`False`) |
| **Gates Passed** | — | **0 / 8** | **8 / 8** | **8 / 8** | **8 / 8** | **2 / 8** |

- **Batch 5 fresh fictional brands (`8` registered brands, `19` branded rows = `38.0%`, `31` brandless = `62.0%`, `0` collisions):** `SurjoTel` (`3`), `KajolTech` (`3`), `BeliMart` (`3`), `KarnaphuliBank` (`2`), `ChampaStyle` (`2`), `KomolPay` (`2`), `ToriCourier` (`2`), `KanchanBazar` (`2`), registered in [`batch5_50/brand_registry.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch5_50/brand_registry.json). *(Note: `extract_dynamic_brands` also tagged `"GaN"` on `syn_b5_0027`'s `"65W GaN fast charger"`, yielding `unique_brands_count=9` with no effect on `brandless_rate`.)*
- **Opener 3-gram uniqueness:** `max_opener_3gram_count = 1` on `Batch 5 (50)` and `2` across `Combined New (250)`.

---

## 5. Surface, 4-View 3-Class, Counterfactual & Repeated-Seed Statistical Diagnostics

### 5.1 Surface-Only, 4-View 3-Class CV & Counterfactual Probes (`n = 1,279` Real Smishing Rows)

| Cohort (`eligible_train +`) | Surface Acc / MF1 | Length Smish AUC | `raw` MF1 / R@1%FPR / CF | `canon_masked` MF1 / R@1% / CF | `brand_masked` MF1 / R@1% / CF | `brand_stripped` MF1 / R@1% / CF |
| :--- | :--- | ---: | :--- | :--- | :--- | :--- |
| `eligible_train` (`3,999`) | 73.14% / 62.86% | 0.7693 | 98.01% / 98.51% / — | 97.86% / 97.80% / — | 97.76% / 97.80% / — | 97.71% / 97.80% / — |
| (1) `+ Pilot 300` (`4,299`) | 69.32% / 62.78% | 0.7341 | 97.55% / 97.61% / `-10.40/-12.82pp` | 97.26% / 95.35% / `0.00/-0.08pp` | 97.05% / 93.09% / **`0.00pp`** | 97.24% / 91.93% / **`0.00pp`** |
| (2) `+ B1..B4` (`4,199`) | 70.40% / 62.81% | 0.7603 | 98.09% / 97.48% / `-5.47/-5.55pp` | 97.85% / 97.48% / **`0.00pp`** | 97.76% / 97.42% / **`0.00pp`** | 97.80% / 97.42% / **`0.00pp`** |
| **(3) `+ Batch 5` (`4,049`)** | **72.56% / 62.97%** | **0.7677** | **98.15% / 98.84% / `-1.80/-1.72pp`** | **97.77% / 97.74% / `0.00pp`** | **97.66% / 97.61% / `0.00pp`** | **97.66% / 97.74% / `0.00pp`** |
| **(4) `+ Comb New 250` (`4,249`)** | **69.99% / 63.00%** | **0.7590** | **98.14% / 98.26% / `-5.79/-5.86pp`** | **97.79% / 96.83% / `0.00pp`** | **97.68% / 97.48% / `0.00pp`** | **97.77% / 97.03% / `0.00pp`** |
| **(5) `+ Full Comb 550` (`4,549`)** | **66.85% / 62.71%** | **0.7278** | **97.49% / 97.55% / `-11.49/-13.53pp`** | **97.14% / 95.74% / `0.00pp`** | **97.12% / 95.03% / `0.00pp`** | **97.06% / 94.19% / `0.00pp`** |

### 5.2 Seed-42 Synthetic-Origin (`is_synthetic`) 4-View Probes

| Probe & Control | Feature View | (1) Pilot (`300`) | (2) `B1..B4` (`200`) | **(3) Batch 5 (`50`)** | **(4) Comb New (`250`)** | **(5) Full Comb (`550`)** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `all_vs_real` | `raw` / `canon` | 0.9979 / 0.9965 | 0.9983 / 0.9966 | **0.9921 / 0.9815** | **0.9984 / 0.9970** | **0.9972 / 0.9941** |
| (vs `3,999` real) | `brand_m` / `brand_s` / `surf` | 0.9916 / 0.9715 / 0.7260 | 0.9952 / 0.9957 / 0.6131 | **0.9783 / 0.9794 / 0.4965** | **0.9952 / 0.9958 / 0.5971** | **0.9903 / 0.9821 / 0.6660** |
| `synth_vs_real_smishing` | `raw` / `canon` | 0.9997 / 0.9986 | 0.9998 / 0.9995 | **0.9980 / 0.9980** | **0.9999 / 0.9997** | **0.9997 / 0.9987** |
| (vs `1,548` real Smish) | `brand_m` / `brand_s` / `surf` | 0.9990 / 0.9929 / 0.7957 | 0.9995 / 0.9995 / 0.7724 | **0.9977 / 0.9980 / 0.8047** | **0.9997 / 0.9998 / 0.7849** | **0.9987 / 0.9959 / 0.7713** |
| `synthetic_promo_vs_real_promo` | `raw` / `canon` | 0.9966 / 0.9940 | 0.9937 / 0.9871 | **0.9702 / 0.9384** | **0.9947 / 0.9896** | **0.9958 / 0.9911** |
| (Uncontrolled vs `942` real) | `brand_m` / `brand_s` / `surf` | 0.9857 / 0.9660 / 0.7167 | 0.9847 / 0.9827 / 0.5282 | **0.9299 / 0.9310 / 0.5032** | **0.9866 / 0.9868 / 0.5416** | **0.9857 / 0.9807 / 0.6342** |
| `promo_stratum_3way` | `raw` / `canon` | 0.9904 / 0.9832 | 0.9786 / 0.9465 | **0.9495 / 0.9003** | **0.9816 / 0.9574** | **0.9799 / 0.9560** |
| `(lang, domain, length_band)` | `brand_m` / `brand_s` / `surf` | 0.9716 / 0.9410 / 0.5087 | 0.9423 / 0.9389 / 0.4796 | **0.8881 / 0.8902 / 0.4117** | **0.9530 / 0.9501 / 0.4883** | **0.9504 / 0.9332 / 0.4694** |
| `promo_stratum_2way` | `raw` / `canon` | 0.9803 / 0.9633 | 0.9683 / 0.9305 | **0.9397 / 0.8840** | **0.9628 / 0.9303** | **0.9627 / 0.9222** |
| `(lang, length_band)` | `brand_m` / `brand_s` / `surf` | 0.9572 / 0.8901 / 0.4595 | 0.9210 / 0.9230 / 0.5138 | **0.8756 / 0.8764 / 0.4048** | **0.9181 / 0.9220 / 0.5095** | **0.9071 / 0.8725 / 0.4598** |

### 5.3 Repeated-Seed (`10` Seeds: `42..51`), Bootstrap 95% CIs (`B = 1,000`) & Matched-Stratum Coverage

- **Methodology & Uncertainty Definitions:** `mean ± SD` uses sample `ddof=1` across 10 `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)` runs (quantifying CV split-randomization sensitivity on the fixed dataset, not generalization error). Row-level and `family_id`-cluster stratified percentile bootstraps (`B=1,000`) are computed on both `seed=42` OOF and 10-seed `mean_oof` probabilities. Cohorts differ in `N_syn` (`50, 200, 250, 261, 511`) and matched-stratum subsets and are **not directly comparable via statistical significance tests**.

| Cohort (`N_syn` Promo) | 3-Way Coverage (`Strata`, `Syn`, `Real`) | 2-Way Coverage (`Strata`, `Syn`, `Real`) | 3-Way `raw` Mean±SD (`clusCI`) | 3-Way `canon_masked` Mean±SD (`clusCI`) | 3-Way `brand_masked` Mean±SD (`clusCI`) | 3-Way `brand_stripped` Mean±SD (`clusCI`) | 3-Way `surface_8feat` Mean±SD (`clusCI`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| (1) Pilot (`261`) | `50/76` (`186/261=71.3%`, `800`) | `11/12` (`243/261=93.1%`, `942`) | `0.9898±0.0016` (`[0.9874, 0.9965]`) | `0.9810±0.0030` (`[0.9791, 0.9913]`) | `0.9688±0.0025` (`[0.9631, 0.9820]`) | `0.9377±0.0074` (`[0.9302, 0.9620]`) | `0.5027±0.0085` (`[0.4639, 0.5484]`) |
| (2) `B1..B4` (`200`) | `35/40` (`191/200=95.5%`, `816`) | `9/10` (`197/200=98.5%`, `931`) | `0.9794±0.0014` (`[0.9746, 0.9890]`) | `0.9494±0.0026` (`[0.9390, 0.9689]`) | `0.9444±0.0032` (`[0.9339, 0.9653]`) | `0.9429±0.0031` (`[0.9318, 0.9638]`) | `0.4870±0.0069` (`[0.4429, 0.5223]`) |
| **(3) Batch 5 (`50`)** | **`22/25` (`47/50=94.0%`, `688`)** | **`9/9` (`50/50=100.0%`, `931`)** | **`0.9492±0.0051` (`[0.9043, 0.9908]`)** | **`0.8952±0.0082` (`[0.8414, 0.9599]`)** | **`0.8971±0.0085` (`[0.8476, 0.9605]`)** | **`0.8893±0.0089` (`[0.8358, 0.9555]`)** | **`0.4024±0.0252` (`[0.2731, 0.4522]`)** |
| **(4) Comb New (`250`)** | `37/44` (`238/250=95.2%`, `833`) | `9/10` (`247/250=98.8%`, `931`) | `0.9825±0.0009` (`[0.9784, 0.9912]`) | `0.9574±0.0022` (`[0.9503, 0.9747]`) | `0.9502±0.0027` (`[0.9430, 0.9697]`) | `0.9516±0.0028` (`[0.9440, 0.9707]`) | `0.4895±0.0076` (`[0.4398, 0.5200]`) |
| **(5) Full Comb (`550`)** | `56/82` (`424/511=83.0%`, `935`) | `11/12` (`490/511=95.9%`, `942`) | `0.9793±0.0016` (`[0.9761, 0.9879]`) | `0.9540±0.0038` (`[0.9496, 0.9696]`) | `0.9467±0.0033` (`[0.9414, 0.9637]`) | `0.9290±0.0046` (`[0.9233, 0.9499]`) | `0.4673±0.0049` (`[0.4323, 0.4996]`) |

*Note: Batch 5 2-way (`100.0%` matched) 10-seed `brand_stripped` AUC is `0.8722±0.0124` (`mean_oof=0.8868`, cluster CI `[0.8256, 0.9381]`).*

---

## 6. Updated Stage B Deficit & Preservation Stop Condition (STOP & REPORT)

Target `T = 1,548` (`387 / language` across all `1,548` `Promotional` rows):

| Class | `eligible_train` | Pilot (`300`) | `B1..B4` (`200`) | **Batch 5 (`50`)** | Combined Total | Target (`T`) | **Remaining Deficit** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 1,509 | 39 | 0 | 0 | 1,548 | 1,548 | **0** |
| `Promotional` | 942 | 261 | 200 | **50** | 1,453 | 1,548 | **95** |
| `Smishing` | 1,548 | 0 | 0 | 0 | 1,548 | 1,548 | **0** |
| **Total** | **3,999** | **300** | **200** | **50** | **4,549** | — | **95** |

- **Exact remaining `Promotional` deficit by language (`95` total):**
  - `Banglish`: `36` | `Bengali`: `26` | `CodeMix`: `25` | `English`: `8` (`36 + 26 + 25 + 8 = 95`).
- **No remaining `95` rows have been generated.** All prior Stage A, Pilot, Batch 1–4, and held-out/OOD files are preserved untouched; stopped awaiting explicit user approval.
