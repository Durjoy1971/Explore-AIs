# Stage B Batch 6 (`50` Promotional Rows) Evaluation & Checkpoint Report (`curation_v2`)

**Date:** 2026-11-20  
**Batch Directory (Isolated):** [`data/generated/curation_v2/batch6_50/`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch6_50)  
**Validation Summary JSON:** [`batch6_50_validation_summary.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch6_50/batch6_50_validation_summary.json)  
**Multi-Cohort Diagnostics JSON:** [`combined_cohorts_diagnostics.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch6_50/combined_cohorts_diagnostics.json)  
**Status:** **Batch 6 (`50` Promotional Rows) Independently Reviewed, Validated & Exported — Stopped before generating remaining `45` `Promotional` samples, awaiting explicit user approval.**

---

## 1. Unified Brand Detection, USSD Realism & Length Alignment (Requirements 1–5)

### 1.1 Language Allocation & Unified Brand Detection
- **Exact Batch 6 Language Allocation (`N = 50` Promotional):** `Banglish = 19`, `Bengali = 14`, `CodeMix = 13`, `English = 4` (`19 + 14 + 13 + 4 = 50`).
- **Unified Canonical Brand Detection ([`schema.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/schema.py) & [`codemix_diversity.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/codemix_diversity.py)):**
  - Added the accepted Batch 5 fictional brands (`FICTIONAL_BATCH5_BRANDS`: `8` canonical brands, `20` aliases) into `DEFAULT_KNOWN_BRANDS` and `DEFAULT_CANONICAL_BRAND_ALIASES` in [`schema.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/schema.py), with regression tests in [`tests/test_schema_masking.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/tests/test_schema_masking.py).
  - Registered `8` fresh Batch 6 fictional brands in [`batch6_50/brand_registry.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch6_50/brand_registry.json) (`PiyaliMart` `3`, `NibirTech` `3`, `AkashNet` `3`, `BarshaBank` `2`, `TonuPay` `2`, `KakonBazar` `2`, `UjanCourier` `2`, `ArshiStyle` `2`; `0` collisions with `DEFAULT_KNOWN_BRANDS`), leaving **`31 / 50 (62.00%)`** naturally brandless (`19 / 50 = 38.00%` branded; `max_brand_count = 3`).
  - Verified **`0` mismatches** across `detect_brand_mentions(text)`, `"<BRAND>" in mask_text_views(text)["brand_masked"]`, and `brand_stripped != canon_masked` on all **`942`** real `eligible_train` Promotional rows (`brandless = 657/942 = 69.75%`) and all **`561`** Pilot + Batch 1–6 Promotional rows (`brandless = 177/561 = 31.55%`; `172/300 = 57.33%` across `Combined New B1..B6`).

### 1.2 USSD & Language Realism
- **Batch 6 USSD (`7 / 50 = 14.00%`, `0` `*1234#` / `*১২৩৪#`):**
  - `5` ASCII codes: `*566*19#` (`Banglish`), `*121*36#` (`Banglish`), `*247*44#` (`English`), `*321*24#` (`Bengali`), `*247*69#` (`CodeMix`).
  - `2` Bengali-digit codes: `*১২১*৬২#` (`Bengali`), `*৫৬৬*৪১#` (`CodeMix`) — **`2 / 4 = 50.0%`** Bengali-digit among Bengali/CodeMix USSD rows, matching `eligible_train` Promotional (`32 / 64 = 50.0%` Bengali-digit in `Bn+CM`; `100%` ASCII in `Banglish`/`English`).

### 1.3 Full Character-Length Distribution (`eligible_train` Reference vs. Batch 6 & Combined New `B1..B6`)

| Cohort | `N` | Min | P10 | **P25** | **P50 (Median)** | **P75** | P90 | Max | **Mean** | **SD (`ddof=1`)** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `eligible_train` (`Promotional`) | 942 | 42 | 58.0 | **66.0** | **74.0** | **99.0** | 140.0 | 448 | **89.20** | **40.77** |
| `B1..B5` (`250`) | 250 | 43 | 64.0 | 70.0 | 81.0 | 99.0 | 143.2 | 173 | 90.13 | 29.59 |
| **`Batch 6` (`batch6_50`)** | **50** | **40** | **56.9** | **65.0** | **74.0** | **98.25** | **135.1** | **168** | **86.04** | **31.69** |
| `Combined New B1..B6` | 300 | 40 | 63.0 | 68.75 | 79.5 | 99.0 | 143.0 | 173 | 89.45 | 29.94 |

Fine-grained character-length bins (`right-exclusive`, matching `compute_length_band`):

| Cohort | `<50` | `50-59` | `60-69` | `70-79` | `80-89` | `90-99` | `100-120` | `121-150` | `>150` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `eligible_train` (`942`) | 20 (2.12%) | 91 (9.66%) | 236 (25.05%) | 205 (21.76%) | 70 (7.43%) | 94 (9.98%) | 83 (8.81%) | 65 (6.90%) | 78 (8.28%) |
| **`Batch 6` (`50`)** | **1 (2.00%)** | **5 (10.00%)** | **12 (24.00%)** | **11 (22.00%)** | **5 (10.00%)** | **4 (8.00%)** | **4 (8.00%)** | **4 (8.00%)** | **4 (8.00%)** |
| `Combined New B1..B6` (`300`) | 6 (2.00%) | 10 (3.33%) | 63 (21.00%) | 71 (23.67%) | 41 (13.67%) | 37 (12.33%) | 24 (8.00%) | 27 (9.00%) | 21 (7.00%) |

Batch 6 reproduces the full `eligible_train` Promotional shape (`P25 = 65.0 vs 66.0`, `P50 = 74.0 vs 74.0`, `P75 = 98.25 vs 99.0`, `mean = 86.04 vs 89.20`; `short 1 [2.0%]`, `medium 41 [82.0%]`, `long 8 [16.0%]`) without artificial truncation.

---

## 2. Integrity, Isolation & Cross-Batch Deduplication Verification

1. **Protected Corpora, 300-Row Pilot & Batches 1–5 Untouched (SHA256 Verified):**
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
   - [`batch5_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch5_50/accepted.jsonl): `a0cc662f85097f40767ddc78c85982fd50fab8d71897d76ce553742e6d3f6da1` (`matches_baseline=True`)

2. **Exact & Near-Duplicate (`cos >= 0.85`) Checks (Raw & Slot-Template) for Batch 6 (`N = 50`):**

| Reference Corpus | Reference Rows | Exact Norm Hits | Near-Dup Raw (`>=0.85`) | Max Raw Cos | Near-Dup Slot (`>=0.85`) | Max Slot Cos | Family Overlap |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `curation_v2/existing_unique.parquet` | 11,821 | **0** | **0** | 0.3295 | **0** | 0.4754 | **0** |
| `curation_v2/quarantined.parquet` | 1,094 | **0** | **0** | 0.2701 | **0** | 0.3529 | — |
| `banglaphish2026_dedup.parquet` (OOD) | 5,416 | **0** | **0** | 0.2349 | **0** | 0.2477 | — |
| `curation_v2/accepted.jsonl` (`Pilot`) | 300 | **0** | **0** | 0.3254 | **0** | 0.4883 | **0** |
| `batch1_50/accepted.jsonl` (`Batch 1`) | 50 | **0** | **0** | 0.4475 | **0** | 0.6249 | **0** |
| `batch2_50/accepted.jsonl` (`Batch 2`) | 50 | **0** | **0** | 0.5019 | **0** | 0.5215 | **0** |
| `batch3_50/accepted.jsonl` (`Batch 3`) | 50 | **0** | **0** | 0.3198 | **0** | 0.4814 | **0** |
| `batch4_50/accepted.jsonl` (`Batch 4`) | 50 | **0** | **0** | 0.2994 | **0** | 0.3639 | **0** |
| `batch5_50/accepted.jsonl` (`Batch 5`) | 50 | **0** | **0** | 0.4321 | **0** | 0.5151 | **0** |
| Intra-batch (`batch6_50/accepted.jsonl`) | 50 | **0** | **0** | 0.2982 | **0** | 0.4350 | 50 unique `family_id`s |

- Automated screening ([`scripts/validate_stage_b_pilot.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/scripts/validate_stage_b_pilot.py) `--diversity-mode new_batch`) matches [`batch6_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch6_50/accepted.jsonl) `50/50` (`matches_validation_run = True`).

---

## 3. Independent Row-by-Row Review, Rejected & Replacement Samples

- **Independent Reviewer:** `independent_reviewer_subagent` auditing [`raw_batch6_50.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch6_50/raw_batch6_50.jsonl) and [`replacements.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch6_50/replacements.jsonl) across 3 rounds (`review_round1.jsonl`, `review_round2.jsonl`, `review_round3.jsonl`), merged in [`review_batch6_50.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch6_50/review_batch6_50.jsonl) (`61` review records).
- **Total candidates screened & reviewed:** **61** (`50` initial + `11` replacements)
  - **Round 1 (`raw_batch6_50.jsonl`, 50 rows):** **41 / 50 (82.00%)** accepted, **9 / 50** rejected (`0002`, `0005`, `0008`, `0011`, `0018`, `0029`, `0030`, `0031`, `0044`).
  - **Round 2 (`replacements.jsonl`, `syn_b6_r_0001`..`syn_b6_r_0009`):** **7 / 9 (77.78%)** accepted, **2 / 9** rejected (`syn_b6_r_0001`, `syn_b6_r_0009`).
  - **Round 3 (`replacements.jsonl`, `syn_b6_r_0010`..`syn_b6_r_0011`):** **2 / 2 (100.00%)** accepted.
- **Overall acceptance across 61 reviewed candidates:** **50 / 61 (81.97%)**; **final accepted (`batch6_50/accepted.jsonl`):** **50 / 50** (`review_status = "accepted"`, `0` `pending_independent_review`).

| Rejected ID | Stratum | Rejection Reasons | Final Accepted Replacement | Resolution Summary |
| :--- | :--- | :--- | :--- | :--- |
| `syn_b6_0002` | Bn \| banking \| med \| `TonuPay` | `cross_batch_scenario_clone`, `near_duplicate_slot_template` | `syn_b6_r_0010` *(via `r_0001`)* | `0002` cloned `syn_b2_0032` water+gas bill (`bslot=0.610`); `r_0001` cloned `syn_b3_0032` tuition fee; `r_0010` introduces fresh insurance/health policy installment cashback |
| `syn_b6_0005` | Bn \| telecom \| med \| `AkashNet` | `near_duplicate_slot_template` | `syn_b6_r_0002` | Replaced `<N> টাকায় <N> জিবি` template (`bslot=0.616`) with `AkashNet` postpaid bill 10% discount via `*১২১*৬২# চাপুন` |
| `syn_b6_0008` | CM \| telecom \| med \| brandless | `cross_batch_scenario_clone` | `syn_b6_r_0003` | Replaced 4G pocket Wi-Fi 30GB clone (`syn_p2_b_0034`) with night unlimited audio/video pack via `*৫৬৬*৪১#` |
| `syn_b6_0011` | CM \| telecom \| long \| brandless | `brand_detection_mismatch`, `metadata_mismatch` | `syn_b6_r_0004` | Removed `আজই শেষ দিন:` colon-opener so `detect_brand_mentions` returns `[]` (`আজই শেষ সুযোগ!`) |
| `syn_b6_0018` | Bg \| telecom \| med \| brandless | `near_duplicate_slot_template`, `unnatural_phrasing` | `syn_b6_r_0005` | Replaced verbless `syn_b5_0020` template (`bslot=0.612`) with finite-verb `Saptahik 7GB net ar 100 min matro 99 takay nite parben.` |
| `syn_b6_0029` | Bg \| telecom \| med \| `AkashNet` | `cross_batch_scenario_clone`, `near_duplicate_slot_template` | `syn_b6_r_0006` | Replaced `syn_b3_r_0001` clone (`bslot=0.690`) with `AkashNet-e *121*36# dial kore 68 takay 140 minute kotha bolun.` |
| `syn_b6_0030` | Bg \| shopping \| med \| brandless | `cross_batch_scenario_clone`, `near_duplicate_slot_template` | `syn_b6_r_0007` | Replaced sweater/hoodie clone (`syn_b3_0010`, `raw=0.609`) with formal blazer & waistcoat clearance |
| `syn_b6_0031` | Bg \| telecom \| med \| brandless | `unnatural_phrasing` | `syn_b6_r_0008` | Fixed clipped CTA `, nite example.org/flash25` -> `paben, dekhun example.org/flash25` |
| `syn_b6_0044` | Bg \| shopping \| med \| `NibirTech` | `cross_batch_scenario_clone`, `near_duplicate_slot_template` | `syn_b6_r_0011` *(via `r_0009`)* | `0044` cloned `syn_b4_0043` keyboard+mouse (`bslot=0.681`); `r_0009` cloned `syn_b4_r_0001` USB hub+cooler; `r_0011` introduces NVMe SSD + laptop RAM combo |
| `syn_b6_r_0001` | Bn \| banking \| med \| `TonuPay` | `cross_batch_scenario_clone` | `syn_b6_r_0010` | Round-2 rejection (tuition fee clone of `syn_b3_0032`); replaced in Round 3 |
| `syn_b6_r_0009` | Bg \| shopping \| med \| `NibirTech` | `cross_batch_scenario_clone` | `syn_b6_r_0011` | Round-2 rejection (USB hub + cooling pad clone of `syn_b4_r_0001`); replaced in Round 3 |

---

## 4. Five-Cohort Distribution, Lexical & Unified Diversity Comparison (`Promotional`)

All **5 cohorts** (`Pilot 300`, `B1..B5 250`, `Batch 6 50`, `Combined New B1..B6 300`, `Full Combined 600`) are evaluated using the single unified brand detector in [`schema.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/schema.py) / [`codemix_diversity.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/codemix_diversity.py).

### 4.1 Distribution, Surface Cues & Lexical Comparison

| Dimension | `eligible_train` (`N=942`) | (1) Original Pilot (`N=261`) | (2) `B1..B5` (`N=250`) | **(3) Batch 6 (`N=50`)** | **(4) Comb New `B1..B6` (`N=300`)** | **(5) Full Comb (`N=561`)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Language** | En 338, CM 234, Bn 229, Bg 141 | Bg 115, Bn 65, CM 60, En 21 | Bg 95, CM 68, Bn 67, En 20 | **Bg 19, Bn 14, CM 13, En 4** | **Bg 114, Bn 81, CM 81, En 24** | **Bg 229, Bn 146, CM 141, En 45** |
| **`short(<50)`** | 20 (2.12%) | 36 (13.79%) | 5 (2.00%) | **1 (2.00%)** | **6 (2.00%)** | **42 (7.49%)** |
| **`medium(50-120)`** | 779 (82.70%) | 120 (45.98%) | 205 (82.00%) | **41 (82.00%)** | **246 (82.00%)** | **366 (65.24%)** |
| **`long(>120)`** | 143 (15.18%) | 105 (40.23%) | 40 (16.00%) | **8 (16.00%)** | **48 (16.00%)** | **153 (27.27%)** |
| **P25 / P50 / P75 Chars** | `66.0` / `74.0` / `99.0` | `91.0` / `116.0` / `127.0` | `70.0` / `81.0` / `99.0` | **`65.0` / `74.0` / `98.25`** | **`68.75` / `79.5` / `99.0`** | **`72.0` / `94.0` / `123.0`** |
| **Mean ± SD Chars** | `89.20 ± 40.77` | `106.66 ± 32.86` | `90.13 ± 29.59` | **`86.04 ± 31.69`** | **`89.45 ± 29.94`** | **`97.46 ± 32.46`** |
| **% Has URL** | 29.83% (`281`) | 22.61% (`59`) | 28.80% (`72`) | **30.00%** (`15`) | **29.00%** (`87`) | **26.02%** (`146`) |
| **% Has Phone** | 8.92% (`84`) | 8.81% (`23`) | 8.00% (`20`) | **8.00%** (`4`) | **8.00%** (`24`) | **8.38%** (`47`) |
| **% Has USSD (ASCII / Bn)** | 12.95% (`122`: `90`/`32`) | 5.75% (`15`: `15`/`0`) | 12.80% (`32`: `30`/`2`) | **14.00%** (`7`: **`5`/`2`**) | **13.00%** (`39`: `35`/`4`) | **9.63%** (`54`: `50`/`4`) |
| **% Has Urgency** | 18.58% (`175`)* | 23.37% (`61`) | 8.40% (`21`) | **12.00%** (`6`) | **9.00%** (`27`) | **15.69%** (`88`) |
| **% Brandless (Unified)** | 69.75% (`657`) | 1.92% (`5`) | 56.40% (`141`) | **62.00%** (`31`) | **57.33%** (`172`) | **31.55%** (`177`) |

*\*Note: `eligible_train` `has_urgency` (`18.58%`) reflects our script's expanded urgency regex including `এখনই`; synthetic cohorts report audited row `has_urgency` metadata.*

### 4.2 Domain & Lexical Indicators
- **Batch 6 Domains (`N = 50`):** `telecom` `15 (30.0%)`, `shopping_ecommerce` `14 (28.0%)`, `personal_or_general` `11 (22.0%)`, `banking_mfs` `5 (10.0%)`, `delivery_parcel` `3 (6.0%)`, `civic_utility_health` `1 (2.0%)`, `employment` `1 (2.0%)`.
- **Batch 6 `Bn+CM` (`N = 27`) Lexical Balance:** `' ও '` `18 (66.67%)`, `' এবং '` `3 (11.11%)`, `' আর '` `1 (3.70%)`, `'থাকছে'` `3 (11.11%)`, `'পাবেন'` `5 (18.52%)`, `'পাচ্ছেন'` `1 (3.70%)`, `'পাওয়া/পাওয়া'` `1 (3.70%)`, `'দেখুন'` `4 (14.81%)`, `'বিস্তারিত'` `1 (3.70%)`, `'ভিজিট করুন'`/`'উপলব্ধ'`/`'লাভ'`/`'মিলবে'` `0`.
- **Batch 6 `Banglish` (`N = 19`) Conjunctions:** `' ar '` `10 (52.63%)`, `' ebong '` `0`, `' o '` `4 (21.05%)`.

### 4.3 Automated Diversity Gates (Unified Detector)

| Gate / Metric | Threshold | (1) Pilot (`300`, `audit`) | (2) `B1..B5` (`250`, `new_batch`) | **(3) Batch 6 (`50`, `new_batch`)** | **(4) Comb New (`300`, `new_batch`)** | **(5) Full Comb (`600`, `audit`)** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `codemix_bn_script_en` | `>= 0.80` | 0.5833 (`False`) | 1.0000 (`True`) | **1.0000** (`13/13`, **True**) | **1.0000** (`81/81`, **True**) | **0.8227 (`True`)** |
| `codemix_hook_plus_pure_en` | `== 0.0` / `<= 0.15` | 0.8500 (`False`) | 0.0000 (`True`) | **0.0000** (`0/13`, **True**) | **0.0000** (`0/81`, **True**) | 0.3617 (`False`) |
| `codemix_intra_sentential_mix` | `>= 0.60` | 0.1333 (`False`) | 0.9265 (`True`) | **1.0000** (`13/13`, **True**) | **0.9383** (`76/81`, **True**) | 0.5957 (`False`) |
| `promo_sent1_excl_rate` | `<= 0.35` | 0.5632 (`False`) | 0.0560 (`True`) | **0.0800** (`4/50`, **True**) | **0.0600** (`18/300`, **True**) | **0.2941 (`True`)** |
| `promo_sent1_quest_rate` | `<= 0.05` | 0.0920 (`False`) | 0.0080 (`True`) | **0.0000** (`0/50`, **True**) | **0.0067** (`2/300`, **True**) | **0.0463 (`True`)** |
| `max_brand_count` | `<= 5` | 25 (`False`) | 4 (`True`) | **3** (**True**) | **4** (**True**) | 25 (`False`) |
| `brandless_rate` (unified) | `>= 0.30` | 0.0192 (`False`) | 0.5640 (`True`) | **0.6200** (`31/50`, **True**) | **0.5733** (`172/300`, **True**) | **0.3155 (`True`)** |
| `ussd_1234_share` | `<= 0.25` | 1.0000 (`False`) | 0.0000 (`True`) | **0.0000** (`0/7`, **True**) | **0.0000** (`0/39`, **True**) | 0.2778 (`False`) |
| **Gates Passed** | — | **0 / 8** | **8 / 8** | **8 / 8** | **8 / 8** | **4 / 8** |

- **Opener 3-gram uniqueness:** `max_opener_3gram_count = 1` (`50/50` unique) on `Batch 6 (50)` and `3` across `Combined New (300)`. Note that under unified detection, `Full Combined (600)` now passes `4/8` audit gates (with `brandless_rate = 0.3155 >= 0.30` and `promo_sent1_quest_rate = 0.0463 <= 0.05` crossing threshold).

---

## 5. Surface, 4-View 3-Class, Counterfactual & Repeated-Seed Statistical Diagnostics

### 5.1 Surface-Only, 4-View 3-Class CV & Counterfactual Probes (`n = 1,279` Real Smishing Rows)

| Cohort (`eligible_train +`) | Surface Acc / MF1 | Length Smish AUC | `raw` MF1 / R@1%FPR / CF | `canon_masked` MF1 / R@1% / CF | `brand_masked` MF1 / R@1% / CF | `brand_stripped` MF1 / R@1% / CF |
| :--- | :--- | ---: | :--- | :--- | :--- | :--- |
| `eligible_train` (`3,999`) | 73.14% / 62.86% | 0.7693 | 98.01% / 98.51% / — | 97.86% / 97.80% / — | 97.76% / 97.80% / — | 97.71% / 97.80% / — |
| (1) `+ Pilot 300` (`4,299`) | 69.32% / 62.78% | 0.7341 | 97.55% / 97.61% / `-10.40/-12.82pp` | 97.26% / 95.35% / `+0.00/-0.08pp` | 97.05% / 93.09% / **`+0.00pp`** | 97.24% / 91.93% / **`+0.00pp`** |
| (2) `+ B1..B5` (`4,249`) | 69.99% / 63.00% | 0.7590 | 98.14% / 98.26% / `-5.79/-5.86pp` | 97.79% / 96.83% / **`+0.00pp`** | 97.68% / 97.48% / **`+0.00pp`** | 97.77% / 97.03% / **`+0.00pp`** |
| **(3) `+ Batch 6` (`4,049`)** | **72.49% / 62.81%** | **0.7678** | **98.07% / 98.71% / `-1.64/-1.56pp`** | **97.74% / 97.48% / `+0.00pp`** | **97.68% / 97.61% / `+0.00pp`** | **97.68% / 97.87% / `+0.00pp`** |
| **(4) `+ Comb New 300` (`4,299`)** | **69.09% / 62.54%** | **0.7579** | **98.00% / 98.19% / `-6.49/-6.65pp`** | **97.78% / 97.93% / `+0.00pp`** | **97.76% / 97.48% / `+0.00pp`** | **97.73% / 97.74% / `+0.00pp`** |
| **(5) `+ Full Comb 600` (`4,599`)** | **66.32% / 62.47%** | **0.7273** | **97.69% / 98.06% / `-14.07/-16.50pp`** | **97.35% / 95.87% / `+0.00pp`** | **97.26% / 94.25% / `+0.00pp`** | **97.28% / 94.44% / `+0.00pp`** |

### 5.2 Seed-42 Synthetic-Origin (`is_synthetic`) 4-View Probes

| Probe & Control | Feature View | (1) Pilot (`300`) | (2) `B1..B5` (`250`) | **(3) Batch 6 (`50`)** | **(4) Comb New (`300`)** | **(5) Full Comb (`600`)** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `all_vs_real` | `raw` / `canon` | 0.9979 / 0.9965 | 0.9984 / 0.9970 | **0.9700 / 0.9713** | **0.9985 / 0.9967** | **0.9975 / 0.9949** |
| (vs `3,999` real) | `brand_m` / `brand_s` / `surf` | 0.9916 / 0.9715 / 0.7260 | 0.9952 / 0.9958 / 0.5971 | **0.9717 / 0.9695 / 0.5387** | **0.9954 / 0.9957 / 0.6019** | **0.9916 / 0.9846 / 0.6546** |
| `synth_vs_real_smishing` | `raw` / `canon` | 0.9997 / 0.9986 | 0.9999 / 0.9997 | **0.9966 / 0.9971** | **0.9999 / 0.9997** | **0.9997 / 0.9988** |
| (vs `1,548` real Smish) | `brand_m` / `brand_s` / `surf` | 0.9990 / 0.9929 / 0.7957 | 0.9997 / 0.9998 / 0.7849 | **0.9972 / 0.9970 / 0.7390** | **0.9997 / 0.9997 / 0.7888** | **0.9988 / 0.9966 / 0.7694** |
| `synthetic_promo_vs_real_promo` | `raw` / `canon` | 0.9966 / 0.9940 | 0.9947 / 0.9896 | **0.9567 / 0.9361** | **0.9961 / 0.9917** | **0.9946 / 0.9892** |
| (Uncontrolled vs `942` real) | `brand_m` / `brand_s` / `surf` | 0.9857 / 0.9660 / 0.7167 | 0.9866 / 0.9868 / 0.5416 | **0.9306 / 0.9301 / 0.4906** | **0.9897 / 0.9890 / 0.5378** | **0.9841 / 0.9794 / 0.6243** |
| `promo_stratum_3way` | `raw` / `canon` | 0.9904 / 0.9832 | 0.9816 / 0.9574 | **0.9475 / 0.9095** | **0.9844 / 0.9646** | **0.9821 / 0.9590** |
| `(lang, domain, length_band)` | `brand_m` / `brand_s` / `surf` | 0.9716 / 0.9410 / 0.5087 | 0.9530 / 0.9501 / 0.4883 | **0.9021 / 0.9001 / 0.4928** | **0.9594 / 0.9584 / 0.5127** | **0.9547 / 0.9373 / 0.4772** |
| `promo_stratum_2way` | `raw` / `canon` | 0.9803 / 0.9633 | 0.9628 / 0.9303 | **0.9351 / 0.9017** | **0.9590 / 0.9224** | **0.9630 / 0.9222** |
| `(lang, length_band)` | `brand_m` / `brand_s` / `surf` | 0.9572 / 0.8901 / 0.4595 | 0.9181 / 0.9220 / 0.5095 | **0.8901 / 0.8885 / 0.4782** | **0.9121 / 0.9153 / 0.5203** | **0.9064 / 0.8750 / 0.4626** |

### 5.3 Surface-Only (`surface_8feat`) ROC-AUC Positive-Class Orientation Verification
Before interpreting any `surface_8feat` ROC-AUC values below `0.50`, we explicitly verified positive-class orientation in `verify_surface_auc_orientation` ([`combined_cohorts_diagnostics.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch6_50/combined_cohorts_diagnostics.json)):
- **Positive-class orientation confirmed:** In every fold of every cohort (`all_folds_column1_is_class1 = true`), `clf.classes_ == [0, 1]`, so `predict_proba(X)[:, 1]` is strictly $P(\text{is\_synthetic\_bin} = 1)$ (`1` = synthetic, `0` = real).
- **Batch 6 orientation measurements (`seed=42`):**
  - Uncontrolled (`50` vs `942`): OOF AUC = `0.4906` (`mean P(syn=1)` = `0.4775` for `y=1` vs `0.4754` for `y=0`), full-fit in-sample AUC = `0.6010`.
  - 3-way stratum-controlled (`50/50` syn vs `818` real): weighted-fit OOF AUC = `0.4928` (`mean P(syn=1)` = `0.4968` vs `0.4990`), weighted-eval OOF AUC = `0.5062`, unweighted-fit OOF AUC = `0.5596`, full-fit in-sample AUC = `0.5484`.
  - 2-way stratum-controlled (`50/50` syn vs `890` real): weighted-fit OOF AUC = `0.4782` (`mean P(syn=1)` = `0.4933` vs `0.5016`), weighted-eval OOF AUC = `0.3921`, unweighted-fit OOF AUC = `0.5043`, full-fit in-sample AUC = `0.5583`.
- **Combined New `B1..B6` (`300` Promo) 3-way:** OOF AUC = `0.5127` (`mean P(syn=1)` = `0.5100` vs `0.5017`), unweighted-fit OOF = `0.5670`, weighted-eval OOF = `0.5955`, in-sample = `0.5121`.
- **Full Combined `600` (`561` Promo) 3-way:** OOF AUC = `0.4772` (`mean P(syn=1)` = `0.5139` vs `0.5274`), unweighted-fit OOF = `0.6467`, weighted-eval OOF = `0.6072`, in-sample = `0.4794`.
- **Interpretation:** Values slightly below `0.50` (`0.45–0.48`) are **not** an inverted label bug and do **not** imply "better than random indistinguishability"; rather, when real vs. synthetic surface features are matched within strata (so true separation is near `0.50`), inverse-stratum-count weighting on small-`N` strata induces mild leave-one-fold-out anti-correlation between training and held-out folds.

### 5.4 Repeated-Seed (`10` Seeds: `42..51`), Bootstrap 95% CIs (`B = 1,000`) & Matched-Stratum Coverage

- **Methodology & Uncertainty Definitions:** `mean ± SD` uses sample `ddof=1` across 10 `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)` runs (quantifying CV split-randomization sensitivity on the fixed dataset, not generalization error). Cluster bootstrap CIs (`B=1,000`) resample `family_id` groups within class on 10-seed `mean_oof` probabilities. Cohorts differ in `N_syn` (`50, 250, 261, 300, 561`) and matched-stratum subsets and are **not directly comparable via significance tests**.

| Cohort (`N_syn` Promo) | 3-Way Coverage (`Strata`, `Syn`, `Real`) | 2-Way Coverage (`Strata`, `Syn`, `Real`) | 3-Way `raw` Mean±SD (`clusCI`) | 3-Way `canon_masked` Mean±SD (`clusCI`) | 3-Way `brand_masked` Mean±SD (`clusCI`) | 3-Way `brand_stripped` Mean±SD (`clusCI`) | 3-Way `surface_8feat` Mean±SD (`clusCI`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| (1) Pilot (`261`) | `50/76` (`186/261=71.26%`, `800`) | `11/12` (`243/261=93.10%`, `942`) | `0.9898±0.0016` (`[0.9874, 0.9965]`) | `0.9810±0.0030` (`[0.9791, 0.9913]`) | `0.9688±0.0025` (`[0.9631, 0.9820]`) | `0.9377±0.0074` (`[0.9302, 0.9620]`) | `0.5027±0.0085` (`[0.4639, 0.5484]`) |
| (2) `B1..B5` (`250`) | `37/44` (`238/250=95.20%`, `833`) | `9/10` (`247/250=98.80%`, `931`) | `0.9825±0.0009` (`[0.9784, 0.9912]`) | `0.9574±0.0022` (`[0.9503, 0.9747]`) | `0.9530±0.0027` (`[0.9467, 0.9715]`) | `0.9515±0.0028` (`[0.9437, 0.9705]`) | `0.4895±0.0076` (`[0.4398, 0.5200]`) |
| **(3) Batch 6 (`50`)** | **`29/29` (`50/50=100.0%`, `818`)** | **`8/8` (`50/50=100.0%`, `890`)** | **`0.9492±0.0063` (`[0.9123, 0.9891]`)** | **`0.9170±0.0097` (`[0.8735, 0.9717]`)** | **`0.9104±0.0102` (`[0.8676, 0.9668]`)** | **`0.9079±0.0109` (`[0.8627, 0.9663]`)** | **`0.4542±0.0362` (`[0.3499, 0.5278]`)** |
| **(4) Comb New (`300`)** | `39/46` (`288/300=96.00%`, `881`) | `10/11` (`297/300=99.00%`, `939`) | `0.9845±0.0013` (`[0.9808, 0.9916]`) | `0.9625±0.0025` (`[0.9574, 0.9771]`) | `0.9567±0.0027` (`[0.9510, 0.9730]`) | `0.9561±0.0029` (`[0.9497, 0.9727]`) | `0.5036±0.0067` (`[0.4573, 0.5306]`) |
| **(5) Full Comb (`600`)** | `56/82` (`474/561=84.49%`, `935`) | `11/12` (`540/561=96.26%`, `942`) | `0.9810±0.0011` (`[0.9775, 0.9884]`) | `0.9564±0.0019` (`[0.9513, 0.9709]`) | `0.9517±0.0024` (`[0.9469, 0.9671]`) | `0.9343±0.0022` (`[0.9278, 0.9534]`) | `0.4690±0.0062` (`[0.4371, 0.5010]`) |

*Note: Batch 6 2-way (`100.0%` matched) 10-seed `brand_stripped` AUC is `0.8789±0.0129` (`mean_oof=0.8970`, cluster CI `[0.8434, 0.9437]`).*

---

## 6. Updated Stage B Deficit & Preservation Stop Condition (STOP & REPORT)

Target `T = 1,548` (`387 / language` across all `1,548` `Promotional` rows):

| Class | `eligible_train` | Pilot (`300`) | `B1..B5` (`250`) | **Batch 6 (`50`)** | Combined Total | Target (`T`) | **Remaining Deficit** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 1,509 | 39 | 0 | 0 | 1,548 | 1,548 | **0** |
| `Promotional` | 942 | 261 | 250 | **50** | 1,503 | 1,548 | **45** |
| `Smishing` | 1,548 | 0 | 0 | 0 | 1,548 | 1,548 | **0** |
| **Total** | **3,999** | **300** | **250** | **50** | **4,599** | — | **45** |

- **Exact remaining `Promotional` deficit by language (`45` total):**
  - `Banglish`: `17` | `Bengali`: `12` | `CodeMix`: `12` | `English`: `4` (`17 + 12 + 12 + 4 = 45`).
- **STOP condition enforced:** None of the remaining `45` samples have been generated. All protected corpora, Stage A exports, 300-row Pilot, and Batches 1–5 remain untouched; awaiting explicit user approval before generating Batch 7 (`45` Promotional rows).
