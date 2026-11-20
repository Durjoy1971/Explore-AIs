# Stage B Batch 4 (`50` Promotional Rows) Evaluation & Checkpoint Report (`curation_v2`)

**Date:** 2026-11-19  
**Batch Directory (Isolated):** [`data/generated/curation_v2/batch4_50/`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch4_50)  
**Pre-Generation Rejection Audit:** [`reports/stage_b_batch3_rejection_audit.md`](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/stage_b_batch3_rejection_audit.md)  
**Validation Summary JSON:** [`batch4_50_validation_summary.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch4_50/batch4_50_validation_summary.json)  
**Multi-Cohort Diagnostics JSON:** [`combined_cohorts_diagnostics.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch4_50/combined_cohorts_diagnostics.json)  
**Status:** **Evaluated & Passed All Automated + Independent Review Gates — Stopped before generating remaining `145` `Promotional` samples, awaiting explicit user approval.**

---

## 1. Pre-Generation Rejection Audit Summary (Batch 3 -> Batch 4)

Before Batch 4 generation began, all **18** rejected Batch 3 candidates (`17` Round 1 + `1` Round 2 `syn_b3_r_0010` out of `68` reviewed, `73.53%` acceptance) were audited in [`reports/stage_b_batch3_rejection_audit.md`](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/stage_b_batch3_rejection_audit.md):

- **Rejection Categories (`29` tags across `18` rows):** `unnatural_phrasing` (`10`), `cross_batch_scenario_clone` (`6`), `shallow_template_clone` (`5`), `near_duplicate_slot_template` (`2`), `unrealistic_ussd_digits` (`2`), `near_duplicate_paraphrase` (`1`), `weak_promotional_cue` (`1`), `metadata_mismatch` (`1`), `domain_mismatch` (`1`).
- **Root Causes Identified:**
  1. **Positional skeleton anchoring (`9` clones):** `raw_batch3_50.jsonl` matched `batch2_50`'s exact `(language, domain)` row ordering in **48/50** positions and `length_band` in **46/50** positions, producing slot-swapped clones (`POS 12, 18, 20, 21, 29, 35, 37, 47, 49`) whose `cos` (`0.21–0.70`) passed the `0.85` automated gate.
  2. **Over-aggressive lexical avoidance (`5` phrasing errors):** Suppressing `dial`, `থাকছে`, `মিলবে`, `পাওয়া যাচ্ছে`, `ভিজিট করুন`, and `পর্যন্ত` induced unnatural swaps (`activate *566*33#`, `check <URL>`, `অবধি ... চালু হয়েছে`, `চলছে এখন`, `Order at` for salon booking), alongside dropped Bengali/Banglish postpositions (`মেয়াদে`/`জন্য`) and one `education` row mislabeled as `employment` (`syn_b3_0046`).
  3. **Reviewer criterion calibration (`17` valid vs. `1` borderline):** `17/18` rejections were valid quality catches; `syn_b3_0022` (`*৩২১*১৫#`) was flagged solely for Bengali-digit USSD despite natural prose (`eligible_train.parquet` Bengali/CodeMix USSD Promotional SMS split `50/50`: `32` ASCII vs `32` Bengali-digit).
- **Batch 4 Process Improvements Applied:** Shuffled `(language, domain, length_band)` order (breaking POS `1..50` skeletons), pre-review `bslot_cos < 0.55` / `raw_cos < 0.45` clone screening, natural usage of `"ও"`/`"আর"`/`"এবং"`/`"পর্যন্ত"`/`"থাকছে"`/`"পাবেন"`/`"dial"`, explicit duration postposition checks, and generator/reviewer alignment on ASCII USSD digits. Round 1 acceptance improved from **66.0% (`33/50`)** in Batch 3 to **80.0% (`40/50`)** in Batch 4 (`79.37% = 50/63` overall).

---

## 2. Integrity, Isolation & Cross-Batch Deduplication Verification

1. **Protected Corpora, 300-Row Pilot & Batches 1–3 Untouched (SHA256 Verified):**
   - [`data/processed/curation_v2/eligible_train.parquet`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/curation_v2/eligible_train.parquet) (`3,999` rows): `d9af8bf952558b91b45db5b5f721da2735ec4b18547cdca59d86e21a8b4993e7`
   - [`data/processed/curation_v2/existing_unique.parquet`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/curation_v2/existing_unique.parquet) (`11,821` rows): `acb40c57870b8987b72a920e914d9e81966201caa26867456437a389d88b3e4f`
   - [`data/processed/curation_v2/provenance.parquet`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/curation_v2/provenance.parquet) (`18,357` rows): `9a0fc3f2de54a7db4f040983504f5907b592dec751db9cb1dfcc07d99c21d443`
   - [`data/processed/curation_v2/quarantined.parquet`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/curation_v2/quarantined.parquet) (`1,094` rows): `c5cb0855abe2f2998e91d6affb8df7a2ae8984daeacb5373a63d4a1294b17cb4`
   - [`data/processed/banglaphish2026_dedup.parquet`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/banglaphish2026_dedup.parquet) (`5,416` rows): `16aaa636aa8cf89cd07b8dd44887c11a7cc0398340c6698d36c2bbb91651c1f1`
   - [`data/generated/curation_v2/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/accepted.jsonl) (`300` rows): `23af149826dec2411d79b61abceda44c564387574c97d7d12468d7f5a101c5ac`
   - [`data/generated/curation_v2/batch1_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch1_50/accepted.jsonl) (`50` rows): `f6bc19c3b0466ac3ce327a51ee88f7e42c49668246a345183ee96c482c507dbd`
   - [`data/generated/curation_v2/batch2_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch2_50/accepted.jsonl) (`50` rows): `5f4cb5afb8aaaf0ce37ab4cedb64639df6fb462d5f0f872e8444c7e85a6d32a6`
   - [`data/generated/curation_v2/batch3_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch3_50/accepted.jsonl) (`50` rows): `abd5c9d931af3a2f9e84ac21d8a006554ae11bc17059729af19a33efe38b60a1`
   - Held-out splits (`smishdetect_validation=708`, `smishdetect_test=1,698`) and OOD benchmark (`banglaphish_ood=5,416`) were never inspected; deduplication ran via isolated cosine-similarity checks only.

2. **Exact & Near-Duplicate (`cos >= 0.85`) Checks (Raw & Slot-Template) for Batch 4 (`N = 50`):**

| Reference Corpus | Reference Rows | Exact Norm Hits | Near-Dup Raw (`cos >= 0.85`) | Max Raw Cos | Near-Dup Slot (`cos >= 0.85`) | Max Slot Cos | Family Overlap |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `curation_v2/existing_unique.parquet` (incl. held-out & OOD) | 11,821 | **0** | **0** | 0.3958 | **0** | 0.4056 | **0** |
| `curation_v2/quarantined.parquet` | 1,094 | **0** | **0** | 0.3143 | **0** | 0.3328 | — |
| `banglaphish2026_dedup.parquet` (OOD) | 5,416 | **0** | **0** | 0.2837 | **0** | 0.2838 | — |
| `curation_v2/accepted.jsonl` (300-row pilot) | 300 | **0** | **0** | 0.3335 | **0** | 0.4005 | **0** |
| `curation_v2/batch1_50/accepted.jsonl` (Batch 1) | 50 | **0** | **0** | 0.3489 | **0** | 0.4627 | **0** |
| `curation_v2/batch2_50/accepted.jsonl` (Batch 2) | 50 | **0** | **0** | 0.3328 | **0** | 0.5045 | **0** |
| `curation_v2/batch3_50/accepted.jsonl` (Batch 3) | 50 | **0** | **0** | 0.3395 | **0** | 0.4163 | **0** |
| Intra-batch (`batch4_50/accepted.jsonl`) | 50 | **0** | **0** | 0.3035 | **0** | 0.3386 | 50 unique `family_id`s |

- Automated screening ([`scripts/validate_stage_b_pilot.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/scripts/validate_stage_b_pilot.py) `--diversity-mode new_batch`) matches [`batch4_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch4_50/accepted.jsonl) `50/50` (`matches_validation_run = True`).
- Unit & regression suite (`.venv/bin/python -m unittest discover -s tests -v` + `.venv/bin/python tests/test_pipeline.py`): **26/26 passed** + **All tests passed!**.

---

## 3. Independent Row-by-Row Review, Rejected & Replacement Samples

- **Independent Reviewer:** `independent_reviewer_swarm` auditing [`raw_batch4_50.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch4_50/raw_batch4_50.jsonl) and [`replacements.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch4_50/replacements.jsonl) across 3 rounds, logged in [`review_batch4_50.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch4_50/review_batch4_50.jsonl).
- **Total candidates screened & reviewed:** **63** (`50` initial + `13` replacements)
  - **Round 1 (`raw_batch4_50.jsonl`, 50 rows):** **40 / 50** accepted (**80.0%**), **10** rejected (`Banglish 8, Bengali 1, CodeMix 1, English 0`; `10` total: `0002, 0010, 0014, 0019, 0023, 0037, 0039, 0044, 0045, 0049`).
  - **Round 2 (`replacements.jsonl` #1–10):** **7 / 10** accepted (**70.0%**), **3** rejected (`syn_b4_r_0004`, `syn_b4_r_0008`, `syn_b4_r_0010`).
  - **Round 3 (`replacements.jsonl` #11–13):** **3 / 3** accepted (**100.0%** — `syn_b4_r_0011`, `syn_b4_r_0012`, `syn_b4_r_0013`).
- **Overall acceptance across 63 reviewed candidates:** **50 / 63** (**79.37%**); **final accepted (`batch4_50/accepted.jsonl`):** **50 / 50** (`review_status = "accepted"`, `0` `pending_independent_review`, `0` automated rejections, `0` metadata/PII/domain errors).

### Rejection Reason Breakdown (`13` Rejected Candidates, `17` Tags)

- `independent_review:unnatural_phrasing`: `11` (`syn_b4_0002, 0010, 0014, 0019, 0023, 0039, 0044, 0045, r_0004, r_0008, r_0010`)
- `independent_review:near_duplicate_slot_template`: `2` (`syn_b4_0019, 0049`)
- `independent_review:cross_batch_scenario_clone`: `2` (`syn_b4_0037, 0049`)
- `independent_review:near_duplicate_paraphrase`: `1` (`syn_b4_0037`)
- `independent_review:shallow_template_clone`: `1` (`syn_b4_0044`)

| Rejected ID | Stratum | Rejection Reasons | Accepted Replacement | Key Fix Applied |
| :--- | :--- | :--- | :--- | :--- |
| `syn_b4_0002` | Bg \| shopping \| med \| `ShankhoTech` | `unnatural_phrasing` | `syn_b4_r_0001` | Fixed conditional `kinle` + progressive `chhar cholche` mismatch to `USB hub-e flat 12% chhar cholche` |
| `syn_b4_0010` | Bg \| employment \| long \| brandless | `unnatural_phrasing` | `syn_b4_r_0002` | Fixed typo `chakrirdatader -> chakridatader` & wooden `apply:` CTA to `bistarito:` |
| `syn_b4_0014` | Bg \| telecom \| long \| brandless | `unnatural_phrasing` | `syn_b4_r_0003` | Replaced stiff `ei masher shob din` / `niyom janun` with idiomatic `shara mash jure` / `details:` |
| `syn_b4_0019` | Bg \| telecom \| med \| brandless | `near_duplicate_slot_template`, `unnatural_phrasing` | `r_0004` -> **`syn_b4_r_0011`** | Fixed missing `meyade`, broke `b3_r_0001` clone; `r_0004` fixed scope clash (`priyo number-e 15 tk talktime bonus`) |
| `syn_b4_0023` | CM \| delivery \| med \| `BoikaliCourier` | `unnatural_phrasing` | `syn_b4_r_0005` | Fixed `চার্জ ছাড়` & `অনলাইন দোকানদারদের` to `অনলাইন উদ্যোক্তাদের ... delivery charge-এ 25% ছাড়` |
| `syn_b4_0037` | Bg \| personal \| med \| brandless | `cross_batch_scenario_clone`, `near_duplicate_paraphrase` | `syn_b4_r_0006` | Replaced eyewear paraphrase of `syn_p2_b_0009` with winter dry-cleaning discount |
| `syn_b4_0039` | Bn \| telecom \| med \| brandless | `unnatural_phrasing` | `syn_b4_r_0007` | Fixed ambiguous `*121*64# ডায়ালে থাকছে` to direct `চালু করতে *121*64# ডায়াল করুন` |
| `syn_b4_0044` | Bg \| telecom \| med \| brandless | `shallow_template_clone`, `unnatural_phrasing` | `r_0008` -> **`syn_b4_r_0012`** | Broke `syn_b4_0020` frame & clipped `nite dial`; `r_0008` fixed `daily` vs `5 din` contradiction |
| `syn_b4_0045` | Bg \| telecom \| med \| brandless | `unnatural_phrasing` | `syn_b4_r_0009` | Fixed `3 din meyad` -> `3 diner jonno` and replaced `Aj rat ...` opener with gaming booster urgency offer |
| `syn_b4_0049` | Bg \| delivery \| med \| `BoikaliCourier` | `cross_batch_scenario_clone`, `near_duplicate_slot_template` | `r_0010` -> **`syn_b4_r_0013`** | Broke `syn_p2_b_0066` corridor clone; `r_0010` fixed `theke...thekei` & `delivery fee-te` locative |

---

## 4. Five-Cohort Distribution, Lexical & Unified Diversity Comparison (`Promotional`)

Per **Requirement 6**, all **5 required cohorts** are reported separately alongside `eligible_train` `Promotional` (`N = 942`) using one **consistent unified brand detector**: `evaluate_codemix_and_entity_diversity(..., known_brands=list(DEFAULT_KNOWN_BRANDS), brand_registry=merged_b1_to_b4)`.

### 4.1 Distribution & Lexical Comparison

| Dimension | `eligible_train` (`N=942`) | (1) Original Pilot (`N=261`) | (2) `B1+B2+B3` (`N=150`) | **(3) Batch 4 (`N=50`)** | **(4) Combined New `B1..B4` (`N=200`)** | **(5) Full Combined (`N=461`)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Language** | En 338, CM 234, Bn 229, Bg 141 | Bg 115, Bn 65, CM 60, En 21 | Bg 57, Bn 41, CM 40, En 12 | **Bg 19, CM 14, Bn 13, En 4** | **Bg 76, Bn 54, CM 54, En 16** | **Bg 191, Bn 119, CM 114, En 37** |
| **`short(<50)`** | 20 (2.12%) | 36 (13.79%) | 3 (2.00%) | **1 (2.00%)** | **4 (2.00%)** | **40 (8.68%)** |
| **`medium(50-120)`** | 779 (82.70%) | 120 (45.98%) | 123 (82.00%) | **41 (82.00%)** | **164 (82.00%)** | **284 (61.61%)** |
| **`long(>120)`** | 143 (15.18%) | 105 (40.23%) | 24 (16.00%) | **8 (16.00%)** | **32 (16.00%)** | **137 (29.72%)** |
| **Median / Mean Chars** | `74.0` / `89.20` | `116.0` / `106.66` | `75.0` / `86.95` | **`95.5` / `102.96`** | **`82.0` / `90.95`** | **`99.0` / `99.85`** |
| **% Has URL** | 29.83% (`281`) | 22.61% (`59`) | 28.67% (`43`) | **28.00%** (`14`) | **28.50%** (`57`) | **25.16%** (`116`) |
| **% Has Phone** | 8.92% (`84`) | 8.81% (`23`) | 8.00% (`12`) | **8.00%** (`4`) | **8.00%** (`16`) | **8.46%** (`39`) |
| **% Has USSD** | 12.95% (`122`) | 5.75% (`15`) | 13.33% (`20`) | **10.00%** (`5`) | **12.50%** (`25`) | **8.68%** (`40`) |
| **% Has Urgency** | 9.98% (`94`)* | 23.37% (`61`) | 8.00% (`12`) | **8.00%** (`4`) | **8.00%** (`16`) | **16.70%** (`77`) |
| **% Brandless (Unified)** | 66.45% (`626`) | 1.92% (`5`) | 52.67% (`79`) | **62.00%** (`31`) | **55.00%** (`110`) | **24.95%** (`115`) |

*\*Note on `has_urgency` in `eligible_train`: [`eligible_train.parquet`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/curation_v2/eligible_train.parquet) has no stored `has_urgency` column; Batch 4's validator computed `9.98% (94/942)` using a broader Bengali+English urgency regex (whereas Batch 3's report regex gave `6.48% = 61/942`). Synthetic rows use each row's audited `has_urgency` metadata field (`8.00%` across `B1`–`B4`).*

### 4.2 Domain Breakdown Across All Five Cohorts (`Promotional`)

| Domain | `eligible_train` (`942`) | (1) Pilot (`261`) | (2) `B1+B2+B3` (`150`) | **(3) Batch 4 (`50`)** | **(4) Comb New (`200`)** | **(5) Full Comb (`461`)** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `telecom` | 281 (29.8%) | 47 | 45 | **15** (30.0%) | **60** (30.0%) | **107** (23.2%) |
| `shopping_ecommerce` | 270 (28.7%) | 58 | 42 | **14** (28.0%) | **56** (28.0%) | **114** (24.7%) |
| `personal_or_general` | 210 (22.3%) | 13 | 33 | **11** (22.0%) | **44** (22.0%) | **57** (12.4%) |
| `banking_mfs` | 89 (9.4%) | 43 | 15 | **5** (10.0%) | **20** (10.0%) | **63** (13.7%) |
| `delivery_parcel` | 66 (7.0%) | 32 | 9 | **3** (6.0%) | **12** (6.0%) | **44** (9.5%) |
| `civic_utility_health` | 15 (1.6%) | 14 | 3 | **1** (2.0%) | **4** (2.0%) | **18** (3.9%) |
| `employment` | 7 (0.7%) | 25 | 2 | **1** (2.0%) | **3** (1.5%) | **28** (6.1%) |
| `education` | 4 (0.4%) | 29 | 1 | **0** (0.0%) | **1** (0.5%) | **30** (6.5%) |

### 4.3 Lexical & Conjunction Comparison (Allowing Natural Words)

| Lexical / CTA Indicator | `eligible_train` | (1) Pilot (`261`) | (2) `B1+B2+B3` (`150`) | **(3) Batch 4 (`50`)** | **(4) Comb New (`200`)** | **(5) Full Comb (`461`)** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Bn+CM `N`** | `463` | `125` | `81` | **`27`** | `108` | `233` |
| Bn+CM `' ও '` | 23 (4.97%) | 27 (21.60%) | 39 (48.15%) | **19 (70.37%)** | 58 (53.70%) | 85 (36.48%) |
| Bn+CM `' এবং '` | 40 (8.64%) | 2 (1.60%) | 6 (7.41%) | **4 (14.81%)** | 10 (9.26%) | 12 (5.15%) |
| Bn+CM `' আর '` | 9 (1.94%) | 4 (3.20%) | 4 (4.94%) | **0 (0.00%)** | 4 (3.70%) | 8 (3.43%) |
| Bn+CM `উপলব্ধ` / `লাভ` | 0 / 0 | 0 / 0 | 0 / 0 | **0 / 0** | 0 / 0 | 0 / 0 |
| Bn+CM `থাকছে` / `পাবেন` / `মিলবে` | 3 / 15 / 0 | 3 / 0 / 0 | 6 / 7 / 4 | **7 / 0 / 0** | 13 / 7 / 4 | 16 / 7 / 4 |
| Bn+CM `পাওয়া` / `ভিজিট করুন` / `করুন <URL>` | 1 / 3 / 3 | 0 / 3 / 6 | 3 / 1 / 4 | **3 / 1 / 3** | 6 / 2 / 7 | 6 / 5 / 13 |
| Bn+CM `দেখুন` / `বিস্তারিত` | 1 / 5 | 2 / 0 | 7 / 5 | **3 / 1** | 10 / 6 | 12 / 6 |
| **Banglish `N_bg`** | `141` | `115` | `57` | **`19`** | `76` | `191` |
| Bg `' ar '` / `' ebong '` / `' o '` | 0 / 4 / 11 | 25 / 1 / 19 | 15 / 4 / 6 | **10 / 0 / 4** | 25 / 4 / 10 | 50 / 5 / 29 |

*Honest observation:* Allowing natural words lifted Batch 4's Round 1 acceptance (`66% -> 80%`), and `উপলব্ধ`/`লাভ` stayed `0/27` naturally, but Bn+CM `' ও '` (`70.37%`), `'থাকছে'` (`25.93%`), Banglish `' ar '` (`52.63%`), and median length (`95.5` vs `74.0` real) re-concentrated relative to Batch 3 (`' ও '` `11.11%`, median `78.0`). Batch 5+ should keep natural usage without bans while sampling shorter medium texts (`~70–80` median chars) and mixing `'আর'`/`'এবং'`/`'ও'` closer to real rates (`~5–10%`).

### 4.4 Automated Diversity Gates & Brand-Detection Definition Discrepancies

| Gate / Metric | Threshold | (1) Pilot (`300`, `audit`) | (2) `B1..B3` (`150`, `new_batch`) | **(3) Batch 4 (`50`, `new_batch`)** | **(4) Comb New (`200`, `new_batch`)** | **(5) Full Comb (`500`, `audit`)** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `codemix_bn_script_en` | `>= 0.80` | 0.5833 (`False`) | 1.0000 (`True`) | **1.0000** (`14/14`, **True**) | **1.0000** (`54/54`, **True**) | 0.7807 (`False`) |
| `codemix_hook_plus_pure_en` | `== 0.00` / `<= 0.15` | 0.8500 (`False`) | 0.0000 (`True`) | **0.0000** (`0/14`, **True**) | **0.0000** (`0/54`, **True**) | 0.4474 (`False`) |
| `codemix_intra_sentential_mix` | `>= 0.60` | 0.1333 (`False`) | 0.9000 (`True`) | **0.9286** (`13/14`*, **True**) | **0.9074** (`49/54`, **True**) | 0.5000 (`False`) |
| `promo_sent1_excl_rate` | `<= 0.35` | 0.5632 (`False`) | 0.0533 (`True`) | **0.0200** (`1/50`, **True**) | **0.0450** (`9/200`, **True**) | **0.3384 (`True`)** |
| `promo_sent1_quest_rate` | `<= 0.05` | 0.0920 (`False`) | 0.0067 (`True`) | **0.0000** (`0/50`, **True**) | **0.0050** (`1/200`, **True**) | 0.0542 (`False`) |
| `max_brand_count` | `<= 5` | 25 (`False`) | 4 (`True`) | **3** (**True**) | **4** (`ShaplaSeba`, **True**) | 25 (`bornotelecom`, `False`) |
| `brandless_rate` (unified) | `>= 0.30` | 0.0192 (`False`) | 0.5267 (`True`) | **0.6200** (`31/50`, **True**) | **0.5500** (`110/200`, **True**) | 0.2495 (`False`) |
| `ussd_1234_share` | `<= 0.25` | 1.0000 (`False`) | 0.0000 (`True`) | **0.0000** (`0/5`, **True**) | **0.0000** (`0/25`, **True**) | 0.3750 (`False`) |
| **Gates Passed** | — | **0 / 8** | **8 / 8** | **8 / 8** | **8 / 8** | **1 / 8** |
| Unique Promotional Brands | — | 42 | 29 | **8** | **37** | **79** |
| Max Opener 3-gram Count | — | 3 (`1.15%`) | 1 (`0.67%`) | **1 (`2.00%`)** | **1 (`0.50%`)** | 3 (`0.65%`) |

*\*Note: Batch 4 CodeMix includes `13` `intra_sentential_mix` + `1` `en_lead_bn_body` (`syn_b4_0029`, `13 + 1 = 14`).*
- **Batch 4 fresh fictional brands (`8` brands, `19` branded rows = `38.0%`, `31` brandless = `62.0%`, `0` collisions):** `juitel` (`3`), `nokshimart` (`3`), `shankhotech` (`3`), `boikalicourier` (`2`), `doyelbazar` (`2`), `ichamotibank` (`2`), `madhobipay` (`2`), `titasstyle` (`2`), registered in [`batch4_50/brand_registry.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch4_50/brand_registry.json).
- **Batch 4 USSD codes (`5` ASCII codes):** `*247*52#`, `*566*42#`, `*321*28#`, `*121*64#`, `*121*48#` (`0` `*1234#`). **Phones (`4`):** `01700-000078`, `01700-000071`, `০১৭০০-০০০০৮৩`, `০১৭০০-০০০০৯২`.
- **Opener 3-gram uniqueness:** `max_opener_3gram_count = 1` across all **`200/200`** rows of `Combined New (B1+B2+B3+B4)`.
- **Brand-detection discrepancy explanation (Req 6):**
  1. **Unified vs. default `evaluate_codemix_and_entity_diversity`:** Calling the diversity function without `known_brands=list(DEFAULT_KNOWN_BRANDS)` omits `REAL_BD_BRANDS` (which would misclassify real `eligible_train` brands) and omits Pilot brand `'নীড়হাট'` (`NeerHaat`) on `syn_p2_c_0003`, inflating Pilot brandless count from `5/261 (1.92%)` to `6/261 (2.30%)` and Full Combined (`500`) from `115/461 (24.95%)` to `116/461 (25.16%)`. Both sets of numbers are stored in `combined_cohorts_diagnostics.json`; all tables above use the unified definition consistently across every cohort.
  2. **Unified vs. `schema.compile_brand_regex`:** `schema.compile_brand_regex` only matches Bengali suffixes `-ের/-ে/-তে/-কে/-র` and misses `-য়` on Batch 1 `syn_b1_0035` (`'শাপলাসেবায়'`), whereas `codemix_diversity.py` Bengali alias matching detects it.

---

## 5. Surface, 4-View 3-Class, Counterfactual & Repeated-Seed Statistical Diagnostics

### 5.1 Surface-Only, 4-View 3-Class CV & Counterfactual Probes (`n = 1,279` Real Smishing Rows)

| Cohort (`eligible_train +`) | Surface Acc / MF1 | Length Smish AUC | `raw` MF1 / R@1%FPR / CF | `canon_masked` MF1 / R@1% / CF | `brand_masked` MF1 / R@1% / CF | `brand_stripped` MF1 / R@1% / CF |
| :--- | :--- | ---: | :--- | :--- | :--- | :--- |
| `eligible_train` (`3,999`) | 73.14% / 62.86% | 0.7693 | 98.01% / 98.51% / — | 97.90% / 97.67% / — | 97.76% / 97.67% / — | 97.81% / 97.87% / — |
| (1) `+ Pilot 300` (`4,299`) | 69.32% / 62.78% | 0.7341 | 97.55% / 97.61% / `-10.40/-12.82pp` | 97.20% / 96.45% / `0.00/-0.08pp` | 97.02% / 95.35% / **`0.00pp`** | 97.26% / 94.83% / **`0.00pp`** |
| (2) `+ B1+B2+B3` (`4,149`) | 71.13% / 62.92% | 0.7650 | 97.90% / 97.48% / `-5.55/-5.63pp` | 97.56% / 97.03% / **`0.00pp`** | 97.41% / 97.29% / **`0.00pp`** | 97.47% / 97.29% / **`0.00pp`** |
| **(3) `+ Batch 4` (`4,049`)** | **72.49% / 63.02%** | **0.7642** | **98.09% / 98.64% / `-1.56/-1.56pp`** | **97.75% / 97.55% / `0.00pp`** | **97.68% / 97.48% / `0.00pp`** | **97.64% / 97.35% / `0.00pp`** |
| **(4) `+ Comb New 200` (`4,199`)** | **70.40% / 62.81%** | **0.7603** | **98.09% / 97.48% / `-5.47/-5.55pp`** | **97.80% / 97.55% / `0.00pp`** | **97.79% / 97.61% / `0.00pp`** | **97.82% / 97.35% / `0.00pp`** |
| **(5) `+ Full Comb 500` (`4,499`)** | **67.13% / 62.49%** | **0.7285** | **97.50% / 97.80% / `-12.12/-14.23pp`** | **97.26% / 95.48% / `0.00pp`** | **97.17% / 95.61% / `0.00pp`** | **97.09% / 95.16% / `0.00pp`** |

### 5.2 Seed-42 Synthetic-Origin (`is_synthetic`) 4-View Probes

| Probe & Control | Feature View | (1) Pilot (`300`) | (2) `B1+B2+B3` (`150`) | **(3) Batch 4 (`50`)** | **(4) Comb New (`200`)** | **(5) Full Comb (`500`)** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `all_vs_real` | `raw` / `canon` | 0.9979 / 0.9976 | 0.9969 / 0.9958 | **0.9959 / 0.9822** | **0.9983 / 0.9975** | **0.9971 / 0.9956** |
| (vs `3,999` real) | `brand_m` / `brand_s` / `surf` | 0.9928 / 0.9789 / 0.7260 | 0.9947 / 0.9945 / 0.6055 | **0.9798 / 0.9813 / 0.5950** | **0.9964 / 0.9968 / 0.6131** | **0.9910 / 0.9849 / 0.6795** |
| `synth_vs_real_smishing` | `raw` / `canon` | 0.9997 / 0.9990 | 0.9999 / 0.9994 | **0.9993 / 0.9976** | **0.9998 / 0.9995** | **0.9996 / 0.9989** |
| (vs `1,548` real Smish) | `brand_m` / `brand_s` / `surf` | 0.9993 / 0.9946 / 0.7957 | 0.9996 / 0.9994 / 0.7775 | **0.9974 / 0.9975 / 0.7374** | **0.9995 / 0.9995 / 0.7724** | **0.9991 / 0.9967 / 0.7716** |
| `synthetic_promo_vs_real_promo` | `raw` / `canon` | 0.9966 / 0.9952 | 0.9931 / 0.9873 | **0.9875 / 0.9524** | **0.9937 / 0.9902** | **0.9950 / 0.9929** |
| (Uncontrolled vs `942` real) | `brand_m` / `brand_s` / `surf` | 0.9871 / 0.9728 / 0.7167 | 0.9842 / 0.9827 / 0.5226 | **0.9445 / 0.9488 / 0.6380** | **0.9878 / 0.9866 / 0.5282** | **0.9872 / 0.9825 / 0.6560** |
| `promo_stratum_3way` | `raw` / `canon` | 0.9904 / 0.9848 | 0.9710 / 0.9478 | **0.9795 / 0.9358** | **0.9786 / 0.9528** | **0.9807 / 0.9644** |
| `(lang, domain, length_band)` | `brand_m` / `brand_s` / `surf` | 0.9735 / 0.9483 / 0.5087 | 0.9421 / 0.9376 / 0.4695 | **0.9280 / 0.9323 / 0.6075** | **0.9467 / 0.9437 / 0.4796** | **0.9577 / 0.9421 / 0.4666** |
| `promo_stratum_2way` | `raw` / `canon` | 0.9803 / 0.9683 | 0.9566 / 0.9196 | **0.9568 / 0.9089** | **0.9683 / 0.9389** | **0.9638 / 0.9295** |
| `(lang, length_band)` | `brand_m` / `brand_s` / `surf` | 0.9603 / 0.9035 / 0.4595 | 0.9124 / 0.9109 / 0.5101 | **0.8996 / 0.9024 / 0.5045** | **0.9308 / 0.9320 / 0.5138** | **0.9133 / 0.8772 / 0.4458** |

### 5.3 Statistical Methodology & Definitions (Requirement 5)

- **What `±` means:** `mean ± SD` reports the **sample standard deviation (`ddof=1`)** of pooled out-of-fold (OOF) ROC-AUC across **10 random CV fold-assignment seeds (`42..51`)** of `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)` grouped by `family_id` (refitting TF-IDF + balanced LogisticRegression inside every fold). This `SD` quantifies **sensitivity to CV fold randomization on the fixed finite dataset** — it is **not** standard error of the mean and **not** generalization uncertainty to unseen external data.
- **How 95% CIs are computed:**
  1. **`seed=42` Bootstrap 95% CI:** Stratified percentile bootstrap (`B = 1,000`, `[2.5, 97.5]` percentiles, resampling rows with replacement within `is_synthetic`) on `seed=42` OOF probabilities (retained for continuity with Batch 3; note it is centered on a single split rather than the 10-seed mean).
  2. **`mean_oof` Bootstrap 95% CI & Cluster Bootstrap 95% CI:** Stratified row-level and `family_id`-cluster percentile bootstraps (`B = 1,000`) on the **10-seed averaged OOF probability per row** (`mean_oof_auc` is slightly higher than single-seed `mean_auc` due to CV-ensemble variance reduction).
  3. **Generalization Caveat:** OOF bootstrap CIs measure finite-sample score uncertainty conditional on the fixed training pool; they do **not** capture model-fitting variance or true out-of-sample generalization error to unseen real-world distributions (which requires held-out `smishdetect_test` and `BanglaPhish-2026` OOD evaluation).
- **Cohort Non-Comparability Warning:** Cohorts differ in synthetic sample size (`N_syn = 50, 150, 200, 261, 461`) and matched-stratum coverage (`71.26%` to `100.0%`). Larger `N_syn` gives linear TF-IDF more positive examples per fold (mechanically raising separability), and different matched-stratum subsets evaluate on different real rows. **Do not claim statistically significant improvement from comparing AUC across unequal-`N_syn` or unequally matched cohorts.**

### 5.4 Repeated-Seed (`10` Seeds: `42..51`), Bootstrap 95% CIs & Matched-Stratum Coverage

#### A. Matched-Stratum Coverage (`Promotional vs Real Promotional`)

| Cohort (`N_syn` Promo) | 3-Way `(lang, dom, len)` Strata (`Matched / Syn`, `Real=60`) | 3-Way Matched Syn | 3-Way Matched Real | 2-Way `(lang, len)` Strata (`Matched / Syn`, `Real=11`) | 2-Way Matched Syn | 2-Way Matched Real | Unmatched 3-Way Synthetic Strata |
| :--- | :---: | ---: | ---: | :---: | ---: | ---: | :--- |
| (1) Original Pilot (`261`) | `50 / 76` | 186/261 (71.26%) | 800/942 (84.93%) | `11 / 12` | 243/261 (93.10%) | 942/942 (100.0%) | `26` strata (`75` rows) |
| (2) `B1+B2+B3` (`150`) | `30 / 33` | 144/150 (96.00%) | 789/942 (83.76%) | `9 / 10` | 148/150 (98.67%) | 931/942 (98.83%) | `3` strata (`Bg\|shop\|long` 2, `Bg\|tel\|short` 2, `CM\|civic\|med` 2) |
| **(3) Batch 4 (`50`)** | **`24 / 27`** | **47/50 (94.00%)** | **749/942 (79.51%)** | **`8 / 9`** | **49/50 (98.00%)** | **922/942 (97.88%)** | **`3` strata (`Bg\|emp\|long` 1, `Bg\|shop\|long` 1, `Bg\|shop\|short` 1)** |
| **(4) Comb New (`200`)** | **`35 / 40`** | **191/200 (95.50%)** | **816/942 (86.62%)** | **`9 / 10`** | **197/200 (98.50%)** | **931/942 (98.83%)** | **`5` strata (`9` rows: `6` from B1+B2 + `3` from B4)** |
| **(5) Full Comb (`461`)** | **`55 / 81`** | **377/461 (81.78%)** | **928/942 (98.51%)** | **`11 / 12`** | **440/461 (95.44%)** | **942/942 (100.0%)** | **`26` strata (`84` rows: `75` Pilot + `6` B1+B2 + `3` B4)** |

#### B. `synthetic_promo_vs_real_promo_stratum_3way` (`10`-Seed Mean±SD, `mean_oof` & Cluster Bootstrap 95% CIs)

| Cohort | `raw` Mean±SD / `mean_oof` Cluster CI | `canon_masked` Mean±SD / `mean_oof` Cluster CI | `brand_masked` Mean±SD / `mean_oof` Cluster CI | `brand_stripped` Mean±SD / `mean_oof` Cluster CI | `surface_8feat` Mean±SD / `mean_oof` Cluster CI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| (1) Pilot (`261`) | `0.9898±0.0016` (`0.9923` `[0.9872, 0.9964]`) | `0.9829±0.0027` (`0.9870` `[0.9811, 0.9921]`) | `0.9713±0.0022` (`0.9751` `[0.9659, 0.9838]`) | `0.9452±0.0069` (`0.9540` `[0.9390, 0.9681]`) | `0.5027±0.0085` (`0.5056` `[0.4638, 0.5468]`) |
| `Batch 1` (`50`) | `0.9316±0.0113` (`0.9428` `[0.8969, 0.9780]`) | `0.8985±0.0166` (`0.9187` `[0.8633, 0.9640]`) | `0.8906±0.0151` (`0.9086` `[0.8557, 0.9550]`) | `0.8558±0.0180` (`0.8733` `[0.8112, 0.9316]`) | `0.4628±0.0177` (`0.4503` `[0.3684, 0.5315]`) |
| `Batch 2` (`50`) | `0.9398±0.0136` (`0.9515` `[0.9225, 0.9764]`) | `0.9008±0.0211` (`0.9185` `[0.8781, 0.9514]`) | `0.8767±0.0251` (`0.8985` `[0.8495, 0.9387]`) | `0.8793±0.0243` (`0.9002` `[0.8532, 0.9399]`) | `0.4222±0.0283` (`0.4138` `[0.3348, 0.4937]`) |
| `Batch 3` (`50`) | `0.9295±0.0121` (`0.9374` `[0.9003, 0.9674]`) | `0.8858±0.0190` (`0.8983` `[0.8484, 0.9411]`) | `0.8677±0.0201` (`0.8797` `[0.8201, 0.9310]`) | `0.8673±0.0196` (`0.8781` `[0.8172, 0.9304]`) | `0.4784±0.0259` (`0.4694` `[0.3863, 0.5514]`) |
| (2) `B1+B2+B3` (`150`) | `0.9738±0.0043` (`0.9788` `[0.9672, 0.9881]`) | `0.9524±0.0078` (`0.9610` `[0.9444, 0.9755]`) | `0.9478±0.0078` (`0.9569` `[0.9391, 0.9721]`) | `0.9417±0.0085` (`0.9508` `[0.9302, 0.9674]`) | `0.4805±0.0113` (`0.4645` `[0.4153, 0.5084]`) |
| **(3) Batch 4 (`50`)** | **`0.9691±0.0077`** (`0.9764` `[0.9603, 0.9899]`) | **`0.9253±0.0127`** (`0.9371` `[0.8968, 0.9682]`) | **`0.9194±0.0128`** (`0.9324` `[0.8895, 0.9656]`) | **`0.9210±0.0138`** (`0.9325` `[0.8874, 0.9678]`) | **`0.5854±0.0189`** (`0.5921` `[0.5064, 0.6860]`) |
| **(4) Comb New (`200`)** | `0.9794±0.0014` (`0.9829` `[0.9754, 0.9895]`) | `0.9563±0.0026` (`0.9624` `[0.9480, 0.9746]`) | `0.9502±0.0035` (`0.9566` `[0.9410, 0.9701]`) | `0.9489±0.0035` (`0.9554` `[0.9389, 0.9697]`) | `0.4870±0.0069` (`0.4816` `[0.4419, 0.5246]`) |
| **(5) Full Comb (`461`)** | `0.9792±0.0015` (`0.9820` `[0.9751, 0.9880]`) | `0.9614±0.0022` (`0.9662` `[0.9565, 0.9754]`) | `0.9564±0.0018` (`0.9614` `[0.9511, 0.9706]`) | `0.9397±0.0029` (`0.9467` `[0.9333, 0.9590]`) | `0.4650±0.0034` (`0.4623` `[0.4308, 0.4961]`) |

*Note: Full `seed=42` row-bootstrap CIs (`Batch 4` 3-way `brand_stripped`: `[0.8965, 0.9641]`; `Combined New 200`: `[0.9253, 0.9591]`; `Full Combined 461`: `[0.9284, 0.9545]`) and 2-way / uncontrolled 10-seed tables (`Batch 4` 2-way `brand_stripped` `0.8970±0.0191`; `Combined New 200` `0.9211±0.0071`; `Full Combined 461` `0.8849±0.0039`) are recorded in [`combined_cohorts_diagnostics.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch4_50/combined_cohorts_diagnostics.json).*

- **Diagnostic explanation of Batch 4 vs. Batch 3 (`brand_stripped` `0.9210` vs `0.8673`):** Removing artificial word bans raised Round 1 naturalness (`66% -> 80%`), but Batch 4's longer median length (`95.5` vs `74.0` real) and lexical clustering (`' ও '` `70.37%`, `থাকছে` `25.93%`, Banglish `' ar '` `52.63%`, `paben`, `wallet`, `25`) drove top 3-way `brand_stripped` features (`word__কছ` `+0.630`, `word__paben` `+0.491`, `word__রথম` `+0.459`, `word__25` `+0.455`, `word__wallet` `+0.421`, `word__ar` `+0.418`, `char__ ও ` `+0.412`, `word__কল phone` `+0.394`) and lifted `surface_8feat` AUC to `0.5854±0.0189` (which dilutes to `0.4870±0.0069` across `Combined New 200`).

---

## 6. Updated Stage B Deficit & Preservation Stop Condition (STOP & REPORT)

Target `T = 1,548` (`387 / language` across all `1,548` `Promotional` rows):

| Class | `eligible_train` | Pilot (`300`) | `B1` (`50`) | `B2` (`50`) | `B3` (`50`) | **Batch 4 (`50`)** | Combined Total | Target (`T`) | **Remaining Deficit** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 1,509 | 39 | 0 | 0 | 0 | 0 | 1,548 | 1,548 | **0** |
| `Promotional` | 942 | 261 | 50 | 50 | 50 | **50** | 1,403 | 1,548 | **145** |
| `Smishing` | 1,548 | 0 | 0 | 0 | 0 | 0 | 1,548 | 1,548 | **0** |
| **Total** | **3,999** | **300** | **50** | **50** | **50** | **50** | **4,499** | — | **145** |

- **Exact remaining `Promotional` deficit by language (`145` total):**
  - `Banglish`: `387 - (141 + 115 + 19 + 19 + 19 + 19) = 55`
  - `Bengali`: `387 - (229 + 65 + 13 + 14 + 14 + 13) = 39`
  - `CodeMix`: `387 - (234 + 60 + 14 + 13 + 13 + 14) = 39`
  - `English`: `387 - (338 + 21 + 4 + 4 + 4 + 4) = 12`
  - **Total remaining:** `55 + 39 + 39 + 12 = 145`
- **No remaining `145` rows have been generated.** All prior Pilot, Batch 1–3, Stage A, and held-out/OOD files are preserved untouched; awaiting explicit user approval before continuing.
