# Stage B Batch 3 (`50` Promotional Rows) Evaluation & Checkpoint Report (`curation_v2`)

**Date:** 2026-11-19  
**Batch Directory (Isolated):** [`data/generated/curation_v2/batch3_50/`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch3_50)  
**Validation Summary JSON:** [`batch3_50_validation_summary.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch3_50/batch3_50_validation_summary.json)  
**Multi-Cohort Diagnostics JSON:** [`combined_cohorts_diagnostics.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch3_50/combined_cohorts_diagnostics.json)  
**Status:** **Evaluated & Passed All Automated + Independent Review Gates — Stopped before generating remaining `195` `Promotional` samples, awaiting explicit user approval.**

---

## 1. Integrity, Isolation & Cross-Batch Deduplication Verification

1. **Protected Corpora, 300-Row Pilot, Batch 1 & Batch 2 Untouched (SHA256 Verified):**
   - [`data/processed/curation_v2/eligible_train.parquet`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/curation_v2/eligible_train.parquet) (`3,999` rows): `d9af8bf952558b91b45db5b5f721da2735ec4b18547cdca59d86e21a8b4993e7`
   - [`data/processed/curation_v2/existing_unique.parquet`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/curation_v2/existing_unique.parquet) (`11,821` rows): `acb40c57870b8987b72a920e914d9e81966201caa26867456437a389d88b3e4f`
   - [`data/processed/curation_v2/provenance.parquet`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/curation_v2/provenance.parquet) (`18,357` rows): `9a0fc3f2de54a7db4f040983504f5907b592dec751db9cb1dfcc07d99c21d443`
   - [`data/processed/curation_v2/quarantined.parquet`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/curation_v2/quarantined.parquet) (`1,094` rows): `c5cb0855abe2f2998e91d6affb8df7a2ae8984daeacb5373a63d4a1294b17cb4`
   - [`data/generated/curation_v2/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/accepted.jsonl) (`300` rows): `23af149826dec2411d79b61abceda44c564387574c97d7d12468d7f5a101c5ac`
   - [`data/generated/curation_v2/batch1_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch1_50/accepted.jsonl) (`50` rows): `f6bc19c3b0466ac3ce327a51ee88f7e42c49668246a345183ee96c482c507dbd`
   - [`data/generated/curation_v2/batch2_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch2_50/accepted.jsonl) (`50` rows): `5f4cb5afb8aaaf0ce37ab4cedb64639df6fb462d5f0f872e8444c7e85a6d32a6`
   - Held-out splits (`smishdetect_validation=708`, `smishdetect_test=1,698`) and OOD benchmark (`banglaphish_ood=5,416`) were never inspected; deduplication ran via isolated similarity checks only.

2. **Exact & Near-Duplicate (`cos >= 0.85`) Checks (Raw & Slot-Template) for Batch 3 (`N = 50`):**

| Reference Corpus | Reference Rows | Exact Norm Hits | Near-Dup Raw (`cos >= 0.85`) | Max Raw Cos | Near-Dup Slot (`cos >= 0.85`) | Max Slot Cos | Family Overlap |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `curation_v2/existing_unique.parquet` (incl. held-out & OOD) | 11,821 | **0** | **0** | 0.3267 | **0** | 0.4746 | **0** |
| `curation_v2/quarantined.parquet` | 1,094 | **0** | **0** | 0.2520 | **0** | 0.3185 | — |
| `banglaphish2026_dedup.parquet` (OOD) | 5,416 | **0** | **0** | 0.2136 | **0** | 0.2269 | — |
| `curation_v2/accepted.jsonl` (300-row pilot) | 300 | **0** | **0** | 0.4825 | **0** | 0.4990 | **0** |
| `curation_v2/batch1_50/accepted.jsonl` (Batch 1) | 50 | **0** | **0** | 0.3903 | **0** | 0.5669 | **0** |
| `curation_v2/batch2_50/accepted.jsonl` (Batch 2) | 50 | **0** | **0** | 0.3508 | **0** | 0.4255 | **0** |
| Intra-batch (`batch3_50/accepted.jsonl`) | 50 | **0** | **0** | 0.2589 | **0** | 0.3428 | 50 unique `family_id`s |

- Automated screening (`scripts/validate_stage_b_pilot.py --diversity-mode new_batch`) matches [`batch3_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch3_50/accepted.jsonl) 50/50 by `id` and `text` (`matches_validation_run = True`).
- Unit & regression suite (`.venv/bin/python -m unittest discover -s tests -v` + `tests/test_pipeline.py`): **26/26 passed** (`77.439s`) + **All tests passed!**.

---

## 2. Independent Row-by-Row Review, Rejected & Replacement Samples

- **Independent Reviewer:** `independent_reviewer_subagent` (dedicated `stage_b_independent_reviewer` subagent auditing [`raw_batch3_50.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch3_50/raw_batch3_50.jsonl) and [`replacements.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch3_50/replacements.jsonl) across 3 review passes, logged in [`review_batch3_50.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch3_50/review_batch3_50.jsonl)).
- **Total candidates screened & reviewed:** **68** (`50` initial + `18` replacements)
- **Round 1 (`raw_batch3_50.jsonl`, 50 rows):** **33 / 50** accepted (`66.0%`), **17** rejected (`Banglish 5, Bengali 5, CodeMix 5, English 2`).
- **Round 2 (`replacements.jsonl` #1–17):** **16 / 17** accepted (`94.12%`), **1** rejected (`syn_b3_r_0010` for ungrammatical Bengali locative `'শীতকালীন শালে দুটি কিনলে'`).
- **Round 3 (`replacements.jsonl` #18, `syn_b3_r_0018`):** **1 / 1** accepted.
- **Overall acceptance across 68 reviewed candidates:** **50 / 68** (**73.53%**); **final accepted (`batch3_50/accepted.jsonl`):** **50 / 50** (`review_status = "accepted"`, `0` `pending_independent_review`, `0` automated schema/domain/phone/boundary rejections).

### Rejection Reason Breakdown (`18` Rejected Candidates)

- `independent_review:unnatural_phrasing`: `10`
- `independent_review:cross_batch_scenario_clone`: `6`
- `independent_review:shallow_template_clone`: `5`
- `independent_review:near_duplicate_slot_template`: `2`
- `independent_review:unrealistic_ussd_digits`: `2`
- `independent_review:domain_mismatch`: `1`
- `independent_review:metadata_mismatch`: `1`
- `independent_review:near_duplicate_paraphrase`: `1`
- `independent_review:weak_promotional_cue`: `1`

### Summary of All 18 Rejected Candidates & Final Accepted Replacements

| Rejected ID | Stratum | Rejection Reasons | Accepted Replacement | Key Fix Applied |
| :--- | :--- | :--- | :--- | :--- |
| `syn_b3_0002` | Bg \| telecom \| med \| brandless | `unnatural_phrasing` | `syn_b3_r_0001` | Replaced `'activate *566*33#'` with idiomatic `'activate korte *566*33# dial korun.'` |
| `syn_b3_0005` | Bg \| telecom \| med \| brandless | `unnatural_phrasing` | `syn_b3_r_0002` | Fixed `'Bidesh roaming'` / `'shob Asia deshe'` calque to natural Banglish + `'link:'` CTA |
| `syn_b3_0006` | Bg \| telecom \| long \| brandless | `unnatural_phrasing` | `syn_b3_r_0003` | Fixed `'free ... kom mullo-te'` contradiction & dangling clause with full verb `'paben'` |
| `syn_b3_0012` | Bg \| personal \| med \| `KodomStyle` | `shallow_template_clone`, `cross_batch_scenario_clone` | `syn_b3_r_0004` | Replaced POS-12 clone with buy-2-get-gift kurta promo (`01700-000052`) |
| `syn_b3_0018` | Bg \| delivery \| med \| `PalkiCourier` | `weak_promotional_cue`, `shallow_template_clone` | `syn_b3_r_0005` | Added explicit `30% delivery fee discount` & broke POS-18 clone template |
| `syn_b3_0020` | Bn \| telecom \| short \| brandless | `unrealistic_ussd_digits`, `shallow_template_clone` | `syn_b3_r_0006` | Replaced Bengali-digit USSD with ASCII `*789*19#` & validity-first short syntax |
| `syn_b3_0021` | Bn \| telecom \| med \| `KorotoaTel` | `unnatural_phrasing`, `near_duplicate_slot_template` | `syn_b3_r_0007` | Replaced clipped `'পাবেন সাত দিন'` POS-21 clone with weekend app bonus offer |
| `syn_b3_0022` | Bn \| telecom \| med \| brandless | `unrealistic_ussd_digits` | `syn_b3_r_0008` | Replaced Bengali-digit USSD with dialable ASCII `*321*15#` & `'প্রবাসী কল রেট'` |
| `syn_b3_0023` | Bn \| telecom \| med \| brandless | `unnatural_phrasing` | `syn_b3_r_0009` | Fixed aspectual clash (`'অবধি ... চালু হয়েছে'`) using fresh music-streaming last-day promo |
| `syn_b3_0029` | Bn \| personal \| med \| `KodomStyle` | `shallow_template_clone`, `cross_batch_scenario_clone` | `syn_b3_r_0010` → **`syn_b3_r_0018`** | Broke POS-29 clone; `r_0010` rejected for locative `'শালে দুটি কিনলে'`, fixed in `r_0018` (`'কদমস্টাইলে দুটি শীতকালীন শাল একসাথে কিনলেই...'`) |
| `syn_b3_0035` | CM \| telecom \| med \| brandless | `near_duplicate_paraphrase`, `cross_batch_scenario_clone` | `syn_b3_r_0011` | Replaced `syn_b2_0035` paraphrase with data-rollover monthly promo |
| `syn_b3_0036` | CM \| telecom \| long \| brandless | `unnatural_phrasing`, `metadata_mismatch` | `syn_b3_r_0012` | Added `'পুরো 30 দিন মেয়াদে'`, unified offer numerals, aligned `has_urgency=False` |
| `syn_b3_0037` | CM \| shopping \| med \| `MayurTech` | `shallow_template_clone`, `cross_batch_scenario_clone` | `syn_b3_r_0013` | Broke POS-37 clone with flat `300 টাকা কম` TWS earbuds flash deal |
| `syn_b3_0038` | CM \| shopping \| med \| brandless | `unnatural_phrasing` | `syn_b3_r_0014` | Fixed English calque `'চলছে এখন'` & replaced diaper/lotion with rooftop gardening promo |
| `syn_b3_0046` | CM \| employment \| med \| brandless | `domain_mismatch`, `unnatural_phrasing` | `syn_b3_r_0015` | Replaced training-batch `education` text with genuine `employment` remote job alert promo |
| `syn_b3_0047` | En \| telecom \| med \| brandless | `near_duplicate_slot_template`, `cross_batch_scenario_clone` | `syn_b3_r_0016` | Replaced midnight-browsing clone with live-match streaming USSD `*247*14#` |
| `syn_b3_0049` | En \| personal \| med \| brandless | `unnatural_phrasing`, `cross_batch_scenario_clone` | `syn_b3_r_0017` | Replaced `'Order at'` salon clone with `'Book via'` weekend family bowling promo |

---

## 3. Separate-Cohort Distribution & Linguistic Diversity Comparison (`Promotional`)

All five required cohorts are reported separately alongside `eligible_train` `Promotional` (`N = 942`) (plus `Batch 1` and `Batch 2` for single-batch trend visibility):

| Dimension | `eligible_train` (`N=942`) | Original Pilot (`N=261`) | `Batch 1 + Batch 2` (`N=100`) | **Batch 3 (`N=50`)** | **Combined New `B1+B2+B3` (`N=150`)** | **Full Combined Synthetic (`N=411`)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Language** | En 338, CM 234, Bn 229, Bg 141 | Bg 115, Bn 65, CM 60, En 21 | Bg 38, Bn 27, CM 27, En 8 | **Bg 19, Bn 14, CM 13, En 4** | **Bg 57, Bn 41, CM 40, En 12** | **Bg 172, Bn 106, CM 100, En 33** |
| **`short(<50)`** | 20 (2.12%) | 36 (13.79%) | 2 (2.00%) | **1 (2.00%)** | **3 (2.00%)** | **39 (9.49%)** |
| **`medium(50-120)`** | 779 (82.70%) | 120 (45.98%) | 82 (82.00%) | **41 (82.00%)** | **123 (82.00%)** | **243 (59.12%)** |
| **`long(>120)`** | 143 (15.18%) | 105 (40.23%) | 16 (16.00%) | **8 (16.00%)** | **24 (16.00%)** | **129 (31.39%)** |
| **Median / Mean Chars** | `74.0` / `89.20` | `116.0` / `106.66` | `74.0` / `86.16` | **`78.0` / `88.52`** | **`75.0` / `86.95`** | **`101.0` / `99.47`** |
| **% Has URL** | 29.83% (`281`) | 22.61% (`59`) | 28.00% (`28`) | **30.00%** (`15`) | **28.67%** (`43`) | **24.82%** (`102`) |
| **% Has Phone** | 8.92% (`84`) | 8.81% (`23`) | 8.00% (`8`) | **8.00%** (`4`) | **8.00%** (`12`) | **8.52%** (`35`) |
| **% Has USSD** | 12.95% (`122`) | 5.75% (`15`) | 14.00% (`14`) | **12.00%** (`6`) | **13.33%** (`20`) | **8.52%** (`35`) |
| **% Has Urgency** | 6.48% (`61`) | 23.37% (`61`) | 8.00% (`8`) | **8.00%** (`4`) | **8.00%** (`12`) | **17.76%** (`73`) |
| **% Brandless (`DEFAULT_KNOWN`+reg)** | 66.45% (`626`) | 1.92% (`5`) | 48.00% (`48`) | **62.00%** (`31`) | **52.67%** (`79`) | **20.44%** (`84`) |

### Domain Breakdown Across All Five Cohorts (`Promotional`)

| Domain | `eligible_train` (`942`) | Original Pilot (`261`) | `B1+B2` (`100`) | **Batch 3 (`50`)** | **Combined New `B1+B2+B3` (`150`)** | **Full Combined (`411`)** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `telecom` | 281 (29.8%) | 47 | 30 | **15** (30.0%) | **45** (30.0%) | **92** (22.4%) |
| `shopping_ecommerce` | 270 (28.7%) | 58 | 28 | **14** (28.0%) | **42** (28.0%) | **100** (24.3%) |
| `personal_or_general` | 210 (22.3%) | 13 | 22 | **11** (22.0%) | **33** (22.0%) | **46** (11.2%) |
| `banking_mfs` | 89 (9.4%) | 43 | 10 | **5** (10.0%) | **15** (10.0%) | **58** (14.1%) |
| `delivery_parcel` | 66 (7.0%) | 32 | 6 | **3** (6.0%) | **9** (6.0%) | **41** (10.0%) |
| `civic_utility_health` | 15 (1.6%) | 14 | 2 | **1** (2.0%) | **3** (2.0%) | **17** (4.1%) |
| `employment` | 7 (0.7%) | 25 | 1 | **1** (2.0%) | **2** (1.3%) | **27** (6.6%) |
| `education` | 4 (0.4%) | 29 | 1 | **0** (0.0%) | **1** (0.7%) | **30** (7.3%) |

### Linguistic Concentration Comparison (Addressing Batch 2 Lexical Spikes)

| Lexical / CTA Indicator | `eligible_train` Promo | Original Pilot (`261`) | Batch 1 (`50`) | Batch 2 (`50`) | `B1+B2` (`100`) | **Batch 3 (`50`)** | **Combined New (`150`)** | **Full Combined (`411`)** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Bn+CM `N`** | `463` | `125` | `27` | `27` | `54` | **`27`** | `81` | `206` |
| Bn+CM `' ও '` | 23 (4.97%) | 27 (21.60%) | 16 (59.26%) | 20 (74.07%) | 36 (66.67%) | **3 (11.11%)** | 39 (48.15%) | 66 (32.04%) |
| Bn+CM `' এবং '` | 40 (8.64%) | 2 (1.60%) | 1 (3.70%) | 1 (3.70%) | 2 (3.70%) | **4 (14.81%)** | 6 (7.41%) | 8 (3.88%) |
| Bn+CM `' আর '` | 9 (1.94%) | 4 (3.20%) | 0 (0.00%) | 1 (3.70%) | 1 (1.85%) | **3 (11.11%)** | 4 (4.94%) | 8 (3.88%) |
| Bn+CM `উপলব্ধ` / `লাভ` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | **0 / 0** | 0 / 0 | 0 / 0 |
| Bn+CM `মিলবে` / `পাওয়া` / `থাকছে` | 0 / 0 / 3 | 0 / 0 / 3 | 0 / 0 / 2 | 4 / 0* / 4 | 4 / 0 / 6 | **0 / 0 / 0** | 4 / 0 / 6 | 4 / 0 / 9 |
| Bn+CM `ভিজিট করুন` / `করুন <URL>` | 3 / 3 | 3 / 6 | 0 / 1 | 1 / 3 | 1 / 4 | **0 / 0** | 1 / 4 | 4 / 10 |
| **URL rows `N_url`** | `281` | `59` | `14` | `14` | `28` | **`15`** | `43` | `102` |
| URL `দেখুন` / `বিস্তারিত` / `লিংক` | 1 / 3 / 0 | 2 / 0 / 0 | 3 / 2 / 0 | 2 / 1 / 0 | 5 / 3 / 0 | **2 / 2 / 1** | 7 / 5 / 1 | 9 / 5 / 1 |
| **Banglish `N_bg`** | `141` | `115` | `19` | `19` | `38` | **`19`** | `57` | `172` |
| Bg `' ar '` / `' ebong '` / `' o '` | 0 / 4 / 10 | 25 / 1 / 19 | 10 / 0 / 0 | 2 / 3 / 4 | 12 / 3 / 4 | **3 / 1 / 2** | 15 / 4 / 6 | 40 / 5 / 25 |

*\*Note: In Batch 2, 3 rows contained `"পাওয়া যাচ্ছে"` (inflected `"পাওয়া"` + `"যাচ্ছে"` matched the `word__ওয়` subtoken in TF-IDF, alongside `"মিলবে"` matching `word__লব`, `"থাকছে"` matching `word__কছ`, and `"ভিজিট করুন <URL>"` / `"আবেদন করুন <URL>"` / `"চেক করুন <URL>"` matching `word__কর url`); Batch 3 reduced all of those patterns to **0/27** (`' ও '` dropped `74.07% -> 11.11%`, balanced with `' এবং '` `14.81%` and `' আর '` `11.11%`).*

---

## 4. Automated Diversity Gates & Brandless Statistics Across All Five Cohorts

All five cohorts were evaluated with the **merged brand registry** (`batch1_50` + `batch2_50` + `batch3_50` + Pilot `DEFAULT_KNOWN_BRANDS` + row metadata). New-batch cohorts use `--diversity-mode new_batch`; the Original Pilot (`300`, `261` Promo) and **Full Combined Synthetic Dataset (`450`, `411` Promo)** use `--diversity-mode audit` and are reported honestly without conflating `B1+B2` or `B1+B2+B3` with the entire synthetic dataset:

| Gate / Metric | Threshold | Original Pilot (`300`, `audit`) | `B1+B2` (`100`, `new_batch`) | **Batch 3 (`50`, `new_batch`)** | **Combined New `B1+B2+B3` (`150`, `new_batch`)** | **Full Combined (`450`, `audit`)** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `codemix_bn_script_en` | `>= 0.80` | 0.5833 (`False`) | 1.0000 (`True`) | **1.0000** (`13/13`, **True**) | **1.0000** (`40/40`, **True**) | 0.7500 (`False`) |
| `codemix_hook_plus_pure_en` | `== 0.00` / `<= 0.15` | 0.8500 (`False`) | 0.0000 (`True`) | **0.0000** (`0/13`, **True**) | **0.0000** (`0/40`, **True**) | 0.5100 (`False`) |
| `codemix_intra_sentential_mix` | `>= 0.60` | 0.1333 (`False`) | 0.9259 (`True`) | **0.8462** (`11/13`*, **True**) | **0.9000** (`36/40`, **True**) | 0.4400 (`False`) |
| `promo_sent1_excl_rate` | `<= 0.35` | 0.5632 (`False`) | 0.0800 (`True`) | **0.0000** (`0/50`, **True**) | **0.0533** (`8/150`, **True**) | 0.3771 (`False`) |
| `promo_sent1_quest_rate` | `<= 0.05` | 0.0920 (`False`) | 0.0100 (`True`) | **0.0000** (`0/50`, **True**) | **0.0067** (`1/150`, **True**) | 0.0608 (`False`) |
| `max_brand_count` | `<= 5` | 25 (`False`) | 4 (`True`) | **3** (**True**) | **4** (`ShaplaSeba`, **True**) | 25 (`bornotelecom`, `False`) |
| `brandless_rate` (diversity fn) | `>= 0.30` | 0.0230 (`False`) | 0.4800 (`True`) | **0.6200** (`31/50`, **True**) | **0.5267** (`79/150`, **True**) | 0.2068 (`False`)† |
| `ussd_1234_share` | `<= 0.25` | 1.0000 (`False`) | 0.0000 (`True`) | **0.0000** (`0/6`, **True**) | **0.0000** (`0/20`, **True**) | 0.4286 (`False`) |
| **All 8 Gates Passed** | — | **False (`0/8`)** | **True (`8/8`)** | **True (`8/8`)** | **True (`8/8`)** | **False (`0/8`)** |
| Unique Promotional Brands | — | 39 | 21 | **8** | **29** | **68** |
| Max Opener 3-gram Count | — | 3 (`1.15%`) | 1 (`1.00%`) | **1 (`2.00%`)** | **1 (`0.67%`)** | 3 (`0.73%`) |

*\*Note: Batch 3 CodeMix intentionally included 11 `intra_sentential_mix` + 2 `en_lead_bn_body` rows (`11 + 2 = 13`). †Note: On `Full Combined (450)`, `evaluate_codemix_and_entity_diversity` detects `85/411 = 20.68%` brandless (`20.44% = 84/411` when adding `DEFAULT_KNOWN_BRANDS` which matches Pilot `syn_p2_c_0003` `'নীড়হাটে'`); `Full Combined` fails all 8 `audit` gates solely because the legacy 300-row Pilot accounts for `261/411 = 63.5%` of synthetic Promotional rows.*

- **Batch 3 fresh fictional brands (`8` brands, `19` branded rows = `38.0%`, `31` naturally brandless = `62.0%`, max count `3`):** `korotoatel` (`3`), `shimulmart` (`3`), `shiulipay` (`3`), `bakulbazar` (`2`), `rupshabank` (`2`), `mayurtech` (`2`), `palkicourier` (`2`), `kodomstyle` (`2`) — registered in [`batch3_50/brand_registry.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch3_50/brand_registry.json). `Combined New (B1+B2+B3, 150)` spans **`29` unique fictional brands** (`max_brand_count = 4`, `0` cross-batch brand overuse).
- **Batch 3 USSD codes (`6` rows):** `*121*53#`, `*566*33#`, `*789*19#`, `*321*15#`, `*654*11#`, `*247*14#` (`0` uses of `*1234#`).
- **Batch 3 fictional phones (`4` rows):** `01700-000052`, `01700000047`, `০১৭০০-০০০০৬৪`, `+8801700000083`.
- **Opener 3-gram uniqueness:** `max_opener_3gram_count = 1` across `Batch 3` (`50/50` unique openers) and across ALL `150` rows of `Combined New (B1+B2+B3)` (`150/150` unique openers!).

---

## 5. Surface, 4-View 3-Class, Counterfactual & Synthetic-Origin Diagnostics

### A. Surface-Only, 4-View 3-Class CV & Counterfactual Probes (`n = 1,279` Real Smishing Rows)

| Cohort (`eligible_train +`) | Surface Acc / MF1 | Length Smish AUC | `raw` MF1 / R@1%FPR / CF | `canon_masked` MF1 / R@1% / CF | `brand_masked` MF1 / R@1% / CF | `brand_stripped` MF1 / R@1% / CF |
| :--- | :--- | ---: | :--- | :--- | :--- | :--- |
| `eligible_train` (`3,999`) | 73.14% / 62.86% | 0.7693 | 98.01% / 98.51% / — | 97.90% / 97.67% / — | 97.76% / 97.67% / — | 97.81% / 97.87% / — |
| `+ Pilot 300` (`4,299`) | 69.32% / 62.78% | 0.7341 | 97.55% / 97.61% / `-10.40/-12.82pp` | 97.20% / 96.45% / `0.00/-0.08pp` | 97.02% / 95.35% / **`0.00pp`** | 97.26% / 94.83% / **`0.00pp`** |
| `+ B1+B2` (`4,099`) | 71.85% / 62.93% | 0.7666 | 97.88% / 97.87% / `-3.21/-3.21pp` | 97.41% / 96.83% / **`0.00pp`** | 97.57% / 96.51% / **`0.00pp`** | 97.43% / 96.77% / **`0.00pp`** |
| **`+ Batch 3` (`4,049`)** | **72.51% / 63.04%** | **0.7675** | **98.09% / 98.77% / `-3.28/-3.28pp`** | **97.59% / 97.35% / `0.00pp`** | **97.58% / 97.42% / `0.00pp`** | **97.48% / 97.80% / `0.00pp`** |
| **`+ Combined New 150` (`4,149`)** | **71.13% / 62.92%** | **0.7650** | **97.90% / 97.48% / `-5.55/-5.63pp`** | **97.56% / 97.03% / `0.00pp`** | **97.41% / 97.29% / `0.00pp`** | **97.47% / 97.29% / `0.00pp`** |
| **`+ Full Combined 450` (`4,449`)** | **67.75% / 62.66%** | **0.7321** | **97.34% / 97.35% / `-11.88/-14.15pp`** | **97.27% / 96.25% / `0.00pp`** | **97.32% / 95.93% / `0.00pp`** | **97.16% / 95.74% / `0.00pp`** |

### B. Synthetic-Origin (`is_synthetic`) 4-View Probes (`seed = 42`)

| Probe & Control | Feature View | Original Pilot (`300`, `261` Promo) | `B1+B2` (`100` Promo) | **Batch 3 (`50` Promo)** | **Combined New (`150` Promo)** | **Full Combined (`450`, `411` Promo)** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `all_vs_real` | `raw` | 0.9979 | 0.9949 | **0.9830** | **0.9969** | **0.9973** |
| (vs `3,999` real) | `canon_masked` | 0.9976 | 0.9921 | **0.9767** | **0.9958** | **0.9965** |
| | `brand_masked` | 0.9928 | 0.9894 | **0.9757** | **0.9947** | **0.9915** |
| | `brand_stripped` | 0.9789 | 0.9909 | **0.9709** | **0.9945** | **0.9854** |
| | `surface_8feat` | 0.7260 | 0.6048 | **0.5941** | **0.6055** | **0.6847** |
| `synth_vs_real_smishing` | `raw` | 0.9997 | 0.9999 | **0.9979** | **0.9999** | **0.9998** |
| (vs `1,548` real Smish) | `canon_masked` | 0.9990 | 0.9996 | **0.9970** | **0.9994** | **0.9991** |
| | `brand_masked` | 0.9993 | 0.9996 | **0.9974** | **0.9996** | **0.9993** |
| | `brand_stripped` | 0.9946 | 0.9995 | **0.9964** | **0.9994** | **0.9968** |
| | `surface_8feat` | 0.7957 | 0.7665 | **0.7633** | **0.7775** | **0.7799** |
| `synthetic_promo_vs_real_promo` | `raw` | 0.9966 | 0.9882 | **0.9566** | **0.9931** | **0.9937** |
| (Uncontrolled vs `942` real Promo) | `canon_masked` | 0.9952 | 0.9820 | **0.9358** | **0.9873** | **0.9914** |
| | `brand_masked` | 0.9871 | 0.9759 | **0.9284** | **0.9842** | **0.9846** |
| | `brand_stripped` | 0.9728 | 0.9752 | **0.9225** | **0.9827** | **0.9777** |
| | `surface_8feat` | 0.7167 | 0.4999 | **0.4958** | **0.5226** | **0.6428** |
| `synthetic_promo_vs_real_promo_stratum_3way` | `raw` | 0.9904 | 0.9637 | **0.9221** | **0.9710** | **0.9795** |
| `(language, domain, length_band)` | `canon_masked` | 0.9848 | 0.9389 | **0.8821** | **0.9478** | **0.9612** |
| | `brand_masked` | 0.9735 | 0.9307 | **0.8598** | **0.9421** | **0.9559** |
| | `brand_stripped` | 0.9483 | 0.9258 | **0.8600** | **0.9376** | **0.9360** |
| | `surface_8feat` | 0.5087 | 0.4792 | **0.4235** | **0.4695** | **0.4749** |
| `synthetic_promo_vs_real_promo_stratum_2way` | `raw` | 0.9803 | 0.9529 | **0.9392** | **0.9566** | **0.9671** |
| `(language, length_band)` | `canon_masked` | 0.9683 | 0.9164 | **0.8790** | **0.9196** | **0.9381** |
| | `brand_masked` | 0.9603 | 0.9171 | **0.8642** | **0.9124** | **0.9256** |
| | `brand_stripped` | 0.9035 | 0.9063 | **0.8690** | **0.9109** | **0.8856** |
| | `surface_8feat` | 0.4595 | 0.5177 | **0.4810** | **0.5101** | **0.4539** |

*Note on `Normal` class (`39` Pilot rows; `0` added in `B1`–`B3`): `synthetic_normal_vs_real_normal` (`39` vs `1,509`) and `synthetic_normal_vs_real_normal_stratum_3way` (`17` strata, `24/39` syn vs `59/1,509` real) are unchanged across Pilot `300` and Full Combined `450`: uncontrolled `raw 0.9847 / canon 0.9827 / brand_m 0.9982 / brand_s 0.9722 / surf 0.9390`; 3-way `raw 0.8397 / canon 0.7486 / brand_m 0.9004 / brand_s 0.7309 / surf 0.6871`.*

### C. Repeated-Seed (`10` Seeds: `42..51`) & Stratified Bootstrap 95% CI (`B = 1,000`) ROC-AUC Comparisons

All values report `10-seed mean ± std` (`[min, max]`) across `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)` (re-fitting TF-IDF inside every fold) plus `95% stratified bootstrap CI` on `seed=42` OOF predictions:

#### 1. `synthetic_promo_vs_real_promo_stratum_3way` (`(language, domain, length_band)`)

| Cohort | `raw` Mean±SD / 95% CI | `canon_masked` Mean±SD / 95% CI | `brand_masked` Mean±SD / 95% CI | `brand_stripped` Mean±SD / 95% CI | `surface_8feat` Mean±SD / 95% CI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Original Pilot (`261`) | `0.9898±0.0016` `[0.9842, 0.9949]` | `0.9829±0.0027` `[0.9778, 0.9905]` | `0.9713±0.0022` `[0.9629, 0.9826]` | `0.9452±0.0069` `[0.9312, 0.9636]` | `0.5027±0.0085` `[0.4668, 0.5519]` |
| Batch 1 (`50`) | `0.9316±0.0113` `[0.8785, 0.9670]` | `0.8985±0.0166` `[0.8156, 0.9374]` | `0.8906±0.0151` `[0.8246, 0.9315]` | `0.8558±0.0180` `[0.7699, 0.9041]` | `0.4628±0.0177` `[0.3798, 0.5409]` |
| Batch 2 (`50`) | `0.9397±0.0136` `[0.9243, 0.9746]` | `0.9008±0.0211` `[0.8829, 0.9535]` | `0.8767±0.0251` `[0.8578, 0.9386]` | `0.8794±0.0243` `[0.8651, 0.9410]` | `0.4222±0.0283` `[0.3460, 0.4921]` |
| `B1+B2` (`100`) | `0.9650±0.0052` `[0.9424, 0.9814]` | `0.9413±0.0065` `[0.9114, 0.9639]` | `0.9344±0.0065` `[0.9010, 0.9574]` | `0.9259±0.0071` `[0.8959, 0.9545]` | `0.4643±0.0108` `[0.4231, 0.5409]` |
| **Batch 3 (`50`)** | **`0.9295±0.0121`** `[0.8826, 0.9574]` | **`0.8858±0.0190`** `[0.8315, 0.9244]` | **`0.8677±0.0201`** `[0.8024, 0.9097]` | **`0.8673±0.0196`** `[0.8014, 0.9111]` | **`0.4784±0.0259`** `[0.3443, 0.5073]` |
| **Comb New `B1+B2+B3` (`150`)** | `0.9738±0.0043` `[0.9566, 0.9836]` | `0.9523±0.0078` `[0.9262, 0.9672]` | `0.9478±0.0077` `[0.9202, 0.9629]` | `0.9417±0.0085` `[0.9133, 0.9595]` | `0.4805±0.0113` `[0.4187, 0.5136]` |
| **Full Combined (`411`)** | `0.9767±0.0019` `[0.9719, 0.9863]` | `0.9590±0.0026` `[0.9496, 0.9714]` | `0.9543±0.0022` `[0.9452, 0.9662]` | `0.9324±0.0034` `[0.9202, 0.9501]` | `0.4678±0.0045` `[0.4427, 0.5086]` |

#### 2. `synthetic_promo_vs_real_promo_stratum_2way` (`(language, length_band)`)

| Cohort | `raw` Mean±SD / 95% CI | `canon_masked` Mean±SD / 95% CI | `brand_masked` Mean±SD / 95% CI | `brand_stripped` Mean±SD / 95% CI | `surface_8feat` Mean±SD / 95% CI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Original Pilot (`261`) | `0.9815±0.0013` `[0.9725, 0.9869]` | `0.9698±0.0017` `[0.9574, 0.9773]` | `0.9604±0.0010` `[0.9500, 0.9699]` | `0.9059±0.0038` `[0.8816, 0.9224]` | `0.4654±0.0114` `[0.4231, 0.4955]` |
| Batch 1 (`50`) | `0.9129±0.0063` `[0.8715, 0.9573]` | `0.8827±0.0139` `[0.8556, 0.9463]` | `0.9006±0.0117` `[0.8741, 0.9572]` | `0.8584±0.0138` `[0.8320, 0.9284]` | `0.4928±0.0204` `[0.4316, 0.5604]` |
| Batch 2 (`50`) | `0.9165±0.0154` `[0.8970, 0.9573]` | `0.8689±0.0162` `[0.8379, 0.9203]` | `0.8505±0.0192` `[0.8296, 0.9113]` | `0.8489±0.0188` `[0.8211, 0.9078]` | `0.4528±0.0146` `[0.3857, 0.5286]` |
| `B1+B2` (`100`) | `0.9492±0.0062` `[0.9305, 0.9713]` | `0.9154±0.0074` `[0.8889, 0.9446]` | `0.9168±0.0069` `[0.8880, 0.9456]` | `0.9046±0.0080` `[0.8751, 0.9373]` | `0.5032±0.0134` `[0.4690, 0.5702]` |
| **Batch 3 (`50`)** | **`0.9233±0.0124`** `[0.9064, 0.9667]` | **`0.8748±0.0152`** `[0.8275, 0.9225]` | **`0.8538±0.0172`** `[0.8074, 0.9119]` | **`0.8596±0.0168`** `[0.8138, 0.9161]` | **`0.4614±0.0150`** `[0.4004, 0.5607]` |
| **Comb New `B1+B2+B3` (`150`)** | `0.9553±0.0085` `[0.9396, 0.9710]` | `0.9180±0.0073` `[0.8961, 0.9416]` | `0.9101±0.0078` `[0.8869, 0.9369]` | `0.9090±0.0075` `[0.8859, 0.9361]` | `0.5027±0.0146` `[0.4662, 0.5535]` |
| **Full Combined (`411`)** | `0.9620±0.0033` `[0.9582, 0.9768]` | `0.9325±0.0049` `[0.9250, 0.9521]` | `0.9212±0.0037` `[0.9109, 0.9392]` | `0.8805±0.0058` `[0.8675, 0.9042]` | `0.4512±0.0049` `[0.4190, 0.4862]` |

#### 3. `synthetic_promo_vs_real_promo` (Uncontrolled vs. `942` Real Promotional)

| Cohort | `raw` Mean±SD / 95% CI | `canon_masked` Mean±SD / 95% CI | `brand_masked` Mean±SD / 95% CI | `brand_stripped` Mean±SD / 95% CI | `surface_8feat` Mean±SD / 95% CI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Original Pilot (`261`) | `0.9969±0.0005` `[0.9939, 0.9987]` | `0.9957±0.0006` `[0.9923, 0.9977]` | `0.9885±0.0008` `[0.9815, 0.9918]` | `0.9747±0.0012` `[0.9626, 0.9812]` | `0.7122±0.0058` `[0.6836, 0.7505]` |
| Batch 1 (`50`) | `0.9547±0.0071` `[0.8885, 0.9808]` | `0.9384±0.0084` `[0.8796, 0.9731]` | `0.9314±0.0087` `[0.8764, 0.9669]` | `0.9131±0.0091` `[0.8484, 0.9577]` | `0.4775±0.0254` `[0.3986, 0.5649]` |
| Batch 2 (`50`) | `0.9732±0.0072` `[0.9705, 0.9931]` | `0.9517±0.0134` `[0.9495, 0.9828]` | `0.9394±0.0127` `[0.9208, 0.9761]` | `0.9395±0.0124` `[0.9181, 0.9761]` | `0.4394±0.0162` `[0.3530, 0.5298]` |
| `B1+B2` (`100`) | `0.9859±0.0029` `[0.9788, 0.9952]` | `0.9777±0.0040` `[0.9694, 0.9913]` | `0.9715±0.0044` `[0.9589, 0.9881]` | `0.9703±0.0056` `[0.9593, 0.9872]` | `0.5033±0.0149` `[0.4353, 0.5686]` |
| **Batch 3 (`50`)** | **`0.9588±0.0055`** `[0.9241, 0.9820]` | **`0.9348±0.0096`** `[0.8992, 0.9668]` | **`0.9290±0.0107`** `[0.8872, 0.9633]` | **`0.9231±0.0098`** `[0.8779, 0.9603]` | **`0.4718±0.0221`** `[0.4028, 0.5838]` |
| **Comb New `B1+B2+B3` (`150`)** | `0.9915±0.0016` `[0.9886, 0.9969]` | `0.9879±0.0019` `[0.9804, 0.9931]` | `0.9853±0.0024` `[0.9767, 0.9906]` | `0.9828±0.0020` `[0.9748, 0.9901]` | `0.5131±0.0133` `[0.4746, 0.5743]` |
| **Full Combined (`411`)** | `0.9942±0.0003` `[0.9898, 0.9965]` | `0.9915±0.0005` `[0.9872, 0.9947]` | `0.9847±0.0007` `[0.9794, 0.9891]` | `0.9796±0.0012` `[0.9700, 0.9847]` | `0.6490±0.0034` `[0.6097, 0.6746]` |

### D. Matched-Stratum Coverage & Diagnostic Interpretation

| Cohort (`N_syn` Promo) | 3-Way `(lang, dom, len)` Strata (`Matched / Syn Total`, `Real Total=60`) | 3-Way Matched Syn (`%`) | 3-Way Matched Real (`%`) | 2-Way `(lang, len)` Strata (`Matched / Syn Total`, `Real Total=11`) | 2-Way Matched Syn (`%`) | 2-Way Matched Real (`%`) | Unmatched Synthetic Strata |
| :--- | :---: | ---: | ---: | :---: | ---: | ---: | :--- |
| Original Pilot (`261`) | `50 / 76` | 186/261 (**71.26%**) | 800/942 (84.93%) | `11 / 12` | 243/261 (**93.10%**) | 942/942 (100.0%) | **26** 3-way strata (`75` syn rows, incl. `18` `Banglish\|short(<50)`) |
| Batch 1 (`50`) | `23 / 26` | 47/50 (94.00%) | 702/942 (74.52%) | `8 / 9` | 49/50 (98.00%) | 922/942 (97.88%) | **3** (`Bg\|shopping\|long`, `Bg\|telecom\|short`, `CM\|civic\|med`) |
| Batch 2 (`50`) | `24 / 27` | 47/50 (94.00%) | 759/942 (80.57%) | `7 / 8` | 49/50 (98.00%) | 882/942 (93.63%) | **3** (`Bg\|shopping\|long`, `Bg\|telecom\|short`, `CM\|civic\|med`) |
| `B1+B2` (`100`) | `26 / 29` | 94/100 (94.00%) | 769/942 (81.63%) | `8 / 9` | 98/100 (98.00%) | 922/942 (97.88%) | **3** (`6` syn rows across the 3 cells above) |
| **Batch 3 (`50`)** | **`27 / 27`** | **50/50 (100.00%)** | **778/942 (82.59%)** | **`8 / 8`** | **50/50 (100.00%)** | **891/942 (94.59%)** | **None (`0` unmatched strata)** |
| **Comb New (`150`)** | `30 / 33` | 144/150 (96.00%) | 789/942 (83.76%) | `9 / 10` | 148/150 (98.67%) | 931/942 (98.83%) | **3** (`6` syn rows from B1+B2 only) |
| **Full Combined (`411`)** | `54 / 80` | 330/411 (80.29%) | 927/942 (98.41%) | `11 / 12` | 391/411 (95.13%) | 942/942 (100.0%) | **26** (`81` syn rows: `75` Pilot + `6` B1+B2) |

1. **100% matched-stratum coverage in Batch 3 (`50/50`):** Every Batch 3 message falls inside real `eligible_train` `(language, domain, length_band)` and `(language, length_band)` support (`0` rows dropped during controlled probing), yet Batch 3 still achieves the **lowest** 10-seed mean 3-way AUC on `canon_masked` (`0.8858 ± 0.0190`) and `brand_masked` (`0.8677 ± 0.0201`), and `0.8673 ± 0.0196` on `brand_stripped` (vs. Pilot `0.9452 ± 0.0069` on `71.26%` coverage).
2. **Elimination of Batch 2's lexical concentrations:** Batch 2's top 3-way `brand_stripped` features (`word__ওয়`, `word__লব`, `word__কর url`, `char__ ও `) disappeared completely from Batch 3's top features (`word__২৫` +0.636, `word__ইন` +0.597, `word__prothom` +0.459, `word__parcel` +0.424, `word__mashe` +0.406). Note that `word__২৫` reflects Bengali digits (`২৫%`) in Batch 3 vs ASCII digits (`25%`) often used in real Bengali SMS, and `word__ইন` (`অনলাইন`/`ডাইন-ইনে`/`ইন্টারনেট`/`হেল্পলাইন`) reflects common Bangladeshi retail terms.
3. **Why pooled cohorts (`B1+B2` `100` -> `Combined New` `150` -> `Full Combined` `411`) show higher AUC than `N=50` batches:** Increasing synthetic sample count (`50 -> 100 -> 150 -> 411`) against the fixed `~778–942` real Promotional pool gives the linear TF-IDF classifier 2x–8x more positive examples per CV fold, sharpening variance and making subtle residual domain/digit distributions more separable (e.g., `Combined New 150` 3-way `brand_stripped` `0.9417 ± 0.0085` vs `Batch 3` `0.8673 ± 0.0196`). On `Full Combined (450)`, `surface_8feat` uncontrolled Promo AUC (`0.6490 ± 0.0034`) and counterfactual raw Smishing drop (`-11.88 / -14.15 pp`) remain dominated by the Pilot (`261`), confirming the importance of evaluating `Full Combined` separately from the new batches and keeping canonical/brand masking active in downstream baselines.

---

## 6. Updated Stage B Deficit & Next Step (STOP & REPORT)

Target `T = 1,548`:

| Class | `eligible_train` | Original Pilot (`300`) | `Batch 1` (`50`) | `Batch 2` (`50`) | **Batch 3 (`50`)** | Combined Total | Target (`T`) | **Remaining Deficit** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 1,509 | 39 | 0 | 0 | 0 | 1,548 | 1,548 | **0** |
| `Promotional` | 942 | 261 | 50 | 50 | **50** | 1,353 | 1,548 | **195** |
| `Smishing` | 1,548 | 0 | 0 | 0 | 0 | 1,548 | 1,548 | **0** |
| **Total** | **3,999** | **300** | **50** | **50** | **50** | **4,449** | — | **195** |

- **Exact remaining `Promotional` deficit by language (for equal `387/lang` across `1,548` Promotional rows):**
  - `Banglish`: `387 - (141 + 115 + 19 + 19 + 19) = 74`
  - `Bengali`: `387 - (229 + 65 + 13 + 14 + 14) = 52`
  - `CodeMix`: `387 - (234 + 60 + 14 + 13 + 13) = 53`
  - `English`: `387 - (338 + 21 + 4 + 4 + 4) = 16`
  - **Total remaining:** `74 + 52 + 53 + 16 = 195`
- **No remaining `195` rows have been generated.** Stopped after Batch 3; awaiting explicit user approval before continuing.
