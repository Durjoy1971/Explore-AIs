# Stage B Batch 1 (`50` Promotional Rows) Evaluation & Checkpoint Report (`curation_v2`)

**Date:** 2026-11-18  
**Batch Directory (Isolated):** `data/generated/curation_v2/batch1_50/`  
**Validation Summary JSON:** `data/generated/curation_v2/batch1_50/batch1_50_validation_summary.json`  
**Status:** **Evaluated & Passed All Automated + Independent Review Gates — Stopped before generating remaining `295` `Promotional` rows, awaiting explicit user approval.**

---

## 1. Integrity, Isolation & Deduplication Verification

1. **Protected Corpora & 300-Row Pilot Untouched:**
   - `data/generated/curation_v2/accepted.jsonl` (`300` rows, SHA256 `23af149826dec2411d79b61abceda44c564387574c97d7d12468d7f5a101c5ac`) remains completely unmodified (`Batch 1` was **not** merged into the pilot file).
   - `data/processed/curation_v2/eligible_train.parquet` (`3,999` rows, SHA256 `d9af8bf952558b91b45db5b5f721da2735ec4b18547cdca59d86e21a8b4993e7`), `existing_unique.parquet` (`11,821` rows, SHA256 `acb40c57870b8987b72a920e914d9e81966201caa26867456437a389d88b3e4f`), and `provenance.parquet` (`18,357` rows, SHA256 `9a0fc3f2de54a7db4f040983504f5907b592dec751db9cb1dfcc07d99c21d443`) remain completely unmodified.
   - `git status --short` shows only the new isolated directory `data/generated/curation_v2/batch1_50/` (plus this report and `CHECKPOINT.md` update).
   - Held-out evaluation splits (`smishdetect_validation=708`, `smishdetect_test=1,698`) and OOD benchmark (`banglaphish_ood=5,416`) were never inspected; deduplication ran via isolated cosine-similarity checks only.

2. **Exact & Near-Duplicate (`cos >= 0.85`) Checks (Raw & Slot-Template):**

| Reference Corpus | Reference Rows | Exact Norm Hits | Near-Dup Raw (`cos >= 0.85`) | Max Raw Cos | Near-Dup Slot (`cos >= 0.85`) | Max Slot Cos | Family Overlap |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `curation_v2/existing_unique.parquet` (incl. held-out & OOD) | 11,821 | **0** | **0** | 0.3726 | **0** | 0.5704 | **0** |
| `curation_v2/quarantined.parquet` | 1,094 | **0** | **0** | 0.2868 | **0** | 0.3438 | — |
| `banglaphish2026_dedup.parquet` (OOD) | 5,416 | **0** | **0** | 0.1949 | **0** | 0.2613 | — |
| `curation_v2/accepted.jsonl` (300-row pilot) | 300 | **0** | **0** | 0.5210 | **0** | 0.5965 | **0** |
| Intra-batch (`batch1_50/accepted.jsonl`) | 50 | **0** | **0** | 0.4570 | **0** | 0.4864 | 50 unique `family_id`s |

- Automated screening output (`scripts/validate_stage_b_pilot.py --diversity-mode new_batch`) matches `@reviewer`'s `data/generated/curation_v2/batch1_50/accepted.jsonl` 50/50 by ID and text (`matches_validation_run = True`).

---

## 2. Independent Review, Rejected & Replacement Samples

- **Total candidates reviewed (`raw_batch1_50.jsonl` + `replacements.jsonl`):** **52** (`reviewer_swarm`)
- **Initial batch acceptance:** **48 / 50** (`96.0%`); **overall acceptance across 52 candidates:** **50 / 52** (**96.15%**)
- **Final accepted (`batch1_50/accepted.jsonl`):** **50** (`review_status = "accepted"`, `0` `pending_independent_review`, `0` automated schema/endpoint/language/boundary rejections)

### Rejected & Replaced Rows (`2 / 52`)

1. **`syn_b1_0020`** (`CodeMix | telecom | medium(50-120) | brandless`)
   - **Original text:** `মাসে মাত্র ১৯৮ টাকায় ১০জিবি ও ৩০০ minute। অ্যাক্টিভেট করতে dial *247*8#.`
   - **Rejection reasons:** `independent_review:unnatural_phrasing`, `independent_review:inconsistent_script_punct`
   - **Reviewer rationale:** Mixed Bengali numeral+unit (`১০জিবি`) with Bengali numeral+English unit (`৩০০ minute`), dropped polite verb (`করুন`) in `অ্যাক্টিভেট করতে dial *247*8#.`, and mixed sentence-1 Bengali dari (`।`) with sentence-2 ASCII period (`.`).
   - **Replacement (`syn_b1_r_0001`, accepted):** `মাসে মাত্র ১৯৮ টাকায় 10GB data ও 300 min পাচ্ছেন, প্যাকটি অ্যাক্টিভেট করতে ডায়াল করুন *247*8#।` (`94` chars; consistent units, full verb, consistent `।`, non-`*1234#` USSD).
2. **`syn_b1_0026`** (`CodeMix | shopping_ecommerce | medium(50-120) | brandless`)
   - **Original text:** `Amader online store-e winter jacket-e flat 20% discount paben today.`
   - **Rejection reasons:** `independent_review:unnatural_phrasing`, `independent_review:topic_overlap`
   - **Reviewer rationale:** Appended English `"today."` onto a Banglish clause read as stilted synthetic insertion, and `"winter jacket-e"` repeated the product phrase from `syn_b1_0007`.
   - **Replacement (`syn_b1_r_0002`, accepted):** `সব skincare ও perfume আইটেমে weekend special অফার হিসেবে ১৫% instant ছাড় পাচ্ছেন।` (`81` chars; distinct product category, natural intra-sentential Bengali-script CodeMix).

---

## 3. Distribution Comparison vs. Real `eligible_train.parquet` (`Promotional`)

Batch 1 (`N = 50`) was designed to match the `eligible_train.parquet` `Promotional` (`N = 942`) length, domain, and surface-cue distributions while allocating languages proportionally to the remaining `345` `Promotional` deficit:

| Dimension | Real `eligible_train` Promo (`N=942`) | Pilot `261` Promo (`N=261`) | **Batch 1 `50` Promo (`N=50`)** | Combined Synthetic Promo (`N=311`) |
| :--- | :--- | :--- | :--- | :--- |
| **Language** | En 338 (35.9%), CM 234 (24.8%), Bn 229 (24.3%), Bg 141 (15.0%) | Bg 115, Bn 65, CM 60, En 21 | **Bg 19 (38%), CM 14 (28%), Bn 13 (26%), En 4 (8%)** | Bg 134, Bn 78, CM 74, En 25 |
| **`short(<50)`** | 20 (2.1%) | 36 (13.8%) | **1 (2.0%)** | 37 (11.9%) |
| **`medium(50-120)`** | 779 (82.7%) | 120 (46.0%) | **41 (82.0%)** | 161 (51.8%) |
| **`long(>120)`** | 143 (15.2%) | 105 (40.2%) | **8 (16.0%)** | 113 (36.3%) |
| **Median / Mean Chars** | `74.0` / `89.2` | `116.0` / `106.66` | **`72.5` / `85.92`** | `110.0` / `103.33` |
| **% Has URL** | 29.83% | 22.61% | **28.00%** (`14/50`) | 23.47% |
| **% Has Phone** | 8.92% | 8.81% | **8.00%** (`4/50`) | 8.68% |
| **% Has USSD** | 12.95% | 5.75% | **14.00%** (`7/50`) | 7.07% |
| **% Has Urgency** | 7.43% (`70/942`) | 23.37% (`61/261`) | **8.00%** (`4/50`) | 20.90% (`65/311`) |
| **% Brandless (detected)** | 66.45% | 1.92% | **36.00%** (`18/50`) | 7.40% (`23/311`) |

### Domain Breakdown (`Batch 1 N = 50` vs `eligible_train` Promotional `N = 942`)

| Domain | `eligible_train` Promo (`N=942`) | **Batch 1 (`N=50`)** | Pilot `261` Promo | Combined `311` Promo |
| :--- | ---: | ---: | ---: | ---: |
| `telecom` | 281 (29.8%) | **15 (30.0%)** | 47 | 62 |
| `shopping_ecommerce` | 270 (28.7%) | **14 (28.0%)** | 58 | 72 |
| `personal_or_general` | 210 (22.3%) | **11 (22.0%)** | 13 | 24 |
| `banking_mfs` | 89 (9.4%) | **5 (10.0%)** | 43 | 48 |
| `delivery_parcel` | 66 (7.0%) | **3 (6.0%)** | 32 | 35 |
| `civic_utility_health` | 15 (1.6%) | **1 (2.0%)** | 14 | 15 |
| `education` | 4 (0.4%) | **1 (2.0%)** | 29 | 30 |
| `employment` | 7 (0.7%) | **0 (0.0%)** | 25 | 25 |

---

## 4. CodeMix Linguistic Diversity, Punctuation & Entity Gates (`--diversity-mode new_batch`)

All **8 / 8** automated diversity gates passed on `batch1_50/accepted.jsonl`:

| Gate / Metric | Batch 1 (`N=50`, `new_batch`) | Threshold | Passed | Pilot `261` / Combined `311` (`audit`) |
| :--- | ---: | :--- | :---: | :--- |
| `codemix_bn_script_en` | **1.0000** (`14/14`) | `>= 0.80` | **True** | `0.5833` / `0.6622` (`49/74`) |
| `codemix_hook_plus_pure_en` | **0.0000** (`0/14`) | `== 0.00` | **True** | `0.8500` / `0.6892` (`51/74`) |
| `codemix_intra_sentential_mix` | **1.0000** (`14/14`) | `>= 0.60` | **True** | `0.1333` / `0.2973` (`22/74`) |
| `promo_sent1_excl_rate` | **0.1000** (`5/50`) | `<= 0.35` | **True** | `0.5632` / `0.4887` (`152/311`) |
| `promo_sent1_quest_rate` | **0.0200** (`1/50`) | `<= 0.05` | **True** | `0.0920` / `0.0804` (`25/311`) |
| `max_brand_count` | **4** (`ShaplaSeba`) | `<= 5` | **True** | `25` (`bornotelecom`) |
| `brandless_rate` | **0.3600** (`18/50`) | `>= 0.30` | **True** | `0.0192` / `0.0772` (`24/311`) |
| `ussd_1234_share` | **0.0000** (`0/7`) | `<= 0.25` | **True** | `1.0000` / `0.6818` (`15/22`) |

- **Opener 3-gram uniqueness:** `max_opener_3gram_count = 1` (`2.0%` max share across `50` rows).
- **Fictional brands (`13` fresh brands in `brand_registry.json`, `0` pilot overuse):** `shaplaseba` (`4`), `meghdoot` (`3`), `nilachalstore` (`3`), `lalbaghmart` (`3`), `bhorerbazar` (`3`), `moniharfashion` (`3`), `rupkothacard` (`3`), `aronyokbooks` (`2`), `shitolcafe` (`2`), `kashfulpay` (`2`), `sonargaoexpress` (`2`), `charukolahealth` (`1`), `dipshikhaacademy` (`1`).
- **USSD variety (`7` rows):** `*121*45#`, `*566*12#`, `*321*4#`, `*247*8#`, `*789*3#`, `*654*2#`, `*412*9#` (`0` uses of `*1234#`).
- **Reserved endpoints & fictional phones:** Only `example.com`, `example.org`, `example.net`, `example.edu.bd` (`14` rows) and `01700-0000XX` / `+88017000000XX` (`4` rows); `0` real Bangladeshi brands and `0` extraction cues.

---

## 5. Surface, 4-View 3-Class, Counterfactual & Synthetic-Origin Diagnostics

### A. Surface-Only & Length-Only Diagnostics (`StratifiedGroupKFold`, `family_id`)

| Metric | `eligible_train` (`N=3,999`) | `+ Batch 1` (`N=4,049`) | `+ Combined 350` (`N=4,349`) |
| :--- | ---: | ---: | ---: |
| Surface-only 5-fold Accuracy | 73.14% | 72.56% (-0.58%) | 68.75% (-4.39%) |
| Surface-only 5-fold Macro-F1 | 62.86% | 63.11% (+0.25%) | 62.73% (-0.13%) |
| Length-only Smishing ROC-AUC | 0.7693 | 0.7679 (-0.0014) | 0.7335 (-0.0358) |
| Length-only Promo vs. Normal ROC-AUC | 0.7414 | 0.7415 (+0.0001) | 0.7416 (+0.0002) |

### B. 4-View 3-Class TF-IDF+LR & Counterfactual Probes (`n = 1,279` Real Smishing Rows)

| Text View | Before 3-Class Macro-F1 / R@1%FPR | `+ Batch 1` (`N=4,049`) Macro-F1 / R@1%FPR | `+ Combined 350` (`N=4,349`) Macro-F1 / R@1%FPR | Counterfactual Drop (`+Batch 1` -> `+Combined 350`) |
| :--- | :--- | :--- | :--- | :--- |
| `raw` | 98.01% / 98.51% | 98.15% / 98.64% | 97.59% / 97.16% | `-2.27 pp` -> `-11.49 pp` (contact) / `-14.07 pp` (contact+brand) |
| `canon_masked` | 97.90% / 97.67% | 97.67% / 97.74% | 97.05% / 95.93% | **`0.00 pp`** -> **`0.00 pp`** |
| `brand_masked` | 97.76% / 97.67% | 97.67% / 97.42% | 96.90% / 95.09% | **`0.00 pp`** -> **`0.00 pp`** |
| `brand_stripped` | 97.81% / 97.87% | 97.64% / 97.87% | 96.85% / 94.77% | **`0.00 pp`** -> **`0.00 pp`** |

### C. Synthetic-Origin (`is_synthetic`) 4-View Probes (`Promotional vs Promotional` & Overall)

Comparing **Pilot (`261` Promo)** vs **Batch 1 (`50` Promo)** vs **Combined (`311` Promo)** against real `eligible_train` `Promotional` (`942` rows):

| Probe & Control | Feature View | Pilot `300` (`261` Promo) ROC-AUC | **Batch 1 (`50` Promo) ROC-AUC** | Delta (`Batch 1 - Pilot`) | Combined `350` (`311` Promo) ROC-AUC |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `synthetic_promo_vs_real_promo` | `raw` | 0.9966 | **0.9417** | **-0.0549** | 0.9937 |
| (Uncontrolled) | `canon_masked` | 0.9952 | **0.9338** | **-0.0614** | 0.9908 |
| | `brand_masked` | 0.9871 | **0.9283** | **-0.0588** | 0.9846 |
| | `brand_stripped` | 0.9728 | **0.9101** | **-0.0627** | 0.9767 |
| | `surface_8feat` | 0.7167 | **0.4791** | **-0.2376** | 0.6799 |
| `synthetic_promo_vs_real_promo_stratum_3way` | `raw` | 0.9904 | **0.9242** | **-0.0662** | 0.9814 |
| `(language, domain, length_band)` | `canon_masked` | 0.9848 | **0.8794** | **-0.1054** | 0.9692 |
| Batch 1: `23` strata (`47/50` syn, `702/942` real) | `brand_masked` | 0.9735 | **0.8787** | **-0.0948** | 0.9611 |
| | `brand_stripped` | 0.9483 | **0.8396** | **-0.1087** | 0.9387 |
| | `surface_8feat` | 0.5087 | **0.4591** | **-0.0496** | 0.4626 |
| `synthetic_promo_vs_real_promo_stratum_2way` | `raw` | 0.9803 | **0.9171** | **-0.0632** | 0.9681 |
| `(language, length_band)` | `canon_masked` | 0.9683 | **0.9048** | **-0.0635** | 0.9459 |
| Batch 1: `8` strata (`49/50` syn, `922/942` real) | `brand_masked` | 0.9603 | **0.9207** | **-0.0396** | 0.9382 |
| | `brand_stripped` | 0.9035 | **0.8818** | **-0.0217** | 0.8840 |
| | `surface_8feat` | 0.4595 | **0.4968** | +0.0373 | 0.4413 |

Additional overall & class-matched probes on **Batch 1 (`N=50`)** and **Combined (`N=350`)**:
- `all_vs_real` (`Batch 1 50 vs 3,999`): `raw` `0.9812`, `canon_masked` `0.9769`, `brand_masked` `0.9713`, `brand_stripped` `0.9635`, `surface_8feat` `0.6025`.
- `all_vs_real` (`Combined 350 vs 3,999`): `raw` `0.9973`, `canon_masked` `0.9965`, `brand_masked` `0.9901`, `brand_stripped` `0.9823`, `surface_8feat` `0.7094`.
- `synth_vs_real_smishing` (`Batch 1 50 vs 1,548`): `raw` `0.9988`, `canon_masked` `0.9981`, `brand_masked` `0.9987`, `brand_stripped` `0.9977`, `surface_8feat` `0.7782`.
- `synthetic_normal_vs_real_normal_stratum_3way` (`Combined 350`, `24/39` syn vs `59/1,509` real): `raw` `0.8397`, `canon_masked` `0.7486`, `brand_masked` `0.9004`, `brand_stripped` `0.7309`, `surface_8feat` `0.6871`.

### D. Key Findings & Residual Limitations

1. **Surface-feature shortcut eliminated on Promotional (`surface_8feat` AUC = `0.4791` / `0.4591`):** By matching `eligible_train` `Promotional` median length (`72.5` vs `74.0`), URL (`28.0%` vs `29.8%`), phone (`8.0%` vs `8.9%`), and USSD (`14.0%` vs `13.0%`) rates, Batch 1 Promotional messages are indistinguishable from real Promotional messages on surface metadata alone.
2. **Large drop in lexical separability (`-0.055` to `-0.109` AUC across masked/stripped views):** Eliminating the `[hook !/?] + [pure EN]` template, reducing `sent1!` from `56.3% -> 10.0%`, varying USSD codes, raising brandless share to `36.0%`, and capping fictional brands at `<= 4` reduced `brand_stripped` 3-way stratum-controlled Promo-vs-Promo AUC from `0.9483` (Pilot) to `0.8396` (Batch 1).
3. **Residual synthetic-origin lexical cues (honest reporting for the remaining `295` rows):** Even in `brand_stripped` `synthetic_promo_vs_real_promo_stratum_3way` (`0.8396`), TF-IDF+LR still picks up subtle stylistic preferences (`word__ar`, `word__সব`, `word__at`, `word__রত`, `word__কছ`, `word__chhar`), plus `example.com`/`.org`/`.net` in `raw` (`0.9242`) and `<BRAND>` frequency in `brand_masked` (`0.8787`, since `64%` of Batch 1 vs `33.5%` of real Promo mention a detectable brand). For the remaining `295` rows, increasing brandless share slightly higher (`45–55%`) and varying Banglish conjunctions (`o` / `ebong` / comma-separated clauses alongside `ar`) and Bengali wording can lower `brand_stripped` separability further.

---

## 6. Updated Stage B Deficit & Next Step (STOP & REPORT)

Target `T = 1,548`:

| Class | `eligible_train` | Pilot (`accepted.jsonl`) | **Batch 1 (`batch1_50/accepted.jsonl`)** | Combined Total | Target (`T`) | **Remaining Deficit** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 1,509 | 39 | 0 | 1,548 | 1,548 | **0** |
| `Promotional` | 942 | 261 | **50** | 1,253 | 1,548 | **295** |
| `Smishing` | 1,548 | 0 | 0 | 1,548 | 1,548 | **0** |
| **Total** | **3,999** | **300** | **50** | **4,349** | — | **295** |

- **Remaining `295` `Promotional` deficit by language (for balanced `387/lang`):** `Banglish +112`, `Bengali +80`, `CodeMix +79`, `English +24`.
- **No remaining `295` rows have been generated.** Ready for user review and approval before proceeding with the remaining `295` `Promotional` rows (in reviewable batches) and subsequent Phase 2–5 CV/test/OOD + Colab GPU experiments.
