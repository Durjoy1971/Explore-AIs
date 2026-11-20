# Stage B Batch 3 Pre-Generation Rejection Audit (`batch3_50` -> Batch 4)

**Date:** 2026-11-19  
**Scope:** Full audit of all **18 rejected candidates** (`17` Round 1 + `1` Round 2 out of `68` reviewed) in Stage B Batch 3 (`data/generated/curation_v2/batch3_50/`) prior to Batch 4 generation.  
**Artifacts Audited (Read-Only):**
- [`data/DATASET_GUIDE.md`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/DATASET_GUIDE.md)
- [`CHECKPOINT.md`](file:///home/bs01415/Documents/Programming/ExploringAIs/CHECKPOINT.md)
- [`reports/stage_b_batch3_50_report.md`](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/stage_b_batch3_50_report.md)
- [`raw_batch3_50.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch3_50/raw_batch3_50.jsonl), [`replacements.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch3_50/replacements.jsonl), [`review_batch3_50.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch3_50/review_batch3_50.jsonl), [`batch3_50_validation_summary.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch3_50/batch3_50_validation_summary.json)
- Reference distributions from [`eligible_train.parquet`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/processed/curation_v2/eligible_train.parquet) (`N = 942` real `Promotional` rows; held-out/OOD splits were never inspected) and prior synthetic cohorts ([`accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/accepted.jsonl), [`batch1_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch1_50/accepted.jsonl), [`batch2_50/accepted.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch2_50/accepted.jsonl)).

---

## Executive Summary

Batch 3's acceptance rate dropped sharply to **73.53% (`50/68`)** — and **66.0% (`33/50`)** in Round 1 — compared with **96.15% (`50/52`, `96.0%` Round 1)** in Batch 1 and **94.34% (`50/53`, `94.0%` Round 1)** in Batch 2. Auditing all 18 rejections reveals two dominant systemic root causes:

1. **Positional & Structural Template Anchoring Across Batches (`9` candidates):** `raw_batch3_50.jsonl` mirrored `batch2_50`'s exact row-by-row `(language, domain)` sequence in **48/50** positions and `length_band` in **46/50** positions. Generating row-for-row against the same positional skeleton caused repeated cross-batch clones at identical positions (`POS 12, 18, 20, 21, 29, 35, 37, 47, 49`). Automated `cos >= 0.85` screening missed all of them (`max raw_cos = 0.543`, `max slot_cos = 0.670`, `max brand_masked slot_cos = 0.700`).
2. **Over-Aggressive Lexical Avoidance & Hurried Rewrites (`10` `unnatural_phrasing` catches):** Trying to suppress Batch 2's TF-IDF-salient tokens (`ও`, `থাকছে`, `মিলবে`, `পাওয়া যাচ্ছে`, `ভিজিট করুন`, `করুন <URL>`, `dial`, `পর্যন্ত`) pushed the generator into unidiomatic substitutes (`activate *566*33#`, `check <URL>`, `Order at` for salon booking, `অবধি ... চালু হয়েছে` aspectual clash, `চলছে এখন` word-order calque) and dropped grammatical postpositions (`মেয়াদে`/`জন্য`).
3. **One Policy-Misaligned Reviewer Catch (`syn_b3_0022`):** `17/18` rejections were valid quality catches. However, `syn_b3_0022` had completely natural Bengali prose and was rejected solely for Bengali-digit USSD (`*৩২১*১৫#`). Empirical inspection of `eligible_train.parquet` shows **50.0% (`32/64`)** of real Bengali/CodeMix USSD Promotional SMS use Bengali digits (`*১২১*...#`), while `Pilot+B1+B2` used 100% ASCII USSD. Batch 4 needs an explicit, shared USSD-digit rule upfront so generator and reviewer do not diverge.

---

## 1. Rejection Reasons by Category (Exact Counts & IDs)

All **18** rejected candidates (`17` from Round 1 `raw_batch3_50.jsonl`, `1` from Round 2 `replacements.jsonl`: `syn_b3_r_0010`) received multi-label reason tags in [`review_batch3_50.jsonl`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch3_50/review_batch3_50.jsonl) (`29` total reason tags across `18` rows; `0` automated schema/dedup rejections).

### 1.1 Flat Reason Counts (`29` Tags Across `18` Rows)

| Rejection Reason Tag | Count | Rejected IDs |
| :--- | ---: | :--- |
| `independent_review:unnatural_phrasing` | **10** | `syn_b3_0002`, `syn_b3_0005`, `syn_b3_0006`, `syn_b3_0021`, `syn_b3_0023`, `syn_b3_0036`, `syn_b3_0038`, `syn_b3_0046`, `syn_b3_0049`, `syn_b3_r_0010` |
| `independent_review:cross_batch_scenario_clone` | **6** | `syn_b3_0012`, `syn_b3_0029`, `syn_b3_0035`, `syn_b3_0037`, `syn_b3_0047`, `syn_b3_0049` |
| `independent_review:shallow_template_clone` | **5** | `syn_b3_0012`, `syn_b3_0018`, `syn_b3_0020`, `syn_b3_0029`, `syn_b3_0037` |
| `independent_review:near_duplicate_slot_template` | **2** | `syn_b3_0021`, `syn_b3_0047` |
| `independent_review:unrealistic_ussd_digits` | **2** | `syn_b3_0020`, `syn_b3_0022` |
| `independent_review:near_duplicate_paraphrase` | **1** | `syn_b3_0035` |
| `independent_review:weak_promotional_cue` | **1** | `syn_b3_0018` |
| `independent_review:metadata_mismatch` | **1** | `syn_b3_0036` |
| `independent_review:domain_mismatch` | **1** | `syn_b3_0046` |

### 1.2 Exact Multi-Label Combinations & Stratum Breakdown (`18` Rows)

- **Exact Reason Combinations (`N = 18`):**
  - `['unnatural_phrasing']`: **6** (`syn_b3_0002`, `0005`, `0006`, `0023`, `0038`, `r_0010`)
  - `['cross_batch_scenario_clone', 'shallow_template_clone']`: **3** (`syn_b3_0012`, `0029`, `0037`)
  - `['shallow_template_clone', 'weak_promotional_cue']`: **1** (`syn_b3_0018`)
  - `['shallow_template_clone', 'unrealistic_ussd_digits']`: **1** (`syn_b3_0020`)
  - `['near_duplicate_slot_template', 'unnatural_phrasing']`: **1** (`syn_b3_0021`)
  - `['unrealistic_ussd_digits']`: **1** (`syn_b3_0022`)
  - `['cross_batch_scenario_clone', 'near_duplicate_paraphrase']`: **1** (`syn_b3_0035`)
  - `['metadata_mismatch', 'unnatural_phrasing']`: **1** (`syn_b3_0036`)
  - `['domain_mismatch', 'unnatural_phrasing']`: **1** (`syn_b3_0046`)
  - `['cross_batch_scenario_clone', 'near_duplicate_slot_template']`: **1** (`syn_b3_0047`)
  - `['cross_batch_scenario_clone', 'unnatural_phrasing']`: **1** (`syn_b3_0049`)
- **By Language (Rejected / Total Reviewed):** `Bengali`: **6/20** (`30.0%`), `CodeMix`: **5/18** (`27.8%`), `Banglish`: **5/24** (`20.8%`), `English`: **2/6** (`33.3%`).
- **By Domain:** `telecom`: **10**, `personal_or_general`: **4**, `shopping_ecommerce`: **2**, `delivery_parcel`: **1**, `employment`: **1**. (`banking_mfs` and `civic_utility_health` had **0** rejections.)

### 1.3 Complete Audit Ledger of All 18 Rejected Candidates

Max cosine similarities below were computed via character 3–5-gram TF-IDF against all `400` prior accepted synthetic rows (`Pilot 300 + B1 50 + B2 50`) on `raw`, `slot_template_text`, and `brand_masked` + `slot_template_text` (`bslot`).

| ID (Round) | Stratum (`Lang \| Dom \| Len \| Brand`) | Rejected Text | Max Sim vs Prior 400 (`raw / slot / bslot` -> Match) | Rejection Reasons | Primary Root Cause |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `syn_b3_0002` (R1) | `Bg \| telecom \| med \| -` | `210 minute talktime shudhu 129 takay 15 din meyad, activate *566*33#.` | `0.289 / 0.404 / 0.443` (`syn_b2_0002`) | `unnatural_phrasing` | Mechanical synonym swap avoiding `dial` (`activate <USSD>`) |
| `syn_b3_0005` (R1) | `Bg \| telecom \| med \| -` | `Bidesh roaming pack-e 25% kom rate shob Asia deshe, check https://example.net/roam-asia` | `0.242 / 0.247 / 0.255` (`syn_p2_b_0033`) | `unnatural_phrasing` | Ungrammatical Banglish (`Bidesh roaming`, `shob Asia deshe`) + abrupt `check <URL>` |
| `syn_b3_0006` (R1) | `Bg \| telecom \| long \| -` | `Notun postpaid connection nile prothom 3 mash 20GB data, 300 min ebong free caller tune sujog ekbarei kom mullo-te, offer link: https://example.com/postpaid-plus` | `0.223 / 0.269 / 0.237` (`syn_b1_0006`) | `unnatural_phrasing` | Semantic contradiction (`free` + `kom mullo-te`), dangling noun (`sujog` without verb), `mullo-te` |
| `syn_b3_0012` (R1) | `Bg \| personal \| med \| KodomStyle` | `KodomStyle-er Eid panjabi, fatua ar kamiz-e 15% special chhar! Call 01700-000052.` | `0.355 / 0.341 / 0.497` (`syn_b2_0012`) | `shallow_template_clone`, `cross_batch_scenario_clone` | POS-12 positional clone of `syn_b1_0012` & `syn_b2_0012` (`<BRAND>-er <ethnic wear>-e 15% ... Call <PHONE>`) |
| `syn_b3_0018` (R1) | `Bg \| delivery \| med \| PalkiCourier` | `PalkiCourier-e Dhaka-r bhetore cake ba fragile gift box delivery 45 taka.` | `0.482 / 0.506 / 0.481` (`syn_b2_0018`) | `shallow_template_clone`, `weak_promotional_cue` | POS-18 clone of `syn_b1_0018` & `syn_b2_0018`; dropped `matro`/discount cue (reads as flat price statement) |
| `syn_b3_0020` (R1) | `Bn \| telecom \| short \| -` | `৪৫ টাকায় ৩ জিবি নেট ৩ দিন, ডায়াল *৭৮৯*১৯#।` | `0.385 / 0.670 / 0.700` (`syn_b1_0034` / `syn_p2_a_0067`) | `shallow_template_clone`, `unrealistic_ussd_digits` | Bengali-digit USSD + slot-template clone of `syn_p2_a_0067` / `syn_b1_0001` / `syn_b1_0034` |
| `syn_b3_0021` (R1) | `Bn \| telecom \| med \| KorotoaTel` | `করতোয়াটেলে মাত্র ৮৮ টাকা রিচার্জে ৪ জিবি ডেটা ও ১২০ মিনিট পাবেন সাত দিন।` | `0.416 / 0.483 / 0.571` (`syn_b1_0035` / `syn_b2_0021`) | `unnatural_phrasing`, `near_duplicate_slot_template` | POS-21 clone of `syn_b2_0021` (`পাওয়া যাচ্ছে` -> `পাবেন সাত দিন` missing `মেয়াদে`, mixed numerals) |
| `syn_b3_0022` (R1) | `Bn \| telecom \| med \| -` | `কম খরচে বিদেশে কথা বলুন প্রতি মিনিট দেড় টাকায়, চালু করতে ডায়াল *৩২১*১৫#।` | `0.254 / 0.404 / 0.404` (`syn_b1_0034`) | `unrealistic_ussd_digits` | Bengali-digit USSD (`*৩২১*১৫#`); **prose itself was natural** (borderline reviewer criterion) |
| `syn_b3_0023` (R1) | `Bn \| telecom \| med \| -` | `আজ রাত ১১টা অবধি ১১৮ টাকায় ৬ জিবি বোনাস ডেটা চালু হয়েছে, লিংক: example.net/data118` | `0.283 / 0.380 / 0.386` (`syn_b2_0035`) | `unnatural_phrasing` | Aspectual clash (`অবধি ... চালু হয়েছে`) from avoiding `পর্যন্ত`/`থাকছে`/`মিলবে` + intra-batch overlap with `0003` |
| `syn_b3_0029` (R1) | `Bn \| personal \| med \| KodomStyle` | `কদমস্টাইলের বাটিক ওড়না আর কুর্তির নতুন ডিজাইনে ২০% ছাড়, কল ০১৭০০-০০০০৬৪।` | `0.386 / 0.316 / 0.401` (`syn_b2_0029`) | `shallow_template_clone`, `cross_batch_scenario_clone` | POS-29 1-for-1 clone of `syn_b2_0029` (`... ওড়না ও/আর ... ২০% ছাড়, কল <PHONE>।`) |
| `syn_b3_0035` (R1) | `CM \| telecom \| med \| -` | `সপ্তাহান্তে 79 টাকায় 5GB streaming data চালু হয়েছে, অফার লিংক: example.com/stream5` | `0.543 / 0.619 / 0.623` (`syn_b2_0035`) | `near_duplicate_paraphrase`, `cross_batch_scenario_clone` | POS-35 near-verbatim paraphrase of `syn_b2_0035` (`Weekend offer হিসেবে 69` -> `সপ্তাহান্তে 79`) |
| `syn_b3_0036` (R1) | `CM \| telecom \| long \| -` | `Monthly family share plan-এ আজই join করলে 40GB shared internet এবং 400 min call বোনাস পাবেন ৩০ দিন, বিস্তারিত: https://example.org/family-share` | `0.265 / 0.276 / 0.276` (`syn_b2_0006` / `syn_b2_0036`) | `unnatural_phrasing`, `metadata_mismatch` | Missing postposition (`৩০ দিন মেয়াদে`), awkward `call বোনাস`, mixed numerals in clause, `has_urgency=false` despite `আজই` |
| `syn_b3_0037` (R1) | `CM \| shopping \| med \| MayurTech` | `ময়ূরটেক শপে bluetooth neckband ও fast charger অর্ডারে 20% instant ছাড়।` | `0.324 / 0.312 / 0.376` (`syn_b2_0037`) | `shallow_template_clone`, `cross_batch_scenario_clone` | POS-37 1-for-1 clone of `syn_b2_0037` (`<BRAND> শপে <g1> ও <g2> অর্ডারে 20% instant <discount/ছাড়>।`) |
| `syn_b3_0038` (R1) | `CM \| shopping \| med \| -` | `ছোট বাচ্চাদের baby diaper আর lotion কম্বো-তে 15% flat discount চলছে এখন।` | `0.329 / 0.381 / 0.395` (`syn_p2_b_0012`) | `unnatural_phrasing` | English word-order calque (`চলছে এখন।` vs `এখন চলছে`) + repeats `syn_p2_b_0012` diaper+lotion pairing |
| `syn_b3_0046` (R1) | `CM \| employment \| med \| -` | `নতুন graphic design, freelancing স্কিল batch-এ ভর্তিতে 35% ছাড়, দেখুন example.edu.bd/skills` | `0.273 / 0.395 / 0.433` (`syn_p2_b_0081`) | `domain_mismatch`, `unnatural_phrasing` | Course admission (`.edu.bd`) belongs to `education`, not `employment`; clunky asyndetic noun stack |
| `syn_b3_0047` (R1) | `En \| telecom \| med \| -` | `Unlimited midnight browsing 1AM-7AM for 7 days at 39 TK, dial *247*14#.` | `0.443 / 0.604 / 0.636` (`syn_p2_b_0136`) | `near_duplicate_slot_template`, `cross_batch_scenario_clone` | POS-47 repeats `syn_p2_b_0136` + `syn_b1_0004` midnight browsing pack, with comma splice before `dial` |
| `syn_b3_0049` (R1) | `En \| personal \| med \| -` | `Flat 25% off hair spa and haircut bookings on weekdays. Order at https://example.com/salon-week` | `0.225 / 0.212 / 0.214` (`syn_b2_0048` / `0041`) | `unnatural_phrasing`, `cross_batch_scenario_clone` | POS-49 repeats `syn_b1_0049` salon/spa 25% off; wrong CTA verb (`Order at` for hair appointments) |
| `syn_b3_r_0010` (R2) | `Bn \| personal \| med \| KodomStyle` | `কদমস্টাইলের শীতকালীন শালে দুটি কিনলে ২০% ছাড় এ সপ্তাহে, হেল্পলাইন ০১৭০০-০০০০৬৪।` | `0.281 / 0.239 / 0.251` (`syn_b1_0029` / `0041`) | `unnatural_phrasing` | Grammatical case error in hurried rewrite (`শালে` locative `-এ` before transitive `দুটি কিনলে`) |

---

## 2. Did Linguistic Diversity Constraints Cause Unnatural Phrasing?

**Short answer:** **Partially yes — specifically anti-TF-IDF lexical avoidance (`dial`, `ভিজিট করুন`, `থাকছে`, `মিলবে`, `পাওয়া যাচ্ছে`, `পর্যন্ত`), NOT conjunction balancing (`ও`/`আর`/`এবং`) or CodeMix structural gates.**

### 2.1 Conjunction Balancing (`ও` / `আর` / `এবং`)
- In `raw_batch3_50.jsonl` (`27` Bn+CM rows), the generator balanced conjunctions cleanly (`' ও '`: `4/27`, `' এবং '`: `5/27`, `' আর '`: `5/27`), closely aligning with real `eligible_train` Bn+CM Promotional (`N = 463`: `' ও '` `4.97%`, `' এবং '` `8.64%`, `' আর '` `1.94%`).
- **Did avoiding `' ও '` cause the grammar errors?** **No.** Two of the rejected rows (`syn_b3_0021`, `syn_b3_0037`) actually *used* `' ও '`, and two (`syn_b3_0029`, `syn_b3_0038`) used `' আর '`. Conjunction diversity did not force ungrammatical Bengali or CodeMix. The only indirect effect was on `syn_b3_0029`, where swapping `'ও'` -> `'আর'` failed to break the `syn_b2_0029` template clone, and the subsequent hurried replacement (`syn_b3_r_0010`) introduced a locative case error (`শালে দুটি কিনলে`).

### 2.2 Anti-TF-IDF Verb & CTA Avoidance (Direct Cause of `5` Unnatural-Phrasing Rejections)
In Batch 3 (`prompt_version = stage_b_batch3_50_v2_lex_diverse`), the generator aggressively avoided Batch 2's top 3-way TF-IDF features (`মিলবে`, `থাকছে`, `পাওয়া যাচ্ছে`, `ভিজিট করুন`, `করুন <URL>` — all reduced to `0/27` in `raw_batch3_50.jsonl`) as well as common CTAs like `dial` and `পর্যন্ত`. This directly produced unidiomatic substitutions in **5** rejected rows:
1. **`syn_b3_0002` (`activate *566*33#`):** Avoided `"dial *566*33#"` (used in `B1`/`B2`) and treated the USSD code as the direct object of `"activate"`. Fixed naturally in `syn_b3_r_0001` (`activate korte *566*33# dial korun.`).
2. **`syn_b3_0005` (`check https://example.net/roam-asia`):** Avoided `"visit"` / `"bistarito"` / `"details"` and appended an abrupt English `"check <URL>"` onto ungrammatical Banglish.
3. **`syn_b3_0023` (`আজ রাত ১১টা অবধি ... চালু হয়েছে`):** Avoided `"পর্যন্ত"` (which appears in `11.23%` = `52/463` of real Bn+CM Promo, whereas `"অবধি"` appears in `0/463` = `0.00%`!) AND avoided `"থাকছে"`/`"মিলবে"`/`"পাওয়া যাচ্ছে"` by choosing `"চালু হয়েছে"` (`has started`), creating an aspectual clash with `"until 11 PM tonight"`.
4. **`syn_b3_0038` (`... 15% flat discount চলছে এখন।`):** Avoiding `"থাকছে"`/`"মিলবে"` while varying verb placement produced an English SVO word-order calque (`"চলছে এখন।"` instead of `"এখন চলছে।"`).
5. **`syn_b3_0049` (`Order at https://example.com/salon-week`):** Avoided `"Reserve"` / `"Book"` / `"Visit"` from `syn_b1_0049` and chose `"Order at"` for salon haircut/hair-spa bookings.

### 2.3 Length & CodeMix Structural Gates
- **CodeMix gates (`bn_script_en >= 0.80`, `intra_sentential_mix >= 0.60`, `0%` `hook + pure EN`):** **Did NOT cause unnatural phrasing.** All `5` CodeMix rejections (`0035, 0036, 0037, 0038, 0046`) were caused by cross-batch cloning (`0035, 0037`), dropped postpositions/metadata mismatch (`0036`), word-order calque (`0038`), or domain mislabeling (`0046`), not by Bengali-script intra-sentential mixing itself (all `13` final accepted CodeMix rows are fluent).
- **Length (`long(>120)`):** Two of the `8` long candidates (`syn_b3_0006` and `syn_b3_0036`) failed due to clause-level coherence errors (self-contradiction + dropped verb in `0006`; dropped postposition + `has_urgency` mismatch in `0036`). Longer multi-benefit sentences require stronger sentence-level proofreading.

---

## 3. Root-Cause Attribution: Template Avoidance vs. Grammar vs. Mixing vs. Reviewer Criteria

| Root Cause Dimension | Affected IDs | Count | Assessment |
| :--- | :--- | :---: | :--- |
| **1. Positional Skeleton Anchoring & Shallow Template Avoidance** | `syn_b3_0012`, `0018`, `0020`, `0021`, `0029`, `0035`, `0037`, `0047`, `0049` (+ topic repeat in `0038`, `0023`) | **9** (primary) | **Major Batch 3 failure mode.** Reusing `batch2_50`'s exact `(language, domain)` row ordering (`48/50` match) and `length_band` (`46/50` match) caused the generator to riff on the exact slot-template at each index (`POS 12, 18, 20, 21, 29, 35, 37, 47, 49`). |
| **2. Grammatical, Semantic & Collocation Mistakes** | `syn_b3_0002`, `0005`, `0006`, `0021`, `0023`, `0036`, `0038`, `0046`, `0049`, `syn_b3_r_0010` | **10** | Missing Bengali postposition (`মেয়াদে`/`জন্য` in `0021`, `0036`), case-marking error (`শালে` in `r_0010`), aspectual clash (`0023`), Banglish noun-adjunct/calque (`0005`, `0038`), semantic contradiction + missing verb (`0006`), comma splice (`0047`), wrong CTA verb (`0002`, `0049`), clunky noun stack (`0046`). |
| **3. Language Mixing & Script/Numeral Consistency** | `syn_b3_0021`, `0036` (minor); `0` language-tag misclassifications | **2** (minor) | Zero rows were rejected for wrong `language` bucket (`Banglish` vs `Bengali` vs `CodeMix` vs `English`). In `0021` (`৪ জিবি ... সাত দিন`) and `0036` (`40GB ... ৩০ দিন`), mixing numeral forms within the same offer clause was flagged alongside genuine postposition omissions. *(Empirical note: in `eligible_train`, `60/229 = 26.2%` of real Bengali Promo and `37/234 = 15.8%` of real CodeMix Promo contain both ASCII and Bengali digits, typically separated by role such as ASCII USSD/GB vs Bengali TK/days.)* |
| **4. Domain / Promotional Cue / Metadata Errors** | `syn_b3_0018`, `0036`, `0046` | **3** | `0018` lacked an explicit discount/promotional cue; `0036` had `has_urgency=false` despite `"আজই join করলে"`; `0046` placed a graphic design/freelancing course (`.edu.bd`) in `employment` instead of `education`. |
| **5. Reviewer Criteria: Valid Catches vs. Overly Strict** | **17 valid** vs **1 borderline/strict** (`syn_b3_0022`) | **18** | **17/18** rejections were valid, high-value catches that protected dataset quality without relaxing standards. **`syn_b3_0022`** was **overly strict against real-world data**: rejected solely for Bengali-digit USSD (`*৩২১*১৫#`), yet **`32/64` (`50.0%`)** of real Bn+CM USSD Promotional SMS in `eligible_train.parquet` use Bengali-digit USSD (`*১২১*৪২০৫#`, `*৫৫৫#`, etc.). |

### Why `cos >= 0.85` Did Not Catch the Clones
Automated deduplication ([`stage_b_pilot.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/stage_b_pilot.py)) correctly passed `0` rows above `0.85`, because swapping fictional brand names (`UtshobFashion -> KodomStyle`, `EkusheyTech -> MayurTech`), product pairs (`smartwatch ও powerbank -> bluetooth neckband ও fast charger`), or numbers (`69 -> 79`) keeps character 3–5-gram cosine between **`0.21` and `0.70`** even on brand-masked slot templates (`bslot_cos = 0.700` on `syn_b3_0020`, `0.636` on `syn_b3_0047`, `0.623` on `syn_b3_0035`, `0.571` on `syn_b3_0021`). Independent human/LLM review remains essential for structural clones in the `0.45–0.75` `bslot_cos` band.

---

## 4. Actionable Batch 4 Generation & Review Rules (No Quality Relaxation)

Apply all **7** rules below in Batch 4 generation and review to raise Round 1 acceptance back above `90%+` while keeping the same strict quality bar:

1. **Break Positional Batch Skeletons (Eliminate Index-Aligned Generation):**
   - **Do NOT** generate Batch 4 in the same row-by-row `(language, domain, length_band)` sequence used in Batches 1–3. Shuffle the target stratum plan randomly before generating scenarios, and explicitly prohibit reusing `B1`/`B2`/`B3` scenarios at matching strata (`<BRAND>-er <ethnic wear>-e 15% ... Call <PHONE>`, `<BRAND>-e Dhaka ... delivery <N> taka`, `<BRAND> শপে <g1> ও <g2> অর্ডারে 20% instant ...`, `midnight browsing`).
2. **Add an Automated Pre-Review `bslot_cos >= 0.55` Structural Clone Warning:**
   - Before handing `raw_batch4_50.jsonl` to `@reviewer`, run a read-only check computing `brand_masked` + `slot_template_text` char-ngram cosine against `Pilot + B1 + B2 + B3` (`450` rows). Keep `0.85` as the hard deduplication gate, and **treat any pair with `bslot_cos >= 0.55` (or `raw_cos >= 0.45`) as a mandatory rewrite trigger** before submission.
3. **Stop Artificial Synonym Blacklisting; Allow Natural High-Frequency Bengali Words:**
   - Do **not** ban `"ও"`, `"আর"`, `"এবং"`, `"পর্যন্ত"`, `"থাকছে"`, `"পাবেন"`, `"পাওয়া যাচ্ছে"`, `"dial"`, `"ভিজিট করুন"`, or `"দেখুন"`. Use them at natural `eligible_train`-like rates rather than forcing awkward substitutes (`activate <USSD>`, `অবধি ... চালু হয়েছে`, `check <URL>`, `Order at` for bookings, `চলছে এখন`).
   - Reserve `"Book via"` / `"Book at"` / `"Schedule at"` for services/appointments and `"Order at"` / `"Shop at"` only for physical products.
4. **Mandatory Grammar, Postposition & Numeral-Consistency Self-Check:**
   - **Duration postpositions:** Whenever a Bengali/CodeMix/Banglish clause specifies duration after a verb (`পাবেন`, `চলবে`, `থাকছে`, `paben`), require an explicit postposition/noun (`মেয়াদে`, `জন্য`, `দিনে` / `din meyad`, `diner jonno` — never bare `... পাবেন ৩০ দিন` or `... পাবেন সাত দিন`).
   - **Case & verb checks:** Check transitive object phrases for spurious locative `-এ` (`শালে দুটি কিনলে` -> `দুটি শাল একসাথে কিনলে`), verify every long (`>120`) sentence has a complete finite verb and no contradictory qualifiers (`free ... kom mullo-te`), and avoid English SVO adverb placement in Bengali (`চলছে এখন` -> `এখন চলছে`).
   - **Clause-internal numeral consistency:** Avoid mixing Bengali digits, spelled-out numbers, and ASCII digits inside the same offer-benefit phrase (e.g., `40GB ... 400 min ... ৩০ দিন`).
5. **Align Generator & Reviewer on an Explicit USSD-Digit Policy:**
   - **Recommendation for Batch 4:** Standardize on **ASCII digits for all USSD codes** (e.g., `*121*45#`, never `*1234#`) across all four languages so every USSD code is GSM-dialable and matches Pilot/B1/B2/B3 conventions (while noting in the final dataset card that `eligible_train` contains both ASCII and Bengali-digit USSD in real carrier SMS). Ensure `@reviewer` and `@generator` share this exact rule so `0` candidates are wasted on USSD script mismatches.
6. **Strict Domain & Promotional-Cue Verification:**
   - **`employment` vs `education`:** Only label a row `employment` if the offer directly concerns jobs, hiring alerts, CV/interview prep, recruitment expos, or freelancer job-portal plans — **never** skill/training course admissions or `.edu.bd` course batches (those belong to `education`).
   - **Explicit promo cue on flat-rate offers:** Every `Promotional` delivery/service row must carry an explicit promotional signal (`ছাড়`, `discount`, `অফার`, `ফ্রি`, `cashback`, `মাত্র`, `reduced`, `cuts ... by X%`) rather than reading like a neutral tariff table.
7. **Metadata & Replacement Consistency Gate:**
   - Automatically verify `has_urgency` matches urgency cues (`আজই`, `আজ রাত`, `শেষ দিন`, `until Sunday`, `ajkei`, `agamikal porjonto`), `has_link` matches URL presence, `char_len` matches `length_band`, and every replacement row (`syn_b4_r_*`) passes the full grammar + `bslot_cos < 0.55` check before Round 2 review.
