# Stage B Synthetic Augmentation Completion Report (`curation_v2`)

**Date:** 2026-11-20  
**Scope:** Full 7-stage Stage B completion synthesis (`pilot_300` + `Batches 1–7`)  
**Related Batch 7 Report:** [`stage_b_batch7_45_report.md`](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/stage_b_batch7_45_report.md)  
**Status:** **STAGE B GENERATION COMPLETE & STOPPED — Total Balanced Pool `4,644` Rows (`1,548` per class, `387` Promotional per language)**

---

## 1. Overview of All Seven Generation Stages (`Pilot` + `Batches 1–7`)

Stage B augmented the leakage-free `curation_v2` `eligible_train` dataset (`3,999` real SMS: `1,509` Normal, `942` Promotional, `1,548` Smishing) to exact class parity (`T = 1,548` per class) and exact within-Promotional language parity (`387` each across `Banglish`, `Bengali`, `CodeMix`, `English`) without generating any synthetic `Smishing` rows.

Across **eight generation campaigns** (the initial `300`-row Pilot followed by seven `45–50`-row iterative Promotional batches `Batches 1–7`), **`718`** candidates underwent automated screening + independent row-by-row review, yielding **`645`** accepted synthetic rows (**`89.83%`** overall acceptance) and **`73`** rejected candidates preserved in batch `rejected.jsonl` logs.

| Stage | Cohort Directory | Target Scope | Screened | Accepted (`Rate`) | Rejected | Class Split (`N / P / S`) | Key Engineering / Linguistic Focus |
| :--- | :--- | :--- | ---: | ---: | ---: | :---: | :--- |
| **Pilot (`300`)** | [`curation_v2/`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2) | Remaining `Normal` + initial `Promotional` | 312 | 300 (96.15%) | 12 | `39 / 261 / 0` | Initial multi-language generation across 8 domains & cross-lingual `family_id` groups |
| **Batch 1 (`50`)** | [`batch1_50/`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch1_50) | Promotional deficit slice 1 | 52 | 50 (96.15%) | 2 | `0 / 50 / 0` | Eliminated Pilot CodeMix `hook!/? + pure EN` bias, reduced `!` & brand concentration, added brandless quota (`>=30%`), retired `*1234#` |
| **Batch 2 (`50`)** | [`batch2_50/`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch2_50) | Promotional deficit slice 2 | 53 | 50 (94.34%) | 3 | `0 / 50 / 0` | Raised naturally brandless share to `60%`; introduced fresh batch-isolated fictional brands |
| **Batch 3 (`50`)** | [`batch3_50/`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch3_50) | Promotional deficit slice 3 | 68 | 50 (73.53%) | 18 | `0 / 50 / 0` | Stricter cross-batch scenario-clone & shallow-template rejection (filtered 18/68 rows) |
| **Batch 4 (`50`)** | [`batch4_50/`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch4_50) | Promotional deficit slice 4 | 63 | 50 (79.37%) | 13 | `0 / 50 / 0` | Merged Batch 1–4 brands into `DEFAULT_KNOWN_BRANDS`; added 10-seed (`42..51`) + bootstrap & cluster-bootstrap CIs and 5-cohort evaluation |
| **Batch 5 (`50`)** | [`batch5_50/`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch5_50) | Promotional deficit slice 5 | 58 | 50 (86.21%) | 8 | `0 / 50 / 0` | Unified brand detection (NFC + Bengali/Banglish inflections, `0` mismatches); accepted Bengali-digit USSDs; `100%` 3-way stratum plan + shuffled order; real `P25/P50/P75` length matching |
| **Batch 6 (`50`)** | [`batch6_50/`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch6_50) | Promotional deficit slice 6 | 61 | 50 (81.97%) | 11 | `0 / 50 / 0` | Preserved unified detector (`0` mismatches) & length match (`P50=74.0`); verified surface-only ROC-AUC positive-class orientation |
| **Batch 7 (`45`)** | [`batch7_45/`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch7_45) | Final Promotional deficit (`45`) | 51 | 45 (88.24%) | 6 | `0 / 45 / 0` | Closed all remaining class & language deficits to `0`; exact `P50=74.0`, `64.44%` brandless, `8/8` diversity gates |
| **Subtotal New (`B1–B7`)** | — | Promotional balancing | **406** | **345 (84.98%)** | **61** | **`0 / 345 / 0`** | Clean 345-row cohort passing `8/8` diversity gates and aligned to real `eligible_train` Promo length/surface profile |
| **Grand Total (`Pilot + B1–B7`)** | — | Full Stage B augmentation | **718** | **645 (89.83%)** | **73** | **`39 / 606 / 0`** | Achieves exact `4,644` balanced training pool (`1,548` per class) |

---

## 2. The Original Pilot (`pilot_300`) and Its Documented Limitations

Audit diagnostics across Batches 1–7 established that while the original `300`-row Pilot (`39` Normal, `261` Promotional) succeeded in supplying the `39` missing `Normal` messages cleanly enough on isolation and schema, its `261` Promotional messages suffered from systematic structural artifacts (measured under unified [`schema.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/schema.py) + [`codemix_diversity.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/codemix_diversity.py)):

1. **CodeMix Discourse & Script Bias:** `85.00%` (`51/60`) of Pilot Promotional `CodeMix` rows followed a `[Bengali/Banglish hook !/?] + [pure English clause]` template (`hook_plus_pure_en_rate = 0.85`), with only `13.33%` true intra-sentential lexical mixing (`threshold >= 0.60`) and only `58.33%` Bengali-script + English (`41.67%` Banglish-Latin).
2. **Hyper-Punctuated Exclamation/Question Rates:** Sentence-1 exclamation rate reached **`56.32%`** (vs `<= 0.35` gate) and sentence-1 question rate reached **`9.20%`** (vs `<= 0.05` gate).
3. **Near-Zero Brandless Rate & Brand Over-Concentration:** Only **`1.92%` (`5/261`)** of Pilot Promotional messages were naturally brandless (vs `69.75%` in real `eligible_train` Promotional!), while fictional brands repeated heavily (`max_brand_count = 25` for `bornotelecom`, `23` each for `nodipay` and `purbachalbank`).
4. **Placeholder USSD Artifact:** All `15/15` (`100%`) USSD codes in the Pilot used synthetic `*1234#`.
5. **Length & Stratum Skew:** Pilot Promotional lengths skewed much longer (`P25/P50/P75 = 91.0 / 116.0 / 127.0`, `mean = 106.66 ± 32.86`; `40.23%` `long(>120)`) than real `eligible_train` Promotional (`66.0 / 74.0 / 99.0`, `mean = 89.20 ± 40.77`; `15.18%` `long`), leaving `pilot_300` surface-only `Promo vs Real Promo` ROC-AUC at **`0.7167`** (vs `0.4532–0.5512` in Batches 5–7 / `B1–B7`) and 3-way stratum coverage at only `71.26%` (`186/261`).
6. **Elevated Counterfactual Smishing Shift:** Adding `pilot_300` shifted `raw` counterfactual Smishing accuracy on `1,279` probed real Smishing rows by **`-10.40 pp` (`contact_swap`) / `-12.82 pp` (`contact_and_brand_swap`)**, compared with `-6.49 pp / -6.65 pp` for `B1–B6` (`300` Promo rows) and `-1.56 pp / -1.56 pp` for `batch7_45`.

Consequently, **`pilot_300` (`audit`) passes `0/8` diversity gates**, whereas **`B1–B7` (`combined_new_345`, `new_batch`) passes `8/8` diversity gates**.

---

## 3. Accepted & Rejected Candidate Statistics and Quality Trajectory Across Stages

Across all `345` new Promotional rows (`Batches 1–7`), **`61`** initial candidates were rejected (`15.02%` rejection rate) and replaced after independent row-by-row review. Rejection causes evolved from shallow surface/typo catches in the Pilot (`unnatural_phrasing: 12`) to strict cross-batch scenario clone filtering (`cross_batch_scenario_clone`), near-duplicate slot-template guards (`near_duplicate_slot_template`), domain/length-band verification, and brand-detection alignment in Batches 3–7.

| Cohort | Promo `P25 / P50 / P75` | `Mean ± SD` | `Brandless %` | `Max Brand` | `Bn-Script CM %` | `Intra-Sent CM %` | `Hook+PureEN %` | `Sent-1 ! %` | `Sent-1 ? %` | `*1234#` Share (`Tot`, `BN-dig`) | Gates (`8/8`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`eligible_train` (`942`)** | `66.0 / 74.0 / 99.0` | `89.20 ± 40.77` | `69.75%` | — | — | — | — | — | — | `— (122, 32)` | — |
| `pilot_300` (`P=261`) | `91.0 / 116.0 / 127.0` | `106.66 ± 32.86` | `1.92%` | `25` | `58.33%` | `13.33%` | `85.00%` | `56.32%` | `9.20%` | `100.0% (15, 0)` | `False (0/8)` |
| `batch1_50` (`50`) | `67.0 / 72.5 / 97.0` | `85.92 ± 29.26` | `36.00%` | `4` | `100.00%` | `100.00%` | `0.00%` | `10.00%` | `2.00%` | `0.00% (7, 0)` | **`True`** |
| `batch2_50` (`50`) | `69.25 / 74.5 / 86.5` | `86.40 ± 29.33` | `60.00%` | `3` | `100.00%` | `84.62%` | `0.00%` | `6.00%` | `0.00%` | `0.00% (7, 0)` | **`True`** |
| `batch3_50` (`50`) | `72.25 / 78.0 / 88.5` | `88.52 ± 28.83` | `62.00%` | `3` | `100.00%` | `84.62%` | `0.00%` | `0.00%` | `0.00%` | `0.00% (6, 0)` | **`True`** |
| `batch4_50` (`50`) | `87.5 / 95.5 / 107.0` | `102.96 ± 25.51` | `62.00%` | `3` | `100.00%` | `92.86%` | `0.00%` | `2.00%` | `0.00%` | `0.00% (5, 0)` | **`True`** |
| `batch5_50` (`50`) | `65.0 / 77.0 / 94.8` | `86.86 ± 32.22` | `62.00%` | `3` | `100.00%` | `100.00%` | `0.00%` | `10.00%` | `2.00%` | `0.00% (7, 2)` | **`True`** |
| `batch6_50` (`50`) | `65.0 / 74.0 / 98.25` | `86.04 ± 31.69` | `62.00%` | `3` | `100.00%` | `100.00%` | `0.00%` | `8.00%` | `0.00%` | `0.00% (7, 2)` | **`True`** |
| `batch7_45` (`45`) | `64.0 / 74.0 / 95.0` | `86.00 ± 33.02` | `64.44%` | `3` | `100.00%` | `100.00%` | `0.00%` | `4.44%` | `0.00%` | `0.00% (6, 2)` | **`True`** |
| **`B1–B7` (`345`)** | `68.0 / 78.0 / 99.0` | `89.00 ± 30.33` | `58.26%` | `4` | `100.00%` | `94.62%` | `0.00%` | `5.80%` | `0.58%` | `0.00% (45, 6)` | **`True (8/8)`** |
| **`Full All` (`P=606`)** | `70.25 / 94.0 / 122.75` | `96.61 ± 32.61` | `33.99%` | `25` | `83.66%` | `62.75%` | `33.33%` | `27.56%` | `4.29%` | `25.00% (60, 6)` | `False (6/8)` |

### Key Pipeline & Quality Milestones Across Stages
- **Batches 1–3:** Eliminated `hook+pure-EN` discourse shortcuts (`85% -> 0%`), cut `sent1_excl` (`56.32% -> 0–10%`), enforced batch-unique fictional brands (`max_brand <= 4`), scaled natural brandless prevalence (`1.92% -> 60–62%`), and hardened cross-batch scenario-clone detection.
- **Batch 4:** Integrated accepted Batch 1–4 fictional brands into `DEFAULT_KNOWN_BRANDS`, established the 5-cohort evaluation framework plus 10-seed (`42..51`) + cluster bootstrap 95% CIs.
- **Batch 5:** Consolidated all brand detection/masking/stripping into one NFC + Bengali/Banglish-inflection-aware canonical implementation (`0` mismatches), added Bengali-digit USSD support matching real `Bn+CM` SMS patterns, shuffled generation ordering across strata, and matched fine-grained `eligible_train` Promotional length distributions (`100%` 3-way stratum coverage in Batches 5–7).
- **Batches 6–7:** Sustained exact `P50 = 74.0` length alignment (`B6` & `B7`), `62–64.4%` brandless Promotional rates, `0` brand mismatches, `100%` matched 3-way strata, and verified surface-AUC positive-class orientation.

---

## 4. Final Balanced Dataset Composition (`curation_v2` + Stage B)

### 4a. Training Pool (`N = 4,644`)

| Split / Source | `Normal` (`Bn / Bg / CM / En`) | `Promotional` (`Bn / Bg / CM / En`) | `Smishing` (`Bn / Bg / CM / En`) | Total |
| :--- | :---: | :---: | :---: | ---: |
| `eligible_train` (`3,999`) | `1,509` (`396 / 417 / 312 / 384`) | `942` (`229 / 141 / 234 / 338`) | `1,548` (`371 / 404 / 357 / 416`) | **3,999** |
| Stage B Pilot (`300`) | `39` (`10 / 8 / 14 / 7`) | `261` (`65 / 115 / 60 / 21`) | `0` | **300** |
| Stage B Batches 1–7 (`345`) | `0` | `345` (`93 / 131 / 93 / 28`) | `0` | **345** |
| **Final Balanced Pool** | **`1,548`** (`406 / 425 / 326 / 391`) | **`1,548`** (`387 / 387 / 387 / 387`) | **`1,548`** (`371 / 404 / 357 / 416`) | **4,644** |

### 4b. Untouched Held-Out Validation, Test & OOD Evaluation Splits (`existing_unique.parquet`, `N = 11,821`)

All evaluation and quarantined artifacts remain untouched and strictly deduplicated (`0` exact or near-duplicate hits `cos >= 0.85` from any synthetic batch):
- **Primary Train Eligible:** `3,999` (`Normal: 1,509`, `Promotional: 942`, `Smishing: 1,548`)
- **SmishDetect Validation (`direct` + `family_overlap`):** `610 + 98 = 708` (`Normal: 265`, `Promotional: 178`, `Smishing: 265`)
- **SmishDetect Test (`direct` + `family_overlap`):** `1,376 + 322 = 1,698` (`Normal: 581`, `Promotional: 384`, `Smishing: 733`)
- **BanglaPhish-2026 OOD (`sms` + `other`):** `1,538 + 3,878 = 5,416` (`Normal: 2,336`, `Promotional: 80`, `Smishing: 3,000`)

---

## 5. Remaining Limitations & Known Diagnostic Properties

1. **High Lexical Synthetic-Origin Separability (`TF-IDF+LR`):** Even under 3-way `(language, domain, length_band)` stratum control and entity masking (`brand_masked`), linear TF-IDF classifiers still distinguish synthetic Promotional rows from real Promotional rows at high ROC-AUC (**`0.9560 ± 0.0021`** for `B1–B7` `345`; **`0.9327 ± 0.0092`** for `batch7_45`; **`0.9508 ± 0.0026`** for `full_combined_645`). In contrast, `surface_8feat` separability is at chance (`0.5083 ± 0.0046` on `combined_new_345`; `0.4301 ± 0.0216` on `batch7_45`). This reflects LLM-authored phrasing consistency and topic freshness relative to scraped real Bangladeshi SMS rather than surface or formatting leaks.
2. **Synthetic Contact & Fictional Entity Exclusivity:** Reserved URLs (`example.com`, `example.org`, `example.net`, `example.edu.bd`), synthetic phones (`01700-0000XX`), and fictional brand registries occur exclusively in synthetic `Normal`/`Promotional` rows (`0` in real `Smishing`), driving a `-11.81 pp` / `-13.92 pp` counterfactual shift in `raw` text view on `full_combined_645` (`-5.94 pp` / `-6.02 pp` on `B1–B7`, vs `0.00 pp` in `brand_masked` / `brand_stripped`).
3. **Legacy `pilot_300` Artifacts in Cumulative `full_combined_645`:** Because `pilot_300` was preserved unmodified (`261` Promo), `full_combined_645` passes `6/8` audit gates (`hook+pure-EN = 0.3333 > 0.15`, `max_brand_count = 25 > 5`). Downstream experiments should compare **`B1–B7`-only / downweighted-Pilot ablations** alongside the full `4,644`-row balanced set.
4. **External Generalization & Human Evaluation:** Bootstrap 95% CIs capture score uncertainty on the fixed dataset; held-out `smishdetect_test`, `BanglaPhish-2026` OOD evaluation, and native Bangladeshi human quality ratings remain external to Stage B curation.

---

## 6. Recommended Next Steps for Model Training & Research Evaluation

1. **Wait for Batch 7 Sign-Off Before Modifying Files:** Upon user approval of Batch 7, optionally merge Batch 7's 8 brands from [`batch7_45/brand_registry.json`](file:///home/bs01415/Documents/Programming/ExploringAIs/data/generated/curation_v2/batch7_45/brand_registry.json) into `DEFAULT_KNOWN_BRANDS` in [`src/bangla_smish/schema.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/schema.py) and verify a reproducible loader for `eligible_train + Batches 1–7` / `+ Pilot`.
2. **Controlled Ablation Matrix (Baseline vs. Stage B Balanced):** Run fold-safe `StratifiedGroupKFold` (`n_splits=5`, grouped by `family_id`, synthetic rows added strictly inside training folds only) comparing:
   - *(a)* **Unbalanced Real Baseline** (`eligible_train`, `3,999`)
   - *(b)* **Original-Only Reweighted / Downsampled** (`eligible_train` with class weighting / undersampling to `942/class`, zero synthetic)
   - *(c)* **`eligible_train + B1–B7` (`345` clean Promotional)** (isolating clean Promotional balancing without Pilot artifacts)
   - *(d)* **Full Stage B Balanced (`4,644`: `eligible_train + Pilot + B1–B7`)**
   across all **4 text views** (`raw`, `canon_masked`, `brand_masked`, `brand_stripped`) to quantify whether synthetic Promotional rows improve real Promotional-vs-Smishing precision/recall and `Recall @ <=1% FPR`.
3. **Held-Out & OOD Evaluation Suite:** Evaluate best checkpoints **once** on untouched `smishdetect_validation` (for threshold calibration), `smishdetect_test` (`1,698`), and `banglaphish_2026_ood` (`5,416` across `sms` and `other`), reporting per-language (`Bengali`, `Banglish`, `CodeMix`, `English`) Macro-F1, Smishing Recall @ `<=1%` FPR, and Normal->Smishing FPR.
4. **Classical -> Neural/Transformer Progression & GPU Training:** After validating classical TF-IDF/Linear/GBM baselines on CPU, train pretrained multilingual/Bengali transformers (`BanglaBERT`, `BanglaBERT-large`, `XLM-RoBERTa`, `mBERT`) on Colab/GPU under identical fold-safe 4-view + OOD protocols, plus robustness tests with counterfactual contact/brand swaps.
