# Post-Stage B — Stage 3 Completion & Experiment Readiness Audit

**Audit Date:** 2026-11-23  
**Repository State:** `main` at `4d341183cb7a6fce0ed773ff33da2a3b757d6f94` (clean working tree at audit start)  
**Policy & Execution Status:** **READ-ONLY AUDIT.** No datasets, splits, configurations, source files, notebooks, or synthetic cohorts were modified. Stage B synthetic generation remains **PERMANENTLY STOPPED** (`4,644` frozen combined pool). **Full Experiments A/B/C/D remain NO-GO.**

---

## 1. Executive Summary

- **[Verified fact]** **Protected Dataset & Bundle Integrity:** All 13 protected inputs in `PROTECTED_INPUT_DIGESTS`, `DataSet.csv` (`5df79924...`), `post_stage_b_bundle` (`v1`, manifest `2e2e76e2...`), and `post_stage_b_bundle_v2` (`v2`, manifest `308d1c88...`) passed live SHA256 and content verification. All `63/63` `unittest` regressions and `tests/test_pipeline.py` passed after the audit checks.
- **[Verified fact]** **Exact Model-View Collision Audit:** Re-running `scripts/audit_post_stage_b_model_views.py` on `v2` reproduced `128/560` failing cross-boundary checks (`64/140` in `brand_masked`, `64/140` in `brand_stripped`) traced to 3 Smishing row pairs (6 singleton families), while `raw` and `canon_masked` had `0/280` exact string collisions (and `0/40` adversarial CV string collisions, `0/6` mutual held-out string collisions).
- **[Verified fact]** **Task A Near-Duplicate Screen:** Reproducing the 12,466-row char-ngram (`3–5`, `char_wb`) TF-IDF cosine `>= 0.85` screen recovered **32 unique cross-boundary pairs** (3 strong-normalized `>= 0.85`, 29 slot-template-only `>= 0.85`; 1 pair overlaps both). **Zero** pairs involve Stage B Pilot (`300`), Batches 1–7 (`345`), or `BanglaPhish-2026` OOD (`5,416`). **Zero** pairs are exact byte/string duplicates (`raw_eq=False`, `canon_eq=False`). However, **11 pairs** are classified as *Ambiguous; requires human review* (e.g., same phishing URL or `canon_masked` CTA-only clones from upstream `smishdetect_2026` / `bangalabarta_2025`), and **21 pairs** are *Semantically related but distinct*.
- **[Verified fact]** **Task B Option A Feasibility:** Restricting confirmatory evaluation to `raw` and `canon_masked` (160 units; 24 planned comparisons vs A) preserves identical ordered A/B training IDs, C/D synthetic training-only additions, identical per-fold validation IDs, and yields `0/280` exact-input collisions in `v2`. Two structural subtleties remain: (i) in `canon_masked`, counterfactual `contact_swap` collapses to `orig` on `1,261/1,279` rows (`98.59%`), leaving `contact_brand_swap` (`238` brand-bearing rows) as the active probe in `canon_masked`; and (ii) `tests/test_post_stage_b_pipeline.py:209` currently asserts 320 units for the full config.
- **[Verified fact]** **Task C Statistical Protocol Gap:** The current runner selects best epochs by unweighted CV loss, selects the `<=1%` Normal FPR threshold on single ~301-row CV folds (`allowed_fp=3`), evaluates held-out/OOD per fold without a 5-fold ensemble/aggregation rule, runs `mcnemar_pvalue` and `paired_family_accuracy_delta` only on OOF accuracy (not on held-out/OOD or on the low-FPR endpoint), and uses a single seed (`42 + fold`).
- **[Verified fact]** **Task D Colab & Drive Storage Blocker:** Although T4 GPU memory is likely feasible with FP16 (`auto` native check), `hf_train_and_predict_fold` sets `save_strategy="epoch"` with **no `save_total_limit`**, and `_read_committed_unit` verifies `trainer_checkpoint` hashes on disk. Even under Option A (160 units × 3 epoch saves = 480 checkpoints), retaining all checkpoints on Google Drive consumes roughly **~1.09 TB** (decimal GB estimate from parameter counts) and risks Drive storage exhaustion and FUSE I/O throttling. Moreover, `PostStageBConfig` lacks `gradient_accumulation_steps`, and local `.venv` lacks `torch`/`transformers` (so peak VRAM and wall-time on T4 remain unmeasured).

---

## 2. Near-Duplicate Candidate Adjudication Table (Task A)

### 2.1 Audit Source & Methodology

- **[Verified fact]** **Source screen located:** Described in [post_stage_b_integrity_and_leakage_report.md](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/post_stage_b_integrity_and_leakage_report.md#L92-L108) (no standalone JSON/script artifact was committed in `scripts/` or `reports/`).
- **[Verified fact]** **Exact reproduction:** Fitting `TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)` across the `12,466` decoded rows in `data/processed/post_stage_b_bundle_v2/` (`real_train_pool`: 3,999; `synthetic_pilot_300`: 300; `synthetic_batches_b1_b7`: 345; `eval_smishdetect_validation`: 708; `eval_smishdetect_test`: 1,698; `eval_banglaphish_ood`: 5,416) using `strong_normalize_text` and `slot_template_text` from [`curation.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/curation.py#L99-L133) reproduced the exact report counts:
  - Strong-normalized `cos >= 0.85`: `real_pool <-> test = 2`, `CV_cross_fold = 1` (appearing once in fold 2 and once in fold 4), `real_pool <-> val = 0`, `test <-> val = 0`, any Pilot/B1–7/OOD `= 0`.
  - Slot-template `cos >= 0.85`: `real_pool <-> test = 13`, `CV_cross_fold = 7` (14 fold-boundary hits: folds 0/1/2/3/4 = 3/3/3/4/1), `real_pool <-> val = 6`, `test <-> val = 4`, any Pilot/B1–7/OOD `= 0`.
  - Combined unique cross-boundary pairs: **32** (`cur_e2a702578e9e <-> cur_40dd2f53ca2f` exceeds `0.85` on both).
- **[Verified fact]** **Why Stage A assigned distinct `family_id`s:** Re-computing `strong_normalize_text` TF-IDF on `curation_v1/existing_unique.parquet` (`11,947` rows) gave `0.8485`, `0.8497`, and `0.8497` for the three strong-normalized pairs (and `0.8485`, `0.8494`, `0.8498` on `curation_v2/existing_unique.parquet`, `11,821` rows) — all strictly `< 0.85`. Adding the 645 Stage B synthetic rows shifted IDF weights enough to nudge those three borderline pairs across `0.85`. Meanwhile, Stage A's `slot_template_key` collapse required punctuation-stripped *exact* equality (`curation.py:626–633`).
- **[Unresolved issue]** **Tokenizer-level check on the 32 pairs:** Local `.venv` does not have `torch`, `transformers`, `tokenizers`, `sentencepiece`, or `normalizer` installed (`ModuleNotFoundError`), and the temporary `/tmp/exploringais-tokenizer-*` directories from the prior Colab audit are absent. Therefore, backbone-specific integer `input_ids` comparison for these 32 pairs could not be executed locally; only `raw`, `canon_masked`, `brand_masked`, and exact-string equality were verified directly (`0/32` exact matches in `raw`, `canon_masked`, or `brand_masked`).

### 2.2 Complete 32-Pair Near-Duplicate Adjudication Table

*(Excerpts are privacy-safe: URLs, domain names, and phone numbers are masked as `<URL>`/`<PHONE>`. Classifications are strictly label-blind and model-blind.)*

| # | Row 1 (Cohort / Fold / Fam) | Row 2 (Cohort / Fold / Fam) | Norm Cos | Slot Cos | Exact `raw` / `canon` Eq | Existing Family Relationship | Scenario vs. Wording & Privacy-Safe Excerpts | Classification |
|---:|---|---|---:|---:|---|---|---|---|
| 1 | `cur_2ee2c999bea1` (`real_pool` F1, `fam_ee1767057b` [n=2], Bn) | `cur_cfa2f9f81a31` (`test`, `fam_735d8c7d90` [n=2], Bn) | **0.8528** | 0.7594 | No / No | Distinct families (`n=2` each, same-label siblings in each). Both trace to `legacy_2024` + `bangalabarta_2025` + `smishdetect_2026`. | **Same campaign & identical URL** (`ebayisapidlld…`), differs only by 2-word action (`লাঞ্চ করার` vs `একদিন কাটানোর`). Ex1: `“শাকিবের সাথে লাঞ্চ করার সুযোগ পেতে আজই আইপিএলে বাজি ধরুন… <URL>”` / Ex2: `“শাকিবের সাথে একদিন কাটানোর সুযোগ পেতে…”` | **Ambiguous; requires human review** |
| 2 | `cur_e2a702578e9e` (`real_pool` F4, `fam_8866d54624` [n=1], Bn) | `cur_40dd2f53ca2f` (`test`, `fam_5408bae462` [n=1], Bn) | **0.8513** | **0.8513** | No / No | Distinct singletons (`sd_train_1066` vs `sd_test_1244`). | Shared everyday personal scenario template (`বিকেলে … খেলা` vs `ম্যাচ`), no URL/contact/brand. Ex1: `“দোস্ত, আজ বিকেলে ফুটবল খেলা দেখবো। তোমার বাসায় আসবো।”` / Ex2: `“দোস্ত, আজ ফুটবল ম্যাচ দেখবো।…”` | **Semantically related but distinct** |
| 3 | `cur_31f35c6da04c` (`real_pool` F4, `fam_3d1e583fac` [n=1], Bn) | `cur_647968080b19` (`real_pool` F2, `fam_ed9a7b3fad` [n=1], Bn) | **0.8501** | 0.7395 | No / No | Distinct singletons (`legacy_1963`/`sd_train_4157` vs `legacy_1961`/`sd_train_288`). | **Same organization & URL** (`www.joinnavy…`), distinct job posts (`সরাসরি কমিশন্ড অফিসার` vs `অফিসার ক্যাডেট`). Ex1: `“বাংলাদেশ নৌবাহিনীতে সরাসরি কমিশন্ড অফিসার পদে নিয়োগ চলছে…”` / Ex2: `“…অফিসার ক্যাডেট পদে নিয়োগ চলছে…”` | **Semantically related but distinct** |
| 4 | `cur_0667dc3a1230` (`real_pool` F0, `fam_f456015736` [n=2], Bn) | `cur_83d1d273eece` (`real_pool` F3, `fam_63aa660615` [n=2], Bn) | 0.7507 | **0.8524** | No / No | Distinct families (`n=2` each). | Same password-reset phishing template, different domains (`passwordupdate…` vs `passwordreset…`) and phrasing (`করতে হবে` vs `অবিলম্বে … করুন`). Ex1: `“আপনার পাসওয়ার্ডটি রিসেট করতে হবে। এখানে ক্লিক করুন: [<URL>]”` / Ex2: `“আপনার পাসওয়ার্ডটি অবিলম্বে রিসেট করুন।…”` | **Ambiguous; requires human review** |
| 5 | `cur_356bfb9efc77` (`real_pool` F2, `fam_3ed27fec7c` [n=1], Bn) | `cur_0aabd655f8b8` (`val`, `fam_df9b72ac6f` [n=4], Bn) | 0.5678 | **0.9476** | No / No | Distinct (`n=1` vs `n=4` Mashrafe BPL family). | **Identical `canon_masked` body except 1 CTA verb** (`ক্লিক করুন` vs `শুরু করুন`); different URLs. Ex1: `“মাশরাফির সাথে একান্ত সাক্ষাতের সুযোগ পেতে বিপিএলে বাজি ধরুন। এখনই ক্লিক করুন: <URL>”` / Ex2: `“…এখনই শুরু করুন: <URL>”` | **Ambiguous; requires human review** |
| 6 | `cur_356bfb9efc77` (`real_pool` F2, `fam_3ed27fec7c` [n=1], Bn) | `cur_4ea4417160ed` (`val`, `fam_df9b72ac6f` [n=4], Bn) | 0.5399 | **0.9080** | No / No | Same as #5 (`val` sibling of #5). | **Identical `canon_masked` body except CTA** (`এখনই ক্লিক করুন` vs `যোগ দিন`); different URLs. Ex1: `“মাশরাফির সাথে একান্ত সাক্ষাতের সুযোগ… এখনই ক্লিক করুন: <URL>”` / Ex2: `“…যোগ দিন: <URL>”` | **Ambiguous; requires human review** |
| 7 | `cur_40845fd1114b` (`real_pool` F3, `fam_1a97c102c8` [n=2], Bn) | `cur_902a0a1a8ac5` (`real_pool` F0, `fam_d7e75c9326` [n=1], Bn) | 0.4542 | **0.9039** | No / No | Distinct (`fam_1a97c102c8` has iPhone+car; `fam_d7e75c9326` singleton). | **Identical `canon_masked` body except CTA** (`শুরু করুন` vs `যোগ দিন`); different URLs. Ex1: `“ক্যাসিনোতে বাজি ধরুন এবং একটি ফ্রি আইফোন জিতুন! শুরু করুন: <URL>”` / Ex2: `“…যোগ দিন: <URL>”` | **Ambiguous; requires human review** |
| 8 | `cur_472b22039982` (`real_pool` F2, `fam_fcf852e8b3` [n=1], Bn) | `cur_7c14e8bae65c` (`test`, `fam_210d66c8f3` [n=3], Bn) | 0.4574 | **0.8732** | No / No | Distinct (`n=1` vs `n=3` casino-trip family). | **Identical `canon_masked` body except 1 word** (`এখনই শুরু করুন` vs `আজই যোগ দিন`); different URLs. Ex1: `“ক্যাসিনোতে বাজি ধরুন এবং ৭ দিনের জন্য একটি ফ্রি ট্রিপ জিতুন! এখনই শুরু করুন: <URL>”` / Ex2: `“…আজই যোগ দিন: <URL>”` | **Ambiguous; requires human review** |
| 9 | `cur_47f292401da6` (`real_pool` F1, `fam_a4a1f83853` [n=1], Bn) | `cur_7936ee1f5382` (`test`, `fam_77db01305f` [n=1], Bn) | 0.7010 | **0.9519** | No / No | Distinct singletons (`sd_train_504` vs `sd_test_156`). | Same Facebook security-alert template (`থেকে` / `এখনই` / `গিয়ে` minor phrasing); distinct lookalike domains (`fb-verification-bd…` vs `fb-secureverify24…`). Ex1: `“Facebook থেকে নিরাপত্তা সতর্কতা: আপনার একাউন্ট বাংলাদেশ থেকে…”` / Ex2: `“Facebook নিরাপত্তা সতর্কতা:…”` | **Ambiguous; requires human review** |
| 10 | `cur_b3194c2c485a` (`val`, `fam_d04705ccba` [n=1], Bn) | `cur_c00c6e3fa781` (`test`, `fam_b7af3c4ecc` [n=1], Bn) | 0.8101 | **0.8911** | No / No | Distinct singletons (`sd_validation_505` vs `sd_test_593`). | Same Banglalink bonus-suspended template (`অ্যাকাউন্ট স্থগিত আছে` vs `স্থগিত`); distinct domains (`bl-offerverify…` vs `bl-offerverify24…`). Ex1: `“বাংলালিংক থেকে সতর্কবার্তা: আপনার রিচার্জ বোনাস অ্যাকাউন্ট স্থগিত আছে…”` / Ex2: `“…আপনার রিচার্জ বোনাস স্থগিত…”` | **Ambiguous; requires human review** |
| 11 | `cur_292fdff7d278` (`real_pool` F2, `fam_baf93eb49b` [n=3], Bg) | `cur_c24e8fee19a2` (`test`, `fam_03f840170c` [n=4], Bg) | 0.7449 | **0.8560** | No / No | Distinct (`n=3` profile-update family vs `n=4` account-update family). | Near-identical template (`profile` vs `account ti`), distinct domains (`facebookprofile…` vs `facebookupdate…`). Ex1: `“apnar facebook profile update korun. ekhane click korun: [<URL>]”` / Ex2: `“Apnar Facebook account ti update korun…”` | **Ambiguous; requires human review** |
| 12 | `cur_63f82fd91d7a` (`real_pool` F2, `fam_ab0f20e0e6` [n=1], Bn) | `cur_da994897d0c3` (`real_pool` F3, `fam_2b6b4a2df2` [n=1], Bn) | 0.7526 | **0.8793** | No / No | Distinct singletons (`sd_train_2401` vs `sd_train_2525`). | Same DBBL fund-hold template, different amounts (`২,৫০,০০০` vs `২,০০,০০০`), domains (`dbbl-fundrelease24…` vs `dbbl-fundrelease…`), and CTA clause. Ex1: `“ডাচ্-বাংলা ব্যাংক সতর্কতা: আপনার অ্যাকাউন্টে ২,৫০,০০০ টাকার ফান্ড হোল্ড…”` / Ex2: `“…২,০০,০০০ টাকার ফান্ড হোল্ড… এটি মুক্ত করতে এখনই <URL> ভিজিট করুন।”` | **Ambiguous; requires human review** |
| 13 | `cur_be7f24b65438` (`real_pool` F0, `fam_976d8e1388` [n=1], Bn) | `cur_c9018d376c83` (`real_pool` F2, `fam_7ff22a7f78` [n=1], Bn) | 0.7721 | **0.8896** | No / No | Distinct singletons (`sd_train_1301` vs `sd_train_199`). | Same bKash mistaken-transfer template, different amounts (`১০,০০০` vs `৮,৫০০`), domains (`bkash-secureverify24…` vs `bkash-secureverify…`), and closing (`স্থগিত` vs `ফ্রিজ`). Ex1: `“বিকাশ একাউন্টে ১০,০০০ টাকা ভুলবশত পাঠানো হয়েছে…”` / Ex2: `“বিকাশ গ্রাহক, আপনার একাউন্টে ৮,৫০০ টাকা…”` | **Ambiguous; requires human review** |
| 14 | `cur_0fcaebdd6fd0` (`real_pool` F3, `fam_32d08e0523` [n=1], Bn) | `cur_3ecfbe652834` (`real_pool` F1, `fam_7b3c36fa90` [n=1], Bn) | 0.4081 | **0.9239** | No / No | Distinct singletons (`legacy_978`/`sd_train_1760` vs `legacy_980`/`sd_train_2103`). | Same EPL betting template, different prize amounts (`১ কোটি` vs `৫০,০০০` টাকা) and different URLs. Ex1: `“ইংলিশ প্রিমিয়ার লিগে বাজি ধরুন এবং ১ কোটি টাকা জিতুন…”` / Ex2: `“…৫০,০০০ টাকা জিতুন…”` | **Semantically related but distinct** |
| 15 | `cur_1d753e1cee76` (`real_pool` F4, `fam_8538df3808` [n=1], En) | `cur_309b6431fb13` (`real_pool` F1, `fam_a7ddf5a275` [n=1], En) | 0.4627 | **0.8725** | No / No | Distinct singletons (`sd_train_2859` vs `sd_train_1839`). | Same Euro Championship betting template, different amounts (`TK50,000` vs `TK 5,000`), verbs (`Click` vs `Start`), and URLs. Ex1: `“Bet on the European Championship and win TK50,000 cashback…”` / Ex2: `“…win TK 5,000 cashback…”` | **Semantically related but distinct** |
| 16 | `cur_1d753e1cee76` (`real_pool` F4, `fam_8538df3808` [n=1], En) | `cur_eb7d06582b65` (`test`, `fam_c025d51b1d` [n=2], En) | 0.4026 | **0.8644** | No / No | Distinct (`n=1` vs `n=2` bilingual Euro 20k family). | Same Euro Championship betting template, different amounts (`TK50,000` vs `20,000 TK`), verbs (`Click` vs `Join`), and URLs. Ex1: `“Bet on the European Championship and win TK50,000…”` / Ex2: `“Bet on European Championship and win 20,000 TK…”` | **Semantically related but distinct** |
| 17 | `cur_309b6431fb13` (`real_pool` F1, `fam_a7ddf5a275` [n=1], En) | `cur_eb7d06582b65` (`test`, `fam_c025d51b1d` [n=2], En) | 0.4652 | **0.8886** | No / No | Distinct (`n=1` vs `n=2` bilingual Euro 20k family). | Same Euro Championship betting template, different amounts (`TK 5,000` vs `20,000 TK`), verbs (`Start` vs `Join`), and URLs. Ex1: `“…win TK 5,000 cashback. Start: <URL>”` / Ex2: `“…win 20,000 TK cashback. Join: <URL>”` | **Semantically related but distinct** |
| 18 | `cur_2861b4182132` (`real_pool` F4, `fam_05dae3504f` [n=1], Bg) | `cur_e20fbf89b529` (`val`, `fam_9186ea6097` [n=1], Bg) | 0.5020 | **0.8664** | No / No | Distinct singletons (`sd_train_1630` vs `sd_validation_227`). | Same Banglish Euro betting template, different amounts (`5,000` vs `20,000`), conjunction/CTA (`ar … Shuru korun` vs `ebong … Jog din`), and URLs. Ex1: `“European championship e baji dhorun ar 5,000 taka…”` / Ex2: `“…ebong 20,000 taka…”` | **Semantically related but distinct** |
| 19 | `cur_2861b4182132` (`real_pool` F4, `fam_05dae3504f` [n=1], Bg) | `cur_ede29997f5d4` (`test`, `fam_d992d2b1c6` [n=1], Bg) | 0.5463 | **0.9464** | No / No | Distinct singletons (`sd_train_1630` vs `sd_test_1015`). | Same Banglish Euro betting template, different amounts (`5,000` vs `50,000`), CTA (`Shuru korun` vs `Click korun`), and URLs. Ex1: `“European championship e baji dhorun ar 5,000 taka…”` / Ex2: `“…ar 50,000 taka…”` | **Semantically related but distinct** |
| 20 | `cur_e20fbf89b529` (`val`, `fam_9186ea6097` [n=1], Bg) | `cur_ede29997f5d4` (`test`, `fam_d992d2b1c6` [n=1], Bg) | 0.4573 | **0.8828** | No / No | Distinct singletons (`sd_validation_227` vs `sd_test_1015`). | Same Banglish Euro betting template (`20,000` vs `50,000`, `Jog din` vs `Click korun`, different URLs). Ex1: `“…ebong 20,000 taka cashback jitun. Jog din: <URL>”` / Ex2: `“…ar 50,000 taka cashback jitun. Click korun: <URL>”` | **Semantically related but distinct** |
| 21 | `cur_42407c54319e` (`real_pool` F1, `fam_bf5ea8e23d` [n=1], Bn) | `cur_5b4b458c215c` (`real_pool` F2, `fam_533f4c0a04` [n=1], Bn) | 0.5643 | **0.9421** | No / No | Distinct singletons (`sd_train_3312` vs `sd_train_2055`). | Same Bengali Euro betting template (`২০,০০০` vs `৫,০০০`, `যোগ দিন` vs `শুরু করুন`, different URLs). Ex1: `“ইউরোপীয় চ্যাম্পিয়নশিপে বাজি ধরুন এবং ২০,০০০ টাকা…”` / Ex2: `“…৫,০০০ টাকা…”` | **Semantically related but distinct** |
| 22 | `cur_42407c54319e` (`real_pool` F1, `fam_bf5ea8e23d` [n=1], Bn) | `cur_4e0fc3118321` (`val`, `fam_358d797117` [n=1], Bn) | 0.5077 | **0.9392** | No / No | Distinct singletons (`sd_train_3312` vs `sd_validation_68`). | Same Bengali Euro betting template (`২০,০০০` vs `৫০,০০০`, `যোগ দিন` vs `ক্লিক করুন`, different URLs). Ex1: `“ইউরোপীয় চ্যাম্পিয়নশিপে… ২০,০০০ টাকা… যোগ দিন: <URL>”` / Ex2: `“…৫০,০০০ টাকা… ক্লিক করুন: <URL>”` | **Semantically related but distinct** |
| 23 | `cur_5b4b458c215c` (`real_pool` F2, `fam_533f4c0a04` [n=1], Bn) | `cur_4e0fc3118321` (`val`, `fam_358d797117` [n=1], Bn) | 0.5759 | **0.9563** | No / No | Distinct singletons (`sd_train_2055` vs `sd_validation_68`). | Same Bengali Euro betting template (`৫,০০০` vs `৫০,০০০`, `শুরু করুন` vs `ক্লিক করুন`, different URLs). Ex1: `“ইউরোপীয় চ্যাম্পিয়নশিপে… ৫,০০০ টাকা… শুরু করুন: <URL>”` / Ex2: `“…৫০,০০০ টাকা… ক্লিক করুন: <URL>”` | **Semantically related but distinct** |
| 24 | `cur_b70e4081fe21` (`real_pool` F3, `fam_8137a4bd83` [n=1], CM) | `cur_e5c24566c9d3` (`val`, `fam_be13e514ff` [n=1], CM) | 0.5321 | **0.9511** | No / No | Distinct singletons (`sd_train_823` vs `sd_validation_691`). | Same CodeMix Euro betting template (`৫০,০০০` vs `৫,০০০`, `Click করুন` vs `Start করুন`, different URLs). Ex1: `“European championship-এ bet ধরুন এবং ৫০,০০০ টাকা…”` / Ex2: `“…৫,০০০ টাকা…”` | **Semantically related but distinct** |
| 25 | `cur_b70e4081fe21` (`real_pool` F3, `fam_8137a4bd83` [n=1], CM) | `cur_b6041ad053cb` (`test`, `fam_c025d51b1d` [n=2], CM) | 0.4367 | **0.9252** | No / No | Distinct (`n=1` vs `n=2` bilingual Euro 20k family). | Same CodeMix Euro betting template (`৫০,০০০` vs `20,000`, `Click করুন` vs `Join`, different URLs). Ex1: `“European championship-এ bet ধরুন এবং ৫০,০০০ টাকা…”` / Ex2: `“…20,000 টাকা…”` | **Semantically related but distinct** |
| 26 | `cur_e5c24566c9d3` (`val`, `fam_be13e514ff` [n=1], CM) | `cur_b6041ad053cb` (`test`, `fam_c025d51b1d` [n=2], CM) | 0.4970 | **0.9147** | No / No | Distinct (`n=1` vs `n=2` bilingual Euro 20k family). | Same CodeMix Euro betting template (`৫,০০০` vs `20,000`, `Start করুন` vs `Join`, different URLs). Ex1: `“European championship-এ bet ধরুন এবং ৫,০০০ টাকা…”` / Ex2: `“…20,000 টাকা…”` | **Semantically related but distinct** |
| 27 | `cur_3a8ee05d8c8d` (`real_pool` F1, `fam_52d65f6f6a` [n=3], CM) | `cur_f0af08bf6ed0` (`test`, `fam_4bb67afb14` [n=2], CM) | 0.3629 | **0.9143** | No / No | Distinct (`n=3` casino car/iPhone/hotel vs `n=2` casino trip). | Same CodeMix casino template, different prize (`free car` vs `free trip`) and URLs. Ex1: `“Casino-তে bet করুন এবং একটি free car win করুন!…”` / Ex2: `“আজই casino-তে bet করুন এবং একটি free trip win করুন!…”` | **Semantically related but distinct** |
| 28 | `cur_3a8ee05d8c8d` (`real_pool` F1, `fam_52d65f6f6a` [n=3], CM) | `cur_f7cfb2326b54` (`test`, `fam_4bb67afb14` [n=2], CM) | 0.3534 | **0.8576** | No / No | Same as #27 (`test` sibling of #27). | Same CodeMix casino template, different prize (`free car` vs `৭ দিনের জন্য একটি free trip`) and URLs. Ex1: `“Casino-তে bet করুন এবং একটি free car…”` / Ex2: `“…৭ দিনের জন্য একটি free trip…”` | **Semantically related but distinct** |
| 29 | `cur_7b913c5a7646` (`real_pool` F1, `fam_52d65f6f6a` [n=3], CM) | `cur_f481de255c11` (`test`, `fam_3ae7bc14a0` [n=1], CM) | 0.4038 | **0.8670** | No / No | Distinct (`n=3` casino family vs singleton). | Same CodeMix casino iPhone template, different verb/CTA (`করুন … Start করুন` vs `ধরুন … Join করুন`) and URLs. Ex1: `“Casino-তে bet করুন এবং একটি free iPhone জিতুন! Start করুন: <URL>”` / Ex2: `“Casino-তে bet ধরুন… Join করুন: <URL>”` | **Semantically related but distinct** |
| 30 | `cur_54ccc4d7656f` (`real_pool` F3, `fam_1a97c102c8` [n=2], Bn) | `cur_3e1b3d2c7bee` (`test`, `fam_210d66c8f3` [n=3], Bn) | 0.3686 | **0.8599** | No / No | Distinct (`n=2` casino iPhone/car vs `n=3` casino trip). | Same Bengali casino template, different prize (`ফ্রি গাড়ি` vs `ফ্রি ট্রিপ`) and URLs. Ex1: `“ক্যাসিনোতে বাজি ধরুন এবং একটি ফ্রি গাড়ি জিতুন!…”` / Ex2: `“আজই ক্যাসিনোতে বাজি ধরুন এবং একটি ফ্রি ট্রিপ জিতুন!…”` | **Semantically related but distinct** |
| 31 | `cur_e9011c46d295` (`real_pool` F0, `fam_84a82c3269` [n=1], En) | `cur_9b7b07f54ebb` (`test`, `fam_e396320454` [n=2], En) | 0.3233 | **0.8554** | No / No | Distinct (`n=1` vs `n=2` football/BPL 5k/10k). | Same English sports-betting template (`IPL today` vs `BPL`), different URLs. Ex1: `“Bet on IPL today and win 10,000 taka cashback. Start: <URL>”` / Ex2: `“Bet on BPL and win 10,000 taka cashback. Start: <URL>”` | **Semantically related but distinct** |
| 32 | `cur_e4b6ade2454a` (`val`, `fam_0a130203db` [n=2], En) | `cur_a8aaa428cc24` (`test`, `fam_194ea98c59` [n=3], En) | 0.5800 | **0.8558** | No / No | Distinct (`n=2` debit-card family vs `n=3` credit-card family). | Different card product (`debit card` vs `credit card`) and domain (`debitcardlimit…` vs `creditcardincrease…`) on same template. Ex1: `“Click here to increase your debit card limit: [<URL>]”` / Ex2: `“Click here to increase your credit card limit: [<URL>]”` | **Semantically related but distinct** |

*(Language abbreviations: Bn = Bengali, Bg = Banglish, En = English, CM = CodeMix. Totals: **0 Confirmed duplicate**, **11 Ambiguous; requires human review**, **21 Semantically related but distinct**.)*

---

## 3. Exact Collision Audit Summary

- **[Verified fact]** Executing `.venv/bin/python scripts/audit_post_stage_b_model_views.py --bundle-dir data/processed/post_stage_b_bundle_v2` exited **1** with `checks: 560`, `failed_checks: 128`, `matching_row_pairs: 128`.
- **[Verified fact]** All 128 collisions occur exclusively in `brand_masked` (`64/140`) and `brand_stripped` (`64/140`), generated by three Smishing row pairs across six singleton families (`cur_43bd66f47b5b` / `cur_c92eaaa1966b` in CV folds 3–4 + counterfactual probes; `cur_29c93892c828` / `cur_7d47602782d3` in held-out validation; `cur_42a3cdc4a3d8` / `cur_cf8256b4cb6e` in held-out test).
- **[Verified fact]** From [`post_stage_b_model_view_collision_adjudication.md`](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/post_stage_b_model_view_collision_adjudication.md#L39-L57): all four pinned tokenizers (`BanglaBERT`, `MuRIL`, `XLM-R`, `DistilBERT`) reproduced the exact `128/560` hits, plus `24` supplementary adversarial CV token-level hits (on `BanglaBERT`, `MuRIL`, `DistilBERT` in `brand_masked` / `brand_stripped`, due to U+200C removal during tokenization). `raw` and `canon_masked` had **0/140** string hits each and **0/140** token hits each across all four tokenizers, plus **0** adversarial token hits.
- **[Unresolved issue]** Because HuggingFace tokenizers are not installed/cached in the local `.venv`, the tokenizer-level check could not be independently re-executed in this session; exact string and bundle checks were re-executed live.

---

## 4. Option A Feasibility Assessment (Task B)

- **[Verified fact]** **Read-only checks passed on `post_stage_b_bundle_v2`:**
  1. **A/B identical ordered real train IDs:** Verified `trA["row_id"].tolist() == trB["row_id"].tolist()` across all 5 folds (`3,199` rows in folds 0–3; `3,200` in fold 4). B weights verified: fold 0 `[0.883458, 1.416113, 0.860640]`, fold 1 `[0.882726, 1.416113, 0.861335]`, fold 2/3 `[0.883458, 1.414235, 0.861335]`, fold 4 `[0.883734, 1.414677, 0.860909]`.
  2. **C cohort membership:** Verified `3,199/3,200` real train + `345` Batches 1–7 (`3,544/3,545` total) across all 5 folds.
  3. **D cohort membership:** Verified `3,199/3,200` real train + `300` Pilot + `345` Batches 1–7 (`3,844/3,845` total) across all 5 folds.
  4. **Synthetic train-only invariant:** Verified `0` synthetic rows in validation (`800` in folds 0–3, `799` in fold 4), held-out validation (`708`), held-out test (`1,698`), OOD (`5,416`), or counterfactual probe (`1,279`).
  5. **Validation membership invariance:** Verified exact ordered `row_id` equality of `val` across A, B, C, and D in every fold.
  6. **Backbone support for `raw` and `canon_masked`:** Verified in [`post_stage_b_tokenization.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_tokenization.py#L10-L20) and [`post_stage_b_runner.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L385-L403) that all four backbones ingest `text_raw` and `text_canon_masked`, applying `normalizer.normalize` only to `csebuetnlp/banglabert`. Additionally, `canon_masked` (`schema.py:538`) uses only contact/USSD masking + NFC and has **zero dependency on the Stage B brand registry**.
  7. **Zero string-level collisions in scope:** `audit_model_view_overlaps(b2, views=("raw", "canon_masked"))` returned `checks: 280, failed_checks: 0, matching_row_pairs: 0`; adversarial CV string audit returned `0/40`; mutual held-out string audit returned `0/6`.
  8. **Independent unit & comparison arithmetic:** `4 experiments × 4 backbones × 2 views × 5 folds = 160` training/evaluation units; `(4 - 1) × 4 × 2 = 24` planned against-A comparisons.

- **[Verified fact]** **Remaining nuances & edge cases that affect Option A:**
  1. **Counterfactual probe behavior under `canon_masked`:** On `1,261 / 1,279` (`98.59%`) counterfactual rows, `orig_text_canon_masked == contact_swap_text_canon_masked` (because `<URL>`/`<PHONE>` mask both original and swapped contacts; the remaining `18` rows differ only by a trailing period consumed by `_SCHEMA_URL_RE` during raw substitution, e.g., `agrani-lottery-claim.net.` → `<URL>` vs `<URL>.`). Thus `contact_swap` is virtually a no-op under `canon_masked`, while `contact_brand_swap` differs from `contact_swap` on the `238 / 1,279` rows (`18.61%`) that contain recognized `REAL_BD_BRANDS` tokens (`PurbachalBank` substitution).
  2. **Upstream template leakage (Task A):** Dropping `brand_masked`/`brand_stripped` eliminates exact model-input collisions, but **11 ambiguous template/URL pairs** and **21 template-slot pairs** from `smishdetect_2026`/`bangalabarta_2025` remain across CV folds and held-out splits. None come from Stage B synthetic data.
  3. **Equal-epoch step imbalance:** Under `batch_size=32`, 3 epochs give A/B `300` updates, C `333` (`+11%`), and D `363` (`+21%`).
  4. **Hardcoded test guard:** [`test_configurations_keep_full_design_and_smoke_caps`](file:///home/bs01415/Documents/Programming/ExploringAIs/tests/test_post_stage_b_pipeline.py#L209) currently asserts `320` total units on `post_stage_b_abcd_config.json`; any future Option A adoption requires updating that test alongside a versioned Option A config/guard.
  5. **Scientific trade-off:** Option A cannot quantify pure brand-shortcut removal (`<BRAND>` vs stripping); that remains deferred unless a collision-safe follow-up (e.g., Option C) is separately authorized.

---

## 5. Proposed Statistical Protocol (Task C — Proposal Only)

### 5.1 Current Code Gaps (Observed, Not Modified)

- **[Verified fact]** In [`post_stage_b_runner.py`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L514-L569), each fold unit sets `threshold` empirically on its single CV validation fold (`n_normal = 301` or `302`, so `allowed_fp = floor(0.01 * 301) = 3`) and transfers it to held-out validation, test, and OOD per fold.
- **[Verified fact]** In [`_aggregate_verified`](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_runner.py#L1054-L1120), `aggregated_results.json`/`csv` do **not** compute 5-fold ensemble (probability-averaged) held-out/OOD predictions, do **not** run paired tests on held-out/OOD or on `smishing_recall_at_1pct_fpr`, and run only `mcnemar_pvalue` (exploratory row-level argmax accuracy) and `paired_family_accuracy_delta` on pooled 5-fold OOF CV predictions with a single seed (`42 + fold`).

### 5.2 Proposed Defensible Statistical Analysis Plan

1. **Primary Research Question:** *Holding model architecture, text representation (`canon_masked` primary; `raw` reference), training epoch schedule, and evaluation sets fixed, does adding Stage B quality-controlled synthetic Promotional SMS (`C`: Batches 1–7; `D`: Pilot + Batches 1–7) improve real-world Bangla Smishing detection and reduce Promotional-vs-Smishing / Normal-vs-Smishing confusion relative to the unbalanced real baseline (`A`) and class-weighted real baseline (`B`)?*
2. **Primary Hypotheses ($H_1$):**
   - $H_{1a}$ (Data augmentation vs. unbalanced real): On `canon_masked`, `C` (clean Batches 1–7) achieves higher held-out `smishdetect_test` Smishing Recall at a frozen `<= 1%` Normal FPR threshold (and higher Macro-F1) than `A` without exceeding the `1%` Normal FPR cap or degrading counterfactual robustness.
   - $H_{1b}$ (Data augmentation vs. class weighting): `C` matches or outperforms `B` (class-weighted real-only) on held-out `smishdetect_test` Macro-F1 and Promotional-vs-Smishing error rates.
   - $H_{1c}$ (Clean batches vs. full Stage B pool): `C` vs `D` evaluates whether adding the earlier `pilot_300` cohort helps or introduces synthetic-origin artifacts compared to `B1–B7` alone.
3. **Primary Endpoint (Recommendation):**
   - **[Recommendation]** **Two-part co-primary or hierarchical primary endpoint on `canon_masked`:**
     - **Primary Endpoint:** **Macro-F1 (%)** on `smishdetect_test` (using 5-fold probability-averaged ensemble predictions per `(experiment, model, view, seed)`), paired with **Smishing Recall (%) at a development-set frozen threshold targeting `<= 1%` Normal FPR** as the deployment-critical operating-point co-primary endpoint.
     - **Why not `recall_at_1pct_fpr` alone as sole primary?** Two verified statistical issues: (a) a single CV fold has only `301–302` Normal rows (`allowed_fp = 3`), making the empirical 99th-percentile Normal score high-variance; and (b) under distribution shift (especially `BanglaPhish` OOD), a frozen in-distribution threshold often violates `actual_fpr <= 1%` (`constraint_met = False` sets `smishing_recall_at_1pct_fpr = None` in `metrics.py:258`), causing missing values in paired contrasts.
     - **Frozen-Threshold Protocol:** Select the operating threshold **strictly on development data** — specifically, compute the `<= 1%` Normal FPR threshold on the **pooled 5-fold OOF predictions** (`n_normal = 1,509`, `allowed_fp = 15`, or alternatively the median/ensemble threshold validated on `smishdetect_validation` if `smishdetect_validation` is predeclared as the dev calibration set; **never** tune on `smishdetect_test` or `banglaphish_ood`) — and apply that **exact float64 threshold unchanged** to `smishdetect_test`, `banglaphish_ood`, and `val_adv_zwj_homoglyph`. Always report `(threshold, actual_fpr, false_positives, n_normal, recall_at_fpr, constraint_met)` even when `constraint_met` is `False`.
4. **Secondary Evaluation Metrics:** Argmax Smishing Recall, Smishing F1, Normal-as-Smishing FPR (argmax & frozen-threshold), Promotional-as-Smishing FPR (argmax & frozen-threshold), Overall Accuracy, ECE (10-bin calibration), ROC-AUC & PR-AUC (threshold-free Smishing-vs-Rest), and per-language (`Bengali`, `Banglish`, `CodeMix`, `English`) Macro-F1.
5. **Paired Comparisons Across A/B/C/D:**
   - Predeclare two contrast families (per primary/reference view):
     - **Primary confirmatory family (`canon_masked`, 4 backbones):** `{B vs A, C vs A, D vs A}` plus `{C vs B, D vs C}` = `12` against-A comparisons (or `20` if including `C vs B` and `D vs C`). Across both `canon_masked` and `raw`, against-A comparisons = **24**.
   - **Test statistic for paired deltas:** Because `smishdetect_test` (`1,698` rows, `1,304` families) and `banglaphish_ood` (`5,416` rows, `4,335` families) have multi-row families, use **paired family-cluster bootstrap (`B >= 2,000` resamples over `family_id`)** for all paired metric deltas ($\Delta\text{Macro-F1}$, $\Delta\text{Recall@1\%FPR}$, $\Delta\text{Accuracy}$, $\Delta\text{Promo}\to\text{Smish}$), yielding percentile 95% CIs and two-sided bootstrap/permutation p-values respecting family correlation. Retain row-level `mcnemar_pvalue` strictly as *exploratory* (`post_stage_b_runner.py:1112`).
6. **Multiple Random Seeds & Reproducibility:**
   - **[Unresolved issue]** Current config specifies 1 seed (`seed = 42`, shifted by `+ fold`). Single-seed runs conflate model initialization variance with treatment effects.
   - **[Recommendation]** For confirmatory claims, run **at least 3 independent seed replicates** (`seeds = (42, 1042, 2042)`, i.e., `160 × 3 = 480` units if budget permits; or first screen on 160 units with `seed=42`, then replicate top/primary comparisons on 3 seeds — **pending Colab compute budget verification**). Never cherry-pick seeds; report seed-mean ± seed-SD alongside family-bootstrap CIs.
7. **Family-Aware Bootstrap CIs:** Resample whole `family_id` clusters (`metrics.py:107–145`) with `n_boot >= 2,000` for final reporting (current runner uses `1,000` on OOF, `500` on per-unit heldout). Note: for `eval_banglaphish_ood` (`4,335` families across `5,416` rows), family clustering is already supported via `family_id`.
8. **Multiple-Comparison Correction:** Apply **Holm–Bonferroni step-down correction** (`holm_adjust_pvalues`) separately within the predeclared confirmatory primary family (`canon_masked`, 12 vs-A contrasts, or 24 across both views) before inspecting secondary contrasts.
9. **Smishing Recall & Normal FPR:** Report both argmax Normal→Smishing FPR and frozen-threshold Normal→Smishing FPR (`false_positives / n_normal`), alongside Smishing Recall at both operating points.
10. **Promotional & Normal Misclassification Analysis:** Explicitly report the full 3×3 confusion matrix and separate **Promotional→Smishing FPR** (`promotional_as_smishing_fpr_argmax` and at frozen threshold) and **Normal→Promotional FPR** across languages (`Bengali`, `Banglish`, `CodeMix`, `English`). This directly tests whether Stage B Promotional augmentation stops models from conflating benign Promotional offers with Smishing.
11. **Evaluation Separation (Strict Anti-Leakage Boundary):**

| Evaluation Regime | Purpose & Allowed Decisions | Prohibited Uses |
|---|---|---|
| **5-Fold Cross-Validation (`real_train_pool`, 3,999)** | Epoch selection (`eval_loss`), architecture/view diagnostics, frozen `<=1%` FPR threshold calibration, within-pool A/B/C/D learning curves | Do NOT claim OOD or held-out generalization solely from CV; note C/D have +11%/+21% optimizer steps |
| **Held-Out In-Distribution (`smishdetect_validation` 708, `smishdetect_test` 1,698)** | `validation`: optional independent threshold verification / model selection if predeclared; `test`: **final confirmatory ID benchmark** (5-fold probability-averaged predictions) | **Never** use `smishdetect_test` for epoch stopping, hyperparameter tuning, or threshold selection |
| **Out-of-Distribution (`banglaphish_ood`, 5,416)** | **Zero-shot OOD generalization** under synthetic-source shift; report overall + `SMS` vs `other` subsets | **Never** select checkpoints, views, or thresholds from OOD |
| **Counterfactual & Adversarial (`counterfactual_smish_probe` 1,279; `val_adv_zwj_homoglyph`)** | **Shortcut & surface robustness** (`cf_contact_brand_swap_pp_delta`, ZWJ/homoglyph drop). In `canon_masked`, `contact_brand_swap` isolates brand-name reliance on 238 brand-bearing rows | Cannot estimate false positives (Smishing-only probes) or standalone classification accuracy |

---

## 6. Colab GPU Readiness Audit (Task D)

### 6.1 Configuration & Hardware Profile

- **[Verified fact]** **Available hardware:** Google Colab (~200 monthly compute units), target GPU NVIDIA Tesla T4 (15 GB / 16 GB nominal VRAM, Turing architecture, **no native BF16**, requires FP16); local machine AMD Ryzen 5 7530U, 32 GB RAM, integrated Radeon GPU (CPU-only for PyTorch/Colab prep).
- **[Verified fact]** **Full config (`experiments/colab/configs/post_stage_b_abcd_config.json`):** `epochs=3`, `batch_size=32`, `eval_batch_size=64`, `max_length=128`, `learning_rate=3e-5`, `weight_decay=0.01`, `warmup_ratio=0.1`, `fp16="auto"`, `keep_model_weights="none"`. **No `gradient_accumulation_steps` field exists** in `PostStageBConfig` (defaults to `1` in `TrainingArguments`).
- **[Verified fact]** **Precision dispatch:** `_resolve_precision("auto")` checks `torch.cuda.is_bf16_supported(including_emulation=False)`, correctly forcing **FP16** on T4 (`post_stage_b_runner.py:351–356`).

### 6.2 Per-Backbone Sizing, Steps, Storage, & Risk Table

| Backbone | Approx Params | FP16 AdamW Static VRAM (18 B/param) | Est. Peak T4 VRAM (`bs=32, len=128`) | Steps / Unit (A·B / C / D) | Est. 1 Epoch Ckpt Size (12 B/param) | Storage for 40 Units × 3 Epochs (Option A) | T4 Runtime / CU Estimate |
|---|---:|---:|---:|---|---:|---:|---|
| `csebuetnlp/banglabert` | ~110 M | ~1.98 GB | ~4–6 GB *(Inferred)* | 300 / 333 / 363 | ~1.32 GB | ~158 GB | **UNVERIFIED** (needs v2 smoke) |
| `distilbert-base-multilingual-cased` | ~135 M | ~2.43 GB | ~4–6 GB *(Inferred)* | 300 / 333 / 363 | ~1.62 GB | ~194 GB | **UNVERIFIED** (old 64-row smoke only) |
| `google/muril-base-cased` | ~237 M | ~4.27 GB | ~6–9 GB *(Inferred)* | 300 / 333 / 363 | ~2.84 GB | ~341 GB | **UNVERIFIED** (needs v2 smoke) |
| `xlm-roberta-base` | ~278 M | ~5.00 GB | ~7–10 GB *(Inferred)* | 300 / 333 / 363 | ~3.34 GB | ~401 GB | **UNVERIFIED** (highest OOM risk) |
| **Total (Option A: 160 units)** | — | — | **Fits 15 GB T4 *(Inferred)*** | **51,840 train steps** | — | **~1.09 TB** | **Cannot estimate reliably without v2 smoke** |

- **[Verified fact]** **CRITICAL STORAGE & DRIVE I/O BLOCKER:** `hf_train_and_predict_fold` (`post_stage_b_runner.py:419–423`) configures `save_strategy="epoch"` with **no `save_total_limit`** and writes checkpoints directly to `artifacts_dir / "checkpoints" / uid / attempt` (on Google Drive `/content/drive/MyDrive/...`). Furthermore, `_read_committed_unit` (`lines 814–818`) **requires** `trainer_checkpoint` to exist and pass SHA256 verification whenever committed units are resumed or aggregated! Even under Option A (160 units instead of 320), 480 retained epoch checkpoints consume **~1.09 TB** on Google Drive, exceeding typical Drive quotas (15 GB free / 100–200 GB standard paid tiers) and causing heavy FUSE write latency (~4–10 GB/unit).
- **[Verified fact]** **Per-unit evaluation cost:** Every unit evaluates `val` (800) × 3 epochs + final `trainer.predict` over `8` slices (`val` 800 + `val_adv` 800 + `sd_val` 708 + `sd_test` 1,698 + `bp_ood` 5,416 + 3× `cf_*` ~768 ≈ `10,190` rows), re-tokenizing all splits from scratch on CPU per unit (`~1.63M` row tokenizations across 160 units).
- **[Inference]** **Batch size & memory:** With `max_length=128` and FP16, all four backbones should fit in 15 GB T4 VRAM at `batch_size=32, eval_batch_size=64`, with `xlm-roberta-base` closest to the ceiling (~7–10 GB). If `xlm-roberta-base` or `google/muril-base-cased` hits OOM or allocator fragmentation, **`batch_size=16` + `gradient_accumulation_steps=2`** is required to preserve effective batch size 32 — which **currently requires adding `gradient_accumulation_steps` to `PostStageBConfig` and `TrainingArguments`**.
- **[Recommendation]** **Realistic Execution Plan for Option A (160 Units):**
  1. **Fix checkpoint retention before full run (separate authorized PR):** Write Trainer checkpoints to fast local `/content/tmp_ckpt/<uid>` during training (or add `save_total_limit=1` / prune non-committed epoch checkpoints after a unit successfully commits `preds.npz` + `metrics.json`, updating `_read_committed_unit` to not require retaining 3 GB optimizer state once `preds.npz` is committed). Keep only `preds.npz`, `metrics.json`, `run_state.json`, and `exports/` on Google Drive (~1–2 GB total instead of ~1.09 TB!).
  2. **Stage execution by backbone (40 units/backbone):** Run 1 backbone at a time (`BanglaBERT` → `DistilBERT` → `MuRIL` → `XLM-R`), allowing clean monitoring of Colab compute-unit burn rate (~200 CU budget) and session timeouts.
  3. **Measure exact T4 timings first:** Run an expanded capped v2 smoke test covering all 4 backbones on 1 fold (`4 units` or full-length 1-unit benchmark) to record exact `train_seconds`, `peak_cuda_allocated_bytes`, Drive write time, and CU/hour burn rate before committing to 160 (or 480 multi-seed) units.

---

## 7. Integrity and Regression Audit (Task E)

### 7.1 Live Verification Results (Executed 2026-11-23)

| Check | Command / Scope | Observed Result |
|---|---|---|
| Git tree state | `git status --short` / `git log -1 --oneline` | Clean tree on `main` at `4d34118` |
| Protected 13 inputs | `verify_protected_inputs(ROOT, strict_digests=True)` | **13/13 matched** (`eligible_train`, `existing_unique`, `provenance`, `quarantined`, `banglaphish2026_dedup`, Pilot `accepted.jsonl`, Batches 1–7 `accepted.jsonl`) |
| Legacy `DataSet.csv` | `sha256_file("DataSet.csv")` | `5df79924b5966513760ad6ec38911cb12a1233a569b1a645b83b5b0936eabf27` (unchanged) |
| Bundle v1 verification | `.venv/bin/python scripts/prepare_post_stage_b_bundle.py --verify-only` | **Exit 0** (`2e2e76e2...`; 15 cohort pairs, 5 folds, 79 historical v1 norm mismatches documented) |
| Bundle v2 verification | `.venv/bin/python scripts/prepare_post_stage_b_bundle.py --verify-only --bundle-dir data/processed/post_stage_b_bundle_v2` | **Exit 0** (`308d1c88...`; 0 norm mismatches, 4,110 SmishDetect source links, 1,279 CF rows) |
| BanglaPhish raw checksums | `cd data/raw/banglaphish2026 && sha256sum -c SHA256SUMS.txt` | **Exit 1** (22/23 OK; only `README.md` differs, all CSV/XLSX/JSONL/split payloads `OK`) |
| Model-view audit (all 4 views) | `.venv/bin/python scripts/audit_post_stage_b_model_views.py --bundle-dir data/processed/post_stage_b_bundle_v2` | **Exit 1** (`128/560` failures, strictly in `brand_masked` & `brand_stripped`) |
| Model-view audit (Option A scope) | `audit_model_view_overlaps(b2, views=("raw", "canon_masked"))` + adv/mutual checks | **0/280** clean/CF hits, **0/40** adv CV hits, **0/6** mutual held-out hits |
| Full test suite | `.venv/bin/python -m unittest discover -s tests -v` | **Exit 0** (**63/63 passed** in 107.557s, 0 skips) |
| Pipeline sanity suite | `.venv/bin/python tests/test_pipeline.py` | **Exit 0** (`All tests passed!`) |

### 7.2 Code & Notebook Audit Findings

- **[Verified fact]** **Resume identity & prediction integrity:** `_run_identity` fingerprints `config_hash`, `dataset_manifest.json` SHA256, runner qualname, and 7 source module SHA256s; `_read_prediction_archive` validates exact ordered `row_id`, `label_idx`, and probability simplex invariants; `_validate_unit` validates metrics and checkpoint hashes.
- **[Verified fact]** **Colab notebook (`experiments/colab/post_stage_b_experiments.ipynb`):** 8 cells match `scripts/generate_post_stage_b_notebook.py`; Cell 1 enforces `RUN_MODE='gpu_smoke_test'` and `AUTHORIZE_FULL_TRAINING=False` by default; Cell 3 mounts Drive to `/content/drive/MyDrive/bangla_smish_post_stage_b_v2`; Cell 4 checks `torch >= 2.6` and pins `normalizer` commit `d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9` for BanglaBERT.
- **[Unresolved issue]** **Drive FUSE locking & checkpoint persistence:** `_run_lock` (`post_stage_b_runner.py:609–619`) uses `fcntl.flock` on `.runner.lock` inside `artifacts_dir` (on Drive FUSE in Colab); POSIX `flock` behavior and checkpoint storage capacity on mounted Google Drive remain unverified on live Colab.

---

## 8. Regression Test Coverage Assessment

- **[Verified fact]** **Existing 63 tests:** Cover curation v1/v2 (`test_curation`, `test_curation_v2`), CodeMix/brand diversity (`test_codemix_diversity`), regex/brand masking (`test_schema_masking`), Stage B pilot gates (`test_stage_b_pilot`), low-FPR/bootstrap/Holm metrics (`test_post_stage_b_metrics`, 6 tests), BanglaBERT normalization & notebook schema/launch guards (`test_post_stage_b_preprocessing`, 6 tests), bundle integrity/splits/CF/full-collision abort (`test_post_stage_b_pipeline`, 10 tests), and runner resume/corruption/T4-FP16/weighted-loss (`test_post_stage_b_runner`, 11 tests).
- **[Unresolved issue]** **Missing Critical Regression Tests (Must Be Added Only When Authorized):**
  1. **Option A scope enforcement test:** Assert that confirmatory `experiments_abcd` configs contain only `("raw", "canon_masked")`, reject `brand_masked`/`brand_stripped`, and update the hardcoded `320` check in `test_configurations_keep_full_design_and_smoke_caps` (`test_post_stage_b_pipeline.py:209`).
  2. **5-fold ensemble & held-out/OOD paired-test regression:** Test 5-fold probability-averaged ensemble scoring, pooled-OOF frozen-threshold transfer to `smishdetect_test` / `banglaphish_ood`, and family-clustered paired delta tests on held-out sets.
  3. **Gradient accumulation + weighted-loss parity test:** Verify that if `gradient_accumulation_steps > 1` is added, weighted CE loss normalization across micro-batches matches single-batch weighted CE (`_weighted_cross_entropy`).
  4. **Checkpoint pruning / Drive-safe storage test:** Verify that once a unit commits `preds.npz` + `metrics.json`, pruning/excluding multi-GB optimizer checkpoints does not break `resume=True` or `aggregate_post_stage_b_results`.
  5. **Adversarial & CF `canon_masked` invariance guard:** Assert `0` collisions on `text_adv_raw`/`text_adv_canon_masked` and document/verify the `18` trailing-period `contact_swap` diffs in `counterfactual_smish_probe`.

---

## 9. Remaining Blockers

1. **Owner policy decision on brand views & Option A** (must approve dropping `brand_masked`/`brand_stripped` from confirmatory grid or specify an alternative like Option C).
2. **Owner adjudication of the 11 ambiguous near-duplicate / URL-template pairs** from Task A (decide whether to document as upstream `SmishDetect` template noise or create a sensitivity/exclusion list — keeping frozen splits untouched unless explicitly revised).
3. **Statistical protocol sign-off** (Task C: 5-fold ensemble vs fold-mean reporting on held-out/OOD, pooled OOF vs fold-wise threshold, multi-seed budget vs single-seed 42, primary endpoint & Holm family).
4. **Google Drive checkpoint storage (~1.09 TB) & `gradient_accumulation_steps` support** (must resolve `save_total_limit` / post-commit checkpoint retention before any full 160-unit run).
5. **Empirically unverified T4 CUDA + Drive FUSE execution** across all 4 backbones (`BanglaBERT`, `MuRIL`, `XLM-R`, `DistilBERT`).

---

## 10. GO / NO-GO Readiness Matrix

| Gate / Phase | Status | Key Evidence / Condition |
|---|---|---|
| Stage B Synthetic Corpus (`4,644` frozen) | **GO (CLOSED)** | All 13 protected SHA256s & v2 bundle verified; 0 class/language deficits |
| Stage 3 Model-View & Near-Dup Audit | **COMPLETE (PENDING OWNER REVIEW)** | 128/560 exact collisions isolated to brand views; 32 heuristic pairs adjudicated |
| Proposed Option A Policy (`raw`, `canon_masked`) | **READY FOR OWNER APPROVAL** | 0/280 exact collisions; A/B/C/D & validation invariants verified; 5 caveats documented |
| Capped v2 Colab Smoke Test (current DistilBERT `gpu_smoke_test`) | **CONDITIONAL GO** | Notebook & v2 bundle pass local checks; only tests DistilBERT `raw` 64/32 capped |
| Capped v2 4-Backbone T4 Benchmark (recommended pre-full) | **RECOMMENDED BEFORE FULL** | Needed to measure `XLM-R` / `MuRIL` peak VRAM, epoch time, CU burn, and Drive FUSE |
| **Full Experiments A/B/C/D (160 or 320 units)** | **NO-GO** | Blocked by owner approvals (Option A, Task A/C) + ~1.09 TB checkpoint retention issue |

---

## 11. Recommended Next Steps

1. **Owner reviews this report** and signs off on: (a) **Option A** (`canon_masked` primary, `raw` reference; exclude `brand_masked`/`brand_stripped`), (b) keeping frozen splits unchanged while noting the 11 ambiguous upstream `SmishDetect` template pairs for a post-hoc sensitivity table, and (c) the **Task C statistical protocol** (5-fold ensemble on held-out/OOD, pooled-OOF threshold transfer, multi-seed plan).
2. **Authorize a targeted pre-full engineering PR** to: (a) configure Option A (`160` units) + add `gradient_accumulation_steps`, (b) fix Drive checkpoint retention (~1.09 TB → ~2 GB after unit commit), (c) implement 5-fold ensemble + family-clustered held-out paired tests in `_aggregate_verified`, and (d) add the 5 missing regression tests.
3. **Run an authorized capped v2 Colab T4 smoke test** (including `XLM-R` and `MuRIL` peak VRAM + CU/hr measurement) on a fresh `ARTIFACT_RUN_ID`.
4. **Only after smoke metrics and CU budget are confirmed**, authorize Full Experiments A/B/C/D.

---

## 12. Exact Commands for the Next Authorized Phase

*(Read-only verification & local Drive-kit staging; do NOT execute until authorized.)*

```bash
# 1. Re-verify clean tree, tests, and v2 bundle locally
git status --short
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tests/test_pipeline.py
.venv/bin/python scripts/prepare_post_stage_b_bundle.py --verify-only --bundle-dir data/processed/post_stage_b_bundle_v2

# 2. Check Option A exact collision scope (once authorized to test Option A or run read-only check)
.venv/bin/python -c '
import sys; sys.path.insert(0, "src")
from pathlib import Path
from bangla_smish.post_stage_b_model_view_audit import audit_model_view_overlaps
res = audit_model_view_overlaps(Path("data/processed/post_stage_b_bundle_v2"), views=("raw", "canon_masked"))
assert res["failed_checks"] == 0, res
print("Option A exact string audit passed:", res)
'

# 3. Stage a fresh upload-ready Google Drive kit for an authorized capped v2 Colab smoke test
.venv/bin/python scripts/prepare_post_stage_b_bundle.py \
  --verify-only \
  --bundle-dir data/processed/post_stage_b_bundle_v2 \
  --export-drive-kit /tmp/exploringais-post-stage-b-v2-owner-kit
```

---

## 13. Final Decisions

### A. Is Stage 3 complete?
**YES (technical audit complete; awaiting owner disposition).** All exact model-view collisions (`128/560`), supplementary adversarial token collisions (`24` across 3 tokenizers), and all `32` cross-boundary near-duplicate candidates (`norm >= 0.85` and `slot >= 0.85`) have been identified, traced to provenance, and adjudicated. Stage 3 closes operationally once the owner records disposition of the 11 ambiguous upstream `SmishDetect` pairs and signs off on Option A.

### B. Is the proposed Option A policy technically safe and ready for owner approval?
**YES, with documented caveats.** Option A (`canon_masked` primary, `raw` reference; excluding `brand_masked`/`brand_stripped`) achieves `0/280` exact model-input collisions across all 4 arms × 5 folds × 7 evaluation surfaces, `0/40` adversarial CV collisions, and `0/6` mutual held-out collisions, while preserving all frozen memberships, A/B training parity, and `canon_masked`'s independence from the Stage B brand registry. Outstanding conditions for owner sign-off: (1) accept that brand-removal ablation is excluded from confirmatory scope, (2) note that `contact_swap` is a `98.59%` no-op in `canon_masked` (leaving `contact_brand_swap` on 238 rows as the active counterfactual test), and (3) accept/document the 11 ambiguous upstream `SmishDetect` template pairs.

### C. Is the project ready for a capped v2 Colab smoke test?
**CONDITIONAL GO (ready for the existing DistilBERT `gpu_smoke_test`, NOT yet sufficient to clear full 160-unit training).** The v2 bundle (`308d1c88...`), 8-cell notebook, and 63/63 local regressions are verified and ready to stage to a fresh Drive workspace for `RUN_MODE='gpu_smoke_test'` (`AUTHORIZE_FULL_TRAINING=False`, fresh `ARTIFACT_RUN_ID`). However, because `gpu_smoke_test_config.json` exercises only `distilbert-base-multilingual-cased` on `raw` with 64/32 capped rows, a single DistilBERT smoke run will **not** measure `xlm-roberta-base` / `google/muril-base-cased` peak VRAM, Drive checkpoint storage (~1.09 TB risk for full runs), or 200-CU budget feasibility. **Full Experiments A/B/C/D remain NO-GO.**
