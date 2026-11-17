# Dataset Consolidation Report — Stage A (`curation_v1`)

Generated reproducibly by `scripts/curate_stage_a.py` (`src/bangla_smish/curation.py`) from existing datasets only (no synthetic generation, undersampling, or oversampling).

## 1. Logical Source Inventory (Before Cleaning)

Each logical dataset is counted once; duplicate exports (`bangalabarta/*.zip`, `banglaphish2026/*.jsonl`, `*.xlsx`, `data/splits/*`) and derived `data/processed/*.parquet` files are excluded from raw source row counts.

| Logical Dataset | Canonical Source File(s) | License | Source Type | Raw Rows | Unique Strict Texts | Retained Primary Rows in `existing_unique` |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| `legacy_2024` | `DataSet.csv` (`5df79924b5...`) | UNDOCUMENTED | unknown | 2580 | 2083 | 0 |
| `bangalabarta_2025` | `data/raw/bangalabarta/BangalaBarta bangla_spam_sms smishing.csv` (`260c2e3bd1...`) | CC BY-NC-SA 4.0 | curated_real_mixed | 2772 | 1205 | 0 |
| `smishdetect_2026` | `data/raw/smishdetect_hf/data/train-00000-of-00001.parquet` (`9fb92ea7e1...`), `data/raw/smishdetect_hf/data/validation-00000-of-00001.parquet` (`a6a652087b...`), `data/raw/smishdetect_hf/data/test-00000-of-00001.parquet` (`2719d61aaf...`) | MIT | augmented_multilingual | 7005 | 6996 | 6531 |
| `banglaphish_2026` | `data/raw/banglaphish2026/data/BanglaPhish-2026_v1.0_MASTER.csv` (`4d8dd5d2c3...`) | CC BY-NC 4.0 | synthetic | 6000 | 5416 | 5416 |
| **Total** | — | — | — | **18357** | — | **11947** |

- **Cross-corpus exact lineage:** `1205` / `1205` unique strict `bangalabarta_2025` texts appear inside `smishdetect_2026` (100.0%). `1126` unique strict `legacy_2024` texts match `bangalabarta_2025` / `smishdetect_2026`. `banglaphish_2026` has `0` exact overlaps with the other three corpora.

## 2. Normalization, Exact Deduplication & Row Reconciliation

- **Normalization rules:** Original `text` is preserved verbatim. Exact deduplication uses `text_norm_strict` (stripping Unicode `Cf` format chars such as ZWJ/ZWNJ/BOM and non-whitespace `Cc` chars, normalizing Unicode `Z*`/newline whitespace to ASCII space, applying `NFKC`, collapsing whitespace, and `.casefold()`), alongside repo `normalize_text()` (`text_norm`).
- **Provenance sidecar:** Every raw logical row (`18357` rows) is preserved in `provenance.parquet` with source file, row index, original ID, original split, original label, language/script tags, disposition, and reason.

| Provenance Disposition | Raw Row Count | % of Raw Total | Notes |
| :--- | ---: | ---: | :--- |
| `canonical_kept` | 11947 | 65.08% | Canonical rows in `existing_unique.parquet` |
| `exact_dup_merged` | 4906 | 26.73% | Exact `text_norm_strict` duplicates merged into canonical/surviving provenance |
| `near_dup_variant_collapsed` | 454 | 2.47% | Reviewed cosmetic/slot-swap variants collapsed in `BB+SD` |
| `quarantined` | 1050 | 5.72% | Conflicting, unusable, or legacy reference-only rows in `quarantined.parquet` |
| **Total Reconciled** | **18357** | **100.00%** | Matches raw source inventory `18357` |

## 3. Near-Duplicate & Template Family Review

- **Clustering method:** Connected components over the union of (a) `text_norm_strict` char-ngram TF-IDF (3–5 grams, cosine ≥ 0.85) and (b) shortcut-and-digit-normalized template text (`<URL>`, `<PHONE>`, `<USSD>`, `<N>`, cosine ≥ 0.85), assigning deterministic `family_id` across all `12401` clean exact-unique texts BEFORE collapsing variants.
- **Total multi-member families reviewed:** **1363** families logged in `near_duplicate_review.jsonl`.
- **Total families across `existing_unique.parquet`:** **10163**.
- **Variants collapsed:** **454** (all in `BB+SD`; `BanglaPhish-2026` OOD benchmark rows are preserved intact with `family_id` assigned):
  - `cosmetic_punctuation_spacing_or_markdown_variant`: **44** variants collapsed with explicit surviving target IDs logged.
  - `numeric_or_shortcut_slot_clone`: **410** variants collapsed with explicit surviving target IDs logged.

| Review Category (`near_duplicate_review.jsonl`) | Families |
| :--- | ---: |
| `cosmetic_or_slot_collapse_only` | 124 |
| `distinct_variants_kept_only` | 316 |
| `mixed_collapse_and_distinct_kept` | 112 |
| `ood_benchmark_kept_with_family_id` | 811 |

Materially distinct near-duplicates (e.g., cross-script translations across `Bengali`/`Banglish`/`English`/`CodeMix`, paraphrases with distinct wording, and campaign messages sharing a malicious URL) are **retained** in `existing_unique.parquet` with shared `family_id` so grouped splits keep them together.

## 4. Quarantined, Conflicting & Excluded Rows

Total quarantined unique strict groups: **968** (covering **1050** raw rows).

| Quarantine Reason | Unique Groups | Raw Rows | Description |
| :--- | ---: | ---: | :--- |
| `conflicting_3class_labels` | 1 | 6 | Identical text assigned conflicting 3-class labels (`Promotional` vs `Smishing`) in `BB+SD` |
| `cross_source_conflict_legacy_spam_vs_normal` | 10 | 42 | Labeled `spam` in `legacy_2024` vs `Normal` in `BB+SD` (includes mislabeled money-demand/account-debit SMS) |
| `legacy_binary_ham_undocumented_license_reference_only` | 449 | 466 | `legacy_2024`-only `ham`; undocumented license and unverified against 3-class Promotional boundary |
| `legacy_binary_spam_reference_only` | 507 | 534 | `legacy_2024`-only binary `spam`; never auto-converted to Promotional/Smishing (undocumented license) |
| `legacy_internal_binary_conflict` | 1 | 2 | Internal `ham` vs `spam` conflict within `legacy_2024` |

## 5. Evaluation Membership & Contamination Exclusions

| Evaluation Group (`eval_group`) | Benchmark Membership (`benchmark_membership`) | Rows in `existing_unique` | Eligible for Training |
| :--- | :--- | ---: | :--- |
| `banglaphish_ood` | `banglaphish_2026_ood_other` | 3878 | `False` |
| `banglaphish_ood` | `banglaphish_2026_ood_sms` | 1538 | `False` |
| `smishdetect_test` | `smishdetect_test_direct` | 1376 | `False` |
| `smishdetect_test` | `smishdetect_test_family_overlap` | 322 | `False` |
| `smishdetect_validation` | `smishdetect_validation_direct` | 610 | `False` |
| `smishdetect_validation` | `smishdetect_validation_family_overlap` | 98 | `False` |
| `train_pool` | `primary_train_eligible` | 4125 | `True` |

- **BanglaPhish-2026 OOD isolation:** `5416` rows preserved out-of-training (`eligible_for_training=False`); cross-corpus family overlap with `BB+SD` = **0**.
- **Held-out split family exclusions:** Any `smishdetect_2026` `train` or `bangalabarta_2025` message sharing a `family_id` with `smishdetect_test` or `smishdetect_validation` is excluded from `eligible_train.parquet` (`smishdetect_test_family_overlap`: **322**, `smishdetect_validation_family_overlap`: **98**).
- **Traceability to `phase2_3`:** `phase2_3_cv_fold` preserves existing `unified_3class_dedup.parquet` / `banglaphish2026_dedup.parquet` fold IDs; `eligible_train.parquet` additionally provides `curation_cv_fold` (5-fold `StratifiedGroupKFold` grouped by `family_id`). Note that for any 5-fold CV over the full `BB+SD` corpus, Stage B augmentation must be derived strictly per-fold from that fold's training partition.

## 6. Consolidated Inventory vs. Eligible Training Class Counts

| Class (`label_3class`) | Consolidated Inventory (`existing_unique`) | Primary Non-OOD (`BB+SD`) | OOD Benchmark (`BanglaPhish-2026`) | Eligible Training (`eligible_train`) |
| :--- | ---: | ---: | ---: | ---: |
| `Normal` | 4728 | 2392 | 2336 | 1546 |
| `Promotional` | 1651 | 1571 | 80 | 1009 |
| `Smishing` | 5568 | 2568 | 3000 | 1570 |
| **Total** | **11947** | **6531** | **5416** | **4125** |

## 7. Language Coverage & Unresolved Issues

### Eligible Training (`eligible_train.parquet`) by Language × Class

| Language | Normal | Promotional | Smishing | Total |
| :--- | ---: | ---: | ---: | ---: |
| `Banglish` | 428 | 281 | 436 | 1145 |
| `Bengali` | 393 | 245 | 381 | 1019 |
| `CodeMix` | 336 | 209 | 355 | 900 |
| `English` | 389 | 274 | 398 | 1061 |

### Unresolved Language & Provenance Issues

- **Script vs. source language tag discrepancies (`existing_unique`):** `5` rows tagged `CodeMix` contain 0 Latin characters (`script_bucket=Bengali`); `26` rows tagged `CodeMix` contain 0 Bengali characters (`script_bucket=Latin`); `3` rows tagged `Bengali` contain 0 Bengali characters (`script_bucket=Latin`, e.g., Banglish/English); `715` rows tagged `Bengali` contain Latin URLs/loanwords (`script_bucket=CodeMix`). Both `language` (source metadata) and `script_bucket` are preserved separately for Stage B auditing.
- **BanglaPhish placeholder artifacts:** `2046` `banglaphish_2026` rows trigger `script_bucket=CodeMix` solely because of ASCII bracket placeholders (`[REDACTED_URL]`, `[DATE_01]`, `[AMOUNT]`, etc.) despite `language=Bangla`.
- **Binary spam in `legacy_2024`:** 507 unique `legacy_2024`-only `spam` texts and 449 unique `legacy_2024`-only `ham` texts remain quarantined (undocumented license; cannot safely map binary spam to `Promotional` vs `Smishing` without re-annotation).
- **Label noise observed via cross-source check:** `legacy_2024` flagged 10 texts as `spam` that `BB+SD` labeled `Normal` (6 genuine money-demand/account-debit SMS plus 4 literary/religious quotes). All were quarantined; any unflagged paraphrases in `BB+SD` merit targeted audit before evaluation claims.

## 8. Proposed Stage B Generation Deficits (`T - n_c`)

Using accepted, eligible training rows (`eligible_train.parquet`, total `N = 4125`), with target `T = max(n_Normal, n_Promotional, n_Smishing) = 1570`:

| Class | Eligible Training Count (`n_c`) | Target (`T`) | Proposed Stage B Deficit (`max(0, T - n_c)`) |
| :--- | ---: | ---: | ---: |
| `Normal` | 1546 | 1570 | 24 |
| `Promotional` | 1009 | 1570 | 561 |
| `Smishing` | 1570 | 1570 | 0 |
| **Total** | **4125** | — | **585** |

Stage B generation has **not** been started (Stage A only). Per `data/DATASET_GUIDE.md`, Stage B should begin with a reviewable pilot of at most 300 accepted synthetic rows allocated strictly within these measured deficits (`Normal`: 24, `Promotional`: 561, `Smishing`: 0) while addressing language/script and scenario balance.

## 9. Output Artifacts (`data/processed/curation_v1/`)

- `data/processed/curation_v1/existing_unique.parquet`
- `data/processed/curation_v1/provenance.parquet`
- `data/processed/curation_v1/quarantined.parquet`
- `data/processed/curation_v1/eligible_train.parquet`
- `data/processed/curation_v1/near_duplicate_review.jsonl`
- `data/processed/curation_v1/curation_summary.json`
