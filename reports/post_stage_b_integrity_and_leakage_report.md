# Post-Stage B dataset integrity and leakage

**Decision: NO-GO for the complete grid.** The versioned dataset passes the frozen-source, row-ID, family-ID, and original-text checks. Two model views produce exactly equal training and evaluation inputs across frozen boundaries. The full run is blocked while the owner reviews that experimental-design issue.

## Frozen sources and manifests

The checkout was `main` at `282ef63b374d9bafa786bf7aabc6a987a73d3777` with empty `git status --short` at preflight. All 249 preflight files under `data/` matched their original SHA256 after creating the separate v2 bundle. The legacy `DataSet.csv` matched its Git HEAD contents. Each of the seven Parquet files in each bundle matched its recorded SHA256, row count, schema column names, and byte size. The 13 protected source digests are recorded in both manifests and the data module. [Protected digest definitions](../src/bangla_smish/post_stage_b_data.py#L42)

| Bundle | Manifest SHA256 |
|---|---|
| [Frozen v1](../data/processed/post_stage_b_bundle/dataset_manifest.json) | `2e2e76e296873ffc843629ccae560dd8df6597debf93cfe2ab522af49dcd7a9b` |
| [Versioned v2](../data/processed/post_stage_b_bundle_v2/dataset_manifest.json) | `308d1c889840c1ec04e1cd6f6751cccfca0e559f2e12c61e18ff2e54d6fa2922` |

The v2 manifest additionally fingerprints all seven source brand registries and the assembled 364-token vocabulary. All seven v2 tables matched the v1 row identity, order, original text, and four clean text views. All 1,279 counterfactual rows and their stored variants were identical. The six standard tables retain the old raw adversarial input as `text_adv_raw` and include three corresponding masked or stripped views.

### Original source payload fingerprints

| Local source payload | SHA256 |
|---|---|
| Legacy `DataSet.csv` | `5df79924b5966513760ad6ec38911cb12a1233a569b1a645b83b5b0936eabf27` |
| BangalaBarta source CSV | `260c2e3bd12c6f35d47eb9be255ec42e5d14f0b3df4c594c40776e99586897e9` |
| SmishDetect train Parquet | `9fb92ea7e185297f4dd66442a7ed5f11ffc82d501c1ca2ea38385f430b9fe18e` |
| SmishDetect validation Parquet | `a6a652087bbc328618cc39720a3ec66be53e106d9184dcf83791bc4da705a1fc` |
| SmishDetect test Parquet | `2719d61aaf82b4b38e256e7e0d5ecee4a19588a7f522b863c713a435d0dffa31` |
| BanglaPhish master CSV | `4d8dd5d2c3423ebb1c2db6dd16db8e0bcc1bf489a5caa99288dcb7cf08fe979c` |

The BanglaPhish publisher checksum inventory matched 22 of its 23 entries, including all dataset payloads. Its local README measured `839a8212e046d9c6443ecccb771ca7ff24b73b754451c5985cf5403ce423c16c` against published `ba8a5bf2d3985b40ad9c15d04142c92840ba471bfbc1fb86a13b7d0c25984310`. The checksum command exited 1 for that documentation mismatch. The README also matched its local preflight digest. Publisher hash validation for the other raw sources was unavailable from the local inventory. The local hashes above are fingerprints from this checkout.

## Cohort measurements

| Cohort | Rows | Families | Normal | Promotional | Smishing |
|---|---:|---:|---:|---:|---:|
| Eligible prior-source training pool | 3,999 | 3,803 | 1,509 | 942 | 1,548 |
| Accepted Pilot | 300 | 292 | 39 | 261 | 0 |
| Accepted Batches 1–7 | 345 | 345 | 0 | 345 | 0 |
| Full training pool | 4,644 | 4,440 | 1,548 | 1,548 | 1,548 |
| Curated SmishDetect validation | 708 | 606 | 265 | 178 | 265 |
| Curated SmishDetect test | 1,698 | 1,304 | 581 | 384 | 733 |
| BanglaPhish OOD | 5,416 | 4,335 | 2,336 | 80 | 3,000 |

| Promotional subset | Bengali | Banglish | English | CodeMix |
|---|---:|---:|---:|---:|
| Eligible prior-source pool | 229 | 141 | 338 | 234 |
| Pilot | 65 | 115 | 21 | 60 |
| Batch 1 | 13 | 19 | 4 | 14 |
| Batch 2 | 14 | 19 | 4 | 13 |
| Batch 3 | 14 | 19 | 4 | 13 |
| Batch 4 | 13 | 19 | 4 | 14 |
| Batch 5 | 13 | 19 | 4 | 14 |
| Batch 6 | 14 | 19 | 4 | 13 |
| Batch 7 | 12 | 17 | 4 | 12 |
| Batches 1–7 | 93 | 131 | 28 | 93 |
| Full training pool | 387 | 387 | 387 | 387 |

The decoded bundle rows match the protected eligible training table, accepted JSONL membership, and frozen curated evaluation groups in order. The BanglaPhish OOD text-label pairs match the 5,416-row deduplicated reference. All 645 accepted synthetic rows have accepted review status and no source-parent IDs in the accepted metadata. The Pilot's 292 families include rows sharing a family within that cohort. All 345 batch families are distinct. The 3,803 real families each map to exactly one fold.

Each of 15 cohort pairs had zero common row IDs, family IDs, exact original SMS strings, or independently recomputed strong-normalized strings. Each cohort had unique row IDs and unique exact and strongly normalized text. Synthetic accepted strings also had zero strong-normalized matches with the entire `existing_unique` or `quarantined` inventories. The strict comparison uses NFKC, casefold, Unicode-format character removal, and whitespace collapse. It leaves original bytes untouched. [Comparison function](../src/bangla_smish/curation.py#L99) · [Live content checks](../src/bangla_smish/post_stage_b_data.py#L417) · [Source alignment](../src/bangla_smish/post_stage_b_data.py#L674)

## Five-fold measurements

A and B use identical ordered training IDs in each fold, including every eligible real training-partition row. A/B/C/D use identical ordered real validation IDs. C includes exactly the 345 accepted batch IDs in every training partition. D adds those IDs and all 300 Pilot IDs. All 20 train/validation partitions have zero row-ID and family-ID intersection. Each of 3,999 eligible real IDs appears in validation once per five-fold experiment.

| Fold | A/B train | C train | D train | Validation | Val N / P / S | Eligible CF rows |
|---|---:|---:|---:|---:|---|---:|
| 0 | 3,199 | 3,544 | 3,844 | 800 | 302 / 189 / 309 | 261 |
| 1 | 3,199 | 3,544 | 3,844 | 800 | 301 / 189 / 310 | 256 |
| 2 | 3,199 | 3,544 | 3,844 | 800 | 302 / 188 / 310 | 252 |
| 3 | 3,199 | 3,544 | 3,844 | 800 | 302 / 188 / 310 | 261 |
| 4 | 3,200 | 3,545 | 3,845 | 799 | 302 / 188 / 309 | 249 |

The eligible counterfactual table contains 1,279 Smishing rows from 1,171 families. Each fold selects exactly the probe rows with eligible validation parents before any smoke cap. Their IDs, families, folds, languages, and labels match eligible parent rows. Every eligible ID appears exactly once across the five fold subsets per experiment. The smoke configuration remains a bounded subset and provides no full probe coverage estimate. [Fold constructor](../src/bangla_smish/post_stage_b_data.py#L785) · [Probe selection](../src/bangla_smish/post_stage_b_data.py#L384) · [All-fold regression](../tests/test_post_stage_b_pipeline.py#L128)

## Findings

| Severity | Evidence and root cause | Disposition and verification |
|---|---|---|
| Critical | `brand_masked` and `brand_stripped` create equal model input strings for separately identified SMS rows. The full grid audit found 128 failing comparisons out of 560, each containing one row pair. All original row and family checks passed. The masking step collapses distinct source strings and can create new identity across boundaries. | The full runner checks this before training. Existing split and evaluation membership stays frozen pending an owner decision. The read-only overlap command exited 1 on v2 and reproduced all 128 failing comparisons. [Model-view audit](../src/bangla_smish/post_stage_b_model_view_audit.py#L30) · [Full runner regression](../tests/test_post_stage_b_pipeline.py#L217) |
| Critical | The original runner passed the same global probe to every fold. Reconstructing the old first-32 smoke probe finds 21 rows with parents assigned outside fold 0. The frozen probe already carried correct fold ownership. | The runner selects only the unit's validation-owned rows, verifies source and family isolation, and uses that subset for prediction and metrics. All-fold and deliberately misassigned-parent tests passed. Exact model-input matches remain called out separately above. [Runner integration](../src/bangla_smish/post_stage_b_runner.py#L613) · [Tampering regression](../tests/test_post_stage_b_pipeline.py#L162) |
| High | The older bundle checker trusted integrity booleans, verified file digests alone, and tested an incomplete set of cohort boundaries. The assembly default reused the frozen output and the optional kit export recursively removed any existing destination. | Assembly and export require fresh destinations. Verification recomputes all 15 boundaries, provenance, class counts, fold probe membership, and Parquet metadata. The manifest-tamper and existing-destination regressions passed, and both explicit bundle verifications exited 0. [Manifest verifier](../src/bangla_smish/post_stage_b_data.py#L629) · [Protected export](../scripts/prepare_post_stage_b_bundle.py#L23) · [Regression](../tests/test_post_stage_b_pipeline.py#L177) |
| Medium | 79 of 345 batch rows in v1 carried `text_norm_strict` values different from the curation normalizer. The old standardizer accepted stored fields and otherwise used lowercase/strip. Recomputing comparisons still returned zero exact overlaps. | v2 derives every synthetic comparison value from a shared strong normalizer. The v1/v2 content regression passed and measured zero normalization metadata mismatches in v2. [Synthetic standardizer](../src/bangla_smish/post_stage_b_data.py#L297) · [Regression](../tests/test_post_stage_b_pipeline.py#L192) |
| Medium | Source provenance distinguishes SmishDetect as `augmented_multilingual` and all 5,416 BanglaPhish OOD rows as `synthetic`. Bundle `is_synthetic` marks the Stage B intervention. The inherited `real-only` shorthand and broad OOD name can suggest different field provenance. | Report eligible-source baselines and synthetic-source OOD generalization. No benchmark membership change is needed. Read-only provenance confirmed all 4,110 linked SmishDetect source records for the 3,999 eligible training IDs come from the raw train split. [Provenance join](../src/bangla_smish/post_stage_b_data.py#L674) |
| Low | One BanglaPhish provider README hash differs from its checksum inventory. The cause of the documentation difference is unverified. The isolated check measured 22/23 and all listed dataset payloads passed. | Record the discrepancy. The 5,416-row derived OOD reference digest and exact text-label alignment passed. No local payload was edited. [Source check](../src/bangla_smish/post_stage_b_data.py#L720) |

Curated SmishDetect validation contains 610 rows with original primary split validation and 98 with original primary split train. Curated test contains 1,376 original primary test, 269 original primary train, and 53 original primary validation. These are frozen curated evaluation groups with prior family-based split assignment. Report them as curated groups when comparing with published raw splits.

### Exact model-input matches requiring disposition

The CV collision is between Smishing rows `cur_43bd66f47b5b` (fold 3, `fam_af79b35a6a`) and `cur_c92eaaa1966b` (fold 4, `fam_3408a3c08d`). Both directions occur in every experiment for both brand views. Each of the three counterfactual variants of the corresponding validation row also equals the same-view training input in those units.

Curated SmishDetect validation row `cur_7d47602782d3` matches training row `cur_29c93892c828` in both brand views. Curated test row `cur_cf8256b4cb6e` matches training row `cur_42a3cdc4a3d8`. All four rows are Smishing. Both training counterparts are in fold 3, so the two held-out matches appear for units 0, 1, 2 and 4 in all four experiments. OOD had zero exact same-view matches. Raw and canon-masked had zero exact same-view matches in the evaluated clean and probe scopes. The audit compares strings before tokenizer normalization and truncation. Backbone-specific token equality remains unverified.

### Heuristic near-duplicate screen

A second diagnostic fit char-word-boundary TF-IDF using 3–5 character n-grams across the 12,466 decoded rows from the six standard cohorts. It counted pairwise cosine scores at or above 0.85 using separately recomputed strong normalization and slot templates. Because IDF depends on the fitted inventory and linguistic similarity can be benign, these are review candidates. The source data remain frozen.

| Boundary | Strong-normalized pairs | Slot-template pairs |
|---|---:|---:|
| Real pool → curated validation | 0 | 6 |
| Real pool → curated test | 2 | 13 |
| Curated validation → curated test | 0 | 4 |
| CV fold 0 | 0 | 3 |
| CV fold 1 | 0 | 3 |
| CV fold 2 | 1 | 3 |
| CV fold 3 | 0 | 4 |
| CV fold 4 | 1 | 1 |
| Any cohort-pair boundary involving Pilot, Batches 1–7, or OOD | 0 | 0 |

The two strict test pairs were `cur_2ee2c999bea1` / `cur_cfa2f9f81a31` (0.8528) and `cur_e2a702578e9e` / `cur_40dd2f53ca2f` (0.8513). The reciprocal CV pair was `cur_31f35c6da04c` / `cur_647968080b19` (0.8501). The corresponding families are distinct in the frozen inventory. The owner should adjudicate these candidates with the separate exact model-input findings before any proposed revised analysis.

## Executed verification

| Command | Actual result |
|---|---|
| `.venv/bin/python scripts/prepare_post_stage_b_bundle.py --verify-only` | Exit 0. Verified v1 physical and decoded integrity, source alignment, fold coverage, and reported 79 legacy normalization metadata mismatches. |
| `.venv/bin/python scripts/prepare_post_stage_b_bundle.py --bundle-dir data/processed/post_stage_b_bundle_v2` | Exit 0. Built a new local 14 MiB bundle with frozen rows and four clean views. |
| `.venv/bin/python scripts/prepare_post_stage_b_bundle.py --verify-only --bundle-dir data/processed/post_stage_b_bundle_v2` | Exit 0. Verified all source and derived digests, content checks, and zero stored normalization mismatches. |
| `.venv/bin/python scripts/audit_post_stage_b_model_views.py --bundle-dir data/processed/post_stage_b_bundle_v2` | Exit 1. Detected 128 failing comparisons of 560. |
| `cd data/raw/banglaphish2026 && sha256sum -c SHA256SUMS.txt` | Exit 1. 22 of 23 entries matched, including every data payload. |
| Five focused `TestPostStageBPipeline` dataset regressions | Exit 0. All five passed in 60.787 seconds using a fresh temporary v2 bundle. |
| Temporary local `export_drive_kit` verification | Exit 0. Copied and verified the v2 bundle under its versioned name, observed `FileExistsError` on a repeat export, and confirmed the original temporary kit manifest bytes remained intact. |

Focused regression command

```bash
.venv/bin/python -m unittest \
  tests.test_post_stage_b_pipeline.TestPostStageBPipeline.test_experiment_abcd_fold_splits_leakage_free \
  tests.test_post_stage_b_pipeline.TestPostStageBPipeline.test_counterfactual_parent_fold_tamper_detected \
  tests.test_post_stage_b_pipeline.TestPostStageBPipeline.test_class_weights_from_hand_calculated_training_distribution \
  tests.test_post_stage_b_pipeline.TestPostStageBPipeline.test_bundle_physical_metadata_and_original_bundle_preservation \
  tests.test_post_stage_b_pipeline.TestPostStageBPipeline.test_recompute_frozen_and_versioned_bundle_integrity -v
```
