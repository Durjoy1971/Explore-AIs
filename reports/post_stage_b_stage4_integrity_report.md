# Post-Stage B — Stage 4 Protected Dataset & Repository Integrity Report

**Date:** 2026-11-24  
**Repository Root:** `/home/bs01415/Documents/Programming/ExploringAIs`  
**Git Branch / Base Commit:** `main` · `4d341183cb7a6fce0ed773ff33da2a3b757d6f94`  
**Integrity Verification Script:** [verify_stage4_integrity.py](file:///home/bs01415/Documents/Programming/ExploringAIs/scripts/verify_stage4_integrity.py)  
**Protected Baseline Manifest:** [protected_baseline_282.json](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/protected_baseline_282.json)

---

## 1. Repository State & Category A Preservation

- **Zero Category A Modifications:** None of the `282` frozen files (`DataSet.csv`, `data/**`, or the historical Stage A/B/Post-Stage B reports) were created, modified, overwritten, or deleted during Stage 4 implementation.
- **Pre-Existing Work Preserved:** Pre-existing changes (`CHECKPOINT.md` updated only within authorized Stage 4 scope, `reports/post_stage_b_stage3_readiness_review.md`, and session `response.md`) were preserved cleanly without any `git commit`, `push`, `merge`, `reset`, or discard.
- **Non-Self-Referential Manifest Design:** `reports/protected_baseline_282.json` lives in Category C and indexes exclusively the exact `282` Category A files (`non_self_referential = true`, `owner_acceptance_status = "ACCEPTED_BY_OWNER_STAGE4"`).

---

## 2. Canonical Inputs (`13/13`) & Frozen Bundles (`v1` and `v2`)

1. **`verify_protected_inputs(..., strict_digests=True)`:** Passed `13 / 13` SHA256 digests (`eligible_train.parquet`, `existing_unique.parquet`, `provenance.parquet`, `quarantined.parquet`, `banglaphish2026_dedup.parquet`, Pilot `accepted.jsonl`, and `batch1_50`..`batch7_45` `accepted.jsonl`).
2. **Frozen Bundle `v1` (`data/processed/post_stage_b_bundle`):** Passed `--verify-only` (`manifest_sha256 = 2e2e76e296873ffc843629ccae560dd8df6597debf93cfe2ab522af49dcd7a9b`, `cohort_pairs_checked = 15`, full-pool `1,548` per class, all 5 `V1_INTEGRITY_CHECKS` true).
3. **Frozen Bundle `v2` (`data/processed/post_stage_b_bundle_v2`):** Passed `--verify-only` (`manifest_sha256 = 308d1c889840c1ec04e1cd6f6751cccfca0e559f2e12c61e18ff2e54d6fa2922`, `cohort_pairs_checked = 15`, `0` normalization mismatches across all tables, all 7 integrity checks true).

---

## 3. Protected `282`-File Inventory Breakdown (`verify_all_stage4_integrity`)

Verified **`282 / 282`** exact path, byte size, and SHA256 matches, explicitly separating historically anchored files from owner-accepted current-state snapshots and scientific artifacts from machine-specific metadata:

| Provenance Tier | Count | `historical_proof` | Description |
|---|---:|:---:|---|
| `git_282ef63` | `175` | `True` | Byte-for-byte Git blob match at commit `282ef63` |
| `git_4d34118_v2_and_reports` | `15` | `True` | `v2` bundle (`8`) + accepted Post-Stage B reports (`7`) at commit `4d34118` |
| `upstream_sha256sums` | `23` | `True` | Verified against `data/raw/banglaphish2026/SHA256SUMS.txt` |
| `consolidation_v1_prefix_and_intrep` | `4` | `True` | 10-hex prefix in `282ef63:reports/dataset_consolidation_v1.md` + full SHA256 in integrity report |
| `protected_input_digests` | `1` | `True` | Raw `eligible_train` / source digest anchored in `PROTECTED_INPUT_DIGESTS` |
| **`current_state_snapshot_accepted_not_historical_proof`** | **`64`** | **`False`** | Owner-accepted current-state snapshots (historical hashes never fabricated) |
| **Total Protected Category A Files** | **`282`** | — | **218 historical + 64 current-state snapshots** |

- **Scientific vs. Transient Machine Metadata Split:**
  - `scientific_dataset_or_report`: **`240`** files (all raw datasets, processed parquets, synthetic batches, `v1`/`v2` bundles, `DataSet.csv`, historical reports).
  - `machine_specific_metadata`: **`42`** files (`data/raw/banglaphish2026/.git/*`, `data/raw/smishdetect_hf/.cache/huggingface/*`).
- **Two Explicitly Distinct Integrity Contracts:**
  - **Contract A — Full Local Repository Integrity (`282/282`):** Verifies all `282` Category A files (`240` scientific + `42` machine metadata; `218` historical + `64` snapshots) in the full local repository via `verify_protected_baseline_manifest(..., check_portable_scientific_only=False)`.
  - **Contract B — Portable Colab Execution Kit Integrity (`33` runtime files, `post_stage_b_stage4_kit_v2`):** Defined by `build_portable_colab_kit_manifest` / `verify_portable_colab_kit` ([post_stage_b_integrity.py](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_integrity.py#L426-L610)). Stages and independently verifies the exact `33` required runtime files (`17` `src/bangla_smish/*.py`, `3` configs, `3` gitignored Stage 4 `reports/*.json`, `pyproject.toml`, the Stage 4 notebook, and all `8` `post_stage_b_bundle_v2/*` files cross-checked against `protected_baseline_282.json`) while setting `verifies_full_repository_282 = False` and explicitly never claiming to verify absent historical repository files. Notebook Cell 5 invokes `verify_portable_colab_kit(WORKSPACE_ROOT)` cleanly in isolated environments.

---

## 4. Frozen Deterministic Test Mask Integrity (`stage4_frozen_test_masks.json`)

- **Target Parquet:** `data/processed/post_stage_b_bundle_v2/eval_smishdetect_test.parquet` (`1,698` rows, `1,304` families).
- **Masked SHA256 Digests:**
  - `primary_family_decontam_1687`: `mask_sha256 = 86527c48b8a803842990b64e127828f9684e4a7cc55c88c1f8b050be591d105d` (excludes `5` families / `11` Smishing rows: `fam_03f840170c`, `fam_210d66c8f3`, `fam_735d8c7d90`, `fam_77db01305f`, `fam_b7af3c4ecc`).
  - `supplementary_full_1698`: `mask_sha256 = e73e775f70a0485b8bf777796a140cdcd8392fa2a1740adc3da4de01dff312d6`.
  - `sensitivity_row_only_1693`: `mask_sha256 = a70fe670d874912f77bae763d1db38adaf407de079cc63b48408c67f4c321c13` (excludes `cur_7936ee1f5382`, `cur_7c14e8bae65c`, `cur_c00c6e3fa781`, `cur_c24e8fee19a2`, `cur_cfa2f9f81a31`).
  - `sensitivity_template_stress_1675`: `mask_sha256 = 0b2e99963ee6f413bff1e06697bb6c607b9f7875149a7190389deca9f752b480` (excludes `12` families / `23` rows).
