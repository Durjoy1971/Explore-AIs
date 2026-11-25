# Post-Stage B Stage 4 v8 Repository & Portable Kit Integrity Report

**Date:** 2026-11-25  
**Verification Mode:** Read-only verification under kernel-enforced `<= 25%` host CPU cap (`effective_host_cpu_pct = 8.30%` and `10.94%`).

---

## 1. Full Repository Integrity (`scripts/verify_stage4_integrity.py --verify-only`)

- **Exit Code:** `0` (`effective_host_cpu_pct = 8.30%`, `wall_seconds = 19.025s`, `cgroup_nr_throttled_delta = 0`)
- **Category A Protected Baseline (`reports/protected_baseline_282.json`):** **`282/282` VERIFIED** (`240` scientific dataset/report files + `42` machine-specific metadata files; **`0` modified, `0` missing**).
- **Protected Inputs:** **`13/13` VERIFIED** (`DataSet.csv`, `data/raw/*`, `data/curated/*`, `data/generated/*`).
- **Frozen Bundles:**
  - `data/processed/post_stage_b_bundle/` (`v1`): verified (`15` cohort pairs checked, `0` mismatches).
  - `data/processed/post_stage_b_bundle_v2/` (`v2`): verified, manifest SHA256 `308d1c889840c1ec04e1cd6f6751cccfca0e559f2e12c61e18ff2e54d6fa2922` (`0` normalization mismatches).
- **Frozen `smishdetect_test` Cohort Masks (`4/4` re-derived and verified):**
  - `primary_family_decontam_1687` (`1,687` rows / `1,299` families): `86527c48b8a803842990b64e127828f9684e4a7cc55c88c1f8b050be591d105d`
  - `supplementary_full_1698` (`1,698` rows / `1,304` families): `e73e775f70a0485b8bf777796a140cdcd8392fa2a1740adc3da4de01dff312d6`
  - `sensitivity_row_only_1693` (`1,693` rows / `1,302` families): `a70fe670d874912f77bae763d1db38adaf407de079cc63b48408c67f4c321c13`
  - `sensitivity_template_stress_1675` (`1,675` rows / `1,292` families): `0b2e99963ee6f413bff1e06697bb6c607b9f7875149a7190389deca9f752b480`
- **Statistical Governance Artifact:** `reports/stage4_statistical_validation_results.json` untouched (`bde8085b77066cbf449a1b17d0a1afa77f2dcbcb4ea1a82f7d0dde68f6fe504f`, `FAIL_OR_INCONCLUSIVE_CONFIRMATORY_DISABLED`, `CONFIRMATORY_DECLARATIONS_ENABLED = False`).

---

## 2. Versioned Portable Colab Kits (`v3`–`v8`) Preservation & `v8` Parity

All five historical portable kits (`v3`, `v4`, `v5`, `v6`, `v7`) were preserved completely untouched alongside the newly staged `v8` kit. Every kit independently passed `verify_portable_colab_kit` (`34/34` runtime files, `8/8` v2 bundle baseline cross-checks, `4/4` live re-derived frozen masks):

| Kit Version | Status | Verified Files | Masks | Notebook SHA256 | `stage4_kit_manifest.json` SHA256 |
| :--- | :--- | ---: | ---: | :--- | :--- |
| `v3` (historical) | Preserved & Verified | `34/34` | `4/4` | `53b35f508f9a4c783ba951e409c0cf324b9992732f629b65ac06eb587f50a219` | `b27dfbd5db27b4312e76fc90e3a265ad899cd484fe75ce934be35378c49fd258` |
| `v4` (historical) | Preserved & Verified | `34/34` | `4/4` | `484302bc2b4543d5816046abc961a44fabf058c292f8e8f7a539858f9dd4e8ed` | `59bc766eb9c246395cc16fac9cee0b5a1bac70b8844255f6b117bdd31cd7b06e` |
| `v5` (historical) | Preserved & Verified | `34/34` | `4/4` | `d0c0c173c629ef97cb3b2c5df60f72e669fdec69e9bf1868abc8c9e695ba468c` | `db7e7eeacc236c7753c41639ee1923f9587c5a9f059c23d7afca6c87e9e6a3f4` |
| `v6` (historical) | Preserved & Verified | `34/34` | `4/4` | `06b227ff536ac9e660e6c36b0da3cd963b26a8ceaab308daa9b1c60f12a594e5` | `528f0d9ab93f12d6c78268829eba78619063c1d83612ee7d29db8907b636f99d` |
| `v7` (historical) | Preserved & Verified | `34/34` | `4/4` | `8b52b0ba06ebb21259f207cfed3a9bf57919b4fb78cff7a25ae3d2d251bfec41` | `38d6a9efb951d570fd2e40643dcd8bb253e215dcd5316a8ae6b0e59308ace0b8` |
| **`v8` (new)** | **Staged & Verified** | **`34/34`** | **`4/4`** | **`70f8f15f858746539eb6935565611310c49bd71a4651624883d68d74f37b75fa`** | **`9933a1907cdd05e648c10955eda5c327d9dc3d4056506078e514faf82f35043c`** |

- **`v8` Repository-to-Kit Parity:** **`34/34` exact SHA256 matches** (`0` mismatches) between canonical repository files, `stage4_colab_kit_upload/bangla_smish_post_stage_b_stage4_v8/`, and `stage4_kit_manifest.json`.
- **Run-Identity Source Hashing (`schema_version = 5`):** `_run_identity()["source_sha256"]` now hashes **all `18/18` runtime modules** listed in `STAGE4_KIT_REQUIRED_SOURCE_FILES` (upgraded from `13/18` in v7).
- **OS Metadata Filtering:** `verify_portable_colab_kit` now ignores `Thumbs.db` and AppleDouble `._*` files in addition to `.DS_Store`, `desktop.ini`, and `.ipynb_checkpoints`, without relaxing rogue-file rejection.
