# Post-Stage B — Stage 4 Statistical Inference & Validation Report

**Date:** 2026-11-24  
**Production Inference Module:** [post_stage_b_stats.py](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_stats.py)  
**Validation Script:** [validate_stage4_fixed_bootstrap.py](file:///home/bs01415/Documents/Programming/ExploringAIs/scripts/validate_stage4_fixed_bootstrap.py)  
**Verified Result Artifact:** [stage4_statistical_validation_results.json](file:///home/bs01415/Documents/Programming/ExploringAIs/reports/stage4_statistical_validation_results.json)  
**Final Validation Verdict:** **`FAIL_OR_INCONCLUSIVE_CONFIRMATORY_DISABLED`** (`CONFIRMATORY_DECLARATIONS_ENABLED = False`)

---

## 1. Fixed-Budget Production Inference Architecture & Multiplicity

- **Fixed-Budget Bootstrap (Replaces Unvalidated Adaptive Rule):**
  - `B_ci = 2,000`: Family-clustered percentile 95% confidence intervals for $\Delta\text{Macro-F1}$ and $\Delta\text{Recall@1\%FPR}$.
  - `B_pval = 50,000`: Null-centered studentized cluster-bootstrap one-sided p-values using the exact `(TP, E)` `(1 - F1)` Macro-F1 influence-function gradient and tie-inclusive `+1` formula `(1 + count(T*_b >= T_obs)) / (B_pval + 1)`.
  - Independent `np.random.SeedSequence([seed, contrast_index, stream_id])` streams (`chunk_size = 500`, `float64`).
  - Joint IUT: $p_{\text{joint}} = \max(p_{\text{Macro-F1}}, p_{\text{Recall}})$.
  - Round 4's adaptive stopping helper remains optional (`method="adaptive_non_confirmatory"`) with `confirmatory_allowed = False`.
- **Multiplicity Hierarchy (Two Separate Confirmatory/Supporting Families + Separate Exploratory Family):**
  1. **Primary Confirmatory Family (`Holm-12`):** `(B vs A, C vs A, D vs A) × 4 backbones` on `canon_masked` (`1,687`-row primary mask), Joint-IUT p-values Holm-adjusted at $\alpha = 0.05$.
  2. **Supporting Endpoint Family (`Holm-24`):** `12` Macro-F1 + `12` Recall individual endpoint p-values on `canon_masked`, Holm-adjusted separately at $\alpha = 0.05$. *(Never pooled into a 36-hypothesis family.)*
  3. **Prespecified Exploratory Pilot Family (`D vs C`):** `4` backbones comparing Batches 1–7 (`C`) vs. Pilot + Batches 1–7 (`D`) under its own separate Holm-4 (Joint-IUT) and Holm-8 (endpoint) summary (`rejected_at_alpha` always `False` for confirmatory claims).
  4. **Reference View (`raw`):** Reported separately alongside descriptive contrasts.
- **Explicit Conditioning Statement:** All inference is strictly conditional on the trained models, initial seed (`42`), and frozen `smishdetect_validation`-calibrated operating thresholds; it does not capture unconditional multi-seed training variance.

---

## 2. Prespecified Validation Criteria & Mathematical/Edge-Case Checks

**Prespecified Parameters & Gates (fixed before execution in `validate_stage4_fixed_bootstrap.py`):**
- `MASTER_SEED = 20261124`, `MAX_WORKERS = 2`, `OMP_NUM_THREADS = 1`, `nice -n 19` (`total_elapsed = 1,474.98 s` / `24.6 min`).
- Boundary null & Holm counts: `N_outer = 1,000`; negative null counts: `N_outer = 500`.
- Single/Joint Type-I & `Holm-12` Primary FWER gate: empirical rate $\le 0.0600$ **and** 95% Wilson lower bound $\le 0.0500$.
- Negative null Type-I gate (`S4`): empirical rate $\le 0.0100$.
- 95% percentile CI coverage gate (`B_ci = 2,000`): within `[0.9300, 0.9700]`.
- `Holm-24` diagnostic FWER bound: $\le 0.0650$ and lower bound $\le 0.0500$.

**Startup Mathematical & Degenerate Assertions (`PASSED`):**
- Analytical `(TP, E)` `(1 - F1)` gradient vs. central finite differences ($\epsilon = 10^{-6}$, 7 cases): `max_rel_err = 9.835e-10 < 1e-8`.
- Exact `alpha = 0.05` lossless short-circuit vs. full unconditional `B_pval = 50,000`: `0 / 100` mismatches.
- Exact `alpha / 12` null-first lossless Holm gate vs. full `24`-endpoint Holm: `0 / 20` mismatches.
- `8` degenerate/edge cases (`zero_disagreement`, `missing_class_promotional`, `missing_smishing_class`, `near_zero_variance_positive`, `near_zero_variance_negative`, `all_correct_zero_errors`, `tiny_G_5`, `single_cluster_G_1`): `0` NaNs/Infs/exceptions; `zero_disagreement` returns `p_macro_f1 = 1.0`, `p_recall = 1.0`, `CI = [0.0, 0.0]`.

---

## 3. Observed Single-Contrast Validation Results (`B_ci=2,000, B_pval=50,000`)

| DGP | Scenario | `N_outer` | Macro-F1 Rate (`± MC SE`, 95% Wilson) | Recall Rate (`± MC SE`, 95% Wilson) | Joint IUT Rate (`± MC SE`, 95% Wilson) | F1 CI95 Cov | Rec CI95 Cov | Gate Status |
|---|---|---:|---|---|---|---|---:|---:|---|
| `emp_1687` | `S1_both_zero` | `1,000` | `0.0390` (`±0.0061`, `[0.0287, 0.0529]`) | `0.0540` (`±0.0071`, `[0.0416, 0.0698]`) | `0.0080` (`±0.0028`, `[0.0041, 0.0157]`) | `0.9620` | `0.9370` | **PASS** |
| `emp_1687` | `S2_f1_zero_rec_pos` | `1,000` | `0.0480` (`±0.0068`, `[0.0364, 0.0631]`) | `0.9630` *(power)* | `0.0480` (`±0.0068`, `[0.0364, 0.0631]`) | `0.9520` | `0.9460` | **PASS** |
| `emp_1687` | `S3_rec_zero_f1_pos` | `1,000` | `0.9060` *(power)* | `0.0560` (`±0.0073`, `[0.0434, 0.0720]`) | `0.0550` (`±0.0072`, `[0.0425, 0.0709]`) | `0.9570` | `0.9550` | **PASS** |
| `emp_1687` | `S4_both_neg` | `500` | `0.0000` (`[0.0000, 0.0076]`) | `0.0000` (`[0.0000, 0.0076]`) | `0.0000` (`[0.0000, 0.0076]`) | `0.9440` | `0.9460` | **PASS** |
| `syn_stress` | `S1_both_zero` | `1,000` | `0.0510` (`±0.0070`, `[0.0390, 0.0664]`) | **`0.0620`** (`±0.0076`, `[0.0487, 0.0787]`) | `0.0090` (`±0.0030`, `[0.0047, 0.0170]`) | `0.9340` | `0.9410` | **FAIL (Rec `0.0620 > 0.0600`)** |
| `syn_stress` | `S2_f1_zero_rec_pos` | `1,000` | `0.0430` (`±0.0064`, `[0.0321, 0.0574]`) | `0.9670` *(power)* | `0.0430` (`±0.0064`, `[0.0321, 0.0574]`) | `0.9580` | `0.9450` | **PASS** |
| `syn_stress` | `S3_rec_zero_f1_pos` | `1,000` | `0.9960` *(power)* | `0.0520` (`±0.0070`, `[0.0399, 0.0676]`) | `0.0520` (`±0.0070`, `[0.0399, 0.0676]`) | `0.9570` | `0.9450` | **PASS** |
| `syn_stress` | `S4_both_neg` | `500` | `0.0000` (`[0.0000, 0.0076]`) | `0.0000` (`[0.0000, 0.0076]`) | `0.0000` (`[0.0000, 0.0076]`) | `0.9540` | `0.9420` | **PASS** |

---

## 4. Observed Multi-Hypothesis Holm FWER Results (`N_outer = 1,000` Each)

| Family Validation Scenario | Family Type | `m` Joint / `2m` End | `Holm-m` Joint FWER (`± MC SE`, 95% Wilson) | `Holm-2m` Null-Endpoint FWER (`± MC SE`, 95% Wilson) | Gate Status |
|---|---|---:|---|---|---|
| `emp_1687_holm12_correlated` | Primary confirmatory | `12` / `24` | `0.0530` (`±0.0071`, `[0.0407, 0.0687]`) | `0.0420` (`±0.0063`, `[0.0312, 0.0563]`) | **PASS** |
| `emp_1687_holm12_independent` | Primary confirmatory | `12` / `24` | `0.0460` (`±0.0066`, `[0.0347, 0.0608]`) | `0.0380` (`±0.0060`, `[0.0278, 0.0517]`) | **PASS** |
| `syn_stress_holm12_correlated` | Primary confirmatory | `12` / `24` | `0.0510` (`±0.0070`, `[0.0390, 0.0664]`) | `0.0450` (`±0.0066`, `[0.0338, 0.0597]`) | **PASS** |
| `emp_1687_exploratory_c_vs_d_holm4` | Exploratory diagnostic | `4` / `8` | `0.0610` (`±0.0076`, `[0.0478, 0.0776]`) | `0.0570` (`±0.0073`, `[0.0443, 0.0731]`) | Diagnostic (`non-primary`) |

---

## 5. Fail-Closed Governance Decision & Consequence

Because `syn_stress::S1_both_zero` Recall Type-I error (`62 / 1,000 = 0.0620`, 95% Wilson CI `[0.048664, 0.078688]`) exceeded the prespecified hard threshold `0.0600`, **zero simulation extensions or threshold tweaks were performed** and the production gate was automatically switched fail-closed:

- **`CONFIRMATORY_DECLARATIONS_ENABLED = False`** and **`STATISTICAL_VALIDATION_STATUS = "FAIL_OR_INCONCLUSIVE_CONFIRMATORY_DISABLED"`** in [post_stage_b_stats.py](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_stats.py#L38-L39) and enforced fail-closed in `compare_models_cluster_bootstrap`, `holm_reject_family` (default `confirmatory_allowed=False`; separated from simulation-only `simulate_holm_reject_family`), `evaluate_stage4_hypothesis_families`, [post_stage_b_ensemble.py](file:///home/bs01415/Documents/Programming/ExploringAIs/src/bangla_smish/post_stage_b_ensemble.py#L232), and all runner exports (`aggregated_results.json`, `aggregated_results.csv`, `post_stage_b_report.md`, and standalone `retraining_metadata.json`).
- All primary `Holm-12` and supporting `Holm-24` hypotheses stay disabled (`"rejected_at_alpha": false`, `"p_value_status": "UNVALIDATED_DIAGNOSTIC_ONLY"`), even if callers attempt to pass `confirmatory_allowed=True`, while preserving descriptive effect sizes, influence-function SEs, `B_ci = 2,000` family-clustered 95% percentile CIs, and nominal p-values for transparent diagnostic reporting unless a future owner-authorized, pre-registered validation explicitly passes.
