# Follow-up Scientific Audit — Resolve Outstanding Issues in Explore-AIs

**Audit Posture:** Strictly **READ-ONLY** follow-up scientific audit of `Durjoy1971/Explore-AIs` (no modifications to source code, datasets, frozen bundles, masks, configurations, checkpoints, or statistical code; zero Colab/GPU runs; zero synthetic data generation; no automatic experiment authorization). All statistics, counts, file paths, symbol names, `family_id`s, and live HTTP checks below were independently verified against repository artifacts and live external endpoints on `2026-11-25`.

---

## 1. Statistical Type-I Error Calibration

**Inspected artifacts:** [`scripts/validate_stage4_fixed_bootstrap.py`](scripts/validate_stage4_fixed_bootstrap.py), [`src/bangla_smish/post_stage_b_stats.py`](src/bangla_smish/post_stage_b_stats.py), [`reports/stage4_statistical_validation_results.json`](reports/stage4_statistical_validation_results.json), [`reports/post_stage_b_stage4_statistical_validation.md`](reports/post_stage_b_stage4_statistical_validation.md).

1. **Why the operational gate correctly fails:**
   - [`scripts/validate_stage4_fixed_bootstrap.py`](scripts/validate_stage4_fixed_bootstrap.py) (lines `70–71`, `636–637`) enforces the prespecified fail-closed rule:
     $$\text{fail if } \hat{p} > \text{MAX\_EMPIRICAL\_TYPE1 } (0.060) \;\lor\; \text{CI}_{95,\text{Wilson}}^{\text{lo}} > \text{MAX\_WILSON\_LOWER } (0.050).$$
   - In `syn_stress::S1_both_zero` ([`reports/stage4_statistical_validation_results.json`](reports/stage4_statistical_validation_results.json#L339-L347), `L663–665`), Recall@$\le 1\%$FPR rejected $k = 62$ out of $N_{\text{outer}} = 1{,}000$ replications ($\hat{p} = 0.0620$, $\text{SE} = 0.007626$, Wilson 95% CI $[0.048664, 0.078688]$).
   - Although the Wilson lower bound ($0.048664 \le 0.050$) did not trigger the second clause, the observed point estimate ($0.0620 > 0.0600$) tripped the first clause, appending the failure to `gate_failures` and preserving `CONFIRMATORY_DECLARATIONS_ENABLED = False` (`STATISTICAL_VALIDATION_STATUS = "FAIL_OR_INCONCLUSIVE_CONFIRMATORY_DISABLED"` in [`src/bangla_smish/post_stage_b_stats.py`](src/bangla_smish/post_stage_b_stats.py#L38-L39)).

2. **Has the underlying Type-I error rate been statistically demonstrated to exceed `0.06` (or `0.05`)?**
   - **No.** Exact binomial tests on $k = 62,\; n = 1{,}000$:
     - Against the operational ceiling $p_0 = 0.06$: two-sided $p = 0.789786$; one-sided ($H_1: p > 0.06$) $p = 0.413543$. The data are completely compatible with a true Type-I error rate $\le 0.06$.
     - Against nominal $\alpha = 0.05$: two-sided $p = 0.094458$; one-sided ($H_1: p > 0.05$) $p = 0.051110$ — borderline and **not** statistically established at $\alpha = 0.05$.
     - Both the Wilson 95% CI $[0.048664, 0.078688]$ and exact Clopper-Pearson 95% CI $[0.047862, 0.078780]$ straddle both $0.05$ and $0.06$.
   - **Conclusion:** `62/1,000 = 0.0620` is an **operational gate failure**, not statistical proof that the true Type-I error exceeds `0.06`.

3. **Do `1,000` Monte Carlo replications provide sufficient precision?**
   - **No — `N_outer = 1,000` is underpowered relative to the `1.0 pp` margin between nominal `0.050` and gate ceiling `0.060`.**
   - At $N = 1{,}000$, $\text{SE}(p=0.05) = \sqrt{0.05 \times 0.95 / 1000} = 0.00689$ ($95\%$ half-width $\approx \pm 1.35\text{ pp}$), and at $k = 62$ the Wilson 95% CI spans $3.00\text{ pp}$ ($0.030024$) — roughly $3\times$ the `1.0 pp` separation between `0.05` and `0.06`.

4. **Additional simulation design to resolve the uncertainty (future owner-authorized, pre-registered only):**
   - Increase boundary-null replications to $N_{\text{outer}} = 5{,}000$ ($\text{SE}(0.05) = 0.00308$, half-width $\pm 0.60\text{ pp}$) or $N_{\text{outer}} = 10{,}000$ ($\text{SE}(0.05) = 0.00218$, half-width $\pm 0.43\text{ pp}$).
   - Run a pre-registered factorial ablation of `syn_stress` ([`scripts/validate_stage4_fixed_bootstrap.py`](scripts/validate_stage4_fixed_bootstrap.py#L122-L125), `L532–542`): Recall@1%FPR uses only Smishing-bearing clusters ($\approx 40\%$ of $G=1{,}299$, i.e., $G_{\text{eff}} \approx 520$), heavy-tail cluster weights $\{1, 2, 4, 8\}$, and high intra-cluster correlation $\rho_m = 0.65$ ($\rho_a = 0.20$). Varying cluster-size skew and $\rho_m$ while comparing normal-quantile vs. $t_{G_{\text{eff}}-1}$ / delete-$1$ cluster-jackknife SEs will separate finite-sample studentized-ratio anti-conservatism from Monte Carlo noise.

5. **Observed point estimate vs. confidence-bound acceptance rule:**
   - Neither a lower-bound-only rule (which rewards noisy small-$N$ runs where true $p > 0.06$ can still yield $\text{CI}_{\text{lo}} \le 0.05$) nor a low-$N$ point-estimate rule alone is sufficient.
   - A defensible future pre-registered rule pairs **adequate precision ($N_{\text{outer}} \ge 5{,}000$)** with a **joint criterion** (e.g., $\hat{p} \le 0.060$ **and** $\text{CI}_{95,\text{Wilson}}^{\text{lo}} \le 0.050$, or an upper-bound equivalence rule $\text{CI}_{95,\text{Wilson}}^{\text{hi}} \le 0.060$ at $N_{\text{outer}} \ge 10{,}000$).
   - **Governance posture preserved:** Do **not** relax `MAX_EMPIRICAL_TYPE1 = 0.060` or enable `CONFIRMATORY_DECLARATIONS_ENABLED`.

---

## 2. Validation FPR Limitations (`n = 265` Normal Messages)

**Inspected artifacts:** [`data/processed/post_stage_b_bundle_v2/dataset_manifest.json`](data/processed/post_stage_b_bundle_v2/dataset_manifest.json#L424-L431), [`src/bangla_smish/metrics.py`](src/bangla_smish/metrics.py#L18-L80), [`src/bangla_smish/post_stage_b_ensemble.py`](src/bangla_smish/post_stage_b_ensemble.py#L23-L28).

- **Discrete Operating-Point Uncertainty:** `eval_smishdetect_validation` has $708$ total rows ($606$ families), of which **$n = 265$ are `Normal`** (`EXPECTED_SMISHDETECT_VAL_NORMAL = 265`). In `smishing_recall_at_fpr` ([`src/bangla_smish/metrics.py`](src/bangla_smish/metrics.py#L46-L55)), `allowed_fp = int(Decimal("0.01") * 265) = 2`, and `threshold = float(np.nextafter(descending[2], np.inf))`. Each false positive shifts empirical FPR by a discrete step of $1/265 = 0.3774\%$, allowing only $k \in \{0, 1, 2\}$ FPs ($k = 3 \implies 1.1321\% > 1\%$).
- **Exact One-Sided 95% Binomial Upper Confidence Bounds ($n = 265$):**
  Calculated via exact Clopper-Pearson $\text{Beta}^{-1}(0.95;\, k+1,\, n-k)$ (with Wilson one-sided 95% UCB at $z_{0.95} = 1.644854$ shown for comparison):

| Validation FPs ($k / 265$) | Empirical FPR | Clopper-Pearson 95% One-Sided UCB | Wilson 95% One-Sided UCB |
|---:|---:|---:|---:|
| **$k = 0$** | `0.0000%` | **`1.1241%`** (`0.011241`) | `1.0106%` (`0.010106`) |
| **$k = 1$** | `0.3774%` | **`1.7775%`** (`0.017775`) | `1.6735%` (`0.016735`) |
| **$k = 2$** | `0.7547%` | **`2.3566%`** (`0.023566`) | `2.2548%` (`0.022548`) |
| *(Ref: $k = 3$)* | *`1.1321%`* | *`2.8998%` (`0.028998`)* | *`2.7986%` (`0.027986`)* |

- **Empirical Operating-Point Selection vs. Population FPR Guarantee:**
  - Even with **$k = 0 / 265$** FPs, the one-sided 95% upper confidence bound (`1.1241%`) still exceeds `1.0%`. At the typical operating boundary **$k = 2 / 265$** (`0.7547%`), true IID population FPR can reach **`2.36%`** at 95% confidence — and real uncertainty is **wider still** due to validation family clustering ($606$ families across $708$ rows), post-selection optimism from calibrating the threshold on those same $265$ rows, and distribution shift to `eval_smishdetect_test` or `eval_banglaphish_ood`.
  - Accordingly, `calibrate_ensemble_threshold` selects a **frozen empirical operating point**, **not** a population-level $\le 1\%$ FPR guarantee (matching `THRESHOLD_TRANSFER_CAVEAT` in [`src/bangla_smish/post_stage_b_ensemble.py`](src/bangla_smish/post_stage_b_ensemble.py#L24-L28)). The frozen validation cohort must remain unchanged.

---

## 3. Training-Arm Class Counts (Reconciliation)

**Inspected artifacts:** [`data/processed/post_stage_b_bundle_v2/dataset_manifest.json`](data/processed/post_stage_b_bundle_v2/dataset_manifest.json), `data/processed/post_stage_b_bundle_v2/{real_train_pool,synthetic_pilot_300,synthetic_batches_b1_b7}.parquet`, [`src/bangla_smish/post_stage_b_data.py`](src/bangla_smish/post_stage_b_data.py#L107-L140).

### Exact Source × Class × Language Breakdown

| Source Cohort | Class | Bengali (`Bn`) | Banglish (`Bg`) | CodeMix (`CM`) | English (`En`) | Total |
|---|---|---:|---:|---:|---:|---:|
| `real_train_pool.parquet` (`3,999`, `3,803` fams) | `Normal` | 396 | 417 | 312 | 384 | **1,509** |
| | `Promotional` | 229 | 141 | 234 | 338 | **942** |
| | `Smishing` | 371 | 404 | 357 | 416 | **1,548** |
| **Real Total** | | **996** | **962** | **903** | **1,138** | **3,999** |
| `synthetic_pilot_300.parquet` (`300`, `292` fams) | `Normal` | 10 | 8 | 14 | 7 | **39** |
| | `Promotional` | 65 | 115 | 60 | 21 | **261** |
| | `Smishing` | 0 | 0 | 0 | 0 | **0** |
| **Pilot Total** | | **75** | **123** | **74** | **28** | **300** |
| `synthetic_batches_b1_b7.parquet` (`345`, `345` fams) | `Normal` | 0 | 0 | 0 | 0 | **0** |
| | `Promotional` | 93 | 131 | 93 | 28 | **345** |
| | `Smishing` | 0 | 0 | 0 | 0 | **0** |
| **B1–B7 Total** | | **93** | **131** | **93** | **28** | **345** |

### Resulting Training-Arm Pools & How Arm D Achieves Balance

| Arm | Composition | `Normal` (Bn/Bg/CM/En) | `Promotional` (Bn/Bg/CM/En) | `Smishing` (Bn/Bg/CM/En) | Full Pool Total |
|---|---|---:|---:|---:|---:|
| **Arm A** | `real_train_pool` (unweighted) | **1,509** (396/417/312/384) | **942** (229/141/234/338) | **1,548** (371/404/357/416) | **3,999** |
| **Arm B** | `real_train_pool` + inverse-freq `class_weight` | **1,509** (396/417/312/384) | **942** (229/141/234/338) | **1,548** (371/404/357/416) | **3,999** |
| **Arm C** | `real_train_pool + B1–B7` | **1,509** (396/417/312/384) | **1,287** (322/272/327/366) | **1,548** (371/404/357/416) | **4,344** |
| **Arm D** | `real_train_pool + pilot_300 + B1–B7` | **1,548** (406/425/326/391) | **1,548** (**387/387/387/387**) | **1,548** (371/404/357/416) | **4,644** |

- **Exact Balancing Mechanism:** Target per-class ceiling is $T = \max_c n_c(\text{real}) = 1{,}548$ (`Smishing`), creating real deficits of **`+39` `Normal`** and **`+606` `Promotional`** (`+645` total).
  1. `synthetic_pilot_300` contributes **`39` `Normal`** (closing `Normal`: $1{,}509 + 39 = 1{,}548$) plus **`261` `Promotional`** (reducing `Promotional` deficit from `606` to `345`).
  2. `synthetic_batches_b1_b7` contributes **`345` `Promotional`** ($261 + 345 = 606$), bringing `Promotional` to $942 + 606 = 1{,}548$ and balancing each Promotional language to **exactly `387`**: Bn ($229+65+93$), Bg ($141+115+131$), CM ($234+60+93$), En ($338+21+28$).
  3. **Important Pool vs. Fold Distinction:** `1,548/class` and `387 Promo/lang` hold in the **full Arm D pool (`4,644`)**, whereas `Arm C` (`4,344`) remains unbalanced (`1,509 / 1,287 / 1,548`) because `pilot_300` is excluded. Inside each CV fold (`build_experiment_fold_data`, [`src/bangla_smish/post_stage_b_data.py`](src/bangla_smish/post_stage_b_data.py#L785-L847)), $1/5$ of `real_train_pool` (`800` in F0–F3, `799` in F4) is held out as real-only CV validation while **all synthetic rows go strictly to the fold training set** (making Arm D fold-train `3,844` in F0–F3 and `3,845` in F4).

---

## 4. Equal Optimizer-Step Investigation

**Inspected artifacts:** [`experiments/colab/configs/post_stage_b_stage4_option_a_config.json`](experiments/colab/configs/post_stage_b_stage4_option_a_config.json), [`src/bangla_smish/post_stage_b_runner.py`](src/bangla_smish/post_stage_b_runner.py#L778-L816).

- **Exact Step Formula (`compute_optimizer_step_budget`, [`src/bangla_smish/post_stage_b_runner.py`](src/bangla_smish/post_stage_b_runner.py#L778-L816)):**
  Under Option A (`num_train_epochs = 3`, `batch_size = 32`, `gradient_accumulation_steps = 1`):
  $$\text{steps/epoch} = \left\lceil \frac{n_{\text{train}}}{32} \right\rceil, \qquad \text{expected\_optimizer\_steps} = 3 \times \left\lceil \frac{\text{steps/epoch}}{1} \right\rceil.$$

| Arm | Folds 0–3 (`val = 800`) $n_{\text{train}}$ | Fold 4 (`val = 799`) $n_{\text{train}}$ | Steps / Epoch | Total Expected Steps (3 ep) | Ratio vs. Arm A |
|---|---:|---:|---:|---:|---:|
| **A & B** | `3,199` | `3,200` | $\lceil 3199/32 \rceil = 100$ / $\lceil 3200/32 \rceil = 100$ | **`300`** | `1.0000` (`+0.0%`) |
| **C** | `3,544` (`+345`) | `3,545` (`+345`) | $\lceil 3544/32 \rceil = 111$ / $\lceil 3545/32 \rceil = 111$ | **`333`** (`+33`) | `1.1100` (**`+11.0%`**) |
| **D** | `3,844` (`+645`) | `3,845` (`+645`) | $\lceil 3844/32 \rceil = 121$ / $\lceil 3845/32 \rceil = 121$ | **`363`** (`+63`) | `1.2100` (**`+21.0%`**) |

- **Estimand & Causal Attribution Implications:**
  - Option A estimates **synthetic augmentation under an equal-epoch (`epochs = 3`) protocol**, **not** under an equal optimization budget.
  - Because `C` performs $+11.0\%$ ($+33$) and `D` performs $+21.0\%$ ($+63$) more gradient updates — while also stretching linear LR warmup/decay over `333`/`363` vs `300` total steps (and `transformers==4.57.1` advances `state.global_step` even when FP16 `GradScaler` skips an overflow step) — **any gain in C/D vs A/B confounds synthetic data/class-balance effects with optimization-budget and LR-schedule effects**.
- **Scientifically Defensible Future Alternative (No Config Changes Now):**
  - Keep Option A untouched for pre-registered consistency; in a post-Option-A exploratory sensitivity ablation, add a **step-matched control** (`max_steps = 300` on C/D, or `max_steps = 363` on A/B) to separate data-composition effects from step-budget effects.

---

## 5. Residual Leakage and Data Coupling

**Inspected artifacts:** [`reports/post_stage_b_stage3_readiness_review.md`](reports/post_stage_b_stage3_readiness_review.md#L11-L71), [`reports/stage4_frozen_test_masks.json`](reports/stage4_frozen_test_masks.json), [`src/bangla_smish/post_stage_b_cohorts.py`](src/bangla_smish/post_stage_b_cohorts.py), [`src/bangla_smish/curation.py`](src/bangla_smish/curation.py#L99-L133).

### Reconciliation of Zero-Overlap Claims vs. the 32 Near-Duplicate Pairs
1. **Why `0` overlap and `32` pairs are simultaneously true:**
   - `audit_post_stage_b_bundle` guarantees **strict `0` overlap** on `row_id`, `family_id`, raw `text`, and `strong_normalize_text` across all $15$ cohort pairs, `max(folds per family) == 1`, and `0/280` exact model-input collisions in `raw`+`canon_masked`.
   - Separately, a 12,466-row cross-boundary `char_wb` `(3, 5)` TF-IDF cosine screen ($\ge 0.85$) over `strong_normalize_text` and `slot_template_text` found **32 unique cross-boundary pairs** (`0` exact duplicates; `0` involving `pilot_300`, `B1–B7`, or `banglaphish_ood`; all from upstream `smishdetect_2026` / `bangalabarta_2025` / `legacy_2024`). Three borderline `strong_normalize_text` pairs sat at `0.8485–0.8498` during Stage A (`11,821` rows) and crossed `0.85` (`0.8501–0.8528`) when `645` Stage B rows shifted IDF weights; Stage A's `slot_template_key` required exact punctuation-stripped equality.
2. **Exact Breakdown of the 32 Pairs (`11` Ambiguous + `21` Template-Slot):**
   - **Adjudication labels:** **`11` *Ambiguous; requires human review*** (`P01, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13`); **`21` *Semantically related but distinct*** (`P02, P03, P14–P32`).
   - **Boundary breakdown:** `real_train <-> test = 14` (`P01, P02, P08, P09, P11, P16, P17, P19, P25, P27–P31`); `real_train <-> real_train` cross-fold `= 8` (`P03, P04, P07, P12–P15, P21`); `real_train <-> val = 6` (`P05, P06, P18, P22–P24`); `val <-> test = 4` (`P10, P20, P26, P32`).
   - **Exact Affected Cohorts, Row Counts, and `family_id`s:**
     - **`real_train_pool`** (`28` hit rows across **`26` families**, `34` total family rows): `fam_05dae3504f`, `fam_1a97c102c8`, `fam_2b6b4a2df2`, `fam_32d08e0523`, `fam_3d1e583fac`, `fam_3ed27fec7c`, `fam_52d65f6f6a`, `fam_533f4c0a04`, `fam_63aa660615`, `fam_7b3c36fa90`, `fam_7ff22a7f78`, `fam_8137a4bd83`, `fam_84a82c3269`, `fam_8538df3808`, `fam_8866d54624`, `fam_976d8e1388`, `fam_a4a1f83853`, `fam_a7ddf5a275`, `fam_ab0f20e0e6`, `fam_baf93eb49b`, `fam_bf5ea8e23d`, `fam_d7e75c9326`, `fam_ed9a7b3fad`, `fam_ee1767057b`, `fam_f456015736`, `fam_fcf852e8b3`.
     - **`eval_smishdetect_validation`** (`7` hit rows across **`6` families**, `10` total family rows): `fam_0a130203db` (P32), `fam_358d797117` (P22, P23), `fam_9186ea6097` (P18, P20), `fam_be13e514ff` (P24, P26), `fam_d04705ccba` (P10), `fam_df9b72ac6f` (P05, P06).
     - **`eval_smishdetect_test`** (`15` hit rows across **`12` families**, `23` total family rows):
       - *5 ambiguous test families (`11` Smishing rows, excluded in `primary_family_decontam_1687`):* `fam_735d8c7d90` [n=2, P01], `fam_210d66c8f3` [n=3, P08], `fam_77db01305f` [n=1, P09], `fam_b7af3c4ecc` [n=1, P10], `fam_03f840170c` [n=4, P11] (`5` hit rows excluded in `sensitivity_row_only_1693`).
       - *7 template-slot distinct test families (`12` rows, additionally excluded in `sensitivity_template_stress_1675`):* `fam_194ea98c59` [n=3, P32], `fam_3ae7bc14a0` [n=1, P29], `fam_4bb67afb14` [n=2, P27, P28], `fam_5408bae462` [n=1, P02], `fam_c025d51b1d` [n=2, P16, P17, P25, P26], `fam_d992d2b1c6` [n=1, P19, P20], `fam_e396320454` [n=2, P31].
3. **Resolved vs. Remaining Risks:**
   - **Resolved at test evaluation:** `primary_family_decontam_1687` removes all `5` ambiguous test families (`11` rows), and `sensitivity_template_stress_1675` removes all `12` hit test families (`23` rows).
   - **Remaining (post-hoc masking cannot undo):** (i) `8` CV cross-fold pairs (`P03, P04, P07, P12–P15, P21`) and `6` `real_train <-> val` pairs (`P05, P06, P18, P22–P24`) remain active during CV epoch selection (`eval_loss`) and `eval_smishdetect_validation` threshold calibration; (ii) `3` `val <-> test` template-slot pairs (`P20, P26, P32`) remain in `1,687` (removed only in `1,675`); and (iii) `21` template-slot pairs remain outside `1,675` as shared-scenario stress cases.

---

## 6. Counterfactual Evaluation (`98.59%` No-Op of `contact_swap`)

**Inspected artifacts:** [`src/bangla_smish/post_stage_b_data.py`](src/bangla_smish/post_stage_b_data.py#L326-L372), [`src/bangla_smish/schema.py`](src/bangla_smish/schema.py#L20-L37,L297-L309), [`src/bangla_smish/post_stage_b_runner.py`](src/bangla_smish/post_stage_b_runner.py#L1362-L1377), `data/processed/post_stage_b_bundle_v2/counterfactual_smish_probe.parquet` ($N = 1{,}279$).

- **Why `contact_swap` is a `98.59%` (`1,261 / 1,279`) no-op in `canon_masked`:**
  - `_build_counterfactual_probe_df` replaces URLs with `"https://example.com/offer"` and phones with `"01700-000011"` in `raw` text, and then runs `mask_text_views()`. `mask_shortcuts()` converts both the original URLs/phones and the counterfactual `"https://example.com/offer"` / `"01700-000011"` back into identical `<URL>` and `<PHONE>` placeholders.
- **What causes the remaining `18 / 1,279` (`1.41%`) differences in `canon_masked`:**
  - Direct inspection of `counterfactual_smish_probe.parquet` confirms **all `18/18` differing rows** (`17` Banglish, `1` English) differ **solely by a single trailing period (`<URL>.` vs. `<URL>`)**: `_URL_RE` ([`src/bangla_smish/schema.py`](src/bangla_smish/schema.py#L20-L24)) excludes trailing punctuation on bare-domain URLs (leaving `<URL>.`) whereas replacing them with `https://example.com/offer` causes `_URL_RE` to consume the trailing `.` into `<URL>`. Thus `contact_swap` has **zero genuine semantic variation in `canon_masked`** (`98.59%` exact no-op + `1.41%` trailing-dot artifact).
- **Active Transformation Rates Across Views ($N = 1{,}279$):**
  - **`raw`:** `contact_swap` is active on **`1,279/1,279` (`100%`)**; `contact_brand_swap` vs `contact_swap` is active on **`238/1,279` (`18.61%`)**.
  - **`canon_masked`:** `contact_swap` is active on **`18/1,279` (`1.41%`, dot-artifact only)**; `contact_brand_swap` vs `contact_swap` is active on **`238/1,279` (`18.61%`)** (`orig` vs `contact_brand_swap` differs on `249` rows = `238` brand swaps + `11` dot-diffs).
  - **`brand_masked` / `brand_stripped`:** `0/1,279` (`0%`) active brand swaps.
- **Evaluation & Reporting Policy:**
  - `_compute_unit_metrics` ([`src/bangla_smish/post_stage_b_runner.py`](src/bangla_smish/post_stage_b_runner.py#L1362-L1377)) currently aggregates counterfactual flip/delta rates across all `1,279` rows, diluting `canon_masked` brand-swap sensitivity by `5.4×`.
  - Future paper tables should report: (1) exact no-op rate (`98.59%`), regex-artifact rate (`1.41%`), active transformation rate (`18.61%` = `238` rows), and semantic-validity rate; (2) **`contact_swap` strictly on `raw`** (`1,279/1,279`); and (3) **`contact_brand_swap` vs `contact_swap` stratified on the `238` active brand-bearing rows** alongside full-cohort totals.

---

## 7. Synthetic Augmentation Interpretation (`0.9560` TF-IDF AUC & `pilot_300` `0/8` Gates)

**Inspected artifacts:** [`src/bangla_smish/stage_b_pilot.py`](src/bangla_smish/stage_b_pilot.py#L303-L632), [`src/bangla_smish/codemix_diversity.py`](src/bangla_smish/codemix_diversity.py#L104-L114,L528-L585), [`reports/stage_b_batch7_45_report.md`](reports/stage_b_batch7_45_report.md#L152-L162,L207-L244), [`data/generated/curation_v2/batch7_45/combined_cohorts_diagnostics.json`](data/generated/curation_v2/batch7_45/combined_cohorts_diagnostics.json).

- **What `0.9560` TF-IDF AUC Measures Exactly:**
  - It is a **synthetic-versus-real source separability diagnostic restricted to the `Promotional` class under stratum-matched conditioning** (NOT `Promotional`-vs-other-classes).
  - **Dataset & Labels:** `combined_new_345` (`B1–B7`, `345` synthetic `Promotional` rows, `is_synthetic_bin = 1`) vs. `real_train_pool` `Promotional` (`942` real rows, `is_synthetic_bin = 0`), restricted to **41 shared 3-way `(language, domain_eff, length_band_eff)` strata** (`333/345 = 96.52%` synthetic; `887/942` real) with **inverse-stratum-count sample weights** (`strata_mode="3way"` in `run_synthetic_origin_diagnostics`).
  - **Classifier & View:** `_make_tfidf_lr()` (`word` 1–2 `min_df=2` + `char_wb` 3–5 `min_df=3`, `sublinear_tf=True`, `LogisticRegression(C=2.0, class_weight="balanced")`) evaluated on **`brand_masked`**.
  - **Splitting & Metric:** 5-fold `StratifiedGroupKFold(shuffle=True)` grouped by `family_id` across **10 seeds (`42..51`)** → **`0.9560 ± 0.0021`** mean ± SD of unweighted OOF ROC-AUC (`seed=42` OOF AUC = `0.9550`; mean per-fold `mean_oof` = `0.9609`).
  - **Comparative `seed=42` 3-way Promo AUCs (`combined_cohorts_diagnostics.json`):** `combined_new_345` (`raw 0.9833`, `canon_masked 0.9618`, `brand_masked 0.9550`, `brand_stripped 0.9549`, `surface_8feat 0.5062`); `pilot_300` (`71.26%` coverage: `raw 0.9904`, `canon 0.9832`, `brand_masked 0.9716` [10-seed `0.9688 ± 0.0025`], `brand_stripped 0.9410`); `full_combined_645` (`85.64%` coverage: `raw 0.9801`, `canon 0.9537`, `brand_masked 0.9495` [10-seed `0.9508 ± 0.0026`]).
- **Implications of `pilot_300` Failing `0/8` Diversity Gates ([`src/bangla_smish/codemix_diversity.py`](src/bangla_smish/codemix_diversity.py#L104-L114,L528-L585)):**
  - `pilot_300` (`audit` mode) fails **all 8** structural/stylistic gates: `banglish_latin_rate` (`0.5833 < 0.80`), `hook_plus_pure_english_body_rate` (`0.8500 > 0.15`), `intra_sentential_codemix_rate` (`0.1333 < 0.60`), `single_sentence_exclamation_rate` (`0.5632 > 0.35`), `single_sentence_question_rate` (`0.0920 > 0.05`), `max_fictional_brand_count` (`25 > 5`), `brandless_rate` (`0.0192 < 0.30`), and `ussd_1234_rate` (`1.0000 > 0.25`). By contrast, `combined_new_345` (`B1–B7`, `new_batch`) passes **`8/8`**, and `full_combined_645` (`audit`) passes `6/8` (failing `hook+pure_en = 0.3333` and `max_brand = 25`).
  - **Scientific consequence:** Arm **C** (`+B1–B7`) tests clean `8/8`-gate augmentation, whereas Arm **D** (`+pilot_300 + B1–B7`) closes class deficits (`1,548/class`) at the cost of injecting `pilot_300`'s `0/8`-gate artifacts. Both `B1–B7` and `pilot_300` retain `~0.95–0.97` lexical separability even after brand/shortcut masking, underscoring why `C vs D` and `C/D vs B` must be evaluated on real held-out/OOD cohorts rather than assumed beneficial.

---

## 8. CheckList Behavioural Test Validity

**Inspected artifacts:** [`data/DATASET_GUIDE.md`](data/DATASET_GUIDE.md#L89-L174), [`src/bangla_smish/schema.py`](src/bangla_smish/schema.py#L18-L134,L538), [`src/bangla_smish/splits.py`](src/bangla_smish/splits.py#L57-L70), [`src/bangla_smish/post_stage_b_ensemble.py`](src/bangla_smish/post_stage_b_ensemble.py#L104-L158).

1. **Programmatic Generation vs. Bangla-Native Human Annotation:**
   - **Valid for programmatic generation (no new human labels):** Slot/magnitude `INV` tests inside previously validated templates (4–6 digit OTP swaps, durations, city/person names, discount percentages, explicit emphasis `"!"`↔`"!!!"`, verified closed Banglish spelling pairs, and non-conjunct NFC↔NFD / `০-৯`↔`0-9` on `raw`).
   - **Requires Bangla-native human annotation/review:** All newly authored **MFT** and **DIR** templates in Categories **1–5** (`OTP`, `Credential`, `Suspension`, `Rewards`, `Urgency`), naturalistic **URL obfuscations** (Category **6**), **ZWJ/ZWNJ conjunct legibility** (Category **7**), and **polysemous Banglish shorthand** (Category **8**, e.g., `tk`, `acc`, `koren`/`korun`, `bal`/`val`).
2. **Potentially Invalid Invariance (`INV`) Assumptions:**
   - **Brand swaps on `raw` (`bKash`↔`Nagad` or `bKash`↔`PurbachalBank`):** Invalid on `raw` because real MFS brands carry empirical class priors in `real_train_pool`, while fictional Stage B brands (`schema.py:50–134`) appear exclusively in `Normal`/`Promotional` synthetic rows (`0` Smishing).
   - **Canonical URL ↔ URL-shortener/obfuscated URL (`raw` vs `canon_masked`):** Invalid INV on `raw` (shorteners/suspicious TLDs carry genuine scam signal). Moreover, verified `.venv/bin/python` checks on `schema.py:18–27` show `_URL_RE` outputs `<URL>[.]xyz/login` for `[.]`, leaks `hxxps://<URL>`, misses `bit[.]ly/3xFake` and `example(dot)com/verify`, and turns ZWNJ-in-URL (`https://bk\u200cash-help.xyz/pin`) into a **100% no-op** (`<URL>`) in `canon_masked`.
   - **Blind ZWJ/ZWNJ & NFC↔NFD (`splits.py:67-69`, `schema.py:538`):** Inserting `\u200c`/`\u200d` blindly between adjacent `\u0980-\u09ff` codepoints splits Bengali conjuncts/vowel-signs (`ক‌া`), breaking legibility; NFD↔NFC swaps are strict **no-ops** on `canon_masked` because `mask_text_views` applies `NFC`.
   - **`_PHONE_RE` symmetric over-masking (`schema.py:34`):** Verified via `.venv/bin/python`: because `_PHONE_RE` (`\b\d{10,13}\b`) does not pass `re.ASCII`, Python 3 matches both ASCII and Bengali digits `০-৯` **symmetrically** (`"TrxID 1234567890 vs ১২৩৪৫৬৭৮৯০"` → `"TrxID <PHONE> vs <PHONE>"`), meaning non-phone 10–13 digit IDs/TrxIDs are over-masked in both scripts, and digit-slot `INV` swaps collapse to no-ops in `canon_masked`.
   - **Banglish polysemy (`tk`, `acc`, `code`/`kod`):** `tk` interacts with `_PHONE_RE` currency negative-lookbehinds (`schema.py:32-33`) and shifts `CodeMix` vs `Banglish` script cues.
3. **Potentially Ambiguous MFT Labels:**
   - Official helpline `16216` (`Normal`) vs personal scammer `017...` (`Smishing`) both collapse to `<PHONE>` in `canon_masked`, making helpline-only MFTs unresolvable in `canon_masked` (`DATASET_GUIDE.md:106`).
   - Aggressive cashback/reward promos (`Promotional`) vs fee/PIN lottery scams (`Smishing`) without explicit extraction mechanisms.
   - Legitimate urgent fraud/KYC advisories (`Normal`) vs coercive suspension/KYC phishing (`Smishing`), and bare `"আপনার OTP {OTP}"` without explicit sharing vs non-sharing clauses (`DATASET_GUIDE.md:105`).
4. **Contamination & Threshold-Tuning Governance:**
   - Store CheckList suites outside `data/generated/` and `data/processed/` (e.g., read-only `eval_suites/behavioral_checklist/` with independent SHA-256 manifest); screen all CheckList seeds against `real_train_pool`, `synthetic_*`, `eval_smishdetect_*`, and `eval_banglaphish_ood` via `strong_normalize_text` + `slot_template_text` + `char_wb` 3–5 TF-IDF $\ge 0.85$.
   - **Never** use CheckList pass/fail rates for early stopping, checkpoint selection, or calibrating `frozen_threshold` (`eval_smishdetect_validation` `265` Normal rows). Report CheckList strictly as post-hoc diagnostic cluster-level failure rates under both argmax and the frozen validation threshold alongside view-specific no-op rates.

---

## 9. Repository and Documentation Corrections

### A. `microsoft/onnxruntime` GitHub HTTP 404 & Android 15 NNAPI Reconciliation
- **Live HTTP Verification (`2026-11-25T08:30:01Z–08:30:03Z`):**
  - `https://github.com/microsoft/onnxruntime` → **`HTTP/2 404`**; `https://github.com/microsoft/onnxruntime-extensions` → **`HTTP/2 404`**; `https://raw.githubusercontent.com/microsoft/onnxruntime/main/{LICENSE,README.md}` → **`HTTP 404`**; `https://api.github.com/repos/microsoft/onnxruntime` → **`HTTP 404 Not Found`** (prior to unauthenticated rate-limit `403`).
  - Meanwhile, `https://github.com/microsoft` (`HTTP 200`), `https://pypi.org/pypi/onnxruntime/json` (`HTTP/2 200`, `onnxruntime==1.32.0`, `MIT`), `https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html` (`HTTP/2 200`, `last-modified: Mon, 16 Nov 2026 14:30:04 GMT`), and `https://developer.android.com/ndk/guides/neuralnetworks` (`HTTP/2 200`, `last-modified: Wed, 28 Oct 2026 15:04:23 GMT`) are live.
  - **Reconciliation:** The HTTP 404 status is **real and confirmed live on 2026-11-25** (matching November 2026 reports of the GitHub repositories being unavailable/offline while PyPI/Maven artifacts and `onnxruntime.ai` documentation remain active). Future Phase 4 builds must pin immutable PyPI/Maven packages and avoid `git+https://github.com/microsoft/onnxruntime*.git`.
- **Android 15 NNAPI Deprecation vs. ORT NNAPI EP vs. CPU/XNNPACK Baselines:**
  1. **Android 15 NNAPI Deprecation:** Google's Android NDK documentation (`developer.android.com/ndk/guides/neuralnetworks`) states **`"NNAPI is deprecated"`** (deprecated in Android 15 / API level 35), noting Google expects most devices to rely on CPU execution and recommends migrating accelerator workloads to LiteRT/vendor delegates.
  2. **ORT's Continued NNAPI EP Documentation:** `onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html` still documents NNAPI EP (Android 8.1+, 9+ recommended) for backward compatibility on older Android OS versions, though its narrow op support partitions Transformer encoders across NNAPI and CPU.
  3. **CPU / XNNPACK as Future Deployment Baselines:** Aligning with both Android NDK guidance and `onnxruntime.ai/docs/tutorials/mobile/`, **ORT CPU (MLAS) for INT8** and **XNNPACK for FP32** (`intra_op_num_threads=1`, `AllowIntraOpSpinning="0"`) are the primary portable Android baselines; NNAPI or vendor NPU EPs (e.g., QNN) should be evaluated only via physical Android device benchmarks.

### B. Portable Kit Versions (`v3–v7` vs. `v3–v8`)
- `stage4_colab_kit_upload/` holds **6 separate immutable snapshot directories** (`bangla_smish_post_stage_b_stage4_v3` through `v8`, `34` files each, built sequentially from `2026-11-24T12:49Z` to `2026-11-25T05:49Z`).
- **`"v3–v7 untouched"`** means Stage 4 v8 staged a new `v8` directory without altering `v3–v7` (each differs from `v8` in the `10` v8-remediated files). **`"6/6 kits (v3–v8) verified"`** means `verify_portable_colab_kit` ([`src/bangla_smish/post_stage_b_integrity.py`](src/bangla_smish/post_stage_b_integrity.py#L536-L646)) verified each directory's **internal self-consistency** against its *own* `stage4_kit_manifest.json` (`34/34` files, `8/8` v2 checks, `4/4` frozen masks). **Only `v8`** (`9933a190...`, notebook `70f8f15f...`) has `34/34` parity with current repo `HEAD`.

---

## 10. Final Deliverable

### A. Correction & Clarification Matrix

| Issue | Original Audit Finding | Verified Correction / Clarification | Severity | Required Action |
|---|---|---|---|---|
| **1. Type-I Error Gate (`62/1,000`)** | Noted `syn_stress::S1_both_zero` Recall `0.0620 > 0.0600` tripped `CONFIRMATORY_DECLARATIONS_ENABLED = False` | Operational gate fails (`0.0620 > 0.0600`), **not** statistical proof that true rate exceeds `0.06` (exact binomial $p = 0.4135$ vs `0.06`, $p = 0.0511$ vs `0.05`; $N=1{,}000$ CI width $3.00\text{ pp}$ is underpowered) | **High** (Inference framing) | Keep `CONFIRMATORY_DECLARATIONS_ENABLED = False`; if future recalibration is authorized, use $N_{\text{outer}} \ge 5{,}000$ + factorial `syn_stress` ablation |
| **2. Validation FPR (`n = 265` Normal)** | Noted "`265`-row validation threshold transfer variance (`<=2` FPs)" | Quantified exact one-sided 95% Clopper-Pearson UCBs: `0/265` → **`1.1241%`**, `1/265` → **`1.7775%`**, `2/265` → **`2.3566%`**. Even `0` FPs does not guarantee $\le 1\%$ population FPR | **High** (Metric interpretation) | Label threshold strictly as a **frozen empirical operating point**, never a population $\le 1\%$ FPR guarantee |
| **3. Arm D Balancing (`1,548/class`, `387/lang`)** | Stated Arm D pool = `4,644` (`1,548/class`, `387/lang` Promo) without showing `pilot_300` Normal/Promo split | Reconciled exact counts: `pilot_300` has **`39` Normal + `261` Promo**; `B1–B7` has **`345` Promo**; Arm C (`4,344`) remains unbalanced (`1,509/1,287/1,548`); Arm D (`4,644` full pool) reaches `1,548/class`, `387/lang` Promo; fold-train is `3,844`/`3,845` | **Medium** (Reporting clarity) | Report full-pool vs fold-train counts and `pilot_300` Normal (`+39`) contribution explicitly in paper tables |
| **4. Equal Optimizer Steps (`+11%` / `+21%`)** | Reported `A/B 300`, `C 333 (+11%)`, `D 363 (+21%)` steps | Verified exact `ceil(n_train/32)*3` across F0–F4 (`3,199/3,544/3,844` and `3,200/3,545/3,845`); Option A estimates **equal-epoch** augmentation, confounding data effects with $+11\%$/$+21\%$ update budget & LR stretching | **High** (Causal attribution) | Frame Option A C/D vs A/B strictly as equal-epoch comparisons; plan optional future `max_steps`-matched sensitivity check |
| **5. Residual Coupling (`32` Pairs)** | Cited "`11` ambiguous + `21` template-slot pairs remain in `BB`/`SD`" | Mapped all `32` pairs (`14` train↔test, `8` CV↔CV, `6` train↔val, `4` val↔test), `26` train / `6` val / `12` test `family_id`s; `1,687` removes `5` ambiguous test fams (`11` rows), `1,675` removes all `12` test fams (`23` rows); `8` CV + `6` train↔val remain active in training/val selection | **Medium** (Leakage precision) | Report both `primary_family_decontam_1687` and `sensitivity_template_stress_1675`, explicitly disclosing CV/val coupling |
| **6. Counterfactual `contact_swap` (`98.59%` No-Op)** | Reported `contact_swap` `98.59%` no-op in `canon_masked` and `238` active `contact_brand_swap` | Verified remaining `18/1,279` (`1.41%`) in `canon_masked` are **100% trailing-period regex artifacts** (`<URL>.` vs `<URL>`), so `contact_swap` has **0%** semantic change in `canon_masked`; `_compute_unit_metrics` dilutes brand swaps over `1,279` vs `238` active rows | **Medium** (Eval metric fidelity) | Restrict `contact_swap` reporting to `raw`; isolate `contact_brand_swap` to the `238` active rows on `canon_masked` + report no-op/artifact rates |
| **7. `0.9560` TF-IDF AUC & `pilot_300` `0/8`** | Summarized as "synthetic Promo `TF-IDF+LR` separability (`0.9560` AUC on `B1–B7`)" | Clarified exact spec: `is_synthetic_bin` (`B1–B7` `345` vs `real` `942` Promo) across `41` shared 3-way strata, `brand_masked`, 10-seed `StratifiedGroupKFold` (`0.9560 ± 0.0021`); enumerated exact `pilot_300` `0/8` vs `B1–B7` `8/8` gates | **Medium** (Method clarity) | State exact 3-way-stratified synthetic-vs-real Promo definition and separate Arm C (`8/8`) vs Arm D (`0/8` pilot) interpretations |
| **8. CheckList Validity & `_PHONE_RE`** | Proposed 8-category MFT/INV/DIR CheckList suite | Classified deterministic INV vs Bangla-native MFT/DIR; flagged invalid `raw` brand/URL INVs, `_URL_RE` obfuscation misses, ZWJ conjunct breakage, NFC↔NFD `canon_masked` no-ops, and symmetric `_PHONE_RE` Unicode digit over-masking (`\b\d{10,13}\b`) | **High** (Probe validity) | Require native-Bangla annotation for MFT/DIR & Banglish lexicons; enforce registry isolation & frozen-threshold decoupling |
| **9. ONNX 404, Android 15 NNAPI & Kits `v3–v8`** | Reported GitHub `onnxruntime` 404, "NNAPI deprecated", and "`v3–v7` untouched" + "`6/6` (`v3–v8`) verified" | Re-confirmed live `HTTP 404` on `github.com/microsoft/onnxruntime*` (`2026-11-25`); distinguished Android 15 NNAPI deprecation vs ORT NNAPI EP legacy docs vs CPU/XNNPACK baselines; clarified `v3–v7` historical vs `v8` HEAD-parity (`34/34`) | **Low–Medium** (Clarity & dep hygiene) | Pin PyPI/Maven ORT artifacts; benchmark CPU (INT8) & XNNPACK (FP32) on real Android; cite `v8` as sole HEAD-parity kit |

---

### B. Revised Synthesis (Deliverables)

1. **Revised Scientific Assessment:**
   Explore-AIs's data-forensic discipline, view-collision isolation (`canon_masked` vs `raw`), frozen mask hierarchy (`1,687` / `1,698` / `1,693` / `1,675`), and cluster-level statistical architecture are materially stronger than the three direct competitors (`Seajon2002`, `ScamShield-AI`, `BartaScope`). However, **the project currently has zero executed GPU/Colab results and cannot yet make confirmatory superiority claims** (`CONFIRMATORY_DECLARATIONS_ENABLED = False`). Moreover, pre-execution clarity requires (i) framing `62/1,000` as an operational gate fail rather than statistical proof of `>6%` Type-I error, (ii) framing the `265`-Normal `<=1%` threshold strictly as an empirical operating point (`2.36%` 95% UCB at `k=2`), (iii) interpreting Option A C/D vs A/B strictly under equal-epoch budgets (`+11%`/`+21%` step confound), and (iv) slicing counterfactual `contact_brand_swap` on its `238` active rows rather than all `1,279`.

2. **Revised P0–P3 Roadmap *(All steps strictly gated on explicit human authorization)*:**
   - **P0 — Unblock Execution & Calibration Gates:** Owner selects dev-only `t4_benchmark` workload option (`B1/B2/B3`), authorizes capped `gpu_smoke_test` to verify live `transformers==4.57.1` downloads, 32-pair live tokenizer gate (`256/256`), and Drive FUSE durability; decide whether to keep confirmatory claims disabled (descriptive + CIs) or authorize a pre-registered $N_{\text{outer}} \ge 5{,}000$ `syn_stress` calibration study.
   - **P1 — Core Stage 4 Option A Matrix:** Upon explicit GO, run `160` equal-epoch units (`A/B/C/D × 4 backbones × canon_masked/raw × 5 folds`), reporting `primary_family_decontam_1687`, `sensitivity_template_stress_1675`, `banglaphish_ood_sms`, `contact_swap` (`raw` only), and active-subset `contact_brand_swap` (`238` rows) with explicit equal-epoch caveat.
   - **P2 — Submission Robustness & Confounder Controls:** Add multi-seed validation ($\ge 3$ seeds) and step-matched (`max_steps = 300` or `363`) exploratory checks on the top backbone/arms; run native-reviewed CheckList 8-category behavioral suite + Bengali-safe ZWJ/Banglish stress tests (with `eval_suites/` isolation & zero threshold tuning), `source_detector`, and training-pool-only Cleanlab audit (`real_train_pool`, `3,999`).
   - **P3 — Future H2 On-Device Android & SLM Paper:** Export single/distilled student (`keep_model_weights != "none"` in a dedicated run) → FP32 ONNX parity (`< 1e-4`) → INT8 QDQ (`qdq_loss_debug`) + `smishdetect_validation` threshold recalibration & Smishing→Normal FN audit → Android tokenizer/`normalizer` parity (`0/N` mismatches) → physical Android SoC benchmarking on **ORT CPU (INT8)** and **XNNPACK (FP32)** (using pinned PyPI/Maven packages, not GitHub source URLs) + optional future `GroupDRO` & selective 4-bit SLM explanation study.

3. **Exact Unresolved Owner Decisions:**
   1. Disposition of the frozen `t4_benchmark` config (`benchmark_workload_fixture_eval: true` → Option `B1`, `B2`, or `B3`) and separate `t4_benchmark` / `gpu_smoke_test` authorization.
   2. Whether to proceed with Option A framing results as **exploratory effects + 95% cluster-bootstrap CIs** (`CONFIRMATORY_DECLARATIONS_ENABLED = False`) or authorize a separate $N_{\text{outer}} \ge 5{,}000$ calibration audit first.
   3. Explicit GO/NO-GO authorization for the 160-unit Option A Colab GPU run (and any later multi-seed / weight-retention follow-up).
   4. Approval of human Bangla-native review protocol before authoring MFT/DIR CheckList behavioral probes.

4. **Conclusions That Remain Valid:**
   - Canonical Stage A curation (`11,821` unique / `10,048` families), Stage B permanent stop (`645` accepted rows), and frozen `v2` bundle (`308d1c88...`) + 4 frozen test masks (`1,687` primary / `1,698` / `1,693` / `1,675`) remain intact.
   - `csebuetnlp/normalizer@d405944d...` isolated to `csebuetnlp/banglabert@9ce791f3...` matches official defaults cleanly (`apply_unicode_norm_last=True`), differing only in `use_fast=True`, `max_length=128`, and `pad_to_multiple_of=8` omission.
   - Explore-AIs's family-cluster studentized bootstrap (`B_pval = 50,000`) + separate Holm control should **not** be replaced by `compare-mt` or `nlp-stat-test`.
   - `TextFlint` (`GPL-3.0`, En/Cn-only), `TextAttack` `Trainer`/unmodified `words_from_text()`, `CleanLearning`'s `StratifiedKFold`, and `Seajon2002` remain rejected.
   - Cross-competitor F1 scores remain non-comparable; `smishdetect_hf` overlap across Explore-AIs, ScamShield, and BartaScope is verified; BartaScope's INT8 FN-inflation risk (`4→13`) remains a mandatory safety check for future Phase 4 quantization.

5. **Conclusions That Must Be Withdrawn or Qualified:**
   - **Qualified:** "`syn_stress` failed Type-I calibration at `62/1,000`" must be qualified as an **operational gate failure (`0.0620 > 0.0600`) under an underpowered `N=1,000` run**, not statistical proof that true Type-I error exceeds `0.06` ($p = 0.4135$).
   - **Qualified:** Any implication that selecting a `<=1%` empirical FPR threshold on `265` validation Normal rows guarantees `<=1%` population FPR must be withdrawn (`k=2` implies `2.36%` one-sided 95% UCB; even `k=0` gives `1.12%`).
   - **Qualified:** Any attribution of Option A Arm C/D differences solely to synthetic data/class balance must be qualified by the **equal-epoch (`+11.0%` / `+21.0%` optimizer steps)** confound.
   - **Qualified:** `contact_swap` in `canon_masked` is **100% non-semantic** (`98.59%` exact no-op + `1.41%` trailing-period regex artifact), and `contact_brand_swap` must be qualified by its `18.61%` (`238/1,279`) active rate.
   - **Withdrawn (from #28 prior to #31):** Any claim that `_PHONE_RE` (`\b\d{10,13}\b`) fails to match 10–13 Bengali digits (`০-৯`) is withdrawn; Python 3 matches both scripts **symmetrically** to `<PHONE>`.
   - **Qualified:** "`6/6` kits (`v3–v8`) verified" means internal self-manifest integrity per snapshot; only **`v8`** matches current repository `HEAD`. And `microsoft/onnxruntime` GitHub `404` applies to GitHub source endpoints on `2026-11-25`, while PyPI/Maven/docs remain active and NNAPI is deprecated in Android 15.
