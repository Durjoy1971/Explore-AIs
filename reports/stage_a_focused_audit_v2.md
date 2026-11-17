# Stage A Focused Audit Report (`curation_v2`)

Generated reproducibly by `scripts/curate_stage_a_v2.py` (`src/bangla_smish/curation_v2.py`) on top of read-only `data/processed/curation_v1/` (zero synthetic rows generated; zero held-out/OOD rows modified).

## 1. Strict Audit Scope & Isolation Guarantees

- **Audited scope:** `curation_v1/eligible_train.parquet` (`train_pool` only, **4125** rows).
- **Held-out & OOD preservation:** `banglaphish_ood` (**5416**), `smishdetect_test` (**1698**), and `smishdetect_validation` (**708**) remain 100% identical to `curation_v1` in texts, labels, languages, and counts.
- **`curation_v1` & original corpora untouched:** All outputs are written exclusively to `data/processed/curation_v2/`.

## 2. Publication-Grade Decision Boundaries & Language Rules

### Label Boundaries (`Normal` / `Promotional` / `Smishing` / `Quarantine`)
1. **`Normal` vs. `Smishing`:** Transactional/service notices (MyGP PIN/OTP codes, telecom call/balance receipts, scheduled maintenance notices, BTRC handset auto-registration, verified `.edu.bd` university tuition notices) are `Normal` even when urgent or containing URLs/OTPs. Fake emergency/family-danger money requests and credential/fee/prize extraction lures are `Smishing`.
2. **`Normal` vs. `Promotional`:** Commercial discounts, data/voice bundles, paid service offers, identifiable educational admissions (e.g., Comilla City College EIIN:134648), and event/program/contest drives inviting active participation (`Mujib` biopic, Museum Day, WTISD contest) are `Promotional`. Pure institutional holiday/anniversary greetings, national tributes, and non-commercial government/regulatory PSAs (Election Commission, BTRC consultation, e-Passport/BRTA, health/safety) are `Normal`.
3. **Mandatory Quarantine (`audit_v2_*`):** Messages whose intent depends on unavailable sender identity or context — including unattributed account deductions (`1500/- taka deducted...`), bare mobile numbers demanding bKash/Rocket/money without personal context, unverified utility disconnect/car servicing/delivery callbacks, outsourced interview/exam calls without verification or extraction cues, unbranded discount mobile callbacks, garbled recruitment texts, and unverified advance-fee/recruitment URLs — are quarantined rather than guessed.

### Linguistic Classification Rules (`Bengali` / `Banglish` / `English` / `CodeMix`)
1. **URL/Placeholder/Loanword invariance:** URLs, USSD codes, numbers, brand names (`bKash`, `Nagad`, `GP`, `Daraz`), and isolated borrowed telecom/banking loanwords (`OTP`, `PIN`, `MB`, `GB`, `TK`, `taka`, `offer`, `bonus`, `cashback`, `recharge`, `account`, `bank`, `call`, `dial`, `verify`) in a Bengali-matrix sentence do **not** invalidate `Bengali` or `Banglish`.
2. **`Bengali`:** Bengali matrix syntax in Bengali Unicode script (`[\u0980-\u09FF]`), allowing isolated loanwords/URLs.
3. **`Banglish`:** Bengali matrix syntax in Roman/Latin script (`[A-Za-z]`), allowing isolated loanwords/URLs.
4. **`English`:** English matrix syntax in Latin script (including Bangladeshi English terms like `taka`, `TK`, `lakh`, `Eid`, `Puja`).
5. **`CodeMix`:** Productive intra-/inter-sentential switching across Bengali and English/Banglish clauses (in either dual Bengali+Latin script or Latin-script English+Banglish clause switching). Note: Bengali sentences in `SmishDetect` carrying substituted Latin words (`click`, `verify`, `number`, `day`, `Hello`) retain `CodeMix` for parity with the unmodified held-out `SmishDetect` test/validation schema.

## 3. Summary of Applied Audit Decisions (`audit_decisions.parquet`)

- **Total decision rows logged:** **398**
- **Label reclassifications (retained in `eligible_train`):** **82**
- **Newly quarantined ambiguous/unverified groups:** **126**
- **Language reclassifications total:** **178** (**178** retained in `eligible_train`; 0 also quarantined for missing sender context)
- **Boundary exemplars kept with documented note:** **14**

### Label Reclassifications (`eligible_train`)

| Transition | Category | Count |
| :--- | :--- | ---: |
| `Normal -> Smishing` | `emergency_family_impersonation_scam` | 5 |
| `Promotional -> Normal` | `civic_regulatory_psa_notice` | 37 |
| `Promotional -> Normal` | `noncommercial_greeting_or_tribute` | 9 |
| `Promotional -> Normal` | `regulatory_btrc_handset_notice` | 4 |
| `Promotional -> Normal` | `service_maintenance_downtime_notice` | 10 |
| `Promotional -> Normal` | `transactional_call_balance_receipt` | 1 |
| `Promotional -> Normal` | `transactional_call_balance_receipt|script_contradiction_bengali_to_english` | 1 |
| `Promotional -> Normal` | `transactional_call_balance_receipt|script_contradiction_codemix_to_english` | 1 |
| `Promotional -> Normal` | `transactional_pin_otp_notice` | 6 |
| `Smishing -> Normal` | `institutional_independence_day_greeting` | 4 |
| `Smishing -> Promotional` | `identifiable_college_admission_ad` | 4 |

### Newly Quarantined Categories (`quarantined.parquet`)

| Original Class | Category | Groups Quarantined |
| :--- | :--- | ---: |
| `Normal` | `bare_number_money_or_call_demand` | 72 |
| `Normal` | `unattributed_account_deduction_alert` | 5 |
| `Normal` | `unattributed_mfs_account_check` | 3 |
| `Normal` | `unattributed_payment_or_bill_demand` | 17 |
| `Normal` | `unverified_delivery_callback` | 3 |
| `Normal` | `unverified_servicing_callback` | 3 |
| `Normal` | `unverified_utility_disconnect_threat` | 2 |
| `Promotional` | `unverified_advance_fee_or_recruitment_url` | 2 |
| `Smishing` | `unverified_interview_or_unbranded_discount` | 19 |

### Language Reclassifications

| Transition | Category | Count |
| :--- | :--- | ---: |
| `Banglish -> CodeMix` | `latin_script_english_banglish_codemix` | 51 |
| `Banglish -> CodeMix` | `script_contradiction_banglish_to_codemix` | 1 |
| `Banglish -> English` | `pure_english_mislabeled_banglish` | 101 |
| `Bengali -> Banglish` | `script_contradiction_bengali_to_banglish` | 1 |
| `Bengali -> CodeMix` | `multiclause_english_or_banglish_in_bengali` | 5 |
| `Bengali -> English` | `transactional_call_balance_receipt|script_contradiction_bengali_to_english` | 1 |
| `CodeMix -> Banglish` | `script_contradiction_codemix_to_banglish` | 3 |
| `CodeMix -> Bengali` | `script_contradiction_codemix_to_bengali` | 3 |
| `CodeMix -> English` | `script_contradiction_codemix_to_english` | 11 |
| `CodeMix -> English` | `transactional_call_balance_receipt|script_contradiction_codemix_to_english` | 1 |

## 4. `curation_v1` vs. `curation_v2` Counts & Reconciliation

| Dataset View / Class | `curation_v1` | `curation_v2` | Delta |
| :--- | ---: | ---: | ---: |
| `existing_unique.parquet` (total) | 11947 | 11821 | -126 |
| `quarantined.parquet` (unique groups) | 968 | 1094 | +126 |
| `provenance.parquet` (raw rows) | 18357 | 18357 | 0 |
| `eligible_train.parquet` (total) | 4125 | 3999 | -126 |
| `eligible_train` — `Normal` | 1546 | 1509 | -37 |
| `eligible_train` — `Promotional` | 1009 | 942 | -67 |
| `eligible_train` — `Smishing` | 1570 | 1548 | -22 |

### `curation_v2` Eligible Training (`N = 3999`) by Class × Language

| Class | Bengali | Banglish | English | CodeMix | Total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 396 | 417 | 384 | 312 | 1509 |
| `Promotional` | 229 | 141 | 338 | 234 | 942 |
| `Smishing` | 371 | 404 | 416 | 357 | 1548 |

## 5. Recomputed Stage B Deficits (`T - n_c`)

From `curation_v2/eligible_train.parquet` (`N = 3999`), target `T = max(n_Normal, n_Promotional, n_Smishing) = 1548`:

| Class | `curation_v2` Eligible (`n_c`) | Target (`T`) | Stage B Deficit (`max(0, T - n_c)`) |
| :--- | ---: | ---: | ---: |
| `Normal` | 1509 | 1548 | 39 |
| `Promotional` | 942 | 1548 | 606 |
| `Smishing` | 1548 | 1548 | 0 |
| **Total** | **3999** | — | **645** |

## 6. Unresolved Ambiguities & Remaining Audit Notes

- **Civic participation campaigns vs. PSAs:** Public event/film/contest promotions (`Mujib` biopic, Museum Day, WTISD essay contest) are retained in `Promotional` under the active-participation rule, while purely informational PSAs are `Normal`. Downstream ablations can inspect both slices via `audit_decisions.parquet`.
- **`CodeMix` loanword parity with held-out `SmishDetect`:** `SmishDetect` tags some Bengali messages containing isolated English substitutions (`click`, `verify`, `number`, `day`, `Hello`) as `CodeMix` across both train and held-out splits; these were preserved as `CodeMix` in `curation_v2` to avoid a train-vs-eval language-tag shift.
- **Held-out splits unedited:** Held-out `smishdetect_validation` (`708`) and `smishdetect_test` (`1,698`) were not audited or altered to prevent leakage; natural-prevalence evaluation noise remains a documented limitation.
