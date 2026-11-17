# Stage B Coverage Matrix & Pilot Generation Plan (From `curation_v2`)

Derived strictly from `data/processed/curation_v2/eligible_train.parquet` (`N = 3999`, `3803` families). Zero held-out evaluation or `BanglaPhish-2026` OOD messages were inspected or used. **No synthetic generation has been run.**

## 1. Recomputed Class Quotas (`T - n_c`)

Target `T = max(n_Normal, n_Promotional, n_Smishing) = 1548`:

| Class | Accepted Eligible (`n_c`) | Target (`T`) | Required Accepted New Rows (`max(0, T - n_c)`) |
| :--- | ---: | ---: | ---: |
| `Normal` | 1509 | 1548 | 39 |
| `Promotional` | 942 | 1548 | 606 |
| `Smishing` | 1548 | 1548 | 0 |
| **Total** | **3999** | — | **645** |

## 2. Measured `curation_v2` Training Coverage Matrix

### A. Class × Language (`eligible_train.parquet`)

| Class | Bengali | Banglish | English | CodeMix | Total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Normal` | 396 | 417 | 384 | 312 | 1509 |
| `Promotional` | 229 | 141 | 338 | 234 | 942 |
| `Smishing` | 371 | 404 | 416 | 357 | 1548 |

### B. Class × Character-Length Band

| Class | `short(<50)` | `medium(50-120)` | `long(>120)` |
| :--- | ---: | ---: | ---: |
| `Normal` | 785 | 561 | 163 |
| `Promotional` | 20 | 779 | 143 |
| `Smishing` | 12 | 1059 | 477 |

### C. Class × Domain / Scenario Bucket (Keyword-Inferred)

| Domain / Scenario | Normal | Promotional | Smishing |
| :--- | ---: | ---: | ---: |
| `banking_mfs` | 14 | 89 | 589 |
| `civic_utility_health` | 27 | 15 | 24 |
| `delivery_parcel` | 0 | 66 | 16 |
| `education` | 91 | 4 | 26 |
| `employment` | 28 | 7 | 182 |
| `personal_or_general` | 1322 | 210 | 574 |
| `shopping_ecommerce` | 6 | 270 | 54 |
| `telecom` | 21 | 281 | 83 |

### D. Shortcut Surface Cues (URL, Contact, Urgency, OTP/PIN)

| Class | Has URL (`True`) | Has Any Contact (`URL/Phone/USSD`) | Has Urgency Cue | Has OTP/PIN/Verify Cue |
| :--- | ---: | ---: | ---: | ---: |
| `Normal` | 11 (0.7%) | 22 (1.5%) | 34 (2.3%) | 6 (0.4%) |
| `Promotional` | 257 (27.3%) | 390 (41.4%) | 70 (7.4%) | 0 (0.0%) |
| `Smishing` | 704 (45.5%) | 1059 (68.4%) | 271 (17.5%) | 191 (12.3%) |

## 3. Key Coverage & Shortcut Gaps Identified

1. **Severe `Promotional` `Banglish` deficit:** Only **141** `Promotional` `Banglish` rows remain after removing mislabeled English/CodeMix rows (vs. **417** Normal and **404** Smishing).
2. **`Normal` URL/urgency/OTP sparsity (anti-shortcut gap):** Only **11** `Normal` messages contain a URL, **34** contain urgency cues, and **6** mention OTP/PIN/verify — causing models to equate URLs, urgency, and OTPs with Smishing.
3. **Length-class skew:** `Normal` is heavily `short(<50)` (**785**), whereas `Smishing` (`477` long) and `Promotional` (`779` medium / `143` long) skew longer.
4. **Domain gaps:** `delivery_parcel` and legitimate `banking_mfs` transactional notifications are underrepresented in `Normal`, while `education`, `delivery_parcel`, and `employment` legitimate offers are sparse in `Promotional`.

## 4. Proposed Stage B Pilot Plan (≤ 300 Accepted Rows)

Per `data/DATASET_GUIDE.md`, Stage B starts with a **300-row** reviewable pilot (strictly within measured deficits, `Smishing = 0`):

| Class | Total Deficit | Pilot Allocation | Bengali | Banglish | CodeMix | English | Primary Coverage Targets |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `Normal` | 39 | **39** | 10 | 8 | 14 | 7 | Medium/long (`50-180` chars) legitimate bank/MFS/delivery/university notices with reserved `example.com`/`.edu.bd` links, urgency, and OTP warnings |
| `Promotional` | 606 | **261** | 65 | 115 | 60 | 21 | Short/medium/long non-coercive offers across retail/e-commerce, banking cashback, telecom, delivery, education, and employment; mix link-present & link-absent |
| `Smishing` | 0 | **0** | 0 | 0 | 0 | 0 | No deficit (`n_Smishing = T`) |
| **Total** | **645** | **300** | **75** | **123** | **74** | **28** | — |

### Pilot Execution & Validation Protocol (Before Scaling to Remaining `Promotional` Deficit)
1. **Scenario-only prompting:** Generate from `(class, language, domain, length_band, formality, link_presence, urgency_presence)` specifications using reserved domains (`example.com`, `example.org`) and fictional IDs/numbers (`+88017000000XX`); never paraphrase held-out rows.
2. **Automated gates:** Enforce schema, `is_synthetic=True`, exact `text_norm_strict` uniqueness, char-ngram cosine `< 0.85` against all existing sources + isolated held-out/OOD contamination filter.
3. **Independent review & surface diagnostics:** Human/independent label review of all 300 pilot rows plus surface-only & length-only AUC check before generating the remaining `Promotional` balance (`606 - 261 = 345` rows).
4. **Fold-safe training:** When running 5-fold CV, include only synthetic rows derived from that fold's training partition.
