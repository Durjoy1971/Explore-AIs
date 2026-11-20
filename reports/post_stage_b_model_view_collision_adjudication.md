# Post-Stage B model-view collision adjudication

**Full A/B/C/D remains NO-GO. Recommend Option A for owner approval.** Restrict the proposed confirmatory analysis to `raw` and `canon_masked` once the remaining readiness gates are cleared. Both views passed this exact-input study across all four tokenizer snapshots. The brand views failed. This report changes no experiment policy, configuration, code, or dataset.

## 1. Exact collision inventory

The reference audit found **three distinct unordered pairs, six rows, six singleton families, and 128 repeated cross-boundary occurrences in 560 checks**. Every flagged check contained exactly one matching row pair. All six labels are **Smishing**. The same 128 occurrences were reproduced independently with each of the four tokenizers. Counting the same checks across tokenizers gives 512 model-specific occurrences in 2,240 checks.

The excerpts below are deliberately short. Contact details, destinations, amounts, and the rest of each original message are omitted. Original text was verified through the frozen bundle, curation provenance, and exact equality at the listed source Parquet row. Source offsets are zero-based. Source files are under `data/raw/smishdetect_hf/data/`. The curated bundle cohort is listed separately from the publisher’s source split.

| Pair | Original row ID | Family ID | Curated cohort and fold | Original SmishDetect file and row offset | Privacy-safe source excerpt |
|---|---|---|---|---|---|
| P1 | `cur_43bd66f47b5b` | `fam_af79b35a6a` | Eligible train, fold 3 | `train-00000-of-00001.parquet`, 1915 | “নগদ অ্যাকাউন্টে ত্রুটি দেখা দিয়েছে। …” |
| P1 | `cur_c92eaaa1966b` | `fam_3408a3c08d` | Eligible train, fold 4 | `train-00000-of-00001.parquet`, 4702 | “সোনালী ব্যাংক অ্যাকাউন্টে ত্রুটি …” |
| P2 | `cur_29c93892c828` | `fam_642e019979` | Eligible train, fold 3 | `train-00000-of-00001.parquet`, 2047 | “Error detected in Nagad account. …” |
| P2 | `cur_7d47602782d3` | `fam_414df89d87` | Curated held-out validation | `validation-00000-of-00001.parquet`, 564 | “Error detected in Sonali Bank account. …” |
| P3 | `cur_42a3cdc4a3d8` | `fam_4f26028f71` | Eligible train, fold 3 | `train-00000-of-00001.parquet`, 3559 | “You have won [amount] taka prize from [brand] …” |
| P3 | `cur_cf8256b4cb6e` | `fam_d7a7a2f623` | Curated held-out test | `test-00000-of-00001.parquet`, 177 | “You have won [amount] taka prize from Nagad …” |

Each hash below is SHA256 of the NFC-normalized UTF-8 stored view string before backbone-specific input preparation. In the 560-check scope, the matching training and evaluation strings are byte-equal after NFC. In P1, the original probe and both swap variants have the same view hash as the matching clean training input. Each row in this table crosses a train/evaluation boundary, has identical labels on both sides, and affects all four tokenizer snapshots listed below.

| Pair | View | Normalized transformed-text SHA256 | Boundary and folds | Occurrences of 560 |
|---|---|---|---|---:|
| P1 | `brand_masked` | `58e2b565598f8ec9a622f3f044c437642d951f3f3271151b922bc081c6708ef6` | CV and three probe variants in folds 3 and 4, reciprocal directions | 32 |
| P1 | `brand_stripped` | `601c0f87ecf9d41437fe6db07fdfec124f2833b8b19e7e9397906c5699a42a6c` | CV and three probe variants in folds 3 and 4, reciprocal directions | 32 |
| P2 | `brand_masked` | `247b0cd0d3dac7c3d41aae408e447591628bfec2da93010a97cf62db7b6557a1` | Held-out validation in units for folds 0, 1, 2, 4 | 16 |
| P2 | `brand_stripped` | `3f8af2c4d6f9c5643206b832b9fc66c50b6b02797f976726b079f7740346a3e5` | Held-out validation in units for folds 0, 1, 2, 4 | 16 |
| P3 | `brand_masked` | `fec6d57086c04ef03cc6b806c55d43aaeb8cf71047e50511a2d256f6c50b8cd4` | Held-out test in units for folds 0, 1, 2, 4 | 16 |
| P3 | `brand_stripped` | `f107c2c9fa4b8dadab99dd5d9597228928d467f298fb30b67c5a1a2ec5c8a3f1` | Held-out test in units for folds 0, 1, 2, 4 | 16 |

In P1, fold 3 scores `cur_43bd66f47b5b` while training on `cur_c92eaaa1966b`. Fold 4 reverses those roles. Each of the four surfaces contributes eight hits per brand view, from four arms and two folds. P2 and P3 each contribute 16 hits per view, from four arms and four folds. Their training counterparts are both in fold 3, so fold-3 models do not train on either counterpart. No primary collision involves a Pilot, Batch 1–7, or OOD row.

## 2. Root cause and four-view check

All three pairs differ in their original contact details and recognized brand spans. Canonical masking replaces contacts and applies NFC. That leaves the brand as a distinguishing span in each pair. The brand-masked view substitutes the same `<BRAND>` token for both brands. The stripped view replaces both spans with whitespace and collapses it. This deterministic many-to-one mapping occurs before tokenization. The frozen family map assigns the six source rows separate IDs, so its isolation check cannot detect the equal downstream representations. The pair P1 probe rewrites collapse to the same masked inputs under the existing view rules. [Schema view implementation](../src/bangla_smish/schema.py#L507) · [Probe construction](../src/bangla_smish/post_stage_b_data.py#L326)

I rebuilt the 20 arm/fold partitions in memory and checked seven evaluation surfaces for each of the four views. The surfaces were clean CV, curated held-out validation, test and OOD, and the three probe variants assigned to the CV fold. I compared exact strings and, separately, complete integer `input_ids` tuples produced using the runner’s shared model-specific preparation, special tokens and configured truncation to 128. I joined each equality back to both row IDs and checked the labels. All four tokenizers reproduced the exact same ordered train/evaluation pairs in the same checks as the string comparison. No model weights or scores were used. [Runner input path](../src/bangla_smish/post_stage_b_runner.py#L386) · [String audit entry point](../scripts/audit_post_stage_b_model_views.py)

| Comparison basis | `raw` | `canon_masked` | `brand_masked` | `brand_stripped` | Total |
|---|---:|---:|---:|---:|---:|
| Stored view strings | 0/140 | 0/140 | 64/140 | 64/140 | 128/560 |
| BanglaBERT · `9ce791f330578f50da6bc52b54205166fb5d1c8c` | 0/140 | 0/140 | 64/140 | 64/140 | 128/560 |
| MuRIL · `afd9f36c7923d54e97903922ff1b260d091d202f` | 0/140 | 0/140 | 64/140 | 64/140 | 128/560 |
| XLM-R · `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` | 0/140 | 0/140 | 64/140 | 64/140 | 128/560 |
| Multilingual DistilBERT · `45c032ab32cc946ad88a166f7cb282f58c753c2e` | 0/140 | 0/140 | 64/140 | 64/140 | 128/560 |

Tokenizer assets came from the pinned local snapshots for `csebuetnlp/banglabert`, `google/muril-base-cased`, `xlm-roberta-base`, and `distilbert-base-multilingual-cased`, with Transformers 4.57.1. BanglaBERT used the published normalizer pinned at `d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9`. Within a check, token tuples are compared directly. No cross-tokenizer comparison of token numbers is meaningful. [Tokenizer preparation](../src/bangla_smish/post_stage_b_tokenization.py) · [Base configuration](../experiments/colab/configs/post_stage_b_abcd_config.json)

### Supplementary checks outside the 560

- The three held-out partitions had zero mutual matches across four views, in 12 comparisons for strings and 12 per tokenizer.
- The runner’s adversarial CV view had zero string matches in 80 checks. Its **token** path added a repeat of P1 in fold 4, from `cur_43bd66f47b5b` training to perturbed `cur_c92eaaa1966b` evaluation. Both sides are Smishing. It occurred for all four arms in each brand view on BanglaBERT, MuRIL and DistilBERT. That is four hits per view on each of those tokenizers, eight per affected tokenizer and **24 additional model-specific hits** altogether. XLM-R had zero. Raw and canonical adversarial checks had zero on all four. This introduces **zero new unique pairs** and is counted outside the primary 128. The perturbed string adds U+200C (zero-width non-joiner) relative to the clean training string in both affected views. The three affected tokenizers map the two strings to equal model inputs.

| Supplementary P1 input | Clean training NFC SHA256 | Adversarial evaluation NFC SHA256 |
|---|---|---|
| `brand_masked` | `58e2b565598f8ec9a622f3f044c437642d951f3f3271151b922bc081c6708ef6` | `687e1f457348ae80227ed03649aff6affeb0cad1c3209c346de5c91fb1d5b6f8` |
| `brand_stripped` | `601c0f87ecf9d41437fe6db07fdfec124f2833b8b19e7e9397906c5699a42a6c` | `1b14b0b47fc030db879f1ab632b11f9727fd572b1f8d14725bb3f80c708a139a` |

**Confirmatory judgement for raw and canonical views.** Both pass the exact model-input isolation gate for the audited bundle, revisions, and sequence length. They are eligible candidates for a narrowed confirmatory protocol after independent review of the heuristic near-duplicate candidates, predeclaration of the endpoint and analysis, and the remaining readiness checks. The previous similarity screen reports two real-to-test candidates and one reciprocal real CV candidate at strong-normalized character cosine ≥0.85, plus additional slot-template candidates. Those are a different question from exact-input equality and are outside these counts. [Prior similarity screen](post_stage_b_integrity_and_leakage_report.md#heuristic-near-duplicate-screen) · [Confirmatory selection and evaluation](https://jmlr.org/papers/v11/cawley10a.html)

## 3. Policy comparison

These are projections from read-only, in-memory membership simulations. Each policy concerns the three adjudicated pairs. A later near-duplicate decision or tokenizer change can alter the counts. No policy output was written to a dataset or split.

| Policy | Rows and families directly affected | Training effect | Evaluation effect | Split requirement and A/B/C/D comparability |
|---|---|---|---|---|
| **A. Exclude the two affected views from confirmatory evaluation** | Six rows and six families implicated in the excluded views. Zero rows or families moved or deleted. | Original fit membership retained for both eligible views. Confirmatory grid scope would change from 320 to 160 units. | All 3,999 OOF, 708 held-out validation, 1,698 test, 5,416 OOD, and 1,279 probe memberships retained in scope. No confirmatory estimate for brand removal or masking. | No split regeneration. A/B retain identical ordered train IDs, C/D retain their accepted additions, and all four share the evaluation IDs within each retained view. |
| **B. Group known CV-collision counterparts together** | Two CV rows in two singleton families enter the same grouping component. The minimal relocation moves one row, one family and one probe parent. The two held-out matches involve four other rows and four families. | No source deletion. Move `cur_43bd66f47b5b` from fold 3 to 4, adding it to fold-3 training and removing it from fold-4 training. Each row still appears in four train folds. | One validation ID and one probe parent change their scoring fold. Held-out files and all evaluation IDs stay. The two held-out Smishing inputs still collide with training in four of five units per arm/view. | Requires a new fold map and probe assignment, with the original version retained. A/B/C/D remain paired if they all use the new map. Cross-view pairing by fitted fold requires the same map for every compared view. |
| **C. Exclude matching training rows for affected views** | Four unique eligible training rows in four singleton families excluded in at least one fold. Four evaluation IDs lose exposure to identical training inputs. The two CV IDs also supply the affected probe parents. Six unique source rows and families overall. | 10 training-row/fold exclusions per arm and affected view. No synthetic exclusions and no reassignment. Exact row and family-closed counts coincide on these data. | All validation, test, OOD and probe rows keep their original assignments and order. Changed exposure covers 2/3,999 CV IDs, 1/708 held-out validation ID, 1/1,698 test ID, 0/5,416 OOD IDs, and 2/1,279 probe parents. | No fold regeneration. Requires a separate, versioned exclusion manifest. Apply the same eligibility mask to all four arms within each affected view. Recompute B’s fold weights and report the resulting cohorts. |

The policy simulations repeated the token comparison on all four pinned tokenizers, with 320 checks per policy and tokenizer across the two relevant views and eight scoring surfaces. Adversarial CV was the eighth surface. Counts below are per tokenizer. The three held-out split-to-split comparisons remained clean in both views for every simulation.

| Simulated policy and views | Clean CV and probes | Held-out validation | Held-out test | OOD and adversarial | Total per tokenizer |
|---|---:|---:|---:|---:|---:|
| A · raw and canonical | 0 | 0 | 0 | 0 | 0/320 |
| B · both brand views, one relocation | 0 | 32 | 32 | 0 | 64/320 |
| C · both brand views, family-closed exclusion | 0 | 0 | 0 | 0 | 0/320 |

### A. Exclude brand views from confirmatory analysis

This is the least disruptive option. It retains the frozen membership, the planned A/B/C/D intervention contrast, the benchmark samples, and the initial class-weight comparison. A narrowed four-backbone grid would have 160 planned units and at most 24 planned against-A comparisons across the retained model/view combinations, down from 48. Predeclare the primary view, endpoint and contrast family before seeing scores. I recommend `canon_masked` as the primary view for the shortcut-resistance question and `raw` as a prespecified reference. Report the brand analysis separately only after a representation-safe protocol exists. The cost is that the confirmatory analysis cannot answer how much masking or stripping brand cues changes performance.

### B. Collision-aware grouping

A minimal move from fold 3 to 4 produces validation sizes `800/800/800/799/800` and probe counts `261/256/252/260/250`. The reverse move would give validation sizes `800/800/800/801/798`. Keeping the two dependent inputs together protects CV and preserves training pool size. It also concentrates two equivalent inputs in one validation unit. A joint collision component should be reflected in uncertainty estimation. A global regrouping algorithm might move more rows than this minimal example.

The held-out collision counterparts participate in four train folds each under any ordinary five-fold assignment. Grouping the eligible pool alone leaves **64 repeated held-out hits** per tokenizer across the original check set. Keeping held-out datasets fixed and solving all brand-view boundaries therefore requires an additional training-eligibility rule. Combining this relocation with the held-out part of C would require eight row-fold exclusions per arm/view. [Leakage detection and avoidance](https://doi.org/10.1145/2382577.2382579)

### C. Exclude conflicting training memberships

Use one label-blind union of exact matches across supported tokenizer revisions and both affected views. Close exclusions over the existing training family IDs, and use identical exclusion rules in all four arms. Compare against the unit’s clean and adversarial CV, assigned probe variants, and fixed held-outs in an isolated integrity step. Pass only a frozen exclusion manifest to the training workflow. Held-out text may be inspected for decontamination, so document that the resulting estimand is generalization after benchmark-aware exclusion. Keep label values and model scores out of the exclusion decision.

| Fold | Affected training IDs removed from each brand-view arm | A / B train size | C train size | D train size |
|---:|---|---:|---:|---:|
| 0 | `cur_29c93892c828`, `cur_42a3cdc4a3d8` | 3,197 | 3,542 | 3,842 |
| 1 | `cur_29c93892c828`, `cur_42a3cdc4a3d8` | 3,197 | 3,542 | 3,842 |
| 2 | `cur_29c93892c828`, `cur_42a3cdc4a3d8` | 3,197 | 3,542 | 3,842 |
| 3 | `cur_c92eaaa1966b` | 3,198 | 3,543 | 3,843 |
| 4 | `cur_29c93892c828`, `cur_42a3cdc4a3d8`, `cur_43bd66f47b5b` | 3,197 | 3,542 | 3,842 |

All removed rows are Smishing. The source pool and accepted additions remain 3,999, +345 for C and +645 for D. Across four arms and two views there would be 80 removed training memberships, or 320 when counted for each of four backbones. The current batch size and epoch count imply unchanged planned update counts of 300/300/333/363 per fold. Loss weights and training inputs still change. A comparison of clean raw/canonical fits against these brand-view fits would combine a view intervention with a training-eligibility change. A separately approved matched-subset sensitivity analysis could estimate a view contrast on common training support. Pooled brand-view OOF inference should account for the two equivalent P1 evaluation inputs as a shared dependency even though their row IDs still score in different folds. The direct exposure counts do not bound the effect on model scores.

## 4. Required regression tests before adopting a policy

1. Pin each tokenizer revision, normalizer and sequence cap in run identity. Use privacy-safe fixtures for contact masking, brand replacement and stripping, and an invisible Unicode variant. Verify exact string differences and model-input equality where expected.
2. Assert zero cross-boundary equal token tuples on every approved arm/fold/view and all scored surfaces. Test the adversarial tokenizer-only path explicitly. Re-run the string check as a fast preliminary check while retaining the token check as the readiness criterion.
3. For A, reject inclusion of either brand view in a confirmatory run and verify unchanged ordered fold/held-out IDs, A/B training equivalence, and C/D augmentation membership. Verify the declared scope of the against-A comparison family.
4. For B, verify grouping closure across existing families, exact input matches and probe parents. Require every evaluation and probe ID exactly once and confirm common assignments across arms. The measured held-out failure is a negative test for grouping alone. Verify a new fold fingerprint and rejection of incompatible resume directories.
5. For C, verify exact exclusion IDs and full family closure, fold counts `2/2/2/1/3`, unaffected arms and synthetic cohorts, and byte-preserved held-out files. Prove label-independent eligibility and recheck B’s inverse-frequency class weights after exclusion. Require unchanged ordered evaluation IDs.
6. Keep output manifests and run identities versioned. Verify zero mutation of the original splits, held-out files, accepted artifacts and configuration during adjudication. Keep the model-view guard active until the owner-approved scope passes.

## 5. Owner approval requested

Approve **Option A** for the confirmatory protocol, conditional on clearing the near-duplicate, statistical and Colab gates. If the scientific question requires an interpretable brand-view analysis, approve **Option C** as a separate follow-up with a frozen exclusion manifest and an explicit benchmark-aware estimand. Option B addresses the CV dependence while leaving both fixed held-out overlaps, so it is insufficient by itself. There is no authorization to train or implement any option in this study.

Audit identity used v2 manifest SHA256 `308d1c889840c1ec04e1cd6f6751cccfca0e559f2e12c61e18ff2e54d6fa2922` and full-config SHA256 `e7d3c4fc44ec4578a5f76fb27664014623f5024a948e75ad7b9fb1c59a4b8b72`. The model-view counts and option projections were computed without a model fit. The original 317 input, source, config and prior-report file hashes were checked for preservation.
