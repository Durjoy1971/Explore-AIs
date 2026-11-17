check the checkpoint.md and after that do this:

The latest technical updates, regression tests, and read-only dry-run are acknowledged.

I approve proceeding to the next Stage B generation phase, but I want to use a controlled, incremental approach rather than generating all 345 Promotional samples at once.

### Phase 1: Generate the First 50 Promotional Samples

Use `--diversity-mode new_batch` and enforce all approved diversity constraints.

Requirements:

1. At least 30–40% brandless Promotional samples.
2. Use explicit brand and alias metadata for every fictional brand.
3. Maximum 3–5 occurrences per fictional brand.
4. Enforce natural intra-sentential CodeMix and the approved script distribution.
5. Avoid the Bengali/Banglish hook followed by a pure English promotional body pattern.
6. Enforce sentence-ending punctuation constraints.
7. Maintain realistic Bangladeshi SMS language and natural promotional intent.
8. Run exact and near-duplicate checks against all existing protected reference corpora.
9. Perform independent row-by-row review and replace rejected samples.
10. Preserve all existing datasets, the original 300 accepted pilot rows, provenance, and evaluation splits.

### Phase 2: Quality Evaluation

After generating and reviewing the first 50 samples:

- Run all automated validation gates.
- Run the synthetic-origin diagnostics using raw, canonical-masked, brand-masked, and brand-stripped views.
- Run class-matched and stratum-controlled comparisons.
- Evaluate CodeMix linguistic diversity and brandless distribution.
- Compare the new batch against real eligible_train distributions.
- Report all rejected and replacement samples.

Do not merge the new samples into the existing accepted pilot or modify existing artifacts without authorization.

### Phase 3: Stop and Report

After completing the first 50 samples, stop and provide a detailed report.

Do not generate the remaining 295 samples until I review and explicitly approve the first batch.

Update CHECKPOINT.md with the exact progress and resume instructions.

The objective is not merely to satisfy numerical diversity constraints, but to produce natural, realistic, linguistically diverse Bangladeshi Promotional SMS samples that do not introduce easily detectable synthetic-origin shortcuts.