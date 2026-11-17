# Agent workflow for dataset curation and augmentation

## Objective

First build a consolidated, deduplicated corpus from the existing datasets.
Then, in a separate balancing stage, add explicitly synthetic examples for
underrepresented training classes based on the measured accepted counts.
Balance classes and relevant subgroups while reducing duplicates, label errors,
and superficial cues. Do not claim that synthetic data represents real-world
message frequencies or that similarity checks prove absolute uniqueness.

## Required order and agent coordination

Use as many collaborating agents as useful within available concurrency. Assign
separate source audits where practical, then a consolidation task and an
independent validation task. Agents may work sequentially when slots are limited.
Use separate intermediate output paths; a coordinating agent owns the final
merge. All agents follow this guide and use the same label and comparison rules.

### Stage A: Existing data only

1. Inventory `DataSet.csv` and dataset files under `data/raw/`. Count each logical
   source once: CSV/JSONL/XLSX versions and train/validation/test exports of the
   same records are alternative representations, not additional independent data.
   Existing `data/processed/` files are derived artifacts, not new source records.
2. Normalize using documented comparison rules, merge across sources, and remove
   exact duplicates globally. Keep every contributing source ID and original label
   in a provenance sidecar rather than losing lineage when choosing one row.
3. Detect near-duplicate and template families. Review before collapsing them;
   similar wording can express different intent. Keep materially distinct messages
   with family IDs, and record reasons for removed variants.
4. Quarantine label conflicts, ambiguous labels, and unusable rows. Do not silently
   map legacy binary spam to either Promotional or Smishing. Keep legacy rows as
   reference-only unless their label and eligibility are established separately.
5. Preserve benchmark membership. A consolidated inventory may include held-out
   rows with explicit eligibility metadata, but the exported training subset must
   exclude evaluation rows and their duplicate/family overlaps. Keep BanglaPhish
   out of primary training while it serves as the out-of-distribution benchmark.
6. Export `data/processed/curation_<version>/existing_unique.parquet`, a provenance
   sidecar, a quarantined-row artifact with reasons, and an eligible training view.
   Include stable IDs, labels, source provenance, synthetic status, family/cluster
   IDs, original split membership, and training/evaluation eligibility.
7. Write `reports/dataset_consolidation_<version>.md` with source counts, accepted
   unique counts by class and language, duplicates removed, unresolved conflicts,
   evaluation exclusions, and the exact/near-duplicate methods and results.
   Distinguish inventory counts from eligible training counts. An independent
   validation pass checks provenance, class totals, and cross-split contamination.

Complete Stage A and report its results before any generation. Do not undersample,
oversample, or generate rows to make its counts look balanced. During a task scoped
to consolidation, leave Stage B as the documented next step.

### Stage B: Fill measured training deficits

Using accepted, eligible training rows, let `n_c` be each class count and set the
default target `T = max(n_Normal, n_Promotional, n_Smishing)`. The required accepted
new rows for class `c` are `max(0, T - n_c)`. Publish this table before generation;
do not infer quotas from raw counts or the combined evaluation inventory.

Generate only for deficient classes, in small reviewable batches. Count accepted,
deduplicated new rows toward the deficit; rejected candidates do not count.
Recompute deficits after review and retain the original-only corpus separately.
If quality or coverage limits prevent reaching a target, report the shortfall.
The generation, review, and evaluation requirements below apply to this stage.

## Read first

- `reports/phase1_dataset_audit.md`: source overlap, duplicates, and shortcut cues.
- `reports/phase2_3_cpu_baselines.md`: baseline and out-of-distribution results.
- `src/bangla_smish/schema.py`: labels, loaders, normalization, and features.
- `src/bangla_smish/audit.py` and `src/bangla_smish/splits.py`: similarity and grouping.
- Source dataset cards and annotation guides under `data/raw/`.

The recorded audit finds that all 1,205 unique BangalaBarta texts occur in
SmishDetect. Treat shared messages as shared provenance, not independent evidence.
Recompute counts for the current inputs; historical reports are not live counts.

## 1. Audit and establish boundaries

1. Inventory input paths, versions or hashes, row counts, labels, provenance, and
   source license declarations. Preserve originals and record transformations.
2. Keep `legacy_2024` as a reference: its undocumented license and binary spam
   labels do not establish permission or reliable three-class labels.
3. Document source restrictions and attribution. Do not assume a combined corpus
   inherits the most permissive license, or invent licenses for generated rows.
   Keep sources separable when redistribution eligibility is unresolved.
4. Freeze evaluation membership before augmentation. Preserve the existing
   BanglaPhish out-of-distribution benchmark and existing benchmark definitions.
   Do not use held-out messages as generation prompts or paraphrase seeds.
5. Use held-out text only in an isolated contamination check that reports rejection
   decisions to the generation stage. Do not return held-out examples to it.
   Describe prior benchmark-driven design decisions when reporting results.
6. Resolve or quarantine conflicting labels; never settle them by majority vote
   alone. Record uncertain provenance as unknown rather than assuming real data.

## 2. Define a coverage plan

Use the canonical labels `Normal`, `Promotional`, and `Smishing`.

- Normal: ordinary communication or legitimate transactional/service messages.
- Promotional: legitimate advertising or offers without deceptive extraction.
- Smishing: deceptive requests intended to obtain credentials, money, sensitive
  information, or installation of malicious software.

Urgency, a link, or an OTP mention alone does not establish smishing. Quarantine
messages whose label depends on unavailable sender identity or context.

After Stage A, start Stage B with a pilot of at most **300 accepted synthetic
training rows**, allocated across deficient classes without exceeding their
measured deficits. Do not automatically generate 100 rows per class. Audit the
pilot before continuing toward the published targets; never count copies as new
unique examples. Keep full curated data separately.
Leave natural-prevalence evaluation sets intact and report their class counts.

Plan coverage across Bengali, Banglish, English, and CodeMix; banking/mobile
finance, delivery, telecom, shopping, education, employment, and personal messages;
short/medium/long messages; formal/informal language; and links present/absent.
Distinguish Banglish from English through annotation: the current script detector
groups both as Latin. Record counts for each class-by-language cell and important
scenario gaps; equal class counts alone do not establish coverage.

Include plausible benign messages with links and urgency, and deceptive messages
without links. Use overlapping length ranges across labels. Avoid making a label
predictable from punctuation, brand names, script, redaction tags, or generator style.

## 3. Generate candidates

Generate from scenario specifications derived from training-only coverage gaps.
Do not simply rewrite each source message or change names, numbers, and URLs.
Vary intent, context, wording, and sentence structure. Track a shared family ID for
paraphrases, translations, counterfactual pairs, and common generation templates.
Use fictional identities and reserved example domains; do not embed real personal
details or active malicious destinations. Apply placeholder conventions consistently
across classes and audit whether the conventions themselves reveal synthetic origin.

Store candidates under `data/generated/<version>/candidates.jsonl`. At minimum,
record `id`, `text`, `label_3class`, `language`, `scenario`, `is_synthetic`,
`source_dataset`, `source_row_ids`, `family_id`, `generator_model`,
`prompt_version`, `generation_batch`, `created_at`, `review_status`, and
`label_rationale`. Use empty source IDs for scenario-only generation; never
fabricate lineage. Keep row metadata out of model input text.

## 4. Review and validate

1. Validate schema, labels, required metadata, and nonempty text.
2. Check exact normalized duplicates against all sources and accepted candidates.
   Add a Unicode/case/invisible-character comparison for detection while preserving
   original text; the existing normalizer only collapses whitespace.
3. Reuse the repository's character similarity clustering (initial cosine threshold
   0.85). Review flagged pairs and semantic/template similarity separately; a score
   below this threshold is not proof of novelty. Quarantine evaluation overlaps.
4. Review every pilot row for label correctness, natural language, realistic SMS
   context, copied content, personal information, and shallow template changes.
   Record reviewer identity, decision, rationale, and disagreements. Prefer an
   independent reviewer; a second pass by the generator is self-review and must be
   labeled as such. Human review is required before treating synthetic labels as
   a trusted reference evaluation set.
5. Measure accepted counts and class-by-language/scenario coverage after rejection.
   Refill genuine gaps, never relax quality checks just to meet a quota.
6. Re-run surface-only and length-only diagnostics. Compare with the original
   training corpus and investigate label cues. Record measurements and limitations;
   no fixed score alone certifies a dataset as free of shortcuts.

## 5. Export and evaluate

Write accepted rows to `data/generated/<version>/accepted.jsonl` and retain rejected
rows with reasons separately. Export a loader-compatible training artifact only
after validation, retaining provenance in a sidecar if necessary.

Keep synthetic augmentation in training for the primary experiment. For each CV
fold, derive augmentation only from that fold's training partition. Keep all rows
sharing a source parent, family, or duplicate cluster within the same partition;
do not reuse a global augmentation pool across folds if it used their held-out rows.
Label any separately constructed synthetic evaluation set as a synthetic stress set.

Compare the unchanged baseline, a balanced original-only training view, and
original-plus-synthetic training under the same evaluation protocol. Report
Macro-F1, smishing recall at <=1% Normal-as-Smishing FPR, subgroup performance,
and out-of-distribution performance. Improvements are experimental outcomes,
not a consequence guaranteed by class balance.

Produce `reports/dataset_curation_<version>.md` with input fingerprints, quotas,
actual counts, duplicate/conflict/rejection statistics, review coverage, surface
diagnostics, split contamination checks, source restrictions, and reproducible
configuration. Mark checks not run as pending. Never report a proposed dataset
or an unexecuted check as completed work.
