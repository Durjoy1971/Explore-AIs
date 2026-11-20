# Post-Stage B implementation report

The working code uses a new v2 dataset and artifact layout. Historical datasets, the v1 manifest and Parquet files, existing reports, and evaluation memberships remain byte-for-byte unchanged. No model was trained and no SMS rows were added.

| Severity and change | Implementation point and cause addressed | Regression and result |
|---|---|---|
| Critical · Experiment B | Both Colab configs set `class_weight`. `post_stage_b_data.py` builds the same ordered A/B fold and computes `N / (3 × n_class)` in Normal/Promotional/Smishing order. `post_stage_b_runner.py` supplies the weights to training cross-entropy only. Mixed-precision logits are promoted to float32 for the weighted calculation. Standard validation CE drives best-epoch selection. | All five ordered A/B comparisons and the hand example `[2/3, 1, 2]` passed. The loss regression checks batch-weighted CE against a hand computation and dispatches standard loss in eval mode. Unsupported B modes fail validation. The real CPU Trainer also matched the hand-weighted loss and unweighted validation loss. |
| Critical · Operating point | `metrics.py:18` compares float64 scores with `>=`, using `nextafter` on the first disallowed Normal score. Budgeting uses the integer count `floor(FPR limit × n_normal)`. Ties at that boundary are excluded together. Missing denominators remain unavailable, and the exported cutoff retains full precision. The runner transfers the clean CV threshold to each downstream slice. | Six metric regressions passed. The fixed-threshold breach case keeps achieved FPR and recall while the constrained summary is empty. |
| Critical · Counterfactual assignment | `post_stage_b_data.py` matches the probe parent by row, label, family and fold against the uncapped CV validation frame and asserts no training row or family matches. The runner applies smoke caps only after selection. Uncapped five-fold aggregation requires complete, unique probe coverage. | Fold audit and the cross-fold tamper regression passed. Each A/B/C/D arm covers 1,279 probe IDs once across its five folds. The model-view collision gate applies independently. |
| Critical · Resume identity and prediction commits | `post_stage_b_runner.py` binds run state to full configuration SHA256, bundle manifest SHA256, runner kind, and source hashes. It validates state and unit hashes, probabilities, ordered IDs and labels for every evaluation slice. Aggregation uses only declared successful commits. Each attempt has a separate path. | Configuration and manifest drift, ZIP corruption, metric corruption, rehashed row shuffle, incomplete checkpoints and orphan files were exercised locally. Failed or superseded attempts remain available. |
| High · Checkpoint preservation | Epoch callback writes a checksum record after Trainer saves model, optimizer, scheduler, RNG and state files. Recovery searches valid saved attempts, checks recorded best checkpoint availability, and retains prior files. | Local checkpoint fallback tests passed on controlled fixtures. CUDA/Drive interruption remains unverified. |
| High · Counts and derived views | `post_stage_b_runner.py` records pre-cap and actual train/validation counts, actual class counts, and actual Stage B synthetic/other composition. `post_stage_b_data.py` records adversarial text for each view, and `post_stage_b_tokenization.py:10` applies the required BanglaBERT normalization to the selected view. | Mock smoke reconstruction A/D each used 64 train and 32 validation rows. Adversarial dispatch, normalization, and Batch 7 alias fixtures passed. |
| High · Bundle writer and analysis | The preparation CLI requires a fresh output path for a rebuild and a fresh directory for kit export. v2 fixes 79 synthetic normalization metadata values, adds three masked adversarial derivatives, and binds the shared registry to the manifest. The verifier checks physical metadata and recomputes source/fold integrity. | v1 and v2 manifest checks passed. Original text, ordered rows and labels, four clean views, raw adversarial view, and all probe inputs agree across versions. |
| Critical · Exact model inputs | `post_stage_b_model_view_audit.py:30` checks exact input identity across requested experiments, folds, views and evaluation sets. The full runner aborts before Trainer if the check detects overlap. | v1 and v2 each reported 128 failed comparisons of 560. The integration fixture verifies abort before training and run-state creation. |
| High · Colab packaging and launch | The regenerated notebook preserves the smoke grid and requires explicit full-run authorization and a distinct artifact ID. It validates package versions and uses the v2 bundle. | Notebook schema, syntax, generated source parity, safe path, and dependency-decision regressions passed. The runner also uses native BF16 detection. The T4 precision regression selected FP16. Historical GPU setup remains owner-reported. |

## Historical smoke reconciliation

Reconstruction used the original A/D config, seed 42, fold 0, ordered frozen rows, and the existing caps. The new mocked smoke run independently checks the same counts. The original prediction archives and runner logs are unavailable locally.

| Unit | Train pre-cap | Val pre-cap | Train actual | Val actual | Train Normal / Promotional / Smishing | Train eligible source / Stage B additions | Val Normal / Promotional / Smishing |
|---|---:|---:|---:|---:|---|---|---|
| A · DistilBERT · raw | 3,199 | 800 | 64 | 32 | 22 / 13 / 29 | 64 / 0 | 9 / 7 / 16 |
| D · DistilBERT · raw | 3,844 | 800 | 64 | 32 | 22 / 22 / 20 | 56 / 8 | 9 / 7 / 16 |

New run output locations include `units/<unit-id>/attempts/<attempt>/` for raw prediction archives and metrics, `checkpoints/<unit-id>/<attempt>/` for resumable Trainer state, and `exports/history/` for immutable summary snapshots. The top-level state and latest export paths serve as the active run pointers. The original historical smoke directory, if supplied, belongs to the older identity and can be inspected separately.

Full-grid configuration preserves four backbones, four views, five frozen folds, three epochs and the seed 42 schedule. Model input isolation requires a design-level decision before it can pass the full preflight. See the [comprehensive findings](post_stage_b_comprehensive_audit.md), [test commands](post_stage_b_test_report.md), and [readiness matrix](post_stage_b_final_readiness_report.md).

## Changed file inventory

### Modified

- `CHECKPOINT.md`
- `experiments/colab/configs/gpu_smoke_test_config.json`
- `experiments/colab/configs/post_stage_b_abcd_config.json`
- `experiments/colab/post_stage_b_experiments.ipynb`
- `scripts/generate_post_stage_b_notebook.py`
- `scripts/prepare_post_stage_b_bundle.py`
- `src/bangla_smish/metrics.py`
- `src/bangla_smish/post_stage_b_data.py`
- `src/bangla_smish/post_stage_b_runner.py`
- `tests/test_post_stage_b_pipeline.py`

### Added

- `data/processed/post_stage_b_bundle_v2/counterfactual_smish_probe.parquet`
- `data/processed/post_stage_b_bundle_v2/dataset_manifest.json`
- `data/processed/post_stage_b_bundle_v2/eval_banglaphish_ood.parquet`
- `data/processed/post_stage_b_bundle_v2/eval_smishdetect_test.parquet`
- `data/processed/post_stage_b_bundle_v2/eval_smishdetect_validation.parquet`
- `data/processed/post_stage_b_bundle_v2/real_train_pool.parquet`
- `data/processed/post_stage_b_bundle_v2/synthetic_batches_b1_b7.parquet`
- `data/processed/post_stage_b_bundle_v2/synthetic_pilot_300.parquet`
- `reports/post_stage_b_colab_readiness_report.md`
- `reports/post_stage_b_comprehensive_audit.md`
- `reports/post_stage_b_final_readiness_report.md`
- `reports/post_stage_b_implementation_report.md`
- `reports/post_stage_b_integrity_and_leakage_report.md`
- `reports/post_stage_b_test_report.md`
- `scripts/audit_post_stage_b_model_views.py`
- `src/bangla_smish/post_stage_b_model_view_audit.py`
- `src/bangla_smish/post_stage_b_tokenization.py`
- `tests/test_post_stage_b_metrics.py`
- `tests/test_post_stage_b_preprocessing.py`
- `tests/test_post_stage_b_runner.py`

The v2 manifest records the base Git commit and creation time. These audited working-tree changes are part of the reproducible release, and each run identity also fingerprints the relevant source files.
