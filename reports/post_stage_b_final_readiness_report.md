# Post-Stage B final readiness

**Decision: NO-GO for full Experiments A/B/C/D.** The full runner blocks exact train-to-evaluation model-input overlaps in the two brand views. The frozen row and family assignments passed independent audits. Both facts matter when interpreting these results.

| Gate | Status | Evidence |
|---|---|---|
| Dataset integrity | PASS | Source and bundle checksums verified. 3,999 eligible, 300 Pilot and 345 batch rows. Full pool has 1,548 per class and 387 per Promotional language. v2 normalization metadata is consistent. |
| Brand registry | PASS | 364 unified entries. Batch 7 alias and inflection fixtures passed. All four clean views were independently reconstructed on 12,466 rows. |
| Experiment A | PASS | Construction verified in all five folds. Unweighted eligible-source baseline, 3,999 candidate rows. |
| Experiment B class weighting | PASS | Same ordered train and validation IDs as A in all five folds. Training-fold weights and train-only loss dispatch verified. |
| Experiment C construction | PASS | 4,344 rows in the candidate pool. All 345 batch rows restricted to training and all five validation sets match A. |
| Experiment D construction | PASS | 4,644 rows in the candidate pool. All 645 additions restricted to training and all five validation sets match A. |
| Family leakage prevention | PASS | Frozen row/family boundaries are clean across all five folds and across the 15 main cohort pairs. Brand-view model-input identity is tracked in the evaluation gates below. |
| Held-out/OOD isolation | FAIL | Source IDs, raw/normalized text and family isolation passed. In each brand view, train contains an equal model input for one held-out validation row and one test row. OOD has zero exact collisions. |
| Low-FPR metric | PASS | Float32/64, ties, zero-FP, missing classes, fixed-threshold breach and exhaustive cutoff fixtures passed. Historical smoke archive replay is NOT VERIFIED because the files are absent locally. |
| Counterfactual isolation | FAIL | Every eligible probe row belongs to exactly one scored validation fold by ID and family. In folds 3 and 4 the brand-view forms of one probe per fold are also present as training inputs. |
| Resume integrity | PASS | Local configuration and manifest drift, corrupted archives and metrics, incomplete checkpoints, ordered prediction verification, and immutable attempt recovery were exercised. Mounted Drive recovery is NOT VERIFIED. |
| Preprocessing consistency | PASS | A–D share view generation and backbone-specific tokenization. View, normalization and adversarial dispatch tested. Exact brand-view overlap remains a separate training preflight failure. |
| Statistical methodology | BLOCKED | Family bootstrap, fold variation, fixed-threshold transfer and exploratory paired analysis are implemented. A confirmatory seed plan, inference protocol, and held-out reporting and selection policy need owner approval. |
| Regression tests | PASS | 63/63 local unit and integration tests passed, plus the standalone pipeline check and notebook schema. See the [test report](post_stage_b_test_report.md). |
| Colab readiness | NOT VERIFIED | Notebook and four CPU tokenizer paths validated. Corrected CUDA, Drive recovery, peak VRAM, runtime, and checkpoint capacity require measurement. |

## Owner decisions before full training

1. Review the three exact model-input pairs and the heuristic near-duplicate candidates. Approve a versioned policy for the affected comparisons. Options include a corrected representation-aware eligibility or fold rule, or a narrower evaluation scope with the brand views withheld. Preserve the current v1 and v2 splits as evidence.
2. Freeze which CV metrics select the model, how fold models produce held-out and OOD results, the confirmatory comparisons and repeat-seed plan. Accept the equal-epoch interpretation or separately authorize an update-matched sensitivity study.
3. Supply a read-only local copy of the old smoke `preds.npz`, metrics, and logs if historical re-scoring matters. Retain the archive in its historical artifact directory.
4. After reviewing the corrections, authorize and run a separately scoped v2 Colab smoke. Record CUDA precision, peak allocated/reserved VRAM, train time, checkpoint sizes, and recovery after an interrupted unit on a fresh run ID. The old DistilBERT smoke does not measure the corrected runner or the other three backbones.
5. Re-run both local verification suites and the model-view audit after any approved changes. Full training requires an explicit later authorization and a new artifact directory.

[Comprehensive audit](post_stage_b_comprehensive_audit.md) · [Implementation](post_stage_b_implementation_report.md) · [Integrity and leakage](post_stage_b_integrity_and_leakage_report.md) · [Tests](post_stage_b_test_report.md) · [Colab readiness](post_stage_b_colab_readiness_report.md)
