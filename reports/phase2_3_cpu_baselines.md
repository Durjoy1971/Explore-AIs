# Phase 2 & Phase 3 (CPU Baselines) — Fair Evaluation & Shortcut-Stress Results

Unified deduplicated 3-class corpus (`BangalaBarta` + `SmishDetect-2026`): **6997** unique non-conflicting messages (clusters: 6493).

| Experiment / Stress Split | N | Accuracy (95% CI) | Macro-F1 (95% CI) | Smishing Recall | Normal→Smish FPR | Recall @ ≤1% FPR | ECE |
| :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: |
| `surface_only_dedup_cv` | 6997 | 70.84% [69.79, 71.92] | 67.85% [66.72, 69.0] | 75.52% | 5.63% | 14.54% | 0.0564 |
| `tfidf_lr_dedup_cv_raw_text` | 6997 | 98.64% [98.37, 98.91] | 98.65% [98.38, 98.92] | 98.75% | 1.17% | 97.83% | 0.0674 |
| `tfidf_lr_dedup_cv_masked_text` | 6997 | 98.06% [97.74, 98.4] | 98.03% [97.71, 98.38] | 97.9% | 1.37% | 96.47% | 0.0624 |
| `tfidf_lr_eval_adversarial_zwj_homoglyph` | 6997 | 97.66% [97.3, 98.01] | 97.67% [97.31, 98.02] | 96.79% | 1.01% | 95.94% | 0.0853 |
| `tfidf_lr_eval_length_matched_40_120` | 4641 | 98.58% [98.23, 98.9] | 98.53% [98.16, 98.86] | 99.0% | 2.31% | 97.55% | 0.0597 |
| `tfidf_lr_eval_banglish_codemix` | 3952 | 98.86% [98.53, 99.19] | 98.88% [98.54, 99.2] | 99.07% | 1.28% | 97.9% | 0.0588 |
| `lodo_train_bangalabarta_test_smishdetect_raw` | 5793 | 78.8% [77.8, 79.8] | 76.32% [75.25, 77.43] | 84.38% | 2.69% | 63.34% | 0.1508 |
| `lodo_train_smishdetect_test_bangalabarta_raw` | 1204 | 95.43% [94.19, 96.51] | 95.12% [93.72, 96.41] | 95.19% | 2.05% | 85.77% | 0.0977 |
| `lodo_train_bangalabarta_test_smishdetect_masked` | 5793 | 70.69% [69.45, 71.79] | 67.61% [66.36, 68.74] | 69.69% | 2.93% | 45.49% | 0.0837 |
| `lodo_train_smishdetect_test_bangalabarta_masked` | 1204 | 94.1% [92.69, 95.35] | 93.54% [91.95, 94.95] | 94.81% | 2.96% | 83.27% | 0.101 |
| `ood_train_unified_test_banglaphish_sms_raw` | 1540 | 66.75% [64.48, 69.03] | 53.12% [49.94, 56.46] | 73.0% | 24.36% | 6.0% | 0.0834 |
| `ood_train_unified_test_banglaphish_sms_masked` | 1540 | 66.82% [64.55, 69.09] | 53.57% [50.46, 56.67] | 70.6% | 17.62% | 8.3% | 0.074 |
| `smishdetect_original_test_split` | 1401 | 98.79% [98.22, 99.36] | 98.77% [98.17, 99.34] | 99.11% | 0.8% | 99.11% | 0.0707 |

## McNemar Significance Tests (Unified Dedup CV)
- **Raw text vs. Shortcut-masked (`<URL>/<PHONE>/<USSD>`)**: p = `0.0` (raw-only correct: 46, masked-only correct: 5)
- **Raw text vs. Adversarial ZWJ/Homoglyph**: p = `0.0` (raw-only correct: 85, adv-only correct: 16)