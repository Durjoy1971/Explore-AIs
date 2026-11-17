# Experimental Log (Synthetic Sample)

## Datasets & Metrics
- Dataset: Synthetic SMS Benchmark (5-fold group split, seed=42)
- Metrics: Macro-F1 (%), Expected Calibration Error (ECE)

## Main Quantitative Results
| Method | Macro-F1 (%) | ECE |
| :--- | ---: | ---: |
| TF-IDF + LR | 84.2 | 0.091 |
| Compact Encoder (Unmasked) | 88.5 | 0.064 |
| Compact Encoder + SMC (Ours) | 91.4 | 0.032 |

## Compute & Reproducibility
- Hardware: Single CPU/GPU evaluation environment, fixed seed 42, batch size 32.
- Limitation: Evaluated only on synthetic sample records in this validation fixture.
