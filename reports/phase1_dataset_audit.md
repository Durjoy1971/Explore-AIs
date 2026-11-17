# Phase 1 — Dataset Verification & Shortcut Audit Report

## 1. Corpus Summary, Licenses, Duplicates & Shortcut Power

| Dataset | License | Provenance | Rows | Unique | Exact Dup % | Near-Dup Reduction % (cos≥0.85) | Label Conflicts | Surface-Only 5-fold Acc | Surface-Only Macro-F1 | Length-Only Risk AUC |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `legacy_2024` | UNDOCUMENTED | unknown | 2580 | 2083 | 19.26% | 6.0% | 1 | 89.15% | 88.79% | 0.8738 |
| `bangalabarta_2025` | CC BY-NC-SA 4.0 | curated_real_mixed | 2772 | 1205 | 56.53% | 5.23% | 1 | 78.09% | 67.61% | 0.8453 |
| `smishdetect_2026` | MIT | augmented_multilingual | 7005 | 6998 | 0.1% | 7.2% | 0 | 70.63% | 63.85% | 0.7662 |
| `banglaphish_2026` | CC BY-NC 4.0 | synthetic | 6000 | 5416 | 9.73% | 20.81% | 0 | 95.44% | 64.01% | 0.9375 |

## 2. Predefined Split Leakage (Train → Val/Test Exact Normalized Overlap)

- **`legacy_2024`**: No predefined train/val/test split shipped.
- **`bangalabarta_2025`**: No predefined train/val/test split shipped.
- **`smishdetect_2026`**: train→val exact overlap = **3** (0.43%), train→test exact overlap = **0** (0.0%)
- **`banglaphish_2026`**: train→val exact overlap = **3** (0.33%), train→test exact overlap = **2** (0.22%)

## 3. Cross-Dataset Exact Normalized-Text Overlap

| Dataset A | Dataset B | Shared Unique Texts | % of A | % of B |
| :--- | :--- | ---: | ---: | ---: |
| `legacy_2024` | `bangalabarta_2025` | 1126 | 54.06% | 93.44% |
| `legacy_2024` | `smishdetect_2026` | 1126 | 54.06% | 16.09% |
| `legacy_2024` | `banglaphish_2026` | 0 | 0.0% | 0.0% |
| `bangalabarta_2025` | `smishdetect_2026` | 1205 | 100.0% | 17.22% |
| `bangalabarta_2025` | `banglaphish_2026` | 0 | 0.0% | 0.0% |
| `smishdetect_2026` | `banglaphish_2026` | 0 | 0.0% | 0.0% |

## 4. Per-Class Surface & Shortcut Confounders

### `legacy_2024`
| Class | Count | Unique | Median Chars | Mean Chars | % Has Latin | % Has URL | % Has Phone | % Has `[REDACTED_*]` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 1061 | 833 | 28.0 | 41.9 | 2.8% | 0.9% | 3.0% | 0.0% |
| Spam_Binary | 1519 | 1251 | 84.0 | 104.1 | 61.4% | 46.2% | 25.0% | 0.0% |

### `bangalabarta_2025`
| Class | Count | Unique | Median Chars | Mean Chars | % Has Latin | % Has URL | % Has Phone | % Has `[REDACTED_*]` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 924 | 439 | 28.0 | 36.3 | 1.5% | 1.2% | 11.5% | 0.0% |
| Promotional | 924 | 246 | 69.0 | 74.8 | 60.9% | 54.1% | 3.7% | 0.0% |
| Smishing | 924 | 521 | 96.0 | 114.7 | 64.7% | 49.0% | 40.0% | 0.0% |

### `smishdetect_2026`
| Class | Count | Unique | Median Chars | Mean Chars | % Has Latin | % Has URL | % Has Phone | % Has `[REDACTED_*]` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 2488 | 2487 | 45.0 | 57.4 | 74.2% | 1.0% | 10.2% | 0.0% |
| Promotional | 1708 | 1705 | 73.0 | 90.5 | 90.5% | 31.3% | 5.3% | 0.0% |
| Smishing | 2809 | 2806 | 97.0 | 115.6 | 92.9% | 47.4% | 33.0% | 0.0% |

### `banglaphish_2026`
| Class | Count | Unique | Median Chars | Mean Chars | % Has Latin | % Has URL | % Has Phone | % Has `[REDACTED_*]` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 2800 | 2336 | 100.0 | 103.4 | 49.2% | 0.0% | 0.0% | 0.7% |
| Promotional | 200 | 80 | 115.0 | 114.9 | 62.5% | 0.0% | 0.0% | 0.0% |
| Smishing | 3000 | 3000 | 216.0 | 212.7 | 30.0% | 0.0% | 0.0% | 6.7% |
