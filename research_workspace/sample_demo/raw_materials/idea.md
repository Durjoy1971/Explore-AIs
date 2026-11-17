# Research Idea Summary (Synthetic Sample)

## Problem Statement
Compact multilingual encoders suffer accuracy drops when surface URL/phone tokens are masked to prevent shortcut learning in SMS smishing detection.

## Proposed Method
We propose **Shortcut-Masked Calibration (SMC)**, combining token-level `<URL>`/`<PHONE>` masking with temperature scaling on validation logits:
$$p(y \mid x) = \text{softmax}(z(x) / \tau)$$
where $\tau > 0$ is calibrated post-hoc on a held-out leakage-free validation split.

## Core Contributions
- Leakage-free evaluation under surface-token masking.
- Post-hoc temperature scaling improving Expected Calibration Error (ECE) without retraining.
