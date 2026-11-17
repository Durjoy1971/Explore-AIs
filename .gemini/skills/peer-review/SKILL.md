---
name: peer-review
description: Performs automated multi-persona peer review, 6-axis literature review evaluation, deterministic research-integrity audits, score-gated manuscript refinement, and LaTeX layout/page-limit checks. Use when reviewing a research draft, running integrity audits, improving reviewer scores, or verifying submission formatting.
version: 1.0.0
---

# Peer Review & Refinement

Evaluates and refines a draft manuscript using deterministic integrity checks, an AgentReview-inspired multi-reviewer ensemble, a conservative 6-axis literature review rubric, and a score-gated revision loop.

## Core Safeguards

- **Deterministic Audit First:** Always run `scripts/audit_submission.py` before and after every revision to ensure zero fabricated citations, zero ungrounded table numbers, and valid LaTeX structure.
- **No Unfounded Experiments:** When a reviewer asks for new experiments, datasets, or ablations not present in `raw_materials/experimental_log.md`, **never** invent results or promise unrun experiments. Clarify existing setup/analysis only when supported by the log.
- **Honest Limitations:** Never delete or suppress valid Limitations or Reproducibility statements to artificially inflate scores.

## Workflow

### Step 1 — Deterministic Submission Audit
Run:
```bash
python3 .gemini/skills/peer-review/scripts/audit_submission.py \
  --tex manuscript/main.tex \
  --citation-map artifacts/citation_map.json \
  --bib manuscript/references.bib \
  --experimental-log raw_materials/experimental_log.md \
  --figures-info manuscript/figures/info.json \
  --report-out reviews/integrity_report.json \
  --double-blind
```
(Add `--compile` if `pdflatex` and `bibtex` are installed in the environment.)

### Step 2 — Baseline Multi-Persona Peer Review & Lit Eval
- Apply [agent-review-rubric.md](references/agent-review-rubric.md) using a 3-reviewer ensemble (Knowledgeable, Responsible, Benign) aggregated into an Area Chair meta-review (`reviews/peer_review_v0.json`).
- Apply [lit-review-rubric.md](references/lit-review-rubric.md) for skeptical 6-axis Introduction/Related Work scoring (`reviews/lit_review_eval.json`).

### Step 3 — Score-Gated Content Refinement (Max 3 Rounds)
1. Address actionable `Weaknesses` and `Questions` strictly using facts from `idea.md`, `experimental_log.md`, and `citation_map.json`.
2. Re-run `scripts/audit_submission.py` on the revised draft.
3. Re-score and evaluate the gate via `scripts/audit_submission.py --score-gate reviews/peer_review_v{k}.json reviews/peer_review_v{k+1}.json`:
   - **Accept & Continue:** `Overall` strictly increases, OR `Overall` ties with `total_drop <= total_gain` across sub-axes (`Originality`, `Quality`, `Clarity`, `Significance`, `Soundness`, `Presentation`, `Contribution`).
   - **Revert & Stop:** `Overall` decreases OR `total_drop > total_gain` on a tie.

### Step 4 — Layout & Formatting Check
- If a compiled PDF exists and `pymupdf`/`Pillow` are installed, run `scripts/pdf_grid_export.py` to generate 2×2 page grids and inspect against [format-checklist.md](references/format-checklist.md).
- Apply layout-only adjustments (e.g., `tabularx`, `\small`, float placement, tightening non-informational adjectives if over page limit) without altering scientific claims, then re-verify with `scripts/audit_submission.py`.

## References & Scripts
- Ensemble peer-review rubric: [agent-review-rubric.md](references/agent-review-rubric.md)
- Skeptical lit-review rubric: [lit-review-rubric.md](references/lit-review-rubric.md)
- Formatting & layout checklist: [format-checklist.md](references/format-checklist.md)
- Submission auditor & score-gate helper: `scripts/audit_submission.py`
- PDF 2×2 grid exporter: `scripts/pdf_grid_export.py`
