---
name: scientific-writing
description: Drafts and assembles submission-ready LaTeX manuscripts from verified outlines, Markdown notes, experimental logs, citation maps, and figures. Use when writing Abstract, Methodology, Experiments, Limitations, Conclusion, constructing booktabs tables, or checking LaTeX manuscript grounding.
version: 1.0.0
---

# Scientific Writing

Transforms `artifacts/outline_v1.json`, `raw_materials/idea.md`, `raw_materials/experimental_log.md`, `artifacts/citation_map.json`, and `manuscript/figures/info.json` into a complete, grounded LaTeX manuscript (`manuscript/main.tex`).

## Core Integrity Rules

1. **Preserve Verified Sections:** Do not alter verified citations or claims in existing Introduction/Related Work sections except for minor notation alignment.
2. **Exact Experimental Grounding:** Construct all `booktabs` tables (`\toprule`, `\midrule`, `\bottomrule`) and quantitative claims verbatim from `raw_materials/experimental_log.md` or `raw_materials/data/`. Never extrapolate, round inconsistently, or fabricate ablation numbers.
3. **Strict Bibliography Closure:** Only use `\cite{...}` (or `\citet`/`\citep` for `natbib`) keys present in `artifacts/citation_map.json` and `manuscript/references.bib`. Ground descriptions of cited work in each entry's verified `abstract`.
4. **Honest Limitations & Reproducibility:** Always include a candid **Limitations** discussion (scope boundaries, dataset constraints, failure modes) and a **Reproducibility** note referencing hardware, seeds, and hyperparameters present in `experimental_log.md`.
5. **Visual Fidelity:** Include every figure listed in `manuscript/figures/info.json` using its exact `figures/<name>` path, placed before Conclusion (or in Appendix after References), and never describe visual patterns that contradict the rendered image.

## Workflow

1. **Select Venue Template:** Initialize `manuscript/main.tex` from `assets/templates/iclr_single_col.tex` (single-column) or `assets/templates/cvpr_two_col.tex` (two-column), or a user-provided venue template.
2. **Draft Remaining Sections:** Follow `outline_v1.json` section-by-section for `Abstract`, `Methodology`, `Experiments` (setup, main results, ablations, compute cost), `Limitations`, `Conclusion`, and optional `Appendix` (after `\bibliography{references}`).
3. **Run Pre-Compile Integrity Gate:**
   ```bash
   python3 .gemini/skills/scientific-writing/scripts/verify_manuscript_integrity.py \
     --tex manuscript/main.tex \
     --citation-map artifacts/citation_map.json \
     --bib manuscript/references.bib \
     --experimental-log raw_materials/experimental_log.md \
     --figures-info manuscript/figures/info.json \
     --double-blind
   ```
   Fix any reported citation, figure, environment-balance, or table-number mismatches before handing off to `peer-review`.

## References, Assets & Scripts
- LaTeX & prose standards: [latex-writing-rules.md](references/latex-writing-rules.md)
- Venue formatting summary: [venue-guidelines-summary.md](references/venue-guidelines-summary.md)
- Templates: [iclr_single_col.tex](assets/templates/iclr_single_col.tex), [cvpr_two_col.tex](assets/templates/cvpr_two_col.tex)
- Integrity checker: `scripts/verify_manuscript_integrity.py`
