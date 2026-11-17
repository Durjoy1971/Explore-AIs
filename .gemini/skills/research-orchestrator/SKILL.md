---
name: research-orchestrator
description: Orchestrates the end-to-end scientific research-writing pipeline from Markdown notes and experimental logs to a verified LaTeX manuscript. Use when initializing a research writing project, running the full paper pipeline, coordinating outline/literature/plotting/writing/peer-review skills, or checking pipeline integrity gates.
version: 1.0.0
---

# Research Orchestrator

Coordinates a modular, multi-stage scientific writing workflow inspired by PaperOrchestra (`arXiv:2604.05018`), adapted natively for Gemini models and strict research-integrity enforcement.

## Core Principles

1. **Workspace Separation:** Keep agent skills strictly under `.gemini/skills/` and keep all research inputs, intermediate artifacts, manuscripts, and reviews under `research_workspace/<project-slug>/`.
2. **Gemini-Only Models:** Use Gemini models exclusively across all reasoning, search, visual critique, and review tasks (e.g., Gemini Pro-tier for outlining, synthesis, writing, and review; Gemini Flash-tier for fast candidate discovery).
3. **Zero Fabrication:** Never invent citations, datasets, experimental results, statistical significance numbers, or scientific contributions. Every claim must trace to `raw_materials/` or verified external bibliography APIs.
4. **Gate Before Advance:** Do not advance to the next stage if a deterministic integrity gate fails.

## Pipeline Workflow

Follow these stages sequentially (Stage 2a and Stage 2b may run in parallel once `outline.json` passes Gate 1):

### Stage 0 — Initialize Project Workspace
Run `scripts/init_project.py` to scaffold `research_workspace/<project-slug>/` (with `raw_materials/`, `artifacts/`, `manuscript/figures/`, and `reviews/`). Place Markdown research inputs (`idea.md`, `experimental_log.md`) and raw data under `raw_materials/`.

### Stage 1 — Research Outline (`research-outline`)
Activate `research-outline` to synthesize `raw_materials/idea.md`, `raw_materials/experimental_log.md`, and the target venue template/guidelines into `artifacts/outline.json`.
- **Gate 1:** Validate `artifacts/outline.json` with `scripts/validate_workspace.py --stage outline` (or `validate_outline.py` from `research-outline`).

### Stage 2a & 2b — Literature Review & Scientific Plotting (Parallelizable)
- **Stage 2a (`literature-review`):** Discover candidate papers via Gemini Search, verify metadata and cutoff dates against Semantic Scholar / Crossref / arXiv, emit `artifacts/citation_map.json`, `manuscript/references.bib`, `artifacts/outline_v1.json`, and draft Introduction + Related Work into `manuscript/main.tex`.
- **Stage 2b (`scientific-plotting`):** Execute `plotting_plan` from `artifacts/outline.json` by writing reproducible Python plotting scripts (for quantitative plots) and TikZ/Mermaid/SVG specifications (for diagrams). Save rendered outputs and `manuscript/figures/info.json`.
- **Gate 2:** Run `scripts/validate_workspace.py --stage assets` to confirm verified citations, valid `.bib` entries, and existing figure assets before full drafting.

### Stage 3 — Scientific Writing (`scientific-writing`)
Activate `scientific-writing` to draft the remaining sections (`Abstract`, `Methodology`, `Experiments`, `Limitations`, `Conclusion`, optional `Reproducibility` / `Appendix`) into `manuscript/main.tex`, preserving verified Introduction/Related Work and formatting `booktabs` tables strictly from `experimental_log.md`.
- **Gate 3:** Run `scripts/validate_workspace.py --stage manuscript` to verify every `\cite{...}` key, `\includegraphics` path, numeric grounding, and LaTeX environment balance.

### Stage 4 — Peer Review & Refinement (`peer-review`)
Activate `peer-review` to run deterministic submission audits, multi-persona AgentReview scoring, 6-axis literature evaluation, score-gated content refinement, and layout/page-limit checks.
- **Gate 4:** Run `scripts/validate_workspace.py --stage final` to verify final audit reports and zero unresolved critical integrity errors.

## References & Scripts
- Contracts & directory schemas: [pipeline-contracts.md](references/pipeline-contracts.md)
- Integrity gates specification: [integrity-gates.md](references/integrity-gates.md)
- Scaffold workspace: `scripts/init_project.py`
- Validate stage/gates: `scripts/validate_workspace.py`
