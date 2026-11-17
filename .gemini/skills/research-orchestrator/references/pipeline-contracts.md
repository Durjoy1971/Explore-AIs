# Pipeline Contracts & Artifact Layout

All project artifacts live under `research_workspace/<project-slug>/`, completely separate from `.gemini/skills/`.

## Directory Structure

```text
research_workspace/<project-slug>/
├── manifest.json                 # Project metadata: venue, cutoff_date, double_blind, model policy
├── raw_materials/
│   ├── idea.md                   # Core methodology, hypotheses, theoretical notes
│   ├── experimental_log.md       # Raw metrics, ablations, tables, system observations
│   ├── data/                     # Optional CSV/JSON raw experimental tables
│   └── figures/                  # Optional user-supplied pre-existing figures
├── artifacts/
│   ├── outline.json              # Output of Stage 1 (research-outline)
│   ├── outline_v1.json           # Outline enriched with verified citation_candidates (Stage 2a)
│   ├── candidates_raw.json       # Unverified candidate pool before API check
│   ├── citation_map.json         # Verified reference metadata keyed by citation_key
│   └── plotting_results.json     # Metadata, script paths, captions for generated figures
├── manuscript/
│   ├── main.tex                  # Complete LaTeX manuscript
│   ├── references.bib            # BibTeX generated strictly from verified citation_map.json
│   ├── figures/                  # Final figures + info.json
│   └── scripts/                  # Reproducible Python plotting scripts
└── reviews/
    ├── integrity_report.json     # Deterministic citation, numeric, and LaTeX audit results
    ├── peer_review_v0.json       # Baseline ensemble review
    ├── peer_review_v*.json       # Refinement iteration reviews
    ├── lit_review_eval.json      # 6-axis literature review score
    └── refinement_worklog.json   # Accepted/reverted edits with score deltas
```

## Inter-Stage Data Contracts

| Stage | Required Inputs | Required Outputs |
| :--- | :--- | :--- |
| `init` | Project slug, venue (`iclr_single` or `cvpr_two`), cutoff (`YYYY-MM`) | `manifest.json`, `raw_materials/` skeleton |
| `outline` | `raw_materials/idea.md`, `raw_materials/experimental_log.md` | `artifacts/outline.json` |
| `assets` (2a) | `artifacts/outline.json` | `artifacts/citation_map.json`, `manuscript/references.bib`, `artifacts/outline_v1.json` |
| `assets` (2b) | `artifacts/outline.json`, `raw_materials/` | `manuscript/figures/info.json`, figure files, `artifacts/plotting_results.json` |
| `manuscript` | `artifacts/outline_v1.json`, `citation_map.json`, `figures/info.json`, `idea.md`, `experimental_log.md` | `manuscript/main.tex` |
| `final` | `manuscript/main.tex`, `manuscript/references.bib`, `raw_materials/experimental_log.md` | `reviews/integrity_report.json`, `reviews/peer_review_v0.json` |
