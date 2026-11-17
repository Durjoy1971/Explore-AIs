# Research Integrity Gates

Every stage transition enforces objective integrity requirements:

1. **Gate 1 — Outline Traceability (`outline`):**
   - `outline.json` must contain `plotting_plan`, `intro_related_work_plan`, and `section_plan`.
   - `figure_id` values must not contain the word `Figure`.
   - Subsections cannot be orphaned (if `X.1` exists, `X.2` must exist).
   - Every quantitative bullet or method claim in `section_plan` must reference `idea.md` or `experimental_log.md`.

2. **Gate 2 — Grounded Assets (`assets`):**
   - Every entry in `artifacts/citation_map.json` must include `verified: true`, a `verification_source` (e.g., `semantic_scholar`, `crossref`, `arxiv`, or `offline_fixture`), and an external identifier (`paper_id`, `doi`, or `arxiv_id`).
   - All verified papers must satisfy `year/publication_date <= manifest.cutoff_date` (when `cutoff_date` is set).
   - Every entry in `manuscript/figures/info.json` must point to an existing non-empty file in `manuscript/figures/`.

3. **Gate 3 — Manuscript Grounding (`manuscript`):**
   - Every `\cite`, `\citet`, `\citep`, `\citeauthor`, `\citeyear` key in `manuscript/main.tex` must exist in **both** `artifacts/citation_map.json` and `manuscript/references.bib`.
   - Every `\includegraphics{...}` target must resolve to an existing file relative to `manuscript/`.
   - All LaTeX `\begin{env}` / `\end{env}` pairs must balance cleanly (e.g., `figure*` closes with `figure*`).
   - All numeric values in LaTeX tables must be traceable to `raw_materials/experimental_log.md` or `raw_materials/data/`.

4. **Gate 4 — Submission Readiness (`final`):**
   - `reviews/integrity_report.json` must report `"status": "pass"` with zero critical errors.
   - If `manifest.double_blind` is `true`, no author names, institutional domains, or `corresponding author` strings may appear in `manuscript/main.tex`.
   - Honest Limitations and Reproducibility discussion must not be suppressed.
