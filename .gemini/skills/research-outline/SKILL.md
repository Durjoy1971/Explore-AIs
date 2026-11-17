---
name: research-outline
description: Synthesizes Markdown research notes (idea.md), experimental logs (experimental_log.md), and target venue guidelines into a structured, integrity-checked paper outline (outline.json). Use when planning a scientific manuscript, designing a plotting plan, structuring Introduction vs Related Work search strategies, or enumerating citation hints.
version: 1.0.0
---

# Research Outline

Transforms unstructured Markdown research inputs (`idea.md`, `experimental_log.md`) and venue guidelines into a structured, verifiable JSON manuscript blueprint (`artifacts/outline.json`).

## Workflow

1. **Synthesize Inputs Together:** Read `raw_materials/idea.md`, `raw_materials/experimental_log.md`, any raw data files in `raw_materials/data/`, and the target venue guidelines. Never analyze inputs in isolation.
2. **Design the Plotting Plan (`plotting_plan`):**
   - Identify essential figures needed to prove the core hypothesis (quantitative plots from `experimental_log.md` and architecture/pipeline diagrams from `idea.md`).
   - Set `plot_type` strictly to `"plot"` or `"diagram"`.
   - Set `data_source` strictly to `"idea.md"`, `"experimental_log.md"`, or `"both"`.
   - Set `aspect_ratio` to one of `"1:1"`, `"3:2"`, `"4:3"`, `"16:9"`, `"21:9"`, `"2:3"`, `"3:4"`, `"4:5"`, `"5:4"`, `"9:16"`, `"1:4"`, `"4:1"`.
   - Name `figure_id` semantically (e.g., `fig_framework_overview`), and NEVER include the word `"Figure"`.
3. **Separate Literature Scopes (`intro_related_work_plan`):**
   - **Introduction Strategy (Macro, 10–15 papers):** Define `hook_hypothesis` and `problem_gap_hypothesis` strictly scoped to the datasets and evaluations present in `experimental_log.md` (no over-claiming), plus 3–5 search directions for impact, surveys, and foundational works.
   - **Related Work Strategy (Micro, 2–4 clusters):** Group directly competing/preceding paradigms. Include `sota_investigation_mission` (respecting `cutoff_date`), `limitation_hypothesis` grounded in `idea.md`, narrow `limitation_search_queries`, and `bridge_to_our_method`. Do NOT invent competitors not evaluated in `experimental_log.md`.
4. **Build Section Plan (`section_plan`) & Citation Hints:**
   - Outline `Abstract`, `Methodology`, `Experiments`, `Limitations`, `Conclusion`, and optional `Appendix`.
   - Enforce structural balance: if `X.1` is created, `X.2` is mandatory (no orphaned single subsections).
   - Reference exact equations, tables, and observations from `idea.md` and `experimental_log.md` via `evidence_sources`.
   - Enumerate exhaustive `citation_hints` for every baseline, dataset, metric, optimizer, and foundational model/framework mentioned. If unsure of authors, write `"research paper or technical report introducing '<Exact Name>'"` — never guess authors.
5. **Validate Output:** Run `scripts/validate_outline.py --outline artifacts/outline.json` before passing to `literature-review` and `scientific-plotting`.

## References, Assets & Scripts
- Schema details: [outline-schema.md](references/outline-schema.md)
- JSON skeleton: [outline_template.json](assets/outline_template.json)
- Validator: `scripts/validate_outline.py`
