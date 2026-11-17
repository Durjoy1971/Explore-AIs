# Outline Schema Reference

`artifacts/outline.json` must contain three top-level keys: `plotting_plan`, `intro_related_work_plan`, and `section_plan`.

## 1. `plotting_plan` (List of Objects)
Each item requires:
- `figure_id` (string): Semantically meaningful identifier (e.g., `fig_ablation_latency`). Must NOT contain `"figure"` (case-insensitive).
- `title` (string): Concise figure title.
- `plot_type` (`"plot"` | `"diagram"`): Use `"plot"` for data-driven charts and `"diagram"` for conceptual/architecture visuals.
- `data_source` (`"idea.md"` | `"experimental_log.md"` | `"both"`).
- `objective` (string): Specific chart type (e.g., grouped bar chart, Pareto curve) and exact scientific takeaway.
- `aspect_ratio` (string): One of `1:1`, `1:4`, `2:3`, `3:2`, `3:4`, `4:1`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`.

## 2. `intro_related_work_plan` (Object)
- `introduction_strategy`:
  - `hook_hypothesis` (string)
  - `problem_gap_hypothesis` (string, strictly scoped to evaluated datasets/tasks)
  - `search_directions` (list of 3–5 query strings)
- `related_work_strategy`:
  - `overview` (string)
  - `subsections` (list of 2–4 cluster objects, each with `subsection_title`, `methodology_cluster`, `sota_investigation_mission`, `limitation_hypothesis`, `limitation_search_queries`, `bridge_to_our_method`)

## 3. `section_plan` (List of Section Objects)
Each section has:
- `section_title` (string)
- `subsections` (list of objects with `subsection_title`, `content_bullets`, `citation_hints`, and `evidence_sources` listing source file/heading references such as `idea.md#Method` or `experimental_log.md#Table1`).

## Integrity Rules
- **No Knowledge Leakage:** Do not recall external results or unpublished author details.
- **Anti-Hallucination Citation Hints:** Only include exact author+title when 100% verified; otherwise use `"research paper or technical report introducing '<Exact Artifact>'"`.
- **No Orphaned Subsections:** If a section uses numbered subsections (e.g., `3.1`), it must have at least two (`3.1`, `3.2`).
