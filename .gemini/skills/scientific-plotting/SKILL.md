---
name: scientific-plotting
description: Designs, generates, critiques, and registers publication-grade scientific plots and conceptual diagrams from experimental logs and methodology notes. Use when generating charts from raw metrics/CSV/JSON data, writing reproducible Matplotlib scripts, creating TikZ/Mermaid/SVG architecture diagrams, or registering figure captions in figures/info.json.
version: 1.0.0
---

# Scientific Plotting

Translates `plotting_plan` entries from `artifacts/outline.json` and verified data from `raw_materials/` into reproducible, publication-ready visuals and captions.

## Design Rules (Research Integrity & Fidelity)

1. **Zero Data Fabrication:** Every coordinate, bar height, error bar, and annotation in a statistical plot must come directly from `raw_materials/experimental_log.md` or `raw_materials/data/`. Never interpolate or invent missing baselines or error bars.
2. **Standalone Reproducible Scripts (Plots):** Never rely on opaque in-memory `exec()`. Save each plot generator as `manuscript/scripts/<figure_id>.py`, load `assets/matplotlib_pub.mplstyle` when Matplotlib is available, and run it in an isolated subprocess via `scripts/run_plot_script.py`.
3. **Structured Vector Diagrams Preferred (Diagrams):** Prefer LaTeX TikZ, Graphviz DOT, or Mermaid/SVG diagrams for system architectures so text labels and module connections remain exact and editable. Only use Gemini image generation as an explicit opt-in, followed by a strict VLM label/hallucination audit.
4. **Caption Rules:** Captions stored in `manuscript/figures/info.json` must be concise plain text without a `"Figure X:"` prefix and without Markdown formatting (LaTeX handles numbering). Never embed the caption title inside the rendered image canvas when LaTeX `\caption{}` will render it below.

## Workflow

1. **Inspect Plan & Data:** Read `plotting_plan` in `artifacts/outline.json` and the referenced `data_source` (`experimental_log.md`, `idea.md`, or `raw_materials/data/`).
2. **Statistical Plots (`plot_type == "plot"`):**
   - Write a self-contained script at `manuscript/scripts/<figure_id>.py` that reads or embeds the exact verified data points and writes `manuscript/figures/<figure_id>.png` (or `.pdf`).
   - Execute and register via:
     ```bash
     python3 .gemini/skills/scientific-plotting/scripts/run_plot_script.py \
       --script manuscript/scripts/<figure_id>.py \
       --output-image manuscript/figures/<figure_id>.png \
       --figure-id <figure_id> \
       --caption "<plain-text caption>" \
       --info-json manuscript/figures/info.json \
       --results-json artifacts/plotting_results.json
     ```
3. **Conceptual Diagrams (`plot_type == "diagram"`):**
   - Author clean TikZ (embedded or standalone `.tex`/`.pdf`) or SVG/Mermaid source under `manuscript/scripts/<figure_id>.*`, render/validate, and register with `--register-only` in `scripts/run_plot_script.py`.
4. **Visual Critique Loop (Max 3 Rounds):**
   - Inspect the rendered figure with a Gemini multimodal model against [visual-style-guide.md](references/visual-style-guide.md) to catch overlapping labels, clipped legends, illegible fonts, or data mismatches.
   - Stop early if no issues remain.

## References, Assets & Scripts
- Style & critique checklist: [visual-style-guide.md](references/visual-style-guide.md)
- Publication Matplotlib stylesheet: [matplotlib_pub.mplstyle](assets/matplotlib_pub.mplstyle)
- Subprocess plot runner & registry: `scripts/run_plot_script.py`
