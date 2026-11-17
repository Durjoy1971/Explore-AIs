# Scientific Visual Style Guide

Adapted from top-tier ML/CV conference aesthetics (NeurIPS/ICLR/CVPR style standards).

## Statistical Plots
- **Palette:** Colorblind-safe Okabe-Ito / Tol muted palette (`#0072B2`, `#D55E00`, `#009E73`, `#CC79A7`, `#E69F00`, `#56B4E9`, `#4D4D4D`). Avoid default saturated primary red/green/blue.
- **Typography:** Sans-serif or serif matching paper font; axis labels >= 10 pt at print scale; tick labels >= 8.5 pt; clear units in parentheses (e.g., `Latency (ms)`).
- **Spines & Grids:** Remove top and right spines; use subtle dashed horizontal/xy gridlines (`alpha=0.25`).
- **Legends & Annotations:** Never occlude data points or bars. Prefer direct bar/line labels or uncluttered top/outside legends.
- **No Internal Figure Numbers:** Do not include `"Figure 1: ..."` inside the plot title or image body.

## Conceptual Diagrams
- **Hierarchy:** Left-to-right or top-to-bottom flow with consistent spacing and arrow weights.
- **Semantic Consistency:** Every module name, tensor shape, and loss symbol must match `idea.md` and `Section 3`.
- **Background:** Pure white (`#FFFFFF`) with subtle pastel container fills (`#F4F6F9`, `#EEF5FF`, `#FFF5EB`) and dark readable text (`#1A1A1A`).

## VLM Critique Checklist (Up to 3 Rounds)
1. **Data Fidelity:** Do plotted values and bar heights match `experimental_log.md`?
2. **Text QA:** Any typos, truncated axis labels, or gibberish text?
3. **Overlap Check:** Any colliding tick labels, legend boxes, or data annotations?
4. **Caption Exclusion:** Confirmed no `"Figure X:"` baked into the graphic?
