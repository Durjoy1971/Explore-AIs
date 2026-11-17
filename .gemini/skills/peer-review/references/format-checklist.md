# Formatting & Visual Layout Checklist

1. **Figures:**
   - Appear near or before first textual reference, and never after Conclusion (unless in Appendix).
   - Every figure in `manuscript/figures/info.json` is included individually (no merging figures).
   - Scaled proportionally (`width=1.0\linewidth` or `\textwidth` for `figure*`); never set both fixed width and height.
2. **Tables:**
   - Appear near or before first reference, never after Conclusion (unless in Appendix).
   - No column/margin overflow: apply `\small` / `\footnotesize`, abbreviate headers, or use `tabularx` with `X` columns.
   - All numeric values match `raw_materials/experimental_log.md`.
3. **Page Limit & Typography:**
   - Main text (up to References) complies with target venue limit.
   - If over limit, trim flowery adjectives or merge tiny paragraphs first—never delete core contributions, main results, Limitations, or Conclusion.
   - Fix orphan/widow headings and check balanced environments (`figure*` closed with `figure*`).
