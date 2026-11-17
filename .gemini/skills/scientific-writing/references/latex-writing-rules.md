# LaTeX & Scientific Writing Rules

## Technical Prose & Rigor
- Adopt a dense, objective, peer-reviewed ML/AI conference tone. Avoid hype adjectives ("revolutionary", "groundbreaking", "meticulously").
- Ground every equation and symbol in `idea.md`; define all variables on first appearance and keep notation consistent across text, tables, and figures.
- Report uncertainty/variance (standard deviations, 95% CIs, seeds, p-values) whenever present in `experimental_log.md`. Never fabricate error bars if single-run numbers were logged; state clearly that single-run values are reported.

## Tables (`booktabs`)
- Use `\toprule`, `\midrule`, `\bottomrule` with `tabular` or `tabularx`.
- Bold the best result per metric column only when clearly higher/lower is better, and indicate direction in headers (`$\uparrow$` or `$\downarrow$`).
- Place table captions **above** tables (single-column ICLR/NeurIPS style) or consistently per target venue rules.

## Figures
- Use `\includegraphics[width=\linewidth]{figures/<exact_filename>}` (or `\textwidth` inside `figure*`). Never set both fixed width and height.
- Never write `"Figure 1:"` inside `\caption{...}`; LaTeX generates figure numbers automatically.
- Ensure every `\begin{figure*}` closes with `\end{figure*}` (not `\end{figure}`).

## Package & Macro Safety
- Keep `\usepackage[capitalize]{cleveref}` spelled `cleveref` (never `cleverref`).
- Place `\appendix` strictly **after** the References (`\bibliography{...}`) on a new page (`\clearpage`).
