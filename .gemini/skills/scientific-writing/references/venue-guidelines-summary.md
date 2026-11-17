# Venue Guidelines Summary

## Preset 1: Single-Column Conference Style (`iclr_single_col`)
- **Layout:** Single-column `article`, 10pt text, ~5.5in text width (`geometry` margin ~1.5in).
- **Page Limit:** Typically 9–10 pages main text (excluding References, Ethics/Reproducibility Statements, and Appendix).
- **Citations:** `natbib` (`\citet{key}` for narrative, `\citep{key}` for parenthetical).
- **Captions:** Below figures, above tables; sentence-case after first word.
- **Optional Pre-Reference Sections:** Ethics Statement, Reproducibility Statement.

## Preset 2: Two-Column Conference Style (`cvpr_two_col`)
- **Layout:** Two-column (`twocolumn`), 10pt text, tight column width (`width=1.0\linewidth` for single-column floats; `figure*`/`table*` only when wide multi-column spans are essential).
- **Page Limit:** Typically 8 pages main text (excluding References).
- **Citations:** Numeric/bracketed (`\cite{key}`) or `ieeenat_fullname` style.
- **Cross-References:** Use `\cref{fig:...}`, `\cref{tab:...}`, `\cref{sec:...}` (`cleveref`).
- **Double-Blind Policy:** Remove author names, emails, affiliations, grant IDs, and non-anonymous GitHub links.
