# Search & Citation Verification Protocol

## Stage 1: Candidate Discovery (Gemini Search-Grounded)
1. **Introduction tasks:** Macro-context queries (surveys, real-world motivation, foundational works).
2. **Related Work tasks:** Micro-level queries per methodology cluster + limitation queries from `outline.json`.
3. **Targeted `citation_hints`:** Exact 1-to-1 resolution of datasets, baselines, optimizers, metrics, and architectures named in `idea.md` or `experimental_log.md`.

Output format (`artifacts/candidates_raw.json`):
```json
[
  {
    "title": "Exact Candidate Paper Title",
    "year": 2024,
    "section": "Related Work: 2.1 Paradigm A",
    "reason": "Direct baseline evaluated in Table 1."
  }
]
```

## Stage 2: Deterministic Verification (`verify_citations.py`)
- Queries Semantic Scholar Graph API (`https://api.semanticscholar.org/graph/v1/paper/search`) sequentially (`1.05s` delay between requests to respect public rate limits, or `SEMANTIC_SCHOLAR_API_KEY` if configured).
- Falls back to Crossref (`https://api.crossref.org/works`) if Semantic Scholar returns no match or HTTP 429.
- Computes normalized similarity via `difflib.SequenceMatcher` (`>= 0.72`, with `+0.08` bonus when `year` matches).
- Rejects entries published after `cutoff_date` (`YYYY-MM`).
- Deduplicates by normalized title and Semantic Scholar/DOI ID.
- Assigns deterministic keys (`AuthorYearFirstTwoContentWords`, appending `a`, `b`, ... on collision).

## Stage 3: Writing Rules for Intro & Related Work
- Cite foundational claims in Introduction and comparative clusters in Related Work.
- Never extrapolate claims beyond what is supported by the verified `abstract` in `citation_map.json`.
- Keep all citations consistent with the venue style (`\cite{}` vs `\citet{}` / `\citep{}`).
