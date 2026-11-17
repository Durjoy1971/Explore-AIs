---
name: literature-review
description: Executes decoupled literature discovery and API-grounded citation verification, builds verified BibTeX and citation maps, and drafts synthesized Introduction and Related Work LaTeX sections. Use when searching for academic papers, verifying citations against Semantic Scholar or Crossref, generating references.bib, or writing Introduction and Related Work.
version: 1.0.0
---

# Literature Review

Implements a two-stage decoupled literature pipeline inspired by PaperOrchestra's `HybridLiteratureAgent`: fast Gemini search-grounded discovery followed by deterministic API metadata verification and thematic synthesis.

## Anti-Fabrication Policy

- **Never cite unverified papers.** Every `\cite{key}` must resolve to an entry in `artifacts/citation_map.json` verified via Semantic Scholar, Crossref, arXiv, or a user-supplied local BibTeX/fixture registry.
- **No citation padding:** Cite papers because they genuinely support framing, baselines, datasets, metrics, or thematic synthesis—never to satisfy an arbitrary citation percentage quota.
- **Timeline & Evaluation Integrity:** Treat papers published after `cutoff_date` as excluded or concurrent (never as prior baselines to beat). Never claim SOTA superiority over a paper unless that paper is explicitly evaluated in `raw_materials/experimental_log.md`.

## Workflow

### Step 1 — Collect Search Tasks & Discover Candidates
Read `artifacts/outline.json` (`intro_related_work_plan` and `section_plan[*].subsections[*].citation_hints`).
Use Gemini with Search Grounding (Flash-tier model) to gather candidate `{title, year, section, reason}` objects into `artifacts/candidates_raw.json`.

### Step 2 — Verify Candidates & Build Citation Map
Run `scripts/verify_citations.py` sequentially against Semantic Scholar / Crossref (or `--mode offline --fixture <file>` for local/offline verification):
```bash
python3 .gemini/skills/literature-review/scripts/verify_citations.py \
  --candidates artifacts/candidates_raw.json \
  --cutoff-date YYYY-MM \
  --output artifacts/citation_map.json
```
Candidates failing fuzzy title verification, temporal cutoff, or missing metadata/abstracts are rejected and logged separately (`rejected_candidates` summary).

### Step 3 — Generate `references.bib` & `outline_v1.json`
Run `scripts/build_bib.py`:
```bash
python3 .gemini/skills/literature-review/scripts/build_bib.py \
  --citation-map artifacts/citation_map.json \
  --outline artifacts/outline.json \
  --bib-out manuscript/references.bib \
  --outline-out artifacts/outline_v1.json
```

### Step 4 — Synthesize Introduction & Related Work
Draft `\section{Introduction}` and `\section{Related Work}` in `manuscript/main.tex` (or `manuscript/sections/intro_related_work.tex` if modular):
- **Introduction:** Macro framing, hook, problem gap strictly scoped to `experimental_log.md`, and concise contribution bullets grounded in `idea.md`.
- **Related Work:** Group prior work into the thematic clusters from `outline_v1.json`, comparing tradeoffs and bridging to our method rather than listing one-sentence paper summaries.
- Use **only** citation keys present in `artifacts/citation_map.json`.

## References & Scripts
- Protocol details: [search-verification-protocol.md](references/search-verification-protocol.md)
- API verifier: `scripts/verify_citations.py`
- BibTeX & outline injector: `scripts/build_bib.py`
