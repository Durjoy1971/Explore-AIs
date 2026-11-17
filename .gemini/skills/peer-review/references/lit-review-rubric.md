# Skeptical Literature Review Rubric (0–100)

Adapted from PaperOrchestra (`autoraters/prompts/lit_review_quality_prompts.py`). Evaluates **only** the Introduction and Related Work sections.

## Anti-Inflation Rules
- Default expectation for unpolished drafts: `45–70`.
- Scores `> 85` require strong, quoted-evidence justification across all 6 axes.
- Mostly descriptive paper-by-paper summaries cap `Critical Analysis <= 60`.
- Vague novelty claims without close baseline comparison cap `Positioning <= 60`.
- Evaluate strictly what is written; do not infer author intent.

## Weighted Axes
1. **Coverage & Completeness (20%):** Breadth across relevant sub-threads, foundational + recent coverage, absence of obvious omissions.
2. **Relevance & Focus (15%):** Tight alignment with the stated research problem; no citation padding.
3. **Critical Analysis & Synthesis (25%):** Thematic grouping, comparison of tradeoffs/limitations rather than sequential lists.
4. **Positioning & Novelty Justification (25%):** Literature-grounded gap and explicit differentiation from closest prior work.
5. **Organization & Writing Quality (10%):** Logical flow, signposting, academic precision.
6. **Citation Rigor (5%):** Every key claim supported by verified, credible citations.
