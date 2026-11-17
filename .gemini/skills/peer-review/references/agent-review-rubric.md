# AgentReview Ensemble Rubric

Adapted from AgentReview / PaperOrchestra (`autoraters/agent_review.py`).

## Reviewer Ensemble & Meta-Review
Run 3 independent Gemini review passes using the **Knowledgeable + Responsible + Benign** reviewer persona, then aggregate with an **Area Chair Meta-Reviewer** and average integer axis scores across the 3 passes.

## Scoring Scales
- **1–4 Axes** (`Originality`, `Quality`, `Clarity`, `Significance`): 1=Low, 2=Medium, 3=High, 4=Very High.
- **1–4 Axes** (`Soundness`, `Presentation`, `Contribution`): 1=Poor, 2=Fair, 3=Good, 4=Excellent.
- **Overall (1–10):**
  - `10`: Top 2% award-quality paper; thorough and field-shaping.
  - `8`: Top 10%; strong support for all claims; highly original.
  - `6`: Sufficient support for major claims; moderate originality.
  - `5`: Main claims insufficiently supported or major technical gaps; leaning reject.
  - `3`: Marginal contributions.
  - `1`: Not thorough enough or out of scope.
- **Confidence (1–5):** 1=Low to 5=Absolute.
- **Decision:** Strictly `"Accept"` or `"Reject"`.

## Required Output JSON Schema
```json
{
  "Summary": "...",
  "Strengths": ["..."],
  "Weaknesses": ["..."],
  "Originality": 3,
  "Quality": 3,
  "Clarity": 3,
  "Significance": 3,
  "Questions": ["..."],
  "Limitations": ["..."],
  "Ethical Concerns": false,
  "Soundness": 3,
  "Presentation": 3,
  "Contribution": 3,
  "Overall": 6,
  "Confidence": 4,
  "Decision": "Accept"
}
```
