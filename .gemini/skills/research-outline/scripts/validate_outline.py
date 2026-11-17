#!/usr/bin/env python3
"""Validate outline.json structure, constraints, and evidence traceability."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

VALID_ASPECT_RATIOS = {
    "1:1", "1:4", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"
}
VALID_DATA_SOURCES = {"idea.md", "experimental_log.md", "both"}


def validate_outline(outline: dict) -> list[str]:
    """Return a list of schema/integrity errors; empty list means valid."""
    errors: list[str] = []

    for key in ("plotting_plan", "intro_related_work_plan", "section_plan"):
        if key not in outline:
            errors.append(f"Missing top-level key: {key}")

    plotting_plan = outline.get("plotting_plan", [])
    if not isinstance(plotting_plan, list):
        errors.append("plotting_plan must be a list.")
    else:
        seen_figs: set[str] = set()
        for idx, item in enumerate(plotting_plan):
            fig_id = str(item.get("figure_id", ""))
            if not fig_id:
                errors.append(f"plotting_plan[{idx}] missing figure_id.")
            elif "figure" in fig_id.lower():
                errors.append(f"plotting_plan[{idx}] figure_id {fig_id!r} must not contain 'figure'.")
            elif fig_id in seen_figs:
                errors.append(f"Duplicate figure_id: {fig_id!r}")
            seen_figs.add(fig_id)

            if item.get("plot_type") not in ("plot", "diagram"):
                errors.append(f"plotting_plan[{idx}] invalid plot_type: {item.get('plot_type')!r}")
            if item.get("data_source") not in VALID_DATA_SOURCES:
                errors.append(f"plotting_plan[{idx}] invalid data_source: {item.get('data_source')!r}")
            if item.get("aspect_ratio") not in VALID_ASPECT_RATIOS:
                errors.append(f"plotting_plan[{idx}] invalid aspect_ratio: {item.get('aspect_ratio')!r}")
            if not item.get("objective"):
                errors.append(f"plotting_plan[{idx}] missing objective.")

    irw = outline.get("intro_related_work_plan", {})
    if not isinstance(irw, dict):
        errors.append("intro_related_work_plan must be an object.")
    else:
        intro = irw.get("introduction_strategy", {})
        if not intro.get("hook_hypothesis") or not intro.get("problem_gap_hypothesis"):
            errors.append("introduction_strategy must include hook_hypothesis and problem_gap_hypothesis.")
        if not intro.get("search_directions"):
            errors.append("introduction_strategy must include non-empty search_directions.")

        rw_subs = irw.get("related_work_strategy", {}).get("subsections", [])
        if len(rw_subs) < 2:
            errors.append("related_work_strategy.subsections should define at least 2 methodology clusters.")

    section_plan = outline.get("section_plan", [])
    if not isinstance(section_plan, list) or not section_plan:
        errors.append("section_plan must be a non-empty list.")
    else:
        for sec in section_plan:
            sec_title = sec.get("section_title", "<unnamed>")
            subs = sec.get("subsections", [])
            if not subs:
                errors.append(f"Section {sec_title!r} has no subsections.")
                continue
            numbered = [s for s in subs if re.match(r"^\d+\.\d+\b", str(s.get("subsection_title", "")).strip())]
            if len(numbered) == 1:
                errors.append(f"Section {sec_title!r} has an orphaned numbered subsection: {numbered[0].get('subsection_title')!r}")
            for sub in subs:
                if not sub.get("content_bullets"):
                    errors.append(f"Subsection {sub.get('subsection_title')!r} has empty content_bullets.")
                if "citation_hints" not in sub:
                    errors.append(f"Subsection {sub.get('subsection_title')!r} missing citation_hints list.")
                if not sub.get("evidence_sources"):
                    errors.append(f"Subsection {sub.get('subsection_title')!r} missing evidence_sources traceability.")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate artifacts/outline.json.")
    parser.add_argument("--outline", required=True, help="Path to outline.json.")
    args = parser.parse_args()

    path = Path(args.outline)
    if not path.exists():
        print(json.dumps({"status": "fail", "errors": [f"File not found: {path}"]}, indent=2))
        return 1

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(json.dumps({"status": "fail", "errors": [f"Invalid JSON: {exc}"]}, indent=2))
        return 1

    errors = validate_outline(data)
    result = {"status": "pass" if not errors else "fail", "errors": errors}
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
