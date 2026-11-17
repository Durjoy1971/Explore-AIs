#!/usr/bin/env python3
"""Initialize an isolated research manuscript project under research_workspace/."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_IDEA_TEMPLATE = """# Research Idea Summary

## Problem Statement
Describe the specific problem and why existing approaches fall short.

## Proposed Method
Detail your core algorithm, equations, and design choices.

## Core Contributions
- Contribution 1
- Contribution 2
"""

DEFAULT_LOG_TEMPLATE = """# Experimental Log

## Datasets & Metrics
- Dataset(s):
- Evaluation Metric(s):

## Main Quantitative Results
| Method | Metric A | Metric B |
| :--- | ---: | ---: |
| Baseline | 0.0 | 0.0 |
| Ours | 0.0 | 0.0 |

## Ablations & Compute Cost
Record verified ablation numbers, seeds, hardware, and runtime observations here.
"""


def sanitize_slug(raw_slug: str) -> str:
    """Normalize a project slug into lowercase alphanumeric/underscore/hyphen."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw_slug.strip()).strip("-_").lower()
    if not cleaned:
        raise ValueError("Project slug cannot be empty after sanitization.")
    return cleaned


def init_project(
    workspace_root: Path,
    slug: str,
    venue: str = "iclr_single",
    cutoff_date: str | None = None,
    double_blind: bool = True,
    overwrite_templates: bool = False,
) -> Path:
    """Create the research_workspace/<slug> folder structure and manifest."""
    safe_slug = sanitize_slug(slug)
    project_dir = workspace_root / "research_workspace" / safe_slug

    subdirs = [
        project_dir / "raw_materials" / "data",
        project_dir / "raw_materials" / "figures",
        project_dir / "artifacts",
        project_dir / "manuscript" / "figures",
        project_dir / "manuscript" / "scripts",
        project_dir / "reviews",
    ]
    for d in subdirs:
        d.mkdir(parents=True, exist_ok=True)

    if not cutoff_date:
        cutoff_date = datetime.now(timezone.utc).strftime("%Y-%m")

    manifest_path = project_dir / "manifest.json"
    if not manifest_path.exists() or overwrite_templates:
        manifest = {
            "project_slug": safe_slug,
            "venue": venue,
            "cutoff_date": cutoff_date,
            "double_blind": double_blind,
            "model_policy": {
                "provider": "gemini_only",
                "reasoning_writer_model": "gemini-3.1-pro-preview",
                "fast_search_model": "gemini-3-flash-preview",
                "review_model": "gemini-3.1-pro-preview",
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    idea_path = project_dir / "raw_materials" / "idea.md"
    if not idea_path.exists() or overwrite_templates:
        idea_path.write_text(DEFAULT_IDEA_TEMPLATE, encoding="utf-8")

    log_path = project_dir / "raw_materials" / "experimental_log.md"
    if not log_path.exists() or overwrite_templates:
        log_path.write_text(DEFAULT_LOG_TEMPLATE, encoding="utf-8")

    return project_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Initialize a research_workspace/<slug> project separated from .gemini/skills/."
    )
    parser.add_argument("--slug", required=True, help="Project folder slug (e.g., bangla-smish-2026).")
    parser.add_argument(
        "--workspace-root",
        default=".",
        help="Root repository directory containing .gemini/ and research_workspace/.",
    )
    parser.add_argument(
        "--venue",
        default="iclr_single",
        choices=["iclr_single", "cvpr_two"],
        help="Target venue layout preset.",
    )
    parser.add_argument(
        "--cutoff-date",
        default=None,
        help="Literature temporal cutoff in YYYY-MM format (defaults to current UTC month).",
    )
    parser.add_argument(
        "--no-double-blind",
        action="store_true",
        help="Disable double-blind anonymity checks.",
    )
    parser.add_argument(
        "--overwrite-templates",
        action="store_true",
        help="Overwrite existing manifest.json, idea.md, and experimental_log.md.",
    )
    args = parser.parse_args()

    project_dir = init_project(
        workspace_root=Path(args.workspace_root).resolve(),
        slug=args.slug,
        venue=args.venue,
        cutoff_date=args.cutoff_date,
        double_blind=not args.no_double_blind,
        overwrite_templates=args.overwrite_templates,
    )
    print(json.dumps({"status": "initialized", "project_dir": str(project_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
