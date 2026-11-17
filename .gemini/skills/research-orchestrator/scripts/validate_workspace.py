#!/usr/bin/env python3
"""Validate research_workspace/<slug> structure and inter-stage integrity gates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> tuple[Any, str | None]:
    if not path.exists():
        return None, f"Missing file: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON in {path}: {exc}"


def check_init(project_dir: Path) -> list[str]:
    errors: list[str] = []
    if ".gemini" in project_dir.parts:
        errors.append("Research artifacts must not reside inside .gemini/.")

    for rel in [
        "manifest.json",
        "raw_materials/idea.md",
        "raw_materials/experimental_log.md",
        "artifacts",
        "manuscript/figures",
        "reviews",
    ]:
        if not (project_dir / rel).exists():
            errors.append(f"Missing required path: {rel}")

    manifest, err = _load_json(project_dir / "manifest.json")
    if err:
        errors.append(err)
    elif isinstance(manifest, dict):
        provider = manifest.get("model_policy", {}).get("provider")
        if provider != "gemini_only":
            errors.append(f"Expected model_policy.provider == 'gemini_only', got {provider!r}")
    return errors


def check_outline(project_dir: Path) -> list[str]:
    errors: list[str] = []
    outline, err = _load_json(project_dir / "artifacts" / "outline.json")
    if err:
        return [err]
    if not isinstance(outline, dict):
        return ["artifacts/outline.json must be a JSON object."]

    for key in ("plotting_plan", "intro_related_work_plan", "section_plan"):
        if key not in outline:
            errors.append(f"outline.json missing top-level key: {key}")

    for item in outline.get("plotting_plan", []):
        fig_id = str(item.get("figure_id", ""))
        if not fig_id or "figure" in fig_id.lower():
            errors.append(f"Invalid figure_id {fig_id!r}: must be non-empty and exclude 'figure'.")
        if item.get("plot_type") not in ("plot", "diagram"):
            errors.append(f"Invalid plot_type for {fig_id!r}: {item.get('plot_type')!r}")

    for sec in outline.get("section_plan", []):
        subs = sec.get("subsections", [])
        titles = [str(s.get("subsection_title", "")) for s in subs]
        if len(titles) == 1 and re.match(r"^\d+\.1\b", titles[0].strip()):
            errors.append(f"Orphaned subsection {titles[0]!r} in section {sec.get('section_title')!r}")
    return errors


def check_assets(project_dir: Path) -> list[str]:
    errors: list[str] = []
    cmap, err = _load_json(project_dir / "artifacts" / "citation_map.json")
    if err:
        errors.append(err)
    elif not isinstance(cmap, dict) or not cmap:
        errors.append("artifacts/citation_map.json must be a non-empty dictionary.")
    else:
        for key, entry in cmap.items():
            if not entry.get("verified"):
                errors.append(f"Unverified citation entry in citation_map.json: {key}")
            if not any(entry.get(k) for k in ("paper_id", "doi", "arxiv_id")):
                errors.append(f"Citation {key} missing external ID (paper_id, doi, or arxiv_id).")

    bib_path = project_dir / "manuscript" / "references.bib"
    if not bib_path.exists() or not bib_path.read_text(encoding="utf-8").strip():
        errors.append("manuscript/references.bib is missing or empty.")

    info_path = project_dir / "manuscript" / "figures" / "info.json"
    info, err = _load_json(info_path)
    if err:
        errors.append(err)
    elif isinstance(info, list):
        for fig in info:
            name = fig.get("name", "")
            fig_file = project_dir / "manuscript" / "figures" / name
            if not name or not fig_file.exists() or fig_file.stat().st_size == 0:
                errors.append(f"Missing or empty figure asset referenced in info.json: {name!r}")
    else:
        errors.append("manuscript/figures/info.json must be a JSON list.")
    return errors


def check_manuscript(project_dir: Path) -> list[str]:
    errors: list[str] = []
    tex_path = project_dir / "manuscript" / "main.tex"
    if not tex_path.exists():
        return ["Missing manuscript/main.tex"]
    tex = tex_path.read_text(encoding="utf-8")

    cmap, err = _load_json(project_dir / "artifacts" / "citation_map.json")
    bib_path = project_dir / "manuscript" / "references.bib"
    bib_text = bib_path.read_text(encoding="utf-8") if bib_path.exists() else ""
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib_text))

    cite_groups = re.findall(r"\\cite[a-zA-Z*]*\s*(?:\[[^\]]*\]\s*)*\{([^}]+)\}", tex)
    used_keys = {k.strip() for grp in cite_groups for k in grp.split(",") if k.strip()}
    if err or not isinstance(cmap, dict):
        errors.append(err or "Invalid citation_map.json")
    else:
        for key in sorted(used_keys):
            if key not in cmap:
                errors.append(f"Cited key {key!r} missing from citation_map.json")
            if key not in bib_keys:
                errors.append(f"Cited key {key!r} missing from references.bib")

    for img_ref in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        candidate = project_dir / "manuscript" / img_ref
        if not candidate.exists():
            errors.append(f"Missing figure file referenced in main.tex: {img_ref}")

    begins = re.findall(r"\\begin\{([^}]+)\}", tex)
    ends = re.findall(r"\\end\{([^}]+)\}", tex)
    if len(begins) != len(ends):
        errors.append(f"Unbalanced LaTeX environments: {len(begins)} \\begin vs {len(ends)} \\end.")
    return errors


def check_final(project_dir: Path) -> list[str]:
    errors: list[str] = []
    report, err = _load_json(project_dir / "reviews" / "integrity_report.json")
    if err:
        errors.append(err)
    elif report.get("status") != "pass":
        errors.append(f"integrity_report.json status is {report.get('status')!r}, expected 'pass'.")
    return errors


STAGE_CHECKS = {
    "init": check_init,
    "outline": check_outline,
    "assets": check_assets,
    "manuscript": check_manuscript,
    "final": check_final,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate research_workspace project gates.")
    parser.add_argument("--project-dir", required=True, help="Path to research_workspace/<slug>.")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "init", "outline", "assets", "manuscript", "final"],
        help="Stage gate to validate.",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    stages = list(STAGE_CHECKS.keys()) if args.stage == "all" else [args.stage]

    results: dict[str, list[str]] = {}
    all_ok = True
    for st in stages:
        errs = STAGE_CHECKS[st](project_dir)
        results[st] = errs
        if errs:
            all_ok = False

    output = {"status": "pass" if all_ok else "fail", "project_dir": str(project_dir), "stages": results}
    print(json.dumps(output, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
