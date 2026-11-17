#!/usr/bin/env python3
"""Deterministic integrity verification for a LaTeX scientific manuscript."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def check_latex_environments(tex: str) -> list[str]:
    """Verify every \\begin{env} matches its \\end{env} in proper LIFO order."""
    errors: list[str] = []
    stack: list[str] = []
    for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", tex):
        kind, env = match.group(1), match.group(2).strip()
        if kind == "begin":
            stack.append(env)
        else:
            if not stack:
                errors.append(f"Unexpected \\end{{{env}}} with empty environment stack.")
            elif stack[-1] != env:
                top = stack.pop()
                errors.append(f"Mismatched environment: opened \\begin{{{top}}} but closed with \\end{{{env}}}.")
            else:
                stack.pop()
    for unclosed in stack:
        errors.append(f"Unclosed LaTeX environment: \\begin{{{unclosed}}}.")
    return errors


def extract_table_numbers(tex: str) -> list[str]:
    """Extract numeric tokens inside tabular/tabularx blocks."""
    numbers: list[str] = []
    for block in re.findall(r"\\begin\{tabularx?\}(.*?)\\end\{tabularx?\}", tex, flags=re.DOTALL):
        # Strip column specifiers and LaTeX spacing args like [t] or {lcc}
        body = re.sub(r"^\s*(?:\[[^\]]*\]\s*)?(?:\{[^}]*\}\s*){1,2}", "", block)
        for token in re.findall(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?=%|\b)", body):
            numbers.append(token)
    return numbers


def check_manuscript_integrity(
    tex_path: Path,
    citation_map_path: Path,
    bib_path: Path,
    experimental_log_path: Path,
    figures_info_path: Path | None = None,
    data_dir: Path | None = None,
    double_blind: bool = False,
) -> dict[str, Any]:
    """Run all deterministic citation, figure, environment, numeric, and anonymity checks."""
    errors: list[str] = []
    warnings: list[str] = []

    if not tex_path.exists():
        return {"status": "fail", "errors": [f"Missing TeX file: {tex_path}"], "warnings": []}

    tex = tex_path.read_text(encoding="utf-8")
    tex_no_comments = "\n".join(line.split("%", 1)[0] for line in tex.splitlines())

    # 1. Citation key closure
    cmap = json.loads(citation_map_path.read_text(encoding="utf-8")) if citation_map_path.exists() else {}
    bib_text = bib_path.read_text(encoding="utf-8") if bib_path.exists() else ""
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib_text))

    cite_groups = re.findall(r"\\cite[a-zA-Z*]*\s*(?:\[[^\]]*\]\s*)*\{([^}]+)\}", tex_no_comments)
    used_keys = sorted({k.strip() for grp in cite_groups for k in grp.split(",") if k.strip()})

    for key in used_keys:
        if key not in cmap:
            errors.append(f"Citation key {key!r} is not in citation_map.json.")
        elif not cmap[key].get("verified"):
            errors.append(f"Citation key {key!r} is marked unverified in citation_map.json.")
        if key not in bib_keys:
            errors.append(f"Citation key {key!r} is missing from references.bib.")

    if not used_keys:
        warnings.append("No \\cite{...} commands detected in manuscript.")

    # 2. Figure references
    manuscript_dir = tex_path.parent
    included_imgs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex_no_comments)
    for img_rel in included_imgs:
        if not (manuscript_dir / img_rel).exists():
            errors.append(f"Included figure does not exist: {manuscript_dir / img_rel}")

    if figures_info_path and figures_info_path.exists():
        info = json.loads(figures_info_path.read_text(encoding="utf-8"))
        included_basenames = {Path(p).name for p in included_imgs}
        for item in info:
            name = item.get("name", "")
            if name and name not in included_basenames:
                warnings.append(f"Figure {name!r} in info.json is not embedded in {tex_path.name}.")

    # 3. LaTeX environment & macro hygiene
    errors.extend(check_latex_environments(tex_no_comments))
    if "cleverref" in tex_no_comments:
        errors.append("Typo detected: 'cleverref' should be 'cleveref'.")

    # 4. Numeric grounding against experimental_log.md and raw data
    source_corpus = experimental_log_path.read_text(encoding="utf-8") if experimental_log_path.exists() else ""
    if data_dir and data_dir.exists():
        for p in data_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".md", ".csv", ".json", ".txt"}:
                source_corpus += "\n" + p.read_text(encoding="utf-8", errors="ignore")

    source_numbers = set(re.findall(r"[-+]?\d+(?:\.\d+)?", source_corpus))
    table_numbers = extract_table_numbers(tex_no_comments)
    ungrounded = sorted({num for num in table_numbers if num not in source_numbers})
    if ungrounded:
        errors.append(f"Table numeric values not found in experimental_log.md/data: {ungrounded}")

    # 5. Double-blind anonymity check
    if double_blind:
        lowered = tex_no_comments.lower()
        if "corresponding author" in lowered:
            errors.append("Double-blind violation: 'corresponding author' found.")
        for url in re.findall(r"https?://github\.com/[^\s}]+", tex_no_comments, flags=re.IGNORECASE):
            errors.append(f"Double-blind violation: non-anonymous GitHub URL {url!r}.")

    return {
        "status": "pass" if not errors else "fail",
        "checked_citations": len(used_keys),
        "checked_figures": len(included_imgs),
        "checked_table_numbers": len(table_numbers),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify LaTeX manuscript integrity.")
    parser.add_argument("--tex", required=True, help="Path to manuscript/main.tex.")
    parser.add_argument("--citation-map", required=True, help="Path to artifacts/citation_map.json.")
    parser.add_argument("--bib", required=True, help="Path to manuscript/references.bib.")
    parser.add_argument("--experimental-log", required=True, help="Path to raw_materials/experimental_log.md.")
    parser.add_argument("--figures-info", default=None, help="Path to manuscript/figures/info.json.")
    parser.add_argument("--data-dir", default=None, help="Optional raw_materials/data directory.")
    parser.add_argument("--double-blind", action="store_true", help="Enable double-blind anonymity checks.")
    args = parser.parse_args()

    result = check_manuscript_integrity(
        tex_path=Path(args.tex),
        citation_map_path=Path(args.citation_map),
        bib_path=Path(args.bib),
        experimental_log_path=Path(args.experimental_log),
        figures_info_path=Path(args.figures_info) if args.figures_info else None,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        double_blind=args.double_blind,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
