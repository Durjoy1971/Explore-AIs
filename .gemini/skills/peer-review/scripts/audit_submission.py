#!/usr/bin/env python3
"""Deterministic submission audit, optional LaTeX compile, and score-gate evaluator."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

SUB_AXES = [
    "Originality",
    "Quality",
    "Clarity",
    "Significance",
    "Soundness",
    "Presentation",
    "Contribution",
]


def check_environments(tex: str) -> list[str]:
    errors: list[str] = []
    stack: list[str] = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", tex):
        kind, env = m.group(1), m.group(2).strip()
        if kind == "begin":
            stack.append(env)
        else:
            if not stack:
                errors.append(f"Unexpected \\end{{{env}}}")
            elif stack[-1] != env:
                top = stack.pop()
                errors.append(f"Mismatched environment: \\begin{{{top}}} closed with \\end{{{env}}}")
            else:
                stack.pop()
    for unclosed in stack:
        errors.append(f"Unclosed environment: \\begin{{{unclosed}}}")
    return errors


def run_audit(
    tex_path: Path,
    citation_map_path: Path,
    bib_path: Path,
    experimental_log_path: Path,
    figures_info_path: Path | None,
    double_blind: bool,
    compile_pdf: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if not tex_path.exists():
        return {"status": "fail", "errors": [f"Missing TeX file: {tex_path}"], "warnings": []}

    tex = tex_path.read_text(encoding="utf-8")
    tex_clean = "\n".join(line.split("%", 1)[0] for line in tex.splitlines())

    # Citations & BibTeX
    cmap = json.loads(citation_map_path.read_text(encoding="utf-8")) if citation_map_path.exists() else {}
    bib_text = bib_path.read_text(encoding="utf-8") if bib_path.exists() else ""
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib_text))

    cite_groups = re.findall(r"\\cite[a-zA-Z*]*\s*(?:\[[^\]]*\]\s*)*\{([^}]+)\}", tex_clean)
    used_keys = sorted({k.strip() for grp in cite_groups for k in grp.split(",") if k.strip()})

    for key in used_keys:
        if key not in cmap or not cmap[key].get("verified"):
            errors.append(f"Unverified or missing citation in citation_map.json: {key}")
        if key not in bib_keys:
            errors.append(f"Citation missing from references.bib: {key}")

    # Figures
    manuscript_dir = tex_path.parent
    imgs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex_clean)
    for img in imgs:
        if not (manuscript_dir / img).exists():
            errors.append(f"Missing figure file: {manuscript_dir / img}")

    if figures_info_path and figures_info_path.exists():
        info = json.loads(figures_info_path.read_text(encoding="utf-8"))
        included_names = {Path(i).name for i in imgs}
        for item in info:
            if item.get("name") and item["name"] not in included_names:
                warnings.append(f"Registered figure not included in TeX: {item['name']}")

    # Table numeric grounding
    log_text = experimental_log_path.read_text(encoding="utf-8") if experimental_log_path.exists() else ""
    log_nums = set(re.findall(r"[-+]?\d+(?:\.\d+)?", log_text))
    table_nums: list[str] = []
    for block in re.findall(r"\\begin\{tabularx?\}(.*?)\\end\{tabularx?\}", tex_clean, flags=re.DOTALL):
        body = re.sub(r"^\s*(?:\[[^\]]*\]\s*)?(?:\{[^}]*\}\s*){1,2}", "", block)
        table_nums.extend(re.findall(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?=%|\b)", body))
    ungrounded = sorted({n for n in table_nums if n not in log_nums})
    if ungrounded:
        errors.append(f"Ungrounded table numbers not in experimental_log.md: {ungrounded}")

    # Environments & integrity sections
    errors.extend(check_environments(tex_clean))
    if "cleverref" in tex_clean:
        errors.append("Invalid package name 'cleverref' (should be 'cleveref').")
    if "limitations" not in tex_clean.lower():
        warnings.append("No Limitations section or discussion detected.")
    if "reproducibility" not in tex_clean.lower():
        warnings.append("No Reproducibility statement detected.")

    if double_blind:
        if "corresponding author" in tex_clean.lower():
            errors.append("Double-blind violation: 'corresponding author' found.")
        if re.search(r"https?://github\.com/", tex_clean, flags=re.IGNORECASE):
            errors.append("Double-blind violation: non-anonymous GitHub link found.")

    compile_status = "skipped"
    if compile_pdf:
        pdflatex = shutil.which("pdflatex")
        bibtex = shutil.which("bibtex")
        if not pdflatex or not bibtex:
            compile_status = "unavailable_no_pdflatex"
            warnings.append("pdflatex/bibtex not found on PATH; skipped PDF compilation.")
        else:
            stem = tex_path.stem
            cmds = [
                [pdflatex, "-interaction=nonstopmode", f"{stem}.tex"],
                [bibtex, stem],
                [pdflatex, "-interaction=nonstopmode", f"{stem}.tex"],
                [pdflatex, "-interaction=nonstopmode", f"{stem}.tex"],
            ]
            compile_status = "success"
            for cmd in cmds:
                proc = subprocess.run(cmd, cwd=manuscript_dir, capture_output=True, text=True, timeout=45)
                if proc.returncode != 0:
                    compile_status = "failed"
                    errors.append(f"LaTeX command failed: {' '.join(cmd)}")
                    break

    return {
        "status": "pass" if not errors else "fail",
        "checked_citations": len(used_keys),
        "checked_figures": len(imgs),
        "checked_table_numbers": len(table_nums),
        "compile_status": compile_status,
        "errors": errors,
        "warnings": warnings,
    }


def evaluate_score_gate(before_path: Path, after_path: Path) -> dict[str, Any]:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))

    old_overall = float(before.get("Overall", 0))
    new_overall = float(after.get("Overall", 0))

    deltas: dict[str, float] = {}
    gain = 0.0
    drop = 0.0
    for axis in SUB_AXES:
        d = float(after.get(axis, 0)) - float(before.get(axis, 0))
        deltas[axis] = d
        if d > 0:
            gain += d
        elif d < 0:
            drop += abs(d)

    if new_overall > old_overall:
        decision = "ACCEPT_SCORE_INCREASE"
        accepted = True
    elif new_overall < old_overall:
        decision = "REJECT_SCORE_DECREASE"
        accepted = False
    elif drop > gain:
        decision = "REJECT_DEGRADATION"
        accepted = False
    else:
        decision = "ACCEPT_NEUTRAL_IMPROVEMENT"
        accepted = True

    return {
        "accepted": accepted,
        "decision": decision,
        "overall_before": old_overall,
        "overall_after": new_overall,
        "total_gain": gain,
        "total_drop": drop,
        "deltas": deltas,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit manuscript integrity or evaluate refinement score gate.")
    parser.add_argument("--tex", default=None, help="Path to manuscript/main.tex.")
    parser.add_argument("--citation-map", default=None, help="Path to artifacts/citation_map.json.")
    parser.add_argument("--bib", default=None, help="Path to manuscript/references.bib.")
    parser.add_argument("--experimental-log", default=None, help="Path to raw_materials/experimental_log.md.")
    parser.add_argument("--figures-info", default=None, help="Path to manuscript/figures/info.json.")
    parser.add_argument("--report-out", default=None, help="Optional output path for integrity_report.json.")
    parser.add_argument("--double-blind", action="store_true", help="Enforce double-blind anonymity checks.")
    parser.add_argument("--compile", action="store_true", help="Attempt pdflatex+bibtex compilation if installed.")
    parser.add_argument("--score-gate", nargs=2, metavar=("BEFORE_JSON", "AFTER_JSON"), help="Evaluate score-gated refinement.")
    args = parser.parse_args()

    if args.score_gate:
        gate = evaluate_score_gate(Path(args.score_gate[0]), Path(args.score_gate[1]))
        print(json.dumps(gate, indent=2))
        return 0 if gate["accepted"] else 1

    if not (args.tex and args.citation_map and args.bib and args.experimental_log):
        parser.error("--tex, --citation-map, --bib, and --experimental-log are required unless --score-gate is used.")

    report = run_audit(
        tex_path=Path(args.tex),
        citation_map_path=Path(args.citation_map),
        bib_path=Path(args.bib),
        experimental_log_path=Path(args.experimental_log),
        figures_info_path=Path(args.figures_info) if args.figures_info else None,
        double_blind=args.double_blind,
        compile_pdf=args.compile,
    )
    if args.report_out:
        out = Path(args.report_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
