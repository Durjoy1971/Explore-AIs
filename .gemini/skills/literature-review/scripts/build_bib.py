#!/usr/bin/env python3
"""Generate references.bib and citation-enriched outline_v1.json from citation_map.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def format_bibtex_entry(key: str, entry: dict[str, Any]) -> str:
    authors = " and ".join(entry.get("authors") or ["Unknown"])
    title = entry.get("title", "Untitled")
    year = entry.get("year") or "0000"
    journal = entry.get("journal")
    venue = entry.get("venue") or "arXiv"

    entry_type = "article" if journal else "inproceedings"
    lines = [
        f"@{entry_type}{{{key},",
        f"  title={{{title}}},",
        f"  author={{{authors}}},",
    ]
    if journal:
        lines.append(f"  journal={{{journal}}},")
    else:
        lines.append(f"  booktitle={{{venue}}},")
    lines.append(f"  year={{{year}}},")

    if entry.get("volume"):
        lines.append(f"  volume={{{entry['volume']}}},")
    if entry.get("pages"):
        lines.append(f"  pages={{{entry['pages']}}},")
    if entry.get("doi"):
        lines.append(f"  doi={{{entry['doi']}}},")
    if entry.get("arxiv_id"):
        lines.append(f"  eprint={{{entry['arxiv_id']}}},")
        lines.append("  archivePrefix={arXiv},")

    # Strip trailing comma on last field
    lines[-1] = lines[-1].rstrip(",")
    lines.append("}")
    return "\n".join(lines)


def inject_citations_into_outline(outline: dict[str, Any], citation_map: dict[str, Any]) -> dict[str, Any]:
    section_to_keys: dict[str, list[str]] = {}
    for key, entry in citation_map.items():
        sec = entry.get("found_in_section", "General")
        section_to_keys.setdefault(sec, []).append(key)

    irw = outline.get("intro_related_work_plan", {})
    intro = irw.get("introduction_strategy", {})
    intro["citation_candidates"] = sorted(set(section_to_keys.get("Introduction", [])))

    for sub in irw.get("related_work_strategy", {}).get("subsections", []):
        sub_title = sub.get("subsection_title", "General")
        keys = section_to_keys.get(f"Related Work: {sub_title}", []) + section_to_keys.get(sub_title, [])
        sub["citation_candidates"] = sorted(set(keys))

    for sec in outline.get("section_plan", []):
        sec_title = sec.get("section_title", "")
        for sub in sec.get("subsections", []):
            sub_title = sub.get("subsection_title", "")
            combined_keys = (
                section_to_keys.get(f"{sec_title} - {sub_title}", [])
                + section_to_keys.get(sub_title, [])
            )
            sub["citation_candidates"] = sorted(set(combined_keys))

    return outline


def main() -> int:
    parser = argparse.ArgumentParser(description="Build references.bib and outline_v1.json.")
    parser.add_argument("--citation-map", required=True, help="Path to verified citation_map.json.")
    parser.add_argument("--bib-out", required=True, help="Path to output references.bib.")
    parser.add_argument("--outline", default=None, help="Optional input outline.json.")
    parser.add_argument("--outline-out", default=None, help="Optional output outline_v1.json.")
    args = parser.parse_args()

    cmap = json.loads(Path(args.citation_map).read_text(encoding="utf-8"))
    for key, entry in cmap.items():
        if not entry.get("verified"):
            raise ValueError(f"Refusing to write BibTeX for unverified entry: {key}")

    bib_entries = [format_bibtex_entry(key, entry) for key, entry in cmap.items()]
    bib_path = Path(args.bib_out)
    bib_path.parent.mkdir(parents=True, exist_ok=True)
    bib_path.write_text("\n\n".join(bib_entries) + ("\n" if bib_entries else ""), encoding="utf-8")

    if args.outline and args.outline_out:
        outline = json.loads(Path(args.outline).read_text(encoding="utf-8"))
        updated = inject_citations_into_outline(outline, cmap)
        out_outline = Path(args.outline_out)
        out_outline.parent.mkdir(parents=True, exist_ok=True)
        out_outline.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"status": "ok", "bib_entries": len(bib_entries), "bib_out": str(bib_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
