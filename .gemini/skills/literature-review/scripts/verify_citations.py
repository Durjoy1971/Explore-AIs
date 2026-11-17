#!/usr/bin/env python3
"""Verify candidate papers against Semantic Scholar / Crossref or an offline fixture."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "for", "to", "of", "and", "is", "are",
    "with", "by", "from", "via", "using", "study", "towards", "toward",
}


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


def title_similarity(query: str, candidate: str, year_hint: int | None, cand_year: int | None) -> float:
    ratio = difflib.SequenceMatcher(None, query.lower().strip(), candidate.lower().strip()).ratio()
    if year_hint and cand_year and int(year_hint) == int(cand_year):
        ratio = min(1.0, ratio + 0.08)
    return ratio


def is_before_cutoff(pub_date: str | None, year: int | None, cutoff_date: str | None) -> bool:
    if not cutoff_date:
        return True
    parts = [int(p) for p in cutoff_date.split("-")]
    c_year = parts[0]
    c_month = parts[1] if len(parts) > 1 else 12

    if pub_date:
        m = re.match(r"^(\d{4})(?:-(\d{2}))?", str(pub_date).strip())
        if m:
            p_year = int(m.group(1))
            p_month = int(m.group(2)) if m.group(2) else None
            if p_year != c_year:
                return p_year < c_year
            if p_month is not None:
                return p_month <= c_month
            return True

    if year is not None:
        return int(year) <= c_year
    return True


def generate_citation_key(authors: list[str], year: int | None, title: str, seen_keys: set[str]) -> str:
    if not authors:
        last_name = "Unknown"
    else:
        last_name = re.sub(r"[^a-zA-Z]", "", authors[0].split()[-1]).capitalize() or "Unknown"

    year_str = str(year) if year else "0000"
    words = re.sub(r"[^a-zA-Z0-9\s]", "", title.lower()).split()
    meaningful = [w.capitalize() for w in words if w not in STOPWORDS]
    title_part = "".join(meaningful[:2]) if meaningful else "Paper"
    base = f"{last_name}{year_str}{title_part}"

    key = base
    suffix_idx = 0
    while key in seen_keys:
        key = f"{base}{chr(ord('a') + suffix_idx)}"
        suffix_idx += 1
    seen_keys.add(key)
    return key


def _http_get_json(url: str, headers: dict[str, str] | None = None, timeout: int = 10) -> dict[str, Any] | None:
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "PaperOrchestra-Skills/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def search_semantic_scholar(title: str, year_hint: int | None, cutoff_date: str | None) -> dict[str, Any] | None:
    params = urllib.parse.urlencode({
        "query": title,
        "limit": 3,
        "fields": "paperId,externalIds,title,authors,venue,year,abstract,citationCount,journal,publicationDate",
    })
    headers = {"User-Agent": "PaperOrchestra-Skills/1.0"}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["X-API-KEY"] = api_key

    data = _http_get_json(f"https://api.semanticscholar.org/graph/v1/paper/search?{params}", headers=headers)
    if not data or not data.get("data"):
        return None

    best, best_score = None, 0.0
    for item in data["data"]:
        cand_title = item.get("title") or ""
        if not cand_title:
            continue
        if not is_before_cutoff(item.get("publicationDate"), item.get("year"), cutoff_date):
            continue
        score = title_similarity(title, cand_title, year_hint, item.get("year"))
        if score > best_score:
            best_score = score
            best = item

    if best and best_score >= 0.72:
        ext_ids = best.get("externalIds") or {}
        journal = best.get("journal") or {}
        return {
            "verification_source": "semantic_scholar",
            "paper_id": best.get("paperId"),
            "doi": ext_ids.get("DOI"),
            "arxiv_id": ext_ids.get("ArXiv"),
            "title": best["title"],
            "authors": [a.get("name", "") for a in best.get("authors", []) if a.get("name")],
            "venue": best.get("venue") or journal.get("name") or ("arXiv" if ext_ids.get("ArXiv") else "Unknown"),
            "year": best.get("year") or year_hint,
            "abstract": (best.get("abstract") or "").strip(),
            "citation_count": best.get("citationCount") or 0,
            "journal": journal.get("name"),
            "volume": journal.get("volume"),
            "pages": journal.get("pages"),
            "publication_date": best.get("publicationDate"),
        }
    return None


def search_crossref(title: str, year_hint: int | None, cutoff_date: str | None) -> dict[str, Any] | None:
    params = urllib.parse.urlencode({"query.title": title, "rows": 3})
    data = _http_get_json(f"https://api.crossref.org/works?{params}")
    items = (data or {}).get("message", {}).get("items", [])
    best, best_score = None, 0.0
    for item in items:
        titles = item.get("title") or []
        if not titles:
            continue
        cand_title = titles[0]
        date_parts = (item.get("issued") or {}).get("date-parts", [[None]])[0]
        cand_year = date_parts[0] if date_parts else None
        cand_month = date_parts[1] if len(date_parts) > 1 else None
        pub_date = f"{cand_year}-{cand_month:02d}" if cand_year and cand_month else None
        if not is_before_cutoff(pub_date, cand_year, cutoff_date):
            continue
        score = title_similarity(title, cand_title, year_hint, cand_year)
        if score > best_score:
            best_score = score
            best = (item, cand_title, cand_year, pub_date)

    if best and best_score >= 0.72:
        item, cand_title, cand_year, pub_date = best
        authors = [
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in item.get("author", [])
            if a.get("family") or a.get("given")
        ]
        container = (item.get("container-title") or ["Unknown"])[0]
        return {
            "verification_source": "crossref",
            "paper_id": item.get("DOI"),
            "doi": item.get("DOI"),
            "arxiv_id": None,
            "title": cand_title,
            "authors": authors,
            "venue": container,
            "year": cand_year or year_hint,
            "abstract": re.sub(r"<[^>]+>", "", item.get("abstract") or "").strip(),
            "citation_count": item.get("is-referenced-by-count") or 0,
            "journal": container if item.get("type") == "journal-article" else None,
            "volume": item.get("volume"),
            "pages": item.get("page"),
            "publication_date": pub_date,
        }
    return None


def search_offline_fixture(
    title: str, year_hint: int | None, cutoff_date: str | None, fixture_records: list[dict[str, Any]]
) -> dict[str, Any] | None:
    best, best_score = None, 0.0
    for rec in fixture_records:
        cand_title = rec.get("title", "")
        if not cand_title:
            continue
        if not is_before_cutoff(rec.get("publication_date"), rec.get("year"), cutoff_date):
            continue
        score = title_similarity(title, cand_title, year_hint, rec.get("year"))
        if score > best_score:
            best_score = score
            best = rec

    if best and best_score >= 0.72:
        return {
            "verification_source": best.get("verification_source", "offline_fixture"),
            "paper_id": best.get("paper_id") or best.get("doi") or best.get("arxiv_id"),
            "doi": best.get("doi"),
            "arxiv_id": best.get("arxiv_id"),
            "title": best["title"],
            "authors": best.get("authors", []),
            "venue": best.get("venue", "Unknown"),
            "year": best.get("year") or year_hint,
            "abstract": best.get("abstract", "").strip(),
            "citation_count": best.get("citation_count", 0),
            "journal": best.get("journal"),
            "volume": best.get("volume"),
            "pages": best.get("pages"),
            "publication_date": best.get("publication_date"),
        }
    return None


def verify_candidates(
    candidates: list[dict[str, Any]],
    cutoff_date: str | None,
    mode: str = "live",
    fixture_records: list[dict[str, Any]] | None = None,
    sleep_sec: float = 1.05,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    citation_map: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    seen_norm_titles: set[str] = set()
    seen_keys: set[str] = set()

    for idx, cand in enumerate(candidates):
        title = str(cand.get("title", "")).strip()
        if not title:
            rejected.append({"candidate": cand, "reason": "Empty title"})
            continue

        year_hint = cand.get("year")
        if mode == "offline":
            match = search_offline_fixture(title, year_hint, cutoff_date, fixture_records or [])
        else:
            if idx > 0 and sleep_sec > 0:
                time.sleep(sleep_sec)
            match = search_semantic_scholar(title, year_hint, cutoff_date)
            if match is None:
                match = search_crossref(title, year_hint, cutoff_date)

        if not match:
            rejected.append({"candidate": cand, "reason": "Unverified or exceeds cutoff_date"})
            continue

        if not any(match.get(k) for k in ("paper_id", "doi", "arxiv_id")):
            rejected.append({"candidate": cand, "reason": "Missing external identifier"})
            continue

        norm = normalize_title(match["title"])
        if norm in seen_norm_titles:
            continue
        seen_norm_titles.add(norm)

        key = generate_citation_key(match.get("authors", []), match.get("year"), match["title"], seen_keys)
        citation_map[key] = {
            "citation_key": key,
            "verified": True,
            "found_in_section": cand.get("section", "General"),
            "reason": cand.get("reason", ""),
            **match,
        }

    return citation_map, rejected


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify candidate citations and emit citation_map.json.")
    parser.add_argument("--candidates", required=True, help="Path to candidates_raw.json.")
    parser.add_argument("--output", required=True, help="Path to output citation_map.json.")
    parser.add_argument("--rejected-out", default=None, help="Optional path for rejected candidates JSON.")
    parser.add_argument("--cutoff-date", default=None, help="Temporal cutoff YYYY-MM.")
    parser.add_argument("--mode", default="live", choices=["live", "offline"], help="Verification mode.")
    parser.add_argument("--fixture", default=None, help="JSON fixture list when --mode offline is used.")
    parser.add_argument("--sleep-sec", type=float, default=1.05, help="Rate-limit delay in live mode.")
    args = parser.parse_args()

    candidates = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    fixture_records = json.loads(Path(args.fixture).read_text(encoding="utf-8")) if args.fixture else []

    citation_map, rejected = verify_candidates(
        candidates=candidates,
        cutoff_date=args.cutoff_date,
        mode=args.mode,
        fixture_records=fixture_records,
        sleep_sec=args.sleep_sec,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(citation_map, indent=2) + "\n", encoding="utf-8")

    if args.rejected_out:
        rej_path = Path(args.rejected_out)
        rej_path.parent.mkdir(parents=True, exist_ok=True)
        rej_path.write_text(json.dumps(rejected, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "ok",
        "verified_count": len(citation_map),
        "rejected_count": len(rejected),
        "output": str(out_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
