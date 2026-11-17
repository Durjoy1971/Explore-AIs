#!/usr/bin/env python3
"""Export compiled PDF pages into 2x2 grid PNG images for VLM layout inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_pdf_grids(pdf_path: Path, output_dir: Path, scale: float = 2.0) -> dict:
    try:
        import fitz  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as exc:
        return {
            "status": "unavailable",
            "reason": f"Optional PDF rendering dependency missing: {exc.name}",
            "grid_paths": [],
        }

    if not pdf_path.exists():
        return {"status": "error", "reason": f"PDF not found: {pdf_path}", "grid_paths": []}

    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(scale, scale)
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))

    grid_paths: list[str] = []
    for idx in range(0, len(pages), 4):
        chunk = pages[idx : idx + 4]
        w, h = chunk[0].size
        canvas = Image.new("RGB", (w * 2, h * 2), (255, 255, 255))
        positions = [(0, 0), (w, 0), (0, h), (w, h)]
        for img, pos in zip(chunk, positions):
            canvas.paste(img, pos)
        out_file = output_dir / f"grid_{idx // 4 + 1}.png"
        canvas.save(out_file)
        grid_paths.append(str(out_file))

    return {"status": "ok", "page_count": len(pages), "grid_paths": grid_paths}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export PDF pages to 2x2 grid PNGs.")
    parser.add_argument("--pdf", required=True, help="Path to compiled PDF.")
    parser.add_argument("--output-dir", required=True, help="Directory for grid PNGs.")
    parser.add_argument("--scale", type=float, default=2.0, help="Render scale factor.")
    args = parser.parse_args()

    res = export_pdf_grids(Path(args.pdf), Path(args.output_dir), scale=args.scale)
    print(json.dumps(res, indent=2))
    return 0 if res["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
