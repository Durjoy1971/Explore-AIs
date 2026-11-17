#!/usr/bin/env python3
"""Run a standalone plotting script in an isolated subprocess and register the output."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _upsert_json_list(path: Path, key_field: str, key_val: str, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                items = loaded
        except json.JSONDecodeError:
            items = []

    updated = [item for item in items if item.get(key_field) != key_val]
    updated.append(entry)
    path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")


def run_plot_script(
    script_path: Path | None,
    output_image: Path,
    figure_id: str,
    caption: str,
    info_json: Path,
    results_json: Path | None,
    python_bin: str = sys.executable,
    timeout_sec: int = 60,
    register_only: bool = False,
) -> dict[str, Any]:
    """Execute plotting script (unless register_only) and update info.json + plotting_results.json."""
    clean_caption = caption.strip()
    if clean_caption.lower().startswith("figure "):
        raise ValueError("Caption must not contain a 'Figure X:' prefix.")

    if not register_only:
        if not script_path or not script_path.exists():
            raise FileNotFoundError(f"Plotting script not found: {script_path}")

        env = dict(os.environ)
        env["MPLBACKEND"] = "Agg"
        proc = subprocess.run(
            [python_bin, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
        )
        if proc.returncode != 0:
            missing_mpl = "No module named 'matplotlib'" in proc.stderr
            return {
                "status": "error",
                "error_type": "matplotlib_missing" if missing_mpl else "script_execution_failed",
                "returncode": proc.returncode,
                "stdout": proc.stdout[-1000:],
                "stderr": proc.stderr[-1500:],
            }

    if not output_image.exists() or output_image.stat().st_size == 0:
        return {
            "status": "error",
            "error_type": "missing_output_image",
            "message": f"Expected non-empty image at {output_image}",
        }

    info_entry = {
        "name": output_image.name,
        "figure_id": figure_id,
        "caption": clean_caption,
    }
    _upsert_json_list(info_json, "name", output_image.name, info_entry)

    if results_json:
        res_entry = {
            "figure_id": figure_id,
            "image_path": f"figures/{output_image.name}",
            "script_path": str(script_path) if script_path else None,
            "caption": clean_caption,
            "size_bytes": output_image.stat().st_size,
        }
        _upsert_json_list(results_json, "figure_id", figure_id, res_entry)

    return {
        "status": "ok",
        "figure_id": figure_id,
        "output_image": str(output_image),
        "size_bytes": output_image.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute and register a scientific plot or diagram.")
    parser.add_argument("--script", default=None, help="Path to standalone plotting script.")
    parser.add_argument("--output-image", required=True, help="Expected rendered figure path.")
    parser.add_argument("--figure-id", required=True, help="Semantic figure_id from outline.json.")
    parser.add_argument("--caption", required=True, help="Plain-text caption without 'Figure X:'.")
    parser.add_argument("--info-json", required=True, help="Path to manuscript/figures/info.json.")
    parser.add_argument("--results-json", default=None, help="Optional path to artifacts/plotting_results.json.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python interpreter for subprocess.")
    parser.add_argument("--timeout", type=int, default=60, help="Subprocess timeout in seconds.")
    parser.add_argument("--register-only", action="store_true", help="Skip script execution and register existing image/diagram.")
    args = parser.parse_args()

    result = run_plot_script(
        script_path=Path(args.script) if args.script else None,
        output_image=Path(args.output_image),
        figure_id=args.figure_id,
        caption=args.caption,
        info_json=Path(args.info_json),
        results_json=Path(args.results_json) if args.results_json else None,
        python_bin=args.python_bin,
        timeout_sec=args.timeout,
        register_only=args.register_only,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
