"""Read-only exact model-input overlap audit for the frozen Post-Stage B splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.post_stage_b_data import verify_manifest
from bangla_smish.post_stage_b_model_view_audit import audit_model_view_overlaps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    args = parser.parse_args()
    verify_manifest(args.bundle_dir, verify_checksums=True)
    result = audit_model_view_overlaps(args.bundle_dir)
    print(json.dumps(result, indent=2))
    return int(bool(result["failed_checks"]))


if __name__ == "__main__":
    raise SystemExit(main())
