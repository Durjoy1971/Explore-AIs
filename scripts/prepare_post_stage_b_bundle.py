"""Local CPU dataset auditing, manifest generation, and Colab bundle assembly."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.post_stage_b_data import (
    assemble_post_stage_b_bundle,
    audit_post_stage_b_bundle,
    sha256_file,
    verify_manifest,
    verify_protected_inputs,
)


def export_drive_kit(project_root: Path, bundle_dir: Path, kit_dir: Path) -> Path:
    """Create a self-contained folder ready to upload to Google Drive (no credentials or raw data)."""
    kit_dir.mkdir(parents=True)

    shutil.copytree(
        project_root / "src" / "bangla_smish",
        kit_dir / "src" / "bangla_smish",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        project_root / "experiments" / "colab" / "configs",
        kit_dir / "experiments" / "colab" / "configs",
    )
    shutil.copy2(project_root / "pyproject.toml", kit_dir / "pyproject.toml")
    shutil.copytree(bundle_dir, kit_dir / "data" / "processed" / bundle_dir.name)
    return kit_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and assemble Post-Stage B Colab dataset bundle.")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Existing bundle to verify or new directory for assembly. Verification defaults to the frozen v1 bundle.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify protected inputs and an existing bundle without rebuilding.",
    )
    parser.add_argument(
        "--export-drive-kit",
        type=Path,
        default=None,
        help="Optional directory to stage a lightweight upload-ready Google Drive folder.",
    )
    args = parser.parse_args()
    if args.bundle_dir is None:
        if not args.verify_only:
            parser.error("--bundle-dir must name a new directory when assembling a bundle")
        args.bundle_dir = ROOT / "data" / "processed" / "post_stage_b_bundle"

    print("1) Verifying 13 protected inputs (SHA256)...")
    protected = verify_protected_inputs(ROOT, strict_digests=True)
    print(f"   Verified {len(protected)} protected inputs untouched.")

    if not args.verify_only:
        print(f"2) Auditing and assembling bundle into {args.bundle_dir}...")
        assemble_post_stage_b_bundle(ROOT, args.bundle_dir, strict_digests=True)
    manifest = verify_manifest(args.bundle_dir, verify_checksums=True)
    audit = audit_post_stage_b_bundle(ROOT, args.bundle_dir, manifest)

    print("3) Bundle & manifest verification summary:")
    print(json.dumps(
        {
            "manifest_version": manifest["manifest_version"],
            "manifest_sha256": sha256_file(args.bundle_dir / "dataset_manifest.json"),
            "brand_vocabulary_count": manifest["brand_vocabulary_count"],
            "integrity_checks": manifest["integrity_checks"],
            "cohort_counts": {k: v.get("by_class", v) for k, v in manifest["cohort_counts"].items()},
            "recomputed_audit": audit,
        },
        indent=2,
        ensure_ascii=False,
    ))

    if args.export_drive_kit is not None:
        staged = export_drive_kit(ROOT, args.bundle_dir, args.export_drive_kit)
        print(f"4) Exported upload-ready Google Drive kit to: {staged}")


if __name__ == "__main__":
    main()
