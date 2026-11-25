"""Stage a versioned Stage 4 Colab execution kit without overwriting existing bundles or Drive workspaces."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.post_stage_b_data import verify_manifest, verify_protected_inputs
from bangla_smish.post_stage_b_integrity import (
    STAGE4_KIT_MANIFEST_NAME,
    STAGE4_KIT_REQUIRED_CONFIGS,
    STAGE4_KIT_REQUIRED_NOTEBOOK,
    STAGE4_KIT_REQUIRED_REPORTS,
    STAGE4_KIT_REQUIRED_SOURCE_FILES,
    build_portable_colab_kit_manifest,
    verify_portable_colab_kit,
)


def export_stage4_colab_kit(
    project_root: str | Path,
    bundle_dir: str | Path,
    kit_dir: str | Path,
) -> dict[str, Any]:
    """Export a fresh, self-contained Stage 4 portable Colab kit and immediately verify it."""
    root = Path(project_root).resolve()
    bdir = Path(bundle_dir).resolve()
    kdir = Path(kit_dir).resolve()
    if kdir.exists() or kdir.is_symlink():
        raise FileExistsError(f"Stage 4 Colab kit target must be a brand-new directory: {kdir}")

    verify_protected_inputs(root, strict_digests=True)
    manifest = verify_manifest(bdir, verify_checksums=True)
    if manifest.get("manifest_version") != "post_stage_b_v2":
        raise ValueError(f"Stage 4 Colab kit requires post_stage_b_v2 bundle, got {manifest.get('manifest_version')!r}")

    kdir.mkdir(parents=True)
    files_to_copy = (
        *STAGE4_KIT_REQUIRED_SOURCE_FILES,
        *STAGE4_KIT_REQUIRED_CONFIGS,
        *STAGE4_KIT_REQUIRED_REPORTS,
        STAGE4_KIT_REQUIRED_NOTEBOOK,
        "pyproject.toml",
    )
    for rel_path in files_to_copy:
        src = root / rel_path
        if not src.is_file():
            raise FileNotFoundError(f"Missing required Stage 4 runtime artifact in repository: {src}")
        dst = kdir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    shutil.copytree(bdir, kdir / "data" / "processed" / bdir.name)

    kit_manifest = build_portable_colab_kit_manifest(kdir, source_root=root, bundle_dir_name=bdir.name)
    (kdir / STAGE4_KIT_MANIFEST_NAME).write_text(json.dumps(kit_manifest, indent=2), encoding="utf-8")
    verify_portable_colab_kit(kdir)
    return kit_manifest


def verify_stage4_colab_kit(kit_dir: str | Path) -> dict[str, Any]:
    """Verify staged Stage 4 portable kit integrity independently of the source repository."""
    return verify_portable_colab_kit(kit_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or verify the Stage 4 versioned Colab execution kit.")
    parser.add_argument("--bundle-dir", type=Path, default=ROOT / "data" / "processed" / "post_stage_b_bundle_v2")
    parser.add_argument("--kit-dir", type=Path, required=True, help="New local staging directory for the Stage 4 kit.")
    parser.add_argument("--verify-only", action="store_true", help="Verify an already-staged Stage 4 kit directory.")
    args = parser.parse_args()

    if args.verify_only:
        res = verify_stage4_colab_kit(args.kit_dir)
    else:
        res = export_stage4_colab_kit(ROOT, args.bundle_dir, args.kit_dir)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
