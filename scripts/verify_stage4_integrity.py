"""Stage 4 CLI to write/verify the 282-file protected manifest, canonical inputs, v1/v2 bundles, and frozen test masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.post_stage_b_cohorts import build_smishdetect_test_cohort_masks, export_frozen_test_masks
from bangla_smish.post_stage_b_integrity import (
    build_protected_baseline_manifest,
    verify_all_stage4_integrity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Stage 4 protected inventory, bundles, and frozen test masks.")
    parser.add_argument(
        "--write-manifests",
        action="store_true",
        help="Write Category-C reports/protected_baseline_282.json and reports/stage4_frozen_test_masks.json.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify all 282 protected files, 13 canonical inputs, v1/v2 bundles, and frozen test masks without writing.",
    )
    parser.add_argument(
        "--skip-tfidf",
        action="store_true",
        help="Skip recomputing the 12,466-row TF-IDF screen when verifying masks quickly.",
    )
    args = parser.parse_args()

    manifest_path = ROOT / "reports" / "protected_baseline_282.json"
    masks_path = ROOT / "reports" / "stage4_frozen_test_masks.json"
    v2_dir = ROOT / "data" / "processed" / "post_stage_b_bundle_v2"

    if args.write_manifests:
        manifest = build_protected_baseline_manifest(ROOT, owner_acceptance_status="ACCEPTED_BY_OWNER_STAGE4")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        masks_payload = export_frozen_test_masks(v2_dir, masks_path, verify_tfidf_reproduction=not args.skip_tfidf)
        print(f"Wrote {manifest_path} ({manifest['total_protected_files']} protected entries)")
        print(f"Wrote {masks_path} ({len(masks_payload['cohorts'])} frozen cohorts)")

    summary = verify_all_stage4_integrity(ROOT, manifest_path if manifest_path.is_file() else None)
    if masks_path.is_file():
        expected_masks = json.loads(masks_path.read_text(encoding="utf-8"))
        live_masks = build_smishdetect_test_cohort_masks(v2_dir, verify_tfidf_reproduction=not args.skip_tfidf)
        for name, expected in expected_masks["cohorts"].items():
            live = live_masks["cohorts"][name]
            if live["mask_sha256"] != expected["mask_sha256"] or live["included_row_ids_sha256"] != expected["included_row_ids_sha256"]:
                raise RuntimeError(f"Frozen test mask mismatch for {name}")
        summary["frozen_test_masks"] = {
            name: {"n_rows": c["n_rows"], "n_families": c["n_families"], "mask_sha256": c["mask_sha256"]}
            for name, c in expected_masks["cohorts"].items()
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
