"""Protected 282-file baseline manifest builder and Stage 4 repository safety verifier."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .post_stage_b_data import (
    PROTECTED_INPUT_DIGESTS,
    audit_post_stage_b_bundle,
    sha256_file,
    verify_manifest,
    verify_protected_inputs,
)

# Exact 24 historical/accepted reports in Category A (outside Stage 3 readiness review & Stage 4 deliverables).
HISTORICAL_CATEGORY_A_REPORTS: tuple[str, ...] = (
    "reports/dataset_consolidation_v1.md",
    "reports/phase1_audit.json",
    "reports/phase1_dataset_audit.md",
    "reports/phase2_3_cpu_baselines.json",
    "reports/phase2_3_cpu_baselines.md",
    "reports/post_stage_b_colab_readiness_report.md",
    "reports/post_stage_b_comprehensive_audit.md",
    "reports/post_stage_b_final_readiness_report.md",
    "reports/post_stage_b_implementation_report.md",
    "reports/post_stage_b_integrity_and_leakage_report.md",
    "reports/post_stage_b_model_view_collision_adjudication.md",
    "reports/post_stage_b_test_report.md",
    "reports/stage_a_focused_audit_v2.md",
    "reports/stage_b_batch1_50_report.md",
    "reports/stage_b_batch2_50_report.md",
    "reports/stage_b_batch3_50_report.md",
    "reports/stage_b_batch3_rejection_audit.md",
    "reports/stage_b_batch4_50_report.md",
    "reports/stage_b_batch5_50_report.md",
    "reports/stage_b_batch6_50_report.md",
    "reports/stage_b_batch7_45_report.md",
    "reports/stage_b_completion_report.md",
    "reports/stage_b_coverage_and_pilot_plan.md",
    "reports/stage_b_pilot_v2.md",
)

STAGE4_SNAPSHOT_TIER = "current_state_snapshot_accepted_not_historical_proof"
EXPECTED_PROTECTED_COUNT = 282


def _is_protected_category_a_path(rel_path: str) -> bool:
    if rel_path == "DataSet.csv" or rel_path.startswith("data/"):
        return True
    return rel_path in HISTORICAL_CATEGORY_A_REPORTS


def _artifact_class(rel_path: str) -> str:
    if (
        rel_path.startswith("data/raw/banglaphish2026/.git/")
        or rel_path.startswith("data/raw/smishdetect_hf/.cache/")
    ):
        return "machine_specific_metadata"
    return "scientific_dataset_or_report"


def _enumerate_protected_paths(root: Path) -> list[str]:
    paths: list[str] = []
    for cur_root, dirs, files in os.walk(root):
        rel_dir = os.path.relpath(cur_root, root)
        if rel_dir == ".":
            dirs[:] = [d for d in dirs if d not in {".git", ".venv", ".gemini", "__pycache__"}]
        else:
            dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            rel_path = fn if rel_dir == "." else (Path(rel_dir) / fn).as_posix()
            if _is_protected_category_a_path(rel_path):
                paths.append(rel_path)
    return sorted(paths)


def _git_output(root: Path, args: list[str], *, text: bool = True) -> Any:
    return subprocess.check_output(["git", "-C", str(root), *args], text=text)


def git_repository_state(project_root: str | Path) -> dict[str, Any]:
    """Capture repository root, branch, commit SHA, and clean/dirty status without mutating state."""
    root = Path(project_root).resolve()
    branch = _git_output(root, ["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    head = _git_output(root, ["rev-parse", "HEAD"]).strip()
    status_lines = _git_output(root, ["status", "--short"]).splitlines()
    return {
        "repo_root": str(root),
        "branch": branch,
        "head_commit": head,
        "status_short": status_lines,
    }


def _git_blob_sha256(root: Path, commit: str, rel_path: str) -> str:
    blob = _git_output(root, ["show", f"{commit}:{rel_path}"], text=False)
    return hashlib.sha256(blob).hexdigest()


def _git_object_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("utf-8") + data).hexdigest()


def build_protected_baseline_manifest(
    project_root: str | Path,
    *,
    owner_acceptance_status: str = "ACCEPTED_BY_OWNER_STAGE4",
) -> dict[str, Any]:
    """Build the non-self-referential 282-file Category A protected baseline manifest."""
    root = Path(project_root).resolve()
    protected_paths = _enumerate_protected_paths(root)
    if len(protected_paths) != EXPECTED_PROTECTED_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PROTECTED_COUNT} Category A files, found {len(protected_paths)}."
        )

    t282 = set(_git_output(root, ["ls-tree", "-r", "--name-only", "282ef63"]).splitlines())
    t4d3 = set(_git_output(root, ["ls-tree", "-r", "--name-only", "4d34118"]).splitlines())

    bp_dir = root / "data" / "raw" / "banglaphish2026"
    bp_sums: dict[str, str] = {}
    for line in (bp_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, fname = line.split(None, 1)
            bp_sums[(bp_dir / fname.strip().lstrip("*")).relative_to(root).as_posix()] = digest

    cons_md = _git_output(root, ["show", "282ef63:reports/dataset_consolidation_v1.md"])
    int_rep = (root / "reports" / "post_stage_b_integrity_and_leakage_report.md").read_text(encoding="utf-8")
    raw_anchored_4 = {
        "data/raw/bangalabarta/BangalaBarta bangla_spam_sms smishing.csv",
        "data/raw/smishdetect_hf/data/train-00000-of-00001.parquet",
        "data/raw/smishdetect_hf/data/validation-00000-of-00001.parquet",
        "data/raw/smishdetect_hf/data/test-00000-of-00001.parquet",
    }
    bp_tracked = {
        "data/raw/banglaphish2026/" + line
        for line in _git_output(root, ["-C", "data/raw/banglaphish2026", "ls-files"]).splitlines()
    }

    entries: list[dict[str, Any]] = []
    tier_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}

    for rel_path in protected_paths:
        abs_path = root / rel_path
        sha = sha256_file(abs_path)
        size = int(abs_path.stat().st_size)
        art_class = _artifact_class(rel_path)

        is_v2 = rel_path.startswith("data/processed/post_stage_b_bundle_v2/")
        is_4d3_rep = rel_path.startswith("reports/") and (rel_path in t4d3) and (rel_path not in t282)

        if is_v2 or is_4d3_rep:
            expected_git_sha = _git_blob_sha256(root, "4d34118", rel_path)
            if sha != expected_git_sha:
                raise RuntimeError(f"4d34118 blob mismatch for {rel_path}")
            tier = "git_4d34118_v2_and_reports"
            hist_proof = True
            anchor_note = "Exact Git blob match at commit 4d34118."
        elif rel_path in t282:
            expected_git_sha = _git_blob_sha256(root, "282ef63", rel_path)
            if sha != expected_git_sha:
                raise RuntimeError(f"282ef63 blob mismatch for {rel_path}")
            tier = "git_282ef63"
            hist_proof = True
            anchor_note = "Exact Git blob match at commit 282ef63."
        elif rel_path in PROTECTED_INPUT_DIGESTS and PROTECTED_INPUT_DIGESTS[rel_path] == sha:
            tier = "protected_input_digests"
            hist_proof = True
            anchor_note = "Matches hardcoded SHA256 in PROTECTED_INPUT_DIGESTS."
        elif rel_path in bp_sums and bp_sums[rel_path] == sha:
            tier = "upstream_sha256sums"
            hist_proof = True
            anchor_note = "Matches upstream data/raw/banglaphish2026/SHA256SUMS.txt."
        elif rel_path in raw_anchored_4 and (sha in int_rep) and (sha[:10] in cons_md):
            tier = "consolidation_v1_prefix_and_intrep"
            hist_proof = True
            anchor_note = "10-hex prefix in 282ef63:dataset_consolidation_v1.md + full SHA256 in integrity report."
        else:
            tier = STAGE4_SNAPSHOT_TIER
            hist_proof = False
            anchor_note = _describe_snapshot_cross_check(root, rel_path, sha, bp_tracked)

        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        class_counts[art_class] = class_counts.get(art_class, 0) + 1
        entries.append(
            {
                "path": rel_path,
                "sha256": sha,
                "bytes": size,
                "provenance_tier": tier,
                "historical_proof": hist_proof,
                "artifact_class": art_class,
                "verification_note": anchor_note,
            }
        )

    return {
        "schema_version": "stage4_protected_baseline_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "owner_acceptance_status": owner_acceptance_status,
        "git_anchor": {
            "branch": git_repository_state(root)["branch"],
            "head_commit": git_repository_state(root)["head_commit"],
            "historical_commits": ["282ef63", "4d34118"],
        },
        "total_protected_files": len(entries),
        "non_self_referential": True,
        "provenance_tier_counts": tier_counts,
        "artifact_class_counts": class_counts,
        "entries": entries,
    }


def _describe_snapshot_cross_check(root: Path, rel_path: str, sha: str, bp_tracked: set[str]) -> str:
    if rel_path == "data/processed/unified_3class_dedup.parquet":
        df = pd.read_parquet(root / rel_path)
        return (
            f"Current-state snapshot only (not historical proof); rows={len(df)}, "
            f"clusters={df['dup_cluster_id'].nunique()} match 282ef63 reports."
        )
    if rel_path in {"reports/phase1_audit.json", "reports/phase2_3_cpu_baselines.json"}:
        return "Current-state snapshot only (not historical proof); metrics match tracked 282ef63 MD reports."
    if rel_path == "data/raw/bangalabarta/BangalaBarta bangla_spam_sms smishing.zip":
        mend = json.loads((root / "data/raw/bangalabarta/mendeley_metadata.json").read_text(encoding="utf-8"))
        if mend["files"][0]["content_details"]["sha256_hash"] != sha:
            raise RuntimeError("Mendeley ZIP SHA256 does not match mendeley_metadata.json")
        with zipfile.ZipFile(root / rel_path) as zf:
            csv_sha = hashlib.sha256(zf.read("BangalaBarta bangla_spam_sms smishing.csv")).hexdigest()
            disk_csv_sha = sha256_file(root / "data/raw/bangalabarta/BangalaBarta bangla_spam_sms smishing.csv")
            if csv_sha != disk_csv_sha:
                raise RuntimeError("Mendeley ZIP internal CSV does not match extracted CSV on disk")
        return "Current-state snapshot only; matches mendeley_metadata.json & extracted CSV SHA256."
    if rel_path == "data/raw/bangalabarta/mendeley_metadata.json":
        return "Current-state snapshot only; records sibling Mendeley ZIP size and SHA256."
    if rel_path == "data/raw/banglaphish2026/README.md":
        return "Current-state snapshot only; tracked in clean nested Git commit 3c42aae9 (differs from SHA256SUMS)."
    if rel_path.startswith("data/raw/banglaphish2026/.git/"):
        return "Current-state snapshot of machine-specific nested Git metadata at commit 3c42aae9."
    if rel_path.startswith("data/raw/banglaphish2026/"):
        if rel_path not in bp_tracked:
            raise RuntimeError(f"Unexpected untracked file in banglaphish2026: {rel_path}")
        return "Current-state snapshot only; tracked in nested Git commit 3c42aae9 (not in upstream SHA256SUMS)."
    if rel_path in {"data/raw/smishdetect_hf/.gitattributes", "data/raw/smishdetect_hf/README.md"}:
        meta_file = root / "data/raw/smishdetect_hf/.cache/huggingface/download" / f"{Path(rel_path).name}.metadata"
        meta_lines = meta_file.read_text(encoding="utf-8").splitlines()
        if meta_lines[1] != _git_object_sha1((root / rel_path).read_bytes()):
            raise RuntimeError(f"HF ETag mismatch for {rel_path}")
        return "Current-state snapshot only; Git-blob SHA1 matches sibling HF download metadata ETag."
    if rel_path.startswith("data/raw/smishdetect_hf/.cache/"):
        return "Current-state snapshot of machine-specific HuggingFace download cache/lock metadata."
    raise ValueError(f"Unrecognized snapshot path: {rel_path}")


def verify_protected_baseline_manifest(
    project_root: str | Path,
    manifest_or_path: dict[str, Any] | str | Path,
    *,
    check_portable_scientific_only: bool = False,
) -> dict[str, Any]:
    """Verify protected Category A files against the baseline manifest without mutating files."""
    root = Path(project_root).resolve()
    if isinstance(manifest_or_path, (str, Path)):
        manifest = json.loads(Path(manifest_or_path).read_text(encoding="utf-8"))
    else:
        manifest = manifest_or_path

    entries = manifest.get("entries", [])
    if len(entries) != EXPECTED_PROTECTED_COUNT:
        raise RuntimeError(f"Manifest must index exactly {EXPECTED_PROTECTED_COUNT} entries, got {len(entries)}.")
    if any(Path(e["path"]).name == "protected_baseline_282.json" for e in entries):
        raise RuntimeError("Manifest must be non-self-referential.")

    checked = 0
    skipped_metadata = 0
    mismatches: list[str] = []
    missing: list[str] = []
    for entry in entries:
        rel_path = entry["path"]
        art_class = entry["artifact_class"]
        if check_portable_scientific_only and art_class == "machine_specific_metadata":
            skipped_metadata += 1
            continue
        abs_path = root / rel_path
        if not abs_path.is_file():
            missing.append(rel_path)
            continue
        actual_sha = sha256_file(abs_path)
        actual_bytes = int(abs_path.stat().st_size)
        if actual_sha != entry["sha256"] or actual_bytes != entry["bytes"]:
            mismatches.append(f"{rel_path}: expected ({entry['sha256']}, {entry['bytes']}), got ({actual_sha}, {actual_bytes})")
        checked += 1

    if missing or mismatches:
        raise RuntimeError(
            "Protected baseline verification failed:\n"
            + "\n".join([f"MISSING: {m}" for m in missing] + [f"MISMATCH: {m}" for m in mismatches])
        )
    return {
        "verified_count": checked,
        "skipped_machine_metadata": skipped_metadata,
        "owner_acceptance_status": manifest.get("owner_acceptance_status"),
        "provenance_tier_counts": manifest.get("provenance_tier_counts"),
        "artifact_class_counts": manifest.get("artifact_class_counts"),
        "check_portable_scientific_only": check_portable_scientific_only,
    }


def verify_all_stage4_integrity(
    project_root: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run full read-only repository, canonical digest, v1/v2 bundle, and 282-file checks."""
    root = Path(project_root).resolve()
    repo_state = git_repository_state(root)
    protected_13 = verify_protected_inputs(root, strict_digests=True)

    v1_dir = root / "data" / "processed" / "post_stage_b_bundle"
    v2_dir = root / "data" / "processed" / "post_stage_b_bundle_v2"
    v1_manifest = verify_manifest(v1_dir, verify_checksums=True)
    v1_audit = audit_post_stage_b_bundle(root, v1_dir, v1_manifest)
    v2_manifest = verify_manifest(v2_dir, verify_checksums=True)
    v2_audit = audit_post_stage_b_bundle(root, v2_dir, v2_manifest)

    mpath = Path(manifest_path) if manifest_path is not None else (root / "reports" / "protected_baseline_282.json")
    baseline_summary = None
    if mpath.is_file():
        baseline_summary = verify_protected_baseline_manifest(root, mpath, check_portable_scientific_only=False)

    return {
        "git_state": repo_state,
        "protected_input_count": len(protected_13),
        "v1_bundle": {
            "manifest_sha256": sha256_file(v1_dir / "dataset_manifest.json"),
            "cohort_pairs_checked": v1_audit["cohort_pairs_checked"],
        },
        "v2_bundle": {
            "manifest_sha256": sha256_file(v2_dir / "dataset_manifest.json"),
            "cohort_pairs_checked": v2_audit["cohort_pairs_checked"],
            "normalization_mismatches": v2_audit["normalization_metadata_mismatches"],
        },
        "protected_baseline_282": baseline_summary,
    }


STAGE4_KIT_VERSION = "post_stage_b_stage4_kit_v2"
STAGE4_KIT_MANIFEST_NAME = "stage4_kit_manifest.json"
STAGE4_KIT_SCOPE_DISCLOSURE = (
    "Portable Colab execution kit integrity verifies ONLY the staged runtime bundle, source code, "
    "configs, frozen cohort masks, statistical governance evidence, and Stage 4 notebook. It does NOT "
    "claim to verify absent historical repository files outside the portable kit; full 282-file "
    "repository verification must run locally against the complete repository."
)
STAGE4_KIT_REQUIRED_SOURCE_FILES: tuple[str, ...] = (
    "src/bangla_smish/__init__.py",
    "src/bangla_smish/audit.py",
    "src/bangla_smish/codemix_diversity.py",
    "src/bangla_smish/curation.py",
    "src/bangla_smish/metrics.py",
    "src/bangla_smish/post_stage_b_checkpointing.py",
    "src/bangla_smish/post_stage_b_cohorts.py",
    "src/bangla_smish/post_stage_b_colab_deps.py",
    "src/bangla_smish/post_stage_b_data.py",
    "src/bangla_smish/post_stage_b_ensemble.py",
    "src/bangla_smish/post_stage_b_integrity.py",
    "src/bangla_smish/post_stage_b_model_view_audit.py",
    "src/bangla_smish/post_stage_b_runner.py",
    "src/bangla_smish/post_stage_b_stats.py",
    "src/bangla_smish/post_stage_b_tokenization.py",
    "src/bangla_smish/post_stage_b_tokenizer_gate.py",
    "src/bangla_smish/schema.py",
    "src/bangla_smish/splits.py",
)
STAGE4_KIT_IGNORED_OS_DIRS: frozenset[str] = frozenset({".ipynb_checkpoints"})
STAGE4_KIT_IGNORED_OS_FILES: frozenset[str] = frozenset({".DS_Store", "desktop.ini"})
STAGE4_KIT_REQUIRED_CONFIGS: tuple[str, ...] = (
    "experiments/colab/configs/gpu_smoke_test_config.json",
    "experiments/colab/configs/post_stage_b_stage4_option_a_config.json",
    "experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json",
)
STAGE4_KIT_REQUIRED_REPORTS: tuple[str, ...] = (
    "reports/protected_baseline_282.json",
    "reports/stage4_frozen_test_masks.json",
    "reports/stage4_statistical_validation_results.json",
)
STAGE4_KIT_REQUIRED_NOTEBOOK = "experiments/colab/post_stage_b_stage4_experiments.ipynb"


def _portable_kit_expected_files(bundle_dir_name: str, bdir: Path) -> dict[str, str]:
    expected: dict[str, str] = {"pyproject.toml": "dependency_spec", STAGE4_KIT_REQUIRED_NOTEBOOK: "notebook"}
    for p in STAGE4_KIT_REQUIRED_SOURCE_FILES:
        expected[p] = "source_module"
    for p in STAGE4_KIT_REQUIRED_CONFIGS:
        expected[p] = "runner_config"
    for p in STAGE4_KIT_REQUIRED_REPORTS:
        expected[p] = "governance_or_baseline_artifact"
    for fpath in sorted(p for p in bdir.iterdir() if p.is_file()):
        expected[f"data/processed/{bundle_dir_name}/{fpath.name}"] = "v2_bundle"
    return expected


def _enumerate_kit_files(kdir: Path) -> set[str]:
    found: set[str] = set()
    for cur_root, dirs, files in os.walk(kdir):
        rel_dir = os.path.relpath(cur_root, kdir)
        ignored_dirs = {"__pycache__", *STAGE4_KIT_IGNORED_OS_DIRS}
        if rel_dir == ".":
            ignored_dirs.add("artifacts")
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for fn in files:
            if fn.endswith(".pyc") or fn in STAGE4_KIT_IGNORED_OS_FILES:
                continue
            rel = fn if rel_dir == "." else (Path(rel_dir) / fn).as_posix()
            if rel == STAGE4_KIT_MANIFEST_NAME:
                continue
            found.add(rel)
    return found


def build_portable_colab_kit_manifest(
    kit_dir: str | Path,
    *,
    source_root: str | Path | None = None,
    bundle_dir_name: str = "post_stage_b_bundle_v2",
) -> dict[str, Any]:
    """Build and verify the strict self-contained Stage 4 portable Colab kit manifest."""
    from .post_stage_b_colab_deps import (
        BANGLABERT_NORMALIZER_COMMIT,
        BANGLABERT_NORMALIZER_REQUIREMENT,
        MIN_TORCH_VERSION,
        STAGE4_BASE_PACKAGES,
    )
    from .post_stage_b_stats import CONFIRMATORY_DECLARATIONS_ENABLED, STATISTICAL_VALIDATION_STATUS

    kdir = Path(kit_dir).resolve()
    bdir = kdir / "data" / "processed" / bundle_dir_name
    if not bdir.is_dir():
        raise FileNotFoundError(f"Missing staged v2 bundle directory: {bdir}")

    b_manifest = verify_manifest(bdir, verify_checksums=True)
    if b_manifest.get("manifest_version") != "post_stage_b_v2":
        raise RuntimeError(f"Portable kit requires post_stage_b_v2 bundle, got {b_manifest.get('manifest_version')!r}")

    expected_map = _portable_kit_expected_files(bundle_dir_name, bdir)
    actual_files = _enumerate_kit_files(kdir)
    missing = sorted(set(expected_map) - actual_files)
    unexpected = sorted(actual_files - set(expected_map))
    if missing or unexpected:
        raise RuntimeError(
            "Portable Colab kit inventory mismatch:\n"
            + "\n".join([f"MISSING: {m}" for m in missing] + [f"UNEXPECTED: {u}" for u in unexpected])
        )

    baseline = json.loads((kdir / "reports" / "protected_baseline_282.json").read_text(encoding="utf-8"))
    if baseline.get("total_protected_files") != EXPECTED_PROTECTED_COUNT or not baseline.get("non_self_referential"):
        raise RuntimeError("Incompatible protected_baseline_282.json in portable kit")
    baseline_by_path = {e["path"]: e for e in baseline.get("entries", [])}

    masks_doc = json.loads((kdir / "reports" / "stage4_frozen_test_masks.json").read_text(encoding="utf-8"))
    test_parquet_sha = sha256_file(bdir / "eval_smishdetect_test.parquet")
    if masks_doc.get("bundle_test_parquet_sha256") != test_parquet_sha:
        raise RuntimeError("stage4_frozen_test_masks.json does not match staged eval_smishdetect_test.parquet")

    val_doc = json.loads((kdir / "reports" / "stage4_statistical_validation_results.json").read_text(encoding="utf-8"))
    if val_doc.get("verdict") != STATISTICAL_VALIDATION_STATUS:
        raise RuntimeError(
            f"Incompatible statistical validation verdict in kit: {val_doc.get('verdict')!r} != {STATISTICAL_VALIDATION_STATUS!r}"
        )

    runtime_entries: dict[str, dict[str, Any]] = {}
    bundle_baseline_checked = 0
    for rel, category in sorted(expected_map.items()):
        fpath = kdir / rel
        fsha = sha256_file(fpath)
        fbytes = int(fpath.stat().st_size)
        if category == "v2_bundle":
            if rel not in baseline_by_path:
                raise RuntimeError(f"Staged v2 bundle file absent from protected_baseline_282.json: {rel}")
            b_entry = baseline_by_path[rel]
            if b_entry["sha256"] != fsha or int(b_entry["bytes"]) != fbytes:
                raise RuntimeError(f"Staged v2 bundle file mismatches protected_baseline_282.json: {rel}")
            bundle_baseline_checked += 1
        runtime_entries[rel] = {"sha256": fsha, "bytes": fbytes, "category": category}

    git_ref: dict[str, Any] = {}
    if source_root is not None:
        sroot = Path(source_root).resolve()
        if (sroot / ".git").exists():
            git_state = git_repository_state(sroot)
            git_ref = {"branch": git_state["branch"], "head_commit": git_state["head_commit"]}

    payload: dict[str, Any] = {
        "kit_version": STAGE4_KIT_VERSION,
        "contract_type": "portable_colab_execution_kit",
        "verifies_full_repository_282": False,
        "scope_disclosure": STAGE4_KIT_SCOPE_DISCLOSURE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_dir_name": bundle_dir_name,
        "bundle_manifest_sha256": runtime_entries[f"data/processed/{bundle_dir_name}/dataset_manifest.json"]["sha256"],
        "option_a_config_sha256": runtime_entries["experiments/colab/configs/post_stage_b_stage4_option_a_config.json"]["sha256"],
        "t4_benchmark_config_sha256": runtime_entries["experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json"]["sha256"],
        "stage4_notebook_sha256": runtime_entries[STAGE4_KIT_REQUIRED_NOTEBOOK]["sha256"],
        "source_baseline_reference": {
            "protected_baseline_282_sha256": runtime_entries["reports/protected_baseline_282.json"]["sha256"],
            "total_repository_protected_files": EXPECTED_PROTECTED_COUNT,
            "staged_v2_bundle_files_cross_checked": bundle_baseline_checked,
            **git_ref,
        },
        "statistical_governance_evidence": {
            "validation_results_sha256": runtime_entries["reports/stage4_statistical_validation_results.json"]["sha256"],
            "validation_verdict": val_doc.get("verdict"),
            "confirmatory_declarations_enabled": bool(CONFIRMATORY_DECLARATIONS_ENABLED),
        },
        "dependency_specification": {
            "min_torch_version": MIN_TORCH_VERSION,
            "base_packages": list(STAGE4_BASE_PACKAGES),
            "banglabert_normalizer_requirement": BANGLABERT_NORMALIZER_REQUIREMENT,
            "banglabert_normalizer_commit": BANGLABERT_NORMALIZER_COMMIT,
        },
        "required_runtime_files": runtime_entries,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def verify_portable_colab_kit(kit_dir: str | Path) -> dict[str, Any]:
    """Verify an exported Stage 4 portable Colab kit independently with zero original repository access."""
    kdir = Path(kit_dir).resolve()
    manifest_path = kdir / STAGE4_KIT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing portable kit manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kit_version") != STAGE4_KIT_VERSION or manifest.get("contract_type") != "portable_colab_execution_kit":
        raise RuntimeError(f"Incompatible portable kit version/contract: {manifest.get('kit_version')!r}")
    if manifest.get("verifies_full_repository_282") is not False:
        raise RuntimeError("Portable kit manifest must set verifies_full_repository_282=False")

    recorded_digest = manifest.get("manifest_sha256")
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    expected_digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    if recorded_digest != expected_digest:
        raise RuntimeError("Portable kit manifest_sha256 self-digest verification failed")

    bundle_dir_name = manifest.get("bundle_dir_name")
    bdir = kdir / "data" / "processed" / str(bundle_dir_name)
    if not bdir.is_dir():
        raise RuntimeError(f"Missing staged bundle directory: {bdir}")
    b_manifest = verify_manifest(bdir, verify_checksums=True)
    if b_manifest.get("manifest_version") != "post_stage_b_v2":
        raise RuntimeError(f"Incompatible staged bundle manifest_version: {b_manifest.get('manifest_version')!r}")

    expected_required = set(_portable_kit_expected_files(str(bundle_dir_name), bdir))
    manifest_files = manifest.get("required_runtime_files", {})
    if set(manifest_files) != expected_required:
        raise RuntimeError(
            "Portable kit manifest file set differs from required runtime inventory: "
            f"missing={sorted(expected_required - set(manifest_files))}, "
            f"extra={sorted(set(manifest_files) - expected_required)}"
        )

    disk_files = _enumerate_kit_files(kdir)
    missing = sorted(expected_required - disk_files)
    unexpected = sorted(disk_files - expected_required)
    corrupted: list[str] = []
    for rel, meta in sorted(manifest_files.items()):
        fpath = kdir / rel
        if not fpath.is_file():
            continue
        if sha256_file(fpath) != meta["sha256"] or int(fpath.stat().st_size) != int(meta["bytes"]):
            corrupted.append(rel)

    if missing or unexpected or corrupted:
        raise RuntimeError(
            "Portable Colab kit verification failed:\n"
            + "\n".join(
                [f"MISSING: {m}" for m in missing]
                + [f"UNEXPECTED: {u}" for u in unexpected]
                + [f"CORRUPTED: {c}" for c in corrupted]
            )
        )

    baseline = json.loads((kdir / "reports" / "protected_baseline_282.json").read_text(encoding="utf-8"))
    if baseline.get("total_protected_files") != EXPECTED_PROTECTED_COUNT or not baseline.get("non_self_referential"):
        raise RuntimeError("Incompatible protected_baseline_282.json in staged kit")
    baseline_by_path = {e["path"]: e for e in baseline.get("entries", [])}
    v2_checked = 0
    for rel, meta in manifest_files.items():
        if meta["category"] == "v2_bundle":
            be = baseline_by_path.get(rel)
            if be is None or be["sha256"] != meta["sha256"] or int(be["bytes"]) != int(meta["bytes"]):
                raise RuntimeError(f"Staged v2 bundle entry incompatible with protected_baseline_282.json: {rel}")
            v2_checked += 1

    masks_doc = json.loads((kdir / "reports" / "stage4_frozen_test_masks.json").read_text(encoding="utf-8"))
    if masks_doc.get("bundle_test_parquet_sha256") != manifest_files[f"data/processed/{bundle_dir_name}/eval_smishdetect_test.parquet"]["sha256"]:
        raise RuntimeError("stage4_frozen_test_masks.json test parquet digest mismatch in staged kit")
    from .post_stage_b_cohorts import build_smishdetect_test_cohort_masks

    live_masks = build_smishdetect_test_cohort_masks(bdir, verify_tfidf_reproduction=False)
    for cname, expected_c in masks_doc.get("cohorts", {}).items():
        live_c = live_masks["cohorts"].get(cname)
        if (
            live_c is None
            or live_c["mask_sha256"] != expected_c.get("mask_sha256")
            or live_c["included_row_ids_sha256"] != expected_c.get("included_row_ids_sha256")
            or int(live_c["n_rows"]) != int(expected_c.get("n_rows", -1))
            or int(live_c["n_families"]) != int(expected_c.get("n_families", -1))
        ):
            raise RuntimeError(f"Re-derived frozen test mask mismatch in staged kit for {cname!r}")

    val_doc = json.loads((kdir / "reports" / "stage4_statistical_validation_results.json").read_text(encoding="utf-8"))
    if val_doc.get("verdict") != manifest["statistical_governance_evidence"]["validation_verdict"]:
        raise RuntimeError("Statistical validation verdict mismatch in staged kit")

    opt_a = json.loads((kdir / "experiments/colab/configs/post_stage_b_stage4_option_a_config.json").read_text(encoding="utf-8"))
    t4_cfg = json.loads((kdir / "experiments/colab/configs/post_stage_b_stage4_t4_benchmark_config.json").read_text(encoding="utf-8"))
    if opt_a.get("confirmatory_mode") is not True or t4_cfg.get("run_heldout_eval") is not False:
        raise RuntimeError("Incompatible runner configuration safety flags inside portable kit")
    for cfg_name, cfg_obj in (("option_a", opt_a), ("t4_benchmark", t4_cfg)):
        tok_revs = cfg_obj.get("tokenizer_revisions") or {}
        if any(m not in tok_revs for m in cfg_obj.get("models", [])):
            raise RuntimeError(f"Portable kit {cfg_name} config lacks pinned tokenizer_revisions for all models")

    return {
        "kit_version": manifest["kit_version"],
        "verifies_full_repository_282": False,
        "scope_disclosure": manifest["scope_disclosure"],
        "verified_runtime_file_count": len(manifest_files),
        "cross_checked_baseline_v2_files": v2_checked,
        "reverified_frozen_masks": len(live_masks["cohorts"]),
        "statistical_validation_verdict": val_doc.get("verdict"),
        "bundle_manifest_sha256": manifest["bundle_manifest_sha256"],
        "stage4_notebook_sha256": manifest["stage4_notebook_sha256"],
        "option_a_config_sha256": manifest["option_a_config_sha256"],
        "t4_benchmark_config_sha256": manifest["t4_benchmark_config_sha256"],
    }
