"""Post-Stage B dataset auditing, bundle assembly, manifest verification, and fold-safe split builders."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .codemix_diversity import collect_brand_aliases
from .schema import (
    CANONICAL_LABELS,
    _PHONE_RE as _SCHEMA_PHONE_RE,
    _URL_RE as _SCHEMA_URL_RE,
    compile_brand_regex,
    extract_metadata_brands,
    mask_text_views,
    REAL_BD_BRANDS,
)
from .splits import perturb_zwj_homoglyph

LABEL_TO_IDX: dict[str, int] = {lbl: idx for idx, lbl in enumerate(CANONICAL_LABELS)}
TEXT_VIEW_COLUMNS: dict[str, str] = {
    "raw": "text_raw",
    "canon_masked": "text_canon_masked",
    "brand_masked": "text_brand_masked",
    "brand_stripped": "text_brand_stripped",
}
_REAL_BRAND_SWAP_RE = compile_brand_regex(REAL_BD_BRANDS)

PROTECTED_INPUT_DIGESTS: dict[str, str] = {
    "data/processed/curation_v2/eligible_train.parquet": "d9af8bf952558b91b45db5b5f721da2735ec4b18547cdca59d86e21a8b4993e7",
    "data/processed/curation_v2/existing_unique.parquet": "acb40c57870b8987b72a920e914d9e81966201caa26867456437a389d88b3e4f",
    "data/processed/curation_v2/provenance.parquet": "9a0fc3f2de54a7db4f040983504f5907b592dec751db9cb1dfcc07d99c21d443",
    "data/processed/curation_v2/quarantined.parquet": "c5cb0855abe2f2998e91d6affb8df7a2ae8984daeacb5373a63d4a1294b17cb4",
    "data/processed/banglaphish2026_dedup.parquet": "16aaa636aa8cf89cd07b8dd44887c11a7cc0398340c6698d36c2bbb91651c1f1",
    "data/generated/curation_v2/accepted.jsonl": "23af149826dec2411d79b61abceda44c564387574c97d7d12468d7f5a101c5ac",
    "data/generated/curation_v2/batch1_50/accepted.jsonl": "f6bc19c3b0466ac3ce327a51ee88f7e42c49668246a345183ee96c482c507dbd",
    "data/generated/curation_v2/batch2_50/accepted.jsonl": "5f4cb5afb8aaaf0ce37ab4cedb64639df6fb462d5f0f872e8444c7e85a6d32a6",
    "data/generated/curation_v2/batch3_50/accepted.jsonl": "abd5c9d931af3a2f9e84ac21d8a006554ae11bc17059729af19a33efe38b60a1",
    "data/generated/curation_v2/batch4_50/accepted.jsonl": "a97f0a912bad36ced9c5f104f2a4596a8b27f875a074bff4bf487cbce9ecbd64",
    "data/generated/curation_v2/batch5_50/accepted.jsonl": "a0cc662f85097f40767ddc78c85982fd50fab8d71897d76ce553742e6d3f6da1",
    "data/generated/curation_v2/batch6_50/accepted.jsonl": "ba4b6329c6a52ffcd6c233a76173eeb5c2ef713be9cbfb1174680dcef2eec682",
    "data/generated/curation_v2/batch7_45/accepted.jsonl": "5e7d237b9b12c08b68ea909b15563a95f2352f33e9a66d19986682f074c38438",
}

SYNTHETIC_BATCH_DIRS: tuple[str, ...] = (
    "batch1_50",
    "batch2_50",
    "batch3_50",
    "batch4_50",
    "batch5_50",
    "batch6_50",
    "batch7_45",
)

BUNDLE_FILENAMES: tuple[str, ...] = (
    "real_train_pool.parquet",
    "synthetic_pilot_300.parquet",
    "synthetic_batches_b1_b7.parquet",
    "eval_smishdetect_validation.parquet",
    "eval_smishdetect_test.parquet",
    "eval_banglaphish_ood.parquet",
    "counterfactual_smish_probe.parquet",
)


@dataclass(frozen=True)
class ExperimentSpec:
    """Definition of one Post-Stage B training cohort experiment."""

    experiment_id: str
    name: str
    description: str
    include_pilot: bool
    include_b1_b7: bool
    default_balance_mode: str  # "none", "downsample", or "class_weight"


EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {
    "A": ExperimentSpec(
        experiment_id="A",
        name="exp_a_unbalanced_real",
        description="Unbalanced Real Baseline (eligible_train only, full pool N=3,999).",
        include_pilot=False,
        include_b1_b7=False,
        default_balance_mode="none",
    ),
    "B": ExperimentSpec(
        experiment_id="B",
        name="exp_b_original_only_balanced",
        description="Original-Only Balanced (eligible_train downsampled to minority class per fold; 942/class full-pool).",
        include_pilot=False,
        include_b1_b7=False,
        default_balance_mode="downsample",
    ),
    "C": ExperimentSpec(
        experiment_id="C",
        name="exp_c_clean_stage_b",
        description="Clean Stage B Augmentation (eligible_train + Batches 1-7, N=4,344; synthetic added to train folds only).",
        include_pilot=False,
        include_b1_b7=True,
        default_balance_mode="none",
    ),
    "D": ExperimentSpec(
        experiment_id="D",
        name="exp_d_full_stage_b",
        description="Full Stage B Balanced Pool (eligible_train + Pilot 300 + Batches 1-7, N=4,644; synthetic added to train folds only).",
        include_pilot=True,
        include_b1_b7=True,
        default_balance_mode="none",
    ),
}


def sha256_file(path: Path) -> str:
    """Compute hex SHA256 digest of a file without modifying it."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _git_head_sha(project_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def verify_protected_inputs(
    project_root: str | Path,
    *,
    strict_digests: bool = True,
) -> dict[str, dict[str, Any]]:
    """Verify existence and SHA256 integrity of all 13 protected datasets and Stage B cohorts."""
    root = Path(project_root)
    inventory: dict[str, dict[str, Any]] = {}
    mismatches: list[str] = []
    for rel_path, expected_sha in PROTECTED_INPUT_DIGESTS.items():
        abs_path = root / rel_path
        if not abs_path.exists():
            raise FileNotFoundError(f"Protected input missing: {abs_path}")
        actual_sha = sha256_file(abs_path)
        matches = actual_sha == expected_sha
        if strict_digests and not matches:
            mismatches.append(f"{rel_path}: expected {expected_sha}, got {actual_sha}")
        inventory[rel_path] = {
            "sha256": actual_sha,
            "expected_sha256": expected_sha,
            "matches_baseline": matches,
            "bytes": abs_path.stat().st_size,
        }
    if mismatches:
        raise RuntimeError("Protected input SHA256 mismatch:\n" + "\n".join(mismatches))
    return inventory


def collect_post_stage_b_brand_vocabulary(
    project_root: str | Path,
    synthetic_records: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Collect unified known + pilot + Batches 1-7 brand aliases without mutating schema.py."""
    root = Path(project_root)
    merged_registry: dict[str, list[str]] = {}
    for batch_name in SYNTHETIC_BATCH_DIRS:
        reg_path = root / "data" / "generated" / "curation_v2" / batch_name / "brand_registry.json"
        if reg_path.exists():
            payload = json.loads(reg_path.read_text(encoding="utf-8"))
            brands_map = payload.get("brands", payload) if isinstance(payload, dict) else {}
            if isinstance(brands_map, dict):
                for key, aliases in brands_map.items():
                    merged_registry.setdefault(str(key), [])
                    if isinstance(aliases, list):
                        merged_registry[str(key)].extend(str(a) for a in aliases)
    alias_dict = collect_brand_aliases(
        synthetic_records or [],
        brand_registry=merged_registry,
    )
    meta_brands = set(extract_metadata_brands(synthetic_records or []))
    all_tokens = {alias for aliases in alias_dict.values() for alias in aliases} | meta_brands
    return sorted(all_tokens, key=len, reverse=True)


def _compute_view_columns(
    texts: list[str],
    *,
    brand_pattern: Any,
    brands: list[str],
    include_adv: bool = True,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {
        "text_raw": [],
        "text_canon_masked": [],
        "text_brand_masked": [],
        "text_brand_stripped": [],
    }
    if include_adv:
        out["text_adv"] = []
    for t in texts:
        views = mask_text_views(t, brands=brands, brand_pattern=brand_pattern)
        out["text_raw"].append(views["raw"])
        out["text_canon_masked"].append(views["canon_masked"])
        out["text_brand_masked"].append(views["brand_masked"])
        out["text_brand_stripped"].append(views["brand_stripped"])
        if include_adv:
            out["text_adv"].append(perturb_zwj_homoglyph(views["raw"], seed=42))
    return out


def _standardize_real_df(
    df: pd.DataFrame,
    *,
    cohort_name: str,
    brands: list[str],
    brand_pattern: Any,
) -> pd.DataFrame:
    texts = df["text"].fillna("").astype(str).tolist()
    view_cols = _compute_view_columns(texts, brand_pattern=brand_pattern, brands=brands, include_adv=True)
    fold_vals = (
        df["curation_cv_fold"].astype(int).tolist()
        if "curation_cv_fold" in df.columns
        else [-1] * len(df)
    )
    src_types = (
        df["source_type"].fillna("SMS").astype(str).tolist()
        if "source_type" in df.columns
        else ["SMS"] * len(df)
    )
    out = pd.DataFrame(
        {
            "row_id": df["canonical_id"].astype(str).tolist(),
            "text": texts,
            "text_norm_strict": df["text_norm_strict"].astype(str).tolist(),
            "label_3class": df["label_3class"].astype(str).tolist(),
            "label_idx": df["label_3class"].map(LABEL_TO_IDX).astype(int).tolist(),
            "language": df["language"].astype(str).tolist(),
            "family_id": df["family_id"].astype(str).tolist(),
            "is_synthetic": [False] * len(df),
            "cohort": [cohort_name] * len(df),
            "curation_cv_fold": fold_vals,
            "source_type": src_types,
            "char_len": [len(t) for t in texts],
            **view_cols,
        }
    )
    return out


def _standardize_synthetic_records(
    records: list[dict[str, Any]],
    *,
    default_cohort: str,
    brands: list[str],
    brand_pattern: Any,
) -> pd.DataFrame:
    texts = [str(r.get("text", "")) for r in records]
    view_cols = _compute_view_columns(texts, brand_pattern=brand_pattern, brands=brands, include_adv=True)
    labels = [str(r["label_3class"]) for r in records]
    return pd.DataFrame(
        {
            "row_id": [str(r["id"]) for r in records],
            "text": texts,
            "text_norm_strict": [str(r.get("text_norm_strict") or r["text"]).strip().lower() for r in records],
            "label_3class": labels,
            "label_idx": [LABEL_TO_IDX[lbl] for lbl in labels],
            "language": [str(r["language"]) for r in records],
            "family_id": [str(r["family_id"]) for r in records],
            "is_synthetic": [True] * len(records),
            "cohort": [str(r.get("generation_batch") or default_cohort) for r in records],
            "curation_cv_fold": [-1] * len(records),
            "source_type": ["SMS"] * len(records),
            "char_len": [len(t) for t in texts],
            **view_cols,
        }
    )


def _build_counterfactual_probe_df(
    real_train_df: pd.DataFrame,
    *,
    brands: list[str],
    brand_pattern: Any,
) -> pd.DataFrame:
    smish = real_train_df[real_train_df["label_3class"] == "Smishing"].copy()
    texts = smish["text"].fillna("").astype(str)
    has_contact = texts.str.contains(_SCHEMA_URL_RE, regex=True) | texts.str.contains(_SCHEMA_PHONE_RE, regex=True)
    probed = smish[has_contact].reset_index(drop=True)

    orig_texts = probed["text"].tolist()
    contact_swap: list[str] = []
    contact_brand_swap: list[str] = []
    for t in orig_texts:
        c_txt = _SCHEMA_URL_RE.sub("https://example.com/offer", t)
        c_txt = _SCHEMA_PHONE_RE.sub("01700-000011", c_txt)
        cb_txt = _REAL_BRAND_SWAP_RE.sub("PurbachalBank", c_txt)
        contact_swap.append(c_txt)
        contact_brand_swap.append(cb_txt)

    orig_views = _compute_view_columns(orig_texts, brand_pattern=brand_pattern, brands=brands, include_adv=False)
    c_views = _compute_view_columns(contact_swap, brand_pattern=brand_pattern, brands=brands, include_adv=False)
    cb_views = _compute_view_columns(contact_brand_swap, brand_pattern=brand_pattern, brands=brands, include_adv=False)

    out = pd.DataFrame(
        {
            "row_id": probed["row_id"].tolist(),
            "family_id": probed["family_id"].tolist(),
            "curation_cv_fold": probed["curation_cv_fold"].astype(int).tolist(),
            "language": probed["language"].tolist(),
            "label_idx": probed["label_idx"].astype(int).tolist(),
            "orig_text_raw": orig_views["text_raw"],
            "orig_text_canon_masked": orig_views["text_canon_masked"],
            "orig_text_brand_masked": orig_views["text_brand_masked"],
            "orig_text_brand_stripped": orig_views["text_brand_stripped"],
            "contact_swap_text_raw": c_views["text_raw"],
            "contact_swap_text_canon_masked": c_views["text_canon_masked"],
            "contact_swap_text_brand_masked": c_views["text_brand_masked"],
            "contact_swap_text_brand_stripped": c_views["text_brand_stripped"],
            "contact_brand_swap_text_raw": cb_views["text_raw"],
            "contact_brand_swap_text_canon_masked": cb_views["text_canon_masked"],
            "contact_brand_swap_text_brand_masked": cb_views["text_brand_masked"],
            "contact_brand_swap_text_brand_stripped": cb_views["text_brand_stripped"],
        }
    )
    return out


def _summarize_df_counts(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_rows": int(len(df)),
        "n_families": int(df["family_id"].nunique()),
        "by_class": {lbl: int((df["label_3class"] == lbl).sum()) for lbl in CANONICAL_LABELS},
        "by_language": {str(k): int(v) for k, v in df["language"].value_counts().sort_index().items()},
    }


def assemble_post_stage_b_bundle(
    project_root: str | Path,
    output_dir: str | Path,
    *,
    strict_digests: bool = True,
) -> dict[str, Any]:
    """Audit protected corpora + Stage B cohorts and export the Colab-ready verified dataset bundle."""
    root = Path(project_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    protected_inventory = verify_protected_inputs(root, strict_digests=strict_digests)

    eligible_df = pd.read_parquet(root / "data/processed/curation_v2/eligible_train.parquet")
    existing_df = pd.read_parquet(root / "data/processed/curation_v2/existing_unique.parquet")

    pilot_records = _load_jsonl(root / "data/generated/curation_v2/accepted.jsonl")
    b1_b7_records: list[dict[str, Any]] = []
    for batch_name in SYNTHETIC_BATCH_DIRS:
        recs = _load_jsonl(root / "data/generated/curation_v2" / batch_name / "accepted.jsonl")
        for r in recs:
            r.setdefault("generation_batch", batch_name)
        b1_b7_records.extend(recs)

    all_syn_records = pilot_records + b1_b7_records
    brands = collect_post_stage_b_brand_vocabulary(root, synthetic_records=all_syn_records)
    brand_pattern = compile_brand_regex(brands)

    real_train = _standardize_real_df(eligible_df, cohort_name="real_train", brands=brands, brand_pattern=brand_pattern)
    sd_val = _standardize_real_df(
        existing_df[existing_df["eval_group"] == "smishdetect_validation"].reset_index(drop=True),
        cohort_name="smishdetect_validation",
        brands=brands,
        brand_pattern=brand_pattern,
    )
    sd_test = _standardize_real_df(
        existing_df[existing_df["eval_group"] == "smishdetect_test"].reset_index(drop=True),
        cohort_name="smishdetect_test",
        brands=brands,
        brand_pattern=brand_pattern,
    )
    bp_ood = _standardize_real_df(
        existing_df[existing_df["eval_group"] == "banglaphish_ood"].reset_index(drop=True),
        cohort_name="banglaphish_ood",
        brands=brands,
        brand_pattern=brand_pattern,
    )
    syn_pilot = _standardize_synthetic_records(
        pilot_records, default_cohort="pilot_300", brands=brands, brand_pattern=brand_pattern
    )
    syn_b1_b7 = _standardize_synthetic_records(
        b1_b7_records, default_cohort="batches_1_7", brands=brands, brand_pattern=brand_pattern
    )
    cf_probe = _build_counterfactual_probe_df(real_train, brands=brands, brand_pattern=brand_pattern)

    # Leakage and count integrity assertions
    train_fams = set(real_train["family_id"])
    val_fams = set(sd_val["family_id"])
    test_fams = set(sd_test["family_id"])
    ood_fams = set(bp_ood["family_id"])
    pilot_fams = set(syn_pilot["family_id"])
    b1_b7_fams = set(syn_b1_b7["family_id"])

    zero_cross_eval_family_overlap = (
        train_fams.isdisjoint(val_fams)
        and train_fams.isdisjoint(test_fams)
        and train_fams.isdisjoint(ood_fams)
        and val_fams.isdisjoint(test_fams)
    )
    zero_synthetic_eval_family_overlap = (
        (pilot_fams | b1_b7_fams).isdisjoint(train_fams | val_fams | test_fams | ood_fams)
        and pilot_fams.isdisjoint(b1_b7_fams)
    )
    fold_family_leakage_max = int(real_train.groupby("family_id")["curation_cv_fold"].nunique().max())

    all_real_texts = set(existing_df["text_norm_strict"].astype(str))
    all_syn_texts = set(syn_pilot["text_norm_strict"]) | set(syn_b1_b7["text_norm_strict"])
    zero_synthetic_real_exact_overlap = len(all_real_texts & all_syn_texts) == 0

    full_balanced = pd.concat([real_train, syn_pilot, syn_b1_b7], ignore_index=True)
    balanced_class_counts = {lbl: int((full_balanced["label_3class"] == lbl).sum()) for lbl in CANONICAL_LABELS}
    exact_stage_b_parity = balanced_class_counts == {"Normal": 1548, "Promotional": 1548, "Smishing": 1548}

    integrity_checks = {
        "zero_cross_eval_family_overlap": bool(zero_cross_eval_family_overlap),
        "zero_synthetic_eval_family_overlap": bool(zero_synthetic_eval_family_overlap),
        "zero_synthetic_real_exact_overlap": bool(zero_synthetic_real_exact_overlap),
        "fold_family_leakage_max_eq_1": fold_family_leakage_max == 1,
        "exact_stage_b_parity_1548_per_class": bool(exact_stage_b_parity),
    }
    if not all(integrity_checks.values()):
        raise RuntimeError(f"Dataset integrity check failed: {integrity_checks}")

    tables = {
        "real_train_pool.parquet": real_train,
        "synthetic_pilot_300.parquet": syn_pilot,
        "synthetic_batches_b1_b7.parquet": syn_b1_b7,
        "eval_smishdetect_validation.parquet": sd_val,
        "eval_smishdetect_test.parquet": sd_test,
        "eval_banglaphish_ood.parquet": bp_ood,
        "counterfactual_smish_probe.parquet": cf_probe,
    }

    bundle_files_meta: dict[str, dict[str, Any]] = {}
    for fname, table in tables.items():
        fpath = out_dir / fname
        table.to_parquet(fpath, index=False)
        bundle_files_meta[fname] = {
            "sha256": sha256_file(fpath),
            "bytes": fpath.stat().st_size,
            "n_rows": int(len(table)),
            "columns": list(table.columns),
        }

    manifest = {
        "manifest_version": "post_stage_b_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head_sha(root),
        "synthetic_generation_status": "PERMANENTLY_STOPPED",
        "brand_vocabulary_count": len(brands),
        "protected_inputs": protected_inventory,
        "bundle_files": bundle_files_meta,
        "integrity_checks": integrity_checks,
        "cohort_counts": {
            "real_train_pool": _summarize_df_counts(real_train),
            "synthetic_pilot_300": _summarize_df_counts(syn_pilot),
            "synthetic_batches_b1_b7": _summarize_df_counts(syn_b1_b7),
            "full_stage_b_balanced_pool": _summarize_df_counts(full_balanced),
            "eval_smishdetect_validation": _summarize_df_counts(sd_val),
            "eval_smishdetect_test": _summarize_df_counts(sd_test),
            "eval_banglaphish_ood": _summarize_df_counts(bp_ood),
            "counterfactual_smish_probe": {"n_rows": int(len(cf_probe))},
        },
        "experiments": {
            exp_id: {
                "name": spec.name,
                "description": spec.description,
                "include_pilot": spec.include_pilot,
                "include_b1_b7": spec.include_b1_b7,
                "default_balance_mode": spec.default_balance_mode,
            }
            for exp_id, spec in EXPERIMENT_SPECS.items()
        },
    }

    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def verify_manifest(bundle_dir: str | Path, *, verify_checksums: bool = True) -> dict[str, Any]:
    """Load and cryptographically verify an assembled Post-Stage B dataset bundle."""
    bdir = Path(bundle_dir)
    manifest_path = bdir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset_manifest.json not found in {bdir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != "post_stage_b_v1":
        raise ValueError(f"Unsupported manifest_version: {manifest.get('manifest_version')}")
    checks = manifest.get("integrity_checks", {})
    if not checks or not all(bool(v) for v in checks.values()):
        raise RuntimeError(f"Manifest failed integrity checks: {checks}")

    bundle_files = manifest.get("bundle_files", {})
    for fname in BUNDLE_FILENAMES:
        fpath = bdir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing bundle file: {fpath}")
        if fname not in bundle_files:
            raise KeyError(f"Manifest missing metadata for {fname}")
        if verify_checksums:
            actual_sha = sha256_file(fpath)
            expected_sha = bundle_files[fname]["sha256"]
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"Checksum mismatch for {fname}: expected {expected_sha}, got {actual_sha}"
                )
    return manifest


def _downsample_balanced_df(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Deterministically downsample each class to the minority class count."""
    counts = df["label_3class"].value_counts()
    min_n = int(counts.min())
    parts = [
        df[df["label_3class"] == lbl].sample(n=min_n, random_state=seed + idx)
        for idx, lbl in enumerate(CANONICAL_LABELS)
    ]
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_experiment_fold_data(
    bundle_dir: str | Path,
    experiment_id: str,
    fold: int,
    *,
    seed: int = 42,
    balance_mode_override: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Construct leakage-free train and validation DataFrames for one experiment and CV fold."""
    if experiment_id not in EXPERIMENT_SPECS:
        raise ValueError(f"Unknown experiment_id={experiment_id!r}; expected one of {tuple(EXPERIMENT_SPECS)}")
    if fold not in (0, 1, 2, 3, 4):
        raise ValueError(f"Invalid fold={fold}; expected 0..4")

    bdir = Path(bundle_dir)
    spec = EXPERIMENT_SPECS[experiment_id]
    balance_mode = balance_mode_override or spec.default_balance_mode

    real_df = pd.read_parquet(bdir / "real_train_pool.parquet")
    tr_real = real_df[real_df["curation_cv_fold"] != fold].reset_index(drop=True)
    va_real = real_df[real_df["curation_cv_fold"] == fold].reset_index(drop=True)

    if balance_mode == "downsample":
        tr_real = _downsample_balanced_df(tr_real, seed=seed + fold * 101)

    train_parts = [tr_real]
    if spec.include_pilot:
        train_parts.append(pd.read_parquet(bdir / "synthetic_pilot_300.parquet"))
    if spec.include_b1_b7:
        train_parts.append(pd.read_parquet(bdir / "synthetic_batches_b1_b7.parquet"))

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed + fold).reset_index(drop=True)

    # Strict anti-leakage assertions
    if bool(va_real["is_synthetic"].any()):
        raise RuntimeError("Validation fold must contain real rows only.")
    if not set(train_df["family_id"]).isdisjoint(set(va_real["family_id"])):
        raise RuntimeError("Family ID leakage detected between train and validation folds.")

    class_weights = None
    if balance_mode == "class_weight":
        counts = train_df["label_idx"].value_counts().to_dict()
        total = float(len(train_df))
        n_cls = float(len(CANONICAL_LABELS))
        class_weights = [total / (n_cls * float(counts.get(i, 1))) for i in range(len(CANONICAL_LABELS))]

    meta = {
        "experiment_id": experiment_id,
        "experiment_name": spec.name,
        "fold": int(fold),
        "balance_mode": balance_mode,
        "n_train": int(len(train_df)),
        "n_train_real": int((~train_df["is_synthetic"]).sum()),
        "n_train_synthetic": int(train_df["is_synthetic"].sum()),
        "n_val": int(len(va_real)),
        "train_by_class": {lbl: int((train_df["label_3class"] == lbl).sum()) for lbl in CANONICAL_LABELS},
        "val_by_class": {lbl: int((va_real["label_3class"] == lbl).sum()) for lbl in CANONICAL_LABELS},
        "class_weights": class_weights,
    }
    return train_df, va_real, meta


def load_heldout_eval_sets(bundle_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load untouched validation, test, OOD, and counterfactual probe tables from the verified bundle."""
    bdir = Path(bundle_dir)
    return {
        "smishdetect_validation": pd.read_parquet(bdir / "eval_smishdetect_validation.parquet"),
        "smishdetect_test": pd.read_parquet(bdir / "eval_smishdetect_test.parquet"),
        "banglaphish_ood": pd.read_parquet(bdir / "eval_banglaphish_ood.parquet"),
        "counterfactual_smish_probe": pd.read_parquet(bdir / "counterfactual_smish_probe.parquet"),
    }
