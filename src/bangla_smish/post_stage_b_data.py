"""Post-Stage B dataset auditing, bundle assembly, manifest verification, and fold-safe split builders."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyarrow.parquet import ParquetFile

from .codemix_diversity import collect_brand_aliases
from .curation import strong_normalize_text
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
CV_FOLDS = (0, 1, 2, 3, 4)
BALANCE_MODES = ("none", "downsample", "class_weight")
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

EXPECTED_COHORT_COUNTS = {
    "real_train_pool.parquet": (1509, 942, 1548),
    "synthetic_pilot_300.parquet": (39, 261, 0),
    "synthetic_batches_b1_b7.parquet": (0, 345, 0),
    "eval_smishdetect_validation.parquet": (265, 178, 265),
    "eval_smishdetect_test.parquet": (581, 384, 733),
    "eval_banglaphish_ood.parquet": (2336, 80, 3000),
}
V1_INTEGRITY_CHECKS = {
    "zero_cross_eval_family_overlap",
    "zero_synthetic_eval_family_overlap",
    "zero_synthetic_real_exact_overlap",
    "fold_family_leakage_max_eq_1",
    "exact_stage_b_parity_1548_per_class",
}


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
        description="Class-Weighted Real Baseline (full eligible_train pool N=3,999).",
        include_pilot=False,
        include_b1_b7=False,
        default_balance_mode="class_weight",
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
        brands_map = json.loads(reg_path.read_text(encoding="utf-8"))["brands"]
        if not isinstance(brands_map, dict) or not brands_map:
            raise ValueError(f"Invalid brand registry: {reg_path}")
        for key, aliases in brands_map.items():
            if not isinstance(aliases, list) or not all(isinstance(alias, str) and alias.strip() for alias in aliases):
                raise ValueError(f"Invalid aliases for {key!r} in {reg_path}")
            merged_registry.setdefault(key, []).extend(aliases)
    alias_dict = collect_brand_aliases(
        synthetic_records or [],
        brand_registry=merged_registry,
    )
    meta_brands = set(extract_metadata_brands(synthetic_records or []))
    all_tokens = {alias for aliases in alias_dict.values() for alias in aliases} | meta_brands
    return sorted(all_tokens, key=lambda token: (-len(token), token))


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
        for view in TEXT_VIEW_COLUMNS:
            out[f"text_adv_{view}"] = []
    for t in texts:
        views = mask_text_views(t, brands=brands, brand_pattern=brand_pattern)
        out["text_raw"].append(views["raw"])
        out["text_canon_masked"].append(views["canon_masked"])
        out["text_brand_masked"].append(views["brand_masked"])
        out["text_brand_stripped"].append(views["brand_stripped"])
        if include_adv:
            adv_text = perturb_zwj_homoglyph(views["raw"], seed=42)
            adv_views = mask_text_views(adv_text, brands=brands, brand_pattern=brand_pattern)
            for view, value in adv_views.items():
                out[f"text_adv_{view}"].append(value)
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
            "text_norm_strict": [strong_normalize_text(text) for text in texts],
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


def select_counterfactual_fold(
    probe: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fold: int,
) -> pd.DataFrame:
    """Select the probe for the full pre-cap validation fold and check its parent rows."""
    if not probe["row_id"].is_unique or not probe["curation_cv_fold"].isin(CV_FOLDS).all():
        raise RuntimeError("Counterfactual probe has duplicate IDs or invalid fold assignments.")
    if not val_df["curation_cv_fold"].eq(fold).all():
        raise RuntimeError("Counterfactual validation frame has an unexpected fold assignment.")
    selected = probe.loc[probe["curation_cv_fold"].eq(fold)].reset_index(drop=True)
    for column in ("row_id", "family_id"):
        if not set(selected[column]).isdisjoint(train_df[column]):
            raise RuntimeError(f"Counterfactual {column} overlaps training in fold {fold}.")

    texts = val_df["text"]
    contact = texts.str.contains(_SCHEMA_URL_RE, regex=True) | texts.str.contains(_SCHEMA_PHONE_RE, regex=True)
    eligible = val_df.loc[val_df["label_3class"].eq("Smishing") & contact]
    columns = ["row_id", "family_id", "curation_cv_fold", "language", "label_idx"]
    if not selected[columns].sort_values("row_id").reset_index(drop=True).equals(
        eligible[columns].sort_values("row_id").reset_index(drop=True)
    ):
        raise RuntimeError(f"Counterfactual rows do not match eligible validation parents in fold {fold}.")
    return selected


def _require_equal_rows(actual: pd.DataFrame, expected: pd.DataFrame, table_name: str) -> None:
    columns = expected.columns.tolist()
    if not actual[columns].reset_index(drop=True).equals(expected.reset_index(drop=True)):
        raise RuntimeError(f"Bundle source rows, labels, or order differ for {table_name}.")


def _audit_bundle_tables(tables: dict[str, pd.DataFrame], *, require_normalization: bool) -> dict[str, Any]:
    """Independently measure cohort membership, duplicate keys, and validation ownership."""
    fingerprints = {}
    normalization_mismatches = {}
    for name, expected_counts in EXPECTED_COHORT_COUNTS.items():
        table = tables[name]
        actual_counts = _summarize_df_counts(table)["by_class"]
        if tuple(actual_counts.values()) != expected_counts or len(table) != sum(expected_counts):
            raise RuntimeError(f"Unexpected cohort size or class counts in {name}: {actual_counts}.")
        expected_labels = table["label_3class"].map(LABEL_TO_IDX)
        expected_folds = CV_FOLDS if name == "real_train_pool.parquet" else (-1,)
        expected_synthetic = name.startswith("synthetic_")
        if (
            not table["label_idx"].eq(expected_labels).all()
            or not table["curation_cv_fold"].isin(expected_folds).all()
            or not table["is_synthetic"].eq(expected_synthetic).all()
        ):
            raise RuntimeError(f"Unexpected label, fold, or Stage B cohort marker in {name}.")
        strict = table["text"].map(strong_normalize_text)
        normalization_mismatches[name] = int(strict.ne(table["text_norm_strict"]).sum())
        fingerprints[name] = {
            "row_id": set(table["row_id"]),
            "family_id": set(table["family_id"]),
            "text": set(table["text"]),
            "strong_normalized_text": set(strict),
        }
        for column in ("row_id", "family_id", "text"):
            if table[column].isna().any() or "" in fingerprints[name][column]:
                raise RuntimeError(f"Missing {column} in {name}.")
        for column in ("row_id", "text", "strong_normalized_text"):
            values = fingerprints[name][column]
            if "" in values or len(values) != len(table):
                raise RuntimeError(f"Empty or duplicate {column} in {name}.")
    if require_normalization and any(normalization_mismatches.values()):
        raise RuntimeError(f"Inconsistent strong normalization: {normalization_mismatches}.")

    pairs_checked = 0
    for left, right in combinations(fingerprints, 2):
        pairs_checked += 1
        for key, left_values in fingerprints[left].items():
            if left_values & fingerprints[right][key]:
                raise RuntimeError(f"Cross-cohort {key} overlap between {left} and {right}.")

    real = tables["real_train_pool.parquet"]
    cf = tables["counterfactual_smish_probe.parquet"]
    if real.groupby("family_id")["curation_cv_fold"].nunique().max() != 1:
        raise RuntimeError("A real family is assigned to multiple folds.")
    per_fold = {}
    scored_probe_ids = []
    for fold in CV_FOLDS:
        train = real.loc[real["curation_cv_fold"].ne(fold)]
        val = real.loc[real["curation_cv_fold"].eq(fold)]
        if val.empty:
            raise RuntimeError(f"Missing real validation fold {fold}.")
        probe = select_counterfactual_fold(cf, train, val, fold)
        scored_probe_ids.extend(probe["row_id"])
        per_fold[fold] = {
            "n_real_train": len(train),
            "n_real_val": len(val),
            "n_counterfactual": len(probe),
            "val_by_class": _summarize_df_counts(val)["by_class"],
        }
    if set(scored_probe_ids) != set(cf["row_id"]) or len(scored_probe_ids) != len(cf):
        raise RuntimeError("The counterfactual probe must be covered exactly once by the five folds.")

    full_train = pd.concat([tables[name] for name in list(EXPECTED_COHORT_COUNTS)[:3]], ignore_index=True)
    promotional_languages = full_train.loc[full_train["label_3class"].eq("Promotional"), "language"].value_counts()
    if promotional_languages.to_dict() != dict.fromkeys(("Bengali", "Banglish", "English", "CodeMix"), 387):
        raise RuntimeError(f"Unexpected full-pool Promotional language counts: {promotional_languages.to_dict()}.")
    return {
        "cohort_pairs_checked": pairs_checked,
        "normalization_metadata_mismatches": normalization_mismatches,
        "promotional_languages_full_pool": promotional_languages.to_dict(),
        "folds": per_fold,
    }


def _brand_input_hashes(project_root: Path) -> dict[str, str]:
    paths = [
        Path("data/generated/curation_v2") / batch / "brand_registry.json"
        for batch in SYNTHETIC_BATCH_DIRS
    ]
    return {path.as_posix(): sha256_file(project_root / path) for path in paths}


def _vocabulary_sha256(vocabulary: list[str]) -> str:
    payload = json.dumps(vocabulary, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assemble_post_stage_b_bundle(
    project_root: str | Path,
    output_dir: str | Path,
    *,
    strict_digests: bool = True,
) -> dict[str, Any]:
    """Audit protected corpora + Stage B cohorts and export the Colab-ready verified dataset bundle."""
    root = Path(project_root)
    out_dir = Path(output_dir)
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(f"Bundle output must be a new directory: {out_dir}")

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

    full_balanced = pd.concat([real_train, syn_pilot, syn_b1_b7], ignore_index=True)
    tables = {
        "real_train_pool.parquet": real_train,
        "synthetic_pilot_300.parquet": syn_pilot,
        "synthetic_batches_b1_b7.parquet": syn_b1_b7,
        "eval_smishdetect_validation.parquet": sd_val,
        "eval_smishdetect_test.parquet": sd_test,
        "eval_banglaphish_ood.parquet": bp_ood,
        "counterfactual_smish_probe.parquet": cf_probe,
    }
    content_audit = _audit_bundle_tables(tables, require_normalization=True)
    integrity_checks = dict.fromkeys(sorted(V1_INTEGRITY_CHECKS), True)
    integrity_checks["zero_all_pairwise_row_family_and_text_overlap"] = True
    integrity_checks["exact_counterfactual_validation_parent_coverage"] = True

    out_dir.mkdir(parents=True)
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
        "manifest_version": "post_stage_b_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head_sha(root),
        "synthetic_generation_status": "PERMANENTLY_STOPPED",
        "brand_vocabulary_count": len(brands),
        "brand_vocabulary_sha256": _vocabulary_sha256(brands),
        "brand_registry_sha256": _brand_input_hashes(root),
        "protected_inputs": protected_inventory,
        "bundle_files": bundle_files_meta,
        "integrity_checks": integrity_checks,
        "content_audit": content_audit,
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
    version = manifest.get("manifest_version")
    if version not in ("post_stage_b_v1", "post_stage_b_v2"):
        raise ValueError(f"Unsupported manifest_version: {manifest.get('manifest_version')}")
    checks = manifest.get("integrity_checks", {})
    required_checks = V1_INTEGRITY_CHECKS
    if version == "post_stage_b_v2":
        required_checks = required_checks | {
            "zero_all_pairwise_row_family_and_text_overlap",
            "exact_counterfactual_validation_parent_coverage",
        }
    if not isinstance(checks, dict) or not required_checks <= checks.keys() or not all(v is True for v in checks.values()):
        raise RuntimeError(f"Manifest failed integrity checks: {checks}")

    bundle_files = manifest.get("bundle_files", {})
    if set(bundle_files) != set(BUNDLE_FILENAMES):
        raise RuntimeError("Manifest bundle files do not match the required inventory.")
    for fname in BUNDLE_FILENAMES:
        fpath = bdir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing bundle file: {fpath}")
        metadata = bundle_files[fname]
        if verify_checksums:
            actual_sha = sha256_file(fpath)
            expected_sha = metadata["sha256"]
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"Checksum mismatch for {fname}: expected {expected_sha}, got {actual_sha}"
                )
        with ParquetFile(fpath) as parquet:
            if (
                parquet.metadata.num_rows != metadata["n_rows"]
                or parquet.schema_arrow.names != metadata["columns"]
                or fpath.stat().st_size != metadata["bytes"]
            ):
                raise RuntimeError(f"Manifest physical metadata mismatch for {fname}.")
    return manifest


def audit_post_stage_b_bundle(
    project_root: str | Path, bundle_dir: str | Path, manifest: dict[str, Any],
) -> dict[str, Any]:
    """Compare decoded bundle rows against protected source tables and recompute checks."""
    root = Path(project_root)
    bdir = Path(bundle_dir)
    if manifest["protected_inputs"] != verify_protected_inputs(root):
        raise RuntimeError("Manifest protected source fingerprints differ from local files.")
    tables = {name: pd.read_parquet(bdir / name) for name in BUNDLE_FILENAMES}
    audit = _audit_bundle_tables(tables, require_normalization=manifest["manifest_version"] == "post_stage_b_v2")
    cohort_counts = {Path(name).stem: _summarize_df_counts(tables[name]) for name in EXPECTED_COHORT_COUNTS}
    full_train = pd.concat([tables[name] for name in list(EXPECTED_COHORT_COUNTS)[:3]], ignore_index=True)
    cohort_counts["full_stage_b_balanced_pool"] = _summarize_df_counts(full_train)
    cohort_counts["counterfactual_smish_probe"] = {"n_rows": len(tables["counterfactual_smish_probe.parquet"])}
    if manifest["cohort_counts"] != cohort_counts:
        raise RuntimeError("Manifest cohort counts differ from decoded tables.")

    real_cols = ["canonical_id", "text", "label_3class", "language", "family_id"]
    eligible = pd.read_parquet(root / "data/processed/curation_v2/eligible_train.parquet")
    existing = pd.read_parquet(root / "data/processed/curation_v2/existing_unique.parquet")
    for name, cohort, source in (
        ("real_train_pool.parquet", "real_train", eligible),
        ("eval_smishdetect_validation.parquet", "smishdetect_validation", existing),
        ("eval_smishdetect_test.parquet", "smishdetect_test", existing),
        ("eval_banglaphish_ood.parquet", "banglaphish_ood", existing),
    ):
        if cohort == "real_train":
            expected = source[real_cols + ["curation_cv_fold"]].copy()
        else:
            expected = source.loc[source["eval_group"].eq(cohort), real_cols].assign(curation_cv_fold=-1)
        expected = expected.rename(columns={"canonical_id": "row_id"}).assign(cohort=cohort)
        _require_equal_rows(tables[name], expected, name)

    provenance = pd.read_parquet(root / "data/processed/curation_v2/provenance.parquet")
    joined = provenance.merge(
        existing[["canonical_id", "family_id", "eval_group"]],
        on="canonical_id", how="inner", validate="many_to_one", suffixes=("_source", "_curated"),
    )
    if not joined["family_id_source"].eq(joined["family_id_curated"]).all():
        raise RuntimeError("Curated source provenance has an inconsistent family ID.")
    smishdetect = joined.loc[joined["dataset"].eq("smishdetect_2026")]
    train_links = smishdetect.loc[smishdetect["eval_group"].eq("train_pool")]
    if set(train_links["canonical_id"]) != set(eligible["canonical_id"]) or set(train_links["orig_split"]) != {"train"}:
        raise RuntimeError("Curated training rows have missing or held-out SmishDetect source links.")
    audit["smishdetect_train_source_links"] = len(train_links)

    ood = pd.read_parquet(root / "data/processed/banglaphish2026_dedup.parquet")
    ood_columns = ["text", "label_3class"]
    _require_equal_rows(
        tables["eval_banglaphish_ood.parquet"].sort_values("text"),
        ood[ood_columns].sort_values("text"),
        "BanglaPhish reference",
    )

    generated = root / "data/generated/curation_v2"
    accepted_groups = {
        "synthetic_pilot_300.parquet": [_load_jsonl(generated / "accepted.jsonl")],
        "synthetic_batches_b1_b7.parquet": [_load_jsonl(generated / batch / "accepted.jsonl") for batch in SYNTHETIC_BATCH_DIRS],
    }
    duplicate_reference = set(existing["text"].map(strong_normalize_text))
    quarantine = pd.read_parquet(root / "data/processed/curation_v2/quarantined.parquet", columns=["text"])
    duplicate_reference.update(quarantine["text"].map(strong_normalize_text))
    for file_name, record_groups in accepted_groups.items():
        counts = [len(group) for group in record_groups]
        expected_counts = [300] if file_name == "synthetic_pilot_300.parquet" else [50] * 6 + [45]
        if counts != expected_counts:
            raise RuntimeError(f"Unexpected accepted source membership sizes for {file_name}: {counts}")
        accepted = pd.DataFrame([record for group in record_groups for record in group])
        if not accepted["review_status"].eq("accepted").all() or accepted["source_row_ids"].map(bool).any():
            raise RuntimeError(f"Unaccepted rows or source-linked augmentation found in {file_name}.")
        source_cols = ["id", "text", "label_3class", "language", "family_id", "generation_batch"]
        expected = accepted[source_cols].rename(columns={"id": "row_id", "generation_batch": "cohort"})
        _require_equal_rows(tables[file_name], expected, file_name)
        if duplicate_reference.intersection(accepted["text"].map(strong_normalize_text)):
            raise RuntimeError(f"Strong normalized text overlap with the curation inventory in {file_name}.")
    all_accepted = [row for groups in accepted_groups.values() for group in groups for row in group]
    brands = collect_post_stage_b_brand_vocabulary(root, synthetic_records=all_accepted)
    if len(brands) != manifest["brand_vocabulary_count"]:
        raise RuntimeError("Manifest brand vocabulary count differs from local source metadata.")
    if manifest["manifest_version"] == "post_stage_b_v2":
        if (
            manifest["brand_vocabulary_sha256"] != _vocabulary_sha256(brands)
            or manifest["brand_registry_sha256"] != _brand_input_hashes(root)
        ):
            raise RuntimeError("Manifest brand inputs differ from frozen source metadata.")
    cf = tables["counterfactual_smish_probe.parquet"]
    real = tables["real_train_pool.parquet"].set_index("row_id")
    if not cf["orig_text_raw"].eq(cf["row_id"].map(real["text"])).all():
        raise RuntimeError("Counterfactual original text differs from its frozen parent.")
    return audit


def _downsample_balanced_df(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Deterministically downsample each class to the minority class count."""
    counts = df["label_3class"].value_counts()
    min_n = int(counts.min())
    parts = [
        df[df["label_3class"] == lbl].sample(n=min_n, random_state=seed + idx)
        for idx, lbl in enumerate(CANONICAL_LABELS)
    ]
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def training_class_weights(label_indices: pd.Series) -> list[float]:
    """Return inverse frequency weights N / (K × n_c) for a training split."""
    counts = label_indices.value_counts().reindex(range(len(CANONICAL_LABELS)), fill_value=0)
    if not label_indices.isin(counts.index).all() or counts.eq(0).any():
        raise ValueError("Class weighting requires valid labels and at least one training row per class.")
    return (len(label_indices) / (len(CANONICAL_LABELS) * counts)).tolist()


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
    if fold not in CV_FOLDS:
        raise ValueError(f"Invalid fold={fold}; expected 0..4")

    bdir = Path(bundle_dir)
    spec = EXPERIMENT_SPECS[experiment_id]
    balance_mode = spec.default_balance_mode if balance_mode_override is None else balance_mode_override
    if balance_mode not in BALANCE_MODES:
        raise ValueError(f"Unsupported balance mode: {balance_mode!r}; expected one of {BALANCE_MODES}")

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
    if not train_df["row_id"].is_unique or not va_real["row_id"].is_unique:
        raise RuntimeError("Duplicate row IDs within a training or validation fold.")
    if not set(train_df["row_id"]).isdisjoint(va_real["row_id"]):
        raise RuntimeError("Row ID leakage detected between train and validation folds.")
    if not set(train_df["family_id"]).isdisjoint(set(va_real["family_id"])):
        raise RuntimeError("Family ID leakage detected between train and validation folds.")

    class_weights = None
    if balance_mode == "class_weight":
        class_weights = training_class_weights(train_df["label_idx"])

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
