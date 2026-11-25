"""Deterministic, read-only Stage 4 test evaluation cohort masks without mutating frozen datasets."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .curation import slot_template_text, strong_normalize_text
from .post_stage_b_data import sha256_file
from .schema import CANONICAL_LABELS

EXPECTED_SD_TEST_ROWS = 1698
EXPECTED_SD_TEST_FAMILIES = 1304

# 5 Smishing test rows identified in Stage 3 Task A with >=0.85 ambiguous similarity to train/val.
AMBIGUOUS_TEST_ROW_IDS: tuple[str, ...] = (
    "cur_7936ee1f5382",
    "cur_7c14e8bae65c",
    "cur_c00c6e3fa781",
    "cur_c24e8fee19a2",
    "cur_cfa2f9f81a31",
)

# 5 families owning the 5 ambiguous test rows (expanding to 11 test Smishing rows).
AMBIGUOUS_TEST_FAMILY_IDS: tuple[str, ...] = (
    "fam_03f840170c",
    "fam_210d66c8f3",
    "fam_735d8c7d90",
    "fam_77db01305f",
    "fam_b7af3c4ecc",
)

# All 12 test families hit by any of the 32 Stage 3 cross-boundary TF-IDF >= 0.85 pairs (23 rows).
TEMPLATE_STRESS_TEST_FAMILY_IDS: tuple[str, ...] = (
    "fam_03f840170c",
    "fam_194ea98c59",
    "fam_210d66c8f3",
    "fam_3ae7bc14a0",
    "fam_4bb67afb14",
    "fam_5408bae462",
    "fam_735d8c7d90",
    "fam_77db01305f",
    "fam_b7af3c4ecc",
    "fam_c025d51b1d",
    "fam_d992d2b1c6",
    "fam_e396320454",
)

NEAR_DUPLICATE_AND_LEAKAGE_CAVEAT = (
    "Evaluation masking removes near-duplicate test rows/families at test scoring time cleanly "
    "and identically across all compared models without modifying frozen files. However, post-hoc "
    "test masking CANNOT undo any indirect influence of train<->CV or train<->val near-duplicates "
    "on epoch checkpoint selection during training, and 21 semantically related template pairs remain "
    "in sensitivity comparisons outside template_stress."
)

COHORT_SPECS: dict[str, dict[str, Any]] = {
    "primary_family_decontam_1687": {
        "role": "primary_confirmatory",
        "rule": "Exclude entire test families owning any ambiguous (>=0.85) cross-split pair.",
        "expected_n_rows": 1687,
        "expected_n_families": 1299,
        "expected_by_class": {"Normal": 581, "Promotional": 384, "Smishing": 722},
    },
    "supplementary_full_1698": {
        "role": "supplementary_benchmark",
        "rule": "Retain full untouched eval_smishdetect_test benchmark (zero exclusions).",
        "expected_n_rows": 1698,
        "expected_n_families": 1304,
        "expected_by_class": {"Normal": 581, "Promotional": 384, "Smishing": 733},
    },
    "sensitivity_row_only_1693": {
        "role": "sensitivity_row_only",
        "rule": "Exclude only the 5 ambiguous test rows individually (preserving sibling rows).",
        "expected_n_rows": 1693,
        "expected_n_families": 1302,
        "expected_by_class": {"Normal": 581, "Promotional": 384, "Smishing": 728},
    },
    "sensitivity_template_stress_1675": {
        "role": "sensitivity_template_stress",
        "rule": "Exclude all 12 test families hit by any of the 32 cross-split TF-IDF >= 0.85 pairs.",
        "expected_n_rows": 1675,
        "expected_n_families": 1292,
        "expected_by_class": {"Normal": 580, "Promotional": 384, "Smishing": 711},
    },
}


def _reproduce_cross_boundary_test_hits(bdir: Path) -> dict[str, Any]:
    """Reproduce the 12,466-row char-ngram (3-5) TF-IDF >= 0.85 screen across cohort boundaries."""
    parts = [
        ("real_train", pd.read_parquet(bdir / "real_train_pool.parquet")),
        ("pilot", pd.read_parquet(bdir / "synthetic_pilot_300.parquet")),
        ("b1_b7", pd.read_parquet(bdir / "synthetic_batches_b1_b7.parquet")),
        ("val", pd.read_parquet(bdir / "eval_smishdetect_validation.parquet")),
        ("test", pd.read_parquet(bdir / "eval_smishdetect_test.parquet")),
        ("ood", pd.read_parquet(bdir / "eval_banglaphish_ood.parquet")),
    ]
    all_rows = pd.concat(
        [df[["row_id", "family_id", "curation_cv_fold"]].assign(_split=name, _text=df["text"]) for name, df in parts],
        ignore_index=True,
    )
    splits = all_rows["_split"].to_numpy()
    folds = all_rows["curation_cv_fold"].to_numpy()
    fams = all_rows["family_id"].to_numpy()
    rids = all_rows["row_id"].to_numpy()

    def _hit_pairs(texts: list[str]) -> set[tuple[int, int]]:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        mat = vec.fit_transform(texts)
        sim = linear_kernel(mat, mat)
        ii, jj = np.where(np.triu(sim, k=1) >= 0.85)
        hits: set[tuple[int, int]] = set()
        for a, b in zip(ii.tolist(), jj.tolist()):
            if fams[a] == fams[b]:
                continue
            cross_split = splits[a] != splits[b]
            cross_fold = splits[a] == "real_train" and splits[b] == "real_train" and folds[a] != folds[b]
            if cross_split or cross_fold:
                hits.add((min(a, b), max(a, b)))
        return hits

    norm_hits = _hit_pairs([strong_normalize_text(t) for t in all_rows["_text"]])
    slot_hits = _hit_pairs([slot_template_text(t) for t in all_rows["_text"]])
    union_hits = norm_hits | slot_hits
    test_hit_rows = sorted(
        {str(rids[idx]) for a, b in union_hits for idx in (a, b) if splits[idx] == "test"}
    )
    test_hit_fams = sorted(
        {str(fams[idx]) for a, b in union_hits for idx in (a, b) if splits[idx] == "test"}
    )
    return {
        "n_corpus_rows": len(all_rows),
        "n_unique_cross_boundary_pairs": len(union_hits),
        "test_hit_row_ids": test_hit_rows,
        "test_hit_family_ids": test_hit_fams,
    }


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_smishdetect_test_cohort_masks(
    bundle_dir: str | Path,
    *,
    verify_tfidf_reproduction: bool = False,
) -> dict[str, Any]:
    """Build and verify the 4 deterministic SmishDetect test evaluation cohort masks."""
    bdir = Path(bundle_dir)
    test_path = bdir / "eval_smishdetect_test.parquet"
    test_df = pd.read_parquet(test_path).reset_index(drop=True)
    if len(test_df) != EXPECTED_SD_TEST_ROWS or test_df["family_id"].nunique() != EXPECTED_SD_TEST_FAMILIES:
        raise RuntimeError(
            f"Unexpected eval_smishdetect_test shape: rows={len(test_df)}, fams={test_df['family_id'].nunique()}"
        )

    tfidf_info = None
    if verify_tfidf_reproduction:
        tfidf_info = _reproduce_cross_boundary_test_hits(bdir)
        if not set(AMBIGUOUS_TEST_ROW_IDS).issubset(tfidf_info["test_hit_row_ids"]):
            raise RuntimeError("Reproduced TF-IDF >= 0.85 screen missing expected AMBIGUOUS_TEST_ROW_IDS.")
        if set(tfidf_info["test_hit_family_ids"]) != set(TEMPLATE_STRESS_TEST_FAMILY_IDS):
            raise RuntimeError(
                f"Reproduced template-stress test families {tfidf_info['test_hit_family_ids']} "
                f"differ from expected {list(TEMPLATE_STRESS_TEST_FAMILY_IDS)}."
            )

    ambig_rows_set = set(AMBIGUOUS_TEST_ROW_IDS)
    ambig_fams_set = set(AMBIGUOUS_TEST_FAMILY_IDS)
    stress_fams_set = set(TEMPLATE_STRESS_TEST_FAMILY_IDS)

    if set(test_df.loc[test_df["row_id"].isin(ambig_rows_set), "family_id"]) != ambig_fams_set:
        raise RuntimeError("AMBIGUOUS_TEST_ROW_IDS do not map cleanly to AMBIGUOUS_TEST_FAMILY_IDS.")

    boolean_masks = {
        "primary_family_decontam_1687": ~test_df["family_id"].isin(ambig_fams_set).to_numpy(dtype=bool),
        "supplementary_full_1698": np.ones(len(test_df), dtype=bool),
        "sensitivity_row_only_1693": ~test_df["row_id"].isin(ambig_rows_set).to_numpy(dtype=bool),
        "sensitivity_template_stress_1675": ~test_df["family_id"].isin(stress_fams_set).to_numpy(dtype=bool),
    }

    cohorts: dict[str, Any] = {}
    for name, mask in boolean_masks.items():
        spec = COHORT_SPECS[name]
        kept = test_df.loc[mask]
        excl = test_df.loc[~mask]
        by_class = {lbl: int((kept["label_3class"] == lbl).sum()) for lbl in CANONICAL_LABELS}
        excl_by_class = {lbl: int((excl["label_3class"] == lbl).sum()) for lbl in CANONICAL_LABELS}
        n_rows = int(len(kept))
        n_fams = int(kept["family_id"].nunique())
        if (
            n_rows != spec["expected_n_rows"]
            or n_fams != spec["expected_n_families"]
            or by_class != spec["expected_by_class"]
        ):
            raise RuntimeError(f"Cohort {name} mismatch: rows={n_rows}, fams={n_fams}, classes={by_class}")

        included_ids = kept["row_id"].astype(str).tolist()
        excluded_ids = sorted(excl["row_id"].astype(str).tolist())
        excluded_fams = sorted(excl["family_id"].astype(str).unique().tolist())
        mask_bits = "".join("1" if b else "0" for b in mask.tolist())
        cohorts[name] = {
            "role": spec["role"],
            "rule": spec["rule"],
            "n_rows": n_rows,
            "n_families": n_fams,
            "by_class": by_class,
            "n_excluded_rows": int(len(excl)),
            "excluded_by_class": excl_by_class,
            "excluded_row_ids": excluded_ids,
            "excluded_family_ids": excluded_fams,
            "included_row_ids_sha256": _sha256_json(included_ids),
            "mask_sha256": hashlib.sha256(mask_bits.encode("ascii")).hexdigest(),
            "mask": mask,
        }

    val_df = pd.read_parquet(bdir / "eval_smishdetect_validation.parquet")
    val_normal = int((val_df["label_3class"] == "Normal").sum())
    return {
        "schema_version": "stage4_test_cohort_masks_v1",
        "bundle_test_parquet_sha256": sha256_file(test_path),
        "test_row_ids_sha256": _sha256_json(test_df["row_id"].astype(str).tolist()),
        "threshold_calibration_independent_of_test_exclusions": True,
        "validation_normal_count": val_normal,
        "ambiguous_test_all_smishing": bool((test_df.loc[~boolean_masks["primary_family_decontam_1687"], "label_3class"] == "Smishing").all()),
        "near_duplicate_and_leakage_caveat": NEAR_DUPLICATE_AND_LEAKAGE_CAVEAT,
        "tfidf_reproduction": tfidf_info,
        "cohorts": cohorts,
    }


def export_frozen_test_masks(
    bundle_dir: str | Path,
    out_path: str | Path,
    *,
    verify_tfidf_reproduction: bool = True,
) -> dict[str, Any]:
    """Write reproducible frozen cohort mask definitions (no raw boolean numpy arrays in JSON)."""
    result = build_smishdetect_test_cohort_masks(bundle_dir, verify_tfidf_reproduction=verify_tfidf_reproduction)
    serializable = {
        k: v for k, v in result.items() if k != "cohorts"
    }
    serializable["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    serializable["cohorts"] = {
        name: {k: v for k, v in entry.items() if k != "mask"}
        for name, entry in result["cohorts"].items()
    }
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    return serializable
