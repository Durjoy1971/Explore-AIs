"""Five-fold probability-averaged ensemble validation, threshold calibration, and split export."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score

from .metrics import _confusion_intervals, expected_calibration_error, smishing_recall_at_fpr
from .post_stage_b_stats import (
    CONFIRMATORY_DECLARATIONS_ENABLED,
    DEFAULT_B_CI,
    DEFAULT_B_PVAL,
    OPTION_A_PRIMARY_ARMS,
    compare_models_cluster_bootstrap,
    evaluate_stage4_hypothesis_families,
    STATISTICAL_VALIDATION_STATUS,
)
from .schema import CANONICAL_LABELS

EXPECTED_SMISHDETECT_VAL_NORMAL = 265
THRESHOLD_TRANSFER_CAVEAT = (
    "Satisfying the <=1% Normal FPR constraint on smishdetect_validation (265 Normal rows, "
    "max 2 FPs) fixes the float64 decision threshold BEFORE evaluating test/OOD and NEVER "
    "guarantees <=1% FPR on smishdetect_test or banglaphish_ood under sampling variability or domain shift."
)


def validate_fold_predictions_for_ensemble(
    fold_entries: dict[int, dict[str, Any]],
    expected_frames: dict[str, pd.DataFrame],
    *,
    required_folds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict[str, Any]:
    """Strictly verify all 5 folds share compatible revisions/hashes and exact row/label/family alignment."""
    if set(fold_entries) != set(required_folds):
        missing = sorted(set(required_folds) - set(fold_entries))
        extra = sorted(set(fold_entries) - set(required_folds))
        raise ValueError(f"Incomplete or unexpected folds for ensemble: missing={missing}, extra={extra}")

    anchor = fold_entries[required_folds[0]]
    anchor_meta = {
        key: anchor.get(key)
        for key in (
            "experiment_id",
            "model_name",
            "text_view",
            "model_revision",
            "tokenizer_revision",
            "config_hash",
            "dataset_manifest_sha256",
        )
    }
    for fold in required_folds:
        entry = fold_entries[fold]
        for key, expected_val in anchor_meta.items():
            if entry.get(key) != expected_val:
                raise ValueError(
                    f"Fold {fold} mismatch on {key}: {entry.get(key)!r} != {expected_val!r}"
                )

    averaged_probs: dict[str, np.ndarray] = {}
    for split_name, frame in expected_frames.items():
        ref_ids = frame["row_id"].to_numpy(dtype=str)
        ref_y = frame["label_idx"].to_numpy(dtype=int)
        ref_fams = frame["family_id"].to_numpy(dtype=str)
        fold_probs: list[np.ndarray] = []
        for fold in required_folds:
            arrays = fold_entries[fold]["arrays"]
            for suffix in ("row_ids", "y", "probs"):
                key = f"{split_name}_{suffix}"
                if key not in arrays:
                    raise ValueError(f"Fold {fold} missing {key} required for ensemble aggregation")
            if not np.array_equal(arrays[f"{split_name}_row_ids"], ref_ids):
                raise ValueError(f"Fold {fold} row_ids misaligned on {split_name}")
            if not np.array_equal(arrays[f"{split_name}_y"], ref_y):
                raise ValueError(f"Fold {fold} labels misaligned on {split_name}")
            if f"{split_name}_family_ids" in arrays and not np.array_equal(arrays[f"{split_name}_family_ids"], ref_fams):
                raise ValueError(f"Fold {fold} family_ids misaligned on {split_name}")

            probs = np.asarray(arrays[f"{split_name}_probs"], dtype=np.float64)
            if (
                probs.shape != (len(frame), len(CANONICAL_LABELS))
                or not np.isfinite(probs).all()
                or (probs < 0.0).any()
                or (probs > 1.0).any()
                or not np.allclose(probs.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6)
            ):
                raise ValueError(f"Invalid probability matrix in fold {fold} for {split_name}")
            fold_probs.append(probs)

        mean_probs = np.mean(np.stack(fold_probs, axis=0), axis=0, dtype=np.float64)
        mean_probs /= mean_probs.sum(axis=1, keepdims=True)
        averaged_probs[split_name] = mean_probs

    return {
        "metadata": anchor_meta,
        "averaged_probs": averaged_probs,
    }


def calibrate_ensemble_threshold(
    val_df: pd.DataFrame,
    val_probs: np.ndarray,
    *,
    max_fpr: float = 0.01,
    expected_normal_count: int | None = EXPECTED_SMISHDETECT_VAL_NORMAL,
) -> dict[str, Any]:
    """Calibrate Smishing threshold strictly on SmishDetect validation probabilities."""
    y_val = val_df["label_idx"].to_numpy(dtype=int)
    normal_idx = CANONICAL_LABELS.index("Normal")
    smish_idx = CANONICAL_LABELS.index("Smishing")
    if expected_normal_count is not None and int((y_val == normal_idx).sum()) != expected_normal_count:
        raise ValueError(
            f"Expected {expected_normal_count} Normal validation rows, got {int((y_val == normal_idx).sum())}"
        )
    cal = smishing_recall_at_fpr(y_val, val_probs, smish_idx, normal_idx, max_fpr=max_fpr)
    if cal["threshold"] is None:
        raise ValueError("Cannot calibrate threshold without Normal validation samples")
    return cal


def _safe_smishing_aucs(y_true: np.ndarray, smish_scores: np.ndarray, smish_idx: int) -> tuple[float | None, float | None]:
    y_bin = (y_true == smish_idx).astype(int)
    if len(np.unique(y_bin)) < 2:
        return None, None
    return (
        round(float(roc_auc_score(y_bin, smish_scores)), 6),
        round(float(average_precision_score(y_bin, smish_scores)), 6),
    )


def evaluate_ensemble_split(
    df: pd.DataFrame,
    probs: np.ndarray,
    *,
    frozen_threshold: float,
    split_name: str,
    calibration_achieved_fpr: float | None,
    n_boot: int = DEFAULT_B_CI,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate aligned ensemble probabilities under a frozen float64 threshold without re-tuning."""
    y_true = df["label_idx"].to_numpy(dtype=int)
    fams = df["family_id"].to_numpy(dtype=str)
    y_pred = probs.argmax(axis=1)
    normal_idx = CANONICAL_LABELS.index("Normal")
    promo_idx = CANONICAL_LABELS.index("Promotional")
    smish_idx = CANONICAL_LABELS.index("Smishing")

    points, intervals = _confusion_intervals(y_true, y_pred, len(CANONICAL_LABELS), smish_idx, n_boot, fams, seed=seed)
    low_fpr = smishing_recall_at_fpr(
        y_true, probs, smish_idx, normal_idx, max_fpr=0.01, operating_threshold=float(frozen_threshold)
    )
    scores = np.asarray(probs[:, smish_idx], dtype=np.float64)
    hit_frozen = scores >= float(frozen_threshold)

    is_normal = y_true == normal_idx
    is_promo = y_true == promo_idx
    n_normal = int(is_normal.sum())
    n_promo = int(is_promo.sum())
    normal_fp = int((hit_frozen & is_normal).sum())
    promo_fp = int((hit_frozen & is_promo).sum())

    roc_auc, pr_auc = _safe_smishing_aucs(y_true, scores, smish_idx)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=list(CANONICAL_LABELS),
        output_dict=True,
        zero_division=0,
    )

    return {
        "split_name": split_name,
        "n_samples": int(len(df)),
        "n_families": int(pd.Series(fams).nunique()),
        "threshold": float(frozen_threshold),
        "threshold_source": "frozen_smishdetect_validation",
        "calibration_achieved_fpr_pct": calibration_achieved_fpr,
        "achieved_fpr_pct": low_fpr["actual_fpr"],
        "normal_false_positives": normal_fp,
        "normal_denominator": n_normal,
        "promotional_false_positives_at_threshold": promo_fp,
        "promotional_denominator": n_promo,
        "promotional_fpr_at_threshold_pct": (100.0 * promo_fp / n_promo) if n_promo else None,
        "smishing_recall_at_frozen_threshold_pct": low_fpr["recall_at_fpr"],
        "smishing_recall_argmax_pct": points["smishing_recall_argmax"],
        "smishing_recall_argmax_ci95_pct": intervals["smishing_recall_argmax"],
        "macro_f1_pct": points["macro_f1"],
        "macro_f1_ci95_pct": intervals["macro_f1"],
        "accuracy_pct": points["accuracy"],
        "accuracy_ci95_pct": intervals["accuracy"],
        "ece": expected_calibration_error(y_true, probs),
        "roc_auc_smishing_vs_rest": roc_auc,
        "pr_auc_smishing_vs_rest": pr_auc,
        "confusion_matrix_argmax": cm,
        "constraint_met": low_fpr["constraint_met"],
        "calibration_caveat": THRESHOLD_TRANSFER_CAVEAT,
        "per_class": {
            lbl: {
                "precision": round(float(report[lbl]["precision"] * 100.0), 4),
                "recall": round(float(report[lbl]["recall"] * 100.0), 4),
                "f1": round(float(report[lbl]["f1-score"] * 100.0), 4),
                "support": int(report[lbl]["support"]),
            }
            for lbl in CANONICAL_LABELS
        },
        "_y_true": y_true,
        "_y_pred": y_pred,
        "_hit_frozen": hit_frozen,
        "_family_ids": fams,
    }


def _strip_private_arrays(entry: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in entry.items() if not k.startswith("_")}


def build_ensemble_evaluation_summary(
    *,
    ensemble_groups: dict[tuple[str, str, str], dict[int, dict[str, Any]]],
    eval_frames: dict[str, pd.DataFrame],
    test_masks: dict[str, np.ndarray] | None = None,
    n_boot_ci: int = DEFAULT_B_CI,
    n_boot_pval: int = DEFAULT_B_PVAL,
    seed: int = 42,
    confirmatory_declarations_enabled: bool = CONFIRMATORY_DECLARATIONS_ENABLED,
) -> dict[str, Any]:
    """Build 5-fold ensembles, calibrate thresholds on validation, score splits/masks, and run Holm tests."""
    ensemble_reports: dict[str, Any] = {}
    scored_primary: dict[tuple[str, str, str], dict[str, Any]] = {}
    scored_full: dict[tuple[str, str, str], dict[str, Any]] = {}
    smish_idx = CANONICAL_LABELS.index("Smishing")

    required_splits = ("smishdetect_validation", "smishdetect_test")
    extra_ood = "banglaphish_ood" in eval_frames
    selected_frames = {name: eval_frames[name] for name in (*required_splits, *(["banglaphish_ood"] if extra_ood else []))}

    for key_tuple, fold_map in sorted(ensemble_groups.items()):
        if set(fold_map) != set(range(5)):
            continue
        exp_id, model_name, text_view = key_tuple
        validated = validate_fold_predictions_for_ensemble(fold_map, selected_frames)
        avg_probs = validated["averaged_probs"]
        val_cal = calibrate_ensemble_threshold(eval_frames["smishdetect_validation"], avg_probs["smishdetect_validation"])
        frozen_thr = float(val_cal["threshold"])
        cal_fpr = val_cal["actual_fpr"]

        test_df = eval_frames["smishdetect_test"]
        test_probs = avg_probs["smishdetect_test"]
        test_cohorts: dict[str, Any] = {}
        active_masks = test_masks or {"supplementary_full_1698": np.ones(len(test_df), dtype=bool)}
        for mask_name, mask_arr in active_masks.items():
            sub_df = test_df.loc[mask_arr].reset_index(drop=True)
            sub_probs = test_probs[mask_arr]
            res = evaluate_ensemble_split(
                sub_df,
                sub_probs,
                frozen_threshold=frozen_thr,
                split_name=f"smishdetect_test::{mask_name}",
                calibration_achieved_fpr=cal_fpr,
                n_boot=n_boot_ci,
                seed=seed,
            )
            test_cohorts[mask_name] = _strip_private_arrays(res)
            if mask_name == "primary_family_decontam_1687":
                scored_primary[key_tuple] = res
            elif mask_name == "supplementary_full_1698":
                scored_full[key_tuple] = res

        val_res = evaluate_ensemble_split(
            eval_frames["smishdetect_validation"],
            avg_probs["smishdetect_validation"],
            frozen_threshold=frozen_thr,
            split_name="smishdetect_validation",
            calibration_achieved_fpr=cal_fpr,
            n_boot=n_boot_ci,
            seed=seed,
        )

        ood_reports: dict[str, Any] = {}
        if extra_ood:
            ood_df = eval_frames["banglaphish_ood"]
            ood_probs = avg_probs["banglaphish_ood"]
            ood_overall = evaluate_ensemble_split(
                ood_df,
                ood_probs,
                frozen_threshold=frozen_thr,
                split_name="banglaphish_ood",
                calibration_achieved_fpr=cal_fpr,
                n_boot=n_boot_ci,
                seed=seed,
            )
            ood_reports["overall"] = _strip_private_arrays(ood_overall)
            if "source_type" in ood_df.columns:
                sms_mask = (ood_df["source_type"] == "SMS").to_numpy()
                other_mask = ~sms_mask
                if int(sms_mask.sum()) >= 5:
                    ood_reports["sms"] = _strip_private_arrays(
                        evaluate_ensemble_split(
                            ood_df.loc[sms_mask].reset_index(drop=True),
                            ood_probs[sms_mask],
                            frozen_threshold=frozen_thr,
                            split_name="banglaphish_ood_sms",
                            calibration_achieved_fpr=cal_fpr,
                            n_boot=n_boot_ci,
                            seed=seed,
                        )
                    )
                if int(other_mask.sum()) >= 5:
                    ood_reports["other"] = _strip_private_arrays(
                        evaluate_ensemble_split(
                            ood_df.loc[other_mask].reset_index(drop=True),
                            ood_probs[other_mask],
                            frozen_threshold=frozen_thr,
                            split_name="banglaphish_ood_other",
                            calibration_achieved_fpr=cal_fpr,
                            n_boot=n_boot_ci,
                            seed=seed,
                        )
                    )

        group_id = f"exp_{exp_id}__{model_name}__{text_view}"
        ensemble_reports[group_id] = {
            "metadata": validated["metadata"],
            "calibration": {
                "split": "smishdetect_validation",
                "n_normal": val_cal["n_normal"],
                "max_allowed_fp": int(0.01 * val_cal["n_normal"]),
                "false_positives": val_cal["false_positives"],
                "achieved_fpr_pct": cal_fpr,
                "frozen_threshold": frozen_thr,
                "constraint_met": val_cal["constraint_met"],
                "smishing_recall_at_threshold_pct": val_cal["recall_at_fpr"],
                "caveat": THRESHOLD_TRANSFER_CAVEAT,
            },
            "smishdetect_validation": _strip_private_arrays(val_res),
            "smishdetect_test_cohorts": test_cohorts,
            "banglaphish_ood": ood_reports,
        }

    # Run statistical contrasts on the primary 1,687 cohort (fallback to full 1,698 if test_masks not supplied)
    target_scored = scored_primary if scored_primary else scored_full
    contrasts: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    contrast_idx = 0
    for (exp_id, model_name, text_view), cur in sorted(target_scored.items()):
        base_arms: list[str] = []
        if exp_id in OPTION_A_PRIMARY_ARMS:
            base_arms.append("A")
        if exp_id == "D":
            base_arms.append("C")
        for base_arm in base_arms:
            base_key = (base_arm, model_name, text_view)
            if base_key not in target_scored:
                continue
            base = target_scored[base_key]
            if not np.array_equal(cur["_family_ids"], base["_family_ids"]) or not np.array_equal(cur["_y_true"], base["_y_true"]):
                raise ValueError("Compared models must evaluate on identical ordered membership")
            contrasts[(text_view, exp_id, base_arm, model_name)] = compare_models_cluster_bootstrap(
                base["_y_true"],
                base["_y_pred"],
                base["_hit_frozen"],
                cur["_y_pred"],
                cur["_hit_frozen"],
                base["_family_ids"],
                smish_idx=smish_idx,
                B_ci=n_boot_ci,
                B_pval=n_boot_pval,
                seed=seed,
                contrast_index=contrast_idx,
                method="fixed_budget",
            )
            contrast_idx += 1

    hypothesis_summary = evaluate_stage4_hypothesis_families(
        contrasts,
        confirmatory_declarations_enabled=confirmatory_declarations_enabled,
    )
    effective_conf = bool(confirmatory_declarations_enabled and CONFIRMATORY_DECLARATIONS_ENABLED)
    return {
        "n_complete_ensembles": len(ensemble_reports),
        "primary_inference_cohort": (
            "primary_family_decontam_1687" if scored_primary else "supplementary_full_1698"
        ),
        "confirmatory_declarations_enabled": effective_conf,
        "statistical_validation_status": (
            "VALIDATED_CONFIRMATORY" if effective_conf else STATISTICAL_VALIDATION_STATUS
        ),
        "ensembles": ensemble_reports,
        "hypothesis_testing": hypothesis_summary,
    }
