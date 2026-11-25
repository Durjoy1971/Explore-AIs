"""Evaluation metrics: Macro-F1, Smishing Recall @ FPR<=1%, Bootstrap CIs, McNemar, ECE."""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Any

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import classification_report


class _ThresholdSelection(Enum):
    EMPIRICAL = "empirical"


def smishing_recall_at_fpr(
    y_true: np.ndarray,
    probs: np.ndarray,
    smish_idx: int,
    normal_idx: int,
    max_fpr: float = 0.01,
    *,
    operating_threshold: float | None | _ThresholdSelection = _ThresholdSelection.EMPIRICAL,
) -> dict[str, Any]:
    """Use float64 score >= threshold and report rates in percent without rounding.

    With empirical selection, exclude the first Normal score outside the integer
    FP budget and its ties using the next representable float64 threshold.
    A supplied threshold is scored unchanged, even when its FPR exceeds the cap.
    """
    if not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must be in [0, 1]")
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    if probs.ndim != 2 or y_true.shape != (len(probs),):
        raise ValueError("Expected aligned labels and probabilities")
    scores = probs[:, smish_idx].astype(np.float64, copy=False)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Expected finite Smishing probabilities in [0, 1]")
    is_smish = y_true == smish_idx
    is_normal = y_true == normal_idx
    n_normal = int(is_normal.sum())
    n_smish = int(is_smish.sum())
    allowed_fp = int(Decimal(str(max_fpr)) * n_normal)

    if operating_threshold is _ThresholdSelection.EMPIRICAL:
        threshold = None
        if n_normal:
            if allowed_fp == n_normal:
                threshold = float(scores.min())
            else:
                descending = np.sort(scores[is_normal])[::-1]
                threshold = float(np.nextafter(descending[allowed_fp], np.inf))
        source = "empirical_same_sample"
    else:
        threshold = operating_threshold
        source = "fixed_validation"
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("Expected a finite threshold")
    fp = recall = fpr = constraint_met = None
    if threshold is not None:
        predicted = scores >= threshold
        fp = int((predicted & is_normal).sum())
        recall = 100.0 * int((predicted & is_smish).sum()) / n_smish if n_smish else None
        fpr = 100.0 * fp / n_normal if n_normal else None
        constraint_met = fp <= allowed_fp if n_normal else None
    return {
        "recall_at_fpr": recall,
        "actual_fpr": fpr,
        "threshold": threshold,
        "false_positives": fp,
        "n_normal": n_normal,
        "n_smishing": n_smish,
        "constraint_met": constraint_met,
        "max_fpr": max_fpr,
        "comparison": "float64_score >= float64_threshold",
        "threshold_source": source,
    }


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Compute multiclass top-confidence Expected Calibration Error (ECE)."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1]) if i > 0 else (confidences >= bins[i]) & (confidences <= bins[i + 1])
        if mask.any():
            ece += (mask.sum() / len(y_true)) * abs(accuracies[mask].mean() - confidences[mask].mean())
    return round(float(ece), 4)


def _cluster_indices(length: int, family_ids: np.ndarray | None) -> tuple[np.ndarray, int]:
    if family_ids is None:
        return np.arange(length), length
    families = np.asarray(family_ids, dtype=str)
    if families.shape != (length,) or np.isin(families, ["", "None", "nan", "<NA>"]).any():
        raise ValueError("Expected one nonempty family ID per prediction row")
    unique, indices = np.unique(families, return_inverse=True)
    return indices, len(unique)


def _confusion_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    smish_idx: int,
    n_resamples: int,
    family_ids: np.ndarray | None,
    seed: int = 42,
) -> tuple[dict[str, float | None], dict[str, list[float] | None]]:
    """Percentile intervals resampling complete families or individual rows."""
    cluster_indices, n_clusters = _cluster_indices(len(y_true), family_ids)
    keys = ("accuracy", "macro_f1", "smishing_recall_argmax")
    if not n_clusters:
        return dict.fromkeys(keys), dict.fromkeys(keys)
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    cells = np.zeros((n_clusters, n_classes, n_classes), dtype=np.int64)
    np.add.at(cells, (cluster_indices, y_true, y_pred), 1)

    def stats(cm: np.ndarray) -> tuple[float, float, float]:
        tp = cm.diagonal()
        support = cm.sum(axis=1)
        denom = support + cm.sum(axis=0)
        class_f1 = np.divide(2 * tp, denom, out=np.zeros(n_classes), where=denom > 0)
        return tp.sum() / cm.sum(), class_f1.mean(), tp[smish_idx] / support[smish_idx] if support[smish_idx] else np.nan

    point = stats(cells.sum(axis=0))
    rng = np.random.default_rng(seed)
    values = np.empty((n_resamples, len(keys)))
    for i in range(n_resamples):
        sample = rng.integers(n_clusters, size=n_clusters)
        values[i] = stats(cells[sample].sum(axis=0))
    estimates = {}
    intervals = {}
    for idx, key in enumerate(keys):
        valid = values[np.isfinite(values[:, idx]), idx]
        estimates[key] = round(float(point[idx]) * 100.0, 2) if np.isfinite(point[idx]) else None
        intervals[key] = np.round(np.percentile(valid, [2.5, 97.5]) * 100.0, 2).tolist() if len(valid) else None
    return estimates, intervals


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high) via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = float(metric_fn(y_true, y_pred))
    vals = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        vals[i] = float(metric_fn(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(point * 100.0, 2), round(float(lo) * 100.0, 2), round(float(hi) * 100.0, 2)


def mcnemar_pvalue(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, Any]:
    """Exact binomial McNemar test comparing two classifiers on identical test instances."""
    if y_true.ndim != 1 or pred_a.shape != y_true.shape or pred_b.shape != y_true.shape:
        raise ValueError("McNemar requires aligned one-dimensional predictions")
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    n01 = int((~correct_a & correct_b).sum())
    n10 = int((correct_a & ~correct_b).sum())
    n_disc = n01 + n10
    pval = 1.0 if n_disc == 0 else float(binomtest(min(n01, n10), n_disc, 0.5).pvalue)
    return {"n_a_only": n10, "n_b_only": n01, "p_value": pval}


def holm_adjust_pvalues(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm familywise-error adjusted p-values for the supplied comparison family."""
    adjusted = {}
    running_max = 0.0
    for rank, (key, pvalue) in enumerate(sorted(pvalues.items(), key=lambda item: item[1])):
        running_max = max(running_max, min(1.0, (len(pvalues) - rank) * pvalue))
        adjusted[key] = running_max
    return adjusted


def paired_family_accuracy_delta(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    family_ids: np.ndarray,
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap paired (B minus A) accuracy deltas over complete families."""
    if y_true.ndim != 1 or pred_a.shape != y_true.shape or pred_b.shape != y_true.shape:
        raise ValueError("Paired bootstrap requires aligned one-dimensional predictions")
    indices, n_clusters = _cluster_indices(len(y_true), family_ids)
    delta = (pred_b == y_true).astype(int) - (pred_a == y_true)
    group_delta = np.bincount(indices, weights=delta, minlength=n_clusters)
    group_sizes = np.bincount(indices, minlength=n_clusters)
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples)
    for i in range(n_resamples):
        draw = rng.integers(n_clusters, size=n_clusters)
        values[i] = group_delta[draw].sum() / group_sizes[draw].sum()
    return {
        "accuracy_delta_pp": round(float(delta.mean()) * 100, 2),
        "accuracy_delta_ci95_pp": np.round(np.percentile(values, [2.5, 97.5]) * 100, 2).tolist(),
    }


def evaluate_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    label_names: tuple[str, ...] = ("Normal", "Promotional", "Smishing"),
    n_boot: int = 1000,
    *,
    family_ids: np.ndarray | None = None,
    operating_threshold: float | None | _ThresholdSelection = _ThresholdSelection.EMPIRICAL,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate predictions and optionally apply a threshold learned on validation."""
    y_pred = probs.argmax(axis=1)
    normal_idx = label_names.index("Normal")
    promo_idx = label_names.index("Promotional")
    smish_idx = label_names.index("Smishing")
    points, intervals = _confusion_intervals(
        y_true, y_pred, len(label_names), smish_idx, n_boot, family_ids, seed=seed
    )
    fpr_stats = smishing_recall_at_fpr(
        y_true, probs, smish_idx, normal_idx, operating_threshold=operating_threshold
    )

    def argmax_false_positive_pct(label_idx: int) -> float | None:
        mask = y_true == label_idx
        return round(float((y_pred[mask] == smish_idx).mean()) * 100.0, 2) if mask.any() else None

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=list(label_names),
        output_dict=True,
        zero_division=0,
    )

    return {
        "n_samples": int(len(y_true)),
        **points,
        **{f"{key}_ci95": interval for key, interval in intervals.items()},
        "bootstrap_unit": "family" if family_ids is not None else "row",
        "smishing_f1": round(float(report["Smishing"]["f1-score"] * 100.0), 2),
        "normal_as_smishing_fpr_argmax": argmax_false_positive_pct(normal_idx),
        "promotional_as_smishing_fpr_argmax": argmax_false_positive_pct(promo_idx),
        "smishing_recall_at_1pct_fpr": fpr_stats["recall_at_fpr"] if fpr_stats["constraint_met"] else None,
        "smishing_low_fpr": fpr_stats,
        "ece": expected_calibration_error(y_true, probs),
        "per_class": {
            lbl: {
                "precision": round(float(report[lbl]["precision"] * 100.0), 2),
                "recall": round(float(report[lbl]["recall"] * 100.0), 2),
                "f1": round(float(report[lbl]["f1-score"] * 100.0), 2),
                "support": int(report[lbl]["support"]),
            }
            for lbl in label_names
        },
    }
