"""Evaluation metrics: Macro-F1, Smishing Recall @ FPR<=1%, Bootstrap CIs, McNemar, ECE."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, classification_report, f1_score


def smishing_recall_at_fpr(
    y_true: np.ndarray,
    probs: np.ndarray,
    smish_idx: int,
    normal_idx: int,
    max_fpr: float = 0.01,
) -> dict[str, float]:
    """Compute Smishing Recall at threshold keeping Normal-as-Smishing FPR <= max_fpr."""
    is_smish = y_true == smish_idx
    is_normal = y_true == normal_idx
    smish_scores = probs[:, smish_idx]

    normal_scores = np.sort(smish_scores[is_normal])[::-1]
    if len(normal_scores) == 0 or is_smish.sum() == 0:
        return {"recall_at_fpr": 0.0, "actual_fpr": 0.0, "threshold": 1.0}

    allowed_fp = int(np.floor(max_fpr * len(normal_scores)))
    if allowed_fp == 0:
        threshold = float(normal_scores[0] + 1e-9)
    else:
        threshold = float(normal_scores[allowed_fp - 1])

    preds_smish = smish_scores >= threshold
    tp = int((preds_smish & is_smish).sum())
    fp = int((preds_smish & is_normal).sum())
    recall = tp / float(is_smish.sum())
    actual_fpr = fp / float(is_normal.sum())
    return {
        "recall_at_fpr": round(recall * 100.0, 2),
        "actual_fpr": round(actual_fpr * 100.0, 2),
        "threshold": round(threshold, 4),
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


def _fast_acc_and_macro_f1_boot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Fast vectorized bootstrap CIs for accuracy and macro-F1 via confusion-matrix bincount."""
    n = len(y_true)
    if n == 0:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    def _stats(yt: np.ndarray, yp: np.ndarray) -> tuple[float, float]:
        cm = np.bincount(yt * n_classes + yp, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0).astype(float) - tp
        fn = cm.sum(axis=1).astype(float) - tp
        denom = 2.0 * tp + fp + fn
        f1s = np.where(denom > 0, (2.0 * tp) / np.where(denom > 0, denom, 1.0), 0.0)
        return float(tp.sum() / n), float(f1s.mean())

    pt_acc, pt_f1 = _stats(y_true, y_pred)
    rng = np.random.default_rng(seed)
    accs = np.empty(n_resamples, dtype=float)
    f1s = np.empty(n_resamples, dtype=float)
    pair_codes = y_true * n_classes + y_pred
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        cm = np.bincount(pair_codes[idx], minlength=n_classes * n_classes).reshape(n_classes, n_classes)
        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0).astype(float) - tp
        fn = cm.sum(axis=1).astype(float) - tp
        denom = 2.0 * tp + fp + fn
        accs[i] = float(tp.sum() / n)
        f1s[i] = float(np.where(denom > 0, (2.0 * tp) / np.where(denom > 0, denom, 1.0), 0.0).mean())

    acc_lo, acc_hi = np.percentile(accs, [2.5, 97.5])
    f1_lo, f1_hi = np.percentile(f1s, [2.5, 97.5])
    return (
        (round(pt_acc * 100.0, 2), round(float(acc_lo) * 100.0, 2), round(float(acc_hi) * 100.0, 2)),
        (round(pt_f1 * 100.0, 2), round(float(f1_lo) * 100.0, 2), round(float(f1_hi) * 100.0, 2)),
    )


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
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    n01 = int((~correct_a & correct_b).sum())
    n10 = int((correct_a & ~correct_b).sum())
    n_disc = n01 + n10
    pval = 1.0 if n_disc == 0 else float(binomtest(min(n01, n10), n_disc, 0.5).pvalue)
    return {"n_a_only": n10, "n_b_only": n01, "p_value": round(pval, 6)}


def evaluate_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    label_names: tuple[str, ...] = ("Normal", "Promotional", "Smishing"),
    n_boot: int = 1000,
) -> dict[str, Any]:
    """Full evaluation bundle matching the research proposal metrics."""
    y_pred = probs.argmax(axis=1)
    (acc, acc_lo, acc_hi), (mf1, mf1_lo, mf1_hi) = _fast_acc_and_macro_f1_boot(
        y_true, y_pred, n_classes=len(label_names), n_resamples=n_boot
    )
    normal_idx = label_names.index("Normal")
    smish_idx = label_names.index("Smishing")
    fpr_stats = smishing_recall_at_fpr(y_true, probs, smish_idx=smish_idx, normal_idx=normal_idx, max_fpr=0.01)

    # Also compute Smishing FPR at argmax prediction
    normal_mask = y_true == normal_idx
    argmax_fpr = float(((y_pred == smish_idx) & normal_mask).sum() / max(1, int(normal_mask.sum())) * 100.0)
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
        "accuracy": acc,
        "accuracy_ci95": [acc_lo, acc_hi],
        "macro_f1": mf1,
        "macro_f1_ci95": [mf1_lo, mf1_hi],
        "smishing_recall_argmax": round(float(report["Smishing"]["recall"] * 100.0), 2),
        "smishing_f1": round(float(report["Smishing"]["f1-score"] * 100.0), 2),
        "normal_as_smishing_fpr_argmax": round(argmax_fpr, 2),
        "smishing_recall_at_1pct_fpr": fpr_stats["recall_at_fpr"],
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
