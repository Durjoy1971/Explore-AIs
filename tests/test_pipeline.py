"""Sanity tests for schema loading, deduplicated cluster splitting, and metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.metrics import evaluate_predictions, smishing_recall_at_fpr
from bangla_smish.schema import CANONICAL_LABELS, mask_shortcuts
from bangla_smish.splits import build_dedup_grouped_cv, perturb_zwj_homoglyph


def test_mask_shortcuts() -> None:
    txt = "Click https://bit.ly/abc or call 01712345678 or *121# [REDACTED_URL]"
    masked = mask_shortcuts(txt)
    assert "https://bit.ly/abc" not in masked
    assert "01712345678" not in masked
    assert "*121#" not in masked
    assert "[REDACTED_URL]" not in masked
    assert "<URL>" in masked and "<PHONE>" in masked and "<USSD>" in masked


def test_dedup_grouped_cv_zero_leakage() -> None:
    df = pd.DataFrame(
        {
            "text": [f"আপনার বিকাশ একাউন্ট {i} বন্ধ হবে" for i in range(6)]
            + [f"আজকে বিকেলে দেখা হবে বন্ধু {i}" for i in range(6)]
            + [f"স্পেশাল অফার ৩০ জিবি ৩০০ টাকা {i}" for i in range(6)],
            "label_3class": ["Smishing"] * 6 + ["Normal"] * 6 + ["Promotional"] * 6,
            "dataset": ["test"] * 18,
        }
    )
    df["text_norm"] = df["text"].str.strip()
    out = build_dedup_grouped_cv(df, n_splits=3, sim_threshold=0.85)
    assert out.groupby("dup_cluster_id")["fold"].nunique().max() == 1
    assert set(out["fold"].unique()).issubset({0, 1, 2})


def test_metrics_and_perturb() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    probs = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.7, 0.2],
            [0.05, 0.05, 0.9],
            [0.1, 0.1, 0.8],
        ]
    )
    res = evaluate_predictions(y_true, probs, label_names=CANONICAL_LABELS, n_boot=50)
    assert res["macro_f1"] == 100.0
    assert res["normal_as_smishing_fpr_argmax"] == 0.0
    r_fpr = smishing_recall_at_fpr(y_true, probs, smish_idx=2, normal_idx=0, max_fpr=0.01)
    assert r_fpr["recall_at_fpr"] == 100.0
    adv = perturb_zwj_homoglyph("আপনার account বন্ধ হবে")
    assert len(adv) >= len("আপনার account বন্ধ হবে")


if __name__ == "__main__":
    test_mask_shortcuts()
    test_dedup_grouped_cv_zero_leakage()
    test_metrics_and_perturb()
    print("All tests passed!")
