"""Regression cases for the Post-Stage B operating point and uncertainty metrics."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bangla_smish.metrics import (
    evaluate_predictions,
    holm_adjust_pvalues,
    paired_family_accuracy_delta,
    smishing_recall_at_fpr,
)


def probabilities(scores, dtype=np.float64):
    scores = np.asarray(scores, dtype=dtype)
    return np.stack((1 - scores, np.zeros_like(scores), scores), axis=1)


class TestPostStageBMetrics(unittest.TestCase):
    def test_zero_fp_float32_and_float64_with_exact_ties(self):
        y = np.array([0, 0, 2, 2])
        for dtype in (np.float32, np.float64):
            maximum = dtype(0.9)
            above = np.nextafter(maximum, dtype(np.inf))
            probs = probabilities([maximum, 0.1, maximum, above], dtype)
            result = smishing_recall_at_fpr(y, probs, 2, 0)
            self.assertEqual(result["false_positives"], 0)
            self.assertEqual(result["n_normal"], 2)
            self.assertEqual(result["actual_fpr"], 0.0)
            self.assertEqual(result["recall_at_fpr"], 50.0)
            self.assertGreater(result["threshold"], float(maximum))
            stored = json.loads(json.dumps(result))
            self.assertEqual(
                np.count_nonzero(probs[y == 0, 2].astype(np.float64) >= stored["threshold"]), 0
            )

    def test_budget_boundary_ties_and_budget_one(self):
        y = np.array([0] * 100 + [2, 2, 2])
        tied = smishing_recall_at_fpr(
            y, probabilities([0.8, 0.8] + [0.1] * 98 + [0.8, 0.9, 0.1]), 2, 0
        )
        self.assertEqual(tied["false_positives"], 0)
        self.assertAlmostEqual(tied["recall_at_fpr"], 100 / 3)
        untied = smishing_recall_at_fpr(
            y, probabilities([0.9, 0.8] + [0.1] * 98 + [0.8, 0.9, 0.1]), 2, 0
        )
        self.assertEqual(untied["false_positives"], 1)
        self.assertEqual(untied["actual_fpr"], 1.0)
        self.assertTrue(untied["constraint_met"])

    def test_missing_classes_and_fixed_threshold_breach(self):
        missing_normal = smishing_recall_at_fpr(np.array([2]), probabilities([0.9]), 2, 0)
        self.assertIsNone(missing_normal["threshold"])
        self.assertIsNone(missing_normal["actual_fpr"])
        self.assertIsNone(missing_normal["constraint_met"])
        missing_smishing = smishing_recall_at_fpr(np.array([0]), probabilities([1.0]), 2, 0)
        self.assertIsNone(missing_smishing["recall_at_fpr"])
        self.assertGreater(missing_smishing["threshold"], 1)
        self.assertEqual(missing_smishing["false_positives"], 0)
        y = np.array([0, 1, 2])
        probs = probabilities([0.9, 0.8, 0.9])
        result = evaluate_predictions(y, probs, n_boot=20, operating_threshold=0.8)
        self.assertEqual(result["smishing_low_fpr"]["actual_fpr"], 100)
        self.assertEqual(result["smishing_low_fpr"]["recall_at_fpr"], 100)
        self.assertEqual(result["smishing_low_fpr"]["threshold_source"], "fixed_validation")
        self.assertIsNone(result["smishing_recall_at_1pct_fpr"])
        self.assertEqual(result["promotional_as_smishing_fpr_argmax"], 100)

    def test_operating_point_matches_exhaustive_threshold_search(self):
        rng = np.random.default_rng(42)
        for dtype in (np.float32, np.float64):
            for n_normal in (1, 9, 99, 100, 101, 500):
                scores = rng.integers(0, 12, size=n_normal + 50) / 11
                labels = np.array([0] * n_normal + [2] * 50)
                probs = probabilities(scores, dtype)
                result = smishing_recall_at_fpr(labels, probs, 2, 0)
                numeric_scores = probs[:, 2].astype(np.float64)
                thresholds = np.append(np.unique(numeric_scores), np.nextafter(1.0, np.inf))
                possibilities = [int((numeric_scores[n_normal:] >= threshold).sum()) for threshold in thresholds
                                 if (numeric_scores[:n_normal] >= threshold).sum() <= n_normal // 100]
                self.assertTrue(result["constraint_met"])
                self.assertEqual(result["recall_at_fpr"], 100.0 * max(possibilities) / 50)
                self.assertEqual(result["false_positives"], int((numeric_scores[:n_normal] >= result["threshold"]).sum()))

    def test_cluster_bootstrap_preserves_perfect_family_dependence(self):
        y = np.full(40, 2)
        probs = probabilities([0.99] * 20 + [0.01] * 20)
        families = np.repeat(["one", "two"], 20)
        result = evaluate_predictions(y, probs, family_ids=families, n_boot=100)
        self.assertEqual(result["bootstrap_unit"], "family")
        self.assertEqual(result["smishing_recall_argmax"], 50)
        self.assertEqual(result["smishing_recall_argmax_ci95"], [0, 100])
        paired = paired_family_accuracy_delta(y, np.zeros(40), probs.argmax(1), families, n_resamples=100)
        self.assertEqual(paired["accuracy_delta_pp"], 50)
        self.assertEqual(paired["accuracy_delta_ci95_pp"], [0, 100])

    def test_holm_step_down_is_monotonic(self):
        result = holm_adjust_pvalues({"a": 0.01, "b": 0.03, "c": 0.04})
        self.assertEqual(result, {"a": 0.03, "b": 0.06, "c": 0.06})


if __name__ == "__main__":
    unittest.main()
