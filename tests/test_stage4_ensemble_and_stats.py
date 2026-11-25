"""Stage 4 regression tests for 5-fold ensembles, validation calibration, fixed-budget stats, and Holm families."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.post_stage_b_cohorts import build_smishdetect_test_cohort_masks
from bangla_smish.post_stage_b_data import load_heldout_eval_sets
from bangla_smish.post_stage_b_ensemble import (
    EXPECTED_SMISHDETECT_VAL_NORMAL,
    build_ensemble_evaluation_summary,
    calibrate_ensemble_threshold,
    validate_fold_predictions_for_ensemble,
)
from bangla_smish.post_stage_b_stats import (
    DEFAULT_B_CI,
    DEFAULT_B_PVAL,
    compare_models_cluster_bootstrap,
    compare_models_reject_only_for_validation,
    evaluate_stage4_hypothesis_families,
    macro_f1_delta_tp_e,
    macro_f1_grad_tp_e,
)


def _synth_probs_for_frame(df: pd.DataFrame, *, seed: int, boost: float = 2.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = df["label_idx"].to_numpy(dtype=int)
    p = rng.dirichlet([0.5, 0.5, 0.5], size=len(y)).astype(np.float64)
    for i, c in enumerate(y):
        p[i, c] += boost
    p /= p.sum(axis=1, keepdims=True)
    return p


class TestStage4EnsembleAndStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v2_bundle = ROOT / "data" / "processed" / "post_stage_b_bundle_v2"
        cls.eval_sets = load_heldout_eval_sets(cls.v2_bundle)
        masks_payload = build_smishdetect_test_cohort_masks(cls.v2_bundle, verify_tfidf_reproduction=False)
        cls.test_masks = {k: v["mask"] for k, v in masks_payload["cohorts"].items()}

    def _make_fold_entries(self, exp_id: str, model: str, view: str, *, boost: float = 2.5) -> dict[int, dict]:
        val_f = self.eval_sets["smishdetect_validation"]
        test_f = self.eval_sets["smishdetect_test"]
        ood_f = self.eval_sets["banglaphish_ood"]
        out = {}
        for fold in range(5):
            seed = 1000 + hash((exp_id, model, view)) % 1000 + fold
            out[fold] = {
                "experiment_id": exp_id,
                "model_name": model,
                "text_view": view,
                "model_revision": "rev_model_123",
                "tokenizer_revision": "rev_tok_123",
                "config_hash": "cfg_abc",
                "dataset_manifest_sha256": "man_xyz",
                "arrays": {
                    "smishdetect_validation_row_ids": val_f["row_id"].to_numpy(dtype=str),
                    "smishdetect_validation_y": val_f["label_idx"].to_numpy(dtype=int),
                    "smishdetect_validation_probs": _synth_probs_for_frame(val_f, seed=seed + 1, boost=boost),
                    "smishdetect_test_row_ids": test_f["row_id"].to_numpy(dtype=str),
                    "smishdetect_test_y": test_f["label_idx"].to_numpy(dtype=int),
                    "smishdetect_test_probs": _synth_probs_for_frame(test_f, seed=seed + 2, boost=boost),
                    "banglaphish_ood_row_ids": ood_f["row_id"].to_numpy(dtype=str),
                    "banglaphish_ood_y": ood_f["label_idx"].to_numpy(dtype=int),
                    "banglaphish_ood_probs": _synth_probs_for_frame(ood_f, seed=seed + 3, boost=boost),
                },
            }
        return out

    def test_ensemble_rejects_incomplete_or_misaligned_folds(self) -> None:
        frames = {"smishdetect_validation": self.eval_sets["smishdetect_validation"]}
        folds = self._make_fold_entries("A", "csebuetnlp/banglabert", "canon_masked")
        ok = validate_fold_predictions_for_ensemble(folds, frames)
        self.assertIn("smishdetect_validation", ok["averaged_probs"])

        # Missing fold
        incomplete = {f: folds[f] for f in range(4)}
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_fold_predictions_for_ensemble(incomplete, frames)

        # Revision mismatch
        bad_rev = self._make_fold_entries("A", "csebuetnlp/banglabert", "canon_masked")
        bad_rev[3]["tokenizer_revision"] = "wrong_tok"
        with self.assertRaisesRegex(ValueError, "tokenizer_revision"):
            validate_fold_predictions_for_ensemble(bad_rev, frames)

        # Misordered row IDs
        bad_ids = self._make_fold_entries("A", "csebuetnlp/banglabert", "canon_masked")
        bad_ids[2]["arrays"]["smishdetect_validation_row_ids"] = bad_ids[2]["arrays"]["smishdetect_validation_row_ids"][::-1]
        with self.assertRaisesRegex(ValueError, "row_ids misaligned"):
            validate_fold_predictions_for_ensemble(bad_ids, frames)

        # Unnormalized probabilities
        bad_probs = self._make_fold_entries("A", "csebuetnlp/banglabert", "canon_masked")
        bad_probs[1]["arrays"]["smishdetect_validation_probs"][0, 0] += 0.5
        with self.assertRaisesRegex(ValueError, "Invalid probability matrix"):
            validate_fold_predictions_for_ensemble(bad_probs, frames)

    def test_threshold_calibration_265_normal_and_frozen_transfer_exports(self) -> None:
        val_f = self.eval_sets["smishdetect_validation"]
        val_probs = _synth_probs_for_frame(val_f, seed=42, boost=3.0)
        cal = calibrate_ensemble_threshold(val_f, val_probs)
        self.assertEqual(cal["n_normal"], EXPECTED_SMISHDETECT_VAL_NORMAL)
        self.assertLessEqual(cal["false_positives"], 2)
        self.assertLessEqual(cal["actual_fpr"], 1.0)
        self.assertTrue(cal["constraint_met"])

    def test_macro_f1_influence_gradient_vs_finite_differences(self) -> None:
        rng = np.random.default_rng(77)
        eps = 1e-6
        for _ in range(10):
            # Cluster-mean scale (~0.05..0.50) matching Round 4 verify_macro_f1_if.py
            vec = rng.uniform(0.05, 0.50, size=12)
            ana = macro_f1_grad_tp_e(vec, scale=1.0)
            fd = np.empty(12, dtype=np.float64)
            for j in range(12):
                vp, vm = vec.copy(), vec.copy()
                vp[j] += eps
                vm[j] -= eps
                fd[j] = (macro_f1_delta_tp_e(vp) - macro_f1_delta_tp_e(vm)) / (2.0 * eps)
            rel_err = float(np.max(np.abs(ana - fd) / np.maximum(np.abs(fd), 1e-12)))
            self.assertLess(rel_err, 1e-8)

    def test_fixed_budget_bootstrap_literal_defaults_and_shortcircuit_parity(self) -> None:
        rng = np.random.default_rng(42)
        y = rng.integers(0, 3, size=300)
        fams = np.repeat(np.arange(150), 2)
        pA = rng.integers(0, 3, size=300)
        pM = np.where(rng.random(300) < 0.75, y, pA)
        hA = (y == 2) & (rng.random(300) < 0.6)
        hM = (y == 2) & (rng.random(300) < 0.85)
        full = compare_models_cluster_bootstrap(
            y, pA, hA, pM, hM, fams, B_ci=DEFAULT_B_CI, B_pval=DEFAULT_B_PVAL, seed=20261124, contrast_index=3
        )
        fast = compare_models_reject_only_for_validation(
            y, pA, hA, pM, hM, fams, alpha=0.05, B_pval=DEFAULT_B_PVAL, seed=20261124, contrast_index=3
        )
        self.assertEqual(full["B_ci"], 2000)
        self.assertEqual(full["B_pval"], 50000)
        self.assertEqual(full["p_macro_f1"] <= 0.05, fast["reject_macro_f1"])
        self.assertEqual(full["p_recall"] <= 0.05, fast["reject_recall"])
        self.assertEqual(full["p_joint_iut"] <= 0.05, fast["reject_joint_iut"])

    def test_multiplicity_separates_holm12_holm24_exploratory_and_raw_and_fail_closed_gate(self) -> None:
        models = (
            "csebuetnlp/banglabert",
            "google/muril-base-cased",
            "xlm-roberta-base",
            "distilbert-base-multilingual-cased",
        )
        contrasts = {}
        for view in ("canon_masked", "raw"):
            for arm_m, arm_a in (("B", "A"), ("C", "A"), ("D", "A"), ("D", "C")):
                for b_idx, model in enumerate(models):
                    p_f1 = 0.0005 if (arm_m == "D" and arm_a == "A" and b_idx == 0) else 0.25
                    p_rec = 0.0008 if (arm_m == "D" and arm_a == "A" and b_idx == 0) else 0.30
                    contrasts[(view, arm_m, arm_a, model)] = {
                        "macro_f1_delta": 0.02,
                        "recall_delta": 0.03,
                        "p_macro_f1": p_f1,
                        "p_recall": p_rec,
                        "p_joint_iut": max(p_f1, p_rec),
                    }

        from unittest.mock import patch

        with patch("bangla_smish.post_stage_b_stats.CONFIRMATORY_DECLARATIONS_ENABLED", True):
            enabled = evaluate_stage4_hypothesis_families(contrasts, confirmatory_declarations_enabled=True)
            self.assertEqual(enabled["confirmatory_status"], "ENABLED_VALIDATED")
            self.assertFalse(enabled["multiplicity_structure"]["pooled_36_used"])
            self.assertEqual(enabled["primary_confirmatory_holm12"]["actual_family_size"], 12)
            self.assertTrue(enabled["primary_confirmatory_holm12"]["complete_family"])
            self.assertEqual(enabled["supporting_endpoints_holm24"]["actual_family_size"], 24)
            self.assertTrue(enabled["supporting_endpoints_holm24"]["complete_family"])
            self.assertEqual(enabled["exploratory_c_vs_d_pilot_contribution"]["actual_joint_family_size"], 4)
            # Check separate Holm-12 vs Holm-24 multipliers: 0.0008 * 12 = 0.0096; 0.0008 * 23 = 0.0184
            p12_adj = enabled["primary_confirmatory_holm12"]["results"]["D_vs_A__csebuetnlp/banglabert__canon_masked"]["holm_adjusted_p_value"]
            p24_adj = enabled["supporting_endpoints_holm24"]["results"]["D_vs_A__csebuetnlp/banglabert__canon_masked__recall"]["holm_adjusted_p_value"]
            self.assertAlmostEqual(p12_adj, 0.0008 * 12, places=8)
            self.assertAlmostEqual(p24_adj, 0.0008 * 23, places=8)  # p_f1=0.0005 rank 0 (*24), p_rec=0.0008 rank 1 (*23)
            # Exploratory C vs D NEVER declares confirmatory superiority
            for v in enabled["exploratory_c_vs_d_pilot_contribution"]["joint_holm4"].values():
                self.assertFalse(v["rejected_at_alpha"])

        # Fail-closed disables all confirmatory superiority declarations EVEN if caller attempts bypass
        disabled = evaluate_stage4_hypothesis_families(contrasts, confirmatory_declarations_enabled=True)
        self.assertEqual(disabled["confirmatory_status"], "FAIL_OR_INCONCLUSIVE_CONFIRMATORY_DISABLED")
        self.assertTrue(disabled["governance_bypass_blocked"])
        self.assertTrue(all(not v["rejected_at_alpha"] for v in disabled["primary_confirmatory_holm12"]["results"].values()))
        self.assertTrue(all(not v["rejected_at_alpha"] for v in disabled["supporting_endpoints_holm24"]["results"].values()))

    def test_full_ensemble_summary_exports_all_required_fields(self) -> None:
        groups = {
            ("A", "csebuetnlp/banglabert", "canon_masked"): self._make_fold_entries("A", "csebuetnlp/banglabert", "canon_masked", boost=1.8),
            ("C", "csebuetnlp/banglabert", "canon_masked"): self._make_fold_entries("C", "csebuetnlp/banglabert", "canon_masked", boost=2.6),
            ("D", "csebuetnlp/banglabert", "canon_masked"): self._make_fold_entries("D", "csebuetnlp/banglabert", "canon_masked", boost=2.8),
        }
        summary = build_ensemble_evaluation_summary(
            ensemble_groups=groups,
            eval_frames=self.eval_sets,
            test_masks=self.test_masks,
            n_boot_ci=50,
            n_boot_pval=100,
            seed=42,
        )
        self.assertEqual(summary["n_complete_ensembles"], 3)
        self.assertEqual(summary["primary_inference_cohort"], "primary_family_decontam_1687")
        ens = summary["ensembles"]["exp_D__csebuetnlp/banglabert__canon_masked"]
        self.assertEqual(ens["calibration"]["n_normal"], 265)
        for cohort in self.test_masks:
            rep = ens["smishdetect_test_cohorts"][cohort]
            for req in (
                "threshold",
                "calibration_achieved_fpr_pct",
                "achieved_fpr_pct",
                "normal_false_positives",
                "normal_denominator",
                "promotional_false_positives_at_threshold",
                "promotional_denominator",
                "smishing_recall_at_frozen_threshold_pct",
                "macro_f1_pct",
                "roc_auc_smishing_vs_rest",
                "pr_auc_smishing_vs_rest",
                "confusion_matrix_argmax",
                "constraint_met",
                "per_class",
            ):
                self.assertIn(req, rep)
        self.assertIn("overall", ens["banglaphish_ood"])
        self.assertIn("C_vs_A__csebuetnlp/banglabert__canon_masked", summary["hypothesis_testing"]["descriptive_effects"])
        self.assertIn("D_vs_C__csebuetnlp/banglabert__canon_masked", summary["hypothesis_testing"]["exploratory_c_vs_d_pilot_contribution"]["joint_holm4"])

    def test_degenerate_cases_and_adaptive_non_confirmatory_flag(self) -> None:
        y = np.array([0, 1, 2] * 20)
        fams = np.repeat(np.arange(30), 2)
        zero = compare_models_cluster_bootstrap(
            y, y, y == 2, y, y == 2, fams, B_ci=100, B_pval=200, seed=42
        )
        self.assertEqual(zero["p_macro_f1"], 1.0)
        self.assertEqual(zero["p_recall"], 1.0)
        self.assertEqual(zero["macro_f1_ci95"], [0.0, 0.0])

        y_no_smish = np.array([0, 1] * 20)
        f_ns = np.repeat(np.arange(20), 2)
        ns = compare_models_cluster_bootstrap(
            y_no_smish, y_no_smish, np.zeros(40, dtype=bool), 1 - y_no_smish, np.zeros(40, dtype=bool),
            f_ns, B_ci=100, B_pval=200, seed=42
        )
        self.assertIn("missing_smishing_class", ns["degenerate_flags"])
        self.assertEqual(ns["p_recall"], 1.0)

        adap = compare_models_cluster_bootstrap(
            y, y, y == 2, y, y == 2, fams, B_ci=50, B_pval=100, method="adaptive_non_confirmatory"
        )
        self.assertFalse(adap["confirmatory_allowed"])
        from bangla_smish.post_stage_b_stats import CONFIRMATORY_DECLARATIONS_ENABLED
        self.assertFalse(CONFIRMATORY_DECLARATIONS_ENABLED)


if __name__ == "__main__":
    unittest.main()
