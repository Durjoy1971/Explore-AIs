"""Stage 4 regression tests for Option A grid, optimizer step budgets, cohorts, and T4 benchmark config."""

from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.post_stage_b_cohorts import (
    AMBIGUOUS_TEST_FAMILY_IDS,
    AMBIGUOUS_TEST_ROW_IDS,
    build_smishdetect_test_cohort_masks,
)
from bangla_smish.post_stage_b_data import build_experiment_fold_data
from bangla_smish.post_stage_b_model_view_audit import audit_model_view_overlaps
from bangla_smish.post_stage_b_runner import (
    OPTION_A_ALLOWED_VIEWS,
    OPTION_A_BACKBONES,
    STAGE4_T4_BENCHMARK_UNITS,
    _expected_units,
    compute_optimizer_step_budget,
    load_config,
)


class TestStage4OptionAAndCohorts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v2_bundle = ROOT / "data" / "processed" / "post_stage_b_bundle_v2"
        cls.option_a_cfg = load_config(
            ROOT / "experiments" / "colab" / "configs" / "post_stage_b_stage4_option_a_config.json"
        )
        cls.t4_cfg = load_config(
            ROOT / "experiments" / "colab" / "configs" / "post_stage_b_stage4_t4_benchmark_config.json"
        )

    def test_option_a_config_160_units_and_rejects_unauthorized_views(self) -> None:
        cfg = self.option_a_cfg
        self.assertTrue(cfg.confirmatory_mode)
        self.assertEqual(cfg.run_mode, "experiments_abcd")
        self.assertEqual(cfg.experiments, ("A", "B", "C", "D"))
        self.assertEqual(cfg.models, OPTION_A_BACKBONES)
        self.assertEqual(cfg.text_views, OPTION_A_ALLOWED_VIEWS)
        self.assertEqual(cfg.folds, (0, 1, 2, 3, 4))
        self.assertEqual(cfg.epochs, 3)
        self.assertEqual(cfg.seed, 42)
        self.assertEqual(len(_expected_units(cfg)), 160)

        for bad_view in ("brand_masked", "brand_stripped"):
            with self.subTest(bad_view=bad_view):
                with self.assertRaisesRegex(ValueError, "Unauthorized text_views"):
                    replace(cfg, text_views=("canon_masked", bad_view))

        with self.assertRaisesRegex(ValueError, "requires both views"):
            replace(cfg, text_views=("canon_masked",))

    def test_exact_abcd_membership_weighting_and_equal_epoch_step_budget(self) -> None:
        real = pd.read_parquet(self.v2_bundle / "real_train_pool.parquet")
        for fold in range(5):
            tr_a, va_a, meta_a = build_experiment_fold_data(self.v2_bundle, "A", fold, seed=42)
            tr_b, va_b, meta_b = build_experiment_fold_data(self.v2_bundle, "B", fold, seed=42)
            tr_c, va_c, meta_c = build_experiment_fold_data(self.v2_bundle, "C", fold, seed=42)
            tr_d, va_d, meta_d = build_experiment_fold_data(self.v2_bundle, "D", fold, seed=42)

            self.assertEqual(len(tr_a) + len(va_a), 3999)
            self.assertEqual(tr_a["row_id"].tolist(), tr_b["row_id"].tolist())
            self.assertIsNone(meta_a["class_weights"])
            self.assertIsNotNone(meta_b["class_weights"])
            self.assertEqual(meta_c["n_train_synthetic"], 345)
            self.assertEqual(meta_c["n_train"] + meta_c["n_val"], 4344)
            self.assertEqual(meta_d["n_train_synthetic"], 645)
            self.assertEqual(meta_d["n_train"] + meta_d["n_val"], 4644)

            for va_df in (va_b, va_c, va_d):
                self.assertEqual(va_df["row_id"].tolist(), va_a["row_id"].tolist())
                self.assertFalse(va_df["is_synthetic"].any())

        # Equal-epoch optimizer steps disclosure for Fold 0 (3,199 real rows)
        b_a = compute_optimizer_step_budget(3199, epochs=3, batch_size=32, gradient_accumulation_steps=1)
        b_c = compute_optimizer_step_budget(3544, epochs=3, batch_size=32, gradient_accumulation_steps=1, arm_a_steps=b_a["expected_optimizer_steps"])
        b_d = compute_optimizer_step_budget(3844, epochs=3, batch_size=32, gradient_accumulation_steps=1, arm_a_steps=b_a["expected_optimizer_steps"])
        self.assertEqual(b_a["expected_optimizer_steps"], 300)
        self.assertEqual(b_c["expected_optimizer_steps"], 333)
        self.assertEqual(b_d["expected_optimizer_steps"], 363)
        self.assertAlmostEqual(b_c["step_ratio_vs_arm_a"], 1.11, places=4)
        self.assertAlmostEqual(b_d["step_ratio_vs_arm_a"], 1.21, places=4)

        # Effective batch size invariance with gradient_accumulation_steps=2, batch_size=16
        b_a_acc = compute_optimizer_step_budget(3199, epochs=3, batch_size=16, gradient_accumulation_steps=2)
        self.assertEqual(b_a_acc["effective_batch_size"], 32)
        self.assertEqual(b_a_acc["expected_optimizer_steps"], 300)

        # Option A representations must have 0 exact collisions in v2 bundle
        overlap = audit_model_view_overlaps(self.v2_bundle, views=OPTION_A_ALLOWED_VIEWS)
        self.assertEqual(overlap["checks"], 280)
        self.assertEqual(overlap["failed_checks"], 0)
        self.assertEqual(len(real), 3999)

    def test_frozen_test_cohort_masks_reproducible_and_match_report_json(self) -> None:
        live = build_smishdetect_test_cohort_masks(self.v2_bundle, verify_tfidf_reproduction=False)
        frozen_json = json.loads((ROOT / "reports" / "stage4_frozen_test_masks.json").read_text(encoding="utf-8"))

        self.assertTrue(live["threshold_calibration_independent_of_test_exclusions"])
        self.assertTrue(live["ambiguous_test_all_smishing"])
        self.assertEqual(live["validation_normal_count"], 265)
        self.assertEqual(
            set(live["cohorts"]),
            {
                "primary_family_decontam_1687",
                "supplementary_full_1698",
                "sensitivity_row_only_1693",
                "sensitivity_template_stress_1675",
            },
        )
        for name, expected in frozen_json["cohorts"].items():
            cur = live["cohorts"][name]
            self.assertEqual(cur["n_rows"], expected["n_rows"])
            self.assertEqual(cur["n_families"], expected["n_families"])
            self.assertEqual(cur["by_class"], expected["by_class"])
            self.assertEqual(cur["mask_sha256"], expected["mask_sha256"])
            self.assertEqual(cur["included_row_ids_sha256"], expected["included_row_ids_sha256"])
            self.assertEqual(cur["excluded_row_ids"], expected["excluded_row_ids"])
            self.assertEqual(cur["excluded_family_ids"], expected["excluded_family_ids"])

        prim = live["cohorts"]["primary_family_decontam_1687"]
        self.assertEqual(prim["excluded_family_ids"], list(AMBIGUOUS_TEST_FAMILY_IDS))
        self.assertEqual(live["cohorts"]["sensitivity_row_only_1693"]["excluded_row_ids"], list(AMBIGUOUS_TEST_ROW_IDS))

    def test_t4_benchmark_config_is_5_units_and_blocks_heldout_eval(self) -> None:
        cfg = self.t4_cfg
        self.assertEqual(cfg.run_mode, "t4_benchmark")
        self.assertFalse(cfg.run_heldout_eval)
        self.assertTrue(cfg.benchmark_workload_fixture_eval)
        self.assertEqual(cfg.unit_schedule, STAGE4_T4_BENCHMARK_UNITS)
        self.assertEqual(len(_expected_units(cfg)), 5)
        with self.assertRaisesRegex(ValueError, "run_heldout_eval=False"):
            replace(cfg, run_heldout_eval=True)


if __name__ == "__main__":
    unittest.main()
