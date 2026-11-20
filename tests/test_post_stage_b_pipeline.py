"""Tests for Post-Stage B dataset assembly, manifest verification, A/B/C/D splits, and resume runner."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.post_stage_b_data import (
    EXPERIMENT_SPECS,
    assemble_post_stage_b_bundle,
    audit_post_stage_b_bundle,
    build_experiment_fold_data,
    select_counterfactual_fold,
    training_class_weights,
    verify_manifest,
    verify_protected_inputs,
)
from bangla_smish.post_stage_b_runner import (
    PostStageBConfig,
    load_config,
    run_post_stage_b_pipeline,
)
from bangla_smish.schema import CANONICAL_LABELS


def _mock_train_eval_fn(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    heldout_sets: dict[str, pd.DataFrame],
    model_name: str,
    text_view: str,
    checkpoint_dir: Path,
    cfg: PostStageBConfig,
    class_weights: list[float] | None,
    seed: int,
    resume_from_checkpoint: bool = True,
) -> dict[str, Any]:
    """Deterministic CPU mock trainer matching hf_train_and_predict_fold output contract."""
    rng = np.random.default_rng(seed)

    def _probs_for(y_idx: np.ndarray) -> np.ndarray:
        probs = rng.dirichlet([0.5, 0.5, 0.5], size=len(y_idx))
        for i, cls in enumerate(y_idx):
            probs[i, cls] += 2.0
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    preds: dict[str, np.ndarray] = {
        "val_probs": _probs_for(val_df["label_idx"].to_numpy(dtype=int)),
    }
    if cfg.run_adversarial_eval:
        n_adv = cfg.smoke_max_eval_samples or len(val_df)
        preds["val_adv_probs"] = _probs_for(val_df["label_idx"].to_numpy(dtype=int)[:n_adv])
    if cfg.run_heldout_eval:
        for k in ("smishdetect_validation", "smishdetect_test", "banglaphish_ood"):
            edf = heldout_sets[k]
            if cfg.smoke_max_eval_samples is not None:
                edf = edf.iloc[: cfg.smoke_max_eval_samples]
            preds[f"{k}_probs"] = _probs_for(edf["label_idx"].to_numpy(dtype=int))
    if cfg.run_counterfactual_eval:
        cf = heldout_sets["counterfactual_smish_probe"]
        if cfg.smoke_max_eval_samples is not None:
            cf = cf.iloc[: cfg.smoke_max_eval_samples]
        y_cf = cf["label_idx"].to_numpy(dtype=int)
        preds["cf_orig_probs"] = _probs_for(y_cf)
        preds["cf_contact_probs"] = _probs_for(y_cf)
        preds["cf_brand_probs"] = _probs_for(y_cf)

    return {
        "train_seconds": 0.05,
        "resumed_from_checkpoint": None,
        "fp16": False,
        "bf16": False,
        "predictions": preds,
    }


class TestPostStageBPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.bundle_dir = Path(cls._tmp.name) / "bundle"
        cls.manifest = assemble_post_stage_b_bundle(ROOT, cls.bundle_dir, strict_digests=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_protected_inputs_and_manifest_integrity(self) -> None:
        protected = verify_protected_inputs(ROOT, strict_digests=True)
        self.assertEqual(len(protected), 13)
        verified = verify_manifest(self.bundle_dir, verify_checksums=True)
        self.assertTrue(all(verified["integrity_checks"].values()))
        self.assertEqual(
            verified["cohort_counts"]["full_stage_b_balanced_pool"]["by_class"],
            {"Normal": 1548, "Promotional": 1548, "Smishing": 1548},
        )

    def test_batch7_brands_masked_in_bundle(self) -> None:
        b1_b7 = pd.read_parquet(self.bundle_dir / "synthetic_batches_b1_b7.parquet")
        shuvro = b1_b7[b1_b7["text_raw"].str.contains("ShuvroTel", na=False)]
        self.assertGreater(len(shuvro), 0)
        for txt in shuvro["text_brand_masked"]:
            self.assertNotIn("ShuvroTel", txt)
            self.assertIn("<BRAND>", txt)

    def test_checksum_tamper_detected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tpath = Path(td) / "bundle"
            shutil.copytree(self.bundle_dir, tpath)
            target = tpath / "synthetic_batches_b1_b7.parquet"
            target.write_bytes(b"corrupted")
            with self.assertRaises(RuntimeError):
                verify_manifest(tpath, verify_checksums=True)

    def test_experiment_abcd_fold_splits_leakage_free(self) -> None:
        real = pd.read_parquet(self.bundle_dir / "real_train_pool.parquet")
        pilot = pd.read_parquet(self.bundle_dir / "synthetic_pilot_300.parquet")
        batches = pd.read_parquet(self.bundle_dir / "synthetic_batches_b1_b7.parquet")
        probe = pd.read_parquet(self.bundle_dir / "counterfactual_smish_probe.parquet")
        expected_synthetic_ids = {"A": set(), "B": set(), "C": set(batches.row_id), "D": set(pilot.row_id) | set(batches.row_id)}
        seen_probe_ids = {experiment: set() for experiment in EXPERIMENT_SPECS}
        for fold in range(5):
            expected_train_ids = set(real.loc[real["curation_cv_fold"].ne(fold), "row_id"])
            expected_val_ids = real.loc[real["curation_cv_fold"].eq(fold), "row_id"].tolist()
            ordered_a_train_ids = None
            for exp_id in EXPERIMENT_SPECS:
                with self.subTest(fold=fold, experiment=exp_id):
                    train, val, meta = build_experiment_fold_data(self.bundle_dir, exp_id, fold=fold, seed=42)
                    self.assertFalse(val["is_synthetic"].any())
                    self.assertEqual(val["row_id"].tolist(), expected_val_ids)
                    self.assertEqual(set(train.loc[~train["is_synthetic"], "row_id"]), expected_train_ids)
                    self.assertEqual(set(train.loc[train["is_synthetic"], "row_id"]), expected_synthetic_ids[exp_id])
                    self.assertTrue(set(train["family_id"]).isdisjoint(val["family_id"]))
                    self.assertTrue(set(train["row_id"]).isdisjoint(val["row_id"]))
                    if exp_id == "A":
                        ordered_a_train_ids = train["row_id"].tolist()
                    if exp_id == "B":
                        self.assertEqual(meta["balance_mode"], "class_weight")
                        self.assertEqual(train["row_id"].tolist(), ordered_a_train_ids)
                        counts = train["label_idx"].value_counts()
                        self.assertEqual(meta["class_weights"], [len(train) / (3 * counts[i]) for i in range(3)])
                    fold_probe = select_counterfactual_fold(probe, train, val, fold)
                    self.assertTrue(fold_probe["curation_cv_fold"].eq(fold).all())
                    self.assertTrue(seen_probe_ids[exp_id].isdisjoint(fold_probe["row_id"]))
                    seen_probe_ids[exp_id].update(fold_probe["row_id"])
        for ids in seen_probe_ids.values():
            self.assertEqual(ids, set(probe["row_id"]))

    def test_counterfactual_parent_fold_tamper_detected(self) -> None:
        train, val, _ = build_experiment_fold_data(self.bundle_dir, "D", fold=0)
        probe = pd.read_parquet(self.bundle_dir / "counterfactual_smish_probe.parquet")
        probe.loc[probe.index[probe["curation_cv_fold"].eq(1)][0], "curation_cv_fold"] = 0
        with self.assertRaisesRegex(RuntimeError, "overlaps training"):
            select_counterfactual_fold(probe, train, val, fold=0)

    def test_class_weights_from_hand_calculated_training_distribution(self) -> None:
        labels = pd.Series([0, 0, 0, 1, 1, 2])
        self.assertEqual(training_class_weights(labels), [2 / 3, 1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "one training row per class"):
            training_class_weights(pd.Series([0, 1]))
        with self.assertRaisesRegex(ValueError, "Unsupported balance mode"):
            build_experiment_fold_data(self.bundle_dir, "B", fold=0, balance_mode_override="unsupported")

    def test_bundle_physical_metadata_and_original_bundle_preservation(self) -> None:
        original_manifest = (self.bundle_dir / "dataset_manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            assemble_post_stage_b_bundle(ROOT, self.bundle_dir)
        self.assertEqual((self.bundle_dir / "dataset_manifest.json").read_bytes(), original_manifest)
        with tempfile.TemporaryDirectory() as tmp_dir:
            copied = Path(tmp_dir) / "bundle"
            shutil.copytree(self.bundle_dir, copied)
            manifest_path = copied / "dataset_manifest.json"
            tampered = json.loads(original_manifest)
            tampered["bundle_files"]["real_train_pool.parquet"]["n_rows"] += 1
            manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "physical metadata mismatch"):
                verify_manifest(copied)

    def test_recompute_frozen_and_versioned_bundle_integrity(self) -> None:
        frozen = ROOT / "data/processed/post_stage_b_bundle"
        v1 = audit_post_stage_b_bundle(ROOT, frozen, verify_manifest(frozen))
        v2 = audit_post_stage_b_bundle(ROOT, self.bundle_dir, verify_manifest(self.bundle_dir))
        self.assertEqual(v1["cohort_pairs_checked"], 15)
        self.assertEqual(v1["normalization_metadata_mismatches"]["synthetic_batches_b1_b7.parquet"], 79)
        self.assertTrue(all(count == 0 for count in v2["normalization_metadata_mismatches"].values()))
        self.assertEqual(sum(data["n_counterfactual"] for data in v2["folds"].values()), 1279)
        frame = pd.read_parquet(self.bundle_dir / "real_train_pool.parquet")
        self.assertTrue({"text_adv_raw", "text_adv_canon_masked", "text_adv_brand_masked", "text_adv_brand_stripped"}.issubset(frame))

    def test_configurations_keep_full_design_and_smoke_caps(self) -> None:
        configs = ROOT / "experiments/colab/configs"
        full = load_config(configs / "post_stage_b_abcd_config.json")
        smoke = load_config(configs / "gpu_smoke_test_config.json")
        self.assertEqual(full.experiments, ("A", "B", "C", "D"))
        self.assertEqual(full.folds, tuple(range(5)))
        self.assertEqual(len(full.models) * len(full.experiments) * len(full.text_views) * len(full.folds), 320)
        self.assertEqual(full.experiment_b_balance_mode, "class_weight")
        self.assertEqual((full.epochs, full.batch_size, full.eval_batch_size, full.max_length), (3, 32, 64, 128))
        self.assertEqual(full.seed, 42)
        self.assertNotEqual(smoke.run_name, full.run_name)
        self.assertEqual((smoke.smoke_max_train_samples, smoke.smoke_max_val_samples, smoke.smoke_max_eval_samples),
                         (64, 32, 32))

    def test_full_pipeline_refuses_model_view_collisions_before_training(self) -> None:
        from unittest.mock import patch

        config = load_config(ROOT / "experiments/colab/configs/post_stage_b_abcd_config.json")
        with tempfile.TemporaryDirectory() as td, patch("bangla_smish.post_stage_b_runner.hf_train_and_predict_fold") as trainer:
            artifacts = Path(td) / "artifacts"
            with self.assertRaisesRegex(RuntimeError, r"(?i)(overlap|collision|leakage)"):
                run_post_stage_b_pipeline(bundle_dir=self.bundle_dir, artifacts_dir=artifacts, cfg=config)
            trainer.assert_not_called()
            self.assertFalse((artifacts / "run_state.json").exists())

    def test_smoke_config_interrupt_resume_and_5fold_aggregation(self) -> None:
        smoke_cfg = load_config(ROOT / "experiments" / "colab" / "configs" / "gpu_smoke_test_config.json")
        self.assertEqual(smoke_cfg.run_mode, "gpu_smoke_test")

        with tempfile.TemporaryDirectory() as td:
            art_dir = Path(td) / "artifacts"
            # Run 1 unit then simulate interruption
            partial = run_post_stage_b_pipeline(
                bundle_dir=self.bundle_dir,
                artifacts_dir=art_dir,
                cfg=smoke_cfg,
                resume=True,
                verify_checksums=True,
                train_eval_fn=_mock_train_eval_fn,
                stop_after_units=1,
            )
            self.assertEqual(partial["completed_unit_count"], 1)

            # Resume and finish remaining unit
            resumed = run_post_stage_b_pipeline(
                bundle_dir=self.bundle_dir,
                artifacts_dir=art_dir,
                cfg=smoke_cfg,
                resume=True,
                verify_checksums=True,
                train_eval_fn=_mock_train_eval_fn,
            )
            self.assertEqual(resumed["completed_unit_count"], 2)
            rows = {row["experiment_id"]: row for row in resumed["unit_rows"]}
            for exp, expected in {
                "A": (3199, 64, 0, {"Normal": 22, "Promotional": 13, "Smishing": 29}),
                "D": (3844, 56, 8, {"Normal": 22, "Promotional": 22, "Smishing": 20}),
            }.items():
                row = rows[exp]
                self.assertEqual((row["n_train_actual"], row["n_val_actual"], row["n_val_pre_cap"]), (64, 32, 800))
                self.assertEqual((row["n_train_pre_cap"], row["n_train_real"], row["n_train_synthetic"], row["train_by_class"]),
                                 expected)
                self.assertEqual(row["val_by_class"], {"Normal": 9, "Promotional": 7, "Smishing": 16})
                self.assertEqual(row["val_low_fpr_false_positives"], 0)
                self.assertEqual(row["val_low_fpr_n_normal"], 9)
            self.assertTrue((art_dir / "exports" / "aggregated_results.json").exists())
            self.assertTrue((art_dir / "exports" / "aggregated_results.csv").exists())
            self.assertTrue((art_dir / "exports" / "post_stage_b_report.md").exists())
            self.assertTrue((art_dir / "logs" / "runner.log").exists())

            # Full 5-fold A vs D mock aggregation check for OOF + McNemar
            abcd_5fold = PostStageBConfig(
                run_name="test_5fold_oof",
                run_mode="gpu_smoke_test",
                experiments=("A", "D"),
                models=("mock-bert",),
                text_views=("raw",),
                folds=(0, 1, 2, 3, 4),
                run_heldout_eval=False,
                run_adversarial_eval=False,
                run_counterfactual_eval=False,
            )
            art_5fold = Path(td) / "artifacts_5fold"
            full_res = run_post_stage_b_pipeline(
                bundle_dir=self.bundle_dir,
                artifacts_dir=art_5fold,
                cfg=abcd_5fold,
                resume=True,
                verify_checksums=True,
                train_eval_fn=_mock_train_eval_fn,
            )
            self.assertEqual(full_res["completed_unit_count"], 10)
            self.assertIn("exp_A__mock-bert__raw", full_res["oof_5fold_summaries"])
            self.assertIn("exp_D__mock-bert__raw", full_res["oof_5fold_summaries"])
            self.assertIn("D_vs_A__mock-bert__raw", full_res["mcnemar_vs_exp_a"])


if __name__ == "__main__":
    unittest.main()
