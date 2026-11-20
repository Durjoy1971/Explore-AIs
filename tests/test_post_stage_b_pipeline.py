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
    build_experiment_fold_data,
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
        val_ids_by_exp = {}
        for exp_id in EXPERIMENT_SPECS:
            tr_df, va_df, meta = build_experiment_fold_data(self.bundle_dir, exp_id, fold=0, seed=42)
            self.assertFalse(bool(va_df["is_synthetic"].any()))
            self.assertTrue(set(tr_df["family_id"]).isdisjoint(set(va_df["family_id"])))
            val_ids_by_exp[exp_id] = va_df["row_id"].tolist()

            if exp_id == "A":
                self.assertEqual(meta["n_train_synthetic"], 0)
                self.assertEqual(meta["n_train"], 3199)
            elif exp_id == "B":
                self.assertEqual(meta["n_train_synthetic"], 0)
                counts = set(meta["train_by_class"].values())
                self.assertEqual(len(counts), 1)  # exact minority parity per fold
            elif exp_id == "C":
                self.assertEqual(meta["n_train_synthetic"], 345)
                self.assertEqual(meta["n_train"], 3199 + 345)
            elif exp_id == "D":
                self.assertEqual(meta["n_train_synthetic"], 645)
                self.assertEqual(meta["n_train"], 3199 + 645)

        # All 4 experiments evaluate on the exact same real validation fold rows
        for exp_id in ("B", "C", "D"):
            self.assertEqual(val_ids_by_exp["A"], val_ids_by_exp[exp_id])

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
