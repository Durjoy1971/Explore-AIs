"""Regression tests for the Post-Stage B runner without GPU training."""
from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish import post_stage_b_runner as runner
from bangla_smish.post_stage_b_data import sha256_file


def _mock_trainer(**kwargs):
    validation = kwargs["val_df"]
    probabilities = np.eye(3)[validation["label_idx"].to_numpy(dtype=int)] * 0.97 + 0.01
    return {"predictions": {"val_probs": probabilities}}


class TestRunnerIntegrity(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.bundle = ROOT / "data" / "processed" / "post_stage_b_bundle"
        self.artifacts = self.root / "artifacts"
        self.config = runner.PostStageBConfig(
            run_name="regression",
            run_mode="gpu_smoke_test",
            experiments=("A",),
            models=("mock",),
            text_views=("raw",),
            folds=(0,),
            smoke_max_train_samples=64,
            smoke_max_val_samples=32,
            run_heldout_eval=False,
            run_adversarial_eval=False,
            run_counterfactual_eval=False,
        )
        self.uid = runner.unit_id("A", "mock", "raw", 0)

    def _run(self, **kwargs):
        inputs = {
            "bundle_dir": self.bundle, "artifacts_dir": self.artifacts,
            "cfg": self.config, "train_eval_fn": _mock_trainer,
        }
        return runner.run_post_stage_b_pipeline(**(inputs | kwargs))

    @property
    def state_path(self):
        return self.artifacts / "run_state.json"

    def _state(self):
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _aggregate(self):
        return runner.aggregate_post_stage_b_results(
            self.bundle, self.artifacts, self.config, train_eval_fn=_mock_trainer
        )

    def test_configuration_and_manifest_changes_rejected_without_mutation(self):
        self._run()
        before = self.state_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self._run(cfg=replace(self.config, epochs=2))
        with self.assertRaisesRegex(ValueError, "separate artifact directory"):
            self._run(resume=False)
        bundle_copy = self.root / "bundle-copy"
        shutil.copytree(self.bundle, bundle_copy)
        manifest = bundle_copy / "dataset_manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self._run(bundle_dir=bundle_copy)
        self.assertEqual(self.state_path.read_bytes(), before)

    def test_corrupt_archive_recovery_preserves_attempt(self):
        self._run()
        previous = self.artifacts / "units" / self.uid / "attempts" / "1" / "preds.npz"
        previous.write_bytes(b"interrupted zip")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self._aggregate()
        self.assertEqual(self._run()["completed_unit_count"], 1)
        state = self._state()
        self.assertEqual(state["committed_units"][self.uid]["attempt"], 2)
        self.assertIn("invalidated", state["attempts"][self.uid][0])
        self.assertEqual(previous.read_bytes(), b"interrupted zip")
        self.assertEqual(len(list((self.artifacts / "exports" / "history").iterdir())), 2)

    def test_corrupt_metric_record_ignored_and_retried(self):
        self._run()
        previous = self.artifacts / "units" / self.uid / "attempts" / "1" / "metrics.json"
        record = json.loads(previous.read_text(encoding="utf-8"))
        record["metrics"]["val_overall"]["n_samples"] += 1
        runner.atomic_write_json(previous, record)
        state = self._state()
        state["committed_units"][self.uid]["metrics_sha256"] = sha256_file(previous)
        runner.atomic_write_json(self.state_path, state)
        with self.assertRaisesRegex(ValueError, "Invalid metric record"):
            self._aggregate()
        self.assertEqual(self._run()["completed_unit_count"], 1)
        self.assertEqual(self._state()["committed_units"][self.uid]["attempt"], 2)

    def test_shuffled_rows_rejected_even_with_updated_digest(self):
        self._run()
        predictions = self.artifacts / "units" / self.uid / "attempts" / "1" / "preds.npz"
        with np.load(predictions, allow_pickle=False) as archive:
            shuffled = {key: archive[key][::-1] for key in archive.files}
        np.savez_compressed(predictions, **shuffled)
        state = self._state()
        state["committed_units"][self.uid]["preds_sha256"] = sha256_file(predictions)
        runner.atomic_write_json(self.state_path, state)
        with self.assertRaisesRegex(ValueError, "row IDs"):
            self._aggregate()
        resumed = self._run()
        self.assertEqual(resumed["completed_unit_count"], 1)
        self.assertEqual(self._state()["committed_units"][self.uid]["attempt"], 2)

    def test_attempt_history_and_only_committed_units_aggregate(self):
        calls = []

        def interrupt_once(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise KeyboardInterrupt
            return _mock_trainer(**kwargs)

        with self.assertRaises(KeyboardInterrupt):
            self._run(train_eval_fn=interrupt_once)
        self.assertEqual(self._state()["attempts"][self.uid][0]["status"], "running")
        uncommitted = self.artifacts / "units" / "unknown" / "attempts" / "1" / "metrics.json"
        runner.atomic_write_json(uncommitted, {"unit_id": "unknown"})
        self.assertEqual(self._run(train_eval_fn=interrupt_once)["completed_unit_count"], 1)
        history = self._state()["attempts"][self.uid]
        self.assertEqual([item["status"] for item in history], ["interrupted", "completed"])
        self._run(train_eval_fn=interrupt_once)
        self.assertEqual(len(calls), 2)

    def test_smoke_counts_show_actual_and_pre_cap_classes(self):
        summary = self._run()
        row = summary["unit_rows"][0]
        self.assertEqual((row["n_train_actual"], row["n_val_actual"]), (64, 32))
        self.assertEqual((row["n_train_pre_cap"], row["n_val_pre_cap"]), (3199, 800))
        self.assertEqual(sum(row["train_by_class"].values()), 64)
        self.assertEqual(sum(row["val_by_class"].values()), 32)
        self.assertEqual((row["n_train_real"], row["n_train_synthetic"]), (64, 0))
        report = (self.artifacts / "exports" / "post_stage_b_report.md").read_text(encoding="utf-8")
        self.assertIn("| 3199 | 64 | 800 | 32 | 64 | 0 |", report)

    def test_checked_checkpoint_recovery_skips_incomplete_and_corrupt_saves(self):
        checkpoint_root = self.root / "checkpoints"
        previous_attempt = checkpoint_root / "1"

        def save(step, *, best=None):
            path = previous_attempt / f"checkpoint-{step}"
            path.mkdir(parents=True)
            for file in (
                "config.json", "training_args.bin", "optimizer.pt", "scheduler.pt",
                "rng_state.pth", "model.safetensors",
            ):
                (path / file).write_bytes(b"trainer output")
            runner.atomic_write_json(path / "trainer_state.json", {
                "global_step": step, "best_model_checkpoint": best,
            })
            runner.atomic_write_json(path / runner._CHECKPOINT_COMMIT, runner._checkpoint_file_hashes(path))
            return path

        complete = save(10)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            runner._checkpoint_file_hashes(complete, require_scaler=True)
        damaged = save(20)
        (damaged / "optimizer.pt").write_bytes(b"changed")
        unavailable_best = save(25, best=str(damaged))
        incomplete = previous_attempt / "checkpoint-30"
        incomplete.mkdir()
        (incomplete / "model.safetensors").write_bytes(b"partial")
        self.assertTrue(runner._valid_checkpoint(unavailable_best))
        self.assertEqual(runner._find_latest_trainer_checkpoint(checkpoint_root / "2"), str(complete))
        self.assertTrue(incomplete.exists())
        (complete / "rng_state.pth").unlink()
        self.assertIsNone(runner._find_latest_trainer_checkpoint(checkpoint_root / "2"))

    def test_unsupported_balance_mode_rejected(self):
        for mode in ("downsample", "unsupported"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "class_weight"):
                replace(self.config, experiment_b_balance_mode=mode)


class TestTrainingDispatch(unittest.TestCase):
    def test_t4_auto_uses_fp16_without_bf16_emulation(self):
        def bf16_support(*, including_emulation=True):
            return including_emulation

        cuda = types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=bf16_support)
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(cuda=cuda)}):
            self.assertEqual(runner._resolve_precision("auto"), (True, False))
            with self.assertRaisesRegex(RuntimeError, "native"):
                runner._resolve_precision("bf16")
            cuda.is_bf16_supported = lambda *, including_emulation: not including_emulation
            self.assertEqual(runner._resolve_precision("auto"), (False, True))

    def test_weighted_loss_matches_hand_calculation_only_in_train_mode(self):
        def mock_cross_entropy(logits, labels, weight):
            shifted = logits - logits.max(axis=1, keepdims=True)
            log_prob = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
            sample_weights = weight[labels]
            return -np.sum(log_prob[np.arange(len(labels)), labels] * sample_weights) / sample_weights.sum()

        torch = types.SimpleNamespace(
            as_tensor=lambda weights, **kwargs: np.array(weights, dtype=kwargs["dtype"]),
            nn=types.SimpleNamespace(functional=types.SimpleNamespace(cross_entropy=mock_cross_entropy)),
        )
        base_calls = []

        class BaseTrainer:
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                base_calls.append((model, inputs, kwargs))
                return (0.75, "unweighted") if return_outputs else 0.75

        class Model:
            training = True

            def __call__(self, **kwargs):
                self.inputs = kwargs
                return types.SimpleNamespace(logits=self)

        model = Model()
        trainer = runner._weighted_trainer_class(BaseTrainer, [1.0, 2.0, 3.0])()
        logits = np.array([[0.0, 0.0, 0.0], [math.log(2), 0.0, 0.0]], dtype=np.float32)

        class Logits(np.ndarray):
            device = "cpu"

        model.float = lambda: logits.view(Logits)
        inputs = {"input_ids": [1, 2], "labels": np.array([0, 1], dtype=int)}
        with patch.dict(sys.modules, {"torch": torch}):
            got, outputs = trainer.compute_loss(model, inputs, return_outputs=True, num_items_in_batch=2)
            self.assertAlmostEqual(float(got), (math.log(3) + 2 * math.log(4)) / 3, places=6)
            self.assertIs(outputs.logits, model)
            self.assertEqual(model.inputs, {"input_ids": [1, 2]})
            self.assertIn("labels", inputs)
            self.assertFalse(trainer.model_accepts_loss_kwargs)
            model.training = False
            self.assertEqual(trainer.compute_loss(model, inputs, return_outputs=True), (0.75, "unweighted"))
            self.assertEqual(base_calls, [(model, inputs, {})])

    def test_adversarial_dispatch_uses_corresponding_view(self):
        frame = pd.DataFrame({f"text_adv_{view}": [view] for view in runner.TEXT_VIEW_COLUMNS})
        for view in runner.TEXT_VIEW_COLUMNS:
            with self.subTest(view=view):
                self.assertEqual(frame[runner._adversarial_column(frame, view)].tolist(), [view])
        legacy = pd.DataFrame({"text_adv": ["old raw"]})
        self.assertEqual(runner._adversarial_column(legacy, "raw"), "text_adv")
        with self.assertRaisesRegex(ValueError, "corrected bundle"):
            runner._adversarial_column(legacy, "brand_masked")


if __name__ == "__main__":
    unittest.main()
