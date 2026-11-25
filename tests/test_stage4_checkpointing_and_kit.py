"""Stage 4 tests for local-SSD checkpointing, 4-state staging scan, fail-closed recovery, notebook, and kit."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.post_stage_b_checkpointing import (
    CheckpointState,
    RecoveryError,
    commit_unit_completion_event,
    prune_superseded_staging,
    recover_latest_valid_checkpoint,
    register_single_writer_lease,
    release_single_writer_lease,
    scan_unit_staging,
    sync_local_checkpoint_to_staging,
)
from bangla_smish.post_stage_b_integrity import (
    build_protected_baseline_manifest,
    verify_protected_baseline_manifest,
)
from bangla_smish.post_stage_b_runner import PostStageBConfig


def _write_fake_local_ckpt(ckpt_dir: Path, step: int, *, best_ckpt: str | None = None, with_scaler: bool = False) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("config.json", "training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"):
        (ckpt_dir / fname).write_bytes(f"{fname}-step-{step}".encode("utf-8"))
    if with_scaler:
        (ckpt_dir / "scaler.pt").write_bytes(f"scaler-{step}".encode("utf-8"))
    (ckpt_dir / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "best_model_checkpoint": best_ckpt}), encoding="utf-8"
    )
    return ckpt_dir


class TestStage4CheckpointingAndKit(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.local_root = self.root / "local_ssd"
        self.staging_root = self.root / "drive_staging"
        self.events_dir = self.root / "drive_events"
        self.pointers_dir = self.root / "drive_pointers"
        self.uid = "exp_D__xlm-roberta-base__canon_masked__fold_0"
        self.cfg_hash = "cfg_hash_stage4"
        self.man_hash = "man_hash_v2"

    def test_four_states_recovery_with_interrupted_events_and_pointers_and_phase3_pruning(self) -> None:
        # 1. Sync epoch 1 (step 121) and epoch 2 (step 242, referencing step 121 or itself)
        c1 = _write_fake_local_ckpt(self.local_root / "ckpt_src" / "checkpoint-121", 121)
        sync1 = sync_local_checkpoint_to_staging(
            local_ckpt_dir=c1, staging_unit_dir=self.staging_root / self.uid,
            events_dir=self.events_dir, pointers_dir=self.pointers_dir,
            uid=self.uid, attempt=1, config_hash=self.cfg_hash, manifest_sha256=self.man_hash,
        )
        c2 = _write_fake_local_ckpt(self.local_root / "ckpt_src" / "checkpoint-242", 242)
        sync2 = sync_local_checkpoint_to_staging(
            local_ckpt_dir=c2, staging_unit_dir=self.staging_root / self.uid,
            events_dir=self.events_dir, pointers_dir=self.pointers_dir,
            uid=self.uid, attempt=1, config_hash=self.cfg_hash, manifest_sha256=self.man_hash,
        )

        # Simulate interrupted upload (missing manifest), interrupted tail event write, and interrupted pointer write
        incomplete = self.staging_root / self.uid / "attempt_002_deadbeef" / "checkpoint-363"
        incomplete.mkdir(parents=True)
        (incomplete / "model.safetensors").write_bytes(b"partial upload")
        (self.events_dir / "event_999999_CHECKPOINT_SYNCED_broken.json").write_text('{"truncated": ', encoding="utf-8")
        (self.pointers_dir / self.uid / "ptr_999999_broken.json").write_text('{"bad": 1}', encoding="utf-8")

        scan = scan_unit_staging(
            staging_unit_dir=self.staging_root / self.uid,
            events_dir=self.events_dir,
            pointers_dir=self.pointers_dir,
            uid=self.uid,
            expected_config_hash=self.cfg_hash,
            expected_manifest_sha256=self.man_hash,
        )
        self.assertEqual(scan["counts_by_state"][CheckpointState.VALID_RECOVERABLE.value], 1)
        self.assertEqual(scan["counts_by_state"][CheckpointState.ALREADY_SUPERSEDED.value], 1)
        self.assertEqual(scan["counts_by_state"][CheckpointState.INCOMPLETE_UPLOAD.value], 1)
        self.assertEqual(scan["latest_valid_rel_path"], sync2["staging_rel_path"])
        self.assertTrue(len(scan["warnings"]) >= 2)

        # Phase 2 recovers step 242 onto fresh local SSD dir cleanly
        restore_dir = self.local_root / "restored_attempt_2"
        rec = recover_latest_valid_checkpoint(scan_report=scan, local_restore_root=restore_dir)
        self.assertTrue(rec["recovered"])
        self.assertEqual(rec["global_step"], 242)
        self.assertEqual(rec["recovery_granularity"], "epoch_level")
        self.assertTrue((rec["restored_local_ckpt"] / "optimizer.pt").is_file())

        # Phase 3 prunes ONLY ALREADY_SUPERSEDED (step 121), preserving step 242 and incomplete dir for inspection
        pruned = prune_superseded_staging(scan_report=scan, dry_run=False)
        self.assertEqual(pruned["pruned_superseded"], [sync1["staging_rel_path"]])
        self.assertFalse(sync1["staged_ckpt_dir"].exists())
        self.assertTrue(sync2["staged_ckpt_dir"].exists())
        self.assertTrue(incomplete.exists())

        # After UNIT_COMMITTED + keep_model_weights="none", step 242 becomes ALREADY_SUPERSEDED and prunes cleanly
        commit_unit_completion_event(
            events_dir=self.events_dir, uid=self.uid, attempt=1,
            config_hash=self.cfg_hash, manifest_sha256=self.man_hash,
            metrics_sha256="m_sha", preds_sha256="p_sha",
        )
        scan_post = scan_unit_staging(
            staging_unit_dir=self.staging_root / self.uid,
            events_dir=self.events_dir, pointers_dir=self.pointers_dir,
            uid=self.uid, expected_config_hash=self.cfg_hash, expected_manifest_sha256=self.man_hash,
            unit_already_committed=True,
        )
        pruned_post = prune_superseded_staging(scan_report=scan_post, dry_run=False)
        self.assertEqual(pruned_post["pruned_superseded"], [sync2["staging_rel_path"]])
        self.assertFalse(sync2["staged_ckpt_dir"].exists())

    def test_corrupt_manifest_and_stale_config_fail_closed_and_preserve_evidence(self) -> None:
        c1 = _write_fake_local_ckpt(self.local_root / "ckpt_c" / "checkpoint-100", 100)
        sync = sync_local_checkpoint_to_staging(
            local_ckpt_dir=c1, staging_unit_dir=self.staging_root / self.uid,
            events_dir=self.events_dir, pointers_dir=self.pointers_dir,
            uid=self.uid, attempt=1, config_hash=self.cfg_hash, manifest_sha256=self.man_hash,
        )
        # Corrupt optimizer payload
        opt = sync["staged_ckpt_dir"] / "optimizer.pt"
        opt.write_bytes(b"corrupted bytes")

        scan = scan_unit_staging(
            staging_unit_dir=self.staging_root / self.uid,
            events_dir=self.events_dir, pointers_dir=self.pointers_dir,
            uid=self.uid, expected_config_hash=self.cfg_hash, expected_manifest_sha256=self.man_hash,
        )
        self.assertEqual(scan["counts_by_state"][CheckpointState.CORRUPT_OR_UNVERIFIABLE.value], 1)
        with self.assertRaises(RecoveryError):
            recover_latest_valid_checkpoint(scan_report=scan, local_restore_root=self.local_root / "restore_fail")

        # Phase 3 NEVER deletes corrupt evidence
        prune_out = prune_superseded_staging(scan_report=scan, dry_run=False)
        self.assertEqual(prune_out["pruned_superseded"], [])
        self.assertEqual(prune_out["preserved_corrupt_evidence"], [sync["staging_rel_path"]])
        self.assertTrue(opt.exists())

        # Stale config_hash check on clean checkpoint also classifies as CORRUPT_OR_UNVERIFIABLE
        c2 = _write_fake_local_ckpt(self.local_root / "ckpt_c2" / "checkpoint-200", 200)
        sync_local_checkpoint_to_staging(
            local_ckpt_dir=c2, staging_unit_dir=self.staging_root / self.uid,
            events_dir=self.events_dir, pointers_dir=self.pointers_dir,
            uid=self.uid, attempt=2, config_hash="old_cfg", manifest_sha256=self.man_hash,
        )
        scan_stale = scan_unit_staging(
            staging_unit_dir=self.staging_root / self.uid,
            events_dir=self.events_dir, pointers_dir=self.pointers_dir,
            uid=self.uid, expected_config_hash=self.cfg_hash, expected_manifest_sha256=self.man_hash,
        )
        self.assertEqual(scan_stale["counts_by_state"][CheckpointState.CORRUPT_OR_UNVERIFIABLE.value], 2)

    def test_single_writer_lease_policy_enforced(self) -> None:
        art = self.root / "run_artifacts"
        lease = register_single_writer_lease(art, writer_id="w1")
        with self.assertRaisesRegex(RuntimeError, "Single-writer policy"):
            register_single_writer_lease(art, writer_id="w2")
        release_single_writer_lease(art, lease["writer_id"])

    def test_protected_baseline_282_and_portable_scientific_check(self) -> None:
        manifest_path = ROOT / "reports" / "protected_baseline_282.json"
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["total_protected_files"], 282)
        self.assertTrue(manifest["non_self_referential"])
        self.assertEqual(manifest["owner_acceptance_status"], "ACCEPTED_BY_OWNER_STAGE4")
        # 64 current-state snapshots must explicitly have historical_proof=False
        snaps = [e for e in manifest["entries"] if e["provenance_tier"] == "current_state_snapshot_accepted_not_historical_proof"]
        self.assertEqual(len(snaps), 64)
        self.assertTrue(all(e["historical_proof"] is False for e in snaps))

        res_full = verify_protected_baseline_manifest(ROOT, manifest_path, check_portable_scientific_only=False)
        res_port = verify_protected_baseline_manifest(ROOT, manifest_path, check_portable_scientific_only=True)
        self.assertEqual(res_full["verified_count"], 282)
        self.assertEqual(res_port["verified_count"], 240)
        self.assertEqual(res_port["skipped_machine_metadata"], 42)
        live = build_protected_baseline_manifest(ROOT, owner_acceptance_status="ACCEPTED_BY_OWNER_STAGE4")
        self.assertEqual(live["provenance_tier_counts"], manifest["provenance_tier_counts"])

    def test_stage4_notebook_generator_parity_and_safety_gates(self) -> None:
        nb_path = ROOT / "experiments" / "colab" / "post_stage_b_stage4_experiments.ipynb"
        nb = nbformat.read(nb_path, as_version=4)
        nbformat.validate(nb)
        gen_path = ROOT / "scripts" / "generate_post_stage_b_stage4_notebook.py"
        spec = importlib.util.spec_from_file_location("generate_post_stage_b_stage4_notebook", gen_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.build_stage4_notebook(), nb)

        ns = {}
        exec(nb.cells[1].source, ns)
        self.assertEqual(ns["RUN_MODE"], "gpu_smoke_test")
        self.assertFalse(ns["AUTHORIZE_T4_BENCHMARK"])
        self.assertFalse(ns["AUTHORIZE_FULL_TRAINING"])
        ns["RUN_MODE"] = "t4_benchmark"
        with self.assertRaisesRegex(RuntimeError, "5-unit T4 benchmark"):
            ns["validate_stage4_launch_settings"]()
        ns["RUN_MODE"] = "experiments_abcd"
        with self.assertRaisesRegex(RuntimeError, "NO-GO"):
            ns["validate_stage4_launch_settings"]()

    def test_stage4_colab_kit_export_and_verification(self) -> None:
        kit_script = ROOT / "scripts" / "prepare_stage4_colab_kit.py"
        spec = importlib.util.spec_from_file_location("prepare_stage4_colab_kit", kit_script)
        kit_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kit_mod)

        staged = self.root / "staged_stage4_kit"
        manifest = kit_mod.export_stage4_colab_kit(
            ROOT, ROOT / "data" / "processed" / "post_stage_b_bundle_v2", staged
        )
        self.assertEqual(manifest["kit_version"], "post_stage_b_stage4_kit_v2")
        verified = kit_mod.verify_stage4_colab_kit(staged)
        self.assertEqual(verified["stage4_notebook_sha256"], manifest["stage4_notebook_sha256"])

    def test_runner_records_retraining_metadata_and_step_budget_in_mock_run(self) -> None:
        from bangla_smish.post_stage_b_runner import run_post_stage_b_pipeline

        cfg = PostStageBConfig(
            run_name="stage4_mock_retrain",
            run_mode="gpu_smoke_test",
            experiments=("A", "D"),
            models=("csebuetnlp/banglabert",),
            text_views=("canon_masked",),
            folds=(0,),
            smoke_max_train_samples=48,
            smoke_max_val_samples=24,
            smoke_max_eval_samples=24,
            keep_model_weights="best_fold0",
            run_heldout_eval=False,
            run_adversarial_eval=False,
            run_counterfactual_eval=False,
        )

        def mock_fn(**kwargs):
            import numpy as np

            val_df = kwargs["val_df"]
            probs = np.eye(3)[val_df["label_idx"].to_numpy(dtype=int)] * 0.94 + 0.02
            return {
                "train_seconds": 0.1,
                "actual_global_step": 6,
                "model_revision": "mock_rev_1",
                "tokenizer_revision": "mock_tok_1",
                "predictions": {"val_probs": probs},
            }

        art = self.root / "mock_artifacts"
        summary = run_post_stage_b_pipeline(
            bundle_dir=ROOT / "data" / "processed" / "post_stage_b_bundle_v2",
            artifacts_dir=art,
            cfg=cfg,
            train_eval_fn=mock_fn,
        )
        self.assertEqual(summary["completed_unit_count"], 2)
        self.assertEqual(summary["weight_retention_policy"]["keep_model_weights"], "best_fold0")
        registry = summary["weight_retention_policy"]["retraining_registry"]
        self.assertEqual(len(registry), 2)
        self.assertEqual(registry[0]["model_revision"], "mock_rev_1")
        self.assertEqual(registry[0]["tokenizer_revision"], "mock_tok_1")
        self.assertEqual(summary["unit_rows"][0]["expected_optimizer_steps"], 6)


if __name__ == "__main__":
    unittest.main()
