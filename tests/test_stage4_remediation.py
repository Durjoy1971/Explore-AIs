"""Comprehensive Stage 4 remediation regression tests covering Blockers 4.1-4.4,
Governance Gaps A/B/C, Report Discrepancies, and T4 Benchmark Fixture Validity.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.post_stage_b_checkpointing import (
    CheckpointState,
    RecoveryError,
    commit_unit_completion_event,
    inspect_writer_leases,
    prune_superseded_staging,
    recover_latest_valid_checkpoint,
    register_single_writer_lease,
    release_single_writer_lease,
    scan_unit_staging,
    sync_local_checkpoint_to_staging,
    verify_writer_lease_still_sole_active,
)
from bangla_smish.post_stage_b_colab_deps import (
    BANGLABERT_NORMALIZER_COMMIT,
    preflight_colab_dependencies,
)
from bangla_smish.post_stage_b_data import load_heldout_eval_sets, sha256_file
from bangla_smish.post_stage_b_ensemble import build_ensemble_evaluation_summary
from bangla_smish.post_stage_b_integrity import verify_portable_colab_kit
from bangla_smish.post_stage_b_runner import (
    PostStageBConfig,
    _build_t4_benchmark_workload_fixture,
    aggregate_post_stage_b_results,
    run_post_stage_b_pipeline,
)
from bangla_smish.post_stage_b_stats import (
    CONFIRMATORY_DECLARATIONS_ENABLED,
    STATISTICAL_VALIDATION_STATUS,
    compare_models_cluster_bootstrap,
    evaluate_stage4_hypothesis_families,
    holm_reject_family,
    simulate_holm_reject_family,
)


def _write_ckpt(ckpt_dir: Path, step: int, *, best_ckpt: str | None = None, with_scaler: bool = False) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("config.json", "training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"):
        (ckpt_dir / fname).write_bytes(f"{fname}-step-{step}".encode("utf-8"))
    if with_scaler:
        (ckpt_dir / "scaler.pt").write_bytes(f"scaler-{step}".encode("utf-8"))
    (ckpt_dir / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "best_model_checkpoint": best_ckpt}), encoding="utf-8"
    )
    return ckpt_dir


class _FakeDist:
    def __init__(self, version: str, direct_url: dict | None = None) -> None:
        self.version = version
        self._direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        if filename == "direct_url.json" and self._direct_url is not None:
            return json.dumps(self._direct_url)
        return None


class TestStage4Remediation(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    # ------------------------------------------------------------------
    # Blocker 4.1: Isolated Portable Colab Kit Integrity
    # ------------------------------------------------------------------
    def test_isolated_portable_kit_verification_and_tamper_rejection(self) -> None:
        kit_script = ROOT / "scripts" / "prepare_stage4_colab_kit.py"
        spec = importlib.util.spec_from_file_location("prepare_stage4_colab_kit", kit_script)
        kit_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kit_mod)

        staged = self.root / "portable_kit"
        exported = kit_mod.export_stage4_colab_kit(
            ROOT, ROOT / "data" / "processed" / "post_stage_b_bundle_v2", staged
        )
        self.assertEqual(exported["kit_version"], "post_stage_b_stage4_kit_v2")
        self.assertFalse(exported["verifies_full_repository_282"])
        self.assertIn("stage4_statistical_validation_results.json", "\n".join(exported["required_runtime_files"]))

        # Verify in a completely isolated child Python process outside repository root (cwd=self.root, PYTHONPATH=staged/src)
        code = (
            "import json, sys; from pathlib import Path; "
            "from bangla_smish.post_stage_b_integrity import verify_portable_colab_kit; "
            f"res = verify_portable_colab_kit(Path({str(staged)!r})); "
            "assert res['verifies_full_repository_282'] is False; "
            "assert res['cross_checked_baseline_v2_files'] == 8; "
            "print(json.dumps(res))"
        )
        env = {**os.environ, "PYTHONPATH": str(staged / "src")}
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(self.root),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        res = json.loads(proc.stdout)
        self.assertEqual(res["statistical_validation_verdict"], STATISTICAL_VALIDATION_STATUS)

        # Reject corrupted required file
        tampered = self.root / "kit_corrupt"
        shutil.copytree(staged, tampered)
        (tampered / "pyproject.toml").write_text("tampered = true\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "CORRUPTED: pyproject.toml"):
            verify_portable_colab_kit(tampered)

        # Reject missing required file
        missing_kit = self.root / "kit_missing"
        shutil.copytree(staged, missing_kit)
        (missing_kit / "reports" / "stage4_statistical_validation_results.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "MISSING: reports/stage4_statistical_validation_results.json"):
            verify_portable_colab_kit(missing_kit)

        # Reject unexpected extra file
        extra_kit = self.root / "kit_extra"
        shutil.copytree(staged, extra_kit)
        (extra_kit / "unexpected_rogue_file.py").write_text("x = 1\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "UNEXPECTED: unexpected_rogue_file.py"):
            verify_portable_colab_kit(extra_kit)

        # Reject incompatible flag (e.g., confirmatory_mode flipped to False on Option A config -> CORRUPTED / flag check)
        bad_cfg_kit = self.root / "kit_bad_cfg"
        shutil.copytree(staged, bad_cfg_kit)
        cfg_p = bad_cfg_kit / "experiments" / "colab" / "configs" / "post_stage_b_stage4_t4_benchmark_config.json"
        cfg_p.write_text(cfg_p.read_text(encoding="utf-8"), encoding="utf-8")
        verify_portable_colab_kit(bad_cfg_kit)  # identical OK
        mod_cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        mod_cfg["run_heldout_eval"] = True
        cfg_p.write_text(json.dumps(mod_cfg), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "CORRUPTED"):
            verify_portable_colab_kit(bad_cfg_kit)

    # ------------------------------------------------------------------
    # Blocker 4.2: Colab Dependency Preflight & Notebook Generator Sync
    # ------------------------------------------------------------------
    def test_colab_dependency_preflight_mocked_and_notebook_cells(self) -> None:
        good_dists = {
            "transformers": _FakeDist("4.57.1"),
            "tokenizers": _FakeDist("0.22.1"),
            "datasets": _FakeDist("3.2.0"),
            "accelerate": _FakeDist("1.2.1"),
            "safetensors": _FakeDist("0.4.5"),
            "sentencepiece": _FakeDist("0.2.0"),
            "packaging": _FakeDist("26.3"),
            "huggingface_hub": _FakeDist("0.34.4"),
            "protobuf": _FakeDist("5.29.0"),
            "regex": _FakeDist("2024.11.6"),
            "pyyaml": _FakeDist("6.0.2"),
            "requests": _FakeDist("2.32.3"),
            "tqdm": _FakeDist("4.67.1"),
            "filelock": _FakeDist("3.16.1"),
            "numpy": _FakeDist("2.1.0"),
            "pandas": _FakeDist("2.2.3"),
            "scipy": _FakeDist("1.14.1"),
            "scikit-learn": _FakeDist("1.5.2"),
            "pyarrow": _FakeDist("18.1.0"),
            "ftfy": _FakeDist("6.0.3"),
            "emoji": _FakeDist("1.4.2"),
            "wcwidth": _FakeDist("0.2.13"),
            "normalizer": _FakeDist(
                "0.0.1",
                {
                    "url": "https://github.com/csebuetnlp/normalizer",
                    "vcs_info": {"commit_id": BANGLABERT_NORMALIZER_COMMIT},
                },
            ),
        }

        def dist_lookup(name: str):
            if name not in good_dists:
                raise PackageNotFoundError(name)
            return good_dists[name]

        def version_lookup(name: str) -> str:
            if name == "torch":
                return "2.6.0+cu124"
            return dist_lookup(name).version

        def import_ok(name: str):
            if name == "torch":
                return types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
            if name == "normalizer":
                return types.SimpleNamespace(__name__=name, normalize=lambda s: s)
            return types.SimpleNamespace(__name__=name)

        res = preflight_colab_dependencies(
            ["csebuetnlp/banglabert", "xlm-roberta-base"],
            install_missing=True,
            require_cuda=True,
            dist_lookup=dist_lookup,
            version_lookup=version_lookup,
            import_module_fn=import_ok,
            run_pip_fn=lambda cmd: self.fail("pip must not run when deps already satisfied"),
        )
        self.assertEqual(res["status"], "verified")
        self.assertIn("normalizer", res["verified_imports"])

        # Incompatible transformers (e.g. 5.0.0) triggers constrained pip install preserving torch + restart instruction
        bad_dists = {**good_dists, "transformers": _FakeDist("5.0.0")}
        pip_calls: list[list[str]] = []

        def fake_pip(cmd: list[str]) -> None:
            pip_calls.append(cmd)
            c_idx = cmd.index("-c") + 1
            c_text = Path(cmd[c_idx]).read_text(encoding="utf-8")
            self.assertIn("torch==2.6.0+cu124", c_text)

        with self.assertRaisesRegex(RuntimeError, "RESTART REQUIRED"):
            preflight_colab_dependencies(
                ["csebuetnlp/banglabert"],
                install_missing=True,
                dist_lookup=lambda n: bad_dists[n],
                version_lookup=version_lookup,
                import_module_fn=import_ok,
                run_pip_fn=fake_pip,
            )
        self.assertEqual(len(pip_calls), 1)
        self.assertIn("transformers>=4.57.1,<5", pip_calls[0])

        # Missing torch or old torch < 2.6 fails closed
        with self.assertRaisesRegex(RuntimeError, "PyTorch >= 2.6"):
            preflight_colab_dependencies(
                ["xlm-roberta-base"],
                dist_lookup=dist_lookup,
                version_lookup=lambda n: "2.5.1+cu121" if n == "torch" else "1.0",
                import_module_fn=import_ok,
            )

        # Notebook Cell 4 & Cell 5 alignment
        nb_path = ROOT / "experiments" / "colab" / "post_stage_b_stage4_experiments.ipynb"
        nb = nbformat.read(nb_path, as_version=4)
        self.assertIn("preflight_colab_dependencies", nb.cells[4].source)
        self.assertIn("verify_portable_colab_kit", nb.cells[5].source)
        self.assertNotIn("verify_protected_baseline_manifest", nb.cells[5].source)

    # ------------------------------------------------------------------
    # Blocker 4.3: Stale Writer Lease Recovery & Zero fcntl.flock
    # ------------------------------------------------------------------
    def test_no_fcntl_and_all_seven_lease_recovery_scenarios(self) -> None:
        runner_src = (ROOT / "src" / "bangla_smish" / "post_stage_b_runner.py").read_text(encoding="utf-8")
        self.assertNotIn("fcntl", runner_src)
        self.assertNotIn("def _run_lock(", runner_src)

        art = self.root / "lease_art"
        # 1. Normal lease acquire & release
        l1 = register_single_writer_lease(art, writer_id="w_normal", run_name="r1", config_hash="h1")
        release_single_writer_lease(art, l1["writer_id"])
        self.assertEqual(inspect_writer_leases(art)["active_leases"], [])

        # 2. Abrupt SIGKILL crash leaves stale lease; 3. Recovery without takeover fails closed
        crash_script = (
            "import sys, time; from pathlib import Path; "
            f"sys.path.insert(0, {str(ROOT / 'src')!r}); "
            "from bangla_smish.post_stage_b_checkpointing import register_single_writer_lease; "
            f"register_single_writer_lease(Path({str(art)!r}), writer_id='w_crashed'); "
            "print('ACQUIRED', flush=True); time.sleep(30)"
        )
        proc = subprocess.Popen([sys.executable, "-c", crash_script], stdout=subprocess.PIPE, text=True)
        try:
            assert proc.stdout is not None
            self.assertEqual(proc.stdout.readline().strip(), "ACQUIRED")
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.poll() is None:
                proc.kill()
        with self.assertRaisesRegex(RuntimeError, "Lease age alone never proves termination"):
            register_single_writer_lease(art, writer_id="w_resume_fail")

        # 4. Authorized stale takeover records immutable takeover_ record & preserves old lease record
        l2 = register_single_writer_lease(
            art,
            writer_id="w_takeover",
            allow_stale_takeover=True,
            expected_stale_writer_ids=["w_crashed"],
            takeover_reason="Owner confirmed Colab runtime killed",
        )
        self.assertEqual(l2["takeover_record"]["superseded_writer_ids"], ["w_crashed"])
        lease_files = list((art / "leases").glob("lease_*.json"))
        takeover_files = list((art / "leases").glob("takeover_*.json"))
        self.assertEqual(len(lease_files), 3)
        self.assertEqual(len(takeover_files), 1)
        release_single_writer_lease(art, "w_takeover")

        # 5. Conflicting active writers detected and guarded
        art2 = self.root / "conflict_art"
        register_single_writer_lease(art2, writer_id="w_a")
        leases_dir = art2 / "leases"
        # Simulate another concurrent FUSE writer writing a second active lease
        l_b = register_single_writer_lease(
            art2,
            writer_id="w_b",
            allow_stale_takeover=True,
            expected_stale_writer_ids=["w_a"],
            takeover_reason="sim_conflict",
        )
        # Put another active lease alongside w_b directly (bypassing guard to simulate race on FUSE)
        from bangla_smish.post_stage_b_checkpointing import _write_new_immutable_json
        _write_new_immutable_json(
            leases_dir / "lease_99999999T000000Z_deadbeef.json",
            {"record_type": "lease_acquired", "writer_id": "w_race", "status": "active", "pid": 99999},
            embed_digest_key="lease_sha256",
        )
        insp = inspect_writer_leases(art2)
        self.assertTrue(insp["conflicting_active_writers"])
        with self.assertRaisesRegex(RuntimeError, "Conflicting active writer"):
            verify_writer_lease_still_sole_active(art2, l_b["writer_id"])

        # 6. Interrupted/corrupt release record fails closed until acknowledged takeover
        art3 = self.root / "corrupt_lease_art"
        l3 = register_single_writer_lease(art3, writer_id="w_clean")
        release_single_writer_lease(art3, l3["writer_id"])
        (art3 / "leases" / "release_9999_broken.json").write_text('{"truncated": ', encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Interrupted or corrupt lease/release"):
            register_single_writer_lease(art3, writer_id="w_next")
        l4 = register_single_writer_lease(
            art3,
            writer_id="w_next",
            allow_stale_takeover=True,
            expected_corrupt_files=["release_9999_broken.json"],
            takeover_reason="Owner verified broken release tail",
        )
        self.assertIn("release_9999_broken.json", l4["takeover_record"]["acknowledged_corrupt_files"])

        # 7. Recovery preserves valid completed units after simulated abrupt stale lease
        bdir = ROOT / "data" / "processed" / "post_stage_b_bundle_v2"
        run_art = self.root / "runner_resume_art"
        cfg = PostStageBConfig(
            run_name="lease_resume_test",
            run_mode="gpu_smoke_test",
            experiments=("A", "D"),
            models=("csebuetnlp/banglabert",),
            text_views=("canon_masked",),
            folds=(0,),
            smoke_max_train_samples=32,
            smoke_max_val_samples=16,
            smoke_max_eval_samples=16,
            run_heldout_eval=False,
            run_adversarial_eval=False,
            run_counterfactual_eval=False,
        )
        calls = {"n": 0}

        def mock_train(**kwargs):
            calls["n"] += 1
            probs = np.eye(3)[kwargs["val_df"]["label_idx"].to_numpy(dtype=int)] * 0.94 + 0.02
            return {"actual_global_step": 1, "train_seconds": 0.05, "predictions": {"val_probs": probs}}

        part = run_post_stage_b_pipeline(
            bundle_dir=bdir, artifacts_dir=run_art, cfg=cfg, train_eval_fn=mock_train, stop_after_units=1
        )
        self.assertEqual(part["completed_unit_count"], 1)
        # Inject an unreleased stale lease simulating abrupt preemption before unit 2
        register_single_writer_lease(run_art, writer_id="crashed_colab_worker")
        with self.assertRaisesRegex(RuntimeError, "Single-writer policy violation"):
            run_post_stage_b_pipeline(bundle_dir=bdir, artifacts_dir=run_art, cfg=cfg, train_eval_fn=mock_train)

        resumed = run_post_stage_b_pipeline(
            bundle_dir=bdir,
            artifacts_dir=run_art,
            cfg=cfg,
            train_eval_fn=mock_train,
            allow_stale_takeover=True,
            expected_stale_writer_ids=["crashed_colab_worker"],
            takeover_reason="Resuming unit 2 after VM reset",
        )
        self.assertEqual(resumed["completed_unit_count"], 2)
        # Unit 1 was preserved and NOT retrained!
        self.assertEqual(calls["n"], 2)

    # ------------------------------------------------------------------
    # Blocker 4.4: Cross-Attempt Best Checkpoint Relocation & Retention
    # ------------------------------------------------------------------
    def test_cross_attempt_best_ckpt_relocation_retention_and_failure_modes(self) -> None:
        staging_unit = self.root / "staging" / "uid1"
        events = self.root / "events"
        pointers = self.root / "pointers"
        uid, cfg_h, man_h = "uid1", "cfg_1", "man_1"

        # 1. Epoch 1 (best=121), Epoch 2 (latest=242 still pointing to old attempt1/checkpoint-121), Epoch 3 (latest=363)
        old_attempt1 = self.root / "old_colab_vm" / "checkpoint-121"
        c1 = _write_ckpt(old_attempt1, 121, best_ckpt=str(old_attempt1), with_scaler=True)
        s1 = sync_local_checkpoint_to_staging(
            local_ckpt_dir=c1, staging_unit_dir=staging_unit, events_dir=events, pointers_dir=pointers,
            uid=uid, attempt=1, config_hash=cfg_h, manifest_sha256=man_h, require_scaler=True,
        )
        c2 = _write_ckpt(self.root / "old_colab_vm" / "checkpoint-242", 242, best_ckpt=str(old_attempt1), with_scaler=True)
        s2 = sync_local_checkpoint_to_staging(
            local_ckpt_dir=c2, staging_unit_dir=staging_unit, events_dir=events, pointers_dir=pointers,
            uid=uid, attempt=1, config_hash=cfg_h, manifest_sha256=man_h, require_scaler=True,
        )
        c3 = _write_ckpt(self.root / "old_colab_vm" / "checkpoint-363", 363, best_ckpt=str(old_attempt1), with_scaler=True)
        s3 = sync_local_checkpoint_to_staging(
            local_ckpt_dir=c3, staging_unit_dir=staging_unit, events_dir=events, pointers_dir=pointers,
            uid=uid, attempt=1, config_hash=cfg_h, manifest_sha256=man_h, require_scaler=True,
        )

        # Remove old attempt 1 local SSD dir so obsolete path is truly gone
        shutil.rmtree(self.root / "old_colab_vm")

        # Retention limit=1 preserves BOTH latest (363) AND distinct pinned best (121), marking middle (242) superseded
        scan = scan_unit_staging(
            staging_unit_dir=staging_unit, events_dir=events, pointers_dir=pointers,
            uid=uid, expected_config_hash=cfg_h, expected_manifest_sha256=man_h, checkpoint_retention_limit=1,
        )
        self.assertEqual(scan["counts_by_state"][CheckpointState.VALID_RECOVERABLE.value], 2)
        self.assertEqual(scan["counts_by_state"][CheckpointState.ALREADY_SUPERSEDED.value], 1)
        self.assertFalse(scan["missing_required_best_checkpoint"])

        # Restore into fresh Attempt 2 local SSD dir; verify relocation & Drive immutability
        attempt2_local = self.root / "attempt2_ssd"
        rec = recover_latest_valid_checkpoint(scan_report=scan, local_restore_root=attempt2_local)
        self.assertTrue(rec["recovered"])
        self.assertTrue(rec["best_model_checkpoint_relocated"])
        self.assertEqual(rec["restored_local_ckpt"].name, "checkpoint-363")
        self.assertEqual(rec["restored_best_local_ckpt"].name, "checkpoint-121")
        for req_f in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt", "training_args.bin", "config.json"):
            self.assertTrue((rec["restored_local_ckpt"] / req_f).is_file())
        relocated_ts = json.loads((rec["restored_local_ckpt"] / "trainer_state.json").read_text(encoding="utf-8"))
        self.assertEqual(relocated_ts["best_model_checkpoint"], str(rec["restored_best_local_ckpt"].resolve()))
        # Drive source trainer_state.json must remain untouched (immutable)!
        drive_ts = json.loads((s3["staged_ckpt_dir"] / "trainer_state.json").read_text(encoding="utf-8"))
        self.assertIn("old_colab_vm", drive_ts["best_model_checkpoint"])

        # Missing/corrupt required distinct best checkpoint must fail closed
        (s1["staged_ckpt_dir"] / "model.safetensors").unlink()
        scan_missing_best = scan_unit_staging(
            staging_unit_dir=staging_unit, events_dir=events, pointers_dir=pointers,
            uid=uid, expected_config_hash=cfg_h, expected_manifest_sha256=man_h, checkpoint_retention_limit=1,
        )
        self.assertTrue(scan_missing_best["missing_required_best_checkpoint"])
        with self.assertRaisesRegex(RecoveryError, "best_model_checkpoint"):
            recover_latest_valid_checkpoint(scan_report=scan_missing_best, local_restore_root=self.root / "fail_missing_best")

        # Partial pruning OSError captured without crashing or losing valid evidence
        scan2 = scan_unit_staging(
            staging_unit_dir=staging_unit, events_dir=events, pointers_dir=pointers,
            uid=uid, expected_config_hash=cfg_h, expected_manifest_sha256=man_h, checkpoint_retention_limit=1,
        )
        with patch("bangla_smish.post_stage_b_checkpointing.shutil.rmtree", side_effect=OSError("Drive FUSE busy")):
            prune_err = prune_superseded_staging(scan_report=scan2, dry_run=False)
        self.assertEqual(len(prune_err["prune_errors"]), 1)
        self.assertEqual(prune_err["pruned_superseded"], [])
        self.assertTrue(s2["staged_ckpt_dir"].exists())
        self.assertEqual(len(prune_err["preserved_corrupt_evidence"]), 1)

        # Sole valid candidate (363) is NEVER pruned even with retention_limit=0
        scan_zero = scan_unit_staging(
            staging_unit_dir=staging_unit, events_dir=events, pointers_dir=pointers,
            uid=uid, expected_config_hash=cfg_h, expected_manifest_sha256=man_h, checkpoint_retention_limit=0,
        )
        self.assertEqual(scan_zero["counts_by_state"][CheckpointState.VALID_RECOVERABLE.value], 1)

        # After UNIT_COMMITTED, all remaining valid checkpoints become superseded and prune cleanly (preserving corrupt evidence)
        commit_unit_completion_event(
            events_dir=events, uid=uid, attempt=1, config_hash=cfg_h, manifest_sha256=man_h,
            metrics_sha256="ms", preds_sha256="ps",
        )
        scan_committed = scan_unit_staging(
            staging_unit_dir=staging_unit, events_dir=events, pointers_dir=pointers,
            uid=uid, expected_config_hash=cfg_h, expected_manifest_sha256=man_h, unit_already_committed=True,
        )
        pruned_final = prune_superseded_staging(scan_report=scan_committed, dry_run=False)
        self.assertFalse(s3["staged_ckpt_dir"].exists())
        self.assertEqual(len(pruned_final["preserved_corrupt_evidence"]), 1)

    # ------------------------------------------------------------------
    # Statistical Governance Bypass Guards, Standalone Retraining Metadata & Benchmark Fixture
    # ------------------------------------------------------------------
    def test_governance_bypass_attempts_exports_and_fixture(self) -> None:
        self.assertFalse(CONFIRMATORY_DECLARATIONS_ENABLED)
        self.assertEqual(STATISTICAL_VALIDATION_STATUS, "FAIL_OR_INCONCLUSIVE_CONFIRMATORY_DISABLED")

        # Attempt A: compare_models_cluster_bootstrap
        y = np.array([0, 1, 2] * 30)
        fams = np.repeat(np.arange(45), 2)
        comp = compare_models_cluster_bootstrap(
            y, y, y == 2, y, y == 2, fams, B_ci=50, B_pval=100, method="fixed_budget"
        )
        self.assertTrue(comp["method_confirmatory_eligible"])
        self.assertFalse(comp["confirmatory_allowed"])
        self.assertEqual(comp["statistical_validation_status"], STATISTICAL_VALIDATION_STATUS)
        self.assertEqual(comp["p_value_status"], "UNVALIDATED_DIAGNOSTIC_ONLY")

        # Attempt B: holm_reject_family default AND explicit confirmatory_allowed=True both stay fail-closed
        h_def = holm_reject_family({"h1": 1e-6})
        h_bypass = holm_reject_family({"h1": 1e-6}, confirmatory_allowed=True)
        for d in (h_def, h_bypass):
            self.assertFalse(d["h1"]["rejected_at_alpha"])
            self.assertFalse(d["h1"]["confirmatory_allowed"])
            self.assertEqual(d["h1"]["p_value_status"], "UNVALIDATED_DIAGNOSTIC_ONLY")
        self.assertTrue(h_bypass["h1"]["governance_bypass_blocked"])

        # Simulation helper stays explicitly separated
        sim_h = simulate_holm_reject_family({"h1": 1e-6})
        self.assertTrue(sim_h["h1"]["simulation_only"])
        self.assertFalse(sim_h["h1"]["production_confirmatory_claim"])

        # Attempt C: build_ensemble_evaluation_summary bypass attempt + standalone exports
        bdir = ROOT / "data" / "processed" / "post_stage_b_bundle_v2"
        eval_sets = load_heldout_eval_sets(bdir)
        val_f = eval_sets["smishdetect_validation"]
        test_f = eval_sets["smishdetect_test"]
        rng = np.random.default_rng(1)
        groups = {}
        for exp_id in ("A", "B"):
            groups[(exp_id, "csebuetnlp/banglabert", "canon_masked")] = {
                fold: {
                    "experiment_id": exp_id,
                    "model_name": "csebuetnlp/banglabert",
                    "text_view": "canon_masked",
                    "model_revision": "r1",
                    "tokenizer_revision": "t1",
                    "config_hash": "c1",
                    "dataset_manifest_sha256": "m1",
                    "arrays": {
                        "smishdetect_validation_row_ids": val_f["row_id"].to_numpy(dtype=str),
                        "smishdetect_validation_y": val_f["label_idx"].to_numpy(dtype=int),
                        "smishdetect_validation_probs": np.eye(3)[val_f["label_idx"].to_numpy(dtype=int)] * 0.9 + 0.03 + rng.uniform(0, 0.01, (len(val_f), 3)),
                        "smishdetect_test_row_ids": test_f["row_id"].to_numpy(dtype=str),
                        "smishdetect_test_y": test_f["label_idx"].to_numpy(dtype=int),
                        "smishdetect_test_probs": np.eye(3)[test_f["label_idx"].to_numpy(dtype=int)] * 0.9 + 0.03 + rng.uniform(0, 0.01, (len(test_f), 3)),
                    },
                }
                for fold in range(5)
            }
        # Normalize rows
        for fmap in groups.values():
            for fd in fmap.values():
                for k in ("smishdetect_validation_probs", "smishdetect_test_probs"):
                    fd["arrays"][k] /= fd["arrays"][k].sum(axis=1, keepdims=True)
        ens_sum = build_ensemble_evaluation_summary(
            ensemble_groups=groups,
            eval_frames={"smishdetect_validation": val_f, "smishdetect_test": test_f},
            n_boot_ci=40,
            n_boot_pval=80,
            seed=42,
            confirmatory_declarations_enabled=True,  # Attempted bypass!
        )
        self.assertFalse(ens_sum["confirmatory_declarations_enabled"])
        self.assertEqual(ens_sum["statistical_validation_status"], STATISTICAL_VALIDATION_STATUS)
        self.assertTrue(ens_sum["hypothesis_testing"]["governance_bypass_blocked"])
        for eff in ens_sum["hypothesis_testing"]["descriptive_effects"].values():
            self.assertFalse(eff["confirmatory_allowed"])

        # End-to-end mocked runner exports: CSV, JSON, MD, and standalone exports/retraining_metadata.json
        art = self.root / "exports_check"
        cfg = PostStageBConfig(
            run_name="remed_exports",
            run_mode="gpu_smoke_test",
            experiments=("A", "D"),
            models=("csebuetnlp/banglabert",),
            text_views=("canon_masked",),
            folds=(0,),
            smoke_max_train_samples=32,
            smoke_max_val_samples=16,
            smoke_max_eval_samples=16,
            run_heldout_eval=False,
            run_adversarial_eval=False,
            run_counterfactual_eval=False,
        )
        summary = run_post_stage_b_pipeline(
            bundle_dir=bdir,
            artifacts_dir=art,
            cfg=cfg,
            train_eval_fn=lambda **kw: {
                "actual_global_step": 3,
                "train_seconds": 0.05,
                "model_revision": "revA",
                "tokenizer_revision": "tokA",
                "predictions": {"val_probs": np.eye(3)[kw["val_df"]["label_idx"].to_numpy(dtype=int)] * 0.94 + 0.02},
            },
        )
        exports_dir = art / "exports"
        retrain_path = exports_dir / "retraining_metadata.json"
        self.assertTrue(retrain_path.is_file())
        retrain_doc = json.loads(retrain_path.read_text(encoding="utf-8"))
        self.assertEqual(retrain_doc["unit_count"], 2)
        self.assertEqual(retrain_doc["units"], summary["weight_retention_policy"]["retraining_registry"])
        self.assertEqual(retrain_doc["registry_sha256"], summary["weight_retention_policy"]["retraining_registry_sha256"])
        self.assertEqual(retrain_doc["statistical_validation_status"], STATISTICAL_VALIDATION_STATUS)
        self.assertFalse(retrain_doc["confirmatory_declarations_enabled"])

        csv_df = pd.read_csv(exports_dir / "aggregated_results.csv")
        self.assertIn("expected_optimizer_steps", csv_df.columns)
        self.assertIn("actual_global_step", csv_df.columns)
        self.assertEqual(list(csv_df["statistical_validation_status"].unique()), [STATISTICAL_VALIDATION_STATUS])
        self.assertFalse(bool(csv_df["confirmatory_declarations_enabled"].any()))

        md_text = (exports_dir / "post_stage_b_report.md").read_text(encoding="utf-8")
        self.assertIn(STATISTICAL_VALIDATION_STATUS, md_text)
        self.assertIn("Equal-Epoch Optimizer-Step Budget Disclosure", md_text)
        self.assertIn("exports/retraining_metadata.json", md_text)

        # Benchmark workload fixture uses ZERO held-out test/OOD rows and matches 7,114 length-paired distribution
        real_df = pd.read_parquet(bdir / "real_train_pool.parquet")
        val_fold0 = real_df[real_df["curation_cv_fold"] == 0].reset_index(drop=True)
        fix_texts, fix_meta = _build_t4_benchmark_workload_fixture(val_fold0, val_f, "text_canon_masked")
        self.assertEqual(len(fix_texts), 7114)
        self.assertEqual(fix_meta["heldout_or_ood_rows_used"], 0)
        self.assertEqual((fix_meta["n_short_rows"], fix_meta["n_long_concat_rows"]), (1698, 5416))
        self.assertGreater(fix_meta["char_len_mean"], 130.0)


if __name__ == "__main__":
    unittest.main()
