"""Targeted Stage 4 Final Remediation regression tests covering:
  1. Enforceable hard <=25% CPU validation wrapper (scripts/run_stage4_capped.py)
  2. Allowed-view 32-pair tokenizer input_ids collision gate & notebook wiring
  3. Checkpoint & lease hardening (expected_stale_writer_ids, corrupt confirmation, commit refresh, prune_errors)
  4. Portable kit OS metadata filtering, live frozen mask re-derivation, transitive deps & pyproject alignment
"""

from __future__ import annotations

import contextlib
import errno
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.post_stage_b_checkpointing import (
    LEGACY_RUNNER_CHECKPOINT_COMMIT,
    ensure_unit_completion_event,
    inspect_writer_leases,
    recover_latest_valid_checkpoint,
    register_single_writer_lease,
    release_single_writer_lease,
    scan_unit_staging,
    sync_local_checkpoint_to_staging,
)
from bangla_smish.post_stage_b_colab_deps import (
    BANGLABERT_NORMALIZER_TRANSITIVE_DEPS,
    STAGE4_BASE_PACKAGES,
    STAGE4_CORE_IMPORTS,
)
from bangla_smish.post_stage_b_integrity import verify_portable_colab_kit
from bangla_smish.post_stage_b_runner import (
    SYNTHETIC_WORKLOAD_FIXTURE_ENABLED,
    PostStageBConfig,
    _CHECKPOINT_COMMIT,
    _IDENTITY_PACKAGE_DISTS,
    _checkpoint_file_hashes,
    _fsync_dir_best_effort,
    _local_checkpoint_base,
    _resolve_package_version,
    _run_identity,
    _valid_checkpoint,
    aggregate_post_stage_b_results,
    atomic_write_json,
    atomic_write_text,
    collect_hardware_fingerprint,
    collect_stable_environment_fingerprint,
    compute_optimizer_step_budget,
    hf_train_and_predict_fold,
    load_config,
    run_post_stage_b_pipeline,
    verify_t4_benchmark_hardware,
)
from bangla_smish.post_stage_b_tokenizer_gate import (
    PINNED_TOKENIZER_REVISIONS,
    STAGE3_32_CANDIDATE_PAIRS,
    load_documented_32_pairs,
    run_allowed_view_tokenizer_collision_gate,
)


def _write_ckpt(ckpt_dir: Path, step: int, *, best_ckpt: str | None = None) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("config.json", "training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"):
        (ckpt_dir / fname).write_bytes(f"{fname}-step-{step}".encode("utf-8"))
    (ckpt_dir / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "best_model_checkpoint": best_ckpt}), encoding="utf-8"
    )
    return ckpt_dir


class _MockTokenizer:
    """Realistic transformers==4.57.1 mock exposing snapshot vocab_file path (no _commit_hash)."""

    def __init__(self, revision: str, *, force_collision: bool = False) -> None:
        self.init_kwargs = (
            {"vocab_file": f"/mock_hf_cache/models--mock/snapshots/{revision}/vocab.txt"}
            if revision
            else {}
        )
        self._force_collision = force_collision

    def __call__(self, texts: list[str], truncation: bool = True, max_length: int = 128, add_special_tokens: bool = True):
        out = []
        for idx, t in enumerate(texts):
            if self._force_collision:
                out.append([101, 202, 303, 102])
            else:
                # Encode deterministic distinct integer codepoints up to max_length - 2
                toks = [101] + [ord(ch) + idx * 0 for ch in t[: max_length - 2]] + [102]
                out.append(toks)
        return {"input_ids": out}


class TestStage4FinalRemediation(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    # ------------------------------------------------------------------
    # 1. Hard CPU Limit Enforcement Wrapper
    # ------------------------------------------------------------------
    def test_cpu_cap_wrapper_enforces_quota_and_affinity_fail_closed(self) -> None:
        capped_path = ROOT / "scripts" / "run_stage4_capped.py"
        spec = importlib.util.spec_from_file_location("run_stage4_capped", capped_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Simulated host with 12 CPUs and unrestricted affinity [0..11] MUST fail closed!
        with (
            patch.object(mod.os, "cpu_count", return_value=12),
            patch.object(mod.os, "sched_getaffinity", return_value=set(range(12))),
        ):
            with self.assertRaisesRegex(RuntimeError, "Hard CPU affinity gate failed"):
                mod.verify_hard_cpu_cap(0.25)

        # Affinity <= 3 but missing/unlimited cgroup cpu.max MUST fail closed!
        with (
            patch.object(mod.os, "cpu_count", return_value=12),
            patch.object(mod.os, "sched_getaffinity", return_value={0, 1, 2}),
            patch.object(mod, "_read_cgroup_cpu_max", return_value=(None, "/sys/fs/cgroup/cpu.max=max 100000")),
        ):
            with self.assertRaisesRegex(RuntimeError, "Hard cgroup CPUQuota gate failed"):
                mod.verify_hard_cpu_cap(0.25)

        # Affinity <= 3 AND cgroup quota <= 3.0 AND child affinity matches [0,1,2] MUST pass!
        with (
            patch.object(mod.os, "cpu_count", return_value=12),
            patch.object(mod.os, "sched_getaffinity", return_value={0, 1, 2}),
            patch.object(mod, "_read_cgroup_cpu_max", return_value=(3.0, "/sys/fs/cgroup/test/cpu.max=300000 100000")),
            patch.object(mod.subprocess, "check_output", return_value="[0, 1, 2]"),
        ):
            res = mod.verify_hard_cpu_cap(0.25)
            self.assertEqual(res["max_allowed_cpus"], 3)
            self.assertEqual(res["parent_affinity"], [0, 1, 2])
            self.assertEqual(res["child_affinity"], [0, 1, 2])
            self.assertEqual(res["cgroup_quota_cpus"], 3.0)

    # ------------------------------------------------------------------
    # 2. Allowed-View 32-Pair Tokenizer Collision Gate & Notebook Integration
    # ------------------------------------------------------------------
    def test_32_pair_allowed_view_tokenizer_collision_gate_mocked_and_notebook(self) -> None:
        v2_dir = ROOT / "data" / "processed" / "post_stage_b_bundle_v2"
        row_map = load_documented_32_pairs(v2_dir)
        self.assertEqual(len(STAGE3_32_CANDIDATE_PAIRS), 32)
        self.assertGreaterEqual(len(row_map), 50)

        # Missing pair from bundle fails closed WITHOUT inventing replacement pairs
        with tempfile.TemporaryDirectory() as td:
            bcopy = Path(td) / "v2_incomplete"
            shutil.copytree(v2_dir, bcopy)
            import pandas as pd
            test_df = pd.read_parquet(bcopy / "eval_smishdetect_test.parquet")
            test_df[test_df["row_id"] != "cur_cfa2f9f81a31"].to_parquet(
                bcopy / "eval_smishdetect_test.parquet", index=False
            )
            with self.assertRaisesRegex(RuntimeError, "refusing to invent replacement pairs"):
                load_documented_32_pairs(bcopy)

        cfg = load_config(ROOT / "experiments" / "colab" / "configs" / "post_stage_b_stage4_option_a_config.json")
        self.assertEqual(cfg.tokenizer_revisions, PINNED_TOKENIZER_REVISIONS)
        export_json = self.root / "tok_gate.json"

        # Mocked distinct tokenizers pass all 4 backbones x 2 views x 32 pairs = 256 checks
        with patch(
            "bangla_smish.post_stage_b_tokenizer_gate.prepare_tokenizer_texts",
            side_effect=lambda texts, model: texts,
        ):
            rep = run_allowed_view_tokenizer_collision_gate(
                v2_dir,
                cfg,
                allowed_views=("raw", "canon_masked"),
                tokenizer_loader=lambda model, rev: _MockTokenizer(rev, force_collision=False),
                export_path=export_json,
            )
        self.assertTrue(rep["passed"])
        self.assertEqual(rep["total_checks"], 4 * 2 * 32)
        self.assertEqual(rep["collision_count"], 0)
        self.assertEqual(rep["config_hash"], cfg.config_hash())
        self.assertEqual(len(rep["pair_ids"]), 32)
        disk_rep = json.loads(export_json.read_text(encoding="utf-8"))
        self.assertEqual(disk_rep["tokenizer_revisions"], PINNED_TOKENIZER_REVISIONS)

        # Unauthorized view rejected fail-closed
        with self.assertRaisesRegex(ValueError, "supports only"):
            run_allowed_view_tokenizer_collision_gate(
                v2_dir, cfg, allowed_views=("brand_masked",),
                tokenizer_loader=lambda m, r: _MockTokenizer(r),
            )

        # Collision detected fails closed
        with (
            patch("bangla_smish.post_stage_b_tokenizer_gate.prepare_tokenizer_texts", side_effect=lambda t, m: t),
            self.assertRaisesRegex(RuntimeError, "FAILED"),
        ):
            run_allowed_view_tokenizer_collision_gate(
                v2_dir, cfg, allowed_views=("raw",),
                tokenizer_loader=lambda m, r: _MockTokenizer(r, force_collision=True),
            )

        # Wrong revision fails closed
        with (
            patch("bangla_smish.post_stage_b_tokenizer_gate.prepare_tokenizer_texts", side_effect=lambda t, m: t),
            self.assertRaisesRegex(RuntimeError, "loaded tokenizer commit"),
        ):
            run_allowed_view_tokenizer_collision_gate(
                v2_dir, cfg, allowed_views=("raw",),
                tokenizer_loader=lambda m, r: _MockTokenizer("0" * 40),
            )

        # Notebook Cells 5 & 6 enforce pre-training gate
        nb = nbformat.read(ROOT / "experiments" / "colab" / "post_stage_b_stage4_experiments.ipynb", as_version=4)
        self.assertIn("run_allowed_view_tokenizer_collision_gate", nb.cells[5].source)
        self.assertIn("TOK_GATE_REPORT", nb.cells[5].source)
        self.assertIn("tempfile.gettempdir()", nb.cells[5].source)
        self.assertNotIn("ARTIFACTS_DIR / 'exports' / 'tokenizer_collision_gate.json'", nb.cells[5].source)
        self.assertIn("TOKENIZER_COLLISION_GATE_PASSED", nb.cells[6].source)
        self.assertIn("tokenizer_gate_report=globals().get('TOK_GATE_REPORT')", nb.cells[6].source)

    # ------------------------------------------------------------------
    # 2B. Cell 5 / Cell 6 Artifact Initialization Integration & Gate Persistence
    # ------------------------------------------------------------------
    @staticmethod
    def _mock_t4_hw(*, driver: str = "550.54.15") -> dict:
        return {
            "cuda_available": True,
            "gpu_count": 1,
            "gpu_name": "Tesla T4",
            "compute_capability": "7.5",
            "nvidia_driver_version": driver,
            "torch_cuda_version": "12.4",
        }

    def _mock_t4_train_eval_fn(self, calls: list[str]):
        def _fn(**kwargs):
            calls.append(f"{kwargs['model_name']}:{kwargs['text_view']}:{kwargs['seed']}")

            def _probs(frame):
                return np.eye(3)[frame["label_idx"].to_numpy(dtype=int)] * 0.94 + 0.02

            val_df = kwargs["val_df"]
            hsets = kwargs["heldout_sets"]
            cf_df = hsets["counterfactual_smish_probe"]
            preds = {
                "val_probs": _probs(val_df),
                "val_adv_probs": _probs(val_df),
                "cf_orig_probs": _probs(cf_df),
                "cf_contact_probs": _probs(cf_df),
                "cf_brand_probs": _probs(cf_df),
            }
            if "smishdetect_validation" in hsets:
                preds["smishdetect_validation_probs"] = _probs(hsets["smishdetect_validation"])
            return {
                "train_seconds": 0.02,
                "actual_global_step": 10,
                "model_revision": kwargs["cfg"].tokenizer_revisions[kwargs["model_name"]],
                "tokenizer_revision": kwargs["cfg"].tokenizer_revisions[kwargs["model_name"]],
                "predictions": preds,
            }

        return _fn

    def _run_mocked_t4_gate(self, v2_dir: Path, cfg: PostStageBConfig, export_path: Path) -> dict:
        with patch(
            "bangla_smish.post_stage_b_tokenizer_gate.prepare_tokenizer_texts",
            side_effect=lambda texts, model: texts,
        ):
            return run_allowed_view_tokenizer_collision_gate(
                v2_dir,
                cfg,
                allowed_views=tuple(v for v in cfg.text_views if v in {"raw", "canon_masked"}),
                tokenizer_loader=lambda model, rev: _MockTokenizer(rev, force_collision=False),
                export_path=export_path,
            )

    def test_cell5_cell6_fresh_init_persistence_resume_and_fail_closed_gate(self) -> None:
        v2_dir = ROOT / "data" / "processed" / "post_stage_b_bundle_v2"
        # Use benchmark_workload_fixture_eval=False on a copy for post-init mocked lifecycle tests;
        # untouched frozen config fail-closed pre-lease behavior is verified separately in v7 tests.
        t4_cfg = replace(
            load_config(ROOT / "experiments" / "colab" / "configs" / "post_stage_b_stage4_t4_benchmark_config.json"),
            benchmark_workload_fixture_eval=False,
        )

        # --- Group 1: Fresh-run initialization and gate-report persistence ---
        preflight_path = (
            Path(tempfile.gettempdir())
            / f"stage4_test_tok_gate_{os.getpid()}_{t4_cfg.config_hash()[:12]}.json"
        )
        self.addCleanup(lambda: preflight_path.unlink(missing_ok=True))
        tok_gate = self._run_mocked_t4_gate(v2_dir, t4_cfg, preflight_path)
        self.assertTrue(preflight_path.is_file())

        with patch(
            "bangla_smish.post_stage_b_runner.collect_hardware_fingerprint",
            return_value=self._mock_t4_hw(),
        ):
            art = self.root / "t4_fresh_run"
            self.assertFalse(art.exists())
            calls: list[str] = []
            partial = run_post_stage_b_pipeline(
                bundle_dir=v2_dir,
                artifacts_dir=art,
                cfg=t4_cfg,
                resume=True,
                verify_checksums=True,
                train_eval_fn=self._mock_t4_train_eval_fn(calls),
                stop_after_units=1,
                tokenizer_gate_report=tok_gate,
            )
            self.assertEqual(partial["completed_unit_count"], 1)
            self.assertEqual(len(calls), 1)
            self.assertTrue((art / "run_state.json").is_file())
            persisted_gate_path = art / "exports" / "tokenizer_collision_gate.json"
            self.assertTrue(persisted_gate_path.is_file())
            original_gate_bytes = persisted_gate_path.read_bytes()
            self.assertEqual(json.loads(original_gate_bytes.decode("utf-8")), tok_gate)

            # Path-based tokenizer_gate_report argument also works on a separate fresh run
            art_from_path = self.root / "t4_fresh_from_path"
            path_calls: list[str] = []
            res_path = run_post_stage_b_pipeline(
                bundle_dir=v2_dir,
                artifacts_dir=art_from_path,
                cfg=t4_cfg,
                resume=True,
                verify_checksums=True,
                train_eval_fn=self._mock_t4_train_eval_fn(path_calls),
                stop_after_units=1,
                tokenizer_gate_report=preflight_path,
            )
            self.assertEqual(res_path["completed_unit_count"], 1)
            self.assertEqual(
                json.loads((art_from_path / "exports" / "tokenizer_collision_gate.json").read_text(encoding="utf-8")),
                tok_gate,
            )

            # --- Group 2: Interrupt/resume preservation & revalidation ---
            # Inject unreleased stale lease simulating abrupt interruption after unit 1
            register_single_writer_lease(art, writer_id="crashed_t4_worker")
            with self.assertRaisesRegex(RuntimeError, "Single-writer policy violation"):
                run_post_stage_b_pipeline(
                    bundle_dir=v2_dir,
                    artifacts_dir=art,
                    cfg=t4_cfg,
                    resume=True,
                    train_eval_fn=self._mock_t4_train_eval_fn(calls),
                    tokenizer_gate_report=tok_gate,
                )

            # Resume with authorized stale takeover and tokenizer_gate_report=None -> revalidates & preserves persisted report
            mid = run_post_stage_b_pipeline(
                bundle_dir=v2_dir,
                artifacts_dir=art,
                cfg=t4_cfg,
                resume=True,
                train_eval_fn=self._mock_t4_train_eval_fn(calls),
                stop_after_units=2,
                allow_stale_takeover=True,
                expected_stale_writer_ids=["crashed_t4_worker"],
                takeover_reason="Owner confirmed stop after unit 1",
                tokenizer_gate_report=None,
            )
            self.assertEqual(mid["completed_unit_count"], 3)
            self.assertEqual(len(calls), 3)
            self.assertEqual(persisted_gate_path.read_bytes(), original_gate_bytes)

            # Resume remaining units passing a freshly re-run Cell 5 report -> validates both & preserves original export
            fresh_gate = self._run_mocked_t4_gate(v2_dir, t4_cfg, preflight_path)
            done = run_post_stage_b_pipeline(
                bundle_dir=v2_dir,
                artifacts_dir=art,
                cfg=t4_cfg,
                resume=True,
                train_eval_fn=self._mock_t4_train_eval_fn(calls),
                tokenizer_gate_report=fresh_gate,
            )
            self.assertEqual(done["completed_unit_count"], 5)
            self.assertEqual(len(calls), 5)
            self.assertEqual(persisted_gate_path.read_bytes(), original_gate_bytes)
            self.assertTrue((art / "exports" / "aggregated_results.json").is_file())
            self.assertTrue((art / "exports" / "retraining_metadata.json").is_file())

            # Corrupted/tampered persisted gate report on disk is rejected fail-closed on resume
            tampered_art = self.root / "t4_tampered_persisted"
            shutil.copytree(art, tampered_art)
            bad_persisted = json.loads((tampered_art / "exports" / "tokenizer_collision_gate.json").read_text(encoding="utf-8"))
            bad_persisted["collision_count"] = 1
            bad_persisted["passed"] = False
            (tampered_art / "exports" / "tokenizer_collision_gate.json").write_text(json.dumps(bad_persisted), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "did not pass cleanly"):
                run_post_stage_b_pipeline(
                    bundle_dir=v2_dir,
                    artifacts_dir=tampered_art,
                    cfg=t4_cfg,
                    resume=True,
                    train_eval_fn=self._mock_t4_train_eval_fn([]),
                    tokenizer_gate_report=fresh_gate,
                )

            # --- Group 3: Fail-closed validation for missing/invalid reports & _read_state guard ---
            # Missing report on fresh pinned-revision run fails closed without creating run_state.json
            missing_art = self.root / "t4_missing_gate"
            missing_calls: list[str] = []
            with self.assertRaisesRegex(RuntimeError, "tokenizer collision gate report is required"):
                run_post_stage_b_pipeline(
                    bundle_dir=v2_dir,
                    artifacts_dir=missing_art,
                    cfg=t4_cfg,
                    train_eval_fn=self._mock_t4_train_eval_fn(missing_calls),
                    tokenizer_gate_report=None,
                )
            self.assertEqual(missing_calls, [])
            self.assertFalse((missing_art / "run_state.json").exists())
            self.assertFalse((missing_art / "exports").exists())

            # Invalid report variants all fail closed before initializing run_state.json
            invalid_cases = [
                ("failed_gate", {**tok_gate, "passed": False, "collision_count": 1, "collisions": [{"pair_id": "P01"}]}, RuntimeError),
                ("wrong_config_hash", {**tok_gate, "config_hash": "0" * 64}, ValueError),
                ("wrong_run_name", {**tok_gate, "run_name": "other_run"}, ValueError),
                ("wrong_schema", {**tok_gate, "schema_version": "wrong_v0"}, ValueError),
                ("wrong_revisions", {**tok_gate, "tokenizer_revisions": {**tok_gate["tokenizer_revisions"], "xlm-roberta-base": "0" * 40}}, ValueError),
                ("wrong_views", {**tok_gate, "allowed_views": ["raw"]}, ValueError),
                ("truncated_results", {**tok_gate, "results": tok_gate["results"][:-1]}, ValueError),
                ("entry_collision_flag", {**tok_gate, "results": [{**tok_gate["results"][0], "input_ids_equal": True}, *tok_gate["results"][1:]]}, RuntimeError),
                ("missing_path", self.root / "does_not_exist.json", RuntimeError),
            ]
            for label, bad_report, exc_type in invalid_cases:
                with self.subTest(invalid_case=label):
                    bad_art = self.root / f"t4_bad_{label}"
                    bad_calls: list[str] = []
                    with self.assertRaises(exc_type):
                        run_post_stage_b_pipeline(
                            bundle_dir=v2_dir,
                            artifacts_dir=bad_art,
                            cfg=t4_cfg,
                            train_eval_fn=self._mock_t4_train_eval_fn(bad_calls),
                            tokenizer_gate_report=bad_report,
                        )
                    self.assertEqual(bad_calls, [])
                    self.assertFalse((bad_art / "run_state.json").exists())
                    self.assertFalse((bad_art / "exports").exists())

            # Unexpected pre-existing artifact in uninitialized artifacts_dir STILL rejected by _read_state
            dirty_art = self.root / "t4_dirty_unidentified"
            (dirty_art / "exports").mkdir(parents=True)
            (dirty_art / "exports" / "tokenizer_collision_gate.json").write_text(json.dumps(tok_gate), encoding="utf-8")
            dirty_calls: list[str] = []
            with self.assertRaisesRegex(ValueError, "Artifact directory contains an unidentified run"):
                run_post_stage_b_pipeline(
                    bundle_dir=v2_dir,
                    artifacts_dir=dirty_art,
                    cfg=t4_cfg,
                    train_eval_fn=self._mock_t4_train_eval_fn(dirty_calls),
                    tokenizer_gate_report=tok_gate,
                )
            self.assertEqual(dirty_calls, [])
            self.assertFalse((dirty_art / "run_state.json").exists())

        # --- Group 4: Notebook generator parity and portable-kit integrity after preflight ---
        gen_path = ROOT / "scripts" / "generate_post_stage_b_stage4_notebook.py"
        gen_spec = importlib.util.spec_from_file_location("generate_post_stage_b_stage4_notebook", gen_path)
        gen_mod = importlib.util.module_from_spec(gen_spec)
        gen_spec.loader.exec_module(gen_mod)
        nb = nbformat.read(ROOT / "experiments" / "colab" / "post_stage_b_stage4_experiments.ipynb", as_version=4)
        nbformat.validate(nb)
        self.assertEqual(gen_mod.build_stage4_notebook(), nb)

        kit_script = ROOT / "scripts" / "prepare_stage4_colab_kit.py"
        kit_spec = importlib.util.spec_from_file_location("prepare_stage4_colab_kit", kit_script)
        kit_mod = importlib.util.module_from_spec(kit_spec)
        kit_spec.loader.exec_module(kit_mod)
        staged = self.root / "kit_v4_preflight_check"
        kit_mod.export_stage4_colab_kit(ROOT, v2_dir, staged)
        # Writing preflight to system temp leaves both WORKSPACE_ROOT and ARTIFACTS_DIR untouched
        self._run_mocked_t4_gate(staged / "data" / "processed" / "post_stage_b_bundle_v2", t4_cfg, preflight_path)
        verified_kit = verify_portable_colab_kit(staged)
        self.assertEqual(verified_kit["verified_runtime_file_count"], 34)
        self.assertFalse((staged / "artifacts").exists())

    # ------------------------------------------------------------------
    # 2C. Full End-to-End Mocked Notebook Lifecycle Simulation (Scenarios A-L)
    # ------------------------------------------------------------------
    def test_full_notebook_end_to_end_lifecycle_scenarios_a_through_l(self) -> None:
        from bangla_smish.post_stage_b_data import sha256_file

        v2_dir = ROOT / "data" / "processed" / "post_stage_b_bundle_v2"
        nb = nbformat.read(ROOT / "experiments" / "colab" / "post_stage_b_stage4_experiments.ipynb", as_version=4)
        nbformat.validate(nb)

        # Stage an isolated portable kit simulating MyDrive/<DRIVE_WORKSPACE_REL>
        kit_script = ROOT / "scripts" / "prepare_stage4_colab_kit.py"
        kit_spec = importlib.util.spec_from_file_location("prepare_stage4_colab_kit", kit_script)
        kit_mod = importlib.util.module_from_spec(kit_spec)
        kit_spec.loader.exec_module(kit_mod)

        fake_drive_root = self.root / "fake_drive" / "MyDrive"
        rel_ws = "bangla_smish_post_stage_b_stage4_v8"
        staged_kit = fake_drive_root / rel_ws
        kit_mod.export_stage4_colab_kit(ROOT, v2_dir, staged_kit)
        canonical_nb_path = staged_kit / "experiments" / "colab" / "post_stage_b_stage4_experiments.ipynb"
        canonical_nb_sha_before = sha256_file(canonical_nb_path)

        fake_torch = types.SimpleNamespace(
            __version__="2.6.0+cu124",
            manual_seed=lambda s: None,
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                manual_seed_all=lambda s: None,
                device_count=lambda: 1,
                get_device_name=lambda idx: "Tesla T4 (Mock)",
                get_device_capability=lambda idx: (7, 5),
                get_device_properties=lambda idx: types.SimpleNamespace(total_memory=15 * (1024**3)),
            ),
            version=types.SimpleNamespace(cuda="12.4"),
        )
        fake_colab_drive = types.SimpleNamespace(mount=lambda path, force_remount=False: None)
        fake_colab = types.SimpleNamespace(drive=fake_colab_drive)

        def _patch_cell3_drive(src: str) -> str:
            return src.replace("Path('/content/drive/MyDrive')", f"Path({str(fake_drive_root)!r})")

        def _exec_cell(ns: dict, cell_idx: int, *, src_override: str | None = None) -> str:
            buf = io.StringIO()
            code = src_override if src_override is not None else nb.cells[cell_idx].source
            with contextlib.redirect_stdout(buf):
                exec(compile(code, f"cell_{cell_idx}", "exec"), ns)
            return buf.getvalue()

        def _fake_nvidia_smi(cmd, **kw):
            if "--query-gpu=driver_version" in cmd:
                return "550.54.15\n"
            return "Tesla T4, 15360 MiB, 550.54.15\n"

        def _run_cells_1_to_3(ns: dict, *, run_id: str, run_mode: str = "t4_benchmark", auth_t4: bool = True, takeover_kwargs: dict | None = None) -> None:
            c1 = (
                nb.cells[1].source
                .replace("RUN_MODE = 'gpu_smoke_test'", f"RUN_MODE = {run_mode!r}")
                .replace("AUTHORIZE_T4_BENCHMARK = False", f"AUTHORIZE_T4_BENCHMARK = {auth_t4!r}")
                .replace("ARTIFACT_RUN_ID = 'stage4-v8-smoke-01'", f"ARTIFACT_RUN_ID = {run_id!r}")
            )
            if takeover_kwargs:
                for key, val in takeover_kwargs.items():
                    c1 = c1.replace(f"{key} = {False if isinstance(val, bool) else repr('')}", f"{key} = {val!r}")
            with (
                patch.dict(sys.modules, {"torch": fake_torch, "google": types.SimpleNamespace(colab=fake_colab), "google.colab": fake_colab}),
                patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
                patch("subprocess.check_output", side_effect=_fake_nvidia_smi),
            ):
                _exec_cell(ns, 1, src_override=c1)
                _exec_cell(ns, 2)
                _exec_cell(ns, 3, src_override=_patch_cell3_drive(nb.cells[3].source))

        # --- Scenario B: Cell 4 dependency installation followed by runtime restart ---
        ns_b: dict = {}
        _run_cells_1_to_3(ns_b, run_id="nb-life-b-01")
        with patch("runpy.run_path", return_value={
            "preflight_colab_dependencies": lambda *a, **kw: (_ for _ in ()).throw(
                RuntimeError("Dependencies installed/updated. RESTART REQUIRED: Click 'Runtime -> Restart session'.")
            )
        }):
            with self.assertRaisesRegex(RuntimeError, "RESTART REQUIRED"):
                _exec_cell(ns_b, 4)
        self.assertFalse(ns_b["ARTIFACTS_DIR"].exists())

        # Simulate kernel restart (fresh empty namespace), re-run Cells 1-4 -> verified cleanly
        ns_b_restarted: dict = {}
        _run_cells_1_to_3(ns_b_restarted, run_id="nb-life-b-01")
        with patch("runpy.run_path", return_value={
            "preflight_colab_dependencies": lambda *a, **kw: {"status": "verified", "torch_version": "2.6.0+cu124", "cuda_available": True}
        }):
            out4 = _exec_cell(ns_b_restarted, 4)
        self.assertIn("verified", out4)
        self.assertFalse(ns_b_restarted["ARTIFACTS_DIR"].exists())

        # --- Scenarios A, C, I, G, J: Fresh init, repeated preflight, Cell 5->6->7, resume after completed unit ---
        ns_main: dict = {}
        _run_cells_1_to_3(ns_main, run_id="nb-life-a-01")
        with patch("runpy.run_path", return_value={
            "preflight_colab_dependencies": lambda *a, **kw: {"status": "verified"}
        }):
            _exec_cell(ns_main, 4)

        # Scenario I: Repeated execution of preflight Cell 5 (twice) leaves WORKSPACE_ROOT & ARTIFACTS_DIR untouched
        with (
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch("bangla_smish.post_stage_b_tokenizer_gate._default_hf_tokenizer_loader", side_effect=lambda m, r: _MockTokenizer(r, force_collision=False)),
            patch("bangla_smish.post_stage_b_tokenizer_gate.prepare_tokenizer_texts", side_effect=lambda t, m: t),
        ):
            _exec_cell(ns_main, 5)
            first_preflight_path = ns_main["TOK_GATE_PREFLIGHT_PATH"]
            self.addCleanup(lambda p=first_preflight_path: p.unlink(missing_ok=True))
            _exec_cell(ns_main, 5)

        self.assertTrue(ns_main["PORTABLE_KIT_VERIFIED"])
        self.assertTrue(ns_main["TOKENIZER_COLLISION_GATE_PASSED"])
        self.assertIsInstance(ns_main["TOK_GATE_REPORT"], dict)
        self.assertFalse(ns_main["ARTIFACTS_DIR"].exists())
        self.assertFalse(first_preflight_path.resolve().is_relative_to(staged_kit.resolve()))
        self.assertEqual(sha256_file(canonical_nb_path), canonical_nb_sha_before)

        # Redirect local_checkpoint_root to temp dir and disable blocked synthetic fixture for post-init mocked lifecycle
        local_ssd_root = self.root / "mock_local_ssd"
        ns_main["cfg"] = replace(ns_main["cfg"], local_checkpoint_root=str(local_ssd_root), benchmark_workload_fixture_eval=False)
        # Re-run gate for modified cfg hash
        ns_main["TOK_GATE_REPORT"] = self._run_mocked_t4_gate(ns_main["BUNDLE_DIR"], ns_main["cfg"], first_preflight_path)

        # Scenario A & C & G: Run Cell 6 stopping after unit 1, verify init + persisted export + local SSD cleanup
        calls_main: list[str] = []
        base_mock_fn = self._mock_t4_train_eval_fn(calls_main)

        def _mock_with_local_ckpt(**kw):
            ckpt_dir = kw["checkpoint_dir"]
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "checkpoint-10").mkdir(exist_ok=True)
            (ckpt_dir / "checkpoint-10" / "dummy.bin").write_bytes(b"ckpt")
            return base_mock_fn(**kw)

        real_run_pipeline = run_post_stage_b_pipeline

        def _cell6_partial(**kw):
            return real_run_pipeline(**kw, train_eval_fn=_mock_with_local_ckpt, stop_after_units=1)

        with (
            patch("bangla_smish.post_stage_b_runner.collect_hardware_fingerprint", return_value=self._mock_t4_hw()),
            patch("bangla_smish.post_stage_b_runner.run_post_stage_b_pipeline", side_effect=_cell6_partial),
        ):
            out6_part = _exec_cell(ns_main, 6)
        self.assertIn("Completed units total: 1 / 5", out6_part)
        art_main = ns_main["ARTIFACTS_DIR"]
        self.assertTrue((art_main / "run_state.json").is_file())
        self.assertTrue((art_main / "exports" / "tokenizer_collision_gate.json").is_file())
        gate_bytes_after_init = (art_main / "exports" / "tokenizer_collision_gate.json").read_bytes()
        # Local SSD temp checkpoints for committed unit 1 were pruned cleanly when keep_model_weights='none'
        u1_local = _local_checkpoint_base(art_main, ns_main["cfg"]) / "exp_D__csebuetnlp__banglabert__canon_masked__fold_0"
        self.assertFalse(u1_local.exists())

        # Scenario G: Restart kernel and resume units 2-5 via Cell 6; unit 1 skipped, gate export preserved
        ns_resume: dict = {}
        _run_cells_1_to_3(ns_resume, run_id="nb-life-a-01")
        with (
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch("bangla_smish.post_stage_b_tokenizer_gate._default_hf_tokenizer_loader", side_effect=lambda m, r: _MockTokenizer(r, force_collision=False)),
            patch("bangla_smish.post_stage_b_tokenizer_gate.prepare_tokenizer_texts", side_effect=lambda t, m: t),
        ):
            _exec_cell(ns_resume, 5)
        ns_resume["cfg"] = replace(ns_resume["cfg"], local_checkpoint_root=str(local_ssd_root), benchmark_workload_fixture_eval=False)
        ns_resume["TOK_GATE_REPORT"] = self._run_mocked_t4_gate(ns_resume["BUNDLE_DIR"], ns_resume["cfg"], first_preflight_path)

        with (
            patch("bangla_smish.post_stage_b_runner.collect_hardware_fingerprint", return_value=self._mock_t4_hw()),
            patch(
                "bangla_smish.post_stage_b_runner.run_post_stage_b_pipeline",
                side_effect=lambda **kw: real_run_pipeline(**kw, train_eval_fn=_mock_with_local_ckpt),
            ),
        ):
            out6_done = _exec_cell(ns_resume, 6)
        self.assertIn("Completed units total: 5 / 5", out6_done)
        self.assertEqual(len(calls_main), 5)  # Unit 1 was NOT retrained!
        self.assertEqual((art_main / "exports" / "tokenizer_collision_gate.json").read_bytes(), gate_bytes_after_init)

        # Scenario J: Cell 7 execution after successful completion
        out7 = _exec_cell(ns_resume, 7)
        self.assertIn("tokenizer_collision_gate.json", out7)
        self.assertIn("Tokenizer gate export: PASSED", out7)
        self.assertIn("Retraining metadata: 5 units", out7)
        self.assertIn("exp_D__csebuetnlp__banglabert__canon_masked__fold_0", out7)

        # --- Scenario D: Failure during tokenizer verification blocks Cell 6 without touching ARTIFACTS_DIR ---
        ns_d: dict = {}
        _run_cells_1_to_3(ns_d, run_id="nb-life-d-01")
        with (
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch("bangla_smish.post_stage_b_tokenizer_gate._default_hf_tokenizer_loader", side_effect=lambda m, r: _MockTokenizer(r, force_collision=True)),
            patch("bangla_smish.post_stage_b_tokenizer_gate.prepare_tokenizer_texts", side_effect=lambda t, m: t),
        ):
            with self.assertRaisesRegex(RuntimeError, "collision gate FAILED"):
                _exec_cell(ns_d, 5)
        if ns_d.get("TOK_GATE_PREFLIGHT_PATH") is not None:
            self.addCleanup(lambda p=ns_d["TOK_GATE_PREFLIGHT_PATH"]: p.unlink(missing_ok=True))
        self.assertFalse(ns_d["TOKENIZER_COLLISION_GATE_PASSED"])
        self.assertIsNone(ns_d["TOK_GATE_REPORT"])
        self.assertFalse(ns_d["ARTIFACTS_DIR"].exists())
        with self.assertRaisesRegex(RuntimeError, "Stale or unverified preflight state"):
            _exec_cell(ns_d, 6)

        # --- Scenario E: Failure during run-state initialization / stale Cell 1 edit without Cells 3 & 5 ---
        ns_e = dict(ns_resume)
        # Re-run Cell 1 with a new ARTIFACT_RUN_ID and jump directly to Cell 6 without Cells 3 & 5
        c1_stale = (
            nb.cells[1].source
            .replace("RUN_MODE = 'gpu_smoke_test'", "RUN_MODE = 't4_benchmark'")
            .replace("AUTHORIZE_T4_BENCHMARK = False", "AUTHORIZE_T4_BENCHMARK = True")
            .replace("ARTIFACT_RUN_ID = 'stage4-v8-smoke-01'", "ARTIFACT_RUN_ID = 'nb-life-e-02'")
        )
        _exec_cell(ns_e, 1, src_override=c1_stale)
        with self.assertRaisesRegex(RuntimeError, "Stale or unverified preflight state"):
            _exec_cell(ns_e, 6)
        self.assertFalse((staged_kit / "artifacts" / "t4_benchmark" / "nb-life-e-02").exists())

        # --- Scenarios F & H: Interruption during a training unit + resume after incomplete unit ---
        ns_f: dict = {}
        _run_cells_1_to_3(ns_f, run_id="nb-life-fh-01")
        with (
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch("bangla_smish.post_stage_b_tokenizer_gate._default_hf_tokenizer_loader", side_effect=lambda m, r: _MockTokenizer(r, force_collision=False)),
            patch("bangla_smish.post_stage_b_tokenizer_gate.prepare_tokenizer_texts", side_effect=lambda t, m: t),
        ):
            _exec_cell(ns_f, 5)
        self.addCleanup(lambda p=ns_f["TOK_GATE_PREFLIGHT_PATH"]: p.unlink(missing_ok=True))
        ns_f["cfg"] = replace(ns_f["cfg"], local_checkpoint_root=str(local_ssd_root), benchmark_workload_fixture_eval=False)
        ns_f["TOK_GATE_REPORT"] = self._run_mocked_t4_gate(ns_f["BUNDLE_DIR"], ns_f["cfg"], ns_f["TOK_GATE_PREFLIGHT_PATH"])

        fh_calls: list[str] = []
        fh_base = self._mock_t4_train_eval_fn(fh_calls)
        interrupted_once = {"done": False}

        def _interrupt_unit1(**kw):
            if not interrupted_once["done"]:
                interrupted_once["done"] = True
                raise KeyboardInterrupt("Simulated Colab preemption mid-unit")
            return fh_base(**kw)

        with (
            patch("bangla_smish.post_stage_b_runner.collect_hardware_fingerprint", return_value=self._mock_t4_hw()),
            patch(
                "bangla_smish.post_stage_b_runner.run_post_stage_b_pipeline",
                side_effect=lambda **kw: real_run_pipeline(**kw, train_eval_fn=_interrupt_unit1, stop_after_units=1),
            ),
        ):
            with self.assertRaises(KeyboardInterrupt):
                _exec_cell(ns_f, 6)

        art_fh = ns_f["ARTIFACTS_DIR"]
        uid0 = "exp_D__csebuetnlp__banglabert__canon_masked__fold_0"
        st_interrupted = json.loads((art_fh / "run_state.json").read_text(encoding="utf-8"))
        self.assertEqual(st_interrupted["attempts"][uid0][0]["status"], "running")
        self.assertNotIn(uid0, st_interrupted["committed_units"])

        # Scenario H: Resume incomplete unit via Cell 6 (with stale lease simulated & authorized takeover)
        register_single_writer_lease(art_fh, writer_id="preempted_t4_vm")
        ns_h: dict = {}
        _run_cells_1_to_3(
            ns_h,
            run_id="nb-life-fh-01",
            takeover_kwargs={
                "ALLOW_STALE_WRITER_TAKEOVER": True,
                "EXPECTED_STALE_WRITER_IDS": "preempted_t4_vm",
                "STALE_TAKEOVER_REASON": "Owner confirmed VM preempted mid-unit 1",
            },
        )
        with (
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch("bangla_smish.post_stage_b_tokenizer_gate._default_hf_tokenizer_loader", side_effect=lambda m, r: _MockTokenizer(r, force_collision=False)),
            patch("bangla_smish.post_stage_b_tokenizer_gate.prepare_tokenizer_texts", side_effect=lambda t, m: t),
        ):
            _exec_cell(ns_h, 5)
        ns_h["cfg"] = replace(ns_h["cfg"], local_checkpoint_root=str(local_ssd_root), benchmark_workload_fixture_eval=False)
        ns_h["TOK_GATE_REPORT"] = self._run_mocked_t4_gate(ns_h["BUNDLE_DIR"], ns_h["cfg"], ns_h["TOK_GATE_PREFLIGHT_PATH"])
        with (
            patch("bangla_smish.post_stage_b_runner.collect_hardware_fingerprint", return_value=self._mock_t4_hw()),
            patch(
                "bangla_smish.post_stage_b_runner.run_post_stage_b_pipeline",
                side_effect=lambda **kw: real_run_pipeline(**kw, train_eval_fn=_interrupt_unit1, stop_after_units=1),
            ),
        ):
            out_h = _exec_cell(ns_h, 6)
        self.assertIn("Completed units total: 1 / 5", out_h)
        st_resumed = json.loads((art_fh / "run_state.json").read_text(encoding="utf-8"))
        self.assertEqual([a["status"] for a in st_resumed["attempts"][uid0]], ["interrupted", "completed"])
        self.assertEqual(st_resumed["committed_units"][uid0]["attempt"], 2)

        # Verify distinct ARTIFACT_RUN_IDs get isolated local SSD checkpoint bases
        self.assertNotEqual(
            _local_checkpoint_base(art_main, ns_main["cfg"]),
            _local_checkpoint_base(art_fh, ns_h["cfg"]),
        )

        # --- Scenario K: Rejection of corrupted or unidentified artifact directories ---
        unid_art = staged_kit / "artifacts" / "t4_benchmark" / "nb-life-k-unid"
        (unid_art / "exports").mkdir(parents=True)
        (unid_art / "exports" / "stray.json").write_text("{}", encoding="utf-8")
        with patch("bangla_smish.post_stage_b_runner.collect_hardware_fingerprint", return_value=self._mock_t4_hw()):
            with self.assertRaisesRegex(ValueError, "unidentified run"):
                real_run_pipeline(
                    bundle_dir=ns_main["BUNDLE_DIR"],
                    artifacts_dir=unid_art,
                    cfg=ns_main["cfg"],
                    train_eval_fn=self._mock_t4_train_eval_fn([]),
                    tokenizer_gate_report=ns_main["TOK_GATE_REPORT"],
                )

            # --- Scenario L: Rejection of mismatched manifest/config/source hashes ---
            state_before_l = (art_main / "run_state.json").read_bytes()
            # 1) Config hash mismatch (changed epochs)
            bad_cfg = replace(ns_main["cfg"], epochs=2)
            with self.assertRaisesRegex(ValueError, "incompatible configuration, manifest, or code"):
                real_run_pipeline(
                    bundle_dir=ns_main["BUNDLE_DIR"],
                    artifacts_dir=art_main,
                    cfg=bad_cfg,
                    train_eval_fn=self._mock_t4_train_eval_fn([]),
                    tokenizer_gate_report=ns_main["TOK_GATE_REPORT"],
                )
            # 2) Source hash mismatch in run_state.json
            src_mismatch_art = self.root / "nb_life_l_src"
            shutil.copytree(art_main, src_mismatch_art)
            st_src = json.loads((src_mismatch_art / "run_state.json").read_text(encoding="utf-8"))
            st_src["identity"]["source_sha256"]["post_stage_b_runner.py"] = "0" * 64
            (src_mismatch_art / "run_state.json").write_text(json.dumps(st_src), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incompatible configuration, manifest, or code"):
                real_run_pipeline(
                    bundle_dir=ns_main["BUNDLE_DIR"],
                    artifacts_dir=src_mismatch_art,
                    cfg=ns_main["cfg"],
                    train_eval_fn=self._mock_t4_train_eval_fn([]),
                    tokenizer_gate_report=ns_main["TOK_GATE_REPORT"],
                )
            # 3) Manifest mismatch (tampered copy of bundle)
            bundle_copy = self.root / "bundle_l_tampered"
            shutil.copytree(v2_dir, bundle_copy)
            (bundle_copy / "dataset_manifest.json").write_bytes((bundle_copy / "dataset_manifest.json").read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "incompatible configuration, manifest, or code"):
                real_run_pipeline(
                    bundle_dir=bundle_copy,
                    artifacts_dir=art_main,
                    cfg=ns_main["cfg"],
                    train_eval_fn=self._mock_t4_train_eval_fn([]),
                    tokenizer_gate_report=ns_main["TOK_GATE_REPORT"],
                )
        self.assertEqual((art_main / "run_state.json").read_bytes(), state_before_l)
        # Canonical notebook remained 100% immutable throughout entire A-L lifecycle
        self.assertEqual(sha256_file(canonical_nb_path), canonical_nb_sha_before)
        self.assertEqual(verify_portable_colab_kit(staged_kit)["verified_runtime_file_count"], 34)

    # ------------------------------------------------------------------
    # 2D. v5 Independent-Review Remediation Tests (MOCKED ONLY — NOT REAL HF/GPU TRAINING)
    # ------------------------------------------------------------------
    def _build_mock_hf_modules(
        self,
        *,
        tok_commit: str | None,
        model_commit: str | None,
        global_step: int | None = 12,
        train_exc: Exception | None = None,
        recorded_calls: dict | None = None,
    ):
        """Construct lightweight mocked torch/datasets/transformers modules for CPU-only unit testing."""
        calls = recorded_calls if recorded_calls is not None else {}
        calls.setdefault("empty_cache_calls", 0)
        calls.setdefault("tok_revisions", [])
        calls.setdefault("model_revisions", [])

        class _FakeTensor(np.ndarray):
            pass

        def _fake_softmax(arr, dim=-1):
            x = np.asarray(arr, dtype=np.float32)
            shifted = x - x.max(axis=dim, keepdims=True)
            ex = np.exp(shifted)
            probs = ex / ex.sum(axis=dim, keepdims=True)
            return types.SimpleNamespace(numpy=lambda: probs)

        fake_torch = types.SimpleNamespace(
            float32=np.float32,
            manual_seed=lambda s: None,
            tensor=lambda data, dtype=None: np.asarray(data, dtype=dtype or np.float32),
            softmax=_fake_softmax,
            version=types.SimpleNamespace(cuda="12.4"),
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                is_bf16_supported=lambda **kw: False,
                manual_seed_all=lambda s: None,
                reset_peak_memory_stats=lambda: None,
                max_memory_allocated=lambda: 1024,
                max_memory_reserved=lambda: 2048,
                empty_cache=lambda: calls.__setitem__("empty_cache_calls", calls["empty_cache_calls"] + 1),
            ),
        )

        class _FakeDataset:
            def __init__(self, payload: dict):
                self._payload = payload
                self._n = len(next(iter(payload.values())))

            @classmethod
            def from_dict(cls, payload: dict):
                return cls(payload)

            def map(self, fn, batched=True, remove_columns=None):
                fn(self._payload)
                return self

            def __len__(self):
                return self._n

        fake_datasets = types.SimpleNamespace(Dataset=_FakeDataset)

        class _FakeAutoTokenizer:
            @classmethod
            def from_pretrained(cls, model_name, revision=None):
                calls["tok_revisions"].append((model_name, revision))
                tok = _MockTokenizer(tok_commit or "")
                tok.save_pretrained = lambda d: Path(d).joinpath("tokenizer_config.json").write_text("{}", encoding="utf-8")
                return tok

        class _FakeAutoModel:
            @classmethod
            def from_pretrained(cls, model_name, num_labels=3, revision=None):
                calls["model_revisions"].append((model_name, revision))
                cfg_obj = types.SimpleNamespace(_commit_hash=model_commit)
                return types.SimpleNamespace(config=cfg_obj)

        class _FakeTrainingArguments:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        class _FakeTrainerCallback:
            pass

        class _FakeTrainer:
            def __init__(self, model=None, args=None, train_dataset=None, eval_dataset=None, **kw):
                self.model = model
                self.args = args
                self.state = types.SimpleNamespace(
                    global_step=global_step,
                    best_model_checkpoint=str(Path(args.output_dir) / "checkpoint-12"),
                    log_history=[],
                )

            def train(self, resume_from_checkpoint=None):
                if train_exc is not None:
                    raise train_exc

            def predict(self, ds):
                n = len(ds)
                logits = np.tile(np.array([[2.0, 0.0, 0.0]], dtype=np.float32), (n, 1))
                return types.SimpleNamespace(predictions=logits)

            def save_model(self, out_dir):
                p = Path(out_dir)
                (p / "config.json").write_text("{}", encoding="utf-8")
                (p / "model.safetensors").write_bytes(b"weights")
                (p / "vocab.txt").write_text("a\n", encoding="utf-8")

        fake_transformers = types.SimpleNamespace(
            AutoModelForSequenceClassification=_FakeAutoModel,
            AutoTokenizer=_FakeAutoTokenizer,
            DataCollatorWithPadding=lambda tokenizer=None: None,
            Trainer=_FakeTrainer,
            TrainerCallback=_FakeTrainerCallback,
            TrainingArguments=_FakeTrainingArguments,
            set_seed=lambda s: None,
        )
        return {"torch": fake_torch, "datasets": fake_datasets, "transformers": fake_transformers}, calls

    def test_v5_review_model_pinning_cuda_cleanup_and_synthetic_heldout_guards(self) -> None:
        """Mocked unit test (NOT real HF Trainer/GPU): verifies model pinning, CUDA cleanup, held-out & fixture guards."""
        import pandas as pd

        self.assertFalse(SYNTHETIC_WORKLOAD_FIXTURE_ENABLED)
        v2_dir = ROOT / "data" / "processed" / "post_stage_b_bundle_v2"
        real_df = pd.read_parquet(v2_dir / "real_train_pool.parquet").head(8).reset_index(drop=True)
        sd_val_df = pd.read_parquet(v2_dir / "eval_smishdetect_validation.parquet").head(6).reset_index(drop=True)
        pinned_xlm = PINNED_TOKENIZER_REVISIONS["xlm-roberta-base"]

        cfg_pinned = PostStageBConfig(
            run_name="v5_pin_test",
            run_mode="gpu_smoke_test",
            experiments=("A",),
            models=("xlm-roberta-base",),
            text_views=("canon_masked",),
            folds=(0,),
            tokenizer_revisions={"xlm-roberta-base": pinned_xlm},
            smoke_max_train_samples=8,
            smoke_max_val_samples=8,
            smoke_max_eval_samples=6,
            run_heldout_eval=False,
            run_adversarial_eval=False,
            run_counterfactual_eval=False,
        )

        # 1. Model revision pinned and verified on happy path + CUDA cleanup called
        mods, calls = self._build_mock_hf_modules(tok_commit=pinned_xlm, model_commit=pinned_xlm, global_step=9)
        with patch.dict(sys.modules, mods):
            out = hf_train_and_predict_fold(
                train_df=real_df,
                val_df=real_df,
                heldout_sets={},
                model_name="xlm-roberta-base",
                text_view="canon_masked",
                checkpoint_dir=self.root / "ckpt_ok",
                cfg=cfg_pinned,
                class_weights=None,
                seed=42,
            )
        self.assertEqual(calls["tok_revisions"], [("xlm-roberta-base", pinned_xlm)])
        self.assertEqual(calls["model_revisions"], [("xlm-roberta-base", pinned_xlm)])
        self.assertEqual(out["model_revision"], pinned_xlm)
        self.assertEqual(out["tokenizer_revision"], pinned_xlm)
        self.assertEqual(out["actual_global_step"], 9)
        self.assertGreaterEqual(calls["empty_cache_calls"], 1)

        # 2. Missing or mismatched model _commit_hash fails closed AND triggers CUDA cleanup in finally
        for label, bad_commit, regex in (
            ("missing_model_commit", None, "did not expose model.config._commit_hash"),
            ("mismatched_model_commit", "0" * 40, "loaded model revision"),
        ):
            with self.subTest(case=label):
                mods_bad, calls_bad = self._build_mock_hf_modules(tok_commit=pinned_xlm, model_commit=bad_commit)
                with patch.dict(sys.modules, mods_bad):
                    with self.assertRaisesRegex(RuntimeError, regex):
                        hf_train_and_predict_fold(
                            train_df=real_df,
                            val_df=real_df,
                            heldout_sets={},
                            model_name="xlm-roberta-base",
                            text_view="canon_masked",
                            checkpoint_dir=self.root / f"ckpt_{label}",
                            cfg=cfg_pinned,
                            class_weights=None,
                            seed=42,
                        )
                self.assertGreaterEqual(calls_bad["empty_cache_calls"], 1)

        # 3. CUDA cleanup executes on training exception (e.g., simulated CUDA OOM)
        mods_oom, calls_oom = self._build_mock_hf_modules(
            tok_commit=pinned_xlm,
            model_commit=pinned_xlm,
            train_exc=RuntimeError("Simulated CUDA OOM"),
        )
        with patch.dict(sys.modules, mods_oom):
            with self.assertRaisesRegex(RuntimeError, "Simulated CUDA OOM"):
                hf_train_and_predict_fold(
                    train_df=real_df,
                    val_df=real_df,
                    heldout_sets={},
                    model_name="xlm-roberta-base",
                    text_view="canon_masked",
                    checkpoint_dir=self.root / "ckpt_oom",
                    cfg=cfg_pinned,
                    class_weights=None,
                    seed=42,
                )
        self.assertGreaterEqual(calls_oom["empty_cache_calls"], 1)

        # 4. Held-out/OOD guard rejects held-out/OOD frames when run_heldout_eval=False
        mods_ho, _ = self._build_mock_hf_modules(tok_commit=pinned_xlm, model_commit=pinned_xlm)
        with patch.dict(sys.modules, mods_ho):
            with self.assertRaisesRegex(RuntimeError, "held-out/OOD dataset"):
                hf_train_and_predict_fold(
                    train_df=real_df,
                    val_df=real_df,
                    heldout_sets={"smishdetect_test": sd_val_df},
                    model_name="xlm-roberta-base",
                    text_view="canon_masked",
                    checkpoint_dir=self.root / "ckpt_ho",
                    cfg=cfg_pinned,
                    class_weights=None,
                    seed=42,
                )

        # 5. Synthetic workload fixture guard blocks execution fail-closed (never generates synthetic text)
        t4_cfg = load_config(ROOT / "experiments" / "colab" / "configs" / "post_stage_b_stage4_t4_benchmark_config.json")
        mods_fix, calls_fix = self._build_mock_hf_modules(tok_commit=pinned_xlm, model_commit=pinned_xlm)
        with (
            patch.dict(sys.modules, mods_fix),
            patch("bangla_smish.post_stage_b_runner._build_t4_benchmark_workload_fixture") as spy_fixture,
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic workload fixture execution is blocked"):
                hf_train_and_predict_fold(
                    train_df=real_df,
                    val_df=real_df,
                    heldout_sets={"smishdetect_validation": sd_val_df},
                    model_name="xlm-roberta-base",
                    text_view="canon_masked",
                    checkpoint_dir=self.root / "ckpt_fix",
                    cfg=replace(t4_cfg, run_adversarial_eval=False, run_counterfactual_eval=False),
                    class_weights=None,
                    seed=42,
                )
            spy_fixture.assert_not_called()
        # Synthetic workload fixture guard now fires BEFORE tokenizer/model initialization (defense-in-depth).
        self.assertEqual(calls_fix["tok_revisions"], [])
        self.assertEqual(calls_fix["model_revisions"], [])

        # 6. Pipeline-level held-out/OOD read guard: run_heldout_eval=False NEVER reads test/OOD parquets
        read_paths: list[str] = []
        orig_read_parquet = pd.read_parquet
        seen_eval_keys: list[set[str]] = []

        def _spy_read_parquet(path, *a, **kw):
            read_paths.append(str(path))
            return orig_read_parquet(path, *a, **kw)

        t4_no_fix = replace(t4_cfg, benchmark_workload_fixture_eval=False)
        preflight = self.root / "t4_ho_gate.json"
        tok_gate = self._run_mocked_t4_gate(v2_dir, t4_no_fix, preflight)

        def _spy_train(**kw):
            seen_eval_keys.append(set(kw["heldout_sets"].keys()))
            return self._mock_t4_train_eval_fn([])(**kw)

        art_ho = self.root / "t4_no_heldout_reads"
        with (
            patch("bangla_smish.post_stage_b_runner.collect_hardware_fingerprint", return_value=self._mock_t4_hw()),
            patch("bangla_smish.post_stage_b_runner.pd.read_parquet", side_effect=_spy_read_parquet),
            patch("bangla_smish.post_stage_b_data.pd.read_parquet", side_effect=_spy_read_parquet),
        ):
            run_post_stage_b_pipeline(
                bundle_dir=v2_dir,
                artifacts_dir=art_ho,
                cfg=t4_no_fix,
                train_eval_fn=_spy_train,
                stop_after_units=1,
                tokenizer_gate_report=tok_gate,
            )
            aggregate_post_stage_b_results(v2_dir, art_ho, t4_no_fix, train_eval_fn=_spy_train)
        self.assertFalse(any("eval_smishdetect_test.parquet" in p or "eval_banglaphish_ood.parquet" in p for p in read_paths))
        self.assertEqual(seen_eval_keys, [{"counterfactual_smish_probe"}])

    def test_v5_review_actual_steps_env_identity_commit_ordering_and_retention_modes(self) -> None:
        """Mocked unit test (NOT real HF Trainer/GPU): verifies step integrity, env identity, commit/event order & retention."""
        import pandas as pd
        from bangla_smish.post_stage_b_data import sha256_file

        v2_dir = ROOT / "data" / "processed" / "post_stage_b_bundle_v2"
        base_cfg = PostStageBConfig(
            run_name="v5_integrity_test",
            run_mode="gpu_smoke_test",
            experiments=("A",),
            models=("xlm-roberta-base",),
            text_views=("canon_masked",),
            folds=(0,),
            smoke_max_train_samples=16,
            smoke_max_val_samples=8,
            smoke_max_eval_samples=8,
            run_heldout_eval=False,
            run_adversarial_eval=False,
            run_counterfactual_eval=False,
        )

        # 1. Missing or zero actual_global_step is NEVER replaced with expected_optimizer_steps
        for step_label, step_val in (("missing", None), ("zero", 0)):
            with self.subTest(actual_step_case=step_label):
                art_step = self.root / f"step_{step_label}"

                def _step_mock(**kw):
                    out = {
                        "train_seconds": 0.01,
                        "predictions": {"val_probs": np.eye(3)[kw["val_df"]["label_idx"].to_numpy(dtype=int)] * 0.94 + 0.02},
                    }
                    if step_val is not None:
                        out["actual_global_step"] = step_val
                    return out

                summary = run_post_stage_b_pipeline(
                    bundle_dir=v2_dir, artifacts_dir=art_step, cfg=base_cfg, train_eval_fn=_step_mock
                )
                row = summary["unit_rows"][0]
                self.assertGreater(row["expected_optimizer_steps"], 0)
                self.assertEqual(row["actual_global_step"], step_val)
                self.assertNotEqual(row["actual_global_step"], row["expected_optimizer_steps"])
                mjson = json.loads(
                    (art_step / "units" / row["unit_id"] / "attempts" / "1" / "metrics.json").read_text(encoding="utf-8")
                )
                self.assertEqual(mjson["expected_optimizer_steps"], row["expected_optimizer_steps"])
                self.assertEqual(mjson["actual_global_step"], step_val)
                rmeta = json.loads((art_step / "exports" / "retraining_metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(rmeta["units"][0]["actual_global_step"], step_val)

        # 2. Stable environment & hardware fingerprints in _run_identity (timestamp-free) reject package drift on resume
        fp = collect_stable_environment_fingerprint()
        hw = collect_hardware_fingerprint()
        self.assertNotIn("timestamp_utc", fp)
        self.assertIn("packages", fp)
        ident = _run_identity(v2_dir, base_cfg, hf_train_and_predict_fold)
        self.assertEqual(ident["schema_version"], 5)
        self.assertEqual(ident["environment_fingerprint"], fp)
        self.assertEqual(ident["hardware_fingerprint"], hw)

        art_env = self.root / "env_drift_art"
        cfg_2f = replace(base_cfg, folds=(0, 1))
        ok_mock = lambda **kw: {
            "train_seconds": 0.01,
            "actual_global_step": 3,
            "predictions": {"val_probs": np.eye(3)[kw["val_df"]["label_idx"].to_numpy(dtype=int)] * 0.94 + 0.02},
        }
        run_post_stage_b_pipeline(
            bundle_dir=v2_dir, artifacts_dir=art_env, cfg=cfg_2f, train_eval_fn=ok_mock, stop_after_units=1
        )
        state_bytes = (art_env / "run_state.json").read_bytes()
        drifted_fp = {**fp, "packages": {**fp["packages"], "transformers": "4.99.9"}}
        with patch("bangla_smish.post_stage_b_runner.collect_stable_environment_fingerprint", return_value=drifted_fp):
            with self.assertRaisesRegex(ValueError, "incompatible configuration, manifest, or code"):
                run_post_stage_b_pipeline(bundle_dir=v2_dir, artifacts_dir=art_env, cfg=cfg_2f, train_eval_fn=ok_mock)
        self.assertEqual((art_env / "run_state.json").read_bytes(), state_bytes)

        # Helper to simulate real hf_train_and_predict_fold contract (creating local ckpt, staging ckpt, best_model)
        def _make_fake_hf_runner(*, global_step: int | None = 3):
            def _fake_hf(**kw):
                ckpt_dir: Path = kw["checkpoint_dir"]
                cpath = _write_ckpt(ckpt_dir / "checkpoint-10", 10)
                atomic_write_json(cpath / _CHECKPOINT_COMMIT, _checkpoint_file_hashes(cpath))
                sync_local_checkpoint_to_staging(
                    local_ckpt_dir=cpath,
                    staging_unit_dir=kw["staging_unit_dir"],
                    events_dir=kw["events_dir"],
                    pointers_dir=kw["pointers_dir"],
                    uid=kw["uid"],
                    attempt=kw["attempt"],
                    config_hash=kw["config_hash"],
                    manifest_sha256=kw["manifest_sha256"],
                )
                bdir_out = kw.get("best_model_export_dir")
                if bdir_out is not None:
                    bdir_out.mkdir(parents=True, exist_ok=True)
                    (bdir_out / "config.json").write_text("{}", encoding="utf-8")
                    (bdir_out / "tokenizer_config.json").write_text("{}", encoding="utf-8")
                    (bdir_out / "model.safetensors").write_bytes(b"weights")
                    (bdir_out / "vocab.txt").write_text("tok\n", encoding="utf-8")
                return {
                    "train_seconds": 0.02,
                    "actual_global_step": global_step,
                    "best_trainer_checkpoint": str(cpath),
                    "model_revision": "rev_x",
                    "tokenizer_revision": "rev_x",
                    "predictions": {"val_probs": np.eye(3)[kw["val_df"]["label_idx"].to_numpy(dtype=int)] * 0.94 + 0.02},
                }

            return _fake_hf

        local_root = self.root / "local_ssd_ret"
        uid0 = "exp_A__xlm-roberta-base__canon_masked__fold_0"
        uid1 = "exp_A__xlm-roberta-base__canon_masked__fold_1"
        from bangla_smish.post_stage_b_tokenization import BANGLABERT_NORMALIZER_COMMIT
        valid_env_fp = {
            "python_version": "3.12.3",
            "torch_cuda_version": "12.4",
            "normalizer_commit": BANGLABERT_NORMALIZER_COMMIT,
            "packages": {dist: "1.2.3" for dist, _ in _IDENTITY_PACKAGE_DISTS},
        }

        # 3A. Patched hf_train_and_predict_fold rejects actual_global_step=0 fail-closed
        art_zero_hf = self.root / "hf_zero_step"
        with (
            patch("bangla_smish.post_stage_b_runner.collect_stable_environment_fingerprint", return_value=valid_env_fp),
            patch("bangla_smish.post_stage_b_runner.hf_train_and_predict_fold", side_effect=_make_fake_hf_runner(global_step=0)),
        ):
            with self.assertRaisesRegex(ValueError, "invalid actual_global_step=0"):
                run_post_stage_b_pipeline(bundle_dir=v2_dir, artifacts_dir=art_zero_hf, cfg=base_cfg)

        # 3B. Completion-event metrics_sha256 stays 100% consistent with final metrics.json even on post-commit prune error
        art_prune_err = self.root / "hf_prune_err"
        cfg_local = replace(base_cfg, local_checkpoint_root=str(local_root), keep_model_weights="none")
        with (
            patch("bangla_smish.post_stage_b_runner.collect_stable_environment_fingerprint", return_value=valid_env_fp),
            patch("bangla_smish.post_stage_b_runner.hf_train_and_predict_fold", side_effect=_make_fake_hf_runner()),
            patch("bangla_smish.post_stage_b_checkpointing.shutil.rmtree", side_effect=OSError("Drive FUSE busy")),
        ):
            run_post_stage_b_pipeline(bundle_dir=v2_dir, artifacts_dir=art_prune_err, cfg=cfg_local)
        metrics_file = art_prune_err / "units" / uid0 / "attempts" / "1" / "metrics.json"
        final_metrics_sha = sha256_file(metrics_file)
        st_pe = json.loads((art_prune_err / "run_state.json").read_text(encoding="utf-8"))
        self.assertEqual(st_pe["committed_units"][uid0]["metrics_sha256"], final_metrics_sha)
        self.assertFalse(st_pe["committed_units"][uid0]["training_checkpoints_pruned_post_commit"])
        self.assertEqual(len(st_pe["committed_units"][uid0]["post_commit_prune_errors"]), 1)
        commit_events = list((art_prune_err / "events").glob("*UNIT_COMMITTED*.json"))
        self.assertEqual(len(commit_events), 1)
        ev_doc = json.loads(commit_events[0].read_text(encoding="utf-8"))
        self.assertEqual(ev_doc["metrics_sha256"], final_metrics_sha)

        # 3C. Failure during _validate_unit preserves recovery checkpoints and emits NO UNIT_COMMITTED event
        art_val_fail = self.root / "hf_val_fail"
        with (
            patch("bangla_smish.post_stage_b_runner.collect_stable_environment_fingerprint", return_value=valid_env_fp),
            patch("bangla_smish.post_stage_b_runner.hf_train_and_predict_fold", side_effect=_make_fake_hf_runner()),
            patch("bangla_smish.post_stage_b_runner._validate_unit", side_effect=ValueError("Simulated validation failure")),
        ):
            with self.assertRaisesRegex(ValueError, "Simulated validation failure"):
                run_post_stage_b_pipeline(bundle_dir=v2_dir, artifacts_dir=art_val_fail, cfg=cfg_local)
        self.assertEqual(list((art_val_fail / "events").glob("*UNIT_COMMITTED*.json")), [])
        self.assertTrue(any((art_val_fail / "staging" / uid0).rglob("checkpoint-10")))
        self.assertTrue((_local_checkpoint_base(art_val_fail, cfg_local) / uid0 / "1" / "checkpoint-10").is_dir())

        # 3D. Failure between _validate_unit and durable run_state.json commit preserves recovery checkpoints & emits NO event
        art_commit_crash = self.root / "hf_commit_crash"
        from bangla_smish import post_stage_b_runner as runner_mod
        orig_atomic_write_json = runner_mod.atomic_write_json

        def _fail_on_commit_state(path: Path, payload):
            if path.name == "run_state.json" and isinstance(payload, dict) and uid0 in payload.get("committed_units", {}):
                raise OSError("Simulated crash during run_state.json commit")
            return orig_atomic_write_json(path, payload)

        with (
            patch("bangla_smish.post_stage_b_runner.collect_stable_environment_fingerprint", return_value=valid_env_fp),
            patch("bangla_smish.post_stage_b_runner.hf_train_and_predict_fold", side_effect=_make_fake_hf_runner()),
            patch("bangla_smish.post_stage_b_runner.atomic_write_json", side_effect=_fail_on_commit_state),
        ):
            with self.assertRaisesRegex(OSError, "Simulated crash during run_state.json commit"):
                run_post_stage_b_pipeline(bundle_dir=v2_dir, artifacts_dir=art_commit_crash, cfg=cfg_local)
        self.assertEqual(list((art_commit_crash / "events").glob("*UNIT_COMMITTED*.json")), [])
        self.assertTrue(any((art_commit_crash / "staging" / uid0).rglob("checkpoint-10")))
        self.assertTrue((_local_checkpoint_base(art_commit_crash, cfg_local) / uid0 / "1" / "checkpoint-10").is_dir())
        st_cc = json.loads((art_commit_crash / "run_state.json").read_text(encoding="utf-8"))
        self.assertNotIn(uid0, st_cc["committed_units"])

        # 4. All three weight-retention modes (none, best_fold0, all) with local_checkpoint_root set
        for mode in ("none", "best_fold0", "all"):
            with self.subTest(keep_model_weights=mode):
                art_ret = self.root / f"ret_mode_{mode}"
                cfg_ret = replace(cfg_2f, local_checkpoint_root=str(local_root), keep_model_weights=mode)
                fake_hf = _make_fake_hf_runner()
                with (
                    patch("bangla_smish.post_stage_b_runner.collect_stable_environment_fingerprint", return_value=valid_env_fp),
                    patch("bangla_smish.post_stage_b_runner.hf_train_and_predict_fold", side_effect=fake_hf),
                ):
                    part = run_post_stage_b_pipeline(
                        bundle_dir=v2_dir, artifacts_dir=art_ret, cfg=cfg_ret, stop_after_units=1
                    )
                    self.assertEqual(part["completed_unit_count"], 1)
                    # Resume fold 1 (verifying fold 0 committed unit via _read_committed_unit!)
                    done = run_post_stage_b_pipeline(bundle_dir=v2_dir, artifacts_dir=art_ret, cfg=cfg_ret)
                    self.assertEqual(done["completed_unit_count"], 2)
                    agg = aggregate_post_stage_b_results(v2_dir, art_ret, cfg_ret)
                    self.assertEqual(agg["completed_unit_count"], 2)

                st_ret = json.loads((art_ret / "run_state.json").read_text(encoding="utf-8"))
                c0 = st_ret["committed_units"][uid0]
                c1 = st_ret["committed_units"][uid1]
                local_base = _local_checkpoint_base(art_ret, cfg_ret)
                if mode == "none":
                    self.assertNotIn("best_model_sha256", c0)
                    self.assertNotIn("trainer_checkpoint", c0)
                    self.assertTrue(c0["training_checkpoints_pruned_post_commit"])
                    self.assertTrue(c1["training_checkpoints_pruned_post_commit"])
                    self.assertFalse((local_base / uid0).exists())
                    self.assertFalse((local_base / uid1).exists())
                elif mode == "best_fold0":
                    self.assertIn("best_model_sha256", c0)
                    self.assertIn("trainer_checkpoint", c0)
                    self.assertTrue((local_base / uid0 / "1" / "checkpoint-10").is_dir())
                    self.assertNotIn("best_model_sha256", c1)
                    self.assertNotIn("trainer_checkpoint", c1)
                    self.assertTrue(c1["training_checkpoints_pruned_post_commit"])
                    self.assertFalse((local_base / uid1).exists())
                else:  # "all"
                    self.assertIn("best_model_sha256", c0)
                    self.assertIn("trainer_checkpoint", c0)
                    self.assertIn("best_model_sha256", c1)
                    self.assertIn("trainer_checkpoint", c1)
                    self.assertTrue((local_base / uid0 / "1" / "checkpoint-10").is_dir())
                    self.assertTrue((local_base / uid1 / "1" / "checkpoint-10").is_dir())

    # ------------------------------------------------------------------
    # 2E. Stage 4 v7 Targeted Remediation Tests (CPU-MOCKED ONLY — NOT REAL HF/GPU)
    # ------------------------------------------------------------------
    def test_v7_pre_lease_synthetic_workload_guard_frozen_t4_config_untouched(self) -> None:
        """Mocked CPU test: incompatible frozen T4 config fails closed BEFORE lease/state/model/training."""
        from bangla_smish.post_stage_b_data import sha256_file

        v2_dir = ROOT / "data" / "processed" / "post_stage_b_bundle_v2"
        frozen_cfg_path = ROOT / "experiments" / "colab" / "configs" / "post_stage_b_stage4_t4_benchmark_config.json"
        frozen_sha_before = sha256_file(frozen_cfg_path)
        frozen_t4_cfg = load_config(frozen_cfg_path)
        self.assertTrue(frozen_t4_cfg.benchmark_workload_fixture_eval)
        self.assertFalse(SYNTHETIC_WORKLOAD_FIXTURE_ENABLED)

        preflight = self.root / "v7_frozen_t4_gate.json"
        tok_gate = self._run_mocked_t4_gate(v2_dir, frozen_t4_cfg, preflight)

        # 1. With custom train_eval_fn spy + valid mock T4 hardware
        art_spy = self.root / "v7_pre_lease_spy_art"
        train_calls: list[str] = []
        with (
            patch("bangla_smish.post_stage_b_runner.collect_hardware_fingerprint", return_value=self._mock_t4_hw()),
            patch("bangla_smish.post_stage_b_runner.register_single_writer_lease") as spy_lease,
        ):
            with self.assertRaisesRegex(RuntimeError, "Fail closed before run initialization"):
                run_post_stage_b_pipeline(
                    bundle_dir=v2_dir,
                    artifacts_dir=art_spy,
                    cfg=frozen_t4_cfg,
                    train_eval_fn=self._mock_t4_train_eval_fn(train_calls),
                    tokenizer_gate_report=tok_gate,
                )
            spy_lease.assert_not_called()
        self.assertEqual(train_calls, [])
        self.assertFalse(art_spy.exists())
        self.assertFalse((art_spy / "run_state.json").exists())
        self.assertFalse((art_spy / "leases").exists())

        # 2. With default hf_train_and_predict_fold spied (zero model initialization or training)
        art_hf_spy = self.root / "v7_pre_lease_hf_spy_art"
        with (
            patch("bangla_smish.post_stage_b_runner.collect_hardware_fingerprint", return_value=self._mock_t4_hw()),
            patch("bangla_smish.post_stage_b_runner.register_single_writer_lease") as spy_lease_hf,
            patch("bangla_smish.post_stage_b_runner.hf_train_and_predict_fold") as spy_hf,
        ):
            with self.assertRaisesRegex(RuntimeError, "Fail closed before run initialization"):
                run_post_stage_b_pipeline(
                    bundle_dir=v2_dir,
                    artifacts_dir=art_hf_spy,
                    cfg=frozen_t4_cfg,
                    tokenizer_gate_report=tok_gate,
                )
            spy_lease_hf.assert_not_called()
            spy_hf.assert_not_called()
        self.assertFalse(art_hf_spy.exists())
        self.assertEqual(sha256_file(frozen_cfg_path), frozen_sha_before)

    def test_v7_strict_t4_hardware_verification_notebook_runner_and_driver_resume(self) -> None:
        """Mocked CPU test (NOT real GPU): verifies T4 hardware enforcement, strict driver resume, and smoke preservation."""
        v2_dir = ROOT / "data" / "processed" / "post_stage_b_bundle_v2"
        t4_no_fix = replace(
            load_config(ROOT / "experiments" / "colab" / "configs" / "post_stage_b_stage4_t4_benchmark_config.json"),
            benchmark_workload_fixture_eval=False,
        )
        preflight = self.root / "v7_hw_gate.json"
        tok_gate = self._run_mocked_t4_gate(v2_dir, t4_no_fix, preflight)

        # 1. Runner pre-lease hardware gate rejects non-T4, multi-GPU, missing CUDA, wrong CC, and missing driver
        bad_hw_cases = [
            ("no_cuda", {**self._mock_t4_hw(), "cuda_available": False, "gpu_count": 0, "gpu_name": None}, "cuda_available=False"),
            ("multi_gpu", {**self._mock_t4_hw(), "gpu_count": 2}, "exactly 1 Tesla T4"),
            ("wrong_gpu_l4", {**self._mock_t4_hw(), "gpu_name": "NVIDIA L4", "compute_capability": "8.9"}, "Tesla T4 GPU"),
            ("wrong_cc", {**self._mock_t4_hw(), "compute_capability": "8.0"}, "compute capability 7.5"),
            ("missing_driver", {**self._mock_t4_hw(), "nvidia_driver_version": None}, "NVIDIA driver version"),
        ]
        for label, bad_hw, regex in bad_hw_cases:
            with self.subTest(runner_hw_case=label):
                art_bad_hw = self.root / f"v7_bad_hw_{label}"
                calls_bad_hw: list[str] = []
                with (
                    patch("bangla_smish.post_stage_b_runner.collect_hardware_fingerprint", return_value=bad_hw),
                    patch("bangla_smish.post_stage_b_runner.register_single_writer_lease") as spy_lease,
                ):
                    with self.assertRaisesRegex(RuntimeError, regex):
                        run_post_stage_b_pipeline(
                            bundle_dir=v2_dir,
                            artifacts_dir=art_bad_hw,
                            cfg=t4_no_fix,
                            train_eval_fn=self._mock_t4_train_eval_fn(calls_bad_hw),
                            tokenizer_gate_report=tok_gate,
                        )
                    spy_lease.assert_not_called()
                self.assertEqual(calls_bad_hw, [])
                self.assertFalse(art_bad_hw.exists())

        # 2. Decision 3 (STRICT): exact nvidia_driver_version change rejects resume WITHOUT mutating run_state.json
        art_drv = self.root / "v7_driver_strict_resume"
        calls_drv: list[str] = []
        with patch(
            "bangla_smish.post_stage_b_runner.collect_hardware_fingerprint",
            return_value=self._mock_t4_hw(driver="550.54.15"),
        ):
            res1 = run_post_stage_b_pipeline(
                bundle_dir=v2_dir,
                artifacts_dir=art_drv,
                cfg=t4_no_fix,
                train_eval_fn=self._mock_t4_train_eval_fn(calls_drv),
                stop_after_units=1,
                tokenizer_gate_report=tok_gate,
            )
        self.assertEqual(res1["completed_unit_count"], 1)
        state_path = art_drv / "run_state.json"
        state_before = state_path.read_bytes()
        state_obj = json.loads(state_before.decode("utf-8"))
        self.assertEqual(state_obj["identity"]["hardware_fingerprint"]["nvidia_driver_version"], "550.54.15")

        with patch(
            "bangla_smish.post_stage_b_runner.collect_hardware_fingerprint",
            return_value=self._mock_t4_hw(driver="550.90.07"),
        ):
            with self.assertRaisesRegex(ValueError, "incompatible configuration, manifest, or code"):
                run_post_stage_b_pipeline(
                    bundle_dir=v2_dir,
                    artifacts_dir=art_drv,
                    cfg=t4_no_fix,
                    train_eval_fn=self._mock_t4_train_eval_fn(calls_drv),
                    tokenizer_gate_report=tok_gate,
                )
        self.assertEqual(state_path.read_bytes(), state_before)
        self.assertEqual(len(calls_drv), 1)

        # 3. Notebook Cell 2 T4 enforcement vs preserved gpu_smoke_test behavior
        nb = nbformat.read(ROOT / "experiments" / "colab" / "post_stage_b_stage4_experiments.ipynb", as_version=4)
        cell2_src = nb.cells[2].source

        def _run_cell2(*, run_mode: str, cuda_ok: bool, gpu_count: int, gpu_name: str, cc: tuple[int, int], driver: str | None):
            mock_torch = types.SimpleNamespace(
                __version__="2.6.0+cu124",
                version=types.SimpleNamespace(cuda="12.4"),
                cuda=types.SimpleNamespace(
                    is_available=lambda: cuda_ok,
                    device_count=lambda: gpu_count,
                    get_device_name=lambda idx: gpu_name,
                    get_device_capability=lambda idx: cc,
                    get_device_properties=lambda idx: types.SimpleNamespace(total_memory=15 * (1024**3)),
                ),
            )
            ns = {"RUN_MODE": run_mode, "REQUIRE_GPU": True}
            with (
                patch.dict(sys.modules, {"torch": mock_torch}),
                patch("shutil.which", return_value="/usr/bin/nvidia-smi" if driver else None),
                patch("subprocess.check_output", return_value=f"{driver}\n" if driver else ""),
            ):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    exec(compile(cell2_src, "cell_2", "exec"), ns)
            return ns, buf.getvalue()

        # Valid Tesla T4 passes Cell 2 in t4_benchmark mode
        ns_ok, out_ok = _run_cell2(
            run_mode="t4_benchmark", cuda_ok=True, gpu_count=1, gpu_name="Tesla T4", cc=(7, 5), driver="550.54.15"
        )
        self.assertEqual(ns_ok["HARDWARE_PREFLIGHT"]["nvidia_driver_version"], "550.54.15")
        self.assertEqual(ns_ok["HARDWARE_PREFLIGHT"]["compute_capability"], "7.5")
        self.assertIn("Verified Tesla T4 benchmark hardware", out_ok)

        # Non-T4 GPU fails Cell 2 in t4_benchmark mode
        with self.assertRaisesRegex(RuntimeError, "requires a Tesla T4 GPU"):
            _run_cell2(run_mode="t4_benchmark", cuda_ok=True, gpu_count=1, gpu_name="NVIDIA L4", cc=(8, 9), driver="550.54.15")
        with self.assertRaisesRegex(RuntimeError, "exactly 1 Tesla T4"):
            _run_cell2(run_mode="t4_benchmark", cuda_ok=True, gpu_count=2, gpu_name="Tesla T4", cc=(7, 5), driver="550.54.15")
        with self.assertRaisesRegex(RuntimeError, "compute capability 7.5"):
            _run_cell2(run_mode="t4_benchmark", cuda_ok=True, gpu_count=1, gpu_name="Tesla T4", cc=(8, 0), driver="550.54.15")
        with self.assertRaisesRegex(RuntimeError, "verified NVIDIA driver version"):
            _run_cell2(run_mode="t4_benchmark", cuda_ok=True, gpu_count=1, gpu_name="Tesla T4", cc=(7, 5), driver=None)

        # Normal gpu_smoke_test still passes on non-T4 GPU (L4) without T4 restriction!
        ns_smoke, _ = _run_cell2(
            run_mode="gpu_smoke_test", cuda_ok=True, gpu_count=1, gpu_name="NVIDIA L4", cc=(8, 9), driver="550.54.15"
        )
        self.assertEqual(ns_smoke["HARDWARE_PREFLIGHT"]["gpu_name"], "NVIDIA L4")

    def test_v7_idempotent_completion_event_reconciliation_and_fault_injection(self) -> None:
        """Mocked CPU fault-injection tests for completion-event integrity and post-commit reconciliation."""
        from bangla_smish.post_stage_b_checkpointing import commit_unit_completion_event
        from bangla_smish.post_stage_b_data import sha256_file

        # --- Part A: Direct unit tests of ensure_unit_completion_event ---
        ev_dir = self.root / "direct_events"
        res_create = ensure_unit_completion_event(
            events_dir=ev_dir, uid="u1", attempt=1,
            config_hash="cfg_a", manifest_sha256="man_a", metrics_sha256="met_a", preds_sha256="prd_a",
        )
        self.assertTrue(res_create["created"])
        self.assertEqual(len(list(ev_dir.glob("*UNIT_COMMITTED*.json"))), 1)

        # Idempotent match on second call -> created=False, still 1 event
        res_reuse = ensure_unit_completion_event(
            events_dir=ev_dir, uid="u1", attempt=1,
            config_hash="cfg_a", manifest_sha256="man_a", metrics_sha256="met_a", preds_sha256="prd_a",
        )
        self.assertFalse(res_reuse["created"])
        self.assertEqual(res_reuse["event_path"], res_create["event_path"])
        self.assertEqual(len(list(ev_dir.glob("*UNIT_COMMITTED*.json"))), 1)

        # Mismatched config/manifest/metrics/preds hashes rejected fail-closed
        for field_name, overrides in (
            ("config_hash", {"config_hash": "cfg_WRONG"}),
            ("manifest_sha256", {"manifest_sha256": "man_WRONG"}),
            ("metrics_sha256", {"metrics_sha256": "met_WRONG"}),
            ("preds_sha256", {"preds_sha256": "prd_WRONG"}),
        ):
            with self.subTest(hash_mismatch=field_name):
                kwargs = dict(
                    events_dir=ev_dir, uid="u1", attempt=1,
                    config_hash="cfg_a", manifest_sha256="man_a", metrics_sha256="met_a", preds_sha256="prd_a",
                )
                kwargs.update(overrides)
                with self.assertRaisesRegex(ValueError, "does not match authoritative commit hashes"):
                    ensure_unit_completion_event(**kwargs)

        # Duplicate matching UNIT_COMMITTED events rejected fail-closed
        commit_unit_completion_event(
            events_dir=ev_dir, uid="u1", attempt=1,
            config_hash="cfg_a", manifest_sha256="man_a", metrics_sha256="met_a", preds_sha256="prd_a",
        )
        with self.assertRaisesRegex(ValueError, "duplicate UNIT_COMMITTED events"):
            ensure_unit_completion_event(
                events_dir=ev_dir, uid="u1", attempt=1,
                config_hash="cfg_a", manifest_sha256="man_a", metrics_sha256="met_a", preds_sha256="prd_a",
            )

        # --- Part B: Pipeline-level fault injection across all 5 interruption/reconciliation cases ---
        v2_dir = ROOT / "data" / "processed" / "post_stage_b_bundle_v2"
        local_root = self.root / "v7_reconcile_ssd"
        cfg_rec = PostStageBConfig(
            run_name="v7_reconcile_test",
            run_mode="gpu_smoke_test",
            experiments=("A",),
            models=("xlm-roberta-base",),
            text_views=("canon_masked",),
            folds=(0,),
            smoke_max_train_samples=16,
            smoke_max_val_samples=8,
            smoke_max_eval_samples=8,
            run_heldout_eval=False,
            run_adversarial_eval=False,
            run_counterfactual_eval=False,
            keep_model_weights="none",
            local_checkpoint_root=str(local_root),
        )
        uid0 = "exp_A__xlm-roberta-base__canon_masked__fold_0"
        train_invocations: list[int] = []

        def _fake_hf_runner(**kw):
            train_invocations.append(1)
            ckpt_dir: Path = kw["checkpoint_dir"]
            cpath = _write_ckpt(ckpt_dir / "checkpoint-3", 3)
            atomic_write_json(cpath / _CHECKPOINT_COMMIT, _checkpoint_file_hashes(cpath))
            sync_local_checkpoint_to_staging(
                local_ckpt_dir=cpath,
                staging_unit_dir=kw["staging_unit_dir"],
                events_dir=kw["events_dir"],
                pointers_dir=kw["pointers_dir"],
                uid=kw["uid"],
                attempt=kw["attempt"],
                config_hash=kw["config_hash"],
                manifest_sha256=kw["manifest_sha256"],
            )
            return {
                "train_seconds": 0.02,
                "actual_global_step": 3,
                "best_trainer_checkpoint": str(cpath),
                "model_revision": "rev_x",
                "tokenizer_revision": "rev_x",
                "predictions": {"val_probs": np.eye(3)[kw["val_df"]["label_idx"].to_numpy(dtype=int)] * 0.94 + 0.02},
            }

        from bangla_smish.post_stage_b_tokenization import BANGLABERT_NORMALIZER_COMMIT
        valid_env_fp = {
            "python_version": "3.12.3",
            "torch_cuda_version": "12.4",
            "normalizer_commit": BANGLABERT_NORMALIZER_COMMIT,
            "packages": {dist: "1.2.3" for dist, _ in _IDENTITY_PACKAGE_DISTS},
        }

        # Case 1 & 2: Failure during event creation AFTER durable run_state.json commit preserves run_state.json
        art_ev_crash = self.root / "v7_crash_before_event"
        with (
            patch("bangla_smish.post_stage_b_runner.collect_stable_environment_fingerprint", return_value=valid_env_fp),
            patch("bangla_smish.post_stage_b_runner.hf_train_and_predict_fold", side_effect=_fake_hf_runner),
            patch("bangla_smish.post_stage_b_runner.ensure_unit_completion_event", side_effect=OSError("Simulated Drive event write failure")),
        ):
            with self.assertRaisesRegex(OSError, "Simulated Drive event write failure"):
                run_post_stage_b_pipeline(bundle_dir=v2_dir, artifacts_dir=art_ev_crash, cfg=cfg_rec)

        self.assertEqual(len(train_invocations), 1)
        st_after_ev_fail = json.loads((art_ev_crash / "run_state.json").read_text(encoding="utf-8"))
        self.assertIn(uid0, st_after_ev_fail["committed_units"])
        self.assertEqual(st_after_ev_fail["attempts"][uid0][-1]["status"], "completed")
        self.assertFalse(st_after_ev_fail["committed_units"][uid0]["training_checkpoints_pruned_post_commit"])
        self.assertEqual(list((art_ev_crash / "events").glob("*UNIT_COMMITTED*.json")), [])
        metrics_file = art_ev_crash / "units" / uid0 / "attempts" / "1" / "metrics.json"
        metrics_bytes_orig = metrics_file.read_bytes()
        # Recovery checkpoints still intact prior to reconciliation
        self.assertTrue(any((art_ev_crash / "staging" / uid0).rglob("checkpoint-3")))
        self.assertTrue((_local_checkpoint_base(art_ev_crash, cfg_rec) / uid0 / "1" / "checkpoint-3").is_dir())

        # Case 3: Resume where event creation succeeds, then pruning fails transiently
        with (
            patch("bangla_smish.post_stage_b_runner.collect_stable_environment_fingerprint", return_value=valid_env_fp),
            patch("bangla_smish.post_stage_b_runner.hf_train_and_predict_fold", side_effect=_fake_hf_runner),
            patch("bangla_smish.post_stage_b_checkpointing.shutil.rmtree", side_effect=OSError("Simulated FUSE prune failure")),
        ):
            run_post_stage_b_pipeline(bundle_dir=v2_dir, artifacts_dir=art_ev_crash, cfg=cfg_rec)

        self.assertEqual(len(train_invocations), 1)  # NO retraining!
        ev_files_after_c3 = list((art_ev_crash / "events").glob("*UNIT_COMMITTED*.json"))
        self.assertEqual(len(ev_files_after_c3), 1)
        self.assertEqual(metrics_file.read_bytes(), metrics_bytes_orig)
        st_c3 = json.loads((art_ev_crash / "run_state.json").read_text(encoding="utf-8"))
        self.assertFalse(st_c3["committed_units"][uid0]["training_checkpoints_pruned_post_commit"])
        self.assertTrue(st_c3["committed_units"][uid0]["post_commit_prune_errors"])

        # Case 4: Restart and reconcile cleanly -> reuses existing event (still 1), finishes pruning, clears errors
        with (
            patch("bangla_smish.post_stage_b_runner.collect_stable_environment_fingerprint", return_value=valid_env_fp),
            patch("bangla_smish.post_stage_b_runner.hf_train_and_predict_fold", side_effect=_fake_hf_runner),
        ):
            run_post_stage_b_pipeline(bundle_dir=v2_dir, artifacts_dir=art_ev_crash, cfg=cfg_rec)
            # Case 5: Duplicate reconciliation attempt -> idempotent no-op, still 1 event, metrics.json untouched
            run_post_stage_b_pipeline(bundle_dir=v2_dir, artifacts_dir=art_ev_crash, cfg=cfg_rec)

        self.assertEqual(len(train_invocations), 1)  # Still zero retraining!
        self.assertEqual(len(list((art_ev_crash / "events").glob("*UNIT_COMMITTED*.json"))), 1)
        self.assertEqual(metrics_file.read_bytes(), metrics_bytes_orig)
        st_reconciled = json.loads((art_ev_crash / "run_state.json").read_text(encoding="utf-8"))
        self.assertTrue(st_reconciled["committed_units"][uid0]["training_checkpoints_pruned_post_commit"])
        self.assertNotIn("post_commit_prune_errors", st_reconciled["committed_units"][uid0])
        self.assertFalse(any((art_ev_crash / "staging" / uid0).rglob("checkpoint-3")))
        self.assertFalse((_local_checkpoint_base(art_ev_crash, cfg_rec) / uid0).exists())
        ev_doc = json.loads(list((art_ev_crash / "events").glob("*UNIT_COMMITTED*.json"))[0].read_text(encoding="utf-8"))
        self.assertEqual(ev_doc["metrics_sha256"], sha256_file(metrics_file))

        # Duplicate pre-existing UNIT_COMMITTED event fails closed on resume WITHOUT mutating run_state.json
        art_dup_ev = self.root / "v7_dup_event_art"
        shutil.copytree(art_ev_crash, art_dup_ev)
        commit_unit_completion_event(
            events_dir=art_dup_ev / "events",
            uid=uid0,
            attempt=1,
            config_hash=st_reconciled["identity"]["config_hash"],
            manifest_sha256=st_reconciled["identity"]["manifest_sha256"],
            metrics_sha256=st_reconciled["committed_units"][uid0]["metrics_sha256"],
            preds_sha256=st_reconciled["committed_units"][uid0]["preds_sha256"],
        )
        with (
            patch("bangla_smish.post_stage_b_runner.collect_stable_environment_fingerprint", return_value=valid_env_fp),
            patch("bangla_smish.post_stage_b_runner.hf_train_and_predict_fold", side_effect=_fake_hf_runner),
        ):
            with self.assertRaisesRegex(ValueError, "duplicate UNIT_COMMITTED events"):
                run_post_stage_b_pipeline(bundle_dir=v2_dir, artifacts_dir=art_dup_ev, cfg=cfg_rec)
        st_after_dup = json.loads((art_dup_ev / "run_state.json").read_text(encoding="utf-8"))
        self.assertIn(uid0, st_after_dup["committed_units"])

    def test_v7_strict_trainer_global_step_validation_and_terminology(self) -> None:
        """Mocked CPU test: enforces actual_global_step == expected Trainer update-step budget for real HF runs."""
        budget = compute_optimizer_step_budget(3520, epochs=3, batch_size=32, gradient_accumulation_steps=1)
        self.assertEqual(budget["expected_optimizer_steps"], 330)
        self.assertEqual(budget["expected_trainer_update_steps"], 330)
        self.assertIn("transformers==4.57.1", budget["step_semantics_disclosure"])
        self.assertIn("GradScaler", budget["step_semantics_disclosure"])

        v2_dir = ROOT / "data" / "processed" / "post_stage_b_bundle_v2"
        cfg_step = PostStageBConfig(
            run_name="v7_step_check",
            run_mode="gpu_smoke_test",
            experiments=("A",),
            models=("xlm-roberta-base",),
            text_views=("canon_masked",),
            folds=(0,),
            epochs=3,
            batch_size=32,
            smoke_max_train_samples=16,
            smoke_max_val_samples=8,
            smoke_max_eval_samples=8,
            run_heldout_eval=False,
            run_adversarial_eval=False,
            run_counterfactual_eval=False,
        )

        def _step_hf_runner(reported_step: int):
            def _fn(**kw):
                cpath = _write_ckpt(kw["checkpoint_dir"] / "checkpoint-3", 3)
                atomic_write_json(cpath / _CHECKPOINT_COMMIT, _checkpoint_file_hashes(cpath))
                return {
                    "train_seconds": 0.01,
                    "actual_global_step": reported_step,
                    "successful_optimizer_weight_updates": None,
                    "best_trainer_checkpoint": str(cpath),
                    "predictions": {"val_probs": np.eye(3)[kw["val_df"]["label_idx"].to_numpy(dtype=int)] * 0.94 + 0.02},
                }

            return _fn

        from bangla_smish.post_stage_b_tokenization import BANGLABERT_NORMALIZER_COMMIT
        valid_env_fp = {
            "python_version": "3.12.3",
            "torch_cuda_version": "12.4",
            "normalizer_commit": BANGLABERT_NORMALIZER_COMMIT,
            "packages": {dist: "1.2.3" for dist, _ in _IDENTITY_PACKAGE_DISTS},
        }
        # Expected Trainer update-step budget for 16 rows / batch 32 / 3 epochs = 3
        for bad_step in (2, 4):
            with self.subTest(bad_step=bad_step):
                with (
                    patch("bangla_smish.post_stage_b_runner.collect_stable_environment_fingerprint", return_value=valid_env_fp),
                    patch("bangla_smish.post_stage_b_runner.hf_train_and_predict_fold", side_effect=_step_hf_runner(bad_step)),
                ):
                    with self.assertRaisesRegex(ValueError, "!= expected Trainer update-step budget 3"):
                        run_post_stage_b_pipeline(bundle_dir=v2_dir, artifacts_dir=self.root / f"step_bad_{bad_step}", cfg=cfg_step)

        # Exact match passes and records expected, actual, and uninstrumented successful weight updates separately
        art_ok = self.root / "step_exact_ok"
        with (
            patch("bangla_smish.post_stage_b_runner.collect_stable_environment_fingerprint", return_value=valid_env_fp),
            patch("bangla_smish.post_stage_b_runner.hf_train_and_predict_fold", side_effect=_step_hf_runner(3)),
        ):
            res = run_post_stage_b_pipeline(bundle_dir=v2_dir, artifacts_dir=art_ok, cfg=cfg_step)
        uid0 = "exp_A__xlm-roberta-base__canon_masked__fold_0"
        mjson = json.loads((art_ok / "units" / uid0 / "attempts" / "1" / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(mjson["expected_optimizer_steps"], 3)
        self.assertEqual(mjson["expected_trainer_update_steps"], 3)
        self.assertEqual(mjson["actual_global_step"], 3)
        self.assertIsNone(mjson["successful_optimizer_weight_updates"])
        self.assertEqual(res["unit_rows"][0]["actual_global_step"], 3)

    def test_v7_hardened_atomic_writes_posix_fsync_fuse_tolerance_and_orphan_cleanup(self) -> None:
        """Verifies PID+UUID atomic writes, POSIX dir fsync, FUSE EINVAL/EOPNOTSUPP tolerance, and orphan cleanup."""
        target = self.root / "atomic_dir" / "state.json"
        fsync_fds: list[int] = []
        orig_fsync = os.fsync

        def _spy_fsync(fd: int):
            fsync_fds.append(fd)
            return orig_fsync(fd)

        with patch("bangla_smish.post_stage_b_runner.os.fsync", side_effect=_spy_fsync):
            atomic_write_text(target, '{"ok": 1}')
        # At least 2 fsync calls: temporary file fd + parent directory fd
        self.assertGreaterEqual(len(fsync_fds), 2)
        self.assertEqual(target.read_text(encoding="utf-8"), '{"ok": 1}')
        self.assertEqual([p.name for p in target.parent.iterdir()], ["state.json"])

        # FUSE-unsupported errnos (EINVAL, EOPNOTSUPP) tolerated cleanly in _fsync_dir_best_effort
        for fuse_err in (errno.EINVAL, errno.EOPNOTSUPP):
            with patch("bangla_smish.post_stage_b_runner.os.fsync", side_effect=OSError(fuse_err, "FUSE unsupported")):
                _fsync_dir_best_effort(target.parent)

        # Unexpected hardware I/O error (EIO) still propagates fail-closed
        with patch("bangla_smish.post_stage_b_runner.os.fsync", side_effect=OSError(errno.EIO, "Disk I/O error")):
            with self.assertRaises(OSError):
                _fsync_dir_best_effort(target.parent)

        # Mid-write failure cleans up orphaned temp file and preserves original target content
        with patch("bangla_smish.post_stage_b_runner.os.fsync", side_effect=OSError(errno.EIO, "Simulated write failure")):
            with self.assertRaises(OSError):
                atomic_write_text(target, '{"corrupt": true}')
        self.assertEqual(target.read_text(encoding="utf-8"), '{"ok": 1}')
        self.assertEqual([p.name for p in target.parent.iterdir()], ["state.json"])

    def test_v7_package_fingerprint_completeness_tokenizers_and_no_installed_fallback(self) -> None:
        """Verifies tokenizers & runtime-critical dist tracking via importlib.metadata with zero 'installed' fallback."""
        from bangla_smish.post_stage_b_tokenization import BANGLABERT_NORMALIZER_COMMIT

        fp = collect_stable_environment_fingerprint()
        self.assertNotIn("timestamp_utc", fp)
        tracked_dists = {dist for dist, _ in _IDENTITY_PACKAGE_DISTS}
        for required_dist in (
            "numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "torch", "transformers", "tokenizers",
            "datasets", "accelerate", "safetensors", "sentencepiece", "packaging", "huggingface_hub",
            "protobuf", "regex", "pyyaml", "requests", "tqdm", "filelock",
            "normalizer", "ftfy", "emoji", "wcwidth",
        ):
            self.assertIn(required_dist, tracked_dists)
            self.assertIn(required_dist, fp["packages"])
            self.assertNotEqual(fp["packages"][required_dist], "installed")
        self.assertNotIn("unicodedata2", tracked_dists)

        # importlib.metadata.version resolves version even when imported module lacks __version__
        with patch("bangla_smish.post_stage_b_runner.importlib_metadata.version", return_value="0.0.1"):
            self.assertEqual(_resolve_package_version("normalizer", "normalizer"), "0.0.1")

        # Module imported without metadata AND without __version__ returns 'unknown_version', NEVER 'installed'
        from importlib import metadata as importlib_metadata
        fake_no_ver = types.SimpleNamespace()
        with (
            patch("bangla_smish.post_stage_b_runner.importlib_metadata.version", side_effect=importlib_metadata.PackageNotFoundError),
            patch("bangla_smish.post_stage_b_runner.importlib.import_module", return_value=fake_no_ver),
        ):
            self.assertEqual(_resolve_package_version("normalizer", "normalizer"), "unknown_version")

        # Real-HF t4_benchmark pre-run validation rejects missing/unknown tokenizers or bad normalizer_commit before lease
        v2_dir = ROOT / "data" / "processed" / "post_stage_b_bundle_v2"
        t4_no_fix = replace(
            load_config(ROOT / "experiments" / "colab" / "configs" / "post_stage_b_stage4_t4_benchmark_config.json"),
            benchmark_workload_fixture_eval=False,
        )
        full_pkgs = {dist: "1.2.3" for dist, _ in _IDENTITY_PACKAGE_DISTS}
        bad_tok_fp = {
            "python_version": "3.12.3",
            "torch_cuda_version": "12.4",
            "normalizer_commit": BANGLABERT_NORMALIZER_COMMIT,
            "packages": {**full_pkgs, "tokenizers": "unknown_version"},
        }
        with (
            patch("bangla_smish.post_stage_b_runner.collect_hardware_fingerprint", return_value=self._mock_t4_hw()),
            patch("bangla_smish.post_stage_b_runner.collect_stable_environment_fingerprint", return_value=bad_tok_fp),
            patch("bangla_smish.post_stage_b_runner.register_single_writer_lease") as spy_lease,
        ):
            art_bad_tok = self.root / "v7_bad_tokenizers_fp"
            with self.assertRaisesRegex(RuntimeError, "tokenizers"):
                run_post_stage_b_pipeline(bundle_dir=v2_dir, artifacts_dir=art_bad_tok, cfg=t4_no_fix)
            spy_lease.assert_not_called()
            self.assertFalse(art_bad_tok.exists())

        bad_norm_fp = {
            "python_version": "3.12.3",
            "torch_cuda_version": "12.4",
            "normalizer_commit": "0" * 40,
            "packages": dict(full_pkgs),
        }
        with (
            patch("bangla_smish.post_stage_b_runner.collect_hardware_fingerprint", return_value=self._mock_t4_hw()),
            patch("bangla_smish.post_stage_b_runner.collect_stable_environment_fingerprint", return_value=bad_norm_fp),
            patch("bangla_smish.post_stage_b_runner.register_single_writer_lease") as spy_lease2,
        ):
            art_bad_norm = self.root / "v7_bad_normalizer_fp"
            with self.assertRaisesRegex(RuntimeError, "pinned normalizer commit"):
                run_post_stage_b_pipeline(bundle_dir=v2_dir, artifacts_dir=art_bad_norm, cfg=t4_no_fix)
            spy_lease2.assert_not_called()
            self.assertFalse(art_bad_norm.exists())

    # ------------------------------------------------------------------
    # 3. Checkpoint & Lease Hardening
    # ------------------------------------------------------------------
    def test_strict_lease_takeover_local_commit_refresh_and_prune_errors(self) -> None:
        # 3A/3B/3C: Even ONE stale lease requires exact non-empty expected_stale_writer_ids
        art = self.root / "strict_lease"
        register_single_writer_lease(art, writer_id="w_only")
        with self.assertRaisesRegex(RuntimeError, "expected_stale_writer_ids is required"):
            register_single_writer_lease(
                art, writer_id="w2", allow_stale_takeover=True, takeover_reason="confirm stopped"
            )
        with self.assertRaisesRegex(RuntimeError, "at least one writer ID"):
            register_single_writer_lease(
                art, writer_id="w2", allow_stale_takeover=True,
                expected_stale_writer_ids=[], takeover_reason="confirm stopped",
            )
        with self.assertRaisesRegex(RuntimeError, "does not match actual unresolved"):
            register_single_writer_lease(
                art, writer_id="w2", allow_stale_takeover=True,
                expected_stale_writer_ids=["wrong_id"], takeover_reason="confirm stopped",
            )
        l_ok = register_single_writer_lease(
            art, writer_id="w2", allow_stale_takeover=True,
            expected_stale_writer_ids=["w_only"], takeover_reason="owner confirmed stop",
        )
        self.assertEqual(l_ok["takeover_record"]["superseded_writer_ids"], ["w_only"])
        release_single_writer_lease(art, "w2")

        # Corrupt record also requires explicit expected_corrupt_files
        (art / "leases" / "release_broken_tail.json").write_text("{bad", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "expected_corrupt_files confirmation"):
            register_single_writer_lease(
                art, writer_id="w3", allow_stale_takeover=True, takeover_reason="ack"
            )
        l3 = register_single_writer_lease(
            art, writer_id="w3", allow_stale_takeover=True,
            expected_corrupt_files=["release_broken_tail.json"], takeover_reason="ack broken tail",
        )
        self.assertEqual(l3["takeover_record"]["acknowledged_corrupt_files"], ["release_broken_tail.json"])
        self.assertTrue((art / "leases" / "release_broken_tail.json").exists())

        # 3D: Local relocation refreshes local post_stage_b_checkpoint_commit.json so _valid_checkpoint passes
        staging = self.root / "st_u"
        events = self.root / "ev"
        ptrs = self.root / "pt"
        old_best = self.root / "old_vm" / "checkpoint-100"
        c1 = _write_ckpt(old_best, 100, best_ckpt=str(old_best))
        sync_local_checkpoint_to_staging(
            local_ckpt_dir=c1, staging_unit_dir=staging, events_dir=events, pointers_dir=ptrs,
            uid="u1", attempt=1, config_hash="ch", manifest_sha256="mh",
        )
        c2 = _write_ckpt(self.root / "old_vm" / "checkpoint-200", 200, best_ckpt=str(old_best))
        sync_local_checkpoint_to_staging(
            local_ckpt_dir=c2, staging_unit_dir=staging, events_dir=events, pointers_dir=ptrs,
            uid="u1", attempt=1, config_hash="ch", manifest_sha256="mh",
        )
        shutil.rmtree(self.root / "old_vm")
        scan = scan_unit_staging(
            staging_unit_dir=staging, events_dir=events, pointers_dir=ptrs,
            uid="u1", expected_config_hash="ch", expected_manifest_sha256="mh", checkpoint_retention_limit=1,
        )
        rec = recover_latest_valid_checkpoint(scan_report=scan, local_restore_root=self.root / "ssd_a2")
        self.assertTrue((rec["restored_local_ckpt"] / LEGACY_RUNNER_CHECKPOINT_COMMIT).is_file())
        self.assertTrue(_valid_checkpoint(rec["restored_local_ckpt"]))
        self.assertTrue(_valid_checkpoint(rec["restored_best_local_ckpt"]))

        # 3E: Mocked runner surfaces checkpoint_prune_errors in unit metrics.json
        bdir = ROOT / "data" / "processed" / "post_stage_b_bundle_v2"
        run_art = self.root / "prune_diag_art"
        cfg = PostStageBConfig(
            run_name="prune_err_diag",
            run_mode="gpu_smoke_test",
            experiments=("A",),
            models=("csebuetnlp/banglabert",),
            text_views=("canon_masked",),
            folds=(0,),
            smoke_max_train_samples=16,
            smoke_max_val_samples=8,
            smoke_max_eval_samples=8,
            run_heldout_eval=False,
            run_adversarial_eval=False,
            run_counterfactual_eval=False,
        )
        run_post_stage_b_pipeline(
            bundle_dir=bdir,
            artifacts_dir=run_art,
            cfg=cfg,
            train_eval_fn=lambda **kw: {
                "actual_global_step": 1,
                "train_seconds": 0.01,
                "checkpoint_prune_errors": [{"phase": "on_save", "rel_path": "attempt_001/checkpoint-1", "error": "Drive busy"}],
                "predictions": {"val_probs": np.eye(3)[kw["val_df"]["label_idx"].to_numpy(dtype=int)] * 0.94 + 0.02},
            },
        )
        mjson = json.loads(
            (run_art / "units" / "exp_A__csebuetnlp__banglabert__canon_masked__fold_0" / "attempts" / "1" / "metrics.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(mjson["checkpoint_prune_errors"]), 1)
        self.assertEqual(mjson["checkpoint_prune_errors"][0]["phase"], "on_save")

    # ------------------------------------------------------------------
    # 4. Portable Kit OS Metadata, Mask Re-derivation, Transitive Deps & pyproject.toml
    # ------------------------------------------------------------------
    def test_kit_os_metadata_rederived_masks_and_dependency_reconciliation(self) -> None:
        kit_script = ROOT / "scripts" / "prepare_stage4_colab_kit.py"
        spec = importlib.util.spec_from_file_location("prepare_stage4_colab_kit", kit_script)
        kit_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kit_mod)

        staged = self.root / "kit_v2_final"
        kit_mod.export_stage4_colab_kit(ROOT, ROOT / "data" / "processed" / "post_stage_b_bundle_v2", staged)

        # 4A: Known OS metadata ignored cleanly without weakening rogue file rejection
        (staged / ".DS_Store").write_bytes(b"finder")
        (staged / "experiments" / "colab" / "desktop.ini").write_text("[.ShellClassInfo]", encoding="utf-8")
        cp_dir = staged / "experiments" / "colab" / ".ipynb_checkpoints"
        cp_dir.mkdir()
        (cp_dir / "post_stage_b_stage4_experiments-checkpoint.ipynb").write_text("{}", encoding="utf-8")
        ok = verify_portable_colab_kit(staged)
        self.assertEqual(ok["reverified_frozen_masks"], 4)

        # Rogue executable/scientific file is STILL rejected!
        bad_rogue = self.root / "kit_rogue"
        shutil.copytree(staged, bad_rogue)
        (bad_rogue / "rogue_script.py").write_text("print('bad')", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "UNEXPECTED: rogue_script.py"):
            verify_portable_colab_kit(bad_rogue)

        # 4B: Tampering frozen mask digest AND re-signing kit manifest still caught by live mask re-derivation
        bad_mask = self.root / "kit_bad_mask"
        shutil.copytree(staged, bad_mask)
        masks_p = bad_mask / "reports" / "stage4_frozen_test_masks.json"
        mdoc = json.loads(masks_p.read_text(encoding="utf-8"))
        mdoc["cohorts"]["primary_family_decontam_1687"]["mask_sha256"] = "0" * 64
        masks_p.write_text(json.dumps(mdoc), encoding="utf-8")
        # Even if manifest were regenerated to match tampered file bytes, verify_portable_colab_kit rejects live mismatch!
        import hashlib
        from bangla_smish.post_stage_b_integrity import STAGE4_KIT_MANIFEST_NAME
        from bangla_smish.post_stage_b_data import sha256_file
        kman = json.loads((bad_mask / STAGE4_KIT_MANIFEST_NAME).read_text(encoding="utf-8"))
        kman["required_runtime_files"]["reports/stage4_frozen_test_masks.json"]["sha256"] = sha256_file(masks_p)
        kman["required_runtime_files"]["reports/stage4_frozen_test_masks.json"]["bytes"] = int(masks_p.stat().st_size)
        body = {k: v for k, v in kman.items() if k != "manifest_sha256"}
        kman["manifest_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        (bad_mask / STAGE4_KIT_MANIFEST_NAME).write_text(json.dumps(kman), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Re-derived frozen test mask mismatch"):
            verify_portable_colab_kit(bad_mask)

        # 4C & 4D: Explicit transitive specs and pyproject.toml [gpu] alignment
        joined = "\n".join(STAGE4_BASE_PACKAGES)
        for pkg in (
            "packaging>=23.0",
            "huggingface_hub>=0.34.0,<1.0",
            "tokenizers>=0.22.0,<=0.23.0",
            "safetensors>=0.4.3",
            "protobuf>=3.20",
            "regex!=2019.12.17",
            "pyyaml>=5.1",
            "requests>=2.31",
            "tqdm>=4.27",
            "filelock>=3.0",
        ):
            self.assertIn(pkg, joined)
        for mod_name in ("packaging", "huggingface_hub", "tokenizers", "google.protobuf", "regex", "yaml", "requests", "tqdm", "filelock"):
            self.assertIn(mod_name, STAGE4_CORE_IMPORTS)
        self.assertEqual(BANGLABERT_NORMALIZER_TRANSITIVE_DEPS, ("regex", "ftfy", "emoji", "wcwidth"))
        self.assertNotIn("unicodedata2", BANGLABERT_NORMALIZER_TRANSITIVE_DEPS)

        pyproj = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        for expected in ("torch>=2.6", "transformers>=4.57.1,<5", "safetensors>=0.4", "sentencepiece>=0.2,<1", "protobuf>=3.20"):
            self.assertIn(expected, pyproj)


if __name__ == "__main__":
    unittest.main()
