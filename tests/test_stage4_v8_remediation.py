"""Stage 4 v8 comprehensive P0-Medium remediation tests (local real-library + CPU-mocked verification).

Honesty boundary:
  - Uses REAL installed local libraries (`importlib.metadata`, `huggingface_hub.try_to_load_from_cache`,
    `pandas`, `numpy`, `pyarrow`, `nbformat`, POSIX filesystem/fsync) against real frozen bundle v2
    and temporary on-disk HF caches / `.dist-info` directories.
  - Uses CPU mocks ONLY for Colab-only packages unavailable locally (`torch`, `transformers`, `datasets`,
    `Trainer`, CUDA/T4 hardware, Google Drive FUSE errnos). Does NOT execute Colab, GPU training,
    synthetic generation, or held-out/OOD model evaluation.
"""

from __future__ import annotations

from dataclasses import replace
import errno
import importlib.metadata as real_metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bangla_smish.metrics import evaluate_predictions, paired_family_accuracy_delta
from bangla_smish.post_stage_b_checkpointing import (
    CheckpointState,
    STAGE4_CHECKPOINT_MANIFEST,
    STAGE4_TIMING_METADATA,
    commit_unit_completion_event,
    inspect_writer_leases,
    prune_superseded_staging,
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
    preflight_colab_dependencies,
)
from bangla_smish.post_stage_b_data import sha256_file
from bangla_smish.post_stage_b_integrity import (
    STAGE4_KIT_REQUIRED_SOURCE_FILES,
    verify_portable_colab_kit,
)
from bangla_smish.post_stage_b_runner import (
    PostStageBConfig,
    SYNTHETIC_WORKLOAD_FIXTURE_ENABLED,
    _BANGLABERT_REQUIRED_PACKAGE_DISTS,
    _CHECKPOINT_COMMIT,
    _IDENTITY_PACKAGE_DISTS,
    _checkpoint_file_hashes,
    _find_latest_trainer_checkpoint,
    _fsync_dir_best_effort,
    _local_checkpoint_base,
    _run_identity,
    _valid_checkpoint,
    _validate_required_packages_fingerprint,
    _verify_trainer_checkpoint,
    _write_prediction_archive,
    aggregate_post_stage_b_results,
    atomic_write_json,
    atomic_write_text,
    collect_stable_environment_fingerprint,
    hf_train_and_predict_fold,
    load_config,
    run_post_stage_b_pipeline,
)
from bangla_smish.post_stage_b_tokenization import (
    BANGLABERT_MODEL,
    BANGLABERT_NORMALIZER_COMMIT,
    extract_commit_hash_from_path,
    resolve_and_verify_tokenizer_revision,
    resolve_hf_cached_tokenizer_files,
)
from bangla_smish.post_stage_b_tokenizer_gate import (
    PINNED_TOKENIZER_REVISIONS,
    run_allowed_view_tokenizer_collision_gate,
)

V2_BUNDLE = ROOT / "data" / "processed" / "post_stage_b_bundle_v2"


def _write_ckpt(
    ckpt_dir: Path,
    step: int,
    *,
    best_ckpt: str | None = None,
    with_scaler: bool = False,
    timing_meta: dict | None = None,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    files = ["config.json", "training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"]
    if with_scaler:
        files.append("scaler.pt")
    for fname in files:
        (ckpt_dir / fname).write_bytes(f"{fname}-step-{step}".encode("utf-8"))
    (ckpt_dir / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "best_model_checkpoint": best_ckpt}), encoding="utf-8"
    )
    if timing_meta is not None:
        (ckpt_dir / STAGE4_TIMING_METADATA).write_text(json.dumps(timing_meta), encoding="utf-8")
    return ckpt_dir


def _realistic_4571_tokenizer(
    *,
    vocab_file: str | None = None,
    name_or_path: str | None = None,
    force_collision: bool = False,
) -> object:
    """Mimic transformers==4.57.1 PreTrainedTokenizerBase: NO _commit_hash on object or init_kwargs."""

    class _Tok4571:
        def __init__(self) -> None:
            self.init_kwargs: dict[str, object] = {}
            if vocab_file is not None:
                self.init_kwargs["vocab_file"] = vocab_file
            if name_or_path is not None:
                self.name_or_path = name_or_path
                self.init_kwargs["name_or_path"] = name_or_path

        def __call__(self, texts: list[str], truncation: bool = True, max_length: int = 128, add_special_tokens: bool = True):
            if force_collision:
                return {"input_ids": [[101, 202, 303, 102] for _ in texts]}
            return {"input_ids": [[101] + [ord(c) for c in t[: max_length - 2]] + [102] for t in texts]}

        def save_pretrained(self, out_dir: str) -> None:
            Path(out_dir).joinpath("tokenizer_config.json").write_text("{}", encoding="utf-8")

    tok = _Tok4571()
    assert not hasattr(tok, "_commit_hash")
    assert "_commit_hash" not in tok.init_kwargs
    return tok


def _mock_t4_hw(*, gpu_count: int = 1) -> dict:
    return {
        "cuda_available": True,
        "gpu_count": gpu_count,
        "gpu_name": "Tesla T4",
        "compute_capability": "7.5",
        "nvidia_driver_version": "550.54.15",
        "torch_cuda_version": "12.4",
    }


def _mock_valid_env_fp() -> dict:
    return {
        "python_version": sys.version.split()[0],
        "torch_cuda_version": "12.4",
        "normalizer_commit": BANGLABERT_NORMALIZER_COMMIT,
        "packages": {dist: "1.2.3" for dist, _ in _IDENTITY_PACKAGE_DISTS},
    }


class TestStage4V8Remediation(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    # ------------------------------------------------------------------
    # Scenario A (P0): Tokenizer revision verification (transformers==4.57.1 compatible)
    # ------------------------------------------------------------------
    def test_scenario_a_tokenizer_revision_verification_and_gate_binding(self) -> None:
        model_name = "xlm-roberta-base"
        pinned = PINNED_TOKENIZER_REVISIONS[model_name]
        wrong_sha = "1" * 40

        # A1. Realistic transformers==4.57.1 tokenizer: NO _commit_hash, snapshot vocab_file path -> PASSES
        snap_vocab = f"/root/.cache/huggingface/hub/models--xlm-roberta-base/snapshots/{pinned}/sentencepiece.bpe.model"
        tok_snap = _realistic_4571_tokenizer(vocab_file=snap_vocab, name_or_path=model_name)
        self.assertEqual(extract_commit_hash_from_path(snap_vocab), pinned)
        self.assertEqual(
            resolve_and_verify_tokenizer_revision(
                tok_snap, model_name=model_name, expected_revision=pinned, allow_hf_cache_lookup=False
            ),
            pinned,
        )

        # A2. Real huggingface_hub.try_to_load_from_cache on a real temporary HF cache directory!
        hf_cache = self.root / "hf_hub_cache"
        snap_dir = hf_cache / "models--xlm-roberta-base" / "snapshots" / pinned
        snap_dir.mkdir(parents=True, exist_ok=True)
        (snap_dir / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
        tok_empty_kw = _realistic_4571_tokenizer(vocab_file=None, name_or_path=model_name)
        cached_found = resolve_hf_cached_tokenizer_files(model_name, pinned, cache_dir=hf_cache)
        self.assertTrue(any(pinned in p for p in cached_found))
        self.assertEqual(
            resolve_and_verify_tokenizer_revision(
                tok_empty_kw,
                model_name=model_name,
                expected_revision=pinned,
                allow_hf_cache_lookup=True,
                cache_dir=hf_cache,
            ),
            pinned,
        )

        # A3. Missing cache + empty init_kwargs -> FAIL CLOSED (no `or expected_rev` fallback!)
        empty_cache = self.root / "hf_empty_cache"
        empty_cache.mkdir()
        with self.assertRaisesRegex(RuntimeError, "unable to establish resolved commit SHA"):
            resolve_and_verify_tokenizer_revision(
                tok_empty_kw,
                model_name=model_name,
                expected_revision=pinned,
                allow_hf_cache_lookup=True,
                cache_dir=empty_cache,
            )

        # A4. Mismatched snapshot commit SHA -> FAIL CLOSED
        wrong_vocab = f"/root/.cache/huggingface/hub/models--xlm-roberta-base/snapshots/{wrong_sha}/tokenizer.json"
        tok_wrong = _realistic_4571_tokenizer(vocab_file=wrong_vocab, name_or_path=model_name)
        with self.assertRaisesRegex(RuntimeError, "!= pinned revision"):
            resolve_and_verify_tokenizer_revision(
                tok_wrong, model_name=model_name, expected_revision=pinned, allow_hf_cache_lookup=False
            )

        # A5. Local/custom loader ignoring revision (non-snapshot path or redirected name_or_path) -> FAIL CLOSED
        tok_local_file = _realistic_4571_tokenizer(vocab_file="/tmp/unpinned_local/vocab.txt", name_or_path=model_name)
        with self.assertRaisesRegex(RuntimeError, "outside a verified HuggingFace snapshots"):
            resolve_and_verify_tokenizer_revision(
                tok_local_file, model_name=model_name, expected_revision=pinned, allow_hf_cache_lookup=True, cache_dir=hf_cache
            )
        tok_local_nop = _realistic_4571_tokenizer(vocab_file=None, name_or_path="/tmp/unpinned_local_model")
        with self.assertRaisesRegex(RuntimeError, "does not match requested repository"):
            resolve_and_verify_tokenizer_revision(
                tok_local_nop, model_name=model_name, expected_revision=pinned, allow_hf_cache_lookup=True, cache_dir=hf_cache
            )

        # A6. Cell 5 gate uses identical fail-closed verifier + binds bundle_manifest_sha256 & tokenizer_package_versions
        cfg_xlm = PostStageBConfig(
            run_name="v8_tok_gate",
            run_mode="gpu_smoke_test",
            experiments=("A",),
            models=(model_name,),
            text_views=("canon_masked",),
            folds=(0,),
            tokenizer_revisions={model_name: pinned},
            smoke_max_train_samples=16,
            smoke_max_val_samples=8,
            smoke_max_eval_samples=8,
            run_heldout_eval=False,
            run_adversarial_eval=False,
            run_counterfactual_eval=False,
        )
        # Gate rejects tokenizer with empty init_kwargs when custom loader provides no snapshot evidence
        with self.assertRaisesRegex(RuntimeError, "unable to establish resolved commit SHA"):
            run_allowed_view_tokenizer_collision_gate(
                V2_BUNDLE,
                cfg_xlm,
                allowed_views=("canon_masked",),
                tokenizer_loader=lambda m, r: _realistic_4571_tokenizer(vocab_file=None, name_or_path=m),
            )
        # Gate passes with realistic snapshot-path tokenizer and records manifest & package bindings
        gate_out = run_allowed_view_tokenizer_collision_gate(
            V2_BUNDLE,
            cfg_xlm,
            allowed_views=("canon_masked",),
            tokenizer_loader=lambda m, r: _realistic_4571_tokenizer(vocab_file=snap_vocab, name_or_path=m),
            export_path=self.root / "gate_a6.json",
        )
        self.assertTrue(gate_out["passed"])
        self.assertEqual(gate_out["bundle_manifest_sha256"], sha256_file(V2_BUNDLE / "dataset_manifest.json"))
        self.assertEqual(
            set(gate_out["tokenizer_package_versions"].keys()),
            {"transformers", "tokenizers", "sentencepiece", "huggingface_hub"},
        )

        # Runner rejects tampered bundle_manifest_sha256 or mismatched tokenizer_package_versions fail-closed
        tampered_manifest_gate = {**gate_out, "bundle_manifest_sha256": "f" * 64}
        with self.assertRaisesRegex(ValueError, "bundle_manifest_sha256 mismatch"):
            run_post_stage_b_pipeline(
                bundle_dir=V2_BUNDLE,
                artifacts_dir=self.root / "art_bad_gate_manifest",
                cfg=cfg_xlm,
                train_eval_fn=lambda **kw: self.fail("must fail before training"),
                tokenizer_gate_report=tampered_manifest_gate,
            )
        tampered_pkg_gate = {
            **gate_out,
            "tokenizer_package_versions": {**gate_out["tokenizer_package_versions"], "tokenizers": "9.9.9"},
        }
        with self.assertRaisesRegex(ValueError, "tokenizer gate package versions do not match live runtime"):
            run_post_stage_b_pipeline(
                bundle_dir=V2_BUNDLE,
                artifacts_dir=self.root / "art_bad_gate_pkg",
                cfg=cfg_xlm,
                train_eval_fn=lambda **kw: self.fail("must fail before training"),
                tokenizer_gate_report=tampered_pkg_gate,
            )

    # ------------------------------------------------------------------
    # Scenario B (P0): Dependency closure without unicodedata2 (real importlib.metadata on temp .dist-info)
    # ------------------------------------------------------------------
    def test_scenario_b_dependency_closure_real_dist_info_no_unicodedata2(self) -> None:
        # B1. Zero references to bogus unicodedata2 across specs
        self.assertNotIn("unicodedata2", BANGLABERT_NORMALIZER_TRANSITIVE_DEPS)
        self.assertNotIn("unicodedata2", _BANGLABERT_REQUIRED_PACKAGE_DISTS)
        self.assertNotIn("unicodedata2", {d for d, _ in _IDENTITY_PACKAGE_DISTS})
        self.assertNotIn("unicodedata2", "\n".join(STAGE4_BASE_PACKAGES))
        self.assertNotIn("unicodedata2", (ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        # B2. Build a real on-disk temporary site-packages with exact .dist-info directories
        site_dir = self.root / "fake_site_packages"
        site_dir.mkdir()
        installed_versions = {
            "torch": "2.6.0+cu124",
            "transformers": "4.57.1",
            "tokenizers": "0.22.1",
            "datasets": "3.2.0",
            "accelerate": "1.2.1",
            "safetensors": "0.4.5",
            "sentencepiece": "0.2.0",
            "packaging": "24.2",
            "huggingface_hub": "0.34.4",
            "protobuf": "5.29.0",
            "regex": "2024.11.6",
            "PyYAML": "6.0.2",
            "requests": "2.32.3",
            "tqdm": "4.67.1",
            "filelock": "3.16.1",
            "numpy": "2.1.0",
            "pandas": "2.2.3",
            "scipy": "1.14.1",
            "scikit_learn": "1.5.2",
            "pyarrow": "18.1.0",
            "normalizer": "0.0.1",
            "ftfy": "6.0.3",
            "emoji": "1.4.2",
            "wcwidth": "0.2.13",
        }
        for dist_name, ver in installed_versions.items():
            di = site_dir / f"{dist_name}-{ver}.dist-info"
            di.mkdir()
            (di / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist_name}\nVersion: {ver}\n", encoding="utf-8")
            if dist_name == "normalizer":
                (di / "direct_url.json").write_text(
                    json.dumps(
                        {
                            "url": "https://github.com/csebuetnlp/normalizer",
                            "vcs_info": {"vcs": "git", "commit_id": BANGLABERT_NORMALIZER_COMMIT},
                        }
                    ),
                    encoding="utf-8",
                )

        def _real_path_dist(name: str):
            norm = name.lower().replace("-", "_")
            for d in real_metadata.PathDistribution.discover(path=[str(site_dir)]):
                dname = (d.metadata["Name"] or "").lower().replace("-", "_")
                if dname == norm:
                    return d
            raise real_metadata.PackageNotFoundError(name)

        def _real_path_ver(name: str) -> str:
            return _real_path_dist(name).version

        def _fake_import(name: str):
            if name == "torch":
                return types.SimpleNamespace(
                    __version__="2.6.0+cu124",
                    version=types.SimpleNamespace(cuda="12.4"),
                    cuda=types.SimpleNamespace(is_available=lambda: True),
                )
            if name == "normalizer":
                return types.SimpleNamespace(__name__=name, normalize=lambda s: s)
            return types.SimpleNamespace(__name__=name)

        # Cell 4 preflight passes with zero pip calls using real PathDistribution metadata (NO unicodedata2!)
        res = preflight_colab_dependencies(
            [BANGLABERT_MODEL, "xlm-roberta-base"],
            install_missing=True,
            require_cuda=True,
            dist_lookup=_real_path_dist,
            version_lookup=_real_path_ver,
            import_module_fn=_fake_import,
            run_pip_fn=lambda cmd: self.fail(f"Unexpected pip call when closure is complete: {cmd}"),
        )
        self.assertEqual(res["status"], "verified")
        for expected_mod in ("tokenizers", "regex", "yaml", "requests", "tqdm", "filelock", "ftfy", "emoji", "wcwidth", "normalizer"):
            self.assertIn(expected_mod, res["verified_imports"])

        # Cell 6 runner fingerprint + _validate_required_packages_fingerprint ALSO pass on the exact same .dist-info closure!
        with (
            patch("bangla_smish.post_stage_b_runner.importlib_metadata.version", side_effect=_real_path_ver),
            patch("bangla_smish.post_stage_b_runner.importlib_metadata.distribution", side_effect=_real_path_dist),
            patch.dict(sys.modules, {"torch": _fake_import("torch")}),
        ):
            fp = collect_stable_environment_fingerprint()
        cfg_bb = PostStageBConfig(
            run_name="v8_dep_closure",
            run_mode="gpu_smoke_test",
            experiments=("A",),
            models=(BANGLABERT_MODEL,),
            text_views=("canon_masked",),
            folds=(0,),
            smoke_max_train_samples=16,
            smoke_max_val_samples=8,
            smoke_max_eval_samples=8,
            run_heldout_eval=False,
            run_adversarial_eval=False,
            run_counterfactual_eval=False,
        )
        _validate_required_packages_fingerprint(fp, cfg_bb)

        # Removing wcwidth or regex or tokenizers triggers fail-closed in BOTH Cell 4 and Cell 6!
        for missing_dist in ("wcwidth", "regex", "tokenizers"):
            with self.subTest(missing_dist=missing_dist):
                def _missing_dist_lookup(name: str, _miss: str = missing_dist):
                    if name.lower().replace("-", "_") == _miss:
                        raise real_metadata.PackageNotFoundError(name)
                    return _real_path_dist(name)

                with self.assertRaisesRegex(RuntimeError, "RESTART REQUIRED"):
                    preflight_colab_dependencies(
                        [BANGLABERT_MODEL],
                        install_missing=True,
                        require_cuda=True,
                        dist_lookup=_missing_dist_lookup,
                        version_lookup=lambda n: _missing_dist_lookup(n).version,
                        import_module_fn=_fake_import,
                        run_pip_fn=lambda cmd: None,
                    )
                bad_fp = {**fp, "packages": {**fp["packages"], missing_dist: "not_installed"}}
                with self.assertRaisesRegex(RuntimeError, missing_dist):
                    _validate_required_packages_fingerprint(bad_fp, cfg_bb)

    # ------------------------------------------------------------------
    # Scenario C (HIGH): Attempt 2+ checkpoint recovery after invalidated Attempt 1 UNIT_COMMITTED
    # ------------------------------------------------------------------
    def test_scenario_c_attempt2_checkpoint_survives_historical_attempt1_commit_event(self) -> None:
        staging = self.root / "scen_c" / "staging" / "u1"
        events = self.root / "scen_c" / "events"
        pointers = self.root / "scen_c" / "pointers"
        # Attempt 1 committed and emitted a historical UNIT_COMMITTED event
        commit_unit_completion_event(
            events_dir=events,
            uid="u1",
            attempt=1,
            config_hash="cfg_h",
            manifest_sha256="man_h",
            metrics_sha256="m1",
            preds_sha256="p1",
        )
        # Attempt 1 metrics were later invalidated; Attempt 2 saves epoch-1 checkpoint
        ckpt_a2 = _write_ckpt(self.root / "ssd_a2" / "checkpoint-10", 10, best_ckpt=str(self.root / "ssd_a2" / "checkpoint-10"))
        sync_local_checkpoint_to_staging(
            local_ckpt_dir=ckpt_a2,
            staging_unit_dir=staging,
            events_dir=events,
            pointers_dir=pointers,
            uid="u1",
            attempt=2,
            config_hash="cfg_h",
            manifest_sha256="man_h",
        )
        # Scanning for Attempt 2 MUST classify Attempt 2's checkpoint as VALID_RECOVERABLE (NOT ALREADY_SUPERSEDED!)
        scan_a2 = scan_unit_staging(
            staging_unit_dir=staging,
            events_dir=events,
            pointers_dir=pointers,
            uid="u1",
            expected_config_hash="cfg_h",
            expected_manifest_sha256="man_h",
            active_attempt=2,
            trust_commit_events=False,
        )
        valid_a2 = [e for e in scan_a2["entries"] if e["state"] == CheckpointState.VALID_RECOVERABLE]
        sup_a2 = [e for e in scan_a2["entries"] if e["state"] == CheckpointState.ALREADY_SUPERSEDED]
        self.assertEqual(len(valid_a2), 1)
        self.assertEqual(valid_a2[0]["attempt"], 2)
        self.assertEqual(sup_a2, [])
        # Pruning during Attempt 2 MUST NOT delete Attempt 2's checkpoint!
        prune_out = prune_superseded_staging(scan_report=scan_a2, dry_run=False)
        self.assertEqual(prune_out["pruned_superseded"], [])
        self.assertTrue(any(staging.rglob("checkpoint-10")))
        # Attempt 3 recovers Attempt 2's checkpoint cleanly
        rec = recover_latest_valid_checkpoint(scan_report=scan_a2, local_restore_root=self.root / "ssd_a3")
        self.assertTrue(rec["recovered"])
        self.assertEqual(rec["restored_local_ckpt"].name, "checkpoint-10")

    # ------------------------------------------------------------------
    # Scenario D (HIGH): Failed Drive staging NEVER silently falls back to unstaged local checkpoint
    # ------------------------------------------------------------------
    def test_scenario_d_no_silent_local_fallback_when_drive_staging_configured(self) -> None:
        real_df = pd.read_parquet(V2_BUNDLE / "real_train_pool.parquet").head(8).reset_index(drop=True)
        pinned = PINNED_TOKENIZER_REVISIONS["xlm-roberta-base"]
        snap_vocab = f"/cache/models--xlm-roberta-base/snapshots/{pinned}/tokenizer.json"
        cfg = PostStageBConfig(
            run_name="v8_no_local_fallback",
            run_mode="gpu_smoke_test",
            experiments=("A",),
            models=("xlm-roberta-base",),
            text_views=("canon_masked",),
            folds=(0,),
            epochs=1,
            batch_size=8,
            tokenizer_revisions={"xlm-roberta-base": pinned},
            smoke_max_train_samples=8,
            smoke_max_val_samples=8,
            smoke_max_eval_samples=8,
            run_heldout_eval=False,
            run_adversarial_eval=False,
            run_counterfactual_eval=False,
        )
        local_unit_root = self.root / "local_ckpt" / "u1"
        # Simulate unstaged local Attempt 1 checkpoint left on local SSD after failed Drive sync
        unstaged_local = _write_ckpt(local_unit_root / "1" / "checkpoint-1", 1)
        atomic_write_json(unstaged_local / _CHECKPOINT_COMMIT, _checkpoint_file_hashes(unstaged_local))
        # Drive staging is empty (sync never completed)
        staging_unit = self.root / "drive" / "staging" / "u1"
        events_dir = self.root / "drive" / "events"
        pointers_dir = self.root / "drive" / "pointers"
        seen_resume_args: list[object] = []

        fake_torch = types.SimpleNamespace(
            manual_seed=lambda s: None,
            tensor=lambda arr, dtype=None: np.asarray(arr, dtype=np.float32),
            float32=np.float32,
            softmax=lambda x, dim=-1: types.SimpleNamespace(
                numpy=lambda: np.exp(x - x.max(axis=-1, keepdims=True)) / np.exp(x - x.max(axis=-1, keepdims=True)).sum(axis=-1, keepdims=True)
            ),
            cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        )

        class _FakeDS:
            def __init__(self, payload: dict) -> None:
                self._n = len(next(iter(payload.values())))

            @classmethod
            def from_dict(cls, payload: dict):
                return cls(payload)

            def map(self, fn, batched=True, remove_columns=None):
                return self

            def __len__(self) -> int:
                return self._n

        class _FakeTrainer:
            def __init__(self, model=None, args=None, callbacks=None, **kw):
                self.args = args
                self.state = types.SimpleNamespace(global_step=1, best_model_checkpoint=str(Path(args.output_dir) / "checkpoint-1"), log_history=[])

            def train(self, resume_from_checkpoint=None):
                seen_resume_args.append(resume_from_checkpoint)

            def predict(self, ds):
                return types.SimpleNamespace(predictions=np.tile(np.array([[2.0, 0.0, 0.0]], dtype=np.float32), (len(ds), 1)))

        fake_tf = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda m, revision=None: _realistic_4571_tokenizer(vocab_file=snap_vocab, name_or_path=m)),
            AutoModelForSequenceClassification=types.SimpleNamespace(from_pretrained=lambda m, num_labels=3, revision=None: types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash=pinned))),
            DataCollatorWithPadding=lambda tokenizer=None: None,
            Trainer=_FakeTrainer,
            TrainerCallback=object,
            TrainingArguments=lambda **kw: types.SimpleNamespace(**kw),
            set_seed=lambda s: None,
        )
        with patch.dict(sys.modules, {"torch": fake_torch, "datasets": types.SimpleNamespace(Dataset=_FakeDS), "transformers": fake_tf}):
            out = hf_train_and_predict_fold(
                train_df=real_df,
                val_df=real_df,
                heldout_sets={},
                model_name="xlm-roberta-base",
                text_view="canon_masked",
                checkpoint_dir=local_unit_root / "2",
                cfg=cfg,
                class_weights=None,
                seed=42,
                resume_from_checkpoint=True,
                staging_unit_dir=staging_unit,
                events_dir=events_dir,
                pointers_dir=pointers_dir,
                uid="u1",
                attempt=2,
                config_hash=cfg.config_hash(),
                manifest_sha256="man_h",
            )
        # MUST NOT fall back to unstaged local Attempt 1 checkpoint!
        self.assertEqual(seen_resume_args, [None])
        self.assertIsNone(out["resumed_from_checkpoint"])

    # ------------------------------------------------------------------
    # Scenario E & F (HIGH): Cross-VM best checkpoint restoration + FP16 scaler.pt validation
    # ------------------------------------------------------------------
    def test_scenarios_e_and_f_cross_vm_best_restore_and_fp16_scaler_validation(self) -> None:
        # Scenario F: _valid_checkpoint & _find_latest_trainer_checkpoint enforce require_scaler=True
        ckpt_no_scaler = _write_ckpt(self.root / "fp16_unit" / "1" / "checkpoint-10", 10, with_scaler=False)
        atomic_write_json(ckpt_no_scaler / _CHECKPOINT_COMMIT, _checkpoint_file_hashes(ckpt_no_scaler, require_scaler=False))
        self.assertTrue(_valid_checkpoint(ckpt_no_scaler, require_scaler=False))
        self.assertFalse(_valid_checkpoint(ckpt_no_scaler, require_scaler=True))
        self.assertIsNone(_find_latest_trainer_checkpoint(self.root / "fp16_unit" / "2", require_scaler=True))

        ckpt_with_scaler = _write_ckpt(self.root / "fp16_unit" / "1" / "checkpoint-20", 20, with_scaler=True)
        atomic_write_json(ckpt_with_scaler / _CHECKPOINT_COMMIT, _checkpoint_file_hashes(ckpt_with_scaler, require_scaler=True))
        self.assertTrue(_valid_checkpoint(ckpt_with_scaler, require_scaler=True))
        self.assertEqual(
            _find_latest_trainer_checkpoint(self.root / "fp16_unit" / "2", require_scaler=True),
            str(ckpt_with_scaler),
        )

        # Scenario E: Cross-VM best-checkpoint restoration + _verify_trainer_checkpoint accepts restored_<hex6>_checkpoint-*
        staging = self.root / "vm_drive" / "staging" / "u1"
        events = self.root / "vm_drive" / "events"
        pointers = self.root / "vm_drive" / "pointers"
        vm1_root = self.root / "vm1_ssd"
        c1 = _write_ckpt(vm1_root / "checkpoint-10", 10, best_ckpt=str(vm1_root / "checkpoint-10"), with_scaler=True)
        sync_local_checkpoint_to_staging(
            local_ckpt_dir=c1, staging_unit_dir=staging, events_dir=events, pointers_dir=pointers,
            uid="u1", attempt=1, config_hash="ch", manifest_sha256="mh", require_scaler=True,
        )
        c2 = _write_ckpt(vm1_root / "checkpoint-20", 20, best_ckpt=str(vm1_root / "checkpoint-10"), with_scaler=True)
        sync_local_checkpoint_to_staging(
            local_ckpt_dir=c2, staging_unit_dir=staging, events_dir=events, pointers_dir=pointers,
            uid="u1", attempt=1, config_hash="ch", manifest_sha256="mh", require_scaler=True,
        )
        # Wipe VM 1 local SSD completely
        shutil.rmtree(vm1_root)
        scan = scan_unit_staging(
            staging_unit_dir=staging, events_dir=events, pointers_dir=pointers,
            uid="u1", expected_config_hash="ch", expected_manifest_sha256="mh",
            checkpoint_retention_limit=1, active_attempt=2, trust_commit_events=False,
        )
        vm2_root = self.root / "vm2_ssd" / "u1" / "2"
        rec = recover_latest_valid_checkpoint(scan_report=scan, local_restore_root=vm2_root)
        self.assertTrue(rec["recovered"])
        self.assertTrue(_valid_checkpoint(rec["restored_local_ckpt"], require_scaler=True))
        self.assertTrue(_valid_checkpoint(rec["restored_best_local_ckpt"], require_scaler=True))
        # Verify restored_<hex6>_checkpoint-<step> path format is accepted by _verify_trainer_checkpoint
        rel_best = rec["restored_best_local_ckpt"].resolve().relative_to(vm2_root.parent.resolve()).as_posix()
        self.assertTrue(_verify_trainer_checkpoint(vm2_root.parent, rel_best, require_scaler=True))

    # ------------------------------------------------------------------
    # Scenario G (HIGH): Lease safety (live same-host PID, TOCTOU, empty-active takeover, conflict preservation)
    # ------------------------------------------------------------------
    def test_scenario_g_lease_safety_live_pid_toctou_and_conflict_preservation(self) -> None:
        # G1. Same-host live PID (another live process, e.g., parent PID) blocks stale takeover fail-closed!
        art_live = self.root / "lease_live_pid"
        other_live_pid = os.getppid()
        with patch("bangla_smish.post_stage_b_checkpointing.os.getpid", return_value=other_live_pid):
            live_lease = register_single_writer_lease(art_live, writer_id="w_live_pid")
        self.assertEqual(live_lease["pid"], other_live_pid)
        with self.assertRaisesRegex(RuntimeError, "is still alive"):
            register_single_writer_lease(
                art_live,
                writer_id="w_takeover",
                allow_stale_takeover=True,
                expected_stale_writer_ids=["w_live_pid"],
                takeover_reason="attempted takeover of live process",
            )
        # Release cleanly
        release_single_writer_lease(art_live, "w_live_pid")

        # G2. Non-empty expected_stale_writer_ids when NO active leases exist fails closed!
        with self.assertRaisesRegex(RuntimeError, "no active writer leases exist"):
            register_single_writer_lease(
                art_live,
                writer_id="w_ghost",
                allow_stale_takeover=True,
                expected_stale_writer_ids=["nonexistent_writer"],
                takeover_reason="stale id mismatch",
            )

        # G3. Post-write TOCTOU race detection preserves both lease files!
        art_race = self.root / "lease_toctou"
        from bangla_smish import post_stage_b_checkpointing as ckpt_mod
        orig_write = ckpt_mod._write_new_immutable_json

        def _inject_concurrent_writer(path: Path, payload: dict, **kw) -> None:
            orig_write(path, payload, **kw)
            if path.name.startswith("lease_") and payload.get("writer_id") == "w_a":
                # Simulate concurrent writer w_b writing its lease in the TOCTOU window
                other = path.parent / "lease_w_b.json"
                orig_write(other, {**payload, "writer_id": "w_b", "pid": 999999}, **kw)

        with patch("bangla_smish.post_stage_b_checkpointing._write_new_immutable_json", side_effect=_inject_concurrent_writer):
            with self.assertRaisesRegex(RuntimeError, "Conflicting active writer detected mid-run"):
                register_single_writer_lease(art_race, writer_id="w_a")
        # Both lease files preserved as conflict evidence!
        lease_files = list((art_race / "leases").glob("lease_*.json"))
        self.assertEqual(len(lease_files), 2)

        # G4. Mid-run supersession/conflict aborts BEFORE commit AND finally: does NOT release conflicted lease!
        art_mid = self.root / "lease_mid_conflict"
        cfg_sm = PostStageBConfig(
            run_name="v8_mid_lease",
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

        def _train_with_rival_appearing(**kw):
            # Inject valid rival active lease while training is running
            rival_path = art_mid / "leases" / "lease_rival_writer.json"
            orig_write(
                rival_path,
                {
                    "record_type": "lease_acquired",
                    "writer_id": "rival_writer",
                    "status": "active",
                    "pid": 12345,
                    "hostname": "other-vm",
                },
                embed_digest_key="lease_sha256",
            )
            return {
                "train_seconds": 0.01,
                "actual_global_step": 1,
                "predictions": {"val_probs": np.eye(3)[kw["val_df"]["label_idx"].to_numpy(dtype=int)] * 0.94 + 0.02},
            }

        with self.assertRaisesRegex(RuntimeError, "Conflicting active writer detected mid-run"):
            run_post_stage_b_pipeline(bundle_dir=V2_BUNDLE, artifacts_dir=art_mid, cfg=cfg_sm, train_eval_fn=_train_with_rival_appearing)
        # Unit was NOT committed and BOTH leases remain active (conflict evidence preserved; finally did not auto-release!)
        st_mid = json.loads((art_mid / "run_state.json").read_text(encoding="utf-8"))
        self.assertEqual(st_mid["committed_units"], {})
        lease_after = inspect_writer_leases(art_mid)
        self.assertEqual(len(lease_after["active_leases"]), 2)

    # ------------------------------------------------------------------
    # Scenario H (HIGH): Resume timing accuracy (attempt vs cumulative seconds & steps_executed_this_attempt)
    # ------------------------------------------------------------------
    def test_scenario_h_resume_timing_accuracy(self) -> None:
        real_df = pd.read_parquet(V2_BUNDLE / "real_train_pool.parquet").head(8).reset_index(drop=True)
        pinned = PINNED_TOKENIZER_REVISIONS["xlm-roberta-base"]
        snap_vocab = f"/cache/models--xlm-roberta-base/snapshots/{pinned}/tokenizer.json"
        cfg = PostStageBConfig(
            run_name="v8_resume_timing",
            run_mode="gpu_smoke_test",
            experiments=("A",),
            models=("xlm-roberta-base",),
            text_views=("canon_masked",),
            folds=(0,),
            epochs=3,
            batch_size=8,
            tokenizer_revisions={"xlm-roberta-base": pinned},
            smoke_max_train_samples=8,
            smoke_max_val_samples=8,
            smoke_max_eval_samples=8,
            run_heldout_eval=False,
            run_adversarial_eval=False,
            run_counterfactual_eval=False,
        )
        # Stage an Epoch-2 checkpoint (step=10) with stage4_timing.json recording 20.0s cumulative / 18.0s pure opt
        staging_unit = self.root / "time_drive" / "staging" / "u1"
        events_dir = self.root / "time_drive" / "events"
        pointers_dir = self.root / "time_drive" / "pointers"
        ckpt_ep2 = _write_ckpt(
            self.root / "ep2_src" / "checkpoint-10",
            10,
            best_ckpt=str(self.root / "ep2_src" / "checkpoint-10"),
            timing_meta={
                "schema_version": "stage4_timing_v1",
                "global_step": 10,
                "resumed_from_step": 0,
                "attempt_train_seconds_at_save": 20.0,
                "cumulative_train_seconds": 20.0,
                "cumulative_pure_opt_seconds": 18.0,
                "cumulative_eval_seconds": 1.0,
                "cumulative_local_ckpt_seconds": 0.5,
                "cumulative_drive_sync_seconds": 0.5,
            },
        )
        sync_local_checkpoint_to_staging(
            local_ckpt_dir=ckpt_ep2,
            staging_unit_dir=staging_unit,
            events_dir=events_dir,
            pointers_dir=pointers_dir,
            uid="u1",
            attempt=1,
            config_hash=cfg.config_hash(),
            manifest_sha256="man_h",
        )

        fake_torch = types.SimpleNamespace(
            manual_seed=lambda s: None,
            tensor=lambda arr, dtype=None: np.asarray(arr, dtype=np.float32),
            float32=np.float32,
            softmax=lambda x, dim=-1: types.SimpleNamespace(
                numpy=lambda: np.exp(x - x.max(axis=-1, keepdims=True)) / np.exp(x - x.max(axis=-1, keepdims=True)).sum(axis=-1, keepdims=True)
            ),
            cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        )

        class _FakeDS:
            def __init__(self, payload: dict) -> None:
                self._n = len(next(iter(payload.values())))

            @classmethod
            def from_dict(cls, payload: dict):
                return cls(payload)

            def map(self, fn, batched=True, remove_columns=None):
                return self

            def __len__(self) -> int:
                return self._n

        class _FakeResumeTrainer:
            def __init__(self, model=None, args=None, callbacks=None, **kw):
                self.args = args
                self.state = types.SimpleNamespace(
                    global_step=15, best_model_checkpoint=str(Path(args.output_dir) / "checkpoint-15"), log_history=[]
                )

            def train(self, resume_from_checkpoint=None):
                assert resume_from_checkpoint is not None
                time.sleep(0.05)

            def predict(self, ds):
                return types.SimpleNamespace(predictions=np.tile(np.array([[2.0, 0.0, 0.0]], dtype=np.float32), (len(ds), 1)))

        fake_tf = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda m, revision=None: _realistic_4571_tokenizer(vocab_file=snap_vocab, name_or_path=m)),
            AutoModelForSequenceClassification=types.SimpleNamespace(from_pretrained=lambda m, num_labels=3, revision=None: types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash=pinned))),
            DataCollatorWithPadding=lambda tokenizer=None: None,
            Trainer=_FakeResumeTrainer,
            TrainerCallback=object,
            TrainingArguments=lambda **kw: types.SimpleNamespace(**kw),
            set_seed=lambda s: None,
        )
        with patch.dict(sys.modules, {"torch": fake_torch, "datasets": types.SimpleNamespace(Dataset=_FakeDS), "transformers": fake_tf}):
            out = hf_train_and_predict_fold(
                train_df=real_df,
                val_df=real_df,
                heldout_sets={},
                model_name="xlm-roberta-base",
                text_view="canon_masked",
                checkpoint_dir=self.root / "ssd_resume" / "2",
                cfg=cfg,
                class_weights=None,
                seed=42,
                resume_from_checkpoint=True,
                staging_unit_dir=staging_unit,
                events_dir=events_dir,
                pointers_dir=pointers_dir,
                uid="u1",
                attempt=2,
                config_hash=cfg.config_hash(),
                manifest_sha256="man_h",
            )
        self.assertEqual(out["resumed_from_step"], 10)
        self.assertEqual(out["actual_global_step"], 15)
        self.assertEqual(out["steps_executed_this_attempt"], 5)
        self.assertEqual(out["prior_resumed_train_seconds"], 20.0)
        self.assertGreater(out["attempt_train_seconds"], 0.03)
        self.assertLess(out["attempt_train_seconds"], 5.0)
        self.assertGreaterEqual(out["cumulative_train_seconds"], 20.03)
        self.assertEqual(out["train_seconds"], out["cumulative_train_seconds"])
        # Cumulative ms/step (~18050ms / 15 = ~1203ms) must NOT be collapsed to ~50ms/15 = ~3ms!
        self.assertGreater(out["optimizer_step_ms"], 1000.0)
        # Attempt ms/step divides attempt pure-opt time by 5 executed steps (not 15!)
        self.assertGreater(out["attempt_optimizer_step_ms"], 5.0)
        self.assertLess(out["attempt_optimizer_step_ms"], 500.0)

    # ------------------------------------------------------------------
    # Scenario I (MEDIUM): Persistence, FUSE ENOSYS, orphan tmp, backup, out-of-order upload, last-ckpt prune safety
    # ------------------------------------------------------------------
    def test_scenario_i_persistence_fuse_enosys_orphan_tmp_backup_and_upload_safety(self) -> None:
        # I1. FUSE ENOSYS tolerated on both file fsync and dir fsync (EIO still raises)
        target = self.root / "fuse_dir" / "out.json"
        with patch("bangla_smish.post_stage_b_runner.os.fsync", side_effect=OSError(errno.ENOSYS, "Function not implemented")):
            atomic_write_text(target, '{"ok": true}')
            _fsync_dir_best_effort(target.parent)
        self.assertEqual(target.read_text(encoding="utf-8"), '{"ok": true}')
        # Also tolerated in _write_prediction_archive
        df_mini = pd.DataFrame({"row_id": ["r1", "r2"], "label_idx": [0, 1]})
        probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32)
        with patch("bangla_smish.post_stage_b_runner.os.fsync", side_effect=OSError(errno.ENOSYS, "Function not implemented")):
            _write_prediction_archive(self.root / "fuse_dir" / "preds.npz", {"val_probs": probs}, {"val": df_mini})
        self.assertTrue((self.root / "fuse_dir" / "preds.npz").is_file())

        # I2. Orphan .run_state.json.tmp.<pid>.<hex8> in fresh artifact dir is cleaned safely, while exports/ STILL fails!
        cfg_sm = PostStageBConfig(
            run_name="v8_orphan_test",
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
        art_orphan = self.root / "art_orphan_tmp"
        art_orphan.mkdir()
        orphan_file = art_orphan / ".run_state.json.tmp.99999.deadbeef"
        orphan_file.write_text('{"partial":', encoding="utf-8")
        ok_mock = lambda **kw: {
            "train_seconds": 0.01,
            "actual_global_step": 1,
            "predictions": {"val_probs": np.eye(3)[kw["val_df"]["label_idx"].to_numpy(dtype=int)] * 0.94 + 0.02},
        }
        run_post_stage_b_pipeline(bundle_dir=V2_BUNDLE, artifacts_dir=art_orphan, cfg=cfg_sm, train_eval_fn=ok_mock)
        self.assertFalse(orphan_file.exists())
        # run_state.backup.json was created on subsequent state writes!
        self.assertTrue((art_orphan / "run_state.backup.json").is_file())

        # Corrupt run_state.json raises clear error pointing to run_state.backup.json
        (art_orphan / "run_state.json").write_text("{corrupt", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "run_state.backup.json"):
            run_post_stage_b_pipeline(bundle_dir=V2_BUNDLE, artifacts_dir=art_orphan, cfg=cfg_sm, train_eval_fn=ok_mock)

        # Stray exports/ in uninitialized dir STILL triggers unidentified run guard!
        art_stray = self.root / "art_stray_exports"
        (art_stray / "exports").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "unidentified run"):
            run_post_stage_b_pipeline(bundle_dir=V2_BUNDLE, artifacts_dir=art_stray, cfg=cfg_sm, train_eval_fn=ok_mock)

        # I3. Out-of-order Drive upload (manifest before payload, NO CHECKPOINT_SYNCED event) -> INCOMPLETE_UPLOAD
        # AND never prunes the last valid checkpoint before replacement has a verified CHECKPOINT_SYNCED event!
        staging = self.root / "ooo_drive" / "staging" / "u1"
        events = self.root / "ooo_drive" / "events"
        pointers = self.root / "ooo_drive" / "pointers"
        c1 = _write_ckpt(self.root / "ooo_src" / "checkpoint-10", 10)
        sync_local_checkpoint_to_staging(
            local_ckpt_dir=c1, staging_unit_dir=staging, events_dir=events, pointers_dir=pointers,
            uid="u1", attempt=1, config_hash="ch", manifest_sha256="mh",
        )
        # Simulate out-of-order upload of step 20: manifest arrived, model.safetensors missing, NO CHECKPOINT_SYNCED event!
        c20 = _write_ckpt(self.root / "ooo_src" / "checkpoint-20", 20)
        res20 = sync_local_checkpoint_to_staging(
            local_ckpt_dir=c20, staging_unit_dir=staging, events_dir=events, pointers_dir=pointers,
            uid="u1", attempt=1, config_hash="ch", manifest_sha256="mh",
        )
        (Path(res20["staged_ckpt_dir"]) / "model.safetensors").unlink()
        Path(res20["event_path"]).unlink()
        # Simulate un-evented step 15 (all files valid on disk, but CHECKPOINT_SYNCED event never landed)
        c15 = _write_ckpt(self.root / "ooo_src" / "checkpoint-15", 15)
        res15 = sync_local_checkpoint_to_staging(
            local_ckpt_dir=c15, staging_unit_dir=staging, events_dir=events, pointers_dir=pointers,
            uid="u1", attempt=1, config_hash="ch", manifest_sha256="mh",
        )
        Path(res15["event_path"]).unlink()
        scan = scan_unit_staging(
            staging_unit_dir=staging, events_dir=events, pointers_dir=pointers,
            uid="u1", expected_config_hash="ch", expected_manifest_sha256="mh",
            checkpoint_retention_limit=1, active_attempt=1, trust_commit_events=False,
        )
        # Step 20 is INCOMPLETE_UPLOAD (not CORRUPT_OR_UNVERIFIABLE!)
        inc_entries = [e for e in scan["entries"] if e["state"] == CheckpointState.INCOMPLETE_UPLOAD]
        corrupt_entries = [e for e in scan["entries"] if e["state"] == CheckpointState.CORRUPT_OR_UNVERIFIABLE]
        sup_entries = [e for e in scan["entries"] if e["state"] == CheckpointState.ALREADY_SUPERSEDED]
        valid_entries = [e for e in scan["entries"] if e["state"] == CheckpointState.VALID_RECOVERABLE]
        self.assertEqual([e["global_step"] for e in inc_entries], [20])
        self.assertEqual(corrupt_entries, [])
        # Step 10 (committed via CHECKPOINT_SYNCED) is NOT superseded by un-evented step 15!
        self.assertEqual(sup_entries, [])
        valid_steps = {e["global_step"] for e in valid_entries}
        self.assertIn(10, valid_steps)

    # ------------------------------------------------------------------
    # Scenario J (MEDIUM): Full 18-module source_sha256, schema_version=5, strict config, bootstrap seed
    # ------------------------------------------------------------------
    def test_scenario_j_identity_completeness_strict_config_and_bootstrap_seed(self) -> None:
        cfg_opt = load_config(ROOT / "experiments" / "colab" / "configs" / "post_stage_b_stage4_option_a_config.json")
        ident = _run_identity(V2_BUNDLE, cfg_opt, hf_train_and_predict_fold)
        self.assertEqual(ident["schema_version"], 5)
        expected_18 = {Path(rel).name for rel in STAGE4_KIT_REQUIRED_SOURCE_FILES}
        self.assertEqual(len(expected_18), 18)
        self.assertEqual(set(ident["source_sha256"].keys()), expected_18)

        # Strict run_name, seed, folds, and confirmatory_mode locks
        for bad_run_name in ("", "../escape", "sub/dir", "bad space", ".hidden"):
            with self.assertRaisesRegex(ValueError, "Invalid run_name"):
                replace(cfg_opt, run_name=bad_run_name)
        for bad_seed in (True, False, -1, "42"):
            with self.assertRaisesRegex(ValueError, "seed must be a non-negative integer"):
                replace(cfg_opt, seed=bad_seed)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "Invalid folds"):
            PostStageBConfig(run_name="ok_name", run_mode="gpu_smoke_test", folds=(True,))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "Option A confirmatory_mode requires run_heldout_eval=True"):
            replace(cfg_opt, run_heldout_eval=False)
        with self.assertRaisesRegex(ValueError, "frozen hyperparameters"):
            replace(cfg_opt, batch_size=16)
        with self.assertRaisesRegex(ValueError, "frozen hyperparameters"):
            replace(cfg_opt, learning_rate=5e-5)

        # Bootstrap seed propagation produces deterministic results for same seed and distinct for different seed
        rng = np.random.default_rng(7)
        y = np.array([0, 1, 2] * 30, dtype=int)
        probs = rng.dirichlet([1.5, 1.5, 1.5], size=len(y))
        fams = np.array([f"fam_{i // 3}" for i in range(len(y))])
        e1 = evaluate_predictions(y, probs, label_names=("Normal", "Promotional", "Smishing"), family_ids=fams, n_boot=80, seed=42)
        e2 = evaluate_predictions(y, probs, label_names=("Normal", "Promotional", "Smishing"), family_ids=fams, n_boot=80, seed=42)
        e3 = evaluate_predictions(y, probs, label_names=("Normal", "Promotional", "Smishing"), family_ids=fams, n_boot=80, seed=999)
        self.assertEqual(e1["macro_f1_ci95"], e2["macro_f1_ci95"])
        self.assertNotEqual(e1["macro_f1_ci95"], e3["macro_f1_ci95"])
        p1 = paired_family_accuracy_delta(y, (y + 1) % 3, probs.argmax(1), fams, n_resamples=80, seed=42)
        p2 = paired_family_accuracy_delta(y, (y + 1) % 3, probs.argmax(1), fams, n_resamples=80, seed=42)
        self.assertEqual(p1["accuracy_delta_ci95_pp"], p2["accuracy_delta_ci95_pp"])

    # ------------------------------------------------------------------
    # Scenario K (MEDIUM): Multi-GPU rejection across ALL modes + gpu_smoke_test package check
    # ------------------------------------------------------------------
    def test_scenario_k_multi_gpu_rejected_all_modes_and_smoke_pkg_check(self) -> None:
        cfg_smoke = load_config(ROOT / "experiments" / "colab" / "configs" / "gpu_smoke_test_config.json")
        cfg_t4 = replace(
            load_config(ROOT / "experiments" / "colab" / "configs" / "post_stage_b_stage4_t4_benchmark_config.json"),
            benchmark_workload_fixture_eval=False,
        )
        cfg_opt = load_config(ROOT / "experiments" / "colab" / "configs" / "post_stage_b_stage4_option_a_config.json")

        for mode_label, cfg_obj, expected_re in (
            ("gpu_smoke_test", cfg_smoke, "does not support multi-GPU execution"),
            ("t4_benchmark", cfg_t4, "exactly 1 Tesla T4"),
            ("experiments_abcd", cfg_opt, "does not support multi-GPU execution"),
        ):
            with self.subTest(mode=mode_label):
                art = self.root / f"multi_gpu_{mode_label}"
                with (
                    patch("bangla_smish.post_stage_b_runner.collect_hardware_fingerprint", return_value=_mock_t4_hw(gpu_count=2)),
                    patch("bangla_smish.post_stage_b_runner.register_single_writer_lease") as spy_lease,
                ):
                    with self.assertRaisesRegex(RuntimeError, expected_re):
                        run_post_stage_b_pipeline(bundle_dir=V2_BUNDLE, artifacts_dir=art, cfg=cfg_obj)
                    spy_lease.assert_not_called()
                    self.assertFalse(art.exists())

        # gpu_smoke_test with real hf_train_and_predict_fold rejects missing required packages before lease
        bad_fp = {**_mock_valid_env_fp(), "packages": {**_mock_valid_env_fp()["packages"], "tokenizers": "not_installed"}}
        art_sm_pkg = self.root / "smoke_missing_pkg"
        with (
            patch("bangla_smish.post_stage_b_runner.collect_hardware_fingerprint", return_value=_mock_t4_hw(gpu_count=1)),
            patch("bangla_smish.post_stage_b_runner.collect_stable_environment_fingerprint", return_value=bad_fp),
            patch("bangla_smish.post_stage_b_runner.register_single_writer_lease") as spy_lease2,
        ):
            with self.assertRaisesRegex(RuntimeError, "tokenizers"):
                run_post_stage_b_pipeline(bundle_dir=V2_BUNDLE, artifacts_dir=art_sm_pkg, cfg=cfg_smoke)
            spy_lease2.assert_not_called()
            self.assertFalse(art_sm_pkg.exists())

    # ------------------------------------------------------------------
    # Scenario L (MEDIUM): Offline CPU aggregation + durable kept checkpoint after local SSD wipe
    # ------------------------------------------------------------------
    def test_scenario_l_offline_cpu_aggregation_and_durable_kept_checkpoint(self) -> None:
        local_ssd = self.root / "ephemeral_colab_ssd"
        art = self.root / "gpu_run_art"
        cfg_keep = PostStageBConfig(
            run_name="v8_durable_keep",
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
            keep_model_weights="all",
            local_checkpoint_root=str(local_ssd),
        )
        uid0 = "exp_A__xlm-roberta-base__canon_masked__fold_0"

        def _fake_hf(**kw):
            cpath = _write_ckpt(kw["checkpoint_dir"] / "checkpoint-3", 3, with_scaler=True)
            atomic_write_json(cpath / _CHECKPOINT_COMMIT, _checkpoint_file_hashes(cpath, require_scaler=True))
            bdir_out = kw["best_model_export_dir"]
            bdir_out.mkdir(parents=True, exist_ok=True)
            (bdir_out / "config.json").write_text("{}", encoding="utf-8")
            (bdir_out / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (bdir_out / "model.safetensors").write_bytes(b"weights")
            (bdir_out / "vocab.txt").write_text("tok\n", encoding="utf-8")
            return {
                "train_seconds": 0.02,
                "actual_global_step": 3,
                "fp16": True,
                "best_trainer_checkpoint": str(cpath),
                "predictions": {"val_probs": np.eye(3)[kw["val_df"]["label_idx"].to_numpy(dtype=int)] * 0.94 + 0.02},
            }

        _fake_hf.__module__ = hf_train_and_predict_fold.__module__
        _fake_hf.__qualname__ = hf_train_and_predict_fold.__qualname__

        # Run on mock Tesla T4 GPU (recording the real hf_train_and_predict_fold runner identity)
        with (
            patch("bangla_smish.post_stage_b_runner.collect_hardware_fingerprint", return_value=_mock_t4_hw()),
            patch("bangla_smish.post_stage_b_runner.collect_stable_environment_fingerprint", return_value=_mock_valid_env_fp()),
            patch("bangla_smish.post_stage_b_runner.hf_train_and_predict_fold", new=_fake_hf),
        ):
            run_post_stage_b_pipeline(bundle_dir=V2_BUNDLE, artifacts_dir=art, cfg=cfg_keep)

        # Wipe ephemeral Colab local SSD completely!
        shutil.rmtree(local_ssd)
        self.assertFalse(local_ssd.exists())
        st = json.loads((art / "run_state.json").read_text(encoding="utf-8"))
        self.assertEqual(st["committed_units"][uid0]["trainer_checkpoint"]["storage"], "artifacts_dir")
        self.assertTrue((art / "checkpoints" / uid0 / "1" / "checkpoint-3").is_dir())

        # Offline CPU aggregation (cuda_available=False, different env) SUCCEEDS and verifies durable checkpoint!
        cpu_hw = {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_name": None,
            "compute_capability": None,
            "nvidia_driver_version": None,
            "torch_cuda_version": None,
        }
        with patch("bangla_smish.post_stage_b_runner.collect_hardware_fingerprint", return_value=cpu_hw):
            agg = aggregate_post_stage_b_results(V2_BUNDLE, art, cfg_keep, offline_cpu_ok=True)
        self.assertEqual(agg["completed_unit_count"], 1)
        self.assertTrue(agg["aggregated_offline_hardware_diff"])

        # Offline aggregation STILL fails closed if config_hash or source_sha256 mismatches!
        with patch("bangla_smish.post_stage_b_runner.collect_hardware_fingerprint", return_value=cpu_hw):
            with self.assertRaisesRegex(ValueError, "cannot aggregate offline"):
                aggregate_post_stage_b_results(V2_BUNDLE, art, replace(cfg_keep, seed=99), offline_cpu_ok=True)

    # ------------------------------------------------------------------
    # Scenario M: Frozen T4 config untouched, fail-closed pre-lease, and kit ignores Thumbs.db / ._ files
    # ------------------------------------------------------------------
    def test_scenario_m_frozen_t4_config_untouched_and_kit_os_metadata(self) -> None:
        t4_path = ROOT / "experiments" / "colab" / "configs" / "post_stage_b_stage4_t4_benchmark_config.json"
        t4_cfg = load_config(t4_path)
        self.assertTrue(t4_cfg.benchmark_workload_fixture_eval)
        self.assertFalse(SYNTHETIC_WORKLOAD_FIXTURE_ENABLED)

        # Direct hf_train_and_predict_fold ALSO fails closed before torch/model init
        real_df = pd.read_parquet(V2_BUNDLE / "real_train_pool.parquet").head(4).reset_index(drop=True)
        with self.assertRaisesRegex(RuntimeError, "synthetic workload fixture execution is blocked"):
            hf_train_and_predict_fold(
                train_df=real_df,
                val_df=real_df,
                heldout_sets={},
                model_name="xlm-roberta-base",
                text_view="canon_masked",
                checkpoint_dir=self.root / "direct_t4_ckpt",
                cfg=t4_cfg,
                class_weights=None,
                seed=42,
            )

        # Kit verifier ignores Thumbs.db and AppleDouble ._ files while still rejecting rogue files
        import importlib.util
        kit_script = ROOT / "scripts" / "prepare_stage4_colab_kit.py"
        spec = importlib.util.spec_from_file_location("prepare_stage4_colab_kit", kit_script)
        kit_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kit_mod)
        staged = self.root / "kit_v8_os"
        kit_mod.export_stage4_colab_kit(ROOT, V2_BUNDLE, staged)
        (staged / "Thumbs.db").write_bytes(b"thumb")
        (staged / "src" / "bangla_smish" / "._post_stage_b_runner.py").write_bytes(b"appledouble")
        ok = verify_portable_colab_kit(staged)
        self.assertEqual(ok["reverified_frozen_masks"], 4)


if __name__ == "__main__":
    unittest.main()
