"""Reusable Post-Stage B GPU training, checkpointing, safe resume, evaluation, and export runner."""

from __future__ import annotations

import errno
import gc
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from .metrics import (
    evaluate_predictions,
    holm_adjust_pvalues,
    mcnemar_pvalue,
    paired_family_accuracy_delta,
)
from .post_stage_b_checkpointing import (
    CheckpointState,
    RECOVERY_GRANULARITY,
    STAGE4_TIMING_METADATA,
    _is_orphan_tmp_filename,
    commit_unit_completion_event,
    ensure_unit_completion_event,
    inspect_writer_leases,
    prune_superseded_staging,
    recover_latest_valid_checkpoint,
    register_single_writer_lease,
    release_single_writer_lease,
    scan_unit_staging,
    sync_local_checkpoint_to_staging,
    verify_writer_lease_still_sole_active,
)
from .post_stage_b_cohorts import build_smishdetect_test_cohort_masks
from .post_stage_b_data import (
    TEXT_VIEW_COLUMNS,
    build_experiment_fold_data,
    load_heldout_eval_sets,
    select_counterfactual_fold,
    sha256_file,
    verify_manifest,
)
from .post_stage_b_ensemble import build_ensemble_evaluation_summary
from .post_stage_b_integrity import STAGE4_KIT_REQUIRED_SOURCE_FILES
from .post_stage_b_model_view_audit import audit_model_view_overlaps
from .post_stage_b_stats import (
    CONFIRMATORY_DECLARATIONS_ENABLED,
    DEFAULT_B_CI,
    DEFAULT_B_PVAL,
    OPTION_A_BACKBONES,
    STATISTICAL_VALIDATION_STATUS,
)
from .post_stage_b_tokenization import (
    BANGLABERT_MODEL,
    BANGLABERT_NORMALIZER_COMMIT,
    extract_commit_hash_from_path,
    prepare_tokenizer_texts,
    resolve_and_verify_tokenizer_revision,
)
from .post_stage_b_tokenizer_gate import (
    ALLOWED_COLLISION_GATE_VIEWS,
    STAGE3_32_CANDIDATE_PAIRS,
    TOKENIZER_GATE_PACKAGE_DISTS,
)
from .schema import CANONICAL_LABELS

LANGUAGES_CANONICAL: tuple[str, ...] = ("Bengali", "Banglish", "CodeMix", "English")
OPTION_A_ALLOWED_VIEWS: tuple[str, ...] = ("canon_masked", "raw")
# Fail-closed guard enforcing the permanent Stage B stop against synthetic text concatenation
# during T4 benchmark execution until owner approves an unmodified dev-data-only alternative.
SYNTHETIC_WORKLOAD_FIXTURE_ENABLED: bool = False
# Runtime-critical package distributions mapped to (dist_name, import_name), reconciled against
# pyproject.toml, post_stage_b_colab_deps.py, upstream transformers==4.57.1 requires_dist, and
# pinned csebuetnlp/normalizer@d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9 (no unicodedata2).
_IDENTITY_PACKAGE_DISTS: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("pyarrow", "pyarrow"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("tokenizers", "tokenizers"),
    ("datasets", "datasets"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("sentencepiece", "sentencepiece"),
    ("packaging", "packaging"),
    ("huggingface_hub", "huggingface_hub"),
    ("protobuf", "google.protobuf"),
    ("regex", "regex"),
    ("pyyaml", "yaml"),
    ("requests", "requests"),
    ("tqdm", "tqdm"),
    ("filelock", "filelock"),
    ("normalizer", "normalizer"),
    ("ftfy", "ftfy"),
    ("emoji", "emoji"),
    ("wcwidth", "wcwidth"),
)
_IDENTITY_PACKAGES: tuple[str, ...] = tuple(dist for dist, _ in _IDENTITY_PACKAGE_DISTS)
_CORE_REQUIRED_PACKAGE_DISTS: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "pyarrow",
    "torch",
    "transformers",
    "tokenizers",
    "datasets",
    "accelerate",
    "safetensors",
    "sentencepiece",
    "packaging",
    "huggingface_hub",
    "protobuf",
    "regex",
    "pyyaml",
    "requests",
    "tqdm",
    "filelock",
)
_BANGLABERT_REQUIRED_PACKAGE_DISTS: tuple[str, ...] = (
    "normalizer",
    "ftfy",
    "emoji",
    "wcwidth",
)
STAGE4_T4_BENCHMARK_UNITS: tuple[tuple[str, str, str, int], ...] = (
    ("D", "csebuetnlp/banglabert", "canon_masked", 0),
    ("D", "google/muril-base-cased", "canon_masked", 0),
    ("D", "xlm-roberta-base", "canon_masked", 0),
    ("D", "distilbert-base-multilingual-cased", "canon_masked", 0),
    ("B", "xlm-roberta-base", "canon_masked", 0),
)


@dataclass
class PostStageBConfig:
    """Configuration for GPU smoke tests, T4 benchmarks, and Experiments A, B, C, and D."""

    run_name: str
    run_mode: str  # "gpu_smoke_test", "experiments_abcd", or "t4_benchmark"
    experiments: tuple[str, ...] = ("A", "B", "C", "D")
    models: tuple[str, ...] = (
        "csebuetnlp/banglabert",
        "google/muril-base-cased",
        "xlm-roberta-base",
        "distilbert-base-multilingual-cased",
    )
    text_views: tuple[str, ...] = ("raw", "canon_masked", "brand_masked", "brand_stripped")
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 3
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 128
    seed: int = 42
    fp16: str = "auto"  # "auto", "fp16", "bf16", "none"
    experiment_b_balance_mode: str = "class_weight"
    run_heldout_eval: bool = True
    run_adversarial_eval: bool = True
    run_counterfactual_eval: bool = True
    keep_model_weights: str = "none"  # "none", "best_fold0", "all"
    smoke_max_train_samples: int | None = None
    smoke_max_val_samples: int | None = None
    smoke_max_eval_samples: int | None = None
    gradient_accumulation_steps: int = 1
    confirmatory_mode: bool = False
    checkpoint_retention_limit: int = 1
    local_checkpoint_root: str | None = None
    unit_schedule: tuple[tuple[str, str, str, int], ...] | None = None
    benchmark_workload_fixture_eval: bool = False
    tokenizer_revisions: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_name, str)
            or not self.run_name
            or ".." in self.run_name
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.run_name)
        ):
            raise ValueError(
                f"Invalid run_name {self.run_name!r}: must match [A-Za-z0-9][A-Za-z0-9._-]* without path traversal"
            )
        if self.run_mode not in {"gpu_smoke_test", "experiments_abcd", "t4_benchmark"}:
            raise ValueError(f"Unsupported run_mode: {self.run_mode!r}")
        if self.experiment_b_balance_mode != "class_weight":
            raise ValueError("Experiment B requires the class_weight balance mode")
        for name, values, allowed in (
            ("experiments", self.experiments, {"A", "B", "C", "D"}),
            ("text_views", self.text_views, TEXT_VIEW_COLUMNS),
        ):
            if not values or len(set(values)) != len(values) or any(v not in allowed for v in values):
                raise ValueError(f"Invalid {name}: {values!r}")
        if (
            not self.folds
            or any(type(f) is not int or f not in set(range(5)) for f in self.folds)
            or len(set(self.folds)) != len(self.folds)
        ):
            raise ValueError(f"Invalid folds: {self.folds!r}")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {self.seed!r}")
        if not self.models or any(
            not re.fullmatch(r"[\w.-]+(?:/[\w.-]+)?", name) or ".." in name for name in self.models
        ):
            raise ValueError(f"Invalid models: {self.models!r}")
        model_slugs = {name.replace("/", "__") for name in self.models}
        if len(model_slugs) != len(self.models):
            raise ValueError("Model identifiers collide")
        if self.fp16 not in {"auto", "fp16", "bf16", "none"}:
            raise ValueError(f"Unsupported fp16 mode: {self.fp16!r}")
        if self.keep_model_weights not in {"none", "best_fold0", "all"}:
            raise ValueError(f"Unsupported keep_model_weights mode: {self.keep_model_weights!r}")
        for key in ("epochs", "batch_size", "eval_batch_size", "max_length", "gradient_accumulation_steps"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.checkpoint_retention_limit) is not int or self.checkpoint_retention_limit < 0:
            raise ValueError("checkpoint_retention_limit must be a non-negative integer")
        for key, valid in (
            ("learning_rate", lambda x: x > 0),
            ("weight_decay", lambda x: x >= 0),
            ("warmup_ratio", lambda x: 0 <= x <= 1),
        ):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or not valid(value):
                raise ValueError(f"Invalid {key}: {value!r}")
        for key in ("smoke_max_train_samples", "smoke_max_val_samples", "smoke_max_eval_samples"):
            value = getattr(self, key)
            if value is not None and (self.run_mode != "gpu_smoke_test" or type(value) is not int or value < 1):
                raise ValueError(f"{key} requires smoke mode and a positive integer")
        if self.unit_schedule is not None:
            if not self.unit_schedule or len(set(self.unit_schedule)) != len(self.unit_schedule):
                raise ValueError("unit_schedule must be non-empty without duplicate units")
            for exp_id, model_name, text_view, fold in self.unit_schedule:
                if (
                    type(fold) is not int
                    or exp_id not in set(self.experiments)
                    or model_name not in set(self.models)
                    or text_view not in set(self.text_views)
                    or fold not in set(self.folds)
                ):
                    raise ValueError(f"unit_schedule entry {(exp_id, model_name, text_view, fold)!r} outside configured factors")
        if self.confirmatory_mode:
            unauthorized = [v for v in self.text_views if v not in OPTION_A_ALLOWED_VIEWS]
            if unauthorized:
                raise ValueError(
                    f"Unauthorized text_views in confirmatory_mode: {unauthorized!r}; only {OPTION_A_ALLOWED_VIEWS} allowed"
                )
            if set(self.text_views) != set(OPTION_A_ALLOWED_VIEWS):
                raise ValueError(f"Option A confirmatory_mode requires both views {OPTION_A_ALLOWED_VIEWS}")
            if self.run_mode != "experiments_abcd":
                raise ValueError("Option A confirmatory_mode requires run_mode='experiments_abcd'")
            if self.experiments != ("A", "B", "C", "D") or self.models != OPTION_A_BACKBONES or self.folds != (0, 1, 2, 3, 4):
                raise ValueError("Option A confirmatory_mode requires exact 4x4x2x5 = 160-unit factorial grid")
            if self.epochs != 3 or self.seed != 42 or self.unit_schedule is not None:
                raise ValueError("Option A confirmatory_mode requires epochs=3, seed=42, and full 160-unit grid")
            if not (self.run_heldout_eval and self.run_adversarial_eval and self.run_counterfactual_eval):
                raise ValueError(
                    "Option A confirmatory_mode requires run_heldout_eval=True, run_adversarial_eval=True, and run_counterfactual_eval=True"
                )
            if (
                self.batch_size != 32
                or self.gradient_accumulation_steps != 1
                or self.max_length != 128
                or not math.isclose(float(self.learning_rate), 3e-5, rel_tol=0.0, abs_tol=1e-12)
                or not math.isclose(float(self.weight_decay), 0.01, rel_tol=0.0, abs_tol=1e-12)
                or not math.isclose(float(self.warmup_ratio), 0.1, rel_tol=0.0, abs_tol=1e-12)
                or self.benchmark_workload_fixture_eval
            ):
                raise ValueError(
                    "Option A confirmatory_mode requires frozen hyperparameters "
                    "(batch_size=32, gradient_accumulation_steps=1, max_length=128, "
                    "learning_rate=3e-5, weight_decay=0.01, warmup_ratio=0.1, benchmark_workload_fixture_eval=False)"
                )
        if self.run_mode == "t4_benchmark" and self.run_heldout_eval:
            raise ValueError("t4_benchmark must set run_heldout_eval=False to prevent held-out/OOD exposure during benchmarking")
        if self.tokenizer_revisions is not None:
            if not isinstance(self.tokenizer_revisions, dict):
                raise ValueError("tokenizer_revisions must be a dict mapping model_name -> 40-hex commit SHA")
            for mname, rev in self.tokenizer_revisions.items():
                if not re.fullmatch(r"[0-9a-f]{40}", str(rev)):
                    raise ValueError(f"Invalid 40-hex tokenizer revision for {mname!r}: {rev!r}")
        if self.confirmatory_mode or self.run_mode == "t4_benchmark":
            if not self.tokenizer_revisions or any(m not in self.tokenizer_revisions for m in self.models):
                raise ValueError("Stage 4 confirmatory_mode and t4_benchmark require pinned tokenizer_revisions for all models")

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> PostStageBConfig:
    """Load a PostStageBConfig from a JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    for tuple_field in ("experiments", "models", "text_views", "folds"):
        if tuple_field in raw and isinstance(raw[tuple_field], list):
            raw[tuple_field] = tuple(raw[tuple_field])
    if raw.get("unit_schedule") is not None:
        raw["unit_schedule"] = tuple(tuple(item) for item in raw["unit_schedule"])
    return PostStageBConfig(**raw)


def save_config(cfg: PostStageBConfig, path: str | Path) -> None:
    """Persist a PostStageBConfig atomically as JSON."""
    atomic_write_json(Path(path), asdict(cfg))


_FUSE_FSYNC_IGNORED_ERRNOS: frozenset[int] = frozenset(
    err
    for err in (
        errno.EINVAL,
        errno.EOPNOTSUPP,
        errno.EPERM,
        errno.EBADF,
        errno.ENOSYS,
        getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
    )
)
_RUN_STATE_ORPHAN_TMP_RE = re.compile(r"^\.run_state\.json\.tmp\.[0-9]+\.[0-9a-f]{8}$")


def _fsync_fd_best_effort(fd: int) -> None:
    """Best-effort fsync on a file descriptor; tolerates FUSE unsupported errnos (including ENOSYS) while propagating EIO/ENOSPC."""
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno not in _FUSE_FSYNC_IGNORED_ERRNOS:
            raise


def _fsync_dir_best_effort(directory: Path) -> None:
    """Best-effort POSIX directory fsync for local-SSD durability; tolerates FUSE mounts cleanly.

    Local POSIX filesystems (e.g., Colab local SSD ext4/overlayfs) persist rename directory entries
    when the parent directory fd is fsynced. Google Drive FUSE mounts do not provide POSIX directory
    fsync durability guarantees and may return EINVAL/EOPNOTSUPP/EPERM/EBADF/ENOSYS, which are tolerated cleanly.
    """
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        dir_fd = os.open(str(directory), flags)
    except OSError as exc:
        if exc.errno in _FUSE_FSYNC_IGNORED_ERRNOS:
            return
        raise
    try:
        _fsync_fd_best_effort(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically on POSIX local filesystems using PID+UUID temp file, best-effort fsync, replace, and dir fsync.

    On Google Drive FUSE mounts, rename/fsync semantics are best-effort only; Drive-safe checkpoint
    recovery relies on immutable UUID staging and SHA256 verification rather than FUSE fsync guarantees.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            _fsync_fd_best_effort(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    _fsync_dir_best_effort(path.parent)


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically using UTF-8 encoding."""
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _write_run_state(state_path: Path, state: dict[str, Any]) -> None:
    """Persist run_state.json atomically with a prior-valid backup and strict readback verification."""
    if state_path.is_file():
        try:
            prior_text = state_path.read_text(encoding="utf-8")
            json.loads(prior_text)
        except (OSError, json.JSONDecodeError):
            prior_text = None
        if prior_text is not None:
            atomic_write_text(state_path.with_name("run_state.backup.json"), prior_text)
    atomic_write_json(state_path, state)
    readback = json.loads(state_path.read_text(encoding="utf-8"))
    if readback != state:
        raise IOError(f"Readback verification mismatch after writing {state_path}")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (when installed) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed

        set_seed(seed)
    except ImportError:
        pass


def _resolve_package_version(dist_name: str, import_name: str) -> str:
    """Resolve exact package version via importlib.metadata first, never returning generic 'installed'."""
    try:
        ver = importlib_metadata.version(dist_name)
        if ver is not None and str(ver).strip():
            return str(ver).strip()
    except importlib_metadata.PackageNotFoundError:
        pass
    try:
        mod = importlib.import_module(import_name)
    except ImportError:
        return "not_installed"
    ver = getattr(mod, "__version__", None)
    if ver is not None and str(ver).strip():
        return str(ver).strip()
    return "unknown_version"


def _query_nvidia_driver_version() -> str | None:
    """Query exact NVIDIA driver version via nvidia-smi without raising on CPU-only hosts."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        return None
    versions = [line.strip() for line in out.splitlines() if line.strip()]
    if not versions:
        return None
    unique = sorted(set(versions))
    return unique[0] if len(unique) == 1 else ",".join(unique)


def collect_hardware_fingerprint() -> dict[str, Any]:
    """Collect a deterministic, timestamp-free GPU/driver hardware fingerprint for run-identity binding."""
    cuda_available = False
    gpu_count = 0
    gpu_name: str | None = None
    compute_capability: str | None = None
    torch_cuda_version: str | None = None
    try:
        import torch

        raw_cuda = getattr(getattr(torch, "version", None), "cuda", None)
        torch_cuda_version = str(raw_cuda) if raw_cuda is not None else None
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_count = int(torch.cuda.device_count())
            if gpu_count >= 1:
                gpu_name = str(torch.cuda.get_device_name(0))
                cap = torch.cuda.get_device_capability(0)
                if isinstance(cap, (tuple, list)) and len(cap) == 2:
                    compute_capability = f"{int(cap[0])}.{int(cap[1])}"
                elif cap is not None:
                    compute_capability = str(cap)
    except ImportError:
        pass
    return {
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "compute_capability": compute_capability,
        "nvidia_driver_version": _query_nvidia_driver_version() if cuda_available else None,
        "torch_cuda_version": torch_cuda_version,
    }


def verify_t4_benchmark_hardware(hw_fp: dict[str, Any]) -> None:
    """Enforce strict single-GPU Tesla T4 hardware requirements for t4_benchmark runs."""
    if not hw_fp.get("cuda_available"):
        raise RuntimeError("Fail closed: t4_benchmark requires a CUDA-enabled Tesla T4 GPU (cuda_available=False).")
    if hw_fp.get("gpu_count") != 1:
        raise RuntimeError(
            f"Fail closed: t4_benchmark requires exactly 1 Tesla T4 GPU, found gpu_count={hw_fp.get('gpu_count')!r}."
        )
    gpu_name = str(hw_fp.get("gpu_name") or "")
    if "tesla t4" not in gpu_name.lower():
        raise RuntimeError(f"Fail closed: t4_benchmark requires a Tesla T4 GPU, found gpu_name={gpu_name!r}.")
    if str(hw_fp.get("compute_capability") or "") != "7.5":
        raise RuntimeError(
            f"Fail closed: t4_benchmark requires Tesla T4 compute capability 7.5, "
            f"found {hw_fp.get('compute_capability')!r}."
        )
    driver = str(hw_fp.get("nvidia_driver_version") or "").strip()
    if not driver:
        raise RuntimeError(
            "Fail closed: t4_benchmark requires a verified NVIDIA driver version (nvidia-smi unavailable or empty)."
        )


def _validate_required_packages_fingerprint(env_fp: dict[str, Any], cfg: PostStageBConfig) -> None:
    """Reject missing or unversioned runtime-critical distributions before initializing real HF training."""
    packages = env_fp.get("packages") or {}
    required = list(_CORE_REQUIRED_PACKAGE_DISTS)
    uses_banglabert = BANGLABERT_MODEL in set(cfg.models)
    if uses_banglabert:
        required.extend(_BANGLABERT_REQUIRED_PACKAGE_DISTS)
    bad = {
        pkg: packages.get(pkg, "not_installed")
        for pkg in required
        if packages.get(pkg) in {None, "", "not_installed", "unknown_version", "installed"}
    }
    if bad:
        raise RuntimeError(
            f"Fail closed: required runtime package versions missing or unresolved in environment fingerprint: {bad}"
        )
    if uses_banglabert and env_fp.get("normalizer_commit") != BANGLABERT_NORMALIZER_COMMIT:
        raise RuntimeError(
            f"Fail closed: csebuetnlp/banglabert requires pinned normalizer commit {BANGLABERT_NORMALIZER_COMMIT!r}, "
            f"found {env_fp.get('normalizer_commit')!r}."
        )


def collect_stable_environment_fingerprint() -> dict[str, Any]:
    """Collect a deterministic, timestamp-free runtime/package fingerprint for run-identity binding."""
    packages = {
        dist_name: _resolve_package_version(dist_name, import_name)
        for dist_name, import_name in _IDENTITY_PACKAGE_DISTS
    }
    torch_cuda_version: str | None = None
    try:
        import torch

        raw_cuda = getattr(getattr(torch, "version", None), "cuda", None)
        torch_cuda_version = str(raw_cuda) if raw_cuda is not None else None
    except ImportError:
        pass
    normalizer_commit: str | None = None
    try:
        dist = importlib_metadata.distribution("normalizer")
        raw_origin = dist.read_text("direct_url.json")
        if raw_origin:
            origin = json.loads(raw_origin)
            commit = (origin.get("vcs_info") or {}).get("commit_id")
            if commit:
                normalizer_commit = str(commit)
    except Exception:
        pass
    return {
        "python_version": sys.version.split()[0],
        "torch_cuda_version": torch_cuda_version,
        "normalizer_commit": normalizer_commit,
        "packages": packages,
    }


def collect_environment_info() -> dict[str, Any]:
    """Collect runtime, GPU, library, and Colab environment diagnostics without credentials."""
    env_fp = collect_stable_environment_fingerprint()
    hw_fp = collect_hardware_fingerprint()
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": env_fp["python_version"],
        "platform": platform.platform(),
        "in_colab": "google.colab" in sys.modules or Path("/content").exists(),
        "cuda_available": hw_fp["cuda_available"],
        "gpu_name": hw_fp["gpu_name"],
        "gpu_vram_gb": None,
        "gpu_count": hw_fp["gpu_count"],
        "compute_capability": hw_fp["compute_capability"],
        "nvidia_driver_version": hw_fp["nvidia_driver_version"],
        "torch_cuda_version": hw_fp["torch_cuda_version"],
        "normalizer_commit": env_fp["normalizer_commit"],
        "packages": dict(env_fp["packages"]),
    }

    try:
        import torch

        if torch.cuda.is_available() and int(torch.cuda.device_count()) >= 1:
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_gb"] = round(float(props.total_memory) / (1024**3), 2)
    except ImportError:
        pass

    return info


def unit_id(experiment_id: str, model_name: str, text_view: str, fold: int) -> str:
    """Return a filesystem-safe deterministic identifier for one (experiment, model, view, fold) unit."""
    model_slug = model_name.replace("/", "__").replace(" ", "_")
    return f"exp_{experiment_id}__{model_slug}__{text_view}__fold_{fold}"


@contextmanager
def _run_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.Logger(f"post_stage_b_{log_path}")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    try:
        yield logger
    finally:
        fh.close()
        sh.close()


_CHECKPOINT_COMMIT = "post_stage_b_checkpoint_commit.json"


def _checkpoint_file_hashes(path: Path, *, require_scaler: bool = False) -> dict[str, str]:
    files = {
        item.name: item
        for item in path.iterdir()
        if item.is_file() and item.name != _CHECKPOINT_COMMIT and not _is_orphan_tmp_filename(item.name)
    }
    required = {"config.json", "trainer_state.json", "training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"}
    if require_scaler:
        required.add("scaler.pt")
    weights = {"model.safetensors", "pytorch_model.bin"}
    if not required <= files.keys() or not weights & files.keys():
        raise ValueError(f"Incomplete training checkpoint: {path}")
    if any(item.stat().st_size == 0 for item in files.values()):
        raise ValueError(f"Empty file in training checkpoint: {path}")
    trainer_state = json.loads(files["trainer_state.json"].read_text(encoding="utf-8"))
    if trainer_state["global_step"] != int(path.name.split("-")[-1]):
        raise ValueError(f"Training checkpoint step mismatch: {path}")
    return {name: sha256_file(item) for name, item in files.items()}


def _valid_checkpoint(path: Path, *, require_scaler: bool = False) -> bool:
    try:
        commit = json.loads((path / _CHECKPOINT_COMMIT).read_text(encoding="utf-8"))
        return commit == _checkpoint_file_hashes(path, require_scaler=require_scaler)
    except (OSError, ValueError, KeyError):
        return False


def _find_latest_trainer_checkpoint(ckpt_dir: Path, *, require_scaler: bool = False) -> str | None:
    """Search this and earlier attempts for the newest checksummed Trainer save."""
    candidates = []
    for path in ckpt_dir.parent.glob("*/checkpoint-*"):
        if path.parent.name.isdecimal() and re.fullmatch(r"checkpoint-[0-9]+", path.name):
            candidates.append((int(path.name.split("-")[-1]), int(path.parent.name), path))
    for _, _, path in sorted(candidates, reverse=True):
        if not _valid_checkpoint(path, require_scaler=require_scaler):
            continue
        state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        best = state.get("best_model_checkpoint")
        if not best:
            return str(path)
        # Reject a save that cannot restore the model Trainer selected in an earlier epoch.
        best_path = Path(best).absolute()
        if best_path.is_relative_to(ckpt_dir.parent.absolute()) and (
            best_path == path.absolute() or _valid_checkpoint(best_path, require_scaler=require_scaler)
        ):
            return str(path)
    return None


def _weighted_cross_entropy(logits: Any, labels: Any, weights: list[float]) -> Any:
    import torch

    logits = logits.float()
    weight = torch.as_tensor(weights, dtype=logits.dtype, device=logits.device)
    return torch.nn.functional.cross_entropy(logits, labels, weight=weight)


def _weighted_trainer_class(trainer_base: type, weights: list[float]) -> type:
    class WeightedTrainer(trainer_base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # The batch-mean loss consumes no Trainer-provided num_items_in_batch.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            if not model.training:
                return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            outputs = model(**{key: value for key, value in inputs.items() if key != "labels"})
            loss = _weighted_cross_entropy(outputs.logits, inputs["labels"], weights)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def _adversarial_column(df: pd.DataFrame, text_view: str) -> str:
    column = f"text_adv_{text_view}"
    if column in df:
        return column
    if text_view == "raw" and "text_adv" in df:
        return "text_adv"
    raise ValueError(f"Bundle lacks adversarial inputs processed for {text_view}; use the corrected bundle")


def _evaluate_frame(df: pd.DataFrame, probs: np.ndarray, *, n_boot: int, **kwargs: Any) -> dict[str, Any]:
    return evaluate_predictions(
        df["label_idx"].to_numpy(dtype=int),
        probs,
        label_names=CANONICAL_LABELS,
        family_ids=df["family_id"].to_numpy(),
        n_boot=n_boot,
        **kwargs,
    )


def _evaluate_by_language(df: pd.DataFrame, probs: np.ndarray, *, n_boot: int = 300, **kwargs: Any) -> dict[str, Any]:
    y = df["label_idx"].to_numpy(dtype=int)
    per_lang: dict[str, Any] = {}
    for lang in LANGUAGES_CANONICAL:
        mask = (df["language"] == lang).to_numpy()
        if int(mask.sum()) >= 5 and len(np.unique(y[mask])) >= 2:
            per_lang[lang] = _evaluate_frame(df.loc[mask], probs[mask], n_boot=n_boot, **kwargs)
        else:
            per_lang[lang] = {"n_samples": int(mask.sum()), "skipped": "insufficient_samples_or_classes"}
    return per_lang


def _resolve_precision(fp16_mode: str) -> tuple[bool, bool]:
    import torch

    if fp16_mode == "none":
        return False, False
    if not torch.cuda.is_available():
        if fp16_mode != "auto":
            raise RuntimeError(f"Requested {fp16_mode} requires CUDA")
        return False, False
    if fp16_mode == "fp16":
        return True, False
    if torch.cuda.is_bf16_supported(including_emulation=False):
        return False, True
    if fp16_mode == "bf16":
        raise RuntimeError("Requested bf16 requires native CUDA bfloat16 support")
    # T4 and V100 use FP16 in automatic mode.
    return True, False


def compute_optimizer_step_budget(
    n_train: int,
    *,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int = 1,
    arm_a_steps: int | None = None,
) -> dict[str, Any]:
    """Compute exact equal-epoch expected Trainer update-step budget and delta vs Arm A."""
    micro_batches_per_epoch = math.ceil(n_train / batch_size)
    steps_per_epoch = math.ceil(micro_batches_per_epoch / gradient_accumulation_steps)
    expected_steps = steps_per_epoch * epochs
    baseline_steps = expected_steps if arm_a_steps is None else arm_a_steps
    return {
        "n_train": int(n_train),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "effective_batch_size": int(batch_size * gradient_accumulation_steps),
        "micro_batches_per_epoch": int(micro_batches_per_epoch),
        "optimizer_steps_per_epoch": int(steps_per_epoch),
        "expected_optimizer_steps": int(expected_steps),
        "expected_trainer_update_steps": int(expected_steps),
        "arm_a_baseline_steps": int(baseline_steps),
        "step_delta_vs_arm_a": int(expected_steps - baseline_steps),
        "step_ratio_vs_arm_a": round(float(expected_steps / max(baseline_steps, 1)), 4),
        "equal_epoch_budget_disclosure": (
            "Equal-epoch schedule fixes epochs=3; synthetic-augmented arms C (+345 rows) and D (+645 rows) "
            "therefore execute more expected Trainer update steps per epoch than real-only arms A and B."
        ),
        "step_semantics_disclosure": (
            "expected_optimizer_steps (expected_trainer_update_steps) is the expected Transformers Trainer "
            "update-step budget under single-GPU drop_last=False with no early stopping. actual_global_step "
            "records Trainer.state.global_step. Verified against transformers==4.57.1 src/transformers/trainer.py "
            "(L2747-2754): state.global_step increments unconditionally on every gradient-accumulation update "
            "step even when FP16 GradScaler skips a weight update on overflow (accelerator.optimizer_step_was_skipped=True); "
            "successful optimizer weight updates are NOT separately instrumented unless explicitly recorded."
        ),
    }


def _build_t4_benchmark_workload_fixture(
    val_df: pd.DataFrame,
    sd_val_df: pd.DataFrame,
    text_col: str,
) -> tuple[list[str], dict[str, Any]]:
    """Build a 7,114-row dev-only length-matched inference fixture with ZERO held-out/OOD exposure."""
    fold_val_texts = [t for t in val_df[text_col].fillna("").astype(str).tolist() if t]
    sd_val_texts = [t for t in sd_val_df[text_col].fillna("").astype(str).tolist() if t]
    pool = sd_val_texts + fold_val_texts
    if not pool:
        raise ValueError("Dev validation pool for T4 benchmark workload fixture is empty")

    # 1,698 single dev texts mirror smishdetect_test (~80-84 chars mean)
    short_texts = [pool[i % len(pool)] for i in range(1698)]
    # 5,416 deterministically paired dev texts mirror banglaphish_ood (~165 chars / ~26 words mean)
    n_pool = len(pool)
    long_texts = [f"{pool[(i * 7) % n_pool]} {pool[(i * 13 + 3) % n_pool]}" for i in range(5416)]
    fixture_texts = short_texts + long_texts

    char_lens = np.array([len(t) for t in fixture_texts], dtype=np.float64)
    word_lens = np.array([len(t.split()) for t in fixture_texts], dtype=np.float64)
    meta = {
        "n_rows": len(fixture_texts),
        "n_short_rows": len(short_texts),
        "n_long_concat_rows": len(long_texts),
        "heldout_or_ood_rows_used": 0,
        "source_splits": ["smishdetect_validation", "fold_cv_val"],
        "char_len_mean": round(float(char_lens.mean()), 2),
        "char_len_p50": float(np.median(char_lens)),
        "char_len_p95": float(np.percentile(char_lens, 95)),
        "word_len_mean": round(float(word_lens.mean()), 2),
        "word_len_p50": float(np.median(word_lens)),
        "word_len_p95": float(np.percentile(word_lens, 95)),
        "limitation_note": (
            "Constructed strictly from dev splits (0 held-out/OOD rows). Synthetic two-message concatenation "
            "matches target char/word length distribution (~165 chars) but cannot replicate true OOD subword "
            "vocabulary fragmentation; runtime extrapolation must include a 20-30% conservative safety margin."
        ),
    }
    return fixture_texts, meta


def hf_train_and_predict_fold(
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
    best_model_export_dir: Path | None = None,
    staging_unit_dir: Path | None = None,
    events_dir: Path | None = None,
    pointers_dir: Path | None = None,
    uid: str | None = None,
    attempt: int | None = None,
    config_hash: str | None = None,
    manifest_sha256: str | None = None,
    lease_check_fn: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Train one HuggingFace sequence classifier fold on local SSD with epoch checkpoints and Drive staging."""
    if (
        cfg.run_mode == "t4_benchmark"
        and cfg.benchmark_workload_fixture_eval
        and not SYNTHETIC_WORKLOAD_FIXTURE_ENABLED
    ):
        raise RuntimeError(
            "Fail closed before model initialization: synthetic workload fixture execution is blocked "
            "under the permanent synthetic-generation stop (SYNTHETIC_WORKLOAD_FIXTURE_ENABLED=False). "
            "Obtain owner approval for an unmodified development-data-only benchmarking alternative."
        )
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    t_init_start = time.perf_counter()
    text_col = TEXT_VIEW_COLUMNS[text_view]
    if not cfg.run_heldout_eval:
        forbidden = sorted({"smishdetect_test", "banglaphish_ood"} & set(heldout_sets))
        if forbidden:
            raise RuntimeError(
                f"Fail closed: held-out/OOD dataset(s) {forbidden} must not be passed when run_heldout_eval=False."
            )
    seed_everything(seed)
    pinned_rev = (cfg.tokenizer_revisions or {}).get(model_name)
    tokenizer = (
        AutoTokenizer.from_pretrained(model_name, revision=pinned_rev)
        if pinned_rev
        else AutoTokenizer.from_pretrained(model_name)
    )
    if pinned_rev:
        tokenizer_revision: str | None = resolve_and_verify_tokenizer_revision(
            tokenizer,
            model_name=model_name,
            expected_revision=str(pinned_rev),
            allow_hf_cache_lookup=True,
        )
    else:
        raw_tok_rev = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash") or getattr(tokenizer, "_commit_hash", None)
        if not raw_tok_rev:
            init_kw = getattr(tokenizer, "init_kwargs", None)
            if isinstance(init_kw, dict):
                for _key in (
                    "vocab_file",
                    "tokenizer_file",
                    "merges_file",
                    "sp_model_file",
                    "sentencepiece_model_file",
                    "tokenizer_config_file",
                ):
                    _val = init_kw.get(_key)
                    if isinstance(_val, (str, Path)):
                        _sha = extract_commit_hash_from_path(_val)
                        if _sha:
                            raw_tok_rev = _sha
                            break
        tokenizer_revision = str(raw_tok_rev) if raw_tok_rev else None

    tok_seconds = 0.0

    def _make_ds(texts: list[str], labels: list[int] | None = None) -> Dataset:
        nonlocal tok_seconds
        t_tok0 = time.perf_counter()
        payload: dict[str, Any] = {"text": texts}
        if labels is not None:
            payload["label"] = labels
        ds = Dataset.from_dict(payload)
        mapped = ds.map(
            lambda b: tokenizer(
                prepare_tokenizer_texts(b["text"], model_name), truncation=True, max_length=cfg.max_length
            ),
            batched=True,
            remove_columns=["text"],
        )
        tok_seconds += time.perf_counter() - t_tok0
        return mapped

    ds_tr = _make_ds(train_df[text_col].fillna("").astype(str).tolist(), train_df["label_idx"].astype(int).tolist())
    ds_val = _make_ds(val_df[text_col].fillna("").astype(str).tolist(), val_df["label_idx"].astype(int).tolist())

    model: Any = None
    trainer: Any = None
    try:
        model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=len(CANONICAL_LABELS), revision=pinned_rev
            )
            if pinned_rev
            else AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(CANONICAL_LABELS))
        )
        resolved_model_rev = getattr(model.config, "_commit_hash", None)
        if pinned_rev:
            if not resolved_model_rev:
                raise RuntimeError(
                    f"Fail closed: loaded model weights for {model_name!r} did not expose model.config._commit_hash "
                    f"(expected pinned revision {pinned_rev!r})."
                )
            if str(resolved_model_rev) != str(pinned_rev):
                raise RuntimeError(
                    f"Fail closed: loaded model revision {resolved_model_rev!r} != pinned revision {pinned_rev!r} for {model_name!r}"
                )
        model_revision = str(resolved_model_rev) if resolved_model_rev else None
        use_fp16, use_bf16 = _resolve_precision(cfg.fp16)
        init_seconds = round(time.perf_counter() - t_init_start - tok_seconds, 4)

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            per_device_eval_batch_size=cfg.eval_batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            fp16=use_fp16,
            bf16=use_bf16,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_strategy="epoch",
            report_to="none",
            seed=seed,
            data_seed=seed,
        )

        drive_sync_seconds = 0.0
        local_ckpt_seconds = 0.0
        eval_seconds = 0.0
        prior_train_seconds = 0.0
        prior_pure_opt_seconds = 0.0
        prior_eval_seconds = 0.0
        prior_local_ckpt_seconds = 0.0
        prior_drive_sync_seconds = 0.0
        resumed_from_step = 0
        t_train_start: float | None = None
        checkpoint_prune_errors: list[dict[str, Any]] = []
        cb_logger = logging.getLogger("bangla_smish.post_stage_b_runner")

        def _record_prune(prune_out: dict[str, Any], *, phase: str) -> None:
            errs = prune_out.get("prune_errors") or []
            if errs:
                cb_logger.warning("Checkpoint pruning errors for %s (%s): %s", uid or checkpoint_dir.parent.name, phase, errs)
                for e in errs:
                    checkpoint_prune_errors.append({"phase": phase, **e})

        active_attempt_num = int(attempt or 1)

        class Stage4CheckpointCallback(TrainerCallback):
            def on_evaluate(self, args, state, control, **kwargs):
                nonlocal eval_seconds
                # Track epoch-end evaluation timing cleanly across HF versions
                if state.log_history and "eval_runtime" in state.log_history[-1]:
                    eval_seconds += float(state.log_history[-1]["eval_runtime"])

            def on_save(self, args, state, control, **kwargs):
                nonlocal local_ckpt_seconds, drive_sync_seconds
                if not args.should_save:
                    return
                if lease_check_fn is not None:
                    lease_check_fn()
                t_loc = time.perf_counter()
                path = checkpoint_dir / f"checkpoint-{state.global_step}"
                elapsed_attempt = max(time.perf_counter() - t_train_start, 0.0) if t_train_start is not None else 0.0
                attempt_pure_opt_so_far = max(
                    elapsed_attempt - eval_seconds - local_ckpt_seconds - drive_sync_seconds, 0.0
                )
                timing_payload = {
                    "schema_version": "stage4_timing_v1",
                    "global_step": int(state.global_step),
                    "resumed_from_step": int(resumed_from_step),
                    "attempt_train_seconds_at_save": round(elapsed_attempt, 4),
                    "cumulative_train_seconds": round(prior_train_seconds + elapsed_attempt, 4),
                    "cumulative_pure_opt_seconds": round(prior_pure_opt_seconds + attempt_pure_opt_so_far, 4),
                    "cumulative_eval_seconds": round(prior_eval_seconds + eval_seconds, 4),
                    "cumulative_local_ckpt_seconds": round(prior_local_ckpt_seconds + local_ckpt_seconds, 4),
                    "cumulative_drive_sync_seconds": round(prior_drive_sync_seconds + drive_sync_seconds, 4),
                }
                atomic_write_json(path / STAGE4_TIMING_METADATA, timing_payload)
                atomic_write_json(path / _CHECKPOINT_COMMIT, _checkpoint_file_hashes(path, require_scaler=use_fp16))
                local_ckpt_seconds += time.perf_counter() - t_loc
                if staging_unit_dir is not None and events_dir is not None and pointers_dir is not None:
                    t_drv = time.perf_counter()
                    # Ensure any earlier local best_model_checkpoint referenced by trainer_state.json is also
                    # durably staged on Drive so cross-VM restore can always recover both latest and best.
                    best_ref = getattr(state, "best_model_checkpoint", None)
                    if best_ref:
                        best_local = Path(best_ref).resolve()
                        if best_local != path.resolve() and best_local.is_dir() and _valid_checkpoint(best_local, require_scaler=use_fp16):
                            pre_scan = scan_unit_staging(
                                staging_unit_dir=staging_unit_dir,
                                events_dir=events_dir,
                                pointers_dir=pointers_dir,
                                uid=uid or checkpoint_dir.parent.name,
                                expected_config_hash=config_hash or cfg.config_hash(),
                                expected_manifest_sha256=manifest_sha256 or "",
                                checkpoint_retention_limit=cfg.checkpoint_retention_limit,
                                active_attempt=active_attempt_num,
                                trust_commit_events=False,
                            )
                            best_step = int(
                                json.loads((best_local / "trainer_state.json").read_text(encoding="utf-8"))["global_step"]
                            )
                            staged_valid_steps = {
                                int(v["global_step"])
                                for v in pre_scan.get("entries", [])
                                if v.get("state") == CheckpointState.VALID_RECOVERABLE
                            }
                            if best_step not in staged_valid_steps:
                                sync_local_checkpoint_to_staging(
                                    local_ckpt_dir=best_local,
                                    staging_unit_dir=staging_unit_dir,
                                    events_dir=events_dir,
                                    pointers_dir=pointers_dir,
                                    uid=uid or checkpoint_dir.parent.name,
                                    attempt=active_attempt_num,
                                    config_hash=config_hash or cfg.config_hash(),
                                    manifest_sha256=manifest_sha256 or "",
                                    require_scaler=use_fp16,
                                )
                    sync_local_checkpoint_to_staging(
                        local_ckpt_dir=path,
                        staging_unit_dir=staging_unit_dir,
                        events_dir=events_dir,
                        pointers_dir=pointers_dir,
                        uid=uid or checkpoint_dir.parent.name,
                        attempt=active_attempt_num,
                        config_hash=config_hash or cfg.config_hash(),
                        manifest_sha256=manifest_sha256 or "",
                        require_scaler=use_fp16,
                    )
                    scan_ep = scan_unit_staging(
                        staging_unit_dir=staging_unit_dir,
                        events_dir=events_dir,
                        pointers_dir=pointers_dir,
                        uid=uid or checkpoint_dir.parent.name,
                        expected_config_hash=config_hash or cfg.config_hash(),
                        expected_manifest_sha256=manifest_sha256 or "",
                        checkpoint_retention_limit=cfg.checkpoint_retention_limit,
                        active_attempt=active_attempt_num,
                        trust_commit_events=False,
                    )
                    _record_prune(prune_superseded_staging(scan_report=scan_ep, dry_run=False), phase="on_save")
                    drive_sync_seconds += time.perf_counter() - t_drv

        trainer_cls = _weighted_trainer_class(Trainer, class_weights) if class_weights is not None else Trainer
        trainer = trainer_cls(
            model=model,
            args=args,
            train_dataset=ds_tr,
            eval_dataset=ds_val,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            processing_class=tokenizer,
            callbacks=[Stage4CheckpointCallback()],
        )

        latest_ckpt: str | None = None
        if resume_from_checkpoint:
            if staging_unit_dir is not None and events_dir is not None and pointers_dir is not None:
                scan = scan_unit_staging(
                    staging_unit_dir=staging_unit_dir,
                    events_dir=events_dir,
                    pointers_dir=pointers_dir,
                    uid=uid or checkpoint_dir.parent.name,
                    expected_config_hash=config_hash or cfg.config_hash(),
                    expected_manifest_sha256=manifest_sha256 or "",
                    checkpoint_retention_limit=cfg.checkpoint_retention_limit,
                    active_attempt=active_attempt_num,
                    trust_commit_events=False,
                )
                recovered = recover_latest_valid_checkpoint(scan_report=scan, local_restore_root=checkpoint_dir)
                if recovered["recovered"]:
                    latest_ckpt = str(recovered["restored_local_ckpt"])
                    _record_prune(prune_superseded_staging(scan_report=scan, dry_run=False), phase="post_restore")
            else:
                latest_ckpt = _find_latest_trainer_checkpoint(checkpoint_dir, require_scaler=use_fp16)

        if latest_ckpt is not None:
            latest_ckpt_path = Path(latest_ckpt)
            tstate_path = latest_ckpt_path / "trainer_state.json"
            if tstate_path.is_file():
                try:
                    resumed_from_step = int(json.loads(tstate_path.read_text(encoding="utf-8")).get("global_step") or 0)
                except (OSError, ValueError, TypeError):
                    resumed_from_step = 0
            timing_path = latest_ckpt_path / STAGE4_TIMING_METADATA
            if timing_path.is_file():
                try:
                    tmeta = json.loads(timing_path.read_text(encoding="utf-8"))
                    prior_train_seconds = float(tmeta.get("cumulative_train_seconds") or 0.0)
                    prior_pure_opt_seconds = float(tmeta.get("cumulative_pure_opt_seconds") or 0.0)
                    prior_eval_seconds = float(tmeta.get("cumulative_eval_seconds") or 0.0)
                    prior_local_ckpt_seconds = float(tmeta.get("cumulative_local_ckpt_seconds") or 0.0)
                    prior_drive_sync_seconds = float(tmeta.get("cumulative_drive_sync_seconds") or 0.0)
                except (OSError, ValueError, TypeError):
                    pass

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_train_start = time.perf_counter()
        trainer.train(resume_from_checkpoint=latest_ckpt)
        attempt_train_seconds = round(time.perf_counter() - t_train_start, 4)

        t_inf0 = time.perf_counter()

        def _predict_probs(texts: list[str]) -> np.ndarray:
            ds_eval = _make_ds(texts)
            logits = trainer.predict(ds_eval).predictions
            return torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()

        predictions: dict[str, np.ndarray] = {
            "val_probs": _predict_probs(val_df[text_col].fillna("").astype(str).tolist()),
        }

        if cfg.run_adversarial_eval:
            adv_texts = val_df[_adversarial_column(val_df, text_view)].fillna("").astype(str).tolist()
            if cfg.smoke_max_eval_samples is not None:
                adv_texts = adv_texts[: cfg.smoke_max_eval_samples]
            predictions["val_adv_probs"] = _predict_probs(adv_texts)

        fixture_meta: dict[str, Any] | None = None
        if cfg.run_heldout_eval:
            for eval_key in ("smishdetect_validation", "smishdetect_test", "banglaphish_ood"):
                edf = heldout_sets[eval_key]
                if cfg.smoke_max_eval_samples is not None:
                    edf = edf.iloc[: cfg.smoke_max_eval_samples]
                predictions[f"{eval_key}_probs"] = _predict_probs(edf[text_col].fillna("").astype(str).tolist())
        elif cfg.run_mode == "t4_benchmark" and cfg.benchmark_workload_fixture_eval:
            if not SYNTHETIC_WORKLOAD_FIXTURE_ENABLED:
                raise RuntimeError(
                    "Fail closed: synthetic workload fixture execution is blocked under the permanent "
                    "synthetic-generation stop (SYNTHETIC_WORKLOAD_FIXTURE_ENABLED=False). "
                    "Obtain owner approval for an unmodified development-data-only benchmarking alternative."
                )
            sd_val_texts = heldout_sets["smishdetect_validation"][text_col].fillna("").astype(str).tolist()
            predictions["smishdetect_validation_probs"] = _predict_probs(sd_val_texts)
            fixture_texts, fixture_meta = _build_t4_benchmark_workload_fixture(
                val_df, heldout_sets["smishdetect_validation"], text_col
            )
            _ = _predict_probs(fixture_texts)

        if cfg.run_counterfactual_eval and "counterfactual_smish_probe" in heldout_sets and len(heldout_sets["counterfactual_smish_probe"]):
            cf_df = heldout_sets["counterfactual_smish_probe"]
            if cfg.smoke_max_eval_samples is not None:
                cf_df = cf_df.iloc[: cfg.smoke_max_eval_samples]
            predictions["cf_orig_probs"] = _predict_probs(cf_df[f"orig_{text_col}"].fillna("").astype(str).tolist())
            predictions["cf_contact_probs"] = _predict_probs(
                cf_df[f"contact_swap_{text_col}"].fillna("").astype(str).tolist()
            )
            predictions["cf_brand_probs"] = _predict_probs(
                cf_df[f"contact_brand_swap_{text_col}"].fillna("").astype(str).tolist()
            )
        inference_seconds = round(time.perf_counter() - t_inf0, 4)

        if best_model_export_dir is not None:
            best_model_export_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(best_model_export_dir))
            tokenizer.save_pretrained(str(best_model_export_dir))

        raw_global_step = getattr(trainer.state, "global_step", None)
        actual_steps = int(raw_global_step) if raw_global_step is not None else None
        steps_executed_this_attempt = (
            max(actual_steps - resumed_from_step, 0) if actual_steps is not None else None
        )
        attempt_pure_opt_sec = max(
            attempt_train_seconds - eval_seconds - local_ckpt_seconds - drive_sync_seconds, 0.0
        )
        cumulative_train_seconds = round(prior_train_seconds + attempt_train_seconds, 4)
        cumulative_pure_opt_sec = prior_pure_opt_seconds + attempt_pure_opt_sec
        # Attempt-local ms/step uses ONLY steps executed in this attempt (prevents resume undercounting).
        attempt_optimizer_step_ms = (
            round(1000.0 * attempt_pure_opt_sec / steps_executed_this_attempt, 3)
            if steps_executed_this_attempt is not None and steps_executed_this_attempt > 0
            else None
        )
        # Cumulative ms/step divides cumulative pure-optimization seconds across resumed attempts by total global_step
        # (falls back to attempt_optimizer_step_ms if a legacy checkpoint lacked stage4_timing.json).
        if resumed_from_step > 0 and prior_train_seconds <= 0.0:
            optimizer_step_ms = attempt_optimizer_step_ms
        elif actual_steps is not None and actual_steps > 0:
            optimizer_step_ms = round(1000.0 * cumulative_pure_opt_sec / actual_steps, 3)
        else:
            optimizer_step_ms = None
        return {
            "train_seconds": cumulative_train_seconds,
            "attempt_train_seconds": attempt_train_seconds,
            "prior_resumed_train_seconds": round(prior_train_seconds, 4),
            "cumulative_train_seconds": cumulative_train_seconds,
            "resumed_from_step": int(resumed_from_step),
            "steps_executed_this_attempt": steps_executed_this_attempt,
            "init_seconds": init_seconds,
            "tokenization_seconds": round(tok_seconds, 4),
            "eval_seconds": round(prior_eval_seconds + eval_seconds, 4),
            "attempt_eval_seconds": round(eval_seconds, 4),
            "inference_seconds": inference_seconds,
            "local_checkpoint_seconds": round(prior_local_ckpt_seconds + local_ckpt_seconds, 4),
            "drive_sync_seconds": round(prior_drive_sync_seconds + drive_sync_seconds, 4),
            "attempt_optimizer_step_ms": attempt_optimizer_step_ms,
            "optimizer_step_ms": optimizer_step_ms,
            "actual_global_step": actual_steps,
            "successful_optimizer_weight_updates": None,
            "resumed_from_checkpoint": latest_ckpt,
            "recovery_granularity": RECOVERY_GRANULARITY,
            "best_trainer_checkpoint": trainer.state.best_model_checkpoint,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "fp16": use_fp16,
            "bf16": use_bf16,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
            "benchmark_fixture_metadata": fixture_meta if cfg.run_mode == "t4_benchmark" else None,
            "checkpoint_prune_errors": checkpoint_prune_errors,
            "predictions": predictions,
        }
    finally:
        trainer = None
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _compute_unit_metrics(
    *,
    val_df: pd.DataFrame,
    heldout_sets: dict[str, pd.DataFrame],
    predictions: dict[str, np.ndarray],
    cfg: PostStageBConfig,
) -> dict[str, Any]:
    n_boot = 100 if cfg.run_mode == "gpu_smoke_test" else 500
    boot_seed = int(cfg.seed)
    val_probs = predictions["val_probs"]
    val_overall = _evaluate_frame(val_df, val_probs, n_boot=n_boot, seed=boot_seed)
    threshold = val_overall["smishing_low_fpr"]["threshold"]

    metrics: dict[str, Any] = {
        "val_overall": val_overall,
        "val_by_language": _evaluate_by_language(
            val_df, val_probs, n_boot=min(n_boot, 250), operating_threshold=threshold, seed=boot_seed
        ),
    }

    if "val_adv_probs" in predictions:
        n_adv = len(predictions["val_adv_probs"])
        metrics["val_adv_zwj_homoglyph"] = _evaluate_frame(
            val_df.iloc[:n_adv], predictions["val_adv_probs"], n_boot=n_boot, operating_threshold=threshold, seed=boot_seed
        )

    for eval_key in ("smishdetect_validation", "smishdetect_test", "banglaphish_ood"):
        pk = f"{eval_key}_probs"
        if pk in predictions:
            edf = heldout_sets[eval_key].iloc[: len(predictions[pk])]
            ep = predictions[pk]
            metrics[eval_key] = _evaluate_frame(edf, ep, n_boot=n_boot, operating_threshold=threshold, seed=boot_seed)
            if eval_key in ("smishdetect_validation", "smishdetect_test"):
                metrics[f"{eval_key}_by_language"] = _evaluate_by_language(
                    edf, ep, n_boot=min(n_boot, 250), operating_threshold=threshold, seed=boot_seed
                )
            if eval_key == "banglaphish_ood" and "source_type" in edf.columns:
                sms_mask = (edf["source_type"] == "SMS").to_numpy()
                other_mask = ~sms_mask
                if int(sms_mask.sum()) >= 5:
                    metrics["banglaphish_ood_sms"] = _evaluate_frame(
                        edf.loc[sms_mask], ep[sms_mask], n_boot=n_boot, operating_threshold=threshold, seed=boot_seed
                    )
                if int(other_mask.sum()) >= 5:
                    metrics["banglaphish_ood_other"] = _evaluate_frame(
                        edf.loc[other_mask], ep[other_mask], n_boot=n_boot, operating_threshold=threshold, seed=boot_seed
                    )

    if "cf_orig_probs" in predictions:
        smish_idx = CANONICAL_LABELS.index("Smishing")
        p_orig = predictions["cf_orig_probs"]
        p_c = predictions["cf_contact_probs"]
        p_cb = predictions["cf_brand_probs"]
        orig_pct = float((p_orig.argmax(1) == smish_idx).mean() * 100.0)
        c_pct = float((p_c.argmax(1) == smish_idx).mean() * 100.0)
        cb_pct = float((p_cb.argmax(1) == smish_idx).mean() * 100.0)
        metrics["counterfactual_probe"] = {
            "n_probed": int(len(p_orig)),
            "smishing_pred_pct_before": round(orig_pct, 2),
            "contact_swap_pct_after": round(c_pct, 2),
            "contact_swap_pp_delta": round(c_pct - orig_pct, 2),
            "contact_brand_swap_pct_after": round(cb_pct, 2),
            "contact_brand_swap_pp_delta": round(cb_pct - orig_pct, 2),
        }
    return metrics


def _should_keep_weights(keep_mode: str, fold: int) -> bool:
    if keep_mode == "all":
        return True
    if keep_mode == "best_fold0":
        return fold == 0
    return False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expected_units(cfg: PostStageBConfig) -> dict[str, tuple[str, str, str, int]]:
    units_iter = cfg.unit_schedule if cfg.unit_schedule is not None else product(
        cfg.experiments, cfg.models, cfg.text_views, cfg.folds
    )
    return {unit_id(*unit): unit for unit in units_iter}


def _run_identity(
    bundle_dir: Path,
    cfg: PostStageBConfig,
    runner_fn: Callable[..., Any],
    *,
    env_fp: dict[str, Any] | None = None,
    hw_fp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_dir = Path(__file__).parent
    runner_name = getattr(runner_fn, "__qualname__", type(runner_fn).__qualname__)
    kit_module_names = sorted(Path(rel).name for rel in STAGE4_KIT_REQUIRED_SOURCE_FILES)
    return {
        "schema_version": 5,
        "run_name": cfg.run_name,
        "run_mode": cfg.run_mode,
        "config_hash": cfg.config_hash(),
        "manifest_sha256": sha256_file(bundle_dir / "dataset_manifest.json"),
        "pinned_model_revisions": dict(sorted((cfg.tokenizer_revisions or {}).items())),
        "runner": f"{runner_fn.__module__}.{runner_name}",
        "environment_fingerprint": env_fp if env_fp is not None else collect_stable_environment_fingerprint(),
        "hardware_fingerprint": hw_fp if hw_fp is not None else collect_hardware_fingerprint(),
        "source_sha256": {
            name: sha256_file(source_dir / name)
            for name in kit_module_names
        },
    }


_OFFLINE_AGGREGATION_STRICT_IDENTITY_KEYS: tuple[str, ...] = (
    "schema_version",
    "run_name",
    "run_mode",
    "config_hash",
    "manifest_sha256",
    "pinned_model_revisions",
    "runner",
    "source_sha256",
)


def _read_state(
    path: Path,
    identity: dict[str, Any],
    *,
    expect_state: bool,
    allow_offline_hardware_diff: bool = False,
) -> dict[str, Any]:
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            backup_path = path.with_name("run_state.backup.json")
            backup_hint = (
                f" A backup file exists at {backup_path}; inspect and recover explicitly before resuming."
                if backup_path.is_file()
                else ""
            )
            raise ValueError(f"Corrupt JSON in {path}: {exc}.{backup_hint}") from exc
        recorded_id = state.get("identity")
        if allow_offline_hardware_diff:
            if not isinstance(recorded_id, dict) or any(
                recorded_id.get(k) != identity.get(k) for k in _OFFLINE_AGGREGATION_STRICT_IDENTITY_KEYS
            ):
                raise ValueError(
                    "Existing run has incompatible configuration, manifest, pinned revisions, or source code; "
                    "cannot aggregate offline"
                )
        elif recorded_id != identity:
            raise ValueError(
                "Existing run has incompatible configuration, manifest, or code (or runtime/hardware environment); "
                "use a fresh artifact directory"
            )
        return state
    if expect_state:
        raise FileNotFoundError(path)
    orphan_state_tmps: list[Path] = []
    for entry in path.parent.iterdir():
        if entry.name == "leases":
            continue
        if entry.is_file() and _RUN_STATE_ORPHAN_TMP_RE.fullmatch(entry.name):
            orphan_state_tmps.append(entry)
            continue
        raise ValueError("Artifact directory contains an unidentified run; use a fresh artifact directory")
    for orphan in orphan_state_tmps:
        try:
            orphan.unlink(missing_ok=True)
        except OSError:
            pass
    return {"identity": identity, "attempts": {}, "committed_units": {}, "created_at_utc": _utc_now()}


def _load_configured_eval_sets(bundle_dir: Path, cfg: PostStageBConfig) -> dict[str, pd.DataFrame]:
    """Load only the evaluation tables authorized by cfg (never reading held-out/OOD parquets when run_heldout_eval=False)."""
    if cfg.run_heldout_eval:
        return load_heldout_eval_sets(bundle_dir)
    eval_sets: dict[str, pd.DataFrame] = {}
    if cfg.run_mode == "t4_benchmark" and cfg.benchmark_workload_fixture_eval:
        eval_sets["smishdetect_validation"] = pd.read_parquet(bundle_dir / "eval_smishdetect_validation.parquet")
    if cfg.run_counterfactual_eval:
        eval_sets["counterfactual_smish_probe"] = pd.read_parquet(bundle_dir / "counterfactual_smish_probe.parquet")
    return eval_sets


_TOKENIZER_GATE_SCHEMA_VERSION = "stage4_tokenizer_collision_gate_v1"
_TOKENIZER_GATE_EXPORT_NAME = "tokenizer_collision_gate.json"


def _load_tokenizer_gate_payload(
    report_or_path: dict[str, Any] | str | Path,
    *,
    source_label: str,
) -> dict[str, Any]:
    if isinstance(report_or_path, dict):
        return report_or_path
    if isinstance(report_or_path, (str, Path)):
        path = Path(report_or_path)
        if not path.is_file():
            raise RuntimeError(f"Fail closed: missing tokenizer collision gate report at {path} ({source_label})")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Fail closed: unreadable or invalid JSON in {source_label} ({path}): {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Fail closed: tokenizer collision gate report ({source_label}) must be a JSON object")
        return payload
    raise ValueError(
        f"Fail closed: unsupported tokenizer_gate_report type ({source_label}): {type(report_or_path).__name__}"
    )


def _validate_tokenizer_gate_report(
    report: dict[str, Any],
    cfg: PostStageBConfig,
    expected_config_hash: str,
    *,
    expected_manifest_sha256: str,
    expected_env_packages: dict[str, Any] | None = None,
    source_label: str,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ValueError(f"Fail closed: tokenizer collision gate report ({source_label}) must be a dict")
    if report.get("schema_version") != _TOKENIZER_GATE_SCHEMA_VERSION:
        raise ValueError(
            f"Fail closed: incompatible tokenizer gate schema_version in {source_label}: "
            f"{report.get('schema_version')!r} != {_TOKENIZER_GATE_SCHEMA_VERSION!r}"
        )
    if report.get("run_name") != cfg.run_name:
        raise ValueError(
            f"Fail closed: tokenizer gate run_name mismatch in {source_label}: "
            f"{report.get('run_name')!r} != {cfg.run_name!r}"
        )
    if report.get("config_hash") != expected_config_hash:
        raise ValueError(
            f"Fail closed: tokenizer gate config_hash mismatch in {source_label}: "
            f"{report.get('config_hash')!r} != {expected_config_hash!r}"
        )
    if report.get("bundle_manifest_sha256") != expected_manifest_sha256:
        raise ValueError(
            f"Fail closed: tokenizer gate bundle_manifest_sha256 mismatch in {source_label}: "
            f"{report.get('bundle_manifest_sha256')!r} != {expected_manifest_sha256!r}"
        )
    gate_pkgs = report.get("tokenizer_package_versions")
    required_gate_pkg_keys = {dist for dist, _ in TOKENIZER_GATE_PACKAGE_DISTS}
    if not isinstance(gate_pkgs, dict) or set(gate_pkgs.keys()) != required_gate_pkg_keys:
        raise ValueError(
            f"Fail closed: tokenizer gate tokenizer_package_versions missing or malformed in {source_label}: "
            f"{gate_pkgs!r}"
        )
    if expected_env_packages is not None:
        mismatched_pkgs = {
            dist: {"gate": gate_pkgs.get(dist), "live": expected_env_packages.get(dist)}
            for dist in sorted(required_gate_pkg_keys)
            if gate_pkgs.get(dist) != expected_env_packages.get(dist)
        }
        if mismatched_pkgs:
            raise ValueError(
                f"Fail closed: tokenizer gate package versions do not match live runtime in {source_label}: "
                f"{mismatched_pkgs!r}"
            )
    if not isinstance(cfg.tokenizer_revisions, dict):
        raise ValueError(
            f"Fail closed: tokenizer gate report ({source_label}) requires pinned cfg.tokenizer_revisions"
        )
    expected_revisions = {m: str(cfg.tokenizer_revisions[m]) for m in cfg.models}
    if report.get("tokenizer_revisions") != expected_revisions:
        raise ValueError(f"Fail closed: tokenizer gate tokenizer_revisions mismatch in {source_label}")
    expected_views = [v for v in cfg.text_views if v in ALLOWED_COLLISION_GATE_VIEWS]
    if not expected_views:
        raise ValueError(
            f"Fail closed: config has no allowed views in {ALLOWED_COLLISION_GATE_VIEWS} for tokenizer gate ({source_label})"
        )
    if report.get("allowed_views") != expected_views:
        raise ValueError(
            f"Fail closed: tokenizer gate allowed_views mismatch in {source_label}: "
            f"{report.get('allowed_views')!r} != {expected_views!r}"
        )
    expected_norm = BANGLABERT_NORMALIZER_COMMIT if BANGLABERT_MODEL in set(cfg.models) else None
    if report.get("banglabert_normalizer_commit") != expected_norm:
        raise ValueError(
            f"Fail closed: tokenizer gate banglabert_normalizer_commit mismatch in {source_label}: "
            f"{report.get('banglabert_normalizer_commit')!r} != {expected_norm!r}"
        )
    expected_pair_ids = [p[0] for p in STAGE3_32_CANDIDATE_PAIRS]
    if report.get("n_pairs") != len(STAGE3_32_CANDIDATE_PAIRS) or report.get("pair_ids") != expected_pair_ids:
        raise ValueError(f"Fail closed: tokenizer gate 32-pair inventory mismatch in {source_label}")
    if report.get("passed") is not True or report.get("collision_count") != 0 or report.get("collisions") != []:
        raise RuntimeError(
            f"Fail closed: tokenizer collision gate did not pass cleanly ({source_label}): "
            f"passed={report.get('passed')!r}, collision_count={report.get('collision_count')!r}"
        )
    expected_total = len(cfg.models) * len(expected_views) * len(STAGE3_32_CANDIDATE_PAIRS)
    results = report.get("results")
    if report.get("total_checks") != expected_total or not isinstance(results, list) or len(results) != expected_total:
        raise ValueError(
            f"Fail closed: tokenizer gate check count mismatch in {source_label}: "
            f"total_checks={report.get('total_checks')!r}, "
            f"len(results)={len(results) if isinstance(results, list) else None!r}, expected={expected_total}"
        )
    expected_keys = {
        (pid, rid_a, rid_b, model_name, expected_revisions[model_name], view)
        for model_name in cfg.models
        for view in expected_views
        for pid, rid_a, rid_b in STAGE3_32_CANDIDATE_PAIRS
    }
    seen_keys: set[tuple[str, str, str, str, str, str]] = set()
    for entry in results:
        if not isinstance(entry, dict) or entry.get("input_ids_equal") is not False:
            raise RuntimeError(
                f"Fail closed: tokenizer gate result entry failed or malformed in {source_label}: {entry!r}"
            )
        seen_keys.add(
            (
                str(entry.get("pair_id", "")),
                str(entry.get("row_id_a", "")),
                str(entry.get("row_id_b", "")),
                str(entry.get("model_name", "")),
                str(entry.get("tokenizer_revision", "")),
                str(entry.get("view", "")),
            )
        )
    if seen_keys != expected_keys:
        raise ValueError(
            f"Fail closed: tokenizer gate results do not match exact expected 32-pair matrix ({source_label})"
        )
    return report


def _resolve_and_validate_tokenizer_gate(
    *,
    cfg: PostStageBConfig,
    expected_config_hash: str,
    expected_manifest_sha256: str,
    expected_env_packages: dict[str, Any] | None,
    tokenizer_gate_report: dict[str, Any] | str | Path | None,
    persisted_export_path: Path,
    state_exists: bool,
) -> dict[str, Any] | None:
    gate_required = bool(cfg.tokenizer_revisions) or cfg.confirmatory_mode or cfg.run_mode == "t4_benchmark"
    persisted_valid: dict[str, Any] | None = None
    if state_exists and persisted_export_path.exists():
        persisted_payload = _load_tokenizer_gate_payload(
            persisted_export_path, source_label="persisted exports/tokenizer_collision_gate.json"
        )
        persisted_valid = _validate_tokenizer_gate_report(
            persisted_payload,
            cfg,
            expected_config_hash,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_env_packages=expected_env_packages,
            source_label="persisted exports/tokenizer_collision_gate.json",
        )

    supplied_valid: dict[str, Any] | None = None
    if tokenizer_gate_report is not None:
        supplied_payload = _load_tokenizer_gate_payload(
            tokenizer_gate_report, source_label="supplied tokenizer_gate_report"
        )
        supplied_valid = _validate_tokenizer_gate_report(
            supplied_payload,
            cfg,
            expected_config_hash,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_env_packages=expected_env_packages,
            source_label="supplied tokenizer_gate_report",
        )

    if gate_required and persisted_valid is None and supplied_valid is None:
        raise RuntimeError(
            "Fail closed: pre-training 32-pair allowed-view tokenizer collision gate report is required "
            "for runs with pinned tokenizer_revisions (pass tokenizer_gate_report from Cell 5)."
        )
    return persisted_valid if persisted_valid is not None else supplied_valid


def _split_for_unit(
    bundle_dir: Path, heldout_sets: dict[str, pd.DataFrame], cfg: PostStageBConfig, exp_id: str, fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    train_df, val_df, split_meta = build_experiment_fold_data(
        bundle_dir, exp_id, fold, seed=cfg.seed,
        balance_mode_override=cfg.experiment_b_balance_mode if exp_id == "B" else None,
    )
    unit_sets = dict(heldout_sets)
    if cfg.run_counterfactual_eval:
        unit_sets["counterfactual_smish_probe"] = select_counterfactual_fold(
            heldout_sets["counterfactual_smish_probe"], train_df, val_df, fold
        )
    split_meta["n_train_pre_cap"] = len(train_df)
    split_meta["n_val_pre_cap"] = len(val_df)
    split_meta["train_by_class_pre_cap"] = split_meta["train_by_class"]
    split_meta["val_by_class_pre_cap"] = split_meta["val_by_class"]
    split_meta["n_train_real_pre_cap"] = split_meta["n_train_real"]
    split_meta["n_train_synthetic_pre_cap"] = split_meta["n_train_synthetic"]
    train_df = train_df.iloc[:cfg.smoke_max_train_samples].reset_index(drop=True)
    val_df = val_df.iloc[:cfg.smoke_max_val_samples].reset_index(drop=True)
    for name, frame in (("train", train_df), ("val", val_df)):
        if frame.empty:
            raise ValueError(f"Empty {name} fold {fold}")
        split_meta[f"n_{name}"] = len(frame)
        split_meta[f"n_{name}_actual"] = len(frame)
        split_meta[f"{name}_by_class"] = {
            label: int((frame["label_3class"] == label).sum()) for label in CANONICAL_LABELS
        }
    split_meta["n_train_real"] = int((~train_df["is_synthetic"]).sum())
    split_meta["n_train_synthetic"] = int(train_df["is_synthetic"].sum())
    if split_meta["class_weights"] is not None and len(train_df) != split_meta["n_train_pre_cap"]:
        counts = split_meta["train_by_class"]
        if any(count == 0 for count in counts.values()):
            raise ValueError("Capped Experiment B training requires every class to compute weights")
        split_meta["class_weights"] = [len(train_df) / (len(CANONICAL_LABELS) * counts[label]) for label in CANONICAL_LABELS]
    arm_a_real_actual = min(split_meta["n_train_real_pre_cap"], cfg.smoke_max_train_samples) if cfg.smoke_max_train_samples else split_meta["n_train_real_pre_cap"]
    arm_a_steps = compute_optimizer_step_budget(
        arm_a_real_actual,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    )["expected_optimizer_steps"]
    split_meta["optimizer_step_budget"] = compute_optimizer_step_budget(
        len(train_df),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        arm_a_steps=arm_a_steps,
    )
    return train_df, val_df, unit_sets, split_meta


def _prediction_frames(
    cfg: PostStageBConfig, val_df: pd.DataFrame, eval_sets: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    frames = {"val": val_df}
    cap = cfg.smoke_max_eval_samples
    if cfg.run_adversarial_eval:
        frames["val_adv"] = val_df.iloc[:cap]
    if cfg.run_heldout_eval:
        for name in ("smishdetect_validation", "smishdetect_test", "banglaphish_ood"):
            frames[name] = eval_sets[name].iloc[:cap]
    elif cfg.run_mode == "t4_benchmark" and cfg.benchmark_workload_fixture_eval:
        frames["smishdetect_validation"] = eval_sets["smishdetect_validation"].iloc[:cap]
    if cfg.run_counterfactual_eval and len(eval_sets["counterfactual_smish_probe"]):
        cf = eval_sets["counterfactual_smish_probe"].iloc[:cap]
        frames.update({key: cf for key in ("cf_orig", "cf_contact", "cf_brand")})
    return frames


def _write_prediction_archive(path: Path, predictions: dict[str, np.ndarray], frames: dict[str, pd.DataFrame]) -> None:
    arrays = dict(predictions)
    for prefix, frame in frames.items():
        arrays[f"{prefix}_row_ids"] = frame["row_id"].to_numpy(dtype=str)
        arrays[f"{prefix}_y"] = frame["label_idx"].to_numpy(dtype=int)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}.npz")
    try:
        with temporary.open("wb") as file:
            np.savez_compressed(file, **arrays)
            file.flush()
            _fsync_fd_best_effort(file.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    _fsync_dir_best_effort(path.parent)


def _read_prediction_archive(path: Path, frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    expected_keys = {f"{prefix}_{suffix}" for prefix in frames for suffix in ("row_ids", "y", "probs")}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys or len(archive.files) != len(expected_keys):
            raise ValueError(f"Prediction archive keys differ from the expected splits: {path}")
        arrays = {name: archive[name] for name in expected_keys}
    for prefix, frame in frames.items():
        row_ids = arrays[f"{prefix}_row_ids"]
        labels = arrays[f"{prefix}_y"]
        probabilities = arrays[f"{prefix}_probs"]
        if row_ids.dtype.kind != "U" or not np.array_equal(row_ids, frame["row_id"].to_numpy(dtype=str)):
            raise ValueError(f"Prediction row IDs are out of order or incorrect for {prefix}")
        if len(np.unique(row_ids)) != len(row_ids):
            raise ValueError(f"Duplicate prediction row IDs for {prefix}")
        if labels.dtype.kind not in "iu" or not np.array_equal(labels, frame["label_idx"].to_numpy(dtype=int)):
            raise ValueError(f"Prediction labels are out of order or incorrect for {prefix}")
        if (
            probabilities.shape != (len(frame), len(CANONICAL_LABELS))
            or probabilities.dtype.kind != "f"
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6)
        ):
            raise ValueError(f"Malformed probability matrix for {prefix}")
    return arrays


def _best_model_hashes(path: Path) -> dict[str, str]:
    files = {item.name: item for item in path.iterdir() if item.is_file()}
    if (
        not {"config.json", "tokenizer_config.json"} <= files.keys()
        or not {"model.safetensors", "pytorch_model.bin"} & files.keys()
        or not {"tokenizer.json", "vocab.txt", "sentencepiece.bpe.model", "spiece.model"} & files.keys()
        or any(item.stat().st_size == 0 for item in files.values())
    ):
        raise ValueError(f"Incomplete exported model: {path}")
    return {name: sha256_file(item) for name, item in files.items()}


def _verify_trainer_checkpoint(checkpoint_root: Path, relative: str, *, require_scaler: bool = False) -> str:
    if not re.fullmatch(r"[1-9][0-9]*/(?:restored_[0-9a-f]{6}_)?checkpoint-[1-9][0-9]*", relative):
        raise ValueError(f"Invalid trainer checkpoint path: {relative}")
    path = checkpoint_root / relative
    if not _valid_checkpoint(path, require_scaler=require_scaler):
        raise ValueError(f"Trainer checkpoint integrity check failed: {path}")
    return sha256_file(path / _CHECKPOINT_COMMIT)


def _validate_unit(
    unit_dir: Path, record: dict[str, Any], uid: str, unit: tuple[str, str, str, int],
    split_meta: dict[str, Any], frames: dict[str, pd.DataFrame], commit: dict[str, Any],
    *, cfg: PostStageBConfig | None = None,
) -> dict[str, np.ndarray]:
    if (
        record["unit_id"] != uid
        or tuple(record[key] for key in ("experiment_id", "model_name", "text_view", "fold")) != unit
        or record["split_meta"] != split_meta
        or record["attempt"] != commit["attempt"]
    ):
        raise ValueError(f"Unit record does not match the expected unit and split: {uid}")
    if cfg is not None and cfg.tokenizer_revisions:
        expected_rev = str(cfg.tokenizer_revisions[unit[1]])
        if record.get("model_revision") != expected_rev or record.get("tokenizer_revision") != expected_rev:
            raise ValueError(
                f"Unit pinned revision mismatch for {uid}: expected {expected_rev!r}, "
                f"got model_revision={record.get('model_revision')!r}, tokenizer_revision={record.get('tokenizer_revision')!r}"
            )
    value = record["train_seconds"]
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid unit train_seconds: {uid}")
    metrics = record["metrics"]
    aliases = {"val": "val_overall", "val_adv": "val_adv_zwj_homoglyph"}
    for key, frame in frames.items():
        if key.startswith("cf_"):
            if metrics["counterfactual_probe"]["n_probed"] != len(frame):
                raise ValueError(f"Incorrect counterfactual count: {uid}")
            continue
        values = metrics[aliases.get(key, key)]
        if values["n_samples"] != len(frame) or not all(
            isinstance(values[metric], (int, float)) and math.isfinite(values[metric])
            for metric in ("accuracy", "macro_f1", "ece")
        ):
            raise ValueError(f"Invalid metric record for {key}: {uid}")
        low_fpr = values["smishing_low_fpr"]
        n_normal = int(frame["label_3class"].eq("Normal").sum())
        fp = low_fpr["false_positives"]
        if (
            low_fpr["n_normal"] != n_normal
            or low_fpr["n_smishing"] != int(frame["label_3class"].eq("Smishing").sum())
            or (fp is not None and (type(fp) is not int or not 0 <= fp <= n_normal))
            or (low_fpr["constraint_met"] and (not n_normal or fp is None or fp / n_normal > 0.01))
        ):
            raise ValueError(f"Invalid low-FPR metric record for {key}: {uid}")
    if "best_model_sha256" in commit and _best_model_hashes(unit_dir / "best_model") != commit["best_model_sha256"]:
        raise ValueError(f"Exported model checksum mismatch: {uid}")
    return _read_prediction_archive(unit_dir / "preds.npz", frames)


def _unit_attempt_dir(artifacts_dir: Path, uid: str, attempt: int) -> Path:
    if type(attempt) is not int or attempt < 1:
        raise ValueError(f"Invalid attempt number for {uid}")
    return artifacts_dir / "units" / uid / "attempts" / str(attempt)


def _local_checkpoint_base(artifacts_dir: Path, cfg: PostStageBConfig) -> Path:
    """Resolve a run-isolated local SSD checkpoint directory so distinct ARTIFACT_RUN_IDs never share checkpoints."""
    if cfg.local_checkpoint_root is None:
        return artifacts_dir / "checkpoints"
    run_scope = f"{artifacts_dir.name}_{hashlib.sha256(str(artifacts_dir.resolve()).encode('utf-8')).hexdigest()[:8]}"
    return Path(cfg.local_checkpoint_root) / cfg.run_name / run_scope


def _next_attempt(
    artifacts_dir: Path,
    uid: str,
    attempts: list[dict[str, Any]],
    *,
    local_ckpt_base: Path | None = None,
) -> int:
    # A hard interruption can leave an attempt directory even if the state write was lost.
    occupied = [entry["attempt"] for entry in attempts]
    ckpt_root = local_ckpt_base if local_ckpt_base is not None else artifacts_dir / "checkpoints"
    for path in (_unit_attempt_dir(artifacts_dir, uid, 1).parent, artifacts_dir / "checkpoints" / uid, ckpt_root / uid):
        if path.exists():
            occupied.extend(int(entry.name) for entry in path.iterdir() if entry.name.isdecimal())
    return max(occupied, default=0) + 1


def _persist_durable_trainer_checkpoint(
    *,
    src_ckpt_dir: Path,
    artifacts_dir: Path,
    uid: str,
    attempt_num: int,
    require_scaler: bool,
) -> dict[str, Any]:
    """Copy a kept trainer checkpoint durably into artifacts_dir/checkpoints/<uid>/<attempt>/ with SHA256 verification."""
    src_resolved = src_ckpt_dir.resolve()
    durable_root = (artifacts_dir / "checkpoints" / uid).resolve()
    if src_resolved.is_relative_to(durable_root):
        relative = src_resolved.relative_to(durable_root).as_posix()
        sha = _verify_trainer_checkpoint(durable_root, relative, require_scaler=require_scaler)
        return {"path": relative, "sha256": sha, "storage": "artifacts_dir", "require_scaler": require_scaler}
    dest_dir = durable_root / str(attempt_num) / src_resolved.name
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=False)
    for item in sorted(src_resolved.iterdir()):
        if not item.is_file() or _is_orphan_tmp_filename(item.name):
            continue
        dst_file = dest_dir / item.name
        tmp_file = dest_dir / f".{item.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        try:
            with open(item, "rb") as f_in, open(tmp_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
                f_out.flush()
                _fsync_fd_best_effort(f_out.fileno())
            os.replace(tmp_file, dst_file)
        except BaseException:
            try:
                tmp_file.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    _fsync_dir_best_effort(dest_dir)
    relative = dest_dir.relative_to(durable_root).as_posix()
    sha = _verify_trainer_checkpoint(durable_root, relative, require_scaler=require_scaler)
    return {"path": relative, "sha256": sha, "storage": "artifacts_dir", "require_scaler": require_scaler}


def _read_committed_unit(
    artifacts_dir: Path, uid: str, unit: tuple[str, str, str, int], split_meta: dict[str, Any],
    frames: dict[str, pd.DataFrame], commit: dict[str, Any],
    *, cfg: PostStageBConfig | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    unit_dir = _unit_attempt_dir(artifacts_dir, uid, commit["attempt"])
    if "trainer_checkpoint" in commit:
        checkpoint = commit["trainer_checkpoint"]
        if checkpoint.get("storage") == "artifacts_dir":
            ckpt_root = artifacts_dir / "checkpoints" / uid
        else:
            ckpt_root = (_local_checkpoint_base(artifacts_dir, cfg) if cfg is not None else artifacts_dir / "checkpoints") / uid
        actual = _verify_trainer_checkpoint(
            ckpt_root, checkpoint["path"], require_scaler=bool(checkpoint.get("require_scaler", False))
        )
        if actual != checkpoint["sha256"]:
            raise ValueError(f"Trainer checkpoint checksum mismatch: {uid}")
    metrics_path = unit_dir / "metrics.json"
    if sha256_file(metrics_path) != commit["metrics_sha256"] or sha256_file(unit_dir / "preds.npz") != commit["preds_sha256"]:
        raise ValueError(f"Committed metrics or prediction archive checksum mismatch: {uid}")
    record = json.loads(metrics_path.read_text(encoding="utf-8"))
    return record, _validate_unit(unit_dir, record, uid, unit, split_meta, frames, commit, cfg=cfg)


def run_post_stage_b_pipeline(
    *,
    bundle_dir: str | Path,
    artifacts_dir: str | Path,
    cfg: PostStageBConfig,
    resume: bool = True,
    verify_checksums: bool = True,
    train_eval_fn: Callable[..., dict[str, Any]] | None = None,
    stop_after_units: int | None = None,
    allow_stale_takeover: bool = False,
    expected_stale_writer_ids: tuple[str, ...] | list[str] | None = None,
    expected_corrupt_files: tuple[str, ...] | list[str] | None = None,
    takeover_reason: str | None = None,
    tokenizer_gate_report: dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Execute units with a verified run identity, immutable single-writer leases, and append-only attempt artifacts."""
    bdir = Path(bundle_dir).resolve()
    adir = Path(artifacts_dir).resolve()
    if adir.is_relative_to(bdir) or bdir.is_relative_to(adir):
        raise ValueError("Bundle and artifacts must use separate directories")
    runner_fn = train_eval_fn or hf_train_and_predict_fold
    return _run_locked(
        bdir,
        adir,
        cfg,
        runner_fn,
        resume,
        verify_checksums,
        stop_after_units,
        allow_stale_takeover=allow_stale_takeover,
        expected_stale_writer_ids=expected_stale_writer_ids,
        expected_corrupt_files=expected_corrupt_files,
        takeover_reason=takeover_reason,
        tokenizer_gate_report=tokenizer_gate_report,
    )


def _validate_pre_run_compatibility(
    cfg: PostStageBConfig,
    runner_fn: Callable[..., Any],
    env_fp: dict[str, Any],
    hw_fp: dict[str, Any],
) -> None:
    """Fail-closed pre-lease, pre-state compatibility validation before creating any run artifacts."""
    if (
        cfg.run_mode == "t4_benchmark"
        and cfg.benchmark_workload_fixture_eval
        and not SYNTHETIC_WORKLOAD_FIXTURE_ENABLED
    ):
        raise RuntimeError(
            "Fail closed before run initialization: t4_benchmark with benchmark_workload_fixture_eval=True "
            "is incompatible with the permanent synthetic-generation stop "
            "(SYNTHETIC_WORKLOAD_FIXTURE_ENABLED=False). "
            "No lease was acquired, no run state was initialized, and no training was executed. "
            "Obtain owner approval for an unmodified development-data-only benchmarking alternative."
        )
    if cfg.run_mode == "t4_benchmark":
        verify_t4_benchmark_hardware(hw_fp)
        if runner_fn is hf_train_and_predict_fold:
            _validate_required_packages_fingerprint(env_fp, cfg)
    elif hw_fp.get("cuda_available") and int(hw_fp.get("gpu_count") or 0) > 1:
        raise RuntimeError(
            f"Fail closed: Stage 4 single-GPU runner does not support multi-GPU execution "
            f"(found gpu_count={hw_fp.get('gpu_count')!r} in run_mode={cfg.run_mode!r}); "
            "TrainingArguments step budgets and checkpoint recovery assume exactly 1 GPU."
        )
    elif cfg.run_mode == "gpu_smoke_test" and runner_fn is hf_train_and_predict_fold:
        _validate_required_packages_fingerprint(env_fp, cfg)


def _run_locked(
    bdir: Path, adir: Path, cfg: PostStageBConfig, runner_fn: Callable[..., dict[str, Any]],
    resume: bool, verify_checksums: bool, stop_after_units: int | None,
    *,
    allow_stale_takeover: bool = False,
    expected_stale_writer_ids: tuple[str, ...] | list[str] | None = None,
    expected_corrupt_files: tuple[str, ...] | list[str] | None = None,
    takeover_reason: str | None = None,
    tokenizer_gate_report: dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    if not verify_checksums:
        raise ValueError("Post-Stage B runs require verified bundle checksums")
    env_fp = collect_stable_environment_fingerprint()
    hw_fp = collect_hardware_fingerprint()
    _validate_pre_run_compatibility(cfg, runner_fn, env_fp, hw_fp)
    identity = _run_identity(bdir, cfg, runner_fn, env_fp=env_fp, hw_fp=hw_fp)
    manifest = verify_manifest(bdir, verify_checksums=True)
    if sha256_file(bdir / "dataset_manifest.json") != identity["manifest_sha256"]:
        raise ValueError("Bundle manifest changed during verification")
    if cfg.run_adversarial_eval and any(view != "raw" for view in cfg.text_views):
        columns = manifest["bundle_files"]["real_train_pool.parquet"]["columns"]
        if any(f"text_adv_{view}" not in columns for view in cfg.text_views if view != "raw"):
            raise ValueError("Masked adversarial evaluation requires the corrected bundle")
    adir.mkdir(parents=True, exist_ok=True)
    lease = register_single_writer_lease(
        adir,
        run_name=cfg.run_name,
        config_hash=identity["config_hash"],
        allow_stale_takeover=allow_stale_takeover,
        expected_stale_writer_ids=expected_stale_writer_ids,
        expected_corrupt_files=expected_corrupt_files,
        takeover_reason=takeover_reason,
    )
    released_cleanly = False
    try:
        state_path = adir / "run_state.json"
        if not resume and state_path.exists():
            raise ValueError("A fresh run requires a separate artifact directory")
        state_exists = state_path.exists()
        state = _read_state(state_path, identity, expect_state=False)
        units = _expected_units(cfg)
        if (set(state["attempts"]) | set(state["committed_units"])) - units.keys():
            raise ValueError("Run state contains unexpected units")
        tok_gate_export_path = adir / "exports" / _TOKENIZER_GATE_EXPORT_NAME
        verified_tok_gate = _resolve_and_validate_tokenizer_gate(
            cfg=cfg,
            expected_config_hash=identity["config_hash"],
            expected_manifest_sha256=identity["manifest_sha256"],
            expected_env_packages=env_fp.get("packages"),
            tokenizer_gate_report=tokenizer_gate_report,
            persisted_export_path=tok_gate_export_path,
            state_exists=state_exists,
        )
        heldout_sets = _load_configured_eval_sets(bdir, cfg)
        if cfg.run_mode == "experiments_abcd":
            overlap = audit_model_view_overlaps(bdir, experiments=cfg.experiments, views=cfg.text_views, folds=cfg.folds)
            if overlap["failed_checks"] > 0:
                raise RuntimeError(
                    f"Exact model-input overlap blocks full training ({overlap['failed_checks']} failed checks)"
                )
            if runner_fn is hf_train_and_predict_fold:
                _validate_required_packages_fingerprint(env_fp, cfg)
        env_info = collect_environment_info()
        state["last_resumed_at_utc"] = _utc_now()
        state["active_writer_id"] = lease["writer_id"]
        state.setdefault("environments", []).append(env_info)
        _write_run_state(state_path, state)
        atomic_write_json(adir / "exports" / "environment_snapshot.json", env_info)
        save_config(cfg, adir / "exports" / "active_config.json")
        if verified_tok_gate is not None and not tok_gate_export_path.exists():
            atomic_write_json(tok_gate_export_path, verified_tok_gate)
        with _run_logger(adir / "logs" / "runner.log") as logger:
            out = _execute_units(
                bdir, adir, cfg, runner_fn, resume, stop_after_units, state, heldout_sets, logger, identity,
                writer_id=lease["writer_id"],
            )
            released_cleanly = True
            return out
    finally:
        try:
            lease_state = inspect_writer_leases(adir)
            active_ids = [entry["writer_id"] for entry in lease_state["active_leases"]]
            still_sole_active = (
                active_ids == [lease["writer_id"]]
                and not lease_state["unacknowledged_corrupt_files"]
            )
            if still_sole_active:
                release_single_writer_lease(
                    adir,
                    lease["writer_id"],
                    exit_reason="normal_completion" if released_cleanly else "exception_or_abort",
                )
        except OSError:
            if released_cleanly:
                raise


def _retraining_metadata_for_unit(
    *,
    cfg: PostStageBConfig,
    exp_id: str,
    model_name: str,
    text_view: str,
    fold: int,
    seed: int,
    split_meta: dict[str, Any],
    run_out: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, Any]:
    opt_budget = split_meta.get("optimizer_step_budget") or {}
    return {
        "unit_id": unit_id(exp_id, model_name, text_view, fold),
        "experiment_id": exp_id,
        "model_name": model_name,
        "model_revision": run_out.get("model_revision"),
        "tokenizer_revision": run_out.get("tokenizer_revision"),
        "text_view": text_view,
        "fold": int(fold),
        "seed": int(seed),
        "epochs": int(cfg.epochs),
        "batch_size": int(cfg.batch_size),
        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
        "effective_batch_size": int(cfg.batch_size * cfg.gradient_accumulation_steps),
        "learning_rate": float(cfg.learning_rate),
        "weight_decay": float(cfg.weight_decay),
        "warmup_ratio": float(cfg.warmup_ratio),
        "max_length": int(cfg.max_length),
        "balance_mode": split_meta.get("balance_mode"),
        "class_weights": split_meta.get("class_weights"),
        "expected_optimizer_steps": opt_budget.get("expected_optimizer_steps"),
        "expected_trainer_update_steps": opt_budget.get("expected_trainer_update_steps", opt_budget.get("expected_optimizer_steps")),
        "optimizer_step_budget": split_meta.get("optimizer_step_budget"),
        "actual_global_step": run_out.get("actual_global_step"),
        "successful_optimizer_weight_updates": run_out.get("successful_optimizer_weight_updates"),
        "keep_model_weights": cfg.keep_model_weights,
        "config_hash": identity["config_hash"],
        "dataset_manifest_sha256": identity["manifest_sha256"],
    }


def _reconcile_committed_unit_post_commit(
    *,
    adir: Path,
    state_path: Path,
    state: dict[str, Any],
    cfg: PostStageBConfig,
    runner_fn: Callable[..., Any],
    identity: dict[str, Any],
    uid: str,
    fold: int,
    commit: dict[str, Any],
    logger: logging.Logger,
    writer_id: str | None = None,
) -> dict[str, Any]:
    """Idempotently reconcile completion event and post-commit checkpoint cleanup for a durably committed unit.

    Authoritative invariant: `run_state.json` already holds the durable commit for `uid`, and `metrics.json`
    is never rewritten by post-commit reconciliation or pruning recovery.
    """
    if writer_id is not None:
        verify_writer_lease_still_sole_active(adir, writer_id)
    keep_this_unit = _should_keep_weights(cfg.keep_model_weights, fold)
    staging_unit_dir = adir / "staging" / uid
    events_dir = adir / "events"
    pointers_dir = adir / "pointers"
    local_ckpt_base = _local_checkpoint_base(adir, cfg)

    event_result: dict[str, Any] | None = None
    state_dirty = False
    if runner_fn is hf_train_and_predict_fold:
        event_result = ensure_unit_completion_event(
            events_dir=events_dir,
            uid=uid,
            attempt=int(commit["attempt"]),
            config_hash=identity["config_hash"],
            manifest_sha256=identity["manifest_sha256"],
            metrics_sha256=commit["metrics_sha256"],
            preds_sha256=commit["preds_sha256"],
        )
        if not keep_this_unit:
            scan_done = scan_unit_staging(
                staging_unit_dir=staging_unit_dir,
                events_dir=events_dir,
                pointers_dir=pointers_dir,
                uid=uid,
                expected_config_hash=identity["config_hash"],
                expected_manifest_sha256=identity["manifest_sha256"],
                unit_already_committed=True,
                active_attempt=int(commit["attempt"]),
                trust_commit_events=True,
            )
            post_commit_prune = prune_superseded_staging(scan_report=scan_done, dry_run=False)
            prune_errors = post_commit_prune.get("prune_errors") or []
            if prune_errors:
                logger.warning("Post-commit checkpoint pruning errors for %s: %s", uid, prune_errors)
                new_errs = [{"phase": "post_commit", **e} for e in prune_errors]
                if commit.get("post_commit_prune_errors") != new_errs:
                    commit["post_commit_prune_errors"] = new_errs
                    state_dirty = True
                if commit.get("training_checkpoints_pruned_post_commit") is not False:
                    commit["training_checkpoints_pruned_post_commit"] = False
                    state_dirty = True
            else:
                if "post_commit_prune_errors" in commit:
                    commit.pop("post_commit_prune_errors", None)
                    state_dirty = True
                if commit.get("training_checkpoints_pruned_post_commit") is not True:
                    commit["training_checkpoints_pruned_post_commit"] = True
                    state_dirty = True

    if not keep_this_unit and cfg.local_checkpoint_root is not None:
        local_unit_ckpt_root = local_ckpt_base / uid
        if local_unit_ckpt_root.exists():
            try:
                shutil.rmtree(local_unit_ckpt_root)
            except OSError as exc:
                logger.warning("Local SSD checkpoint cleanup warning for %s: %s", uid, exc)

    if state_dirty:
        if writer_id is not None:
            verify_writer_lease_still_sole_active(adir, writer_id)
        _write_run_state(state_path, state)
    return {"event_result": event_result, "state_updated": state_dirty}


def _execute_units(
    bdir: Path, adir: Path, cfg: PostStageBConfig, runner_fn: Callable[..., dict[str, Any]],
    resume: bool, stop_after_units: int | None, state: dict[str, Any], heldout_sets: dict[str, pd.DataFrame],
    logger: logging.Logger, identity: dict[str, Any], *, writer_id: str | None = None,
) -> dict[str, Any]:
    state_path = adir / "run_state.json"
    executed = 0

    def _verify_lease() -> None:
        if writer_id is not None:
            verify_writer_lease_still_sole_active(adir, writer_id)

    for uid, unit in _expected_units(cfg).items():
        _verify_lease()
        exp_id, model_name, text_view, fold = unit
        if stop_after_units is not None and executed >= stop_after_units:
            break
        train_df, val_df, unit_sets, split_meta = _split_for_unit(bdir, heldout_sets, cfg, exp_id, fold)
        frames = _prediction_frames(cfg, val_df, unit_sets)
        attempts = state["attempts"].setdefault(uid, [])
        committed = state["committed_units"].get(uid)
        if committed is not None:
            try:
                if not attempts or (
                    attempts[-1]["attempt"] != committed["attempt"]
                    or attempts[-1]["status"] != "completed"
                    or "invalidated" in attempts[-1]
                ):
                    raise ValueError("Commit is not linked to a successfully completed attempt")
                _read_committed_unit(adir, uid, unit, split_meta, frames, committed, cfg=cfg)
            except (OSError, ValueError, EOFError, KeyError, TypeError, BadZipFile) as exc:
                if not attempts or attempts[-1].get("attempt") != committed.get("attempt"):
                    raise ValueError(f"Inconsistent attempt history for {uid}") from exc
                attempts[-1]["invalidated"] = {"at_utc": _utc_now(), "reason": str(exc)}
                state["committed_units"].pop(uid)
                logger.warning("Retrying invalid committed unit %s: %s", uid, exc)
            else:
                _reconcile_committed_unit_post_commit(
                    adir=adir,
                    state_path=state_path,
                    state=state,
                    cfg=cfg,
                    runner_fn=runner_fn,
                    identity=identity,
                    uid=uid,
                    fold=fold,
                    commit=committed,
                    logger=logger,
                    writer_id=writer_id,
                )
                logger.info("SKIP verified unit: %s", uid)
                continue
        if attempts and attempts[-1]["status"] == "running":
            attempts[-1].update(status="interrupted", ended_at_utc=_utc_now())
        local_ckpt_base = _local_checkpoint_base(adir, cfg)
        attempt = {
            "attempt": _next_attempt(adir, uid, attempts, local_ckpt_base=local_ckpt_base),
            "status": "running",
            "started_at_utc": _utc_now(),
        }
        attempts.append(attempt)
        _verify_lease()
        _write_run_state(state_path, state)

        unit_dir = _unit_attempt_dir(adir, uid, attempt["attempt"])
        checkpoint_dir = local_ckpt_base / uid / str(attempt["attempt"])
        staging_unit_dir = adir / "staging" / uid
        events_dir = adir / "events"
        pointers_dir = adir / "pointers"
        keep_this_unit = _should_keep_weights(cfg.keep_model_weights, fold)
        best_model_dir = (
            unit_dir / "best_model"
            if runner_fn is hf_train_and_predict_fold and keep_this_unit
            else None
        )
        expected_steps = int(split_meta["optimizer_step_budget"]["expected_optimizer_steps"])
        logger.info(
            "RUN unit=%s attempt=%d train=%d/%d val=%d/%d expected_trainer_steps=%d",
            uid, attempt["attempt"],
            split_meta["n_train_actual"], split_meta["n_train_pre_cap"],
            split_meta["n_val_actual"], split_meta["n_val_pre_cap"],
            expected_steps,
        )
        try:
            kwargs: dict[str, Any] = {}
            if runner_fn is hf_train_and_predict_fold:
                if best_model_dir is not None:
                    kwargs["best_model_export_dir"] = best_model_dir
                kwargs.update(
                    {
                        "staging_unit_dir": staging_unit_dir,
                        "events_dir": events_dir,
                        "pointers_dir": pointers_dir,
                        "uid": uid,
                        "attempt": attempt["attempt"],
                        "config_hash": identity["config_hash"],
                        "manifest_sha256": identity["manifest_sha256"],
                        "lease_check_fn": _verify_lease,
                    }
                )
            run_out = runner_fn(
                train_df=train_df, val_df=val_df, heldout_sets=unit_sets, model_name=model_name,
                text_view=text_view, checkpoint_dir=checkpoint_dir, cfg=cfg,
                class_weights=split_meta["class_weights"], seed=cfg.seed + fold,
                resume_from_checkpoint=resume, **kwargs,
            )
            _verify_lease()
            actual_global_step = run_out.get("actual_global_step")
            if runner_fn is hf_train_and_predict_fold:
                if type(actual_global_step) is not int or actual_global_step <= 0:
                    raise ValueError(
                        f"Fail closed: Trainer reported invalid actual_global_step={actual_global_step!r} for {uid}"
                    )
                if actual_global_step != expected_steps:
                    raise ValueError(
                        f"Fail closed: Trainer actual_global_step={actual_global_step} != "
                        f"expected Trainer update-step budget {expected_steps} for {uid}"
                    )
            preds_path = unit_dir / "preds.npz"
            _write_prediction_archive(preds_path, run_out["predictions"], frames)
            predictions = _read_prediction_archive(preds_path, frames)
            unit_metrics = _compute_unit_metrics(val_df=val_df, heldout_sets=unit_sets, predictions=predictions, cfg=cfg)
            retrain_meta = _retraining_metadata_for_unit(
                cfg=cfg, exp_id=exp_id, model_name=model_name, text_view=text_view, fold=fold,
                seed=cfg.seed + fold, split_meta=split_meta, run_out=run_out, identity=identity,
            )
            record = {
                "unit_id": uid, "experiment_id": exp_id, "model_name": model_name, "text_view": text_view,
                "fold": fold, "seed": cfg.seed + fold, "attempt": attempt["attempt"], "split_meta": split_meta,
                "train_seconds": run_out.get("train_seconds", 0.0),
                "attempt_train_seconds": run_out.get("attempt_train_seconds", run_out.get("train_seconds", 0.0)),
                "prior_resumed_train_seconds": run_out.get("prior_resumed_train_seconds", 0.0),
                "cumulative_train_seconds": run_out.get("cumulative_train_seconds", run_out.get("train_seconds", 0.0)),
                "resumed_from_step": run_out.get("resumed_from_step", 0),
                "steps_executed_this_attempt": run_out.get("steps_executed_this_attempt"),
                "init_seconds": run_out.get("init_seconds"),
                "tokenization_seconds": run_out.get("tokenization_seconds"),
                "eval_seconds": run_out.get("eval_seconds"),
                "inference_seconds": run_out.get("inference_seconds"),
                "local_checkpoint_seconds": run_out.get("local_checkpoint_seconds"),
                "drive_sync_seconds": run_out.get("drive_sync_seconds"),
                "attempt_optimizer_step_ms": run_out.get("attempt_optimizer_step_ms"),
                "optimizer_step_ms": run_out.get("optimizer_step_ms"),
                "expected_optimizer_steps": expected_steps,
                "expected_trainer_update_steps": expected_steps,
                "actual_global_step": actual_global_step,
                "successful_optimizer_weight_updates": run_out.get("successful_optimizer_weight_updates"),
                "fp16": run_out.get("fp16"),
                "bf16": run_out.get("bf16"),
                "peak_cuda_allocated_bytes": run_out.get("peak_cuda_allocated_bytes"),
                "peak_cuda_reserved_bytes": run_out.get("peak_cuda_reserved_bytes"),
                "model_revision": run_out.get("model_revision"),
                "tokenizer_revision": run_out.get("tokenizer_revision"),
                "resumed_from_checkpoint": run_out.get("resumed_from_checkpoint"),
                "recovery_granularity": run_out.get("recovery_granularity", RECOVERY_GRANULARITY),
                "retraining_metadata": retrain_meta,
                "benchmark_fixture_metadata": run_out.get("benchmark_fixture_metadata"),
                "checkpoint_prune_errors": list(run_out.get("checkpoint_prune_errors") or []),
                "metrics": unit_metrics, "completed_at_utc": _utc_now(),
            }
            metrics_path = unit_dir / "metrics.json"
            atomic_write_json(metrics_path, record)
            # Durable readback verification before computing commit SHA256 or validating unit.
            disk_record = json.loads(metrics_path.read_text(encoding="utf-8"))
            if disk_record != record:
                raise IOError(f"Durable readback mismatch for {metrics_path}")
            commit: dict[str, Any] = {
                "attempt": attempt["attempt"],
                "metrics_sha256": sha256_file(metrics_path),
                "preds_sha256": sha256_file(preds_path),
                "training_checkpoints_pruned_post_commit": False,
            }
            if best_model_dir is not None:
                commit["best_model_sha256"] = _best_model_hashes(best_model_dir)
            if runner_fn is hf_train_and_predict_fold:
                best = run_out.get("best_trainer_checkpoint")
                if not best:
                    raise ValueError("Trainer did not select a validation checkpoint")
                if keep_this_unit:
                    commit["trainer_checkpoint"] = _persist_durable_trainer_checkpoint(
                        src_ckpt_dir=Path(best),
                        artifacts_dir=adir,
                        uid=uid,
                        attempt_num=attempt["attempt"],
                        require_scaler=bool(run_out.get("fp16")),
                    )
            # Validate unit artifacts (using durable disk_record) BEFORE state commit, event emission, or pruning.
            _validate_unit(unit_dir, disk_record, uid, unit, split_meta, frames, commit, cfg=cfg)
        except Exception as exc:
            attempt.update(status="failed", ended_at_utc=_utc_now(), error=repr(exc))
            try:
                _verify_lease()
                _write_run_state(state_path, state)
            except Exception:
                pass
            logger.exception("FAILED unit=%s attempt=%d", uid, attempt["attempt"])
            raise
        # Verify lease ownership immediately before durable authoritative commit in run_state.json.
        _verify_lease()
        attempt.update(status="completed", ended_at_utc=_utc_now())
        state["committed_units"][uid] = commit
        _write_run_state(state_path, state)
        _reconcile_committed_unit_post_commit(
            adir=adir,
            state_path=state_path,
            state=state,
            cfg=cfg,
            runner_fn=runner_fn,
            identity=identity,
            uid=uid,
            fold=fold,
            commit=commit,
            logger=logger,
            writer_id=writer_id,
        )
        executed += 1
        logger.info("DONE unit=%s val_macro_f1=%s%%", uid, unit_metrics["val_overall"]["macro_f1"])
    _verify_lease()
    return _aggregate_verified(bdir, adir, cfg, state, heldout_sets)


def aggregate_post_stage_b_results(
    bundle_dir: str | Path,
    artifacts_dir: str | Path,
    cfg: PostStageBConfig,
    *,
    train_eval_fn: Callable[..., dict[str, Any]] | None = None,
    offline_cpu_ok: bool = True,
) -> dict[str, Any]:
    """Validate and export the expected units committed by this run (supports offline CPU re-aggregation)."""
    bdir, adir = Path(bundle_dir).resolve(), Path(artifacts_dir).resolve()
    lease_state = inspect_writer_leases(adir)
    if lease_state["active_leases"] or lease_state["unacknowledged_corrupt_files"]:
        raise RuntimeError(
            "Cannot aggregate while unresolved active writer lease(s) or corrupt lease records exist: "
            f"active={[l['writer_id'] for l in lease_state['active_leases']]}, "
            f"corrupt={[c['file'] for c in lease_state['unacknowledged_corrupt_files']]}"
        )
    identity = _run_identity(bdir, cfg, train_eval_fn or hf_train_and_predict_fold)
    verify_manifest(bdir, verify_checksums=True)
    if sha256_file(bdir / "dataset_manifest.json") != identity["manifest_sha256"]:
        raise ValueError("Bundle manifest changed during verification")
    state = _read_state(
        adir / "run_state.json",
        identity,
        expect_state=True,
        allow_offline_hardware_diff=offline_cpu_ok,
    )
    summary = _aggregate_verified(bdir, adir, cfg, state, _load_configured_eval_sets(bdir, cfg))
    summary["aggregated_offline_hardware_diff"] = bool(state.get("identity") != identity)
    return summary


def _aggregate_verified(
    bdir: Path, adir: Path, cfg: PostStageBConfig, state: dict[str, Any], eval_sets: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    units = _expected_units(cfg)
    commits = state["committed_units"]
    if commits.keys() - units.keys():
        raise ValueError("Run state contains unexpected committed units")
    rows: list[dict[str, Any]] = []
    retraining_registry: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str], dict[int, dict[str, np.ndarray]]] = {}
    ensemble_groups: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = {}
    split_cache = {}
    identity_cfg_hash = state["identity"]["config_hash"]
    identity_manifest_sha = state["identity"]["manifest_sha256"]
    for uid, unit in units.items():
        if uid not in commits:
            continue
        exp_id, model_name, text_view, fold = unit
        commit = commits[uid]
        history = state["attempts"][uid]
        if not history or (
            history[-1]["attempt"] != commit["attempt"] or history[-1]["status"] != "completed" or "invalidated" in history[-1]
        ):
            raise ValueError(f"Commit is not linked to a successfully completed attempt: {uid}")
        split_key = (exp_id, fold)
        if split_key not in split_cache:
            _, val_df, unit_sets, split_meta = _split_for_unit(bdir, eval_sets, cfg, exp_id, fold)
            split_cache[split_key] = split_meta, _prediction_frames(cfg, val_df, unit_sets)
        split_meta, frames = split_cache[split_key]
        rec, arrays = _read_committed_unit(adir, uid, unit, split_meta, frames, commit, cfg=cfg)
        metrics = rec["metrics"]
        val = metrics["val_overall"]
        cf = metrics.get("counterfactual_probe", {})
        opt_budget = split_meta.get("optimizer_step_budget", {})
        row = {
            "unit_id": uid, "experiment_id": exp_id, "model_name": model_name, "text_view": text_view, "fold": fold,
            **{
                field: split_meta[field] for field in (
                    "n_train_pre_cap", "n_val_pre_cap", "n_train_actual", "n_val_actual",
                    "n_train_real", "n_train_synthetic", "train_by_class", "val_by_class",
                )
            },
            "expected_optimizer_steps": opt_budget.get("expected_optimizer_steps"),
            "actual_global_step": rec.get("actual_global_step"),
            "effective_batch_size": opt_budget.get("effective_batch_size", cfg.batch_size * cfg.gradient_accumulation_steps),
            "step_ratio_vs_arm_a": opt_budget.get("step_ratio_vs_arm_a"),
            "statistical_validation_status": (
                "VALIDATED_CONFIRMATORY" if CONFIRMATORY_DECLARATIONS_ENABLED else STATISTICAL_VALIDATION_STATUS
            ),
            "confirmatory_declarations_enabled": bool(CONFIRMATORY_DECLARATIONS_ENABLED),
            "val_macro_f1": val["macro_f1"],
            "val_accuracy": val["accuracy"],
            "val_recall_at_1pct_fpr": val["smishing_recall_at_1pct_fpr"],
            "val_low_fpr_threshold": val["smishing_low_fpr"]["threshold"],
            "val_low_fpr_achieved_pct": val["smishing_low_fpr"]["actual_fpr"],
            "val_low_fpr_false_positives": val["smishing_low_fpr"]["false_positives"],
            "val_low_fpr_n_normal": val["smishing_low_fpr"]["n_normal"],
            "val_normal_as_smish_fpr": val["normal_as_smishing_fpr_argmax"],
            "val_ece": val["ece"],
            "sd_test_macro_f1": metrics.get("smishdetect_test", {}).get("macro_f1"),
            "bp_ood_macro_f1": metrics.get("banglaphish_ood", {}).get("macro_f1"),
            "cf_n_probed": cf.get("n_probed"),
            "cf_contact_brand_pp_delta": cf.get("contact_brand_swap_pp_delta"),
            "train_seconds": rec["train_seconds"],
            "attempt_train_seconds": rec.get("attempt_train_seconds", rec["train_seconds"]),
            "prior_resumed_train_seconds": rec.get("prior_resumed_train_seconds", 0.0),
            "cumulative_train_seconds": rec.get("cumulative_train_seconds", rec["train_seconds"]),
            "resumed_from_step": rec.get("resumed_from_step", 0),
            "steps_executed_this_attempt": rec.get("steps_executed_this_attempt"),
            "init_seconds": rec.get("init_seconds"),
            "tokenization_seconds": rec.get("tokenization_seconds"),
            "eval_seconds": rec.get("eval_seconds"),
            "inference_seconds": rec.get("inference_seconds"),
            "local_checkpoint_seconds": rec.get("local_checkpoint_seconds"),
            "drive_sync_seconds": rec.get("drive_sync_seconds"),
            "attempt_optimizer_step_ms": rec.get("attempt_optimizer_step_ms"),
            "optimizer_step_ms": rec.get("optimizer_step_ms"),
            "peak_cuda_allocated_bytes": rec["peak_cuda_allocated_bytes"],
            "peak_cuda_reserved_bytes": rec["peak_cuda_reserved_bytes"],
        }
        rows.append(row)
        if "retraining_metadata" in rec:
            retraining_registry.append(rec["retraining_metadata"])
        paired_arrays = {key: arrays[key] for key in ("val_row_ids", "val_y", "val_probs")}
        if "cf_orig_row_ids" in arrays:
            paired_arrays["cf_orig_row_ids"] = arrays["cf_orig_row_ids"]
        groups.setdefault((exp_id, model_name, text_view), {})[fold] = paired_arrays
        ensemble_groups.setdefault((exp_id, model_name, text_view), {})[fold] = {
            "experiment_id": exp_id,
            "model_name": model_name,
            "text_view": text_view,
            "model_revision": rec.get("model_revision"),
            "tokenizer_revision": rec.get("tokenizer_revision"),
            "config_hash": identity_cfg_hash,
            "dataset_manifest_sha256": identity_manifest_sha,
            "arrays": arrays,
        }

    real = pd.read_parquet(bdir / "real_train_pool.parquet")
    oof_summaries: dict[str, Any] = {}
    oof_pairs: dict[tuple[str, str, str], tuple[np.ndarray, ...]] = {}
    probe_counts = {}
    n_boot = 100 if cfg.run_mode == "gpu_smoke_test" else 1000
    for (exp_id, model_name, text_view), fold_map in groups.items():
        if set(fold_map) != set(range(5)):
            continue
        key = f"exp_{exp_id}__{model_name}__{text_view}"
        if cfg.run_counterfactual_eval and cfg.smoke_max_eval_samples is None:
            probe_ids = [
                row_id for f in range(5) for row_id in fold_map[f].get("cf_orig_row_ids", [])
            ]
            expected = set(eval_sets["counterfactual_smish_probe"]["row_id"])
            if len(probe_ids) != len(expected) or set(probe_ids) != expected:
                raise ValueError(f"Counterfactual OOF coverage mismatch: {key}")
            probe_counts[key] = len(probe_ids)
        if cfg.smoke_max_val_samples is not None:
            continue
        ids = np.concatenate([fold_map[f]["val_row_ids"] for f in range(5)])
        ys = np.concatenate([fold_map[f]["val_y"] for f in range(5)])
        ps = np.concatenate([fold_map[f]["val_probs"] for f in range(5)])
        reference = pd.concat([real[real["curation_cv_fold"] == f] for f in range(5)])
        if (
            len(np.unique(ids)) != len(real)
            or not np.array_equal(ids, reference["row_id"].to_numpy(dtype=str))
            or not np.array_equal(ys, reference["label_idx"].to_numpy(dtype=int))
        ):
            raise ValueError(f"OOF predictions do not match the ordered canonical real rows: {key}")
        family_ids = reference["family_id"].to_numpy()
        oof = _evaluate_frame(reference, ps, n_boot=n_boot, seed=int(cfg.seed))
        fold_rows = [r for r in rows if (r["experiment_id"], r["model_name"], r["text_view"]) == (exp_id, model_name, text_view)]
        oof["fold_macro_f1_sd_pp"] = float(np.std([r["val_macro_f1"] for r in fold_rows], ddof=1))
        oof_summaries[key] = oof
        oof_pairs[(exp_id, model_name, text_view)] = (ids, ys, ps.argmax(1), family_ids)

    mcnemar_vs_exp_a: dict[str, Any] = {}
    for (exp_id, model_name, text_view), (ids, ys, preds, families) in oof_pairs.items():
        if exp_id == "A" or ("A", model_name, text_view) not in oof_pairs:
            continue
        base_ids, base_y, base_preds, base_families = oof_pairs[("A", model_name, text_view)]
        if not all(np.array_equal(a, b) for a, b in ((ids, base_ids), (ys, base_y), (families, base_families))):
            raise ValueError("McNemar comparison requires identical ordered row IDs, labels, and families")
        result = mcnemar_pvalue(ys, base_preds, preds)
        result["interpretation"] = "Exploratory row-level test. Family and training-fold dependencies can violate independence."
        result.update(paired_family_accuracy_delta(ys, base_preds, preds, families, n_resamples=n_boot, seed=int(cfg.seed)))
        mcnemar_vs_exp_a[f"{exp_id}_vs_A__{model_name}__{text_view}"] = result
    planned_comparisons = (len(cfg.experiments) - 1) * len(cfg.models) * len(cfg.text_views) if "A" in cfg.experiments else 0
    if mcnemar_vs_exp_a and len(mcnemar_vs_exp_a) == planned_comparisons:
        adjusted = holm_adjust_pvalues({name: result["p_value"] for name, result in mcnemar_vs_exp_a.items()})
        for name, p_value in adjusted.items():
            mcnemar_vs_exp_a[name]["holm_adjusted_p_value"] = p_value

    ensemble_evaluation: dict[str, Any] | None = None
    complete_heldout_groups = {
        k: v
        for k, v in ensemble_groups.items()
        if set(v) == set(range(5))
        and cfg.run_heldout_eval
        and cfg.smoke_max_eval_samples is None
        and cfg.smoke_max_val_samples is None
    }
    if complete_heldout_groups:
        test_masks = None
        if len(eval_sets["smishdetect_test"]) == 1698:
            mask_payload = build_smishdetect_test_cohort_masks(bdir, verify_tfidf_reproduction=False)
            test_masks = {name: entry["mask"] for name, entry in mask_payload["cohorts"].items()}
        ensemble_evaluation = build_ensemble_evaluation_summary(
            ensemble_groups=complete_heldout_groups,
            eval_frames=eval_sets,
            test_masks=test_masks,
            n_boot_ci=100 if cfg.run_mode == "gpu_smoke_test" else DEFAULT_B_CI,
            n_boot_pval=200 if cfg.run_mode == "gpu_smoke_test" else DEFAULT_B_PVAL,
            seed=cfg.seed,
        )

    stat_status = "VALIDATED_CONFIRMATORY" if CONFIRMATORY_DECLARATIONS_ENABLED else STATISTICAL_VALIDATION_STATUS
    registry_canonical = json.dumps(retraining_registry, sort_keys=True, ensure_ascii=False).encode("utf-8")
    registry_sha256 = hashlib.sha256(registry_canonical).hexdigest()
    retraining_metadata_doc = {
        "schema_version": "stage4_retraining_metadata_v1",
        "run_name": cfg.run_name,
        "run_mode": cfg.run_mode,
        "generated_at_utc": _utc_now(),
        "keep_model_weights": cfg.keep_model_weights,
        "selection_rule": "Configured statically via PostStageBConfig.keep_model_weights; NEVER selected using heldout/OOD performance.",
        "statistical_validation_status": stat_status,
        "confirmatory_declarations_enabled": bool(CONFIRMATORY_DECLARATIONS_ENABLED),
        "unit_count": len(retraining_registry),
        "registry_sha256": registry_sha256,
        "units": retraining_registry,
    }

    summary = {
        "run_name": cfg.run_name, "run_mode": cfg.run_mode, "generated_at_utc": retraining_metadata_doc["generated_at_utc"],
        "completed_unit_count": len(rows), "expected_unit_count": len(units),
        "statistical_governance": {
            "statistical_validation_status": stat_status,
            "confirmatory_declarations_enabled": bool(CONFIRMATORY_DECLARATIONS_ENABLED),
            "confirmatory_mode_grid_guard": bool(cfg.confirmatory_mode),
            "note": (
                "confirmatory_mode=True restricts only the 160-unit experimental grid and NEVER authorizes "
                "statistical superiority claims while confirmatory_declarations_enabled=False."
            ),
        },
        "unit_rows": rows,
        "oof_5fold_summaries": oof_summaries,
        "oof_counterfactual_probe_counts": probe_counts,
        "mcnemar_vs_exp_a": mcnemar_vs_exp_a,
        "planned_mcnemar_comparisons": planned_comparisons,
        "ensemble_5fold_evaluation": ensemble_evaluation,
        "weight_retention_policy": {
            "keep_model_weights": cfg.keep_model_weights,
            "selection_rule": retraining_metadata_doc["selection_rule"],
            "retraining_metadata_file": "exports/retraining_metadata.json",
            "retraining_registry_sha256": registry_sha256,
            "retraining_registry": retraining_registry,
        },
    }
    _write_summary(adir / "exports", cfg, summary, retraining_metadata_doc)
    return summary


def _write_summary(
    exports_dir: Path,
    cfg: PostStageBConfig,
    summary: dict[str, Any],
    retraining_metadata_doc: dict[str, Any],
) -> None:
    # Preserve earlier partial summaries for review while maintaining the conventional latest paths.
    snapshot_dir = exports_dir / "history" / summary["generated_at_utc"].replace(":", "-")
    rows = summary["unit_rows"]
    frame = pd.DataFrame(rows)
    for field in ("train_by_class", "val_by_class"):
        if field in frame:
            frame[field] = frame[field].map(json.dumps)
    csv_text = frame.to_csv(index=False) if rows else ""
    gov = summary["statistical_governance"]
    md = [
        f"# Post-Stage B Run Report for `{cfg.run_name}` (`{cfg.run_mode}`)", "",
        f"Completed units **{len(rows)}/{summary['expected_unit_count']}**.",
        "",
        "## Statistical Governance & Retraining Provenance",
        f"- **Statistical validation status:** `{gov['statistical_validation_status']}`",
        f"- **Confirmatory superiority declarations enabled:** `{gov['confirmatory_declarations_enabled']}` (p-values are unvalidated diagnostic only; descriptive deltas & CIs preserved)",
        f"- **Config `confirmatory_mode` (grid guard only):** `{gov['confirmatory_mode_grid_guard']}`",
        f"- **Retraining metadata artifact:** `exports/retraining_metadata.json` (`units={retraining_metadata_doc['unit_count']}`, `registry_sha256={retraining_metadata_doc['registry_sha256'][:16]}...`, `keep_model_weights={cfg.keep_model_weights}`)",
        "",
        "| Unit | Train pre-cap | Train actual | Validation pre-cap | Validation actual | Train real | Train synthetic | Train N/P/S | Validation N/P/S | CV Macro-F1 % | CV recall at ≤1% FPR % | Threshold | FPR % | FP / Normal | Held-out test F1 % | OOD F1 % | CF rows | CF brand swap Δ pp | Expected opt steps | Actual global step | Effective batch | Step ratio vs A | Statistical status | Confirmatory enabled |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        train_class = "/".join(str(row["train_by_class"][label]) for label in CANONICAL_LABELS)
        val_class = "/".join(str(row["val_by_class"][label]) for label in CANONICAL_LABELS)
        md.append(
            f"| `{row['unit_id']}` | {row['n_train_pre_cap']} | {row['n_train_actual']} | {row['n_val_pre_cap']} | "
            f"{row['n_val_actual']} | {row['n_train_real']} | {row['n_train_synthetic']} | {train_class} | {val_class} | "
            f"{row['val_macro_f1']} | {row['val_recall_at_1pct_fpr']} | {row['val_low_fpr_threshold']} | "
            f"{row['val_low_fpr_achieved_pct']} | {row['val_low_fpr_false_positives']} / {row['val_low_fpr_n_normal']} | "
            f"{row['sd_test_macro_f1']} | {row['bp_ood_macro_f1']} | "
            f"{row['cf_n_probed']} | {row['cf_contact_brand_pp_delta']} | "
            f"{row['expected_optimizer_steps']} | {row['actual_global_step']} | {row['effective_batch_size']} | "
            f"{row['step_ratio_vs_arm_a']} | `{row['statistical_validation_status']}` | {row['confirmatory_declarations_enabled']} |"
        )
    if rows:
        md.extend([
            "",
            "## Equal-Epoch Optimizer-Step Budget Disclosure (Expected Trainer Update-Step Budget vs. Actual `global_step`)",
            "",
            "- **Step-count semantics (`transformers==4.57.1`):** `expected_optimizer_steps` is the expected Trainer "
            "update-step budget; `actual_global_step` is `Trainer.state.global_step` (incremented per gradient-accumulation "
            "step even if FP16 `GradScaler` skips a weight update on overflow); successful optimizer weight updates are "
            "not separately instrumented unless explicitly populated.",
        ])
        seen_arms: set[str] = set()
        for row in rows:
            arm = row["experiment_id"]
            if arm in seen_arms:
                continue
            seen_arms.add(arm)
            md.append(
                f"- Arm `{arm}`: train_actual=`{row['n_train_actual']}`, epochs=`{cfg.epochs}`, "
                f"effective_batch_size=`{row['effective_batch_size']}`, expected_trainer_update_steps=`{row['expected_optimizer_steps']}`, "
                f"actual_global_step=`{row['actual_global_step']}`, step_ratio_vs_arm_a=`{row['step_ratio_vs_arm_a']}`"
            )
    if summary["oof_5fold_summaries"]:
        md.extend(["", "## Five-fold OOF validation", ""])
        for key, metrics in summary["oof_5fold_summaries"].items():
            md.append(
                f"- `{key}` has macro-F1 **{metrics['macro_f1']}%** and family-bootstrap CI {metrics['macro_f1_ci95']}. "
                f"The across-fold macro-F1 SD is {metrics['fold_macro_f1_sd_pp']:.2f} pp."
            )
    report = "\n".join(md) + "\n"
    for target in (snapshot_dir, exports_dir):
        atomic_write_json(target / "aggregated_results.json", summary)
        atomic_write_json(target / "retraining_metadata.json", retraining_metadata_doc)
        atomic_write_text(target / "aggregated_results.csv", csv_text)
        atomic_write_text(target / "post_stage_b_report.md", report)
