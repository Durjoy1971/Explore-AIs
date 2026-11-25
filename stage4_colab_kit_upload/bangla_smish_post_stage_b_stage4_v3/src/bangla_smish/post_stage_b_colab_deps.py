"""Stage 4 Colab runtime dependency preflight, constrained installation, and post-install verification.

Can be executed both directly (via `runpy.run_path` before importing heavy GPU modules) and via standard
package import after `src/` is on `sys.path`. Never installs GPU packages when invoked locally in mocked tests.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any, Callable, Iterable

from packaging.requirements import Requirement
from packaging.version import Version

try:
    from .post_stage_b_tokenization import (
        BANGLABERT_MODEL,
        BANGLABERT_NORMALIZER_COMMIT,
        BANGLABERT_NORMALIZER_REQUIREMENT,
    )
except ImportError:
    _tok_path = Path(__file__).with_name("post_stage_b_tokenization.py")
    _tok_ns: dict[str, Any] = {}
    exec(_tok_path.read_text(encoding="utf-8"), _tok_ns)
    BANGLABERT_MODEL = _tok_ns["BANGLABERT_MODEL"]
    BANGLABERT_NORMALIZER_COMMIT = _tok_ns["BANGLABERT_NORMALIZER_COMMIT"]
    BANGLABERT_NORMALIZER_REQUIREMENT = _tok_ns["BANGLABERT_NORMALIZER_REQUIREMENT"]

MIN_TORCH_VERSION = "2.6"

STAGE4_BASE_PACKAGES: tuple[str, ...] = (
    "transformers>=4.57.1,<5",
    "datasets>=2.20,<5",
    "accelerate>=0.33,<2",
    "safetensors>=0.4",
    "sentencepiece>=0.2,<1",
    "packaging>=23.0",
    "huggingface_hub>=0.24,<2",
    "protobuf>=3.20",
    "numpy>=1.26",
    "pandas>=2.1",
    "scipy>=1.11",
    "scikit-learn>=1.4",
    "pyarrow>=15.0",
)

STAGE4_CORE_IMPORTS: tuple[str, ...] = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "safetensors",
    "sentencepiece",
    "packaging",
    "huggingface_hub",
    "google.protobuf",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "pyarrow",
)

# Documented indirect runtime dependencies pulled by csebuetnlp/normalizer during Colab install
BANGLABERT_NORMALIZER_TRANSITIVE_DEPS: tuple[str, ...] = (
    "unicodedata2",
    "ftfy",
    "emoji",
    "urllib3",
)


def required_packages_for_models(models: Iterable[str]) -> list[str]:
    """Return pinned/bounded pip requirement specs required for the configured backbones."""
    specs = list(STAGE4_BASE_PACKAGES)
    if BANGLABERT_MODEL in set(models):
        specs.append(BANGLABERT_NORMALIZER_REQUIREMENT)
    return specs


def required_imports_for_models(models: Iterable[str]) -> list[str]:
    """Return top-level Python module names that must import cleanly before training starts."""
    mods = list(STAGE4_CORE_IMPORTS)
    if BANGLABERT_MODEL in set(models):
        mods.append("normalizer")
    return mods


def check_requirement_status(
    spec: str,
    *,
    dist_lookup: Callable[[str], Any] = distribution,
) -> dict[str, Any]:
    """Verify whether a PEP 508 or pinned VCS URL requirement is satisfied by an installed distribution."""
    req = Requirement(spec)
    try:
        dist = dist_lookup(req.name)
    except PackageNotFoundError:
        return {
            "spec": spec,
            "name": req.name,
            "installed_version": None,
            "satisfied": False,
            "reason": "missing",
        }

    installed_ver = getattr(dist, "version", "0.0.0")
    if req.url:
        raw_origin = dist.read_text("direct_url.json") or "{}"
        origin = json.loads(raw_origin)
        actual_url = origin.get("url")
        actual_commit = (origin.get("vcs_info") or {}).get("commit_id")
        ok = (
            actual_url == "https://github.com/csebuetnlp/normalizer"
            and actual_commit == BANGLABERT_NORMALIZER_COMMIT
        )
        return {
            "spec": spec,
            "name": req.name,
            "installed_version": installed_ver,
            "direct_url": actual_url,
            "commit_id": actual_commit,
            "satisfied": bool(ok),
            "reason": "ok" if ok else f"vcs_mismatch(url={actual_url!r}, commit={actual_commit!r})",
        }

    try:
        ok = Version(str(installed_ver)) in req.specifier
    except Exception:
        ok = False
    return {
        "spec": spec,
        "name": req.name,
        "installed_version": str(installed_ver),
        "satisfied": bool(ok),
        "reason": "ok" if ok else f"version_incompatible({installed_ver!r} not in {req.specifier!s})",
    }


def preflight_colab_dependencies(
    models: Iterable[str],
    *,
    install_missing: bool = True,
    require_cuda: bool = True,
    dist_lookup: Callable[[str], Any] | None = None,
    version_lookup: Callable[[str], str] | None = None,
    import_module_fn: Callable[[str], Any] | None = None,
    run_pip_fn: Callable[[list[str]], Any] | None = None,
) -> dict[str, Any]:
    """Verify Colab runtime torch/CUDA + Transformers stack; install constrained updates and fail closed if restart needed."""
    model_list = list(models)
    dist_fn = dist_lookup or distribution
    ver_fn = version_lookup or version
    imp_fn = import_module_fn or importlib.import_module
    pip_fn = run_pip_fn or subprocess.check_call

    try:
        torch_ver = str(ver_fn("torch"))
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Select a Google Colab GPU runtime with PyTorch >= 2.6 before running Stage 4."
        ) from exc

    if Version(torch_ver) < Version(MIN_TORCH_VERSION):
        raise RuntimeError(
            f"Incompatible torch=={torch_ver}: Stage 4 requires PyTorch >= {MIN_TORCH_VERSION} for safe Transformers checkpoint loading."
        )

    specs = required_packages_for_models(model_list)
    statuses = [check_requirement_status(spec, dist_lookup=dist_fn) for spec in specs]
    unsatisfied = [st for st in statuses if not st["satisfied"]]

    if unsatisfied:
        summary_bad = ", ".join(f"{st['spec']} [{st['reason']}]" for st in unsatisfied)
        if not install_missing:
            raise RuntimeError(
                f"Fail-closed dependency preflight failure (install_missing=False): {summary_bad}"
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            constraint_path = Path(tmpdir) / "constraints.txt"
            constraint_path.write_text(f"torch=={torch_ver}\n", encoding="utf-8")
            cmd = [sys.executable, "-m", "pip", "install", "-q", "-c", str(constraint_path), *specs]
            pip_fn(cmd)
        raise RuntimeError(
            f"Dependencies installed/updated with torch=={torch_ver} constraint preserved ({summary_bad}). "
            "RESTART REQUIRED: Click 'Runtime -> Restart session' in the SAME Colab GPU runtime, then re-run from Cell 1."
        )

    imports_to_check = required_imports_for_models(model_list)
    imported: dict[str, Any] = {}
    for mod_name in imports_to_check:
        try:
            imported[mod_name] = imp_fn(mod_name)
        except Exception as exc:
            raise RuntimeError(f"Required runtime dependency failed to import cleanly: {mod_name!r} ({exc})") from exc

    torch_mod = imported["torch"]
    cuda_available = bool(torch_mod.cuda.is_available())
    if require_cuda and not cuda_available:
        raise RuntimeError(
            "CUDA GPU is unavailable after dependency verification. Select a GPU runtime (T4/L4/A100) before training."
        )
    if BANGLABERT_MODEL in set(model_list) and not callable(getattr(imported["normalizer"], "normalize", None)):
        raise RuntimeError("`normalizer` imported but does not expose callable `normalize(text)`.")

    installed_versions = {st["name"]: st["installed_version"] for st in statuses}
    installed_versions["torch"] = torch_ver
    return {
        "status": "verified",
        "torch_version": torch_ver,
        "cuda_available": cuda_available,
        "installed_versions": installed_versions,
        "verified_imports": imports_to_check,
        "banglabert_normalizer_commit": BANGLABERT_NORMALIZER_COMMIT if BANGLABERT_MODEL in set(model_list) else None,
    }
