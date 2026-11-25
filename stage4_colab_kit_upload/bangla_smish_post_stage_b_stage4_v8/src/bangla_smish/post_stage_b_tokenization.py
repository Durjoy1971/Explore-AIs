"""Backbone-specific preparation of text views and transformers==4.57.1 tokenizer revision verification.

NOTE: Keep top-level imports stdlib-only because `post_stage_b_colab_deps.py` may `exec()` this file
during Cell 4 preflight before `bangla_smish` is imported as a package.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable

BANGLABERT_MODEL = "csebuetnlp/banglabert"
BANGLABERT_NORMALIZER_COMMIT = "d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9"
BANGLABERT_NORMALIZER_REQUIREMENT = (
    "normalizer @ git+https://github.com/csebuetnlp/normalizer@" + BANGLABERT_NORMALIZER_COMMIT
)

_COMMIT_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
_SNAPSHOT_COMMIT_RE = re.compile(r"(?:^|/)snapshots/([0-9a-f]{40})(?:/|$)")
_TOKENIZER_PATH_KEYS: tuple[str, ...] = (
    "vocab_file",
    "tokenizer_file",
    "merges_file",
    "sp_model_file",
    "sentencepiece_model_file",
    "tokenizer_config_file",
)
_HF_TOKENIZER_PROBE_FILES: tuple[str, ...] = (
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
    "merges.txt",
)


def extract_commit_hash_from_path(path_value: str | Path | None) -> str | None:
    """Extract a 40-hex HuggingFace snapshot commit SHA from a resolved cache path, or None."""
    if path_value is None:
        return None
    posix = Path(str(path_value)).as_posix()
    match = _SNAPSHOT_COMMIT_RE.search(posix)
    if match is None:
        return None
    sha = match.group(1).lower()
    return sha if _COMMIT_SHA40_RE.fullmatch(sha) else None


def resolve_hf_cached_tokenizer_files(
    model_name: str,
    revision: str,
    *,
    cache_dir: str | Path | None = None,
) -> list[str]:
    """Locate local cached HuggingFace snapshot tokenizer files for (model_name, revision) without network calls."""
    resolved: list[str] = []
    seen: set[str] = set()

    def _record(candidate: Any) -> None:
        if isinstance(candidate, (str, Path)):
            s = str(candidate)
            if s and s not in seen and os.path.isfile(s):
                seen.add(s)
                resolved.append(s)

    # 1. Prefer transformers.utils.hub.cached_file (local_files_only=True) when transformers is available.
    try:
        from transformers.utils.hub import cached_file as hf_cached_file

        for fname in _HF_TOKENIZER_PROBE_FILES:
            try:
                p = hf_cached_file(
                    model_name,
                    fname,
                    revision=revision,
                    cache_dir=str(cache_dir) if cache_dir is not None else None,
                    local_files_only=True,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                )
            except Exception:
                continue
            _record(p)
    except ImportError:
        pass

    # 2. Also query huggingface_hub.try_to_load_from_cache (works even without transformers imported).
    if not resolved:
        try:
            from huggingface_hub import try_to_load_from_cache

            for fname in _HF_TOKENIZER_PROBE_FILES:
                try:
                    p = try_to_load_from_cache(
                        repo_id=model_name,
                        filename=fname,
                        revision=revision,
                        cache_dir=cache_dir,
                    )
                except Exception:
                    continue
                _record(p)
        except ImportError:
            pass

    return resolved


def resolve_and_verify_tokenizer_revision(
    tokenizer: Any,
    *,
    model_name: str,
    expected_revision: str,
    allow_hf_cache_lookup: bool = True,
    cache_dir: str | Path | None = None,
    cache_resolver: Callable[[str, str], Any] | None = None,
) -> str:
    """Verify the actual resolved 40-hex commit SHA of a loaded tokenizer fail-closed.

    Compatible with `transformers==4.57.1`, where `PreTrainedTokenizerBase.from_pretrained` pops
    `_commit_hash` and never stores it on `tokenizer.init_kwargs` or `tokenizer._commit_hash`.
    Inspects:
      1. Any explicit `_commit_hash` on `tokenizer.init_kwargs` or `tokenizer._commit_hash` (if present).
      2. Resolved tokenizer file paths (`vocab_file`, `tokenizer_file`, etc.) and `name_or_path` on the
         loaded tokenizer object (failing closed immediately on non-snapshot local files or repo mismatches).
      3. Cached HuggingFace snapshot files for `(model_name, expected_revision)` when `allow_hf_cache_lookup=True`
         or `cache_resolver` is supplied.
    Never falls back to `expected_revision` without verified commit evidence.
    """
    expected = str(expected_revision or "").strip().lower()
    if not _COMMIT_SHA40_RE.fullmatch(expected):
        raise RuntimeError(
            f"Fail closed: expected_revision for {model_name!r} must be a 40-hex commit SHA, got {expected_revision!r}."
        )

    observed_commits: set[str] = set()
    raw_init_kwargs = getattr(tokenizer, "init_kwargs", None)
    init_kwargs: dict[str, Any] = raw_init_kwargs if isinstance(raw_init_kwargs, dict) else {}

    # 1. Explicit _commit_hash attribute or init_kwargs entry (if exposed by any subclass/version).
    for explicit_val in (init_kwargs.get("_commit_hash"), getattr(tokenizer, "_commit_hash", None)):
        if explicit_val is not None and str(explicit_val).strip():
            sha = str(explicit_val).strip().lower()
            if not _COMMIT_SHA40_RE.fullmatch(sha):
                raise RuntimeError(
                    f"Fail closed: tokenizer for {model_name!r} exposed malformed _commit_hash={explicit_val!r}."
                )
            observed_commits.add(sha)

    # 2. Resolved file paths stored on tokenizer.init_kwargs or direct tokenizer attributes.
    candidate_paths: list[tuple[str, str]] = []
    for key in _TOKENIZER_PATH_KEYS:
        val_kw = init_kwargs.get(key)
        if isinstance(val_kw, (str, Path)) and str(val_kw).strip():
            candidate_paths.append((f"init_kwargs[{key!r}]", str(val_kw).strip()))
        val_attr = getattr(tokenizer, key, None)
        if isinstance(val_attr, (str, Path)) and str(val_attr).strip():
            candidate_paths.append((key, str(val_attr).strip()))

    for source_key, path_str in candidate_paths:
        sha = extract_commit_hash_from_path(path_str)
        if sha is None:
            raise RuntimeError(
                f"Fail closed: loaded tokenizer for {model_name!r} resolved {source_key}={path_str!r} "
                f"outside a verified HuggingFace snapshots/<40-hex> commit directory (expected {expected!r})."
            )
        observed_commits.add(sha)

    # 3. Check name_or_path for unpinned local directory redirections.
    raw_nop = getattr(tokenizer, "name_or_path", None)
    if not raw_nop:
        raw_nop = init_kwargs.get("name_or_path")
    if isinstance(raw_nop, (str, Path)) and str(raw_nop).strip():
        nop_str = str(raw_nop).strip()
        nop_sha = extract_commit_hash_from_path(nop_str)
        if nop_sha is not None:
            observed_commits.add(nop_sha)
        elif nop_str != model_name:
            raise RuntimeError(
                f"Fail closed: loaded tokenizer name_or_path={nop_str!r} does not match requested "
                f"repository {model_name!r} or a pinned snapshot directory for {expected!r}."
            )

    # 4. Verify cached HuggingFace snapshot files when loading via HF cache / explicit cache_resolver.
    if cache_resolver is not None or allow_hf_cache_lookup:
        if cache_resolver is not None:
            raw_cached = cache_resolver(model_name, expected)
            if raw_cached is None:
                cached_paths: list[str] = []
            elif isinstance(raw_cached, (str, Path)):
                cached_paths = [str(raw_cached)]
            else:
                cached_paths = [str(p) for p in raw_cached if p is not None and str(p).strip()]
        else:
            cached_paths = resolve_hf_cached_tokenizer_files(model_name, expected, cache_dir=cache_dir)

        for cpath in cached_paths:
            csha = extract_commit_hash_from_path(cpath)
            if csha is None:
                raise RuntimeError(
                    f"Fail closed: cached tokenizer file {cpath!r} for {model_name!r} does not contain "
                    f"a verified 40-hex HuggingFace snapshot commit SHA (expected {expected!r})."
                )
            observed_commits.add(csha)

    if not observed_commits:
        raise RuntimeError(
            f"Fail closed: unable to establish resolved commit SHA for tokenizer {model_name!r} "
            f"(expected pinned revision {expected!r}; tokenizer exposed no snapshot paths or _commit_hash "
            f"and no verified HuggingFace cache snapshot was resolved)."
        )
    if observed_commits != {expected}:
        raise RuntimeError(
            f"Fail closed: loaded tokenizer commit(s) {sorted(observed_commits)!r} != "
            f"pinned revision {expected!r} for {model_name!r}."
        )
    return expected


def prepare_tokenizer_texts(texts: list[str], model_name: str) -> list[str]:
    """Apply BanglaBERT's published normalization to the already selected view."""
    if model_name != BANGLABERT_MODEL:
        return texts
    try:
        from normalizer import normalize
    except ImportError as exc:
        raise RuntimeError(
            f"BanglaBERT requires the authors' text normalizer. Install {BANGLABERT_NORMALIZER_REQUIREMENT!r}."
        ) from exc
    return [normalize(text) for text in texts]
