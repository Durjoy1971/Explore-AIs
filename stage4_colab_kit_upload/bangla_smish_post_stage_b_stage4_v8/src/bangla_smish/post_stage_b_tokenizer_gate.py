"""Deterministic pre-training validation gate for the 32 documented Stage 3 candidate pairs on allowed views."""

from __future__ import annotations

import errno
import importlib
from importlib import metadata as importlib_metadata
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .post_stage_b_data import TEXT_VIEW_COLUMNS, sha256_file
from .post_stage_b_tokenization import (
    BANGLABERT_MODEL,
    BANGLABERT_NORMALIZER_COMMIT,
    prepare_tokenizer_texts,
    resolve_and_verify_tokenizer_revision,
)

ALLOWED_COLLISION_GATE_VIEWS: tuple[str, ...] = ("raw", "canon_masked")
TOKENIZER_GATE_PACKAGE_DISTS: tuple[tuple[str, str], ...] = (
    ("transformers", "transformers"),
    ("tokenizers", "tokenizers"),
    ("sentencepiece", "sentencepiece"),
    ("huggingface_hub", "huggingface_hub"),
)

PINNED_TOKENIZER_REVISIONS: dict[str, str] = {
    "csebuetnlp/banglabert": "9ce791f330578f50da6bc52b54205166fb5d1c8c",
    "google/muril-base-cased": "afd9f36c7923d54e97903922ff1b260d091d202f",
    "xlm-roberta-base": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
    "distilbert-base-multilingual-cased": "45c032ab32cc946ad88a166f7cb282f58c753c2e",
}

# Exact 32 documented Stage 3 candidate cross-boundary pairs (reports/post_stage_b_stage3_readiness_review.md#L38-L69).
# Never invent replacement pairs if any row is missing.
STAGE3_32_CANDIDATE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("P01", "cur_2ee2c999bea1", "cur_cfa2f9f81a31"),
    ("P02", "cur_e2a702578e9e", "cur_40dd2f53ca2f"),
    ("P03", "cur_31f35c6da04c", "cur_647968080b19"),
    ("P04", "cur_0667dc3a1230", "cur_83d1d273eece"),
    ("P05", "cur_356bfb9efc77", "cur_0aabd655f8b8"),
    ("P06", "cur_356bfb9efc77", "cur_4ea4417160ed"),
    ("P07", "cur_40845fd1114b", "cur_902a0a1a8ac5"),
    ("P08", "cur_472b22039982", "cur_7c14e8bae65c"),
    ("P09", "cur_47f292401da6", "cur_7936ee1f5382"),
    ("P10", "cur_b3194c2c485a", "cur_c00c6e3fa781"),
    ("P11", "cur_292fdff7d278", "cur_c24e8fee19a2"),
    ("P12", "cur_63f82fd91d7a", "cur_da994897d0c3"),
    ("P13", "cur_be7f24b65438", "cur_c9018d376c83"),
    ("P14", "cur_0fcaebdd6fd0", "cur_3ecfbe652834"),
    ("P15", "cur_1d753e1cee76", "cur_309b6431fb13"),
    ("P16", "cur_1d753e1cee76", "cur_eb7d06582b65"),
    ("P17", "cur_309b6431fb13", "cur_eb7d06582b65"),
    ("P18", "cur_2861b4182132", "cur_e20fbf89b529"),
    ("P19", "cur_2861b4182132", "cur_ede29997f5d4"),
    ("P20", "cur_e20fbf89b529", "cur_ede29997f5d4"),
    ("P21", "cur_42407c54319e", "cur_5b4b458c215c"),
    ("P22", "cur_42407c54319e", "cur_4e0fc3118321"),
    ("P23", "cur_5b4b458c215c", "cur_4e0fc3118321"),
    ("P24", "cur_b70e4081fe21", "cur_e5c24566c9d3"),
    ("P25", "cur_b70e4081fe21", "cur_b6041ad053cb"),
    ("P26", "cur_e5c24566c9d3", "cur_b6041ad053cb"),
    ("P27", "cur_3a8ee05d8c8d", "cur_f0af08bf6ed0"),
    ("P28", "cur_3a8ee05d8c8d", "cur_f7cfb2326b54"),
    ("P29", "cur_7b913c5a7646", "cur_f481de255c11"),
    ("P30", "cur_54ccc4d7656f", "cur_3e1b3d2c7bee"),
    ("P31", "cur_e9011c46d295", "cur_9b7b07f54ebb"),
    ("P32", "cur_e4b6ade2454a", "cur_a8aaa428cc24"),
)


def _default_hf_tokenizer_loader(model_name: str, revision: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Allowed-view tokenizer collision gate failed closed: `transformers` is unavailable in this environment."
        ) from exc
    return AutoTokenizer.from_pretrained(model_name, revision=revision)


def load_documented_32_pairs(bundle_dir: str | Path) -> dict[str, dict[str, str]]:
    """Resolve all 32 Stage 3 candidate pairs read-only from the frozen v2 bundle without inventing replacements."""
    bdir = Path(bundle_dir).resolve()
    parquets = (
        "real_train_pool.parquet",
        "synthetic_pilot_300.parquet",
        "synthetic_batches_b1_b7.parquet",
        "eval_smishdetect_validation.parquet",
        "eval_smishdetect_test.parquet",
        "eval_banglaphish_ood.parquet",
    )
    cols_needed = ["row_id", *[TEXT_VIEW_COLUMNS[v] for v in ALLOWED_COLLISION_GATE_VIEWS]]
    needed_ids = {rid for _, a, b in STAGE3_32_CANDIDATE_PAIRS for rid in (a, b)}

    row_map: dict[str, dict[str, str]] = {}
    for fn in parquets:
        p = bdir / fn
        if not p.is_file():
            raise FileNotFoundError(f"Missing required bundle parquet for 32-pair lookup: {p}")
        df = pd.read_parquet(p, columns=cols_needed)
        sub = df[df["row_id"].isin(needed_ids)]
        for rec in sub.to_dict(orient="records"):
            rid = str(rec["row_id"])
            row_map[rid] = {TEXT_VIEW_COLUMNS[v]: str(rec[TEXT_VIEW_COLUMNS[v]]) for v in ALLOWED_COLLISION_GATE_VIEWS}

    missing_pairs = [
        (pid, a, b)
        for pid, a, b in STAGE3_32_CANDIDATE_PAIRS
        if (a not in row_map) or (b not in row_map)
    ]
    if missing_pairs:
        raise RuntimeError(
            f"32-pair inventory incomplete in {bdir} (refusing to invent replacement pairs): {missing_pairs}"
        )
    return row_map


def _resolve_gate_package_version(dist_name: str, import_name: str) -> str:
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


def _collect_tokenizer_gate_package_versions() -> dict[str, str]:
    return {
        dist_name: _resolve_gate_package_version(dist_name, import_name)
        for dist_name, import_name in TOKENIZER_GATE_PACKAGE_DISTS
    }


def _atomic_write_gate_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    ignored = {errno.EINVAL, errno.EOPNOTSUPP, errno.EPERM, errno.EBADF, errno.ENOSYS, getattr(errno, "ENOTSUP", errno.EOPNOTSUPP)}
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, ensure_ascii=False))
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError as exc:
                if exc.errno not in ignored:
                    raise
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def run_allowed_view_tokenizer_collision_gate(
    bundle_dir: str | Path,
    cfg: Any,
    *,
    allowed_views: tuple[str, ...] = ALLOWED_COLLISION_GATE_VIEWS,
    tokenizer_loader: Callable[[str, str], Any] | None = None,
    cache_dir: str | Path | None = None,
    cache_resolver: Callable[[str, str], Any] | None = None,
    export_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare actual tokenizer input_ids across all 32 documented pairs on allowed views (raw, canon_masked)."""
    views = tuple(allowed_views)
    if not views or any(v not in ALLOWED_COLLISION_GATE_VIEWS for v in views):
        raise ValueError(f"Tokenizer collision gate supports only {ALLOWED_COLLISION_GATE_VIEWS}, got {views!r}")

    revisions = getattr(cfg, "tokenizer_revisions", None)
    if not isinstance(revisions, dict):
        raise RuntimeError("Fail closed: cfg.tokenizer_revisions must map every model to a frozen 40-hex commit SHA.")
    for model_name in cfg.models:
        rev = revisions.get(model_name)
        if not rev or not re.fullmatch(r"[0-9a-f]{40}", str(rev)):
            raise RuntimeError(f"Fail closed: missing or invalid pinned 40-hex tokenizer revision for {model_name!r}: {rev!r}")

    bdir = Path(bundle_dir).resolve()
    manifest_path = bdir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing dataset_manifest.json in {bdir}")
    bundle_manifest_sha256 = sha256_file(manifest_path)

    row_map = load_documented_32_pairs(bdir)
    loader = tokenizer_loader or _default_hf_tokenizer_loader
    allow_cache = (tokenizer_loader is None) or (cache_resolver is not None) or (cache_dir is not None)

    pair_results: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    loaded_revisions: dict[str, str] = {}

    for model_name in cfg.models:
        expected_rev = str(revisions[model_name])
        tok = loader(model_name, expected_rev)
        verified_rev = resolve_and_verify_tokenizer_revision(
            tok,
            model_name=model_name,
            expected_revision=expected_rev,
            allow_hf_cache_lookup=allow_cache,
            cache_dir=cache_dir,
            cache_resolver=cache_resolver,
        )
        loaded_revisions[model_name] = verified_rev

        for view in views:
            col = TEXT_VIEW_COLUMNS[view]
            for pid, rid_a, rid_b in STAGE3_32_CANDIDATE_PAIRS:
                txt_a = row_map[rid_a][col]
                txt_b = row_map[rid_b][col]
                prep_a, prep_b = prepare_tokenizer_texts([txt_a, txt_b], model_name)
                enc = tok([prep_a, prep_b], truncation=True, max_length=int(cfg.max_length), add_special_tokens=True)
                ids_a = tuple(int(x) for x in enc["input_ids"][0])
                ids_b = tuple(int(x) for x in enc["input_ids"][1])
                equal = ids_a == ids_b
                entry = {
                    "pair_id": pid,
                    "row_id_a": rid_a,
                    "row_id_b": rid_b,
                    "model_name": model_name,
                    "tokenizer_revision": verified_rev,
                    "view": view,
                    "len_a": len(ids_a),
                    "len_b": len(ids_b),
                    "input_ids_equal": equal,
                }
                pair_results.append(entry)
                if equal:
                    collisions.append(entry)

    report = {
        "schema_version": "stage4_tokenizer_collision_gate_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": cfg.run_name,
        "config_hash": cfg.config_hash(),
        "bundle_manifest_sha256": bundle_manifest_sha256,
        "tokenizer_package_versions": _collect_tokenizer_gate_package_versions(),
        "allowed_views": list(views),
        "tokenizer_revisions": loaded_revisions,
        "banglabert_normalizer_commit": BANGLABERT_NORMALIZER_COMMIT if BANGLABERT_MODEL in set(cfg.models) else None,
        "n_pairs": len(STAGE3_32_CANDIDATE_PAIRS),
        "pair_ids": [p[0] for p in STAGE3_32_CANDIDATE_PAIRS],
        "total_checks": len(pair_results),
        "collision_count": len(collisions),
        "collisions": collisions,
        "passed": len(collisions) == 0,
        "results": pair_results,
    }
    if export_path is not None:
        _atomic_write_gate_json(Path(export_path), report)

    if collisions:
        raise RuntimeError(
            f"Allowed-view tokenizer collision gate FAILED ({len(collisions)}/{len(pair_results)} collisions): "
            f"{[(c['model_name'], c['view'], c['pair_id']) for c in collisions]}"
        )
    return report
