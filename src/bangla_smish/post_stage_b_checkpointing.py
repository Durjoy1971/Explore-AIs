"""Stage 4 Drive-safe local-SSD checkpointing, UUID staging, and 3-phase recovery lifecycle.

Guarantees:
  - Training checkpoints are written to Colab local SSD (never directly on Google Drive).
  - Staging uses immutable UUID-disambiguated paths (`attempt_<n>_<uuid>/checkpoint-<step>`),
    per-file SHA256 manifests, full readback verification, append-only commit events, and
    versioned pointer records.
  - Zero reliance on Google Drive FUSE `fcntl.flock`, `O_EXCL`, `fsync`, or atomic rename guarantees.
  - Operational single-writer lease protocol detects unresolved or conflicting active writers and
    corrupted/interrupted lease records, failing closed by default unless explicit owner-confirmed
    stale-writer takeover is authorized (never treating lease age alone as proof of termination).
  - Recovery follows three ordered phases:
      Phase 1: Read-only 4-state scan (VALID_RECOVERABLE, INCOMPLETE_UPLOAD,
               CORRUPT_OR_UNVERIFIABLE, ALREADY_SUPERSEDED) without deleting evidence.
      Phase 2: Validated restore to local SSD from newest VALID_RECOVERABLE candidate (plus any
               distinct pinned best_model_checkpoint, relocating only the restored local
               `trainer_state.json` reference while keeping Drive staging immutable).
      Phase 3: Verified-only pruning of ALREADY_SUPERSEDED checkpoints after commit/restore.
  - Recovery granularity is strictly epoch-level (never claims mid-epoch step-exact restore).
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import socket
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .post_stage_b_data import sha256_file

STAGE4_CHECKPOINT_MANIFEST = "stage4_checkpoint_manifest.json"
LEGACY_RUNNER_CHECKPOINT_COMMIT = "post_stage_b_checkpoint_commit.json"
STAGE4_TIMING_METADATA = "stage4_timing.json"
RECOVERY_GRANULARITY = "epoch_level"
REQUIRED_CKPT_FILES: tuple[str, ...] = (
    "config.json",
    "trainer_state.json",
    "training_args.bin",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
)
WEIGHT_FILE_CANDIDATES: tuple[str, ...] = ("model.safetensors", "pytorch_model.bin")
_ORPHAN_TMP_NAME_RE = re.compile(r"^\..*\.tmp\.[0-9]+\.[0-9a-f]{8}(?:\.[A-Za-z0-9]+)?$")
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


class CheckpointState(str, Enum):
    VALID_RECOVERABLE = "VALID_RECOVERABLE"
    INCOMPLETE_UPLOAD = "INCOMPLETE_UPLOAD"
    CORRUPT_OR_UNVERIFIABLE = "CORRUPT_OR_UNVERIFIABLE"
    ALREADY_SUPERSEDED = "ALREADY_SUPERSEDED"


class RecoveryError(RuntimeError):
    """Raised fail-closed when staging contains unverifiable/corrupt state or missing best checkpoint."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fsync_fd_best_effort(fd: int) -> None:
    """Best-effort fsync on a file descriptor; tolerates FUSE unsupported errnos (including ENOSYS)."""
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno not in _FUSE_FSYNC_IGNORED_ERRNOS:
            raise


def _fsync_dir_best_effort(directory: Path) -> None:
    """Best-effort POSIX directory fsync; tolerates FUSE mounts cleanly while propagating real I/O errors."""
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


def _atomic_write_text_local(path: Path, text: str) -> None:
    """Atomically write UTF-8 text via PID+UUID temp file, best-effort fsync, os.replace, and dir fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            _fsync_fd_best_effort(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    _fsync_dir_best_effort(path.parent)


def _copy_file_verified_best_effort_fsync(src: Path, dst: Path) -> None:
    """Copy a file to staging/local destination with flush + best-effort fsync."""
    with src.open("rb") as f_in, dst.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
        f_out.flush()
        _fsync_fd_best_effort(f_out.fileno())


def _is_orphan_tmp_filename(name: str) -> bool:
    return bool(_ORPHAN_TMP_NAME_RE.fullmatch(name))


def _pid_is_alive_on_local_host(pid: int) -> bool:
    """Check whether a positive PID is alive on the current POSIX host."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _write_new_immutable_json(path: Path, payload: dict[str, Any], *, embed_digest_key: str | None = None) -> str:
    """Write a brand-new immutable JSON artifact with best-effort fsync and readback verification."""
    path.parent.mkdir(parents=True, exist_ok=True)
    to_write = dict(payload)
    if embed_digest_key is not None:
        canonical = json.dumps(to_write, sort_keys=True, ensure_ascii=False).encode("utf-8")
        to_write[embed_digest_key] = _sha256_bytes(canonical)
    encoded = json.dumps(to_write, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    expected_sha = _sha256_bytes(encoded)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable staging file: {path}")
    with path.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        _fsync_fd_best_effort(handle.fileno())
    _fsync_dir_best_effort(path.parent)
    readback = path.read_bytes()
    if _sha256_bytes(readback) != expected_sha:
        raise IOError(f"Readback verification failed for immutable file: {path}")
    return expected_sha


def _verify_embedded_digest_json(path: Path, digest_key: str) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    obj = json.loads(raw)
    if not isinstance(obj, dict) or digest_key not in obj:
        raise ValueError(f"Missing {digest_key} in {path}")
    recorded = obj[digest_key]
    body = {k: v for k, v in obj.items() if k != digest_key}
    expected = _sha256_bytes(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    if recorded != expected:
        raise ValueError(f"Embedded digest mismatch in {path}")
    return obj


def _collect_local_ckpt_files(ckpt_dir: Path, *, require_scaler: bool = False) -> dict[str, dict[str, Any]]:
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Local checkpoint directory missing: {ckpt_dir}")
    files = {
        p.name: p
        for p in ckpt_dir.iterdir()
        if p.is_file() and p.name != STAGE4_CHECKPOINT_MANIFEST and not _is_orphan_tmp_filename(p.name)
    }
    required = set(REQUIRED_CKPT_FILES)
    if require_scaler:
        required.add("scaler.pt")
    if not required <= files.keys() or not (set(WEIGHT_FILE_CANDIDATES) & files.keys()):
        raise ValueError(f"Incomplete checkpoint files in {ckpt_dir}: present={sorted(files)}")
    if any(p.stat().st_size == 0 for p in files.values()):
        raise ValueError(f"Zero-byte file detected in {ckpt_dir}")
    trainer_state = json.loads(files["trainer_state.json"].read_text(encoding="utf-8"))
    m = re.fullmatch(r"checkpoint-([0-9]+)", ckpt_dir.name)
    if not m or int(trainer_state.get("global_step", -1)) != int(m.group(1)):
        raise ValueError(f"global_step mismatch in {ckpt_dir}")
    return {
        name: {"sha256": sha256_file(p), "bytes": int(p.stat().st_size)}
        for name, p in sorted(files.items())
    }


def inspect_writer_leases(artifacts_dir: Path) -> dict[str, Any]:
    """Read-only inspection of immutable lease, release, and takeover records (audit trail preserved)."""
    leases_dir = artifacts_dir / "leases"
    if not leases_dir.is_dir():
        return {
            "active_leases": [],
            "released_writer_ids": [],
            "taken_over_writer_ids": [],
            "corrupt_records": [],
            "unacknowledged_corrupt_files": [],
            "takeover_records": [],
            "conflicting_active_writers": False,
            "total_lease_records": 0,
        }

    released_ids: set[str] = set()
    taken_over_ids: set[str] = set()
    acked_corrupt_files: set[str] = set()
    corrupt_records: list[dict[str, str]] = []
    takeover_records: list[dict[str, Any]] = []
    lease_docs: list[dict[str, Any]] = []

    for rp in sorted(leases_dir.glob("release_*.json")):
        try:
            obj = _verify_embedded_digest_json(rp, "lease_sha256")
            if obj.get("status") != "released" or not obj.get("writer_id"):
                raise ValueError("Invalid release record fields")
            released_ids.add(str(obj["writer_id"]))
        except Exception as exc:
            corrupt_records.append({"file": rp.name, "kind": "release", "error": str(exc)})

    for tp in sorted(leases_dir.glob("takeover_*.json")):
        try:
            obj = _verify_embedded_digest_json(tp, "lease_sha256")
            if obj.get("record_type") != "stale_lease_takeover" or not obj.get("authorized_by_owner"):
                raise ValueError("Invalid takeover record")
            takeover_records.append(obj)
            for wid in obj.get("superseded_writer_ids", []):
                taken_over_ids.add(str(wid))
            for fname in obj.get("acknowledged_corrupt_files", []):
                acked_corrupt_files.add(str(fname))
        except Exception as exc:
            corrupt_records.append({"file": tp.name, "kind": "takeover", "error": str(exc)})

    for lp in sorted(leases_dir.glob("lease_*.json")):
        try:
            obj = _verify_embedded_digest_json(lp, "lease_sha256")
            if obj.get("status") != "active" or not obj.get("writer_id"):
                raise ValueError("Invalid lease record fields")
            lease_docs.append({**obj, "_file": lp.name})
        except Exception as exc:
            corrupt_records.append({"file": lp.name, "kind": "lease", "error": str(exc)})

    resolved_ids = released_ids | taken_over_ids
    active_leases = [d for d in lease_docs if str(d["writer_id"]) not in resolved_ids]
    unacked_corrupt = [c for c in corrupt_records if c["file"] not in acked_corrupt_files]

    return {
        "active_leases": active_leases,
        "released_writer_ids": sorted(released_ids),
        "taken_over_writer_ids": sorted(taken_over_ids),
        "corrupt_records": corrupt_records,
        "unacknowledged_corrupt_files": unacked_corrupt,
        "takeover_records": takeover_records,
        "conflicting_active_writers": len(active_leases) > 1,
        "total_lease_records": len(lease_docs),
    }


def register_single_writer_lease(
    artifacts_dir: Path,
    *,
    writer_id: str | None = None,
    run_name: str | None = None,
    config_hash: str | None = None,
    allow_stale_takeover: bool = False,
    expected_stale_writer_ids: Iterable[str] | None = None,
    expected_corrupt_files: Iterable[str] | None = None,
    takeover_reason: str | None = None,
    force_takeover: bool = False,
) -> dict[str, Any]:
    """Acquire an immutable single-writer lease without Drive FUSE locking assumptions.

    Never treats lease age alone as proof that a prior writer stopped: unresolved active leases or
    corrupted/interrupted lease records fail closed unless the owner explicitly authorizes
    `allow_stale_takeover=True` with non-empty `expected_stale_writer_ids` (matching all unresolved
    active leases) and a non-empty `takeover_reason` after confirming stop state.
    All superseded leases, takeovers, and corrupt records remain preserved on disk for audit.
    """
    leases_dir = artifacts_dir / "leases"
    leases_dir.mkdir(parents=True, exist_ok=True)
    lease_token = writer_id or f"writer_{uuid.uuid4().hex[:12]}"
    if force_takeover and not (allow_stale_takeover and takeover_reason):
        raise ValueError("Refusing force_takeover without explicit allow_stale_takeover=True and takeover_reason")

    state = inspect_writer_leases(artifacts_dir)
    active_leases = state["active_leases"]
    active_ids = [str(l["writer_id"]) for l in active_leases]
    unacked_corrupt = state["unacknowledged_corrupt_files"]

    if lease_token in active_ids:
        raise RuntimeError(f"Writer ID {lease_token!r} is already active in {artifacts_dir}")

    if (active_leases or unacked_corrupt) and not allow_stale_takeover:
        if unacked_corrupt and not active_leases:
            raise RuntimeError(
                "Interrupted or corrupt lease/release record(s) detected: "
                f"{[c['file'] for c in unacked_corrupt]}. Verify no writer is active and authorize "
                "stale takeover explicitly with expected_corrupt_files and takeover_reason before resuming."
            )
        details = [
            f"{l['writer_id']}(pid={l.get('pid')}, host={l.get('hostname')}, started={l.get('created_at_utc')})"
            for l in active_leases
        ]
        conflict_note = "CONFLICTING ACTIVE WRITERS DETECTED! " if len(active_leases) > 1 else ""
        raise RuntimeError(
            f"Single-writer policy violation: {conflict_note}unresolved active writer lease(s) exist ({details}). "
            "Lease age alone never proves termination; confirm the prior runtime stopped and pass "
            "allow_stale_takeover=True with explicit expected_stale_writer_ids and takeover_reason, or finish cleanly."
        )

    takeover_doc: dict[str, Any] | None = None
    if allow_stale_takeover:
        reason = (takeover_reason or "").strip()
        if not reason:
            raise ValueError("Authorized stale-writer takeover requires a non-empty takeover_reason")
        if active_ids:
            if expected_stale_writer_ids is None:
                raise RuntimeError(
                    "Takeover rejected fail-closed: explicit non-empty expected_stale_writer_ids is required "
                    f"for every stale takeover (unresolved active={active_ids})."
                )
            expected_set = {str(x).strip() for x in expected_stale_writer_ids if str(x).strip()}
            if not expected_set:
                raise RuntimeError(
                    "Takeover rejected fail-closed: expected_stale_writer_ids must contain at least one writer ID."
                )
            if expected_set != set(active_ids):
                raise RuntimeError(
                    "Takeover rejected fail-closed: expected_stale_writer_ids "
                    f"{sorted(expected_set)} does not match actual unresolved active writer(s) {sorted(set(active_ids))}."
                )
            cur_host = socket.gethostname()
            cur_pid = os.getpid()
            for lease_entry in active_leases:
                l_pid = lease_entry.get("pid")
                if (
                    lease_entry.get("hostname") == cur_host
                    and isinstance(l_pid, int)
                    and l_pid != cur_pid
                    and _pid_is_alive_on_local_host(l_pid)
                ):
                    raise RuntimeError(
                        "Takeover rejected fail-closed: active lease writer "
                        f"{lease_entry['writer_id']!r} (pid={l_pid} on host={cur_host!r}) is still alive."
                    )
        elif expected_stale_writer_ids is not None:
            expected_set = {str(x).strip() for x in expected_stale_writer_ids if str(x).strip()}
            if expected_set:
                raise RuntimeError(
                    "Takeover rejected fail-closed: expected_stale_writer_ids "
                    f"{sorted(expected_set)} specified, but no active writer leases exist."
                )
        if unacked_corrupt:
            actual_corrupt_set = {c["file"] for c in unacked_corrupt}
            if expected_corrupt_files is None:
                raise RuntimeError(
                    "Takeover rejected fail-closed: unacknowledged corrupt lease/release record(s) "
                    f"{sorted(actual_corrupt_set)} require explicit expected_corrupt_files confirmation."
                )
            expected_corrupt_set = {str(x).strip() for x in expected_corrupt_files if str(x).strip()}
            if expected_corrupt_set != actual_corrupt_set:
                raise RuntimeError(
                    "Takeover rejected fail-closed: expected_corrupt_files "
                    f"{sorted(expected_corrupt_set)} != actual corrupt records {sorted(actual_corrupt_set)}."
                )
        if active_ids or unacked_corrupt:
            takeover_doc = {
                "record_type": "stale_lease_takeover",
                "authorized_by_owner": True,
                "new_writer_id": lease_token,
                "superseded_writer_ids": active_ids,
                "acknowledged_corrupt_files": [c["file"] for c in unacked_corrupt],
                "takeover_reason": reason,
                "note": "Owner-confirmed takeover; age alone is never treated as proof of process death.",
                "recorded_at_utc": _utc_now(),
            }
            tp = leases_dir / f"takeover_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}.json"
            _write_new_immutable_json(tp, takeover_doc, embed_digest_key="lease_sha256")

    lease_doc = {
        "record_type": "lease_acquired",
        "writer_id": lease_token,
        "status": "active",
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "run_name": run_name,
        "config_hash": config_hash,
        "created_at_utc": _utc_now(),
        "operational_policy": "single_writer_per_run_no_drive_fuse_locking_assumption",
    }
    lease_path = leases_dir / f"lease_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}.json"
    _write_new_immutable_json(lease_path, lease_doc, embed_digest_key="lease_sha256")
    # Post-write TOCTOU verification: detect any concurrent writer that raced in during lease write.
    # On conflict, fail closed WITHOUT releasing our lease so conflict evidence remains visible to all peers.
    verify_writer_lease_still_sole_active(artifacts_dir, lease_token)
    if takeover_doc is not None:
        lease_doc["takeover_record"] = takeover_doc
    return lease_doc


def release_single_writer_lease(
    artifacts_dir: Path,
    writer_id: str,
    *,
    exit_reason: str = "normal_completion",
) -> Path:
    leases_dir = artifacts_dir / "leases"
    release_doc = {
        "record_type": "lease_released",
        "writer_id": writer_id,
        "status": "released",
        "exit_reason": exit_reason,
        "released_at_utc": _utc_now(),
    }
    rel_path = leases_dir / f"release_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}.json"
    _write_new_immutable_json(rel_path, release_doc, embed_digest_key="lease_sha256")
    return rel_path


def verify_writer_lease_still_sole_active(artifacts_dir: Path, writer_id: str) -> None:
    """Best-effort operational guard verifying no conflicting writer or takeover occurred mid-run."""
    state = inspect_writer_leases(artifacts_dir)
    if state["unacknowledged_corrupt_files"]:
        raise RuntimeError(
            "Writer safety check failed: corrupt/interrupted lease record detected mid-run: "
            f"{[c['file'] for c in state['unacknowledged_corrupt_files']]}"
        )
    active_ids = [str(l["writer_id"]) for l in state["active_leases"]]
    others = [wid for wid in active_ids if wid != writer_id]
    if others:
        raise RuntimeError(f"Conflicting active writer detected mid-run: expected only {writer_id!r}, found {active_ids!r}")
    if writer_id not in active_ids:
        raise RuntimeError(f"Current writer lease {writer_id!r} is no longer active (released or superseded)")


def _append_commit_event(events_dir: Path, event_type: str, payload: dict[str, Any]) -> tuple[Path, str]:
    events_dir.mkdir(parents=True, exist_ok=True)
    seq = len(list(events_dir.glob("event_*.json"))) + 1
    doc = {
        "seq": seq,
        "event_id": uuid.uuid4().hex[:12],
        "event_type": event_type,
        "recorded_at_utc": _utc_now(),
        **payload,
    }
    path = events_dir / f"event_{seq:06d}_{event_type}_{doc['event_id']}.json"
    sha = _write_new_immutable_json(path, doc, embed_digest_key="event_sha256")
    return path, sha


def _append_versioned_pointer(pointers_dir: Path, uid: str, payload: dict[str, Any]) -> tuple[Path, str]:
    unit_ptr_dir = pointers_dir / uid
    unit_ptr_dir.mkdir(parents=True, exist_ok=True)
    seq = len(list(unit_ptr_dir.glob("ptr_*.json"))) + 1
    doc = {
        "seq": seq,
        "pointer_id": uuid.uuid4().hex[:12],
        "recorded_at_utc": _utc_now(),
        **payload,
    }
    path = unit_ptr_dir / f"ptr_{seq:06d}_{doc['pointer_id']}.json"
    sha = _write_new_immutable_json(path, doc, embed_digest_key="pointer_sha256")
    return path, sha


def _parse_step_from_ckpt_name(name: str | None) -> int | None:
    if not name:
        return None
    m = re.fullmatch(r"checkpoint-([0-9]+)", Path(name).name)
    return int(m.group(1)) if m else None


def sync_local_checkpoint_to_staging(
    *,
    local_ckpt_dir: Path,
    staging_unit_dir: Path,
    events_dir: Path,
    pointers_dir: Path,
    uid: str,
    attempt: int,
    config_hash: str,
    manifest_sha256: str,
    require_scaler: bool = False,
) -> dict[str, Any]:
    """Copy a local SSD epoch checkpoint to an immutable UUID staging dir with full readback verification."""
    local_files = _collect_local_ckpt_files(local_ckpt_dir, require_scaler=require_scaler)
    trainer_state = json.loads((local_ckpt_dir / "trainer_state.json").read_text(encoding="utf-8"))
    global_step = int(trainer_state["global_step"])
    best_raw = trainer_state.get("best_model_checkpoint")
    best_rel = Path(best_raw).name if best_raw else None
    best_step = _parse_step_from_ckpt_name(best_rel)

    attempt_uuid = f"attempt_{attempt:03d}_{uuid.uuid4().hex[:8]}"
    staged_ckpt = staging_unit_dir / attempt_uuid / local_ckpt_dir.name
    staged_ckpt.mkdir(parents=True, exist_ok=False)

    for fname, meta in local_files.items():
        dst = staged_ckpt / fname
        _copy_file_verified_best_effort_fsync(local_ckpt_dir / fname, dst)
        if sha256_file(dst) != meta["sha256"] or int(dst.stat().st_size) != meta["bytes"]:
            raise IOError(f"Staging readback SHA256/size mismatch for {dst}")
    _fsync_dir_best_effort(staged_ckpt)

    manifest_doc = {
        "schema_version": "stage4_ckpt_manifest_v1",
        "recovery_granularity": RECOVERY_GRANULARITY,
        "unit_id": uid,
        "attempt": int(attempt),
        "attempt_uuid": attempt_uuid,
        "global_step": global_step,
        "config_hash": config_hash,
        "dataset_manifest_sha256": manifest_sha256,
        "require_scaler": bool(require_scaler),
        "best_model_checkpoint_rel": best_rel,
        "best_model_checkpoint_step": best_step,
        "files": local_files,
        "staged_at_utc": _utc_now(),
    }
    manifest_sha = _write_new_immutable_json(
        staged_ckpt / STAGE4_CHECKPOINT_MANIFEST, manifest_doc, embed_digest_key="manifest_sha256"
    )
    rel_path = staged_ckpt.relative_to(staging_unit_dir).as_posix()
    event_path, event_sha = _append_commit_event(
        events_dir,
        "CHECKPOINT_SYNCED",
        {
            "unit_id": uid,
            "attempt": int(attempt),
            "global_step": global_step,
            "staging_rel_path": rel_path,
            "checkpoint_manifest_sha256": manifest_sha,
            "config_hash": config_hash,
            "dataset_manifest_sha256": manifest_sha256,
        },
    )
    ptr_path, ptr_sha = _append_versioned_pointer(
        pointers_dir,
        uid,
        {
            "unit_id": uid,
            "attempt": int(attempt),
            "global_step": global_step,
            "staging_rel_path": rel_path,
            "checkpoint_manifest_sha256": manifest_sha,
            "event_sha256": event_sha,
        },
    )
    return {
        "staged_ckpt_dir": staged_ckpt,
        "staging_rel_path": rel_path,
        "global_step": global_step,
        "checkpoint_manifest_sha256": manifest_sha,
        "event_path": event_path,
        "pointer_path": ptr_path,
        "pointer_sha256": ptr_sha,
    }


def _read_unit_commit_events(
    events_dir: Path,
    uid: str,
    *,
    active_attempt: int | None = None,
    trust_commit_events: bool = True,
) -> tuple[bool, int | None, dict[str, set[str]], list[str]]:
    """Scan append-only commit events tolerating interrupted tail writes cleanly.

    When `active_attempt` is supplied, only `UNIT_COMMITTED` events with `attempt >= active_attempt`
    mark the unit committed, preventing historical invalidated Attempt 1 events from deleting Attempt 2+
    checkpoints. Returns `(unit_committed, latest_synced_step, synced_manifests_by_rel_path, warnings)`.
    """
    if not events_dir.is_dir():
        return False, None, {}, []
    warnings: list[str] = []
    unit_committed = False
    latest_synced_step: int | None = None
    synced_manifests: dict[str, set[str]] = {}
    for ep in sorted(events_dir.glob("event_*.json")):
        try:
            ev = _verify_embedded_digest_json(ep, "event_sha256")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"Corrupt/interrupted event ignored during scan ({ep.name}): {exc}")
            continue
        if ev.get("unit_id") != uid:
            continue
        if ev.get("event_type") == "UNIT_COMMITTED":
            if trust_commit_events:
                ev_attempt = ev.get("attempt")
                if active_attempt is None or (isinstance(ev_attempt, int) and ev_attempt >= int(active_attempt)):
                    unit_committed = True
        elif ev.get("event_type") == "CHECKPOINT_SYNCED":
            step = int(ev.get("global_step", -1))
            latest_synced_step = step if latest_synced_step is None else max(latest_synced_step, step)
            rel_p = ev.get("staging_rel_path")
            m_sha = ev.get("checkpoint_manifest_sha256")
            if isinstance(rel_p, str) and isinstance(m_sha, str):
                synced_manifests.setdefault(rel_p, set()).add(m_sha)
    return unit_committed, latest_synced_step, synced_manifests, warnings


def _inspect_pointers(pointers_dir: Path, uid: str) -> list[str]:
    unit_ptr_dir = pointers_dir / uid
    if not unit_ptr_dir.is_dir():
        return []
    warnings: list[str] = []
    for pp in sorted(unit_ptr_dir.glob("ptr_*.json")):
        try:
            _verify_embedded_digest_json(pp, "pointer_sha256")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"Corrupt/interrupted pointer ignored during scan ({pp.name}): {exc}")
    return warnings


def scan_unit_staging(
    *,
    staging_unit_dir: Path,
    events_dir: Path,
    pointers_dir: Path,
    uid: str,
    expected_config_hash: str,
    expected_manifest_sha256: str,
    unit_already_committed: bool = False,
    checkpoint_retention_limit: int = 1,
    active_attempt: int | None = None,
    trust_commit_events: bool = True,
) -> dict[str, Any]:
    """Phase 1: Read-only 4-state scan classifying all staged checkpoints without deleting anything."""
    ev_committed, _, synced_manifests, ev_warnings = _read_unit_commit_events(
        events_dir,
        uid,
        active_attempt=active_attempt,
        trust_commit_events=trust_commit_events,
    )
    ptr_warnings = _inspect_pointers(pointers_dir, uid)
    committed = bool(unit_already_committed or ev_committed)

    candidates: list[dict[str, Any]] = []
    if staging_unit_dir.is_dir():
        for attempt_dir in sorted(p for p in staging_unit_dir.iterdir() if p.is_dir()):
            for ckpt_dir in sorted(p for p in attempt_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")):
                rel_path = ckpt_dir.relative_to(staging_unit_dir).as_posix()
                manifest_file = ckpt_dir / STAGE4_CHECKPOINT_MANIFEST
                if not manifest_file.is_file():
                    candidates.append(
                        {
                            "path": ckpt_dir,
                            "rel_path": rel_path,
                            "state": CheckpointState.INCOMPLETE_UPLOAD,
                            "reason": "Missing stage4_checkpoint_manifest.json (interrupted upload).",
                            "global_step": -1,
                            "attempt": -1,
                        }
                    )
                    continue
                try:
                    manifest = _verify_embedded_digest_json(manifest_file, "manifest_sha256")
                    manifest_file_sha = sha256_file(manifest_file)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    candidates.append(
                        {
                            "path": ckpt_dir,
                            "rel_path": rel_path,
                            "state": CheckpointState.CORRUPT_OR_UNVERIFIABLE,
                            "reason": f"Manifest unverifiable or truncated: {exc}",
                            "global_step": -1,
                            "attempt": -1,
                        }
                    )
                    continue

                if (
                    manifest.get("unit_id") != uid
                    or manifest.get("config_hash") != expected_config_hash
                    or manifest.get("dataset_manifest_sha256") != expected_manifest_sha256
                ):
                    candidates.append(
                        {
                            "path": ckpt_dir,
                            "rel_path": rel_path,
                            "state": CheckpointState.CORRUPT_OR_UNVERIFIABLE,
                            "reason": "Stale or mismatched run config_hash / manifest_sha256 / unit_id.",
                            "global_step": int(manifest.get("global_step", -1)),
                            "attempt": int(manifest.get("attempt", -1)),
                        }
                    )
                    continue

                req_files = set(REQUIRED_CKPT_FILES)
                if manifest.get("require_scaler"):
                    req_files.add("scaler.pt")
                files_meta = manifest.get("files", {})
                if (
                    not isinstance(files_meta, dict)
                    or not req_files <= set(files_meta)
                    or not (set(WEIGHT_FILE_CANDIDATES) & set(files_meta))
                ):
                    candidates.append(
                        {
                            "path": ckpt_dir,
                            "rel_path": rel_path,
                            "state": CheckpointState.CORRUPT_OR_UNVERIFIABLE,
                            "reason": "Manifest lacks required checkpoint or weight entries.",
                            "global_step": int(manifest.get("global_step", -1)),
                            "attempt": int(manifest.get("attempt", -1)),
                        }
                    )
                    continue

                has_synced_event = manifest_file_sha in synced_manifests.get(rel_path, set())
                corrupt_reason: str | None = None
                incomplete_reason: str | None = None
                for fname, fmeta in files_meta.items():
                    fpath = ckpt_dir / fname
                    expected_bytes = int(fmeta["bytes"])
                    if not fpath.is_file():
                        if not has_synced_event:
                            incomplete_reason = (
                                f"Out-of-order async upload: manifest present without CHECKPOINT_SYNCED "
                                f"event and payload file missing ({fname})."
                            )
                            break
                        corrupt_reason = f"Missing staged file: {fname}"
                        break
                    actual_bytes = int(fpath.stat().st_size)
                    if actual_bytes < expected_bytes and not has_synced_event:
                        incomplete_reason = (
                            f"Out-of-order async upload: partial file {fname} ({actual_bytes}/{expected_bytes} bytes) "
                            "without CHECKPOINT_SYNCED event."
                        )
                        break
                    if actual_bytes != expected_bytes or sha256_file(fpath) != fmeta["sha256"]:
                        corrupt_reason = f"SHA256/byte mismatch on staged file: {fname}"
                        break
                if incomplete_reason is not None:
                    candidates.append(
                        {
                            "path": ckpt_dir,
                            "rel_path": rel_path,
                            "state": CheckpointState.INCOMPLETE_UPLOAD,
                            "reason": incomplete_reason,
                            "global_step": int(manifest.get("global_step", -1)),
                            "attempt": int(manifest.get("attempt", -1)),
                        }
                    )
                    continue
                if corrupt_reason is not None:
                    candidates.append(
                        {
                            "path": ckpt_dir,
                            "rel_path": rel_path,
                            "state": CheckpointState.CORRUPT_OR_UNVERIFIABLE,
                            "reason": corrupt_reason,
                            "global_step": int(manifest.get("global_step", -1)),
                            "attempt": int(manifest.get("attempt", -1)),
                        }
                    )
                    continue

                candidates.append(
                    {
                        "path": ckpt_dir,
                        "rel_path": rel_path,
                        "state": CheckpointState.VALID_RECOVERABLE,
                        "reason": "All manifest and payload SHA256 digests verified.",
                        "global_step": int(manifest["global_step"]),
                        "attempt": int(manifest["attempt"]),
                        "has_synced_event": has_synced_event,
                        "manifest": manifest,
                    }
                )

    valid_list = sorted(
        [c for c in candidates if c["state"] == CheckpointState.VALID_RECOVERABLE],
        key=lambda c: (c["global_step"], c["attempt"], c["rel_path"]),
        reverse=True,
    )
    keep_n = max(1, int(checkpoint_retention_limit))
    pinned_best_rel: str | None = None
    pinned_best_entry: dict[str, Any] | None = None
    missing_required_best = False

    if committed:
        for c in valid_list:
            c["state"] = CheckpointState.ALREADY_SUPERSEDED
            c["reason"] = "Unit already committed; training checkpoint superseded."
    elif valid_list:
        latest = valid_list[0]
        latest_fully_committed_to_staging = bool(latest.get("has_synced_event", False))
        pinned_best_rel = latest["manifest"].get("best_model_checkpoint_rel")
        if pinned_best_rel:
            if latest["path"].name == pinned_best_rel:
                pinned_best_entry = latest
            else:
                matches = [older for older in valid_list[1:] if older["path"].name == pinned_best_rel]
                if matches:
                    pinned_best_entry = matches[0]
                else:
                    missing_required_best = True
        for idx, entry in enumerate(valid_list[1:], start=1):
            is_pinned = pinned_best_entry is not None and entry["rel_path"] == pinned_best_entry["rel_path"]
            within_retention = idx < keep_n
            if is_pinned:
                entry["reason"] = (
                    f"Pinned as best_model_checkpoint by latest checkpoint step {latest['global_step']}."
                )
            elif not latest_fully_committed_to_staging:
                entry["reason"] = (
                    f"Preserved because newer checkpoint step {latest['global_step']} lacks verified CHECKPOINT_SYNCED event."
                )
            elif within_retention:
                entry["reason"] = (
                    f"Retained within checkpoint_retention_limit={keep_n} (rank {idx + 1})."
                )
            else:
                entry["state"] = CheckpointState.ALREADY_SUPERSEDED
                entry["reason"] = (
                    f"Superseded by newer verified checkpoint step {latest['global_step']} "
                    f"(retention_limit={keep_n})."
                )

    return {
        "unit_id": uid,
        "unit_committed": committed,
        "expected_config_hash": expected_config_hash,
        "expected_manifest_sha256": expected_manifest_sha256,
        "checkpoint_retention_limit": keep_n,
        "warnings": ev_warnings + ptr_warnings,
        "entries": candidates,
        "counts_by_state": {
            st.value: sum(1 for c in candidates if c["state"] == st)
            for st in CheckpointState
        },
        "latest_valid_rel_path": None if (committed or not valid_list) else valid_list[0]["rel_path"],
        "pinned_best_checkpoint_name": pinned_best_rel,
        "pinned_best_rel_path": None if (committed or pinned_best_entry is None) else pinned_best_entry["rel_path"],
        "missing_required_best_checkpoint": missing_required_best,
    }


def _refresh_local_ckpt_commit_file(ckpt_dir: Path) -> None:
    """Atomically refresh local post_stage_b_checkpoint_commit.json in the restored local SSD directory."""
    hashes = {
        p.name: sha256_file(p)
        for p in sorted(ckpt_dir.iterdir())
        if (
            p.is_file()
            and p.name not in {LEGACY_RUNNER_CHECKPOINT_COMMIT, STAGE4_CHECKPOINT_MANIFEST}
            and not _is_orphan_tmp_filename(p.name)
        )
    }
    _atomic_write_text_local(
        ckpt_dir / LEGACY_RUNNER_CHECKPOINT_COMMIT,
        json.dumps(hashes, indent=2, sort_keys=True),
    )


def _restore_single_ckpt_to_local(item: dict[str, Any], local_restore_root: Path) -> Path:
    src_dir: Path = item["path"]
    manifest = item["manifest"]
    dst_dir = local_restore_root / src_dir.name
    if dst_dir.exists():
        dst_dir = local_restore_root / f"restored_{uuid.uuid4().hex[:6]}_{src_dir.name}"
    dst_dir.mkdir(parents=True, exist_ok=False)
    try:
        for fname, fmeta in manifest["files"].items():
            dst_file = dst_dir / fname
            _copy_file_verified_best_effort_fsync(src_dir / fname, dst_file)
            if sha256_file(dst_file) != fmeta["sha256"] or int(dst_file.stat().st_size) != int(fmeta["bytes"]):
                raise RecoveryError(f"Local restore readback failed for {dst_file}; preserving staging evidence.")
        _fsync_dir_best_effort(dst_dir)
        _refresh_local_ckpt_commit_file(dst_dir)
    except BaseException:
        shutil.rmtree(dst_dir, ignore_errors=True)
        raise
    return dst_dir


def recover_latest_valid_checkpoint(
    *,
    scan_report: dict[str, Any],
    local_restore_root: Path,
    fail_on_corrupt_without_valid: bool = True,
) -> dict[str, Any]:
    """Phase 2: Restore newest VALID_RECOVERABLE checkpoint + any distinct best checkpoint onto local SSD.

    Relocates `best_model_checkpoint` strictly inside the restored local copy of `trainer_state.json`,
    never modifying the immutable source checkpoint on Drive staging. Fails closed if a distinct
    best checkpoint is required but missing/corrupt.
    """
    entries = scan_report["entries"]
    valid = sorted(
        [e for e in entries if e["state"] == CheckpointState.VALID_RECOVERABLE],
        key=lambda c: (c["global_step"], c["attempt"], c["rel_path"]),
        reverse=True,
    )
    corrupt = [e for e in entries if e["state"] == CheckpointState.CORRUPT_OR_UNVERIFIABLE]
    if not valid:
        if fail_on_corrupt_without_valid and corrupt:
            raise RecoveryError(
                f"Fail-closed recovery halt for {scan_report['unit_id']}: corrupt/unverifiable staging "
                f"with no valid fallback ({[c['reason'] for c in corrupt]}). Preserving all evidence."
            )
        return {
            "recovered": False,
            "restored_local_ckpt": None,
            "restored_best_local_ckpt": None,
            "best_model_checkpoint_relocated": False,
            "recovery_granularity": RECOVERY_GRANULARITY,
            "scan_report": scan_report,
        }

    chosen = valid[0]
    chosen_manifest = chosen["manifest"]
    expected_uid = scan_report.get("unit_id")
    expected_cfg = scan_report.get("expected_config_hash")
    expected_man = scan_report.get("expected_manifest_sha256")
    pinned_rel = chosen_manifest.get("best_model_checkpoint_rel")

    best_source_item: dict[str, Any] | None = None
    if pinned_rel:
        if pinned_rel == chosen["path"].name:
            best_source_item = chosen
        else:
            matches = [e for e in valid[1:] if e["path"].name == pinned_rel]
            if not matches or scan_report.get("missing_required_best_checkpoint"):
                raise RecoveryError(
                    f"Fail-closed recovery halt for {expected_uid}: required best_model_checkpoint "
                    f"{pinned_rel!r} referenced by step {chosen['global_step']} is missing or unverifiable."
                )
            best_source_item = matches[0]

    for item in (best_source_item, chosen):
        if item is None:
            continue
        im = item["manifest"]
        if (
            (expected_uid and im.get("unit_id") != expected_uid)
            or (expected_cfg and im.get("config_hash") != expected_cfg)
            or (expected_man and im.get("dataset_manifest_sha256") != expected_man)
        ):
            raise RecoveryError(f"Identity mismatch during Phase 2 restore for {item['rel_path']}")

    local_restore_root.mkdir(parents=True, exist_ok=True)
    restored_paths: list[Path] = []
    restored_best_dir: Path | None = None
    if best_source_item is not None and best_source_item["rel_path"] != chosen["rel_path"]:
        restored_best_dir = _restore_single_ckpt_to_local(best_source_item, local_restore_root)
        restored_paths.append(restored_best_dir)

    restored_latest_dir = _restore_single_ckpt_to_local(chosen, local_restore_root)
    restored_paths.append(restored_latest_dir)
    if best_source_item is not None and best_source_item["rel_path"] == chosen["rel_path"]:
        restored_best_dir = restored_latest_dir

    relocated = False
    if restored_best_dir is not None:
        if not restored_best_dir.is_dir():
            raise RecoveryError(f"Restored best checkpoint directory missing: {restored_best_dir}")
        best_state = json.loads((restored_best_dir / "trainer_state.json").read_text(encoding="utf-8"))
        expected_best_step = chosen_manifest.get("best_model_checkpoint_step")
        if expected_best_step is not None and int(best_state.get("global_step", -1)) != int(expected_best_step):
            raise RecoveryError(
                f"Restored best checkpoint global_step {best_state.get('global_step')} != expected {expected_best_step}"
            )
        local_ts_path = restored_latest_dir / "trainer_state.json"
        ts_doc = json.loads(local_ts_path.read_text(encoding="utf-8"))
        ts_doc["best_model_checkpoint"] = str(restored_best_dir.resolve())
        _atomic_write_text_local(local_ts_path, json.dumps(ts_doc, indent=2, ensure_ascii=False))
        _refresh_local_ckpt_commit_file(restored_latest_dir)
        relocated = True

    return {
        "recovered": True,
        "restored_local_ckpt": restored_latest_dir,
        "restored_best_local_ckpt": restored_best_dir,
        "best_model_checkpoint_relocated": relocated,
        "restored_all_ckpts": restored_paths,
        "global_step": chosen["global_step"],
        "source_staging_rel_path": chosen["rel_path"],
        "recovery_granularity": RECOVERY_GRANULARITY,
        "scan_report": scan_report,
    }


def prune_superseded_staging(
    *,
    scan_report: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Phase 3: Prune ONLY ALREADY_SUPERSEDED staged directories after verified recovery/commit.

    Never deletes VALID_RECOVERABLE, INCOMPLETE_UPLOAD, or CORRUPT_OR_UNVERIFIABLE evidence.
    Records OSError failures per directory cleanly without falsely reporting failed deletions as pruned.
    """
    pruned: list[str] = []
    prune_errors: list[dict[str, str]] = []
    preserved_corrupt: list[str] = []
    preserved_incomplete: list[str] = []
    preserved_valid: list[str] = []
    for entry in scan_report["entries"]:
        st = entry["state"]
        p: Path = entry["path"]
        if st == CheckpointState.ALREADY_SUPERSEDED:
            if dry_run:
                pruned.append(entry["rel_path"])
                continue
            try:
                if p.exists():
                    shutil.rmtree(p)
                    parent = p.parent
                    if parent.is_dir() and not any(parent.iterdir()):
                        parent.rmdir()
                pruned.append(entry["rel_path"])
            except OSError as exc:
                prune_errors.append({"rel_path": entry["rel_path"], "error": str(exc)})
        elif st == CheckpointState.CORRUPT_OR_UNVERIFIABLE:
            preserved_corrupt.append(entry["rel_path"])
        elif st == CheckpointState.INCOMPLETE_UPLOAD:
            preserved_incomplete.append(entry["rel_path"])
        elif st == CheckpointState.VALID_RECOVERABLE:
            preserved_valid.append(entry["rel_path"])

    return {
        "dry_run": bool(dry_run),
        "pruned_superseded": pruned,
        "prune_errors": prune_errors,
        "preserved_valid": preserved_valid,
        "preserved_incomplete": preserved_incomplete,
        "preserved_corrupt_evidence": preserved_corrupt,
    }


def commit_unit_completion_event(
    *,
    events_dir: Path,
    uid: str,
    attempt: int,
    config_hash: str,
    manifest_sha256: str,
    metrics_sha256: str,
    preds_sha256: str,
) -> tuple[Path, str]:
    """Record verified unit commit event before Phase 3 cleanup prunes epoch training checkpoints."""
    return _append_commit_event(
        events_dir,
        "UNIT_COMMITTED",
        {
            "unit_id": uid,
            "attempt": int(attempt),
            "config_hash": config_hash,
            "dataset_manifest_sha256": manifest_sha256,
            "metrics_sha256": metrics_sha256,
            "preds_sha256": preds_sha256,
        },
    )


def ensure_unit_completion_event(
    *,
    events_dir: Path,
    uid: str,
    attempt: int,
    config_hash: str,
    manifest_sha256: str,
    metrics_sha256: str,
    preds_sha256: str,
) -> dict[str, Any]:
    """Idempotently verify or emit the single authoritative UNIT_COMMITTED event for (uid, attempt).

    Rules (fail closed):
      - Scans `events_dir/event_*.json`, ignoring corrupt/interrupted tail files with warnings.
      - Selects all valid `UNIT_COMMITTED` events matching `(unit_id == uid, attempt == int(attempt))`.
      - Rejects duplicate matching events (`len(matches) > 1`).
      - Rejects any matching event whose `config_hash`, `dataset_manifest_sha256`, `metrics_sha256`,
        or `preds_sha256` differs from the authoritative `run_state.json` commit.
      - Accepts exactly one matching valid event without creating a duplicate.
      - Creates the event via `commit_unit_completion_event(...)` when none exists.
    """
    target_attempt = int(attempt)
    warnings: list[str] = []
    matches: list[tuple[Path, dict[str, Any]]] = []
    if events_dir.is_dir():
        for ep in sorted(events_dir.glob("event_*.json")):
            try:
                ev = _verify_embedded_digest_json(ep, "event_sha256")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                warnings.append(f"Corrupt/interrupted event ignored during reconciliation ({ep.name}): {exc}")
                continue
            if ev.get("event_type") != "UNIT_COMMITTED":
                continue
            if ev.get("unit_id") != uid or ev.get("attempt") != target_attempt:
                continue
            matches.append((ep, ev))

    if len(matches) > 1:
        names = [p.name for p, _ in matches]
        raise ValueError(
            f"Fail closed: duplicate UNIT_COMMITTED events found for unit={uid!r} attempt={target_attempt}: {names}"
        )

    if len(matches) == 1:
        ep, ev = matches[0]
        expected_fields = {
            "config_hash": config_hash,
            "dataset_manifest_sha256": manifest_sha256,
            "metrics_sha256": metrics_sha256,
            "preds_sha256": preds_sha256,
        }
        mismatches = {
            k: {"event": ev.get(k), "expected": v}
            for k, v in expected_fields.items()
            if ev.get(k) != v
        }
        if mismatches:
            raise ValueError(
                f"Fail closed: existing UNIT_COMMITTED event {ep.name} for unit={uid!r} attempt={target_attempt} "
                f"does not match authoritative commit hashes: {mismatches}"
            )
        return {
            "created": False,
            "event_path": ep,
            "event_sha256": sha256_file(ep),
            "embedded_event_sha256": str(ev["event_sha256"]),
            "warnings": warnings,
        }

    event_path, event_sha = commit_unit_completion_event(
        events_dir=events_dir,
        uid=uid,
        attempt=target_attempt,
        config_hash=config_hash,
        manifest_sha256=manifest_sha256,
        metrics_sha256=metrics_sha256,
        preds_sha256=preds_sha256,
    )
    created_ev = _verify_embedded_digest_json(event_path, "event_sha256")
    return {
        "created": True,
        "event_path": event_path,
        "event_sha256": event_sha,
        "embedded_event_sha256": str(created_ev["event_sha256"]),
        "warnings": warnings,
    }
