"""Enforceable hard CPU-cap validation runner (<= 25% of host CPU capacity) with kernel telemetry.

Pre-execution gate (fails closed before running target command):
  1. Computes max_cpus = max(1, floor(max_cpu_fraction * host_cpus)) (3 CPUs on a 12-CPU host at 0.25).
  2. Verifies effective process CPU affinity count <= max_cpus.
  3. Verifies active cgroup v2 `cpu.max` CFS quota / period <= max_cpus (e.g., `300000 100000` = 3.0 CPUs = 25%).
  4. Spawns a child Python process and asserts exact affinity inheritance <= max_cpus.

Post-execution telemetry:
  - Reads cgroup `cpu.stat` (`usage_usec`, `nr_throttled`, `throttled_usec`) before and after command execution,
    wall-clock duration, and `resource.getrusage(RUSAGE_CHILDREN)` to report actual host CPU % utilization.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _current_cgroup_relpath() -> str:
    for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3:
            return parts[2].strip()
    return ""


def _read_cgroup_cpu_max(cg_rel: str) -> tuple[float | None, str]:
    base = Path("/sys/fs/cgroup") / cg_rel.lstrip("/")
    for d in [base, *base.parents]:
        p = d / "cpu.max"
        if p.is_file():
            raw = p.read_text(encoding="utf-8").strip()
            tokens = raw.split()
            if len(tokens) == 2 and tokens[0] != "max":
                quota, period = int(tokens[0]), int(tokens[1])
                if period > 0:
                    return quota / float(period), f"{p}={raw}"
            return None, f"{p}={raw}"
        if d == Path("/sys/fs/cgroup"):
            break
    return None, "missing_cpu_max"


def _read_cgroup_cpu_stat(cg_rel: str) -> dict[str, int]:
    p = Path("/sys/fs/cgroup") / cg_rel.lstrip("/") / "cpu.stat"
    if not p.is_file():
        return {}
    out: dict[str, int] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            out[parts[0]] = int(parts[1])
    return out


def verify_hard_cpu_cap(max_cpu_fraction: float = 0.25) -> dict[str, Any]:
    """Fail closed unless kernel CPU affinity, cgroup quota, and child inheritance are <= 25% host capacity."""
    host_cpus = int(os.cpu_count() or 1)
    max_cpus = max(1, int(math.floor(max_cpu_fraction * host_cpus + 1e-9)))

    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) > max_cpus:
        raise RuntimeError(
            f"Hard CPU affinity gate failed: process allowed {len(affinity)} CPUs {affinity}, max allowed is {max_cpus} "
            f"({max_cpu_fraction*100:.0f}% of {host_cpus}). Wrap with `taskset -c 0-{max_cpus - 1}`."
        )

    cg_rel = _current_cgroup_relpath()
    quota_cpus, quota_src = _read_cgroup_cpu_max(cg_rel)
    if quota_cpus is None or quota_cpus > (max_cpus + 1e-6):
        raise RuntimeError(
            f"Hard cgroup CPUQuota gate failed ({quota_src}, parsed_cpus={quota_cpus}); "
            f"required <= {max_cpus:.2f} CPUs ({max_cpu_fraction*100:.0f}% of {host_cpus}). "
            f"Wrap with `systemd-run --user --scope -p CPUQuota={max_cpus * 100}%`."
        )

    child_raw = subprocess.check_output(
        [sys.executable, "-c", "import os, json; print(json.dumps(sorted(os.sched_getaffinity(0))))"],
        text=True,
    ).strip()
    child_affinity = json.loads(child_raw)
    if child_affinity != affinity or len(child_affinity) > max_cpus:
        raise RuntimeError(
            f"Child subprocess affinity inheritance failed: parent={affinity}, child={child_affinity}"
        )

    return {
        "host_logical_cpus": host_cpus,
        "max_cpu_fraction": max_cpu_fraction,
        "max_allowed_cpus": max_cpus,
        "parent_affinity": affinity,
        "child_affinity": child_affinity,
        "cgroup_relpath": cg_rel,
        "cgroup_cpu_max": quota_src,
        "cgroup_quota_cpus": round(quota_cpus, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute command only under verified hard <=25% CPU controls.")
    parser.add_argument("--max-cpu-fraction", type=float, default=0.25)
    parser.add_argument("--telemetry-out", type=Path, default=None)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("Provide command after `--`")

    pre = verify_hard_cpu_cap(max_cpu_fraction=args.max_cpu_fraction)
    stat_before = _read_cgroup_cpu_stat(pre["cgroup_relpath"])
    ru_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()

    proc = subprocess.run(cmd)

    wall_sec = max(time.perf_counter() - t0, 1e-6)
    ru_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    stat_after = _read_cgroup_cpu_stat(pre["cgroup_relpath"])

    child_cpu_sec = (ru_after.ru_utime - ru_before.ru_utime) + (ru_after.ru_stime - ru_before.ru_stime)
    host_cpus = pre["host_logical_cpus"]
    cg_delta_usec = (
        stat_after["usage_usec"] - stat_before["usage_usec"]
        if "usage_usec" in stat_before and "usage_usec" in stat_after
        else None
    )
    cg_throttled_delta = (
        stat_after.get("nr_throttled", 0) - stat_before.get("nr_throttled", 0)
        if "nr_throttled" in stat_after
        else None
    )
    effective_host_pct = (
        round((cg_delta_usec / 1e6) / (wall_sec * host_cpus) * 100.0, 2)
        if cg_delta_usec is not None
        else round(child_cpu_sec / (wall_sec * host_cpus) * 100.0, 2)
    )

    telemetry = {
        **pre,
        "command": cmd,
        "returncode": int(proc.returncode),
        "wall_seconds": round(wall_sec, 3),
        "rusage_children_cpu_seconds": round(child_cpu_sec, 3),
        "cgroup_usage_seconds": round(cg_delta_usec / 1e6, 3) if cg_delta_usec is not None else None,
        "cgroup_nr_throttled_delta": cg_throttled_delta,
        "effective_host_cpu_pct": effective_host_pct,
    }
    if args.telemetry_out is not None:
        args.telemetry_out.parent.mkdir(parents=True, exist_ok=True)
        args.telemetry_out.write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
    print("STAGE4_CAPPED_TELEMETRY=" + json.dumps(telemetry, sort_keys=True))
    sys.exit(int(proc.returncode))


if __name__ == "__main__":
    main()
