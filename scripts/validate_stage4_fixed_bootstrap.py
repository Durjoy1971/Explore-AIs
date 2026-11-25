"""Focused CPU-throttled statistical validation of Stage 4 fixed-budget cluster bootstrap.

Prespecified simulation counts and acceptance criteria are fixed in code PRIOR to execution.
Runs under OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 19
with bounded ProcessPoolExecutor(max_workers=2) to protect concurrent Docker workloads.

Validates the exact production code in `bangla_smish.post_stage_b_stats`:
  1. Startup mathematical identity assertions:
     - 7-pair analytical (TP, E) (1 - F1) gradient vs central finite differences (eps=1e-6)
     - Multinomial cluster collapse equivalence & lossless alpha=0.05 lower-bound short-circuit
       parity (0 / 100 mismatches vs full unconditional B_pval = 50,000).
  2. Degenerate & edge-case suite (8 cases: zero disagreement, missing classes, near-zero variance, tiny G).
  3. Single-contrast Type-I error (B_pval = 50,000) & 95% percentile CI coverage (B_ci = 2,000):
     - Empirical-Structure-Matched Synthetic DGP (n = 1,687, G = 1,299 from frozen primary mask)
     - Synthetic Stress DGP (G = 1,299 unequal heavy-tail clusters, rho_A=0.20, rho_M=0.65)
     - S1 (both zero), S2 (F1=0, Rec>0), S3 (Rec=0, F1>0) [N_outer=1,000 each], S4 (both neg) [N_outer=500]
  4. Multi-hypothesis FWER (Holm-12 Primary Joint-IUT & separate Holm-24 Supporting Endpoints) +
     Exploratory 4-hypothesis C-vs-D family, with exact B_pval = 50,000 (N_outer = 1,000 each).
  5. Fail-closed verdict: sets `confirmatory_declarations_enabled` based strictly on prespecified gates.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.post_stage_b_cohorts import build_smishdetect_test_cohort_masks
from bangla_smish.post_stage_b_stats import (
    DEFAULT_B_CI,
    DEFAULT_B_PVAL,
    _collapse_f1,
    _collapse_rec,
    _macro_f1_obs_stats,
    _recall_obs_stats,
    _stream_studentized_count,
    compare_models_cluster_bootstrap,
    compute_cluster_tables,
    macro_f1_delta_tp_e,
    macro_f1_grad_tp_e,
    simulate_holm_reject_family,
)

MASTER_SEED = 20261124
MAX_WORKERS = 2
Z_975 = 1.959963984540054
TAU = float(norm.ppf(0.99))

# Prespecified simulation counts (NEVER altered after inspecting results)
N_OUTER_BOUNDARY = 1_000
N_OUTER_NEG = 500
N_OUTER_HOLM = 1_000

# Prespecified acceptance criteria
MAX_EMPIRICAL_TYPE1 = 0.060
MAX_WILSON_LOWER = 0.050
MAX_NEGATIVE_TYPE1 = 0.010
MIN_CI_COVERAGE = 0.930
MAX_CI_COVERAGE = 0.970
MAX_HOLM24_DIAG_FWER = 0.065

_WORKER_CTX: dict[str, Any] = {}


def wilson_ci95(k: int, n: int) -> list[float]:
    p = k / float(n)
    z2 = Z_975 * Z_975
    denom = 1.0 + z2 / float(n)
    center = (p + z2 / (2.0 * float(n))) / denom
    half = Z_975 * math.sqrt(max(p * (1.0 - p) + z2 / (4.0 * float(n)), 0.0) / float(n)) / denom
    return [round(max(0.0, center - half), 6), round(min(1.0, center + half), 6)]


def _build_exact_cm(
    pi_fracs: list[Fraction],
    q_fracs: list[Fraction],
    f1_targets: list[Fraction],
    p01_ratio: Fraction = Fraction(1, 2),
) -> np.ndarray:
    P = [[Fraction(0)] * 3 for _ in range(3)]
    for c in range(3):
        P[c][c] = f1_targets[c] * (pi_fracs[c] + q_fracs[c]) / 2
    r = [pi_fracs[c] - P[c][c] for c in range(3)]
    k = [q_fracs[c] - P[c][c] for c in range(3)]
    lo = max(Fraction(0), r[0] - k[2], k[1] - r[2])
    hi = min(r[0], k[1], r[0] + r[1] - k[2])
    if lo > hi:
        raise ValueError(f"Infeasible CM: lo={lo} > hi={hi}")
    p01 = lo + p01_ratio * (hi - lo)
    p02, p21 = r[0] - p01, k[1] - p01
    p12 = k[2] - p02
    p10, p20 = r[1] - p12, r[2] - p21
    P[0][1], P[0][2], P[1][0], P[1][2], P[2][0], P[2][1] = p01, p02, p10, p12, p20, p21
    return np.array([[float(x) for x in row] for row in P], dtype=np.float64)


def _generate_one_pair(
    rng: np.random.Generator,
    y_fam: np.ndarray,
    m_fam: np.ndarray,
    PA: np.ndarray,
    PM: np.ndarray,
    pi: np.ndarray,
    rec_a: float,
    rec_m: float,
    *,
    rho_a: float = 0.20,
    rho_m: float = 0.65,
    rho_am_fam: float = 0.40,
    rho_am_item: float = 0.55,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    G = len(y_fam)
    N = int(m_fam.sum())
    fam = np.repeat(np.arange(G, dtype=np.int64), m_fam)
    yr = np.repeat(y_fam, m_fam)

    zf1, zf2 = rng.standard_normal(G), rng.standard_normal(G)
    ze1, ze2 = rng.standard_normal(N), rng.standard_normal(N)
    bA = zf1
    bM = rho_am_fam * zf1 + math.sqrt(1.0 - rho_am_fam**2) * zf2
    eA = ze1
    eM = rho_am_item * ze1 + math.sqrt(1.0 - rho_am_item**2) * ze2
    uA = math.sqrt(rho_a) * bA[fam] + math.sqrt(1.0 - rho_a) * eA
    uM = math.sqrt(rho_m) * bM[fam] + math.sqrt(1.0 - rho_m) * eM
    vA, vM = ndtr(uA), ndtr(uM)

    cA_cdf = np.cumsum(PA / pi[:, None], axis=1)
    cM_cdf = np.cumsum(PM / pi[:, None], axis=1)
    pA = np.where(vA < cA_cdf[yr, 0], 0, np.where(vA < cA_cdf[yr, 1], 1, 2))
    pM = np.where(vM < cM_cdf[yr, 0], 0, np.where(vM < cM_cdf[yr, 1], 1, 2))

    # Binary recall hit@1%FPR on Smishing rows (and False on others; compute_cluster_tables filters by y==2)
    z_rec_A = ndtr(uA)  # aligned latent prob for Recall positive hit
    z_rec_M = ndtr(uM)
    hitA = (yr == 2) & (z_rec_A >= (1.0 - rec_a))
    hitM = (yr == 2) & (z_rec_M >= (1.0 - rec_m))
    return yr, pA, hitA, pM, hitM, fam


def _lossless_bound_pvalue_at_05(
    *,
    y_true: np.ndarray,
    pred_a: np.ndarray,
    hit_a: np.ndarray,
    pred_m: np.ndarray,
    hit_m: np.ndarray,
    family_ids: np.ndarray,
    seed: int,
    contrast_idx: int,
    B_ci: int = DEFAULT_B_CI,
    B_pval: int = DEFAULT_B_PVAL,
) -> dict[str, Any]:
    """Compute exact B_ci=2,000 CI + lossless <=0.05 exact B_pval=50,000 p-values using production code."""
    tables = compute_cluster_tables(y_true, pred_a, hit_a, pred_m, hit_m, family_ids)
    G = tables["G"]
    f1_obs = _macro_f1_obs_stats(tables["X_int"], G)
    rec_obs = _recall_obs_stats(tables["R_raw"], G)
    ss0, ss1, ss2, ss3 = (
        np.random.SeedSequence([seed, contrast_idx, s]) for s in (0, 1, 2, 3)
    )
    probs_f1, XQu = _collapse_f1(tables["X_int"])
    probs_rec, Ru64 = _collapse_rec(tables["R_raw"])

    # CI check (B_ci = 2,000)
    from bangla_smish.post_stage_b_stats import _eval_f1_chunk, _eval_rec_chunk
    if f1_obs["degenerate_zero_var"] and f1_obs["delta"] == 0.0:
        f1_ci = [0.0, 0.0]
    else:
        M0 = np.random.default_rng(ss0).multinomial(G, probs_f1, size=B_ci)
        db0, _ = _eval_f1_chunk(M0, XQu, G, f1_obs["delta"], f1_obs["se"])
        lo, hi = np.percentile(db0, [2.5, 97.5])
        f1_ci = [float(lo), float(hi)]

    if rec_obs["degenerate_zero_var"] and rec_obs["delta"] == 0.0:
        rec_ci = [0.0, 0.0]
    else:
        M1 = np.random.default_rng(ss1).multinomial(G, probs_rec, size=B_ci)
        db1, _ = _eval_rec_chunk(M1, Ru64, G, rec_obs["delta"], rec_obs["se"])
        lo, hi = np.percentile(db1, [2.5, 97.5])
        rec_ci = [float(lo), float(hi)]

    # Exact lossless p-value if <= 0.05 (since Holm max cutoff across ANY family at alpha=0.05 is 0.05!)
    if f1_obs["degenerate_zero_var"]:
        p_f1 = 1.0
    else:
        k1, n1 = _stream_studentized_count(
            rng=np.random.default_rng(ss2), probs=probs_f1, mat=XQu, G=G,
            d_obs=f1_obs["delta"], se_obs=f1_obs["se"], B_pval=B_pval, is_f1=True,
            shortcircuit_alpha=0.05, chunk_size=500,
        )
        p_f1 = (1 + k1) / (B_pval + 1) if n1 == B_pval else float(max((1 + k1) / (B_pval + 1), 0.050001))

    if rec_obs["degenerate_zero_var"]:
        p_rec = 1.0
    else:
        k2, n2 = _stream_studentized_count(
            rng=np.random.default_rng(ss3), probs=probs_rec, mat=Ru64, G=G,
            d_obs=rec_obs["delta"], se_obs=rec_obs["se"], B_pval=B_pval, is_f1=False,
            shortcircuit_alpha=0.05, chunk_size=500,
        )
        p_rec = (1 + k2) / (B_pval + 1) if n2 == B_pval else float(max((1 + k2) / (B_pval + 1), 0.050001))

    return {
        "p_macro_f1": p_f1,
        "p_recall": p_rec,
        "p_joint": max(p_f1, p_rec),
        "f1_ci95": f1_ci,
        "rec_ci95": rec_ci,
    }


def _eval_holm_family_lossless(
    pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]],
    *,
    seed_base: int,
    sim_idx: int,
    B_pval: int = DEFAULT_B_PVAL,
) -> tuple[bool, bool]:
    """Lossless Null-First alpha/m gate for m-hyp Joint-IUT & 2m-hyp Supporting Holm families.

    Each entry is (y_true, pA, hA, pM, hM, fam, is_f1_null). With m true-null and at most m
    alternative endpoints, neither Holm-m nor Holm-2m can reject any null hypothesis unless
    min(p_null) <= 0.05 / m. When that triggers (~5%), evaluates all 2m endpoints cleanly at 0.05.
    """
    m_hyp = len(pairs)
    gate_alpha = 0.05 / float(m_hyp)
    any_null_below_gate = False
    cached_tables: list[Any] = []

    for j, (yr, pA, hA, pM, hM, fam, is_f1_null) in enumerate(pairs):
        tables = compute_cluster_tables(yr, pA, hA, pM, hM, fam)
        cached_tables.append(tables)
        G = tables["G"]
        c_idx = sim_idx * 16 + j
        if is_f1_null:
            obs = _macro_f1_obs_stats(tables["X_int"], G)
            if not obs["degenerate_zero_var"]:
                probs, mat = _collapse_f1(tables["X_int"])
                _, n_done = _stream_studentized_count(
                    rng=np.random.default_rng(np.random.SeedSequence([seed_base, c_idx, 2])),
                    probs=probs, mat=mat, G=G, d_obs=obs["delta"], se_obs=obs["se"],
                    B_pval=B_pval, is_f1=True, shortcircuit_alpha=gate_alpha, chunk_size=500,
                )
                if n_done == B_pval:
                    any_null_below_gate = True
        else:
            obs = _recall_obs_stats(tables["R_raw"], G)
            if not obs["degenerate_zero_var"]:
                probs, mat = _collapse_rec(tables["R_raw"])
                _, n_done = _stream_studentized_count(
                    rng=np.random.default_rng(np.random.SeedSequence([seed_base, c_idx, 3])),
                    probs=probs, mat=mat, G=G, d_obs=obs["delta"], se_obs=obs["se"],
                    B_pval=B_pval, is_f1=False, shortcircuit_alpha=gate_alpha, chunk_size=500,
                )
                if n_done == B_pval:
                    any_null_below_gate = True

    if not any_null_below_gate:
        return False, False

    pvals_joint: dict[str, float] = {}
    pvals_end: dict[str, float] = {}
    null_end_keys: set[str] = set()
    for j, ((yr, pA, hA, pM, hM, fam, is_f1_null), tables) in enumerate(zip(pairs, cached_tables)):
        G = tables["G"]
        c_idx = sim_idx * 16 + j
        f1_obs = _macro_f1_obs_stats(tables["X_int"], G)
        rec_obs = _recall_obs_stats(tables["R_raw"], G)
        if f1_obs["degenerate_zero_var"]:
            p_f1 = 1.0
        else:
            probs1, mat1 = _collapse_f1(tables["X_int"])
            k1, n1 = _stream_studentized_count(
                rng=np.random.default_rng(np.random.SeedSequence([seed_base, c_idx, 2])),
                probs=probs1, mat=mat1, G=G, d_obs=f1_obs["delta"], se_obs=f1_obs["se"],
                B_pval=B_pval, is_f1=True, shortcircuit_alpha=0.05, chunk_size=500,
            )
            p_f1 = (1 + k1) / (B_pval + 1) if n1 == B_pval else float(max((1 + k1) / (B_pval + 1), 0.050001))
        if rec_obs["degenerate_zero_var"]:
            p_rec = 1.0
        else:
            probs2, mat2 = _collapse_rec(tables["R_raw"])
            k2, n2 = _stream_studentized_count(
                rng=np.random.default_rng(np.random.SeedSequence([seed_base, c_idx, 3])),
                probs=probs2, mat=mat2, G=G, d_obs=rec_obs["delta"], se_obs=rec_obs["se"],
                B_pval=B_pval, is_f1=False, shortcircuit_alpha=0.05, chunk_size=500,
            )
            p_rec = (1 + k2) / (B_pval + 1) if n2 == B_pval else float(max((1 + k2) / (B_pval + 1), 0.050001))
        pvals_joint[f"h{j}"] = max(p_f1, p_rec)
        pvals_end[f"h{j}_f1"] = p_f1
        pvals_end[f"h{j}_rec"] = p_rec
        null_end_keys.add(f"h{j}_f1" if is_f1_null else f"h{j}_rec")

    h_joint = simulate_holm_reject_family(pvals_joint, alpha=0.05)
    h_end = simulate_holm_reject_family(pvals_end, alpha=0.05)
    return (
        any(v["rejected_at_alpha"] for v in h_joint.values()),
        any(h_end[k]["rejected_at_alpha"] for k in null_end_keys),
    )


def _startup_checks() -> dict[str, Any]:
    """Run FD gradient checks + 100-sim exact short-circuit parity checks + 8 degenerate cases."""
    # 1. Analytical gradient vs central finite differences across valid matrices (cluster-mean scale)
    fd_rng = np.random.default_rng(MASTER_SEED)
    max_rel_err = 0.0
    for trial in range(7):
        raw = fd_rng.uniform(0.05, 0.50, size=12)
        grad_ana = macro_f1_grad_tp_e(raw, scale=1.0)
        grad_fd = np.empty(12, dtype=np.float64)
        eps = 1e-6
        for j in range(12):
            vp, vm = raw.copy(), raw.copy()
            vp[j] += eps
            vm[j] -= eps
            grad_fd[j] = (macro_f1_delta_tp_e(vp) - macro_f1_delta_tp_e(vm)) / (2.0 * eps)
        rel_err = float(np.max(np.abs(grad_ana - grad_fd) / np.maximum(np.abs(grad_fd), 1e-12)))
        max_rel_err = max(max_rel_err, rel_err)
    assert max_rel_err < 1e-8, f"FD check failed: {max_rel_err}"

    # 2. 100-sim exact parity between lossless short-circuit at alpha=0.05 and full unconditional B_pval=50,000
    y_t = np.array([0, 1, 2] * 30, dtype=np.int64)
    fams_t = np.repeat(np.arange(45), 2)
    mismatches = 0
    for sim_i in range(100):
        lrng = np.random.default_rng(MASTER_SEED + 1000 + sim_i)
        pA = lrng.integers(0, 3, size=len(y_t))
        pM = pA.copy()
        flip = lrng.random(len(y_t)) < 0.25
        pM[flip] = lrng.integers(0, 3, size=int(flip.sum()))
        hA = (y_t == 2) & (lrng.random(len(y_t)) < 0.70)
        hM = (y_t == 2) & (lrng.random(len(y_t)) < 0.78)
        full = compare_models_cluster_bootstrap(
            y_t, pA, hA, pM, hM, fams_t, B_ci=200, B_pval=DEFAULT_B_PVAL, seed=MASTER_SEED, contrast_index=sim_i
        )
        fast = _lossless_bound_pvalue_at_05(
            y_true=y_t, pred_a=pA, hit_a=hA, pred_m=pM, hit_m=hM, family_ids=fams_t,
            seed=MASTER_SEED, contrast_idx=sim_i, B_ci=200, B_pval=DEFAULT_B_PVAL,
        )
        if (full["p_macro_f1"] <= 0.05) != (fast["p_macro_f1"] <= 0.05) or (full["p_recall"] <= 0.05) != (fast["p_recall"] <= 0.05):
            mismatches += 1
    assert mismatches == 0, f"Lossless short-circuit mismatch count={mismatches}"

    # 3. Degenerate & edge-case suite
    deg_results: dict[str, Any] = {}
    cases = {
        "zero_disagreement": (
            np.array([0, 1, 2] * 20), np.array([0, 1, 2] * 20), np.array([False, False, True] * 20),
            np.array([0, 1, 2] * 20), np.array([False, False, True] * 20), np.repeat(np.arange(30), 2),
        ),
        "missing_class_promotional": (
            np.array([0, 2] * 30), np.array([0, 2] * 30), np.array([False, True] * 30),
            np.array([2, 2] * 30), np.array([False, True] * 30), np.repeat(np.arange(30), 2),
        ),
        "missing_smishing_class": (
            np.array([0, 1] * 30), np.array([0, 1] * 30), np.zeros(60, dtype=bool),
            np.array([1, 1] * 30), np.zeros(60, dtype=bool), np.repeat(np.arange(30), 2),
        ),
        "near_zero_variance_positive": (
            np.array([0, 1, 2] * 20), np.array([1] + [0, 1, 2] * 19 + [0, 1]), np.array([False, False, True] * 20),
            np.array([0, 1, 2] * 20), np.array([False, False, True] * 20), np.repeat(np.arange(30), 2),
        ),
        "near_zero_variance_negative": (
            np.array([0, 1, 2] * 20), np.array([0, 1, 2] * 20), np.array([False, False, True] * 20),
            np.array([1] + [0, 1, 2] * 19 + [0, 1]), np.array([False, False, True] * 20), np.repeat(np.arange(30), 2),
        ),
        "all_correct_zero_errors": (
            np.array([0, 1, 2] * 25), np.array([0, 1, 2] * 25), np.array([False, False, True] * 25),
            np.array([0, 1, 2] * 25), np.array([False, False, True] * 25), np.repeat(np.arange(25), 3),
        ),
        "tiny_G_5": (
            np.array([0, 1, 2] * 5), np.array([0, 1, 0, 0, 1, 2] + [0, 1, 2] * 3), np.array([False, False, True] * 5),
            np.array([0, 1, 2] * 5), np.array([False, False, True] * 5), np.repeat(np.arange(5), 3),
        ),
        "single_cluster_G_1": (
            np.array([0, 1, 2] * 4), np.array([0, 1, 2] * 4), np.array([False, False, True] * 4),
            np.array([0, 0, 2] * 4), np.array([False, False, True] * 4), np.repeat(np.array(["only_fam"]), 12),
        ),
    }
    for cname, args in cases.items():
        res = compare_models_cluster_bootstrap(*args, B_ci=200, B_pval=1000, seed=MASTER_SEED)
        for num_field in ("macro_f1_delta", "macro_f1_se", "p_macro_f1", "recall_delta", "recall_se", "p_recall", "p_joint_iut"):
            assert math.isfinite(res[num_field]), f"Non-finite {num_field} in {cname}"
        deg_results[cname] = {
            "p_macro_f1": res["p_macro_f1"],
            "p_recall": res["p_recall"],
            "flags": res["degenerate_flags"],
        }
    assert deg_results["zero_disagreement"]["p_macro_f1"] == 1.0
    assert deg_results["zero_disagreement"]["p_recall"] == 1.0

    # 4. 20-family Holm-12 lossless gate parity check against full 24-endpoint evaluation (B_pval = 5,000 for unit check)
    holm_mismatches = 0
    for f_i in range(20):
        frng = np.random.default_rng(MASTER_SEED + 5000 + f_i)
        pairs = []
        p_j_full, p_e_full, nulls_f = {}, {}, set()
        for j in range(12):
            yr = np.array([0, 1, 2] * 30, dtype=np.int64)
            fam = np.repeat(np.arange(45), 2)
            pA = frng.integers(0, 3, size=len(yr))
            is_f1_null = (j % 2 == 0)
            # Inject occasional borderline/strong signals to exercise the alpha/12 trigger branch!
            pM = pA.copy() if is_f1_null else np.where(frng.random(len(yr)) < 0.70, yr, pA)
            hA = (yr == 2) & (frng.random(len(yr)) < 0.65)
            hM = (yr == 2) & (frng.random(len(yr)) < (0.65 if not is_f1_null else 0.90))
            if j == 0 and f_i % 3 == 0:
                hM = (yr == 2) & (frng.random(len(yr)) < 0.60)
                pM = np.where(frng.random(len(yr)) < 0.95, yr, pA)
            pairs.append((yr, pA, hA, pM, hM, fam, is_f1_null))
            full_c = compare_models_cluster_bootstrap(
                yr, pA, hA, pM, hM, fam, B_ci=10, B_pval=5_000, seed=MASTER_SEED + 99, contrast_index=f_i * 16 + j
            )
            p_j_full[f"h{j}"] = full_c["p_joint_iut"]
            p_e_full[f"h{j}_f1"] = full_c["p_macro_f1"]
            p_e_full[f"h{j}_rec"] = full_c["p_recall"]
            nulls_f.add(f"h{j}_f1" if is_f1_null else f"h{j}_rec")
        r_j_fast, r_e_fast = _eval_holm_family_lossless(pairs, seed_base=MASTER_SEED + 99, sim_idx=f_i, B_pval=5_000)
        hj_full = simulate_holm_reject_family(p_j_full, alpha=0.05)
        he_full = simulate_holm_reject_family(p_e_full, alpha=0.05)
        r_j_full = any(v["rejected_at_alpha"] for v in hj_full.values())
        r_e_full = any(he_full[k]["rejected_at_alpha"] for k in nulls_f)
        if r_j_fast != r_j_full or r_e_fast != r_e_full:
            holm_mismatches += 1
    assert holm_mismatches == 0, f"Holm lossless gate mismatch count={holm_mismatches}"

    return {
        "max_fd_rel_err": max_rel_err,
        "shortcircuit_parity_mismatches_per_100": mismatches,
        "holm12_parity_mismatches_per_20": holm_mismatches,
        "degenerate_cases": deg_results,
    }


def _init_worker(ctx: dict[str, Any]) -> None:
    _WORKER_CTX.clear()
    _WORKER_CTX.update(ctx)


def _run_single_contrast_sim(task: tuple[str, str, int]) -> tuple[bool, bool, bool, bool, bool]:
    dgp_name, scen_name, sim_idx = task
    ctx = _WORKER_CTX["dgps"][dgp_name]
    scen = ctx["scenarios"][scen_name]
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, hash(dgp_name + scen_name) & 0xFFFFFFFF, sim_idx]))
    yr, pA, hA, pM, hM, fam = _generate_one_pair(
        rng, ctx["y_fam"], ctx["m_fam"], ctx["PA"], scen["PM"], ctx["pi"], ctx["rec_a"], scen["rec_m"]
    )
    out = _lossless_bound_pvalue_at_05(
        y_true=yr, pred_a=pA, hit_a=hA, pred_m=pM, hit_m=hM, family_ids=fam,
        seed=MASTER_SEED + 17, contrast_idx=sim_idx,
    )
    rej_f1 = out["p_macro_f1"] <= 0.05
    rej_rec = out["p_recall"] <= 0.05
    rej_joint = rej_f1 and rej_rec
    cov_f1 = out["f1_ci95"][0] <= scen["true_d_f1"] <= out["f1_ci95"][1]
    cov_rec = out["rec_ci95"][0] <= scen["true_d_rec"] <= out["rec_ci95"][1]
    return rej_f1, rej_rec, rej_joint, cov_f1, cov_rec


def _run_holm_sim(task: tuple[str, bool, int, int]) -> tuple[bool, bool]:
    """Run m-hypothesis family (m=12 confirmatory or m=4 exploratory C vs D) via lossless alpha/m gate."""
    dgp_name, correlated_baseline, m_hyp, sim_idx = task
    ctx = _WORKER_CTX["dgps"][dgp_name]
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, 777 + m_hyp + int(correlated_baseline), sim_idx]))
    y_fam, m_fam, pi, PA, rec_a = ctx["y_fam"], ctx["m_fam"], ctx["pi"], ctx["PA"], ctx["rec_a"]
    G = len(y_fam)
    N = int(m_fam.sum())
    fam = np.repeat(np.arange(G, dtype=np.int64), m_fam)
    yr = np.repeat(y_fam, m_fam)
    cA_cdf = np.cumsum(PA / pi[:, None], axis=1)

    pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]] = []
    for j in range(m_hyp):
        is_f1_null = (j % 2 == 0)
        scen = ctx["scenarios"]["S2_f1_zero_rec_pos" if is_f1_null else "S3_rec_zero_f1_pos"]
        if correlated_baseline and j > 0 and j % 3 != 0:
            pass
        else:
            zf1 = rng.standard_normal(G)
            ze1 = rng.standard_normal(N)
            uA = math.sqrt(0.20) * zf1[fam] + math.sqrt(1.0 - 0.20) * ze1
            vA = ndtr(uA)
            pA = np.where(vA < cA_cdf[yr, 0], 0, np.where(vA < cA_cdf[yr, 1], 1, 2))
            hA = (yr == 2) & (vA >= (1.0 - rec_a))

        zf2 = rng.standard_normal(G)
        ze2 = rng.standard_normal(N)
        bM = 0.40 * zf1 + math.sqrt(1.0 - 0.16) * zf2
        eM = 0.55 * ze1 + math.sqrt(1.0 - 0.3025) * ze2
        vM = ndtr(math.sqrt(0.65) * bM[fam] + math.sqrt(1.0 - 0.65) * eM)
        cM_cdf = np.cumsum(scen["PM"] / pi[:, None], axis=1)
        pM = np.where(vM < cM_cdf[yr, 0], 0, np.where(vM < cM_cdf[yr, 1], 1, 2))
        hM = (yr == 2) & (vM >= (1.0 - scen["rec_m"]))
        pairs.append((yr, pA, hA, pM, hM, fam, is_f1_null))

    return _eval_holm_family_lossless(pairs, seed_base=MASTER_SEED + 99, sim_idx=sim_idx, B_pval=DEFAULT_B_PVAL)


def _build_dgps() -> dict[str, Any]:
    # 1. Empirical-Structure-Matched Synthetic DGP from frozen 1,687-row primary cohort
    masks = build_smishdetect_test_cohort_masks(ROOT / "data/processed/post_stage_b_bundle_v2", verify_tfidf_reproduction=False)
    m_1687 = masks["cohorts"]["primary_family_decontam_1687"]["mask"]
    test_df = pd.read_parquet(ROOT / "data/processed/post_stage_b_bundle_v2/eval_smishdetect_test.parquet").loc[m_1687]
    fam_df = test_df.groupby("family_id").agg(y=("label_idx", "first"), m=("row_id", "size")).reset_index(drop=True)
    y_emp = fam_df["y"].to_numpy(dtype=np.int64)
    m_emp = fam_df["m"].to_numpy(dtype=np.int64)
    c0, c1, c2 = (int((test_df["label_idx"] == i).sum()) for i in range(3))
    n_tot = len(test_df)
    pi_emp_f = [Fraction(c0, n_tot), Fraction(c1, n_tot), Fraction(c2, n_tot)]
    pi_emp = np.array([c0, c1, c2], dtype=np.float64) / float(n_tot)

    PA_emp = _build_exact_cm(pi_emp_f, pi_emp_f, [Fraction(4, 5), Fraction(19, 25), Fraction(21, 25)], Fraction(3, 5))
    PM0_emp = _build_exact_cm(pi_emp_f, [Fraction(c0 + 24, n_tot), Fraction(c1 - 24, n_tot), Fraction(c2, n_tot)], [Fraction(3, 4), Fraction(17, 20), Fraction(4, 5)], Fraction(2, 5))
    PMp_emp = _build_exact_cm(pi_emp_f, [Fraction(c0 - 12, n_tot), Fraction(c1 + 12, n_tot), Fraction(c2, n_tot)], [Fraction(21, 25), Fraction(4, 5), Fraction(22, 25)], Fraction(1, 2))
    PMn_emp = _build_exact_cm(pi_emp_f, [Fraction(c0 + 18, n_tot), Fraction(c1 - 18, n_tot), Fraction(c2, n_tot)], [Fraction(38, 50), Fraction(18, 25), Fraction(4, 5)], Fraction(1, 2))

    # 2. Synthetic Stress DGP (unequal heavy-tail cluster sizes, G = 1,299)
    rng_s = np.random.default_rng(MASTER_SEED + 55)
    G_s = 1299
    pi_s_f = [Fraction(35, 100), Fraction(25, 100), Fraction(40, 100)]
    pi_s = np.array([0.35, 0.25, 0.40], dtype=np.float64)
    y_syn = rng_s.choice(3, size=G_s, p=pi_s).astype(np.int64)
    m_syn = rng_s.choice(np.array([1, 2, 4, 8], dtype=np.int64), size=G_s, p=[0.75, 0.14, 0.07, 0.04])
    PA_syn = _build_exact_cm(pi_s_f, pi_s_f, [Fraction(4, 5), Fraction(19, 25), Fraction(21, 25)], Fraction(1, 2))
    PM0_syn = _build_exact_cm(pi_s_f, [Fraction(38, 100), Fraction(22, 100), Fraction(40, 100)], [Fraction(3, 4), Fraction(17, 20), Fraction(4, 5)], Fraction(1, 3))
    PMp_syn = _build_exact_cm(pi_s_f, [Fraction(34, 100), Fraction(26, 100), Fraction(40, 100)], [Fraction(86, 100), Fraction(82, 100), Fraction(88, 100)], Fraction(1, 2))
    PMn_syn = _build_exact_cm(pi_s_f, [Fraction(39, 100), Fraction(21, 100), Fraction(40, 100)], [Fraction(74, 100), Fraction(70, 100), Fraction(78, 100)], Fraction(2, 3))

    return {
        "emp_1687": {
            "G": len(y_emp),
            "N": int(m_emp.sum()),
            "y_fam": y_emp,
            "m_fam": m_emp,
            "pi": pi_emp,
            "PA": PA_emp,
            "rec_a": 0.72,
            "scenarios": {
                "S1_both_zero": {"PM": PM0_emp, "rec_m": 0.72, "true_d_f1": 0.0, "true_d_rec": 0.0, "null_f1": True, "null_rec": True, "neg": False},
                "S2_f1_zero_rec_pos": {"PM": PM0_emp, "rec_m": 0.80, "true_d_f1": 0.0, "true_d_rec": 0.08, "null_f1": True, "null_rec": False, "neg": False},
                "S3_rec_zero_f1_pos": {"PM": PMp_emp, "rec_m": 0.72, "true_d_f1": 0.04, "true_d_rec": 0.0, "null_f1": False, "null_rec": True, "neg": False},
                "S4_both_neg": {"PM": PMn_emp, "rec_m": 0.65, "true_d_f1": -0.04, "true_d_rec": -0.07, "null_f1": False, "null_rec": False, "neg": True},
            },
        },
        "syn_stress": {
            "G": len(y_syn),
            "N": int(m_syn.sum()),
            "y_fam": y_syn,
            "m_fam": m_syn,
            "pi": pi_s,
            "PA": PA_syn,
            "rec_a": 0.72,
            "scenarios": {
                "S1_both_zero": {"PM": PM0_syn, "rec_m": 0.72, "true_d_f1": 0.0, "true_d_rec": 0.0, "null_f1": True, "null_rec": True, "neg": False},
                "S2_f1_zero_rec_pos": {"PM": PM0_syn, "rec_m": 0.80, "true_d_f1": 0.0, "true_d_rec": 0.08, "null_f1": True, "null_rec": False, "neg": False},
                "S3_rec_zero_f1_pos": {"PM": PMp_syn, "rec_m": 0.72, "true_d_f1": 0.05333333333333334, "true_d_rec": 0.0, "null_f1": False, "null_rec": True, "neg": False},
                "S4_both_neg": {"PM": PMn_syn, "rec_m": 0.65, "true_d_f1": -0.06, "true_d_rec": -0.07, "null_f1": False, "null_rec": False, "neg": True},
            },
        },
    }


def _summarize_rate(k: int, n: int) -> dict[str, Any]:
    rate = k / float(n)
    se = math.sqrt(max(rate * (1.0 - rate), 0.0) / float(n))
    return {"k": int(k), "n": int(n), "rate": round(rate, 6), "mc_se": round(se, 6), "wilson_ci95": wilson_ci95(k, n)}


def main() -> None:
    t_start = time.perf_counter()
    print(f"Starting Stage 4 fixed-budget statistical validation (B_ci={DEFAULT_B_CI}, B_pval={DEFAULT_B_PVAL}, max_workers={MAX_WORKERS})...")
    startup = _startup_checks()
    print(f"Startup assertions PASSED: max_fd_rel_err={startup['max_fd_rel_err']:.3e}, parity_mismatches={startup['shortcircuit_parity_mismatches_per_100']}/100")

    dgps = _build_dgps()
    ctx = {"dgps": {k: {fk: v[fk] for fk in v} for k, v in dgps.items()}}

    single_results: dict[str, Any] = {}
    holm_results: dict[str, Any] = {}
    gate_failures: list[str] = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_init_worker, initargs=(ctx,)) as pool:
        for dgp_name in ("emp_1687", "syn_stress"):
            for scen_name in ("S1_both_zero", "S2_f1_zero_rec_pos", "S3_rec_zero_f1_pos", "S4_both_neg"):
                scen = dgps[dgp_name]["scenarios"][scen_name]
                n_sims = N_OUTER_NEG if scen["neg"] else N_OUTER_BOUNDARY
                t0 = time.perf_counter()
                out = list(pool.map(_run_single_contrast_sim, [(dgp_name, scen_name, i) for i in range(n_sims)], chunksize=25))
                elapsed = round(time.perf_counter() - t0, 2)
                k_f1 = sum(int(x[0]) for x in out)
                k_rec = sum(int(x[1]) for x in out)
                k_joint = sum(int(x[2]) for x in out)
                k_cov_f1 = sum(int(x[3]) for x in out)
                k_cov_rec = sum(int(x[4]) for x in out)

                entry = {
                    "dgp": dgp_name,
                    "scenario": scen_name,
                    "n_outer": n_sims,
                    "elapsed_seconds": elapsed,
                    "macro_f1": _summarize_rate(k_f1, n_sims),
                    "recall": _summarize_rate(k_rec, n_sims),
                    "joint_iut": _summarize_rate(k_joint, n_sims),
                    "ci95_coverage_macro_f1": _summarize_rate(k_cov_f1, n_sims),
                    "ci95_coverage_recall": _summarize_rate(k_cov_rec, n_sims),
                }
                single_results[f"{dgp_name}::{scen_name}"] = entry
                print(
                    f"[Single] {dgp_name:10s} {scen_name:18s} N={n_sims:4d} ({elapsed:6.1f}s) | "
                    f"F1={entry['macro_f1']['rate']:.4f} Rec={entry['recall']['rate']:.4f} "
                    f"IUT={entry['joint_iut']['rate']:.4f} CI95=({entry['ci95_coverage_macro_f1']['rate']:.4f}, {entry['ci95_coverage_recall']['rate']:.4f})"
                )

                # Apply prespecified gates
                if scen["neg"]:
                    if entry["joint_iut"]["rate"] > MAX_NEGATIVE_TYPE1 or entry["macro_f1"]["rate"] > MAX_NEGATIVE_TYPE1 or entry["recall"]["rate"] > MAX_NEGATIVE_TYPE1:
                        gate_failures.append(f"{dgp_name}::{scen_name} negative Type-I exceeded {MAX_NEGATIVE_TYPE1}")
                else:
                    if scen["null_f1"] and (entry["macro_f1"]["rate"] > MAX_EMPIRICAL_TYPE1 or entry["macro_f1"]["wilson_ci95"][0] > MAX_WILSON_LOWER):
                        gate_failures.append(f"{dgp_name}::{scen_name} Macro-F1 Type-I gate failed: {entry['macro_f1']}")
                    if scen["null_rec"] and (entry["recall"]["rate"] > MAX_EMPIRICAL_TYPE1 or entry["recall"]["wilson_ci95"][0] > MAX_WILSON_LOWER):
                        gate_failures.append(f"{dgp_name}::{scen_name} Recall Type-I gate failed: {entry['recall']}")
                    if entry["joint_iut"]["rate"] > MAX_EMPIRICAL_TYPE1 or entry["joint_iut"]["wilson_ci95"][0] > MAX_WILSON_LOWER:
                        gate_failures.append(f"{dgp_name}::{scen_name} Joint IUT Type-I gate failed: {entry['joint_iut']}")

                for cov_key in ("ci95_coverage_macro_f1", "ci95_coverage_recall"):
                    cov_r = entry[cov_key]["rate"]
                    if not (MIN_CI_COVERAGE <= cov_r <= MAX_CI_COVERAGE):
                        gate_failures.append(f"{dgp_name}::{scen_name} {cov_key}={cov_r:.4f} outside [{MIN_CI_COVERAGE}, {MAX_CI_COVERAGE}]")

        holm_specs = [
            ("emp_1687_holm12_correlated", "emp_1687", True, 12, True),
            ("emp_1687_holm12_independent", "emp_1687", False, 12, True),
            ("syn_stress_holm12_correlated", "syn_stress", True, 12, True),
            ("emp_1687_exploratory_c_vs_d_holm4", "emp_1687", True, 4, False),
        ]
        for h_name, dgp_name, corr, m_hyp, is_primary in holm_specs:
            t0 = time.perf_counter()
            out = list(pool.map(_run_holm_sim, [(dgp_name, corr, m_hyp, i) for i in range(N_OUTER_HOLM)], chunksize=10))
            elapsed = round(time.perf_counter() - t0, 2)
            k_j = sum(int(x[0]) for x in out)
            k_e = sum(int(x[1]) for x in out)
            res = {
                "dgp": dgp_name,
                "correlated_baseline": corr,
                "m_hypotheses": m_hyp,
                "is_primary_confirmatory_family": is_primary,
                "n_outer": N_OUTER_HOLM,
                "elapsed_seconds": elapsed,
                "holm_joint_fwer": _summarize_rate(k_j, N_OUTER_HOLM),
                "holm_supporting_endpoints_fwer": _summarize_rate(k_e, N_OUTER_HOLM),
            }
            holm_results[h_name] = res
            print(
                f"[Holm-{m_hyp:2d}] {h_name:34s} N={N_OUTER_HOLM} ({elapsed:6.1f}s) | "
                f"JointFWER={res['holm_joint_fwer']['rate']:.4f} {res['holm_joint_fwer']['wilson_ci95']} | "
                f"EndFWER={res['holm_supporting_endpoints_fwer']['rate']:.4f}"
            )
            if is_primary:
                jf = res["holm_joint_fwer"]
                if jf["rate"] > MAX_EMPIRICAL_TYPE1 or jf["wilson_ci95"][0] > MAX_WILSON_LOWER:
                    gate_failures.append(f"{h_name} Joint FWER gate failed: {jf}")
            ef = res["holm_supporting_endpoints_fwer"]
            if ef["rate"] > MAX_HOLM24_DIAG_FWER or ef["wilson_ci95"][0] > MAX_WILSON_LOWER:
                gate_failures.append(f"{h_name} Supporting endpoint FWER diagnostic exceeded {MAX_HOLM24_DIAG_FWER}: {ef}")

    verdict_enabled = len(gate_failures) == 0
    total_elapsed = round(time.perf_counter() - t_start, 2)
    payload = {
        "schema_version": "stage4_fixed_budget_validation_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_elapsed_seconds": total_elapsed,
        "master_seed": MASTER_SEED,
        "max_workers": MAX_WORKERS,
        "production_params": {
            "B_ci": DEFAULT_B_CI,
            "B_pval": DEFAULT_B_PVAL,
            "gradient": "exact_tp_e_with_1_minus_f1",
            "multiplicity": "Separate Holm-12 Primary Joint-IUT + Separate Holm-24 Supporting + Exploratory Holm-4 C-vs-D",
        },
        "prespecified_criteria": {
            "N_outer_boundary": N_OUTER_BOUNDARY,
            "N_outer_neg": N_OUTER_NEG,
            "N_outer_holm": N_OUTER_HOLM,
            "max_empirical_type1": MAX_EMPIRICAL_TYPE1,
            "max_wilson_lower": MAX_WILSON_LOWER,
            "max_negative_type1": MAX_NEGATIVE_TYPE1,
            "ci_coverage_band": [MIN_CI_COVERAGE, MAX_CI_COVERAGE],
            "max_holm24_diag_fwer": MAX_HOLM24_DIAG_FWER,
        },
        "startup_checks": startup,
        "single_contrast_results": single_results,
        "holm_family_results": holm_results,
        "gate_failures": gate_failures,
        "confirmatory_declarations_enabled": verdict_enabled,
        "verdict": "PASS_CONFIRMATORY_ENABLED" if verdict_enabled else "FAIL_OR_INCONCLUSIVE_CONFIRMATORY_DISABLED",
    }

    out_path = ROOT / "reports" / "stage4_statistical_validation_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path} in {total_elapsed:.1f}s -> Verdict: {payload['verdict']} (failures={len(gate_failures)})")


if __name__ == "__main__":
    main()
