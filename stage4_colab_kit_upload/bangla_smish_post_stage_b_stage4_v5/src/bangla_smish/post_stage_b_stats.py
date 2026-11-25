"""Stage 4 fixed-budget family-cluster statistical inference and multiplicity control.

Default confirmatory inference uses fixed-budget multinomial-collapsed family-cluster
bootstrap:
  - B_ci   = 2,000  for 95% family-clustered percentile confidence intervals
  - B_pval = 50,000 for null-centered studentized cluster-bootstrap one-sided p-values
with the exact (TP, E) (1 - F1) Macro-F1 influence-function gradient, Joint IUT
p_joint = max(p_macro_f1, p_recall), and separate Holm-12 (primary confirmatory),
Holm-24 (supporting endpoints), and Holm-4 (exploratory C vs D) multiplicity families.

Note: The adaptive bootstrap helper is retained strictly as an optional non-confirmatory
diagnostic pending larger outer validation (Round 4 Adaptive H2 = 164/3,000 = 0.0547).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .metrics import _cluster_indices, holm_adjust_pvalues
from .schema import CANONICAL_LABELS

DEFAULT_B_CI = 2_000
DEFAULT_B_PVAL = 50_000
DEFAULT_MASTER_SEED = 20261124
OPTION_A_PRIMARY_ARMS = ("B", "C", "D")
OPTION_A_BACKBONES = (
    "csebuetnlp/banglabert",
    "google/muril-base-cased",
    "xlm-roberta-base",
    "distilbert-base-multilingual-cased",
)

# Fail-closed gate controlled by the focused Stage 4 validation verdict
# (reports/stage4_statistical_validation_results.json -> syn_stress::S1_both_zero Recall 0.0620 > 0.0600).
CONFIRMATORY_DECLARATIONS_ENABLED: bool = False
STATISTICAL_VALIDATION_STATUS: str = "FAIL_OR_INCONCLUSIVE_CONFIRMATORY_DISABLED"

INFERENCE_CONDITIONING_STATEMENT = (
    "Conditional on the trained models, initial seed, and frozen smishdetect_validation-"
    "calibrated operating thresholds; does not represent unconditional multi-seed variance."
)

_IU, _JU = np.triu_indices(12)
_MULT = np.where(_IU == _JU, 1.0, 2.0).astype(np.float64)


def macro_f1_grad_tp_e(vec: np.ndarray, scale: float) -> np.ndarray:
    """Exact (TP, E) analytical gradient of Macro-F1(M) - Macro-F1(A) with (1 - F1) factors.

    vec ordering: [tpA_0..2, eA_0..2, tpM_0..2, eM_0..2].
    scale = 1.0 for cluster means, or scale = float(G) for aggregate cluster totals.
    """
    v_in = np.asarray(vec, dtype=np.float64)
    if v_in.shape != (12,):
        raise ValueError(f"Expected 12-vector, got shape {v_in.shape}")
    k = 3.0
    tpA, eA, tpM, eM = v_in[0:3], v_in[3:6], v_in[6:9], v_in[9:12]
    dA = 2.0 * tpA + eA
    dM = 2.0 * tpM + eM
    fA = np.divide(2.0 * tpA, dA, out=np.zeros(3, dtype=np.float64), where=dA > 0)
    fM = np.divide(2.0 * tpM, dM, out=np.zeros(3, dtype=np.float64), where=dM > 0)
    grad = np.zeros(12, dtype=np.float64)
    grad[0:3] = np.divide(-2.0 * scale * (1.0 - fA), k * dA, out=np.zeros(3, dtype=np.float64), where=dA > 0)
    grad[3:6] = np.divide(+1.0 * scale * fA, k * dA, out=np.zeros(3, dtype=np.float64), where=dA > 0)
    grad[6:9] = np.divide(+2.0 * scale * (1.0 - fM), k * dM, out=np.zeros(3, dtype=np.float64), where=dM > 0)
    grad[9:12] = np.divide(-1.0 * scale * fM, k * dM, out=np.zeros(3, dtype=np.float64), where=dM > 0)
    return grad


def macro_f1_delta_tp_e(vec: np.ndarray) -> float:
    """Compute Macro-F1(M) - Macro-F1(A) from 12-vector (totals or means)."""
    v_in = np.asarray(vec, dtype=np.float64)
    tpA, eA, tpM, eM = v_in[0:3], v_in[3:6], v_in[6:9], v_in[9:12]
    dA = 2.0 * tpA + eA
    dM = 2.0 * tpM + eM
    fA = np.divide(2.0 * tpA, dA, out=np.zeros(3, dtype=np.float64), where=dA > 0)
    fM = np.divide(2.0 * tpM, dM, out=np.zeros(3, dtype=np.float64), where=dM > 0)
    return float(fM.mean() - fA.mean())


def compute_cluster_tables(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    hit_a: np.ndarray,
    pred_m: np.ndarray,
    hit_m: np.ndarray,
    family_ids: np.ndarray | None,
    *,
    smish_idx: int = 2,
) -> dict[str, Any]:
    """Aggregate aligned per-row predictions into per-family (TP, E) and Recall@threshold tables."""
    y = np.asarray(y_true, dtype=np.int64)
    pA = np.asarray(pred_a, dtype=np.int64)
    hA = np.asarray(hit_a, dtype=bool)
    pM = np.asarray(pred_m, dtype=np.int64)
    hM = np.asarray(hit_m, dtype=bool)
    n = len(y)
    if not (y.ndim == 1 and pA.shape == (n,) and pM.shape == (n,) and hA.shape == (n,) and hM.shape == (n,)):
        raise ValueError("Aligned 1D arrays of equal length are required")

    fam_idx, n_clusters = _cluster_indices(n, family_ids)
    X_int = np.empty((n_clusters, 12), dtype=np.int64)
    for c in range(3):
        yc = y == c
        pAc = pA == c
        pMc = pM == c
        X_int[:, c] = np.bincount(fam_idx, weights=yc & pAc, minlength=n_clusters).astype(np.int64)
        X_int[:, 3 + c] = np.bincount(fam_idx, weights=yc ^ pAc, minlength=n_clusters).astype(np.int64)
        X_int[:, 6 + c] = np.bincount(fam_idx, weights=yc & pMc, minlength=n_clusters).astype(np.int64)
        X_int[:, 9 + c] = np.bincount(fam_idx, weights=yc ^ pMc, minlength=n_clusters).astype(np.int64)

    is_smish = y == smish_idx
    n2_g = np.bincount(fam_idx, weights=is_smish, minlength=n_clusters).astype(np.int64)
    d_rec_g = np.bincount(
        fam_idx, weights=is_smish.astype(np.int64) * (hM.astype(np.int64) - hA.astype(np.int64)), minlength=n_clusters
    ).astype(np.int64)
    R_raw = np.column_stack([d_rec_g, n2_g])
    return {
        "n_rows": n,
        "G": n_clusters,
        "X_int": X_int,
        "R_raw": R_raw,
        "missing_classes": [CANONICAL_LABELS[c] for c in range(3) if not bool((y == c).any())],
        "n_smishing_rows": int(is_smish.sum()),
    }


def _macro_f1_obs_stats(X_int: np.ndarray, G: int) -> dict[str, Any]:
    X64 = X_int.astype(np.float64, copy=False)
    tot = X64.sum(axis=0)
    delta = macro_f1_delta_tp_e(tot)
    v_obs = macro_f1_grad_tp_e(tot, float(G))
    psi = X64 @ v_obs
    centered_ss = float(np.sum((psi - psi.mean()) ** 2))
    var = centered_ss / (G * (G - 1)) if G > 1 else 0.0
    se = math.sqrt(max(var, 0.0))
    return {
        "delta": delta,
        "se": se,
        "grad": v_obs,
        "psi": psi,
        "degenerate_zero_var": bool(se <= 1e-12),
    }


def _recall_obs_stats(R_raw: np.ndarray, G: int) -> dict[str, Any]:
    R64 = R_raw.astype(np.float64, copy=False)
    tot_d = float(R64[:, 0].sum())
    tot_n2 = float(R64[:, 1].sum())
    if tot_n2 <= 0.0:
        return {"delta": 0.0, "se": 0.0, "missing_smishing": True, "degenerate_zero_var": True}
    delta = tot_d / tot_n2
    resid = (R64[:, 0] - delta * R64[:, 1]) / tot_n2
    var = float(np.sum(resid**2)) * (G / (G - 1)) if G > 1 else 0.0
    se = math.sqrt(max(var, 0.0))
    return {
        "delta": delta,
        "se": se,
        "missing_smishing": False,
        "degenerate_zero_var": bool(se <= 1e-12),
    }


def _eval_f1_chunk(M_counts: np.ndarray, XQu: np.ndarray, G: int, d_obs: float, se_obs: float) -> tuple[np.ndarray, int]:
    TM = M_counts.astype(np.float64) @ XQu
    T = TM[:, :12]
    M = TM[:, 12:]
    dA = 2.0 * T[:, 0:3] + T[:, 3:6]
    dM = 2.0 * T[:, 6:9] + T[:, 9:12]
    FA = np.divide(2.0 * T[:, 0:3], dA, out=np.zeros((len(T), 3), dtype=np.float64), where=dA > 0)
    FM = np.divide(2.0 * T[:, 6:9], dM, out=np.zeros((len(T), 3), dtype=np.float64), where=dM > 0)
    d_b = FM.mean(axis=1) - FA.mean(axis=1)
    V = np.zeros((len(T), 12), dtype=np.float64)
    V[:, 0:3] = np.divide(-(2.0 * G / 3.0) * (1.0 - FA), dA, out=np.zeros((len(T), 3), dtype=np.float64), where=dA > 0)
    V[:, 3:6] = np.divide(+(G / 3.0) * FA, dA, out=np.zeros((len(T), 3), dtype=np.float64), where=dA > 0)
    V[:, 6:9] = np.divide(+(2.0 * G / 3.0) * (1.0 - FM), dM, out=np.zeros((len(T), 3), dtype=np.float64), where=dM > 0)
    V[:, 9:12] = np.divide(-(G / 3.0) * FM, dM, out=np.zeros((len(T), 3), dtype=np.float64), where=dM > 0)
    quad = np.sum(M * (V[:, _IU] * V[:, _JU]), axis=1)
    denom_g = float(G * max(G - 1, 1))
    se_b = np.sqrt(np.maximum(quad / denom_g, 1e-24))
    t_obs = d_obs / max(se_obs, 1e-12)
    t_star = (d_b - d_obs) / se_b
    return d_b, int(np.sum(t_star >= t_obs))


def _eval_rec_chunk(M_counts: np.ndarray, Ru64: np.ndarray, G: int, d_obs: float, se_obs: float) -> tuple[np.ndarray, int]:
    T = M_counts.astype(np.float64) @ Ru64
    valid = T[:, 1] > 0
    d_b = np.divide(T[:, 0], T[:, 1], out=np.zeros(len(T), dtype=np.float64), where=valid)
    quad = np.maximum(T[:, 2] - 2.0 * d_b * T[:, 3] + (d_b**2) * T[:, 4], 0.0)
    scale_g = float(G / max(G - 1, 1))
    se_b = np.sqrt(
        np.divide(quad * scale_g, np.maximum(T[:, 1] ** 2, 1.0), out=np.full(len(T), 1e-24), where=valid)
    )
    se_b = np.maximum(se_b, 1e-12)
    t_obs = d_obs / max(se_obs, 1e-12)
    t_star = (d_b - d_obs) / se_b
    # When a bootstrap draw misses every Smishing cluster (T[:, 1] == 0), treat t_star as -inf unless d_obs==0.
    t_star = np.where(valid, t_star, -np.inf)
    return d_b, int(np.sum(t_star >= t_obs))


def _collapse_f1(X_int: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xu, counts = np.unique(X_int, axis=0, return_counts=True)
    Xu64 = Xu.astype(np.float64)
    XQu = np.hstack([Xu64, (Xu64[:, _IU] * Xu64[:, _JU]) * _MULT])
    return counts.astype(np.float64) / float(X_int.shape[0]), XQu


def _collapse_rec(R_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Ru, counts = np.unique(R_raw, axis=0, return_counts=True)
    Ru64 = np.column_stack([Ru[:, 0], Ru[:, 1], Ru[:, 0] ** 2, Ru[:, 0] * Ru[:, 1], Ru[:, 1] ** 2]).astype(np.float64)
    return counts.astype(np.float64) / float(R_raw.shape[0]), Ru64


def _stream_studentized_count(
    *,
    rng: np.random.Generator,
    probs: np.ndarray,
    mat: np.ndarray,
    G: int,
    d_obs: float,
    se_obs: float,
    B_pval: int,
    is_f1: bool,
    shortcircuit_alpha: float | None = None,
    chunk_size: int = 500,
) -> tuple[int, int]:
    """Run B_pval draws in bounded chunks; optional lossless early stop ONLY for binary reject(alpha) check."""
    max_k_to_reject: int | None = None
    if shortcircuit_alpha is not None:
        max_k_to_reject = int(math.floor(shortcircuit_alpha * (B_pval + 1))) - 1
        if max_k_to_reject < 0:
            return B_pval, 0

    k_exceed = 0
    done = 0
    eval_fn = _eval_f1_chunk if is_f1 else _eval_rec_chunk
    while done < B_pval:
        cur = min(chunk_size, B_pval - done)
        M_c = rng.multinomial(G, probs, size=cur)
        _, k_c = eval_fn(M_c, mat, G, d_obs, se_obs)
        k_exceed += k_c
        done += cur
        if max_k_to_reject is not None and k_exceed > max_k_to_reject:
            break
    return k_exceed, done


def compare_models_cluster_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    hit_a: np.ndarray,
    pred_m: np.ndarray,
    hit_m: np.ndarray,
    family_ids: np.ndarray | None,
    *,
    smish_idx: int = 2,
    B_ci: int = DEFAULT_B_CI,
    B_pval: int = DEFAULT_B_PVAL,
    seed: int = DEFAULT_MASTER_SEED,
    contrast_index: int = 0,
    method: str = "fixed_budget",
) -> dict[str, Any]:
    """Compare Model M minus Model A via fixed-budget family-cluster bootstrap."""
    if B_ci < 1 or B_pval < 1:
        raise ValueError("Bootstrap counts B_ci and B_pval must be positive")
    if method not in {"fixed_budget", "adaptive_non_confirmatory"}:
        raise ValueError(f"Unsupported bootstrap method: {method!r}")

    tables = compute_cluster_tables(y_true, pred_a, hit_a, pred_m, hit_m, family_ids, smish_idx=smish_idx)
    G = tables["G"]
    X_int = tables["X_int"]
    R_raw = tables["R_raw"]

    f1_obs = _macro_f1_obs_stats(X_int, G)
    rec_obs = _recall_obs_stats(R_raw, G)

    degenerate_flags: list[str] = []
    if tables["missing_classes"]:
        degenerate_flags.append("missing_class:" + ",".join(tables["missing_classes"]))
    if G < 2:
        degenerate_flags.append("insufficient_clusters")
    if f1_obs["degenerate_zero_var"]:
        degenerate_flags.append("f1_zero_or_near_zero_variance")
    if rec_obs["missing_smishing"]:
        degenerate_flags.append("missing_smishing_class")
    elif rec_obs["degenerate_zero_var"]:
        degenerate_flags.append("recall_zero_or_near_zero_variance")

    # Separate deterministic independent streams derived from (seed, contrast_index, stream_id)
    rng_ci_f1, rng_ci_rec, rng_p_f1, rng_p_rec = (
        np.random.default_rng(np.random.SeedSequence([int(seed), int(contrast_index), s]))
        for s in (0, 1, 2, 3)
    )

    probs_f1, XQu = _collapse_f1(X_int)
    probs_rec, Ru64 = _collapse_rec(R_raw)

    # Stream 0 & 1: Percentile 95% CIs (B_ci = 2,000)
    if G < 2 or (f1_obs["degenerate_zero_var"] and f1_obs["delta"] == 0.0):
        f1_ci = [float(f1_obs["delta"]), float(f1_obs["delta"])]
    else:
        M_ci_f1 = rng_ci_f1.multinomial(G, probs_f1, size=B_ci)
        d_b_f1, _ = _eval_f1_chunk(M_ci_f1, XQu, G, f1_obs["delta"], f1_obs["se"])
        lo, hi = np.percentile(d_b_f1, [2.5, 97.5])
        f1_ci = [float(lo), float(hi)]

    if G < 2 or rec_obs["missing_smishing"] or (rec_obs["degenerate_zero_var"] and rec_obs["delta"] == 0.0):
        rec_ci = [float(rec_obs["delta"]), float(rec_obs["delta"])]
    else:
        M_ci_rec = rng_ci_rec.multinomial(G, probs_rec, size=B_ci)
        d_b_rec, _ = _eval_rec_chunk(M_ci_rec, Ru64, G, rec_obs["delta"], rec_obs["se"])
        lo, hi = np.percentile(d_b_rec, [2.5, 97.5])
        rec_ci = [float(lo), float(hi)]

    # Stream 2 & 3: Null-centered studentized one-sided p-values (B_pval = 50,000)
    if G < 2 or f1_obs["degenerate_zero_var"]:
        k_f1, executed_f1, p_f1 = B_pval, 0, 1.0
    else:
        k_f1, executed_f1 = _stream_studentized_count(
            rng=rng_p_f1, probs=probs_f1, mat=XQu, G=G,
            d_obs=f1_obs["delta"], se_obs=f1_obs["se"], B_pval=B_pval, is_f1=True,
        )
        p_f1 = float((1 + k_f1) / (executed_f1 + 1))

    if G < 2 or rec_obs["missing_smishing"] or rec_obs["degenerate_zero_var"]:
        k_rec, executed_rec, p_rec = B_pval, 0, 1.0
    else:
        k_rec, executed_rec = _stream_studentized_count(
            rng=rng_p_rec, probs=probs_rec, mat=Ru64, G=G,
            d_obs=rec_obs["delta"], se_obs=rec_obs["se"], B_pval=B_pval, is_f1=False,
        )
        p_rec = float((1 + k_rec) / (executed_rec + 1))

    p_joint = float(max(p_f1, p_rec))
    mc_se_f1 = float(math.sqrt(p_f1 * (1.0 - p_f1) / (B_pval + 1)))
    mc_se_rec = float(math.sqrt(p_rec * (1.0 - p_rec) / (B_pval + 1)))

    method_eligible = bool(method == "fixed_budget")
    confirmatory_allowed = bool(method_eligible and CONFIRMATORY_DECLARATIONS_ENABLED)
    return {
        "method": method,
        "method_confirmatory_eligible": method_eligible,
        "confirmatory_allowed": confirmatory_allowed,
        "statistical_validation_status": (
            "VALIDATED_CONFIRMATORY" if CONFIRMATORY_DECLARATIONS_ENABLED else STATISTICAL_VALIDATION_STATUS
        ),
        "p_value_status": (
            "VALIDATED_CONFIRMATORY" if confirmatory_allowed else "UNVALIDATED_DIAGNOSTIC_ONLY"
        ),
        "inference_conditioning": INFERENCE_CONDITIONING_STATEMENT,
        "n_rows": tables["n_rows"],
        "n_families": G,
        "n_smishing_rows": tables["n_smishing_rows"],
        "B_ci": int(B_ci),
        "B_pval": int(B_pval),
        "macro_f1_delta": float(f1_obs["delta"]),
        "macro_f1_delta_pp": round(float(f1_obs["delta"]) * 100.0, 4),
        "macro_f1_se": float(f1_obs["se"]),
        "macro_f1_ci95": f1_ci,
        "macro_f1_ci95_pp": [round(x * 100.0, 4) for x in f1_ci],
        "p_macro_f1": p_f1,
        "k_macro_f1": int(k_f1),
        "mc_se_p_macro_f1": mc_se_f1,
        "recall_delta": float(rec_obs["delta"]),
        "recall_delta_pp": round(float(rec_obs["delta"]) * 100.0, 4),
        "recall_se": float(rec_obs["se"]),
        "recall_ci95": rec_ci,
        "recall_ci95_pp": [round(x * 100.0, 4) for x in rec_ci],
        "p_recall": p_rec,
        "k_recall": int(k_rec),
        "mc_se_p_recall": mc_se_rec,
        "p_joint_iut": p_joint,
        "degenerate_flags": degenerate_flags,
    }


def compare_models_reject_only_for_validation(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    hit_a: np.ndarray,
    pred_m: np.ndarray,
    hit_m: np.ndarray,
    family_ids: np.ndarray | None,
    *,
    alpha: float = 0.05,
    B_pval: int = DEFAULT_B_PVAL,
    seed: int = DEFAULT_MASTER_SEED,
    contrast_index: int = 0,
) -> dict[str, Any]:
    """Exact lossless short-circuit test used ONLY by Monte Carlo validation scripts."""
    tables = compute_cluster_tables(y_true, pred_a, hit_a, pred_m, hit_m, family_ids)
    G = tables["G"]
    f1_obs = _macro_f1_obs_stats(tables["X_int"], G)
    rec_obs = _recall_obs_stats(tables["R_raw"], G)
    ss_f1 = np.random.SeedSequence([int(seed), int(contrast_index), 2])
    ss_rec = np.random.SeedSequence([int(seed), int(contrast_index), 3])

    if G < 2 or f1_obs["degenerate_zero_var"]:
        rej_f1 = False
    else:
        probs_f1, XQu = _collapse_f1(tables["X_int"])
        k1, n1 = _stream_studentized_count(
            rng=np.random.default_rng(ss_f1), probs=probs_f1, mat=XQu, G=G,
            d_obs=f1_obs["delta"], se_obs=f1_obs["se"], B_pval=B_pval, is_f1=True,
            shortcircuit_alpha=alpha,
        )
        rej_f1 = (n1 == B_pval) and ((1 + k1) / (B_pval + 1) <= alpha)

    if G < 2 or rec_obs["missing_smishing"] or rec_obs["degenerate_zero_var"]:
        rej_rec = False
    else:
        probs_rec, Ru64 = _collapse_rec(tables["R_raw"])
        k2, n2 = _stream_studentized_count(
            rng=np.random.default_rng(ss_rec), probs=probs_rec, mat=Ru64, G=G,
            d_obs=rec_obs["delta"], se_obs=rec_obs["se"], B_pval=B_pval, is_f1=False,
            shortcircuit_alpha=alpha,
        )
        rej_rec = (n2 == B_pval) and ((1 + k2) / (B_pval + 1) <= alpha)

    return {
        "reject_macro_f1": bool(rej_f1),
        "reject_recall": bool(rej_rec),
        "reject_joint_iut": bool(rej_f1 and rej_rec),
    }


def simulate_holm_reject_family(
    raw_pvalues: dict[str, float],
    *,
    alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Simulation-only Holm rejection counter for Monte Carlo validation scripts (never for production claims)."""
    if not raw_pvalues:
        return {}
    adjusted = holm_adjust_pvalues(raw_pvalues)
    out: dict[str, dict[str, Any]] = {}
    for key, raw_p in raw_pvalues.items():
        adj_p = float(adjusted[key])
        out[key] = {
            "raw_p_value": float(raw_p),
            "holm_adjusted_p_value": adj_p,
            "rejected_at_alpha": bool(adj_p <= alpha),
            "family_size": len(raw_pvalues),
            "alpha": float(alpha),
            "simulation_only": True,
            "production_confirmatory_claim": False,
        }
    return out


def holm_reject_family(
    raw_pvalues: dict[str, float],
    *,
    alpha: float = 0.05,
    confirmatory_allowed: bool = False,
) -> dict[str, dict[str, Any]]:
    """Apply Holm-Bonferroni correction cleanly inside a single declared hypothesis family (fail-closed default)."""
    if not raw_pvalues:
        return {}
    adjusted = holm_adjust_pvalues(raw_pvalues)
    effective_allowed = bool(confirmatory_allowed and CONFIRMATORY_DECLARATIONS_ENABLED)
    bypass_blocked = bool(confirmatory_allowed and not CONFIRMATORY_DECLARATIONS_ENABLED)
    val_status = "VALIDATED_CONFIRMATORY" if CONFIRMATORY_DECLARATIONS_ENABLED else STATISTICAL_VALIDATION_STATUS
    out: dict[str, dict[str, Any]] = {}
    for key, raw_p in raw_pvalues.items():
        adj_p = float(adjusted[key])
        out[key] = {
            "raw_p_value": float(raw_p),
            "holm_adjusted_p_value": adj_p,
            "rejected_at_alpha": bool(effective_allowed and (adj_p <= alpha)),
            "family_size": len(raw_pvalues),
            "alpha": float(alpha),
            "declarations_enabled": effective_allowed,
            "confirmatory_allowed": effective_allowed,
            "governance_bypass_blocked": bypass_blocked,
            "statistical_validation_status": val_status,
            "p_value_status": "VALIDATED_CONFIRMATORY" if effective_allowed else "UNVALIDATED_DIAGNOSTIC_ONLY",
        }
    return out


def evaluate_stage4_hypothesis_families(
    contrasts: dict[tuple[str, str, str, str], dict[str, Any]],
    *,
    alpha: float = 0.05,
    confirmatory_declarations_enabled: bool = CONFIRMATORY_DECLARATIONS_ENABLED,
) -> dict[str, Any]:
    """Organize contrasts into the approved separate Holm families.

    contrasts key: (view, arm_m, arm_a, backbone) -> compare_models_cluster_bootstrap result.
    Strictly preserves:
      1. primary_confirmatory_holm12  : (B/C/D vs A) x 4 backbones on canon_masked (12 Joint-IUT)
      2. supporting_endpoints_holm24  : 12 Macro-F1 + 12 Recall on canon_masked (24 supporting)
      3. exploratory_c_vs_d_pilot     : (D vs C) x 4 backbones on canon_masked (Holm-4 / Holm-8)
      4. reference_view_raw           : raw view contrasts (descriptive + reference Holm, non-primary)
    """
    primary_joint_raw: dict[str, float] = {}
    supporting_endpoints_raw: dict[str, float] = {}
    exploratory_joint_raw: dict[str, float] = {}
    exploratory_endpoints_raw: dict[str, float] = {}
    raw_view_entries: dict[str, Any] = {}

    for (view, arm_m, arm_a, backbone), stats in sorted(contrasts.items()):
        cid = f"{arm_m}_vs_{arm_a}__{backbone}__{view}"
        if view == "canon_masked" and arm_a == "A" and arm_m in OPTION_A_PRIMARY_ARMS:
            primary_joint_raw[cid] = float(stats["p_joint_iut"])
            supporting_endpoints_raw[f"{cid}__macro_f1"] = float(stats["p_macro_f1"])
            supporting_endpoints_raw[f"{cid}__recall"] = float(stats["p_recall"])
        elif view == "canon_masked" and (arm_m, arm_a) == ("D", "C"):
            exploratory_joint_raw[cid] = float(stats["p_joint_iut"])
            exploratory_endpoints_raw[f"{cid}__macro_f1"] = float(stats["p_macro_f1"])
            exploratory_endpoints_raw[f"{cid}__recall"] = float(stats["p_recall"])
        elif view == "raw":
            raw_view_entries[cid] = stats

    effective_enabled = bool(confirmatory_declarations_enabled and CONFIRMATORY_DECLARATIONS_ENABLED)
    bypass_blocked = bool(confirmatory_declarations_enabled and not CONFIRMATORY_DECLARATIONS_ENABLED)
    val_status = "VALIDATED_CONFIRMATORY" if CONFIRMATORY_DECLARATIONS_ENABLED else STATISTICAL_VALIDATION_STATUS

    primary_holm12 = holm_reject_family(
        primary_joint_raw,
        alpha=alpha,
        confirmatory_allowed=effective_enabled,
    )
    supporting_holm24 = holm_reject_family(
        supporting_endpoints_raw,
        alpha=alpha,
        confirmatory_allowed=effective_enabled,
    )
    exploratory_holm4 = holm_reject_family(
        exploratory_joint_raw,
        alpha=alpha,
        confirmatory_allowed=False,  # Always exploratory (Pilot contribution), never confirmatory superiority
    )
    exploratory_holm8 = holm_reject_family(
        exploratory_endpoints_raw,
        alpha=alpha,
        confirmatory_allowed=False,
    )

    return {
        "confirmatory_declarations_enabled": effective_enabled,
        "confirmatory_status": "ENABLED_VALIDATED" if effective_enabled else STATISTICAL_VALIDATION_STATUS,
        "statistical_validation_status": val_status,
        "governance_bypass_blocked": bypass_blocked,
        "inference_conditioning": INFERENCE_CONDITIONING_STATEMENT,
        "multiplicity_structure": {
            "primary_confirmatory_family": "Holm-12 Joint-IUT on canon_masked (B/C/D vs A x 4 backbones)",
            "supporting_endpoint_family": "Separate Holm-24 on individual canon_masked co-primary endpoints",
            "exploratory_pilot_family": "Separate Holm-4 (Joint-IUT) and Holm-8 (endpoints) for D vs C on canon_masked",
            "pooled_36_used": False,
        },
        "primary_confirmatory_holm12": {
            "expected_family_size": 12,
            "actual_family_size": len(primary_joint_raw),
            "complete_family": len(primary_joint_raw) == 12,
            "results": primary_holm12,
        },
        "supporting_endpoints_holm24": {
            "expected_family_size": 24,
            "actual_family_size": len(supporting_endpoints_raw),
            "complete_family": len(supporting_endpoints_raw) == 24,
            "results": supporting_holm24,
        },
        "exploratory_c_vs_d_pilot_contribution": {
            "designation": "prespecified_exploratory_outside_primary_12",
            "expected_joint_family_size": 4,
            "actual_joint_family_size": len(exploratory_joint_raw),
            "joint_holm4": exploratory_holm4,
            "endpoint_holm8": exploratory_holm8,
        },
        "reference_view_raw_contrasts": raw_view_entries,
        "descriptive_effects": {
            f"{arm_m}_vs_{arm_a}__{backbone}__{view}": stats
            for (view, arm_m, arm_a, backbone), stats in sorted(contrasts.items())
        },
    }
