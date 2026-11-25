"""Audit equal model-view strings across the frozen training and evaluation boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .post_stage_b_data import (
    EXPERIMENT_SPECS,
    TEXT_VIEW_COLUMNS,
    build_experiment_fold_data,
    load_heldout_eval_sets,
)


def matching_rows(
    train: pd.DataFrame, evaluation: pd.DataFrame, train_column: str, eval_column: str
) -> list[dict[str, str]]:
    """Report row and family identities for equal input strings without exposing texts."""
    columns = ["row_id", "family_id"]
    train_rows = train[columns + [train_column]].rename(columns={train_column: "input"})
    eval_rows = evaluation[columns + [eval_column]].rename(columns={eval_column: "input"})
    matches = train_rows.merge(eval_rows, on="input", suffixes=("_train", "_evaluation"))
    return matches.drop(columns="input").to_dict("records")


def audit_model_view_overlaps(
    bundle_dir: Path,
    *,
    experiments: Sequence[str] = tuple(EXPERIMENT_SPECS),
    views: Sequence[str] = tuple(TEXT_VIEW_COLUMNS),
    folds: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict[str, Any]:
    """Check exact string identity for every requested split and evaluation."""
    heldout = load_heldout_eval_sets(bundle_dir)
    probe = heldout.pop("counterfactual_smish_probe")
    coverage = 0
    findings = []
    for experiment in experiments:
        for fold in folds:
            train, validation, _ = build_experiment_fold_data(bundle_dir, experiment, fold)
            fold_probe = probe.loc[probe["curation_cv_fold"].eq(fold)]
            for view in views:
                column = TEXT_VIEW_COLUMNS[view]
                evaluations = [
                    ("cv_validation", validation, column),
                    *((name, frame, column) for name, frame in heldout.items()),
                    *(("counterfactual_" + variant, fold_probe, variant + "_" + column)
                      for variant in ("orig", "contact_swap", "contact_brand_swap")),
                ]
                for evaluation, frame, eval_column in evaluations:
                    coverage += 1
                    matches = matching_rows(train, frame, column, eval_column)
                    if matches:
                        findings.append({
                            "experiment": experiment,
                            "fold": fold,
                            "view": view,
                            "evaluation": evaluation,
                            "matches": matches,
                        })
    return {
        "checks": coverage,
        "failed_checks": len(findings),
        "matching_row_pairs": sum(len(result["matches"]) for result in findings),
        "findings": findings,
    }
