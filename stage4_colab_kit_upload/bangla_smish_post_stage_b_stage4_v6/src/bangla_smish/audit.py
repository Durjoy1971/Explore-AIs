"""Phase 1 Dataset Verification: duplicates, split leakage, shortcuts, and cross-corpus overlap."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors

from .schema import extract_surface_features


def compute_near_dup_clusters(texts: list[str], threshold: float = 0.85) -> tuple[np.ndarray, int]:
    """Group texts with char-ngram TF-IDF cosine >= threshold into connected components."""
    n = len(texts)
    if n == 0:
        return np.array([], dtype=int), 0
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    X = vec.fit_transform(texts)
    nn = NearestNeighbors(metric="cosine", radius=1.0 - threshold, algorithm="brute", n_jobs=-1)
    nn.fit(X)
    adj = nn.radius_neighbors_graph(X, mode="connectivity")

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    rows, cols = adj.nonzero()
    for r, c in zip(rows, cols):
        if r < c:
            union(int(r), int(c))

    roots = np.array([find(i) for i in range(n)], dtype=int)
    _, cluster_ids = np.unique(roots, return_inverse=True)
    n_clusters = int(cluster_ids.max() + 1) if n > 0 else 0
    return cluster_ids, n_clusters


def audit_single_dataset(name: str, df: pd.DataFrame) -> dict[str, Any]:
    """Audit duplicates, label conflicts, split leakage, surface confounders, and shortcut power."""
    feats = extract_surface_features(df, "text")
    work = pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

    total = len(work)
    unique_norm = work["text_norm"].nunique()
    exact_dup_count = total - unique_norm
    exact_dup_pct = round(exact_dup_count / max(1, total) * 100.0, 2)

    # Label conflicts among identical normalized texts
    lbl_per_text = work.groupby("text_norm")["label_3class"].nunique()
    conflicting_texts = int((lbl_per_text > 1).sum())

    # Cross-split leakage if dataset ships with predefined train/val/test splits
    split_leakage = {}
    if set(work["orig_split"].unique()) - {"unspecified"}:
        train_texts = set(work.loc[work["orig_split"] == "train", "text_norm"])
        for sp in ("validation", "test"):
            sp_texts = work.loc[work["orig_split"] == sp, "text_norm"]
            if len(sp_texts) > 0:
                leaked = int(sp_texts.isin(train_texts).sum())
                split_leakage[f"train_to_{sp}_exact_rows"] = leaked
                split_leakage[f"train_to_{sp}_pct"] = round(leaked / len(sp_texts) * 100.0, 2)

    # Near-duplicate clustering on unique normalized texts
    uniq_df = work.drop_duplicates(subset=["text_norm"]).reset_index(drop=True)
    cluster_ids, n_clusters = compute_near_dup_clusters(uniq_df["text_norm"].tolist(), threshold=0.85)
    near_dup_reduction_pct = round((len(uniq_df) - n_clusters) / max(1, len(uniq_df)) * 100.0, 2)

    # Surface stats by class
    class_surface = {}
    for cls, grp in work.groupby("label_3class"):
        class_surface[str(cls)] = {
            "count": int(len(grp)),
            "unique_norm": int(grp["text_norm"].nunique()),
            "median_char_len": float(grp["char_len"].median()),
            "mean_char_len": round(float(grp["char_len"].mean()), 1),
            "pct_has_latin": round(float(grp["has_latin"].mean() * 100.0), 1),
            "pct_has_url": round(float(grp["has_url"].mean() * 100.0), 1),
            "pct_has_phone": round(float(grp["has_phone"].mean() * 100.0), 1),
            "pct_has_redacted_tag": round(float(grp["has_redacted_tag"].mean() * 100.0), 1),
        }

    # How much can surface features ALONE predict the labels? (Shortcut diagnostic)
    surf_cols = ["char_len", "word_len", "has_bn", "has_latin", "has_url", "has_phone", "has_ussd", "has_redacted_tag"]
    X_surf = uniq_df[surf_cols].to_numpy(dtype=float)
    y_3 = uniq_df["label_3class"].astype(str).to_numpy(dtype=object)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    pred_3 = cross_val_predict(clf, X_surf, y_3, cv=cv)
    surf_acc_3 = round(float(accuracy_score(y_3, pred_3) * 100.0), 2)
    surf_f1_3 = round(float(f1_score(y_3, pred_3, average="macro") * 100.0), 2)

    # Binary risk AUC using ONLY char_len and surface features
    y_bin = uniq_df["label_binary_risk"].to_numpy(dtype=int)
    len_auc = round(float(roc_auc_score(y_bin, uniq_df["char_len"].values)), 4)
    if len_auc < 0.5:
        len_auc = round(1.0 - len_auc, 4)
    prob_bin = cross_val_predict(clf, X_surf, y_bin, cv=cv, method="predict_proba")[:, 1]
    surf_bin_auc = round(float(roc_auc_score(y_bin, prob_bin)), 4)

    return {
        "dataset": name,
        "license": str(work["license"].iloc[0]),
        "is_synthetic": str(work["is_synthetic"].iloc[0]),
        "total_rows": int(total),
        "unique_norm_rows": int(unique_norm),
        "exact_dup_rows": int(exact_dup_count),
        "exact_dup_pct": exact_dup_pct,
        "conflicting_label_texts": conflicting_texts,
        "near_dup_clusters_085": int(n_clusters),
        "near_dup_within_unique_pct": near_dup_reduction_pct,
        "split_leakage": split_leakage,
        "surface_only_5fold_acc_pct": surf_acc_3,
        "surface_only_5fold_macro_f1_pct": surf_f1_3,
        "length_only_risk_roc_auc": len_auc,
        "surface_only_risk_roc_auc": surf_bin_auc,
        "class_surface_stats": class_surface,
    }


def compute_cross_dataset_overlap(datasets: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Measure exact normalized-text overlap across every pair of datasets."""
    names = list(datasets.keys())
    text_sets = {k: set(v["text_norm"]) for k, v in datasets.items()}
    overlaps = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            inter = text_sets[a] & text_sets[b]
            overlaps.append(
                {
                    "dataset_a": a,
                    "dataset_b": b,
                    "shared_unique_texts": len(inter),
                    "pct_of_a": round(len(inter) / max(1, len(text_sets[a])) * 100.0, 2),
                    "pct_of_b": round(len(inter) / max(1, len(text_sets[b])) * 100.0, 2),
                }
            )
    return overlaps
