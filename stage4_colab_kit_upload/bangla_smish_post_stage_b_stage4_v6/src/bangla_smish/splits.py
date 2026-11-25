"""Leakage-free split construction, LODO splits, and adversarial/shortcut stress perturbations."""

from __future__ import annotations

import hashlib
import random
import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .audit import compute_near_dup_clusters
from .schema import extract_surface_features, mask_shortcuts, normalize_text

# Benign Bangla character substitutions / invisible joiner injections seen in adversarial SMS
_ZWJ = "\u200d"
_ZWNJ = "\u200c"
_HOMOGLYPH_MAP = {
    "a": "а",  # Cyrillic small a
    "e": "е",  # Cyrillic small ie
    "o": "о",  # Cyrillic small o
    "p": "р",  # Cyrillic small er
    "c": "с",  # Cyrillic small es
}


def build_dedup_grouped_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    sim_threshold: float = 0.85,
) -> pd.DataFrame:
    """Deduplicate by normalized text, cluster near-duplicates, and assign StratifiedGroupKFold."""
    # Drop conflicting texts across labels rather than picking arbitrarily
    lbl_cnt = df.groupby("text_norm")["label_3class"].nunique()
    valid_texts = set(lbl_cnt[lbl_cnt == 1].index)
    clean = df[df["text_norm"].isin(valid_texts)].drop_duplicates(subset=["text_norm"]).reset_index(drop=True)

    cluster_ids, _ = compute_near_dup_clusters(clean["text_norm"].tolist(), threshold=sim_threshold)
    clean["dup_cluster_id"] = cluster_ids
    clean["fold"] = -1

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y = clean["label_3class"].values
    groups = clean["dup_cluster_id"].values
    for fold_idx, (_, test_idx) in enumerate(sgkf.split(clean, y, groups)):
        clean.loc[test_idx, "fold"] = fold_idx

    feats = extract_surface_features(clean, "text")
    for c in feats.columns:
        clean[c] = feats[c].values
    clean["text_masked"] = clean["text"].map(lambda t: mask_shortcuts(t, mask_urls=True, mask_phones=True))
    return clean


def perturb_zwj_homoglyph(text: str, seed: int = 42) -> str:
    """Inject subtle ZWJ/ZWNJ into Bangla words and Latin homoglyphs to simulate evasion."""
    rng = random.Random(hashlib.md5(f"{seed}:{text}".encode("utf-8")).hexdigest())
    chars = list(text)
    out = []
    for i, ch in enumerate(chars):
        if ch in _HOMOGLYPH_MAP and rng.random() < 0.25:
            out.append(_HOMOGLYPH_MAP[ch])
        else:
            out.append(ch)
        if "\u0980" <= ch <= "\u09ff" and i + 1 < len(chars) and "\u0980" <= chars[i + 1] <= "\u09ff":
            if rng.random() < 0.08:
                out.append(_ZWJ if rng.random() < 0.5 else _ZWNJ)
    return "".join(out)


def build_length_matched_subset(df: pd.DataFrame, min_len: int = 40, max_len: int = 120) -> pd.DataFrame:
    """Filter to overlapping character-length band where length shortcuts fail."""
    if "char_len" not in df.columns:
        lens = df["text"].astype(str).str.len()
    else:
        lens = df["char_len"]
    return df[(lens >= min_len) & (lens <= max_len)].reset_index(drop=True)
