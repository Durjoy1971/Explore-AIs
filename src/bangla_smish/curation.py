"""Stage A reproducible dataset consolidation, normalization, quarantine, and family review."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold

from .audit import compute_near_dup_clusters
from .schema import (
    CANONICAL_LABELS,
    detect_script_bucket,
    extract_surface_features,
    mask_shortcuts,
    normalize_text,
)

SPLIT_PRIORITY = {"test": 0, "validation": 1, "train": 2, "unspecified": 3}
DATASET_PRIORITY = {"smishdetect_2026": 0, "banglaphish_2026": 1, "bangalabarta_2025": 2, "legacy_2024": 3}

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)
_DIGITS_RE = re.compile(r"\d+", re.UNICODE)
_MASKED_NON_WORD_RE = re.compile(r"[^\w<>]+", re.UNICODE)


@dataclass(frozen=True)
class SourceSpec:
    """Metadata describing a canonical logical source dataset."""

    dataset: str
    files: tuple[str, ...]
    license: str
    synthetic_status: str
    notes: str


LOGICAL_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        dataset="legacy_2024",
        files=("DataSet.csv",),
        license="UNDOCUMENTED",
        synthetic_status="unknown",
        notes="Binary ham/spam CSV without header; kept reference-only unless corroborated.",
    ),
    SourceSpec(
        dataset="bangalabarta_2025",
        files=("data/raw/bangalabarta/BangalaBarta bangla_spam_sms smishing.csv",),
        license="CC BY-NC-SA 4.0",
        synthetic_status="curated_real_mixed",
        notes="Mendeley 10.17632/jfkfbw3gzh.3; .zip archive contains identical CSV (counted once).",
    ),
    SourceSpec(
        dataset="smishdetect_2026",
        files=(
            "data/raw/smishdetect_hf/data/train-00000-of-00001.parquet",
            "data/raw/smishdetect_hf/data/validation-00000-of-00001.parquet",
            "data/raw/smishdetect_hf/data/test-00000-of-00001.parquet",
        ),
        license="MIT",
        synthetic_status="augmented_multilingual",
        notes="HuggingFace shariul-islam/bengali-sms-smishing-dataset; train/validation/test parquets.",
    ),
    SourceSpec(
        dataset="banglaphish_2026",
        files=("data/raw/banglaphish2026/data/BanglaPhish-2026_v1.0_MASTER.csv",),
        license="CC BY-NC 4.0",
        synthetic_status="synthetic",
        notes="MASTER.csv canonical; .jsonl/.xlsx and data/splits/* are identical exports (counted once); OOD benchmark.",
    ),
)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """Compute SHA-256 hex digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strong_normalize_text(text: Any) -> str:
    """Normalize text for exact comparison via Cf-removal, NFKC, whitespace collapse, and casefold."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    cleaned: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == "Cf":
            continue
        if cat.startswith("Z") or ch in ("\n", "\r", "\t", "\v", "\f"):
            cleaned.append(" ")
        elif cat == "Cc":
            continue
        else:
            cleaned.append(ch)
    normalized = unicodedata.normalize("NFKC", "".join(cleaned))
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def strip_markdown_links(text: str) -> str:
    """Unwrap Markdown [label](url) links and escaped underscores for variant comparison."""
    return _MD_LINK_RE.sub(r"\2", text).replace("\\_", "_")


def cosmetic_key(text: str) -> str:
    """Key ignoring spacing, punctuation, and Markdown link wrappers while preserving words and digits."""
    return _NON_WORD_RE.sub("", strong_normalize_text(strip_markdown_links(text)))


def slot_template_text(text: str) -> str:
    """Normalize shortcuts (URL/phone/USSD) and digits to <N> for template clustering."""
    base = strong_normalize_text(strip_markdown_links(text))
    masked = mask_shortcuts(base, mask_urls=True, mask_phones=True)
    return _DIGITS_RE.sub("<N>", masked.casefold())


def slot_template_key(text: str) -> str:
    """Punctuation-free slot-template key for detecting numeric/phone/code slot-swap clones."""
    return _MASKED_NON_WORD_RE.sub("", slot_template_text(text))


def is_unusable_text(text: str, norm_strict: str) -> bool:
    """Return True if text is empty after strict normalization or contains Unicode replacement chars."""
    return not norm_strict or "\ufffd" in text


def inventory_sources(root: Path) -> list[dict[str, Any]]:
    """Compute file-level sizes, SHA-256 checksums, and logical source metadata."""
    records: list[dict[str, Any]] = []
    for spec in LOGICAL_SOURCES:
        file_infos = []
        for rel in spec.files:
            p = root / rel
            file_infos.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
        records.append(
            {
                "dataset": spec.dataset,
                "license": spec.license,
                "synthetic_status": spec.synthetic_status,
                "notes": spec.notes,
                "files": file_infos,
            }
        )
    return records


def load_raw_records_with_provenance(root: Path) -> pd.DataFrame:
    """Load all canonical logical sources once with full file, row-index, split, and metadata provenance."""
    rows: list[dict[str, Any]] = []

    # 1. legacy_2024 (DataSet.csv, no header)
    leg_rel = "DataSet.csv"
    with open(root / leg_rel, "r", encoding="utf-8", errors="replace", newline="") as f:
        for idx, row in enumerate(csv.reader(f)):
            if len(row) < 2:
                continue
            raw_lbl = row[0].strip()
            lbl_lower = raw_lbl.lower()
            text = row[1]
            mapped = "Normal" if lbl_lower == "ham" else ("Spam_Binary" if lbl_lower == "spam" else f"Ambiguous:{raw_lbl}")
            rows.append(
                {
                    "dataset": "legacy_2024",
                    "source_file": leg_rel,
                    "source_row_index": idx,
                    "orig_id": f"legacy_{idx}",
                    "orig_split": "unspecified",
                    "orig_label": raw_lbl,
                    "mapped_label_3class": mapped,
                    "source_language_raw": "unspecified",
                    "source_type": "SMS",
                    "domain_id": "unspecified",
                    "domain": "unspecified",
                    "text": text,
                    "license": "UNDOCUMENTED",
                    "source_synthetic_status": "unknown",
                }
            )

    # 2. bangalabarta_2025
    bb_rel = "data/raw/bangalabarta/BangalaBarta bangla_spam_sms smishing.csv"
    bb_df = pd.read_csv(root / bb_rel)
    bb_map = {"normal": "Normal", "promo": "Promotional", "smish": "Smishing"}
    for idx, r in enumerate(bb_df.itertuples(index=False)):
        raw_lbl = str(r.label).strip()
        text = "" if pd.isna(r.text) else str(r.text)
        rows.append(
            {
                "dataset": "bangalabarta_2025",
                "source_file": bb_rel,
                "source_row_index": idx,
                "orig_id": f"bb_{idx}",
                "orig_split": "unspecified",
                "orig_label": raw_lbl,
                "mapped_label_3class": bb_map.get(raw_lbl.lower(), f"Ambiguous:{raw_lbl}"),
                "source_language_raw": "unspecified",
                "source_type": "SMS",
                "domain_id": "unspecified",
                "domain": "unspecified",
                "text": text,
                "license": "CC BY-NC-SA 4.0",
                "source_synthetic_status": "curated_real_mixed",
            }
        )

    # 3. smishdetect_2026
    sd_map = {"normal": "Normal", "promo": "Promotional", "smish": "Smishing"}
    for split in ("train", "validation", "test"):
        sd_rel = f"data/raw/smishdetect_hf/data/{split}-00000-of-00001.parquet"
        sd_df = pd.read_parquet(root / sd_rel)
        for idx, r in enumerate(sd_df.itertuples(index=False)):
            raw_lbl = str(r.label).strip()
            text = "" if pd.isna(r.text) else str(r.text)
            rows.append(
                {
                    "dataset": "smishdetect_2026",
                    "source_file": sd_rel,
                    "source_row_index": idx,
                    "orig_id": f"sd_{split}_{idx}",
                    "orig_split": split,
                    "orig_label": raw_lbl,
                    "mapped_label_3class": sd_map.get(raw_lbl.lower(), f"Ambiguous:{raw_lbl}"),
                    "source_language_raw": str(r.source).strip(),
                    "source_type": "SMS",
                    "domain_id": "unspecified",
                    "domain": "unspecified",
                    "text": text,
                    "license": "MIT",
                    "source_synthetic_status": "augmented_multilingual",
                }
            )

    # 4. banglaphish_2026
    bp_rel = "data/raw/banglaphish2026/data/BanglaPhish-2026_v1.0_MASTER.csv"
    bp_df = pd.read_csv(root / bp_rel)
    for idx, r in enumerate(bp_df.itertuples(index=False)):
        raw_lbl = str(r.label).strip()
        lbl_low = raw_lbl.lower()
        dom = "" if pd.isna(r.domain) else str(r.domain).strip()
        if lbl_low == "scam":
            mapped = "Smishing"
        elif lbl_low == "legitimate":
            mapped = "Promotional" if "promotional" in dom.lower() else "Normal"
        else:
            mapped = f"Ambiguous:{raw_lbl}"
        text = "" if pd.isna(r.text_bn) else str(r.text_bn)
        rows.append(
            {
                "dataset": "banglaphish_2026",
                "source_file": bp_rel,
                "source_row_index": idx,
                "orig_id": str(r.sample_id).strip(),
                "orig_split": str(r.benchmark_split).strip(),
                "orig_label": raw_lbl,
                "mapped_label_3class": mapped,
                "source_language_raw": str(r.language).strip(),
                "source_type": str(r.source_type).strip(),
                "domain_id": str(r.domain_id).strip(),
                "domain": dom,
                "text": text,
                "license": "CC BY-NC 4.0",
                "source_synthetic_status": "synthetic",
            }
        )

    df = pd.DataFrame(rows)
    df["provenance_row_id"] = df["dataset"] + ":" + df["orig_id"]
    df["text_norm"] = df["text"].map(normalize_text)
    df["text_norm_strict"] = df["text"].map(strong_normalize_text)
    df["text_hash_strict"] = df["text_norm_strict"].map(sha256_text)
    df["script_bucket"] = df["text"].map(detect_script_bucket)
    df["is_unusable"] = [is_unusable_text(t, s) for t, s in zip(df["text"], df["text_norm_strict"])]
    return df


def classify_quarantine_reason(grp: pd.DataFrame) -> str | None:
    """Determine quarantine reason for a strict-normalized text group, or None if clean."""
    if grp["is_unusable"].any():
        return "empty_or_corrupted_text"

    valid_3c_set = set(CANONICAL_LABELS)
    non_legacy = grp[grp["dataset"] != "legacy_2024"]
    legacy = grp[grp["dataset"] == "legacy_2024"]

    if len(non_legacy) == 0:
        leg_labels = set(legacy["orig_label"].str.lower())
        if len(leg_labels) > 1:
            return "legacy_internal_binary_conflict"
        lbl = next(iter(leg_labels))
        if lbl == "spam":
            return "legacy_binary_spam_reference_only"
        if lbl == "ham":
            return "legacy_binary_ham_undocumented_license_reference_only"
        return "legacy_ambiguous_label"

    nl_labels = set(non_legacy["mapped_label_3class"])
    if not nl_labels.issubset(valid_3c_set):
        return "ambiguous_3class_label"
    if len(nl_labels) > 1:
        return "conflicting_3class_labels"

    canon_lbl = next(iter(nl_labels))
    if len(legacy) > 0:
        leg_labels = set(legacy["orig_label"].str.lower())
        if "spam" in leg_labels and canon_lbl == "Normal":
            return "cross_source_conflict_legacy_spam_vs_normal"
        if "ham" in leg_labels and canon_lbl in ("Promotional", "Smishing"):
            return "cross_source_conflict_legacy_ham_vs_non_normal"
    return None


def _resolve_language(row: pd.Series) -> str:
    """Resolve canonical language tag from source metadata and script bucket."""
    src_lang = str(row["source_language_raw"])
    ds_name = str(row["dataset"])
    if ds_name == "banglaphish_2026":
        return "Bengali" if src_lang == "Bangla" else src_lang
    if src_lang in ("Bengali", "Banglish", "English", "CodeMix"):
        return src_lang
    return str(row["script_bucket"])


def _select_canonical_row(grp: pd.DataFrame) -> pd.Series:
    """Pick the deterministic canonical representative row within a clean exact-duplicate group."""
    work = grp[grp["dataset"] != "legacy_2024"].copy()
    work["_sp"] = work["orig_split"].map(SPLIT_PRIORITY).fillna(9).astype(int)
    work["_dp"] = work["dataset"].map(DATASET_PRIORITY).fillna(9).astype(int)
    work["_lang_mismatch"] = (
        ((work["source_language_raw"] == "CodeMix") & (work["script_bucket"] == "Bengali"))
        | ((work["source_language_raw"] == "Bengali") & (work["script_bucket"] == "Latin"))
    ).astype(int)
    work["_non_nfc"] = (work["text"] != work["text"].map(lambda t: unicodedata.normalize("NFC", t))).astype(int)
    work["_has_md_esc"] = work["text"].str.contains(r"\]\(https?://|\\_", regex=True).astype(int)
    work = work.sort_values(["_sp", "_dp", "_lang_mismatch", "_non_nfc", "_has_md_esc", "orig_id"])
    return work.iloc[0]


def _pick_representative_order(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows so evaluation/non-escaped/NFC canonical representatives come first."""
    work = df.copy()
    work["_sp"] = work["original_split"].map(SPLIT_PRIORITY).fillna(9).astype(int)
    work["_dp"] = work["primary_dataset"].map(DATASET_PRIORITY).fillna(9).astype(int)
    work["_non_nfc"] = (work["text"] != work["text"].map(lambda t: unicodedata.normalize("NFC", t))).astype(int)
    work["_has_md_esc"] = work["text"].str.contains(r"\]\(https?://|\\_", regex=True).astype(int)
    return work.sort_values(["_sp", "_dp", "_non_nfc", "_has_md_esc", "primary_orig_id"])


def _build_union_families(df: pd.DataFrame, threshold: float = 0.85) -> np.ndarray:
    """Cluster exact-unique records using union of raw-strict and slot-template char-ngram cosine >= threshold."""
    n = len(df)
    if n == 0:
        return np.array([], dtype=int)

    c_raw, _ = compute_near_dup_clusters(df["text_norm_strict"].tolist(), threshold=threshold)
    c_tmpl, _ = compute_near_dup_clusters(df["text"].map(slot_template_text).tolist(), threshold=threshold)

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

    for cluster_arr in (c_raw, c_tmpl):
        first_seen: dict[int, int] = {}
        for idx, cid in enumerate(cluster_arr):
            cid_int = int(cid)
            if cid_int in first_seen:
                union(first_seen[cid_int], idx)
            else:
                first_seen[cid_int] = idx

    roots = np.array([find(i) for i in range(n)], dtype=int)
    _, normalized = np.unique(roots, return_inverse=True)
    return normalized


def _load_phase2_3_fold_map(root: Path) -> dict[str, int]:
    """Read existing phase2_3 CV fold assignments read-only for traceability."""
    fold_map: dict[str, int] = {}
    for rel in ("data/processed/unified_3class_dedup.parquet", "data/processed/banglaphish2026_dedup.parquet"):
        p = root / rel
        if p.exists():
            pdf = pd.read_parquet(p, columns=["text_norm", "fold"])
            for tn, fld in zip(pdf["text_norm"], pdf["fold"]):
                fold_map[str(tn)] = int(fld)
    return fold_map


def run_stage_a_curation(
    root: str | Path = ".",
    output_dir: str | Path | None = None,
    sim_threshold: float = 0.85,
) -> dict[str, Any]:
    """Execute full Stage A consolidation, quarantine, near-dup family review, and export."""
    root_path = Path(root)
    out_dir = Path(output_dir) if output_dir is not None else root_path / "data" / "processed" / "curation_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    source_inventory = inventory_sources(root_path)
    raw_df = load_raw_records_with_provenance(root_path)
    phase2_3_folds = _load_phase2_3_fold_map(root_path)

    return curate_from_raw_dataframe(
        raw_df=raw_df,
        out_dir=out_dir,
        source_inventory=source_inventory,
        phase2_3_folds=phase2_3_folds,
        sim_threshold=sim_threshold,
    )


def curate_from_raw_dataframe(
    raw_df: pd.DataFrame,
    out_dir: Path,
    source_inventory: list[dict[str, Any]] | None = None,
    phase2_3_folds: dict[str, int] | None = None,
    sim_threshold: float = 0.85,
) -> dict[str, Any]:
    """Core Stage A pipeline operating on a pre-loaded raw provenance DataFrame."""
    out_dir.mkdir(parents=True, exist_ok=True)
    phase2_3_folds = phase2_3_folds or {}
    source_inventory = source_inventory or []

    quarantine_rows: list[dict[str, Any]] = []
    clean_canon_rows: list[dict[str, Any]] = []
    group_status: dict[str, tuple[str, str, str]] = {}

    for thash, grp in raw_df.groupby("text_hash_strict", sort=True):
        q_reason = classify_quarantine_reason(grp)
        if q_reason is not None:
            qid = f"quar_{thash[:12]}"
            group_status[thash] = ("quarantined", qid, q_reason)
            rep_row = grp.sort_values(["dataset", "orig_id"]).iloc[0]
            quarantine_rows.append(
                {
                    "quarantine_id": qid,
                    "text": rep_row["text"],
                    "text_norm": rep_row["text_norm"],
                    "text_norm_strict": rep_row["text_norm_strict"],
                    "text_hash_strict": thash,
                    "quarantine_reason": q_reason,
                    "observed_orig_labels": "|".join(sorted(grp["orig_label"].astype(str).unique())),
                    "observed_mapped_labels": "|".join(sorted(grp["mapped_label_3class"].astype(str).unique())),
                    "contributing_datasets": ",".join(sorted(grp["dataset"].unique())),
                    "contributing_orig_ids": ",".join(grp["provenance_row_id"].tolist()),
                    "contributing_splits": ",".join(sorted(grp["orig_split"].unique())),
                    "raw_row_count": int(len(grp)),
                    "eligible_for_training": False,
                }
            )
            continue

        rep = _select_canonical_row(grp)
        cid = f"cur_{thash[:12]}"
        group_status[thash] = ("clean", cid, "exact_strict_match")

        nl_grp = grp[grp["dataset"] != "legacy_2024"]
        best_split = min(nl_grp["orig_split"].unique(), key=lambda s: SPLIT_PRIORITY.get(s, 9))
        contrib_ds = sorted(grp["dataset"].unique())
        lodo_grp = (
            "banglaphish_ood"
            if rep["dataset"] == "banglaphish_2026"
            else ("bangalabarta_2025" if "bangalabarta_2025" in contrib_ds else "smishdetect_only")
        )
        clean_canon_rows.append(
            {
                "canonical_id": cid,
                "text": rep["text"],
                "text_norm": rep["text_norm"],
                "text_norm_strict": rep["text_norm_strict"],
                "text_hash_strict": thash,
                "label_3class": rep["mapped_label_3class"],
                "label_binary_risk": 1 if rep["mapped_label_3class"] == "Smishing" else 0,
                "language": _resolve_language(rep),
                "source_language_raw": rep["source_language_raw"],
                "script_bucket": rep["script_bucket"],
                "is_synthetic": False,
                "source_synthetic_status": rep["source_synthetic_status"],
                "primary_dataset": rep["dataset"],
                "primary_orig_id": rep["orig_id"],
                "primary_provenance_row_id": rep["provenance_row_id"],
                "contributing_datasets": ",".join(contrib_ds),
                "contributing_count": int(len(grp)),
                "license": rep["license"],
                "original_split": best_split,
                "lodo_group": lodo_grp,
                "phase2_3_cv_fold": int(phase2_3_folds.get(rep["text_norm"], -1)),
                "source_type": rep["source_type"],
                "domain_id": rep["domain_id"],
                "domain": rep["domain"],
            }
        )

    quarantine_cols = [
        "quarantine_id",
        "text",
        "text_norm",
        "text_norm_strict",
        "text_hash_strict",
        "quarantine_reason",
        "observed_orig_labels",
        "observed_mapped_labels",
        "contributing_datasets",
        "contributing_orig_ids",
        "contributing_splits",
        "raw_row_count",
        "eligible_for_training",
    ]
    quarantined_df = (
        pd.DataFrame(quarantine_rows, columns=quarantine_cols).sort_values("quarantine_id").reset_index(drop=True)
    )
    exact_unique_df = pd.DataFrame(clean_canon_rows).sort_values("canonical_id").reset_index(drop=True)

    # Cluster exact-unique records into near-duplicate / template families BEFORE collapsing variants
    exact_unique_df["_cluster_num"] = _build_union_families(exact_unique_df, threshold=sim_threshold)
    cluster_to_fam_id: dict[int, str] = {}
    for cnum, grp in exact_unique_df.groupby("_cluster_num"):
        joined = "|".join(sorted(grp["canonical_id"].tolist()))
        cluster_to_fam_id[int(cnum)] = f"fam_{sha256_text(joined)[:10]}"
    exact_unique_df["family_id"] = exact_unique_df["_cluster_num"].map(cluster_to_fam_id)

    vec_raw = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    X_raw = vec_raw.fit_transform(exact_unique_df["text_norm_strict"]) if len(exact_unique_df) else None

    exact_unique_df["_cosmetic_key"] = exact_unique_df["text"].map(cosmetic_key)
    exact_unique_df["_slot_key"] = exact_unique_df["text"].map(slot_template_key)

    ordered_df = _pick_representative_order(exact_unique_df)
    collapsed_map: dict[str, tuple[str, str]] = {}

    non_bp = ordered_df[ordered_df["primary_dataset"] != "banglaphish_2026"]
    seen_cosmetic: dict[tuple[str, str], str] = {}
    for row in non_bp.itertuples(index=False):
        key = (row._cosmetic_key, row.label_3class)
        if key in seen_cosmetic:
            collapsed_map[row.canonical_id] = (seen_cosmetic[key], "cosmetic_punctuation_spacing_or_markdown_variant")
        else:
            seen_cosmetic[key] = row.canonical_id

    seen_slot: dict[tuple[str, str], str] = {}
    for row in non_bp.itertuples(index=False):
        if row.canonical_id in collapsed_map:
            continue
        key = (row._slot_key, row.label_3class)
        if key in seen_slot:
            collapsed_map[row.canonical_id] = (seen_slot[key], "numeric_or_shortcut_slot_clone")
        else:
            seen_slot[key] = row.canonical_id

    retained_mask = ~exact_unique_df["canonical_id"].isin(collapsed_map)
    existing_unique = exact_unique_df[retained_mask].copy().reset_index(drop=True)

    extra_counts: dict[str, int] = {}
    extra_datasets: dict[str, set[str]] = {}
    extra_splits: dict[str, set[str]] = {}
    exact_by_cid = exact_unique_df.set_index("canonical_id")
    for rem_cid, (kept_cid, _) in collapsed_map.items():
        rem_row = exact_by_cid.loc[rem_cid]
        extra_counts[kept_cid] = extra_counts.get(kept_cid, 0) + int(rem_row["contributing_count"])
        extra_datasets.setdefault(kept_cid, set()).update(str(rem_row["contributing_datasets"]).split(","))
        extra_splits.setdefault(kept_cid, set()).add(str(rem_row["original_split"]))

    for idx, row in existing_unique.iterrows():
        cid = row["canonical_id"]
        if cid in extra_counts:
            existing_unique.at[idx, "contributing_count"] = int(row["contributing_count"]) + extra_counts[cid]
            ds_set = set(str(row["contributing_datasets"]).split(",")) | extra_datasets[cid]
            existing_unique.at[idx, "contributing_datasets"] = ",".join(sorted(ds_set))
            sp_set = {str(row["original_split"])} | extra_splits[cid]
            existing_unique.at[idx, "original_split"] = min(sp_set, key=lambda s: SPLIT_PRIORITY.get(s, 9))
            if "bangalabarta_2025" in ds_set and existing_unique.at[idx, "lodo_group"] == "smishdetect_only":
                existing_unique.at[idx, "lodo_group"] = "bangalabarta_2025"

    exact_idx_by_cid = {cid: idx for idx, cid in enumerate(exact_unique_df["canonical_id"])}
    review_records: list[dict[str, Any]] = []
    for fam_id, fgrp in exact_unique_df.groupby("family_id", sort=True):
        if len(fgrp) <= 1:
            continue
        idxs = [exact_idx_by_cid[c] for c in fgrp["canonical_id"]]
        sim_mat = cosine_similarity(X_raw[idxs])
        triu = np.triu_indices(len(idxs), k=1)
        min_cos = round(float(sim_mat[triu].min()), 4)
        max_cos = round(float(sim_mat[triu].max()), 4)

        kept_members = []
        collapsed_variants = []
        for r in fgrp.itertuples(index=False):
            if r.canonical_id in collapsed_map:
                kept_cid, reason = collapsed_map[r.canonical_id]
                kept_orig = str(exact_by_cid.loc[kept_cid, "primary_orig_id"])
                collapsed_variants.append(
                    {
                        "removed_exact_canonical_id": r.canonical_id,
                        "removed_orig_id": r.primary_orig_id,
                        "kept_canonical_id": kept_cid,
                        "kept_orig_id": kept_orig,
                        "reason": reason,
                        "orig_split": r.original_split,
                        "language": r.language,
                        "text": r.text,
                    }
                )
            else:
                kept_members.append(
                    {
                        "canonical_id": r.canonical_id,
                        "orig_id": r.primary_orig_id,
                        "orig_split": r.original_split,
                        "language": r.language,
                        "text": r.text,
                    }
                )

        ds_scope = "banglaphish_2026" if (fgrp["primary_dataset"] == "banglaphish_2026").all() else "bb_sd"
        n_cos = sum(1 for v in collapsed_variants if v["reason"] == "cosmetic_punctuation_spacing_or_markdown_variant")
        n_slot = sum(1 for v in collapsed_variants if v["reason"] == "numeric_or_shortcut_slot_clone")
        if ds_scope == "banglaphish_2026":
            rev_cat = "ood_benchmark_kept_with_family_id"
            note = "OOD benchmark family preserved intact; shared family_id links near-duplicate/placeholder variants."
        elif collapsed_variants and len(kept_members) > 1:
            rev_cat = "mixed_collapse_and_distinct_kept"
            note = (
                f"Collapsed {n_cos} cosmetic and {n_slot} slot-swap clones; "
                f"kept {len(kept_members)} materially distinct realizations (cross-script/paraphrase/campaign) with shared family_id."
            )
        elif collapsed_variants:
            rev_cat = "cosmetic_or_slot_collapse_only"
            note = f"Collapsed {n_cos} cosmetic and {n_slot} slot-swap variants to 1 canonical representative."
        else:
            rev_cat = "distinct_variants_kept_only"
            note = "All members are materially distinct (cross-language, distinct wording, or distinct content sharing URL); kept with shared family_id."

        review_records.append(
            {
                "family_id": fam_id,
                "dataset_scope": ds_scope,
                "label_3class_set": sorted(fgrp["label_3class"].unique().tolist()),
                "total_exact_unique_members": int(len(fgrp)),
                "retained_count": int(len(kept_members)),
                "collapsed_count": int(len(collapsed_variants)),
                "review_category": rev_cat,
                "min_pairwise_cosine_raw": min_cos,
                "max_pairwise_cosine_raw": max_cos,
                "kept_members": kept_members,
                "collapsed_variants": collapsed_variants,
                "review_notes": note,
            }
        )

    fam_sizes = existing_unique["family_id"].value_counts().to_dict()
    existing_unique["family_size"] = existing_unique["family_id"].map(fam_sizes).astype(int)

    bp_fams = set(existing_unique.loc[existing_unique["primary_dataset"] == "banglaphish_2026", "family_id"])
    test_fams = set(existing_unique.loc[existing_unique["original_split"] == "test", "family_id"])
    val_fams = set(existing_unique.loc[existing_unique["original_split"] == "validation", "family_id"])

    eval_groups: list[str] = []
    bench_memberships: list[str] = []
    eligible_flags: list[bool] = []
    for row in existing_unique.itertuples(index=False):
        if row.primary_dataset == "banglaphish_2026":
            eval_groups.append("banglaphish_ood")
            bench_memberships.append(
                "banglaphish_2026_ood_sms" if row.source_type == "SMS" else "banglaphish_2026_ood_other"
            )
            eligible_flags.append(False)
        elif row.family_id in bp_fams:
            eval_groups.append("banglaphish_ood_overlap")
            bench_memberships.append("banglaphish_ood_family_overlap")
            eligible_flags.append(False)
        elif row.original_split == "test":
            eval_groups.append("smishdetect_test")
            bench_memberships.append("smishdetect_test_direct")
            eligible_flags.append(False)
        elif row.family_id in test_fams:
            eval_groups.append("smishdetect_test")
            bench_memberships.append("smishdetect_test_family_overlap")
            eligible_flags.append(False)
        elif row.original_split == "validation":
            eval_groups.append("smishdetect_validation")
            bench_memberships.append("smishdetect_validation_direct")
            eligible_flags.append(False)
        elif row.family_id in val_fams:
            eval_groups.append("smishdetect_validation")
            bench_memberships.append("smishdetect_validation_family_overlap")
            eligible_flags.append(False)
        else:
            eval_groups.append("train_pool")
            bench_memberships.append("primary_train_eligible")
            eligible_flags.append(True)

    existing_unique["eval_group"] = eval_groups
    existing_unique["benchmark_membership"] = bench_memberships
    existing_unique["eligible_for_training"] = eligible_flags

    existing_unique["text_masked"] = existing_unique["text"].map(lambda t: mask_shortcuts(t, mask_urls=True, mask_phones=True))
    surf = extract_surface_features(existing_unique, "text")
    for col in surf.columns:
        if col != "script_bucket":
            existing_unique[col] = surf[col].values

    drop_tmp = [c for c in existing_unique.columns if c.startswith("_")]
    existing_unique = existing_unique.drop(columns=drop_tmp).sort_values("canonical_id").reset_index(drop=True)

    eligible_train = existing_unique[existing_unique["eligible_for_training"]].copy().reset_index(drop=True)
    eligible_train["curation_cv_fold"] = -1
    if len(eligible_train) >= 5 and eligible_train["family_id"].nunique() >= 5:
        n_splits = min(5, int(eligible_train["label_3class"].value_counts().min()), int(eligible_train["family_id"].nunique()))
        if n_splits >= 2:
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
            for fold_idx, (_, val_idx) in enumerate(
                sgkf.split(eligible_train, eligible_train["label_3class"].values, eligible_train["family_id"].values)
            ):
                eligible_train.loc[val_idx, "curation_cv_fold"] = fold_idx

    # Build provenance sidecar covering every raw row
    kept_primary_prov_ids = set(existing_unique["primary_provenance_row_id"])
    exact_primary_prov_ids = set(exact_unique_df["primary_provenance_row_id"])
    fam_by_cid = existing_unique.set_index("canonical_id")["family_id"].to_dict()

    prov_canonical_ids: list[str] = []
    prov_family_ids: list[str] = []
    prov_dispositions: list[str] = []
    prov_reasons: list[str] = []

    for row in raw_df.itertuples(index=False):
        status, gid, reason = group_status[row.text_hash_strict]
        if status == "quarantined":
            prov_canonical_ids.append(gid)
            prov_family_ids.append("")
            prov_dispositions.append("quarantined")
            prov_reasons.append(reason)
        elif gid in collapsed_map:
            surviving_cid, collapse_reason = collapsed_map[gid]
            prov_canonical_ids.append(surviving_cid)
            prov_family_ids.append(fam_by_cid[surviving_cid])
            if row.provenance_row_id in exact_primary_prov_ids:
                prov_dispositions.append("near_dup_variant_collapsed")
                prov_reasons.append(collapse_reason)
            else:
                prov_dispositions.append("exact_dup_merged")
                prov_reasons.append(f"exact_dup_of_collapsed_variant:{collapse_reason}")
        else:
            prov_canonical_ids.append(gid)
            prov_family_ids.append(fam_by_cid[gid])
            if row.provenance_row_id in kept_primary_prov_ids:
                prov_dispositions.append("canonical_kept")
                prov_reasons.append("canonical_representative")
            else:
                prov_dispositions.append("exact_dup_merged")
                prov_reasons.append("exact_normalized_duplicate")

    provenance_df = raw_df[
        [
            "provenance_row_id",
            "text_hash_strict",
            "dataset",
            "source_file",
            "source_row_index",
            "orig_id",
            "orig_split",
            "orig_label",
            "mapped_label_3class",
            "source_language_raw",
            "script_bucket",
            "license",
            "source_synthetic_status",
        ]
    ].copy()
    provenance_df.insert(1, "canonical_id", prov_canonical_ids)
    provenance_df.insert(2, "family_id", prov_family_ids)
    provenance_df["disposition"] = prov_dispositions
    provenance_df["disposition_reason"] = prov_reasons

    existing_path = out_dir / "existing_unique.parquet"
    provenance_path = out_dir / "provenance.parquet"
    quarantined_path = out_dir / "quarantined.parquet"
    eligible_path = out_dir / "eligible_train.parquet"
    review_path = out_dir / "near_duplicate_review.jsonl"
    summary_path = out_dir / "curation_summary.json"

    existing_unique.to_parquet(existing_path, index=False)
    provenance_df.to_parquet(provenance_path, index=False)
    quarantined_df.to_parquet(quarantined_path, index=False)
    eligible_train.to_parquet(eligible_path, index=False)
    with open(review_path, "w", encoding="utf-8") as f:
        for rec in review_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    elig_cls_counts = {lbl: int((eligible_train["label_3class"] == lbl).sum()) for lbl in CANONICAL_LABELS}
    target_t = max(elig_cls_counts.values()) if elig_cls_counts else 0
    stage_b_deficits = {lbl: int(max(0, target_t - cnt)) for lbl, cnt in elig_cls_counts.items()}

    # Class-by-language breakdown for consolidated inventory vs eligible training
    inv_lang_cls = pd.crosstab(existing_unique["language"], existing_unique["label_3class"]).to_dict(orient="index")
    elig_lang_cls = pd.crosstab(eligible_train["language"], eligible_train["label_3class"]).to_dict(orient="index")

    summary = {
        "source_inventory": source_inventory,
        "raw_row_counts_by_dataset": {k: int(v) for k, v in raw_df["dataset"].value_counts().items()},
        "raw_total_rows": int(len(raw_df)),
        "disposition_counts_raw": {k: int(v) for k, v in provenance_df["disposition"].value_counts().items()},
        "quarantined_groups_total": int(len(quarantined_df)),
        "quarantined_groups_by_reason": {k: int(v) for k, v in quarantined_df["quarantine_reason"].value_counts().items()},
        "quarantined_raw_rows_by_reason": {
            k: int(v)
            for k, v in provenance_df.loc[provenance_df["disposition"] == "quarantined", "disposition_reason"]
            .value_counts()
            .items()
        },
        "exact_unique_clean_before_collapse": int(len(exact_unique_df)),
        "near_dup_collapsed_variants": {
            "total": int(len(collapsed_map)),
            "by_reason": {k: int(v) for k, v in pd.Series([v[1] for v in collapsed_map.values()]).value_counts().items()}
            if collapsed_map
            else {},
        },
        "near_dup_review_families": {
            "total_multi_member_families": int(len(review_records)),
            "by_category": {
                k: int(v) for k, v in pd.Series([r["review_category"] for r in review_records]).value_counts().items()
            }
            if review_records
            else {},
        },
        "existing_unique_total": int(len(existing_unique)),
        "existing_unique_families": int(existing_unique["family_id"].nunique()),
        "existing_unique_by_class": {lbl: int((existing_unique["label_3class"] == lbl).sum()) for lbl in CANONICAL_LABELS},
        "existing_unique_by_dataset": {k: int(v) for k, v in existing_unique["primary_dataset"].value_counts().items()},
        "existing_unique_by_eval_group": {k: int(v) for k, v in existing_unique["eval_group"].value_counts().items()},
        "existing_unique_by_benchmark_membership": {
            k: int(v) for k, v in existing_unique["benchmark_membership"].value_counts().items()
        },
        "existing_unique_by_language": {k: int(v) for k, v in existing_unique["language"].value_counts().items()},
        "existing_unique_lang_by_class": inv_lang_cls,
        "eligible_train_total": int(len(eligible_train)),
        "eligible_train_families": int(eligible_train["family_id"].nunique()),
        "eligible_train_by_class": elig_cls_counts,
        "eligible_train_by_language": {k: int(v) for k, v in eligible_train["language"].value_counts().items()},
        "eligible_train_lang_by_class": elig_lang_cls,
        "stage_b_target_T": int(target_t),
        "stage_b_deficits": stage_b_deficits,
        "artifacts": {
            "existing_unique": str(existing_path),
            "provenance": str(provenance_path),
            "quarantined": str(quarantined_path),
            "eligible_train": str(eligible_path),
            "near_duplicate_review": str(review_path),
            "summary": str(summary_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
