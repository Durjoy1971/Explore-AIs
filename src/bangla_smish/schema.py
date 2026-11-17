"""Unified schema, dataset loaders, normalization, and shortcut/PII feature extractors."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd

CANONICAL_LABELS = ("Normal", "Promotional", "Smishing")

_WS_RE = re.compile(r"\s+")
_BN_CHAR_RE = re.compile(r"[\u0980-\u09FF]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
_URL_RE = re.compile(
    r"(?:https?://\S+|www\.\S+|\b[a-z0-9-]+\.(?:com|net|org|li|ly|me|bd|info|xyz|top|tk|ml|ga|cf|io|co)/\S*|\b(?:bit\.ly|cutt\.ly|t\.me|tinyurl\.com|rb\.gy|shorturl\.at)/\S*)",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(
    r"(?:\+?880[-\s]?1[3-9]\d{8}\b|\b01[3-9]\d{8}\b|\b\d{10,13}\b)"
)
_USSD_RE = re.compile(r"\*\d+(?:\*\d+)*#")
_REDACTED_RE = re.compile(r"\[REDACTED_[A-Z_]+\]")


def normalize_text(text: Any) -> str:
    """Collapse whitespace and strip outer spaces while preserving script/punctuation."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _WS_RE.sub(" ", text.strip())


def detect_script_bucket(text: str) -> str:
    """Classify text script into Bengali, Latin, CodeMix, or Other."""
    has_bn = bool(_BN_CHAR_RE.search(text))
    has_lat = bool(_LATIN_CHAR_RE.search(text))
    if has_bn and has_lat:
        return "CodeMix"
    if has_bn:
        return "Bengali"
    if has_lat:
        return "Latin"
    return "Other"


def mask_shortcuts(text: str, mask_urls: bool = True, mask_phones: bool = True) -> str:
    """Replace URLs, phone numbers, USSD codes, and pre-redacted tags with neutral tokens."""
    out = text
    if mask_urls:
        out = _REDACTED_RE.sub(
            lambda m: "<URL>" if "URL" in m.group(0) else ("<PHONE>" if "PHONE" in m.group(0) else "<TAG>"),
            out,
        )
        out = _URL_RE.sub("<URL>", out)
    if mask_phones:
        out = _PHONE_RE.sub("<PHONE>", out)
        out = _USSD_RE.sub("<USSD>", out)
    return normalize_text(out)


def extract_surface_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Compute surface/shortcut indicator columns for auditing and trivial baselines."""
    s = df[text_col].fillna("").astype(str)
    feats = pd.DataFrame(index=df.index)
    feats["char_len"] = s.str.len()
    feats["word_len"] = s.str.split().str.len()
    feats["has_bn"] = s.str.contains(_BN_CHAR_RE, regex=True)
    feats["has_latin"] = s.str.contains(_LATIN_CHAR_RE, regex=True)
    feats["has_url"] = s.str.contains(_URL_RE, regex=True)
    feats["has_phone"] = s.str.contains(_PHONE_RE, regex=True)
    feats["has_ussd"] = s.str.contains(_USSD_RE, regex=True)
    feats["has_redacted_tag"] = s.str.contains(_REDACTED_RE, regex=True)
    feats["script_bucket"] = [detect_script_bucket(t) for t in s]
    return feats


def load_legacy_2024(root: Path) -> pd.DataFrame:
    """Load the repository's original DataSet.csv (binary spam/ham, undocumented provenance)."""
    path = root / "DataSet.csv"
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        for idx, row in enumerate(csv.reader(f)):
            if len(row) < 2:
                continue
            raw_label = row[0].strip().lower()
            text = row[1]
            # Binary legacy schema cannot separate Promotional vs Smishing cleanly
            mapped = "Normal" if raw_label == "ham" else "Spam_Binary"
            rows.append(
                {
                    "dataset": "legacy_2024",
                    "orig_id": f"legacy_{idx}",
                    "orig_split": "unspecified",
                    "orig_label": raw_label,
                    "label_3class": mapped,
                    "label_binary_risk": 0 if raw_label == "ham" else 1,
                    "text": text,
                    "text_norm": normalize_text(text),
                    "source_lang_tag": detect_script_bucket(text),
                    "license": "UNDOCUMENTED",
                    "is_synthetic": "unknown",
                }
            )
    return pd.DataFrame(rows)


def load_bangalabarta_2025(root: Path) -> pd.DataFrame:
    """Load BangalaBarta (Mendeley 10.17632/jfkfbw3gzh.3, CC BY-NC-SA 4.0)."""
    path = root / "data/raw/bangalabarta/BangalaBarta bangla_spam_sms smishing.csv"
    df = pd.read_csv(path)
    label_map = {"normal": "Normal", "promo": "Promotional", "smish": "Smishing"}
    out = pd.DataFrame(
        {
            "dataset": "bangalabarta_2025",
            "orig_id": [f"bb_{i}" for i in range(len(df))],
            "orig_split": "unspecified",
            "orig_label": df["label"].astype(str).str.strip(),
            "label_3class": df["label"].astype(str).str.strip().str.lower().map(label_map),
            "label_binary_risk": df["label"].astype(str).str.strip().str.lower().eq("smish").astype(int),
            "text": df["text"].fillna("").astype(str),
            "text_norm": df["text"].fillna("").astype(str).map(normalize_text),
            "source_lang_tag": df["text"].fillna("").astype(str).map(detect_script_bucket),
            "license": "CC BY-NC-SA 4.0",
            "is_synthetic": "curated_real_mixed",
        }
    )
    return out


def load_smishdetect_2026(root: Path) -> pd.DataFrame:
    """Load shariul-islam/bengali-sms-smishing-dataset (HF, MIT license)."""
    parts = []
    label_map = {"normal": "Normal", "promo": "Promotional", "smish": "Smishing"}
    for split in ("train", "validation", "test"):
        p = root / f"data/raw/smishdetect_hf/data/{split}-00000-of-00001.parquet"
        df = pd.read_parquet(p)
        part = pd.DataFrame(
            {
                "dataset": "smishdetect_2026",
                "orig_id": [f"sd_{split}_{i}" for i in range(len(df))],
                "orig_split": split,
                "orig_label": df["label"].astype(str).str.strip(),
                "label_3class": df["label"].astype(str).str.strip().str.lower().map(label_map),
                "label_binary_risk": df["label"].astype(str).str.strip().str.lower().eq("smish").astype(int),
                "text": df["text"].fillna("").astype(str),
                "text_norm": df["text"].fillna("").astype(str).map(normalize_text),
                "source_lang_tag": df["source"].astype(str).str.strip(),
                "license": "MIT",
                "is_synthetic": "augmented_multilingual",
            }
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def load_banglaphish_2026(root: Path) -> pd.DataFrame:
    """Load BanglaPhish-2026 (GitHub/Zenodo 10.5281/zenodo.20501760, CC BY-NC 4.0)."""
    path = root / "data/raw/banglaphish2026/data/BanglaPhish-2026_v1.0_MASTER.csv"
    df = pd.read_csv(path)
    # Note: domain 'Promotional/Marketing Messages' (200 rows) is under legitimate in source
    is_promo = df["domain"].astype(str).str.contains("Promotional", case=False, na=False)
    label_3class = []
    for lbl, promo in zip(df["label"].astype(str).str.lower(), is_promo):
        if lbl == "scam":
            label_3class.append("Smishing")
        elif promo:
            label_3class.append("Promotional")
        else:
            label_3class.append("Normal")
    out = pd.DataFrame(
        {
            "dataset": "banglaphish_2026",
            "orig_id": df["sample_id"].astype(str),
            "orig_split": df["benchmark_split"].astype(str),
            "orig_label": df["label"].astype(str),
            "label_3class": label_3class,
            "label_binary_risk": df["label"].astype(str).str.lower().eq("scam").astype(int),
            "text": df["text_bn"].fillna("").astype(str),
            "text_norm": df["text_bn"].fillna("").astype(str).map(normalize_text),
            "source_lang_tag": df["script"].astype(str) + "|" + df["source_type"].astype(str),
            "license": "CC BY-NC 4.0",
            "is_synthetic": "synthetic",
        }
    )
    return out


def load_all_datasets(root: str | Path = ".") -> dict[str, pd.DataFrame]:
    """Load all 4 corpora into a dictionary keyed by dataset name."""
    root_path = Path(root)
    return {
        "legacy_2024": load_legacy_2024(root_path),
        "bangalabarta_2025": load_bangalabarta_2025(root_path),
        "smishdetect_2026": load_smishdetect_2026(root_path),
        "banglaphish_2026": load_banglaphish_2026(root_path),
    }
