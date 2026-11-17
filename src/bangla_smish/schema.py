"""Unified schema, dataset loaders, normalization, and shortcut/PII feature extractors."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

CANONICAL_LABELS = ("Normal", "Promotional", "Smishing")

_WS_RE = re.compile(r"\s+")
_BN_CHAR_RE = re.compile(r"[\u0980-\u09FF]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
_URL_RE = re.compile(
    r"(?:"
    r"https?://[^\s<>\"'\]\[()।]+"
    r"|www\.[^\s<>\"'\]\[()।]+"
    r"|(?<![A-Za-z0-9_.@])(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+"
    r"(?:com|net|org|edu|gov|bd|info|xyz|top|tk|ml|ga|cf|io|co|li|ly|me|in|uk|us|biz|club|online|site|store|app|link|pro)"
    r"(?:\.[a-z]{2,3})?(?:/[^\s<>\"'\]\[()।]*)?(?=$|[\s.,;!?)\]}'\"।])"
    r")",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(
    r"(?:"
    r"(?<!\d)\+?(?:880|৮৮০)[-\s]?(?:1|১)[0-9০-৯](?:[-\s]?[0-9০-৯]{8}|[0-9০-৯]{2}[-\s]?[0-9০-৯]{6}|[0-9০-৯]{2}[-\s]?[0-9০-৯]{3}[-\s]?[0-9০-৯]{3}|[-\s]?[0-9০-৯]{4}[-\s]?[0-9০-৯]{4})(?!\d)"
    r"|(?<!\d)[0০][1১][0-9০-৯](?:[-\s]?[0-9০-৯]{8}|[0-9০-৯]{2}[-\s]?[0-9০-৯]{6}|[0-9০-৯]{2}[-\s]?[0-9০-৯]{3}[-\s]?[0-9০-৯]{3})(?!\d)"
    r"|(?<![\d.])(?<!৳)(?<!৳\s)(?<!bdt)(?<!bdt\s)(?<!tk)(?<!tk\s)(?<!tk\.)(?<!tk\.\s)(?<!rs)(?<!rs\s)(?<!rs\.)(?<!rs\.\s)(?<!টাকা)(?<!টাকা\s)"
    r"(?:16|১৬)[0-9০-৯]{3}(?![\d.])(?!\s*(?:tk|taka|bdt|টাকা|/-)(?![A-Za-z0-9_\u0980-\u09FF]))"
    r"|\b\d{10,13}\b"
    r")",
    re.IGNORECASE,
)
_USSD_RE = re.compile(r"\*\d+(?:\*\d+)*#")
_REDACTED_RE = re.compile(r"\[REDACTED_[A-Z_]+\]")

REAL_BD_BRANDS: tuple[str, ...] = (
    "bKash", "Nagad", "Rocket", "Upay", "Grameenphone", "Robi", "Banglalink", "Airtel",
    "Teletalk", "Skitto", "Daraz", "Pathao", "Shwapno", "Chaldal", "Foodpanda", "Pickaboo",
    "Rokomari", "Evaly", "DBBL", "Dutch-Bangla", "Dutch Bangla", "BRAC Bank", "Sonali Bank",
    "Islami Bank", "City Bank", "EBL", "GP", "MyGP", "BL",
    "বিকাশ", "নগদ", "রকেট", "উপায়", "উপায়", "গ্রামীণফোন", "রবি", "বাংলালিংক", "এয়ারটেল",
    "টেলিটক", "দারাজ", "পাঠাও", "স্বপ্ন", "চালডাল", "ফুডপান্ডা", "ডাচ-বাংলা", "ব্র্যাক ব্যাংক",
    "সোনালী ব্যাংক", "ইসলামী ব্যাংক",
)
FICTIONAL_PILOT_BRANDS: tuple[str, ...] = (
    "PurbachalBank", "NodiPay", "BornoTelecom", "TarangaNet", "SuroviMart", "MeghnaHealth",
    "ShikorParcel", "KothaEdu", "DighaExpress", "AalokStore", "AloAcademy", "UrmiTech",
    "KapotakshoCourier", "BonnaKitchen", "DishariSkills", "KironCare", "ChitroBooks",
    "PalliFresh", "ChayaCinema", "SurmaGym", "TistaTours", "NoyonOptics", "RupaliThreads",
    "KallolElectronics", "JhinukJewelry", "PadmaBazar", "HaatGhor", "SagarikaFashion",
    "ShondhaCafe", "ModhumotiClub", "TarangaGreen", "BoshontoBakery", "BikalpoMart",
    "Chayaneer University", "Chayaneer", "Barendra Power Grid", "Barendra",
    "Rupnagar Water Supply", "NeerHaat",
    "Boishakhi Fabrics", "Anondo Books", "Srijon Career Hub", "Borno Training Hub",
    "পূর্বাচলব্যাংক", "নদীপে", "বর্ণটেলিকম", "তরঙ্গনেট", "তরঙ্গগ্রিন", "সুরভীমার্ট",
    "মেঘনাহেলথ", "শিকড়পার্সেল", "শিকড়পার্সেল", "কথাএডু", "দীঘাএক্সপ্রেস", "আলোকস্টোর",
    "আলোঅ্যাকাডেমি", "ঊর্মিটেক", "কপোতাক্ষকুরিয়ার", "কপোতাক্ষ", "বন্যাকিচেন",
    "দিশারীস্কিলস", "কিরণকেয়ার", "চিত্রবুকস", "পল্লীফ্রেশ", "ছায়াসিনেমা", "সুরমাজিম",
    "তিস্তাট্যুরস", "নয়নঅপটিক্স", "রূপালীথ্রেডস", "কল্লোলইলেকট্রনিক্স", "ঝিনুকজুয়েলারি",
    "পদ্মাবাজার", "হাটঘর", "সাগরিকাফ্যাশন", "সন্ধ্যাক্যাফে", "মধুমতীক্লাব",
    "বসন্তবেকারি", "বিকল্পমার্ট", "ছায়ানীড় বিশ্ববিদ্যালয়", "ছায়ানীড়", "ছায়ানীড়",
    "বরেন্দ্র বিদ্যুৎ কেন্দ্র", "নীড়হাট",
    "বৈশাখী ফেব্রিক্স", "আনন্দ বুকস", "সৃজন ক্যারিয়ার হাব", "সৃজন ক্যারিয়ার হাব",
)
DEFAULT_KNOWN_BRANDS: tuple[str, ...] = REAL_BD_BRANDS + FICTIONAL_PILOT_BRANDS


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


_CAMEL_DYNAMIC_BRAND_RE = re.compile(r"\b[A-Z][a-z]{1,}[A-Z][A-Za-z0-9]*\b")
_COLON_LATIN_BRAND_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9&'.-]*(?:\s+[A-Z][A-Za-z0-9&'.-]*){0,2})\s*:")
_COLON_BN_BRAND_RE = re.compile(r"^\s*([\u0980-\u09FF]{2,}(?:\s+[\u0980-\u09FF]{2,}){0,2})\s*:")
_NON_BRAND_CAMEL = frozenset({
    "whatsapp", "youtube", "facebook", "instagram", "linkedin", "tiktok", "telegram",
    "smarttv", "wifi", "trxid", "iphone", "ipad", "android", "onedrive", "dropbox",
})
_COLON_OPENER_STOPWORDS = frozenset({
    "alert", "notice", "update", "reminder", "urgent", "important", "warning", "security",
    "info", "note", "code", "link", "portal", "helpline", "support", "help", "status",
    "offer", "deal", "promo", "news", "breaking", "dear", "hi", "hello", "otp", "pin",
    "congratulations", "attention", "announcement", "advisory",
    "সতর্কতা", "জরুরি", "বিজ্ঞপ্তি", "জরুরি বিজ্ঞপ্তি", "নোটিশ", "নিরাপত্তা সতর্কতা",
    "তথ্য", "অফার", "সুখবর", "খবর", "হ্যালো", "সহায়তা", "সাহায্য", "শুভ", "দৃষ্টি আকর্ষণ",
})
_BRAND_META_KEYS = ("brand_names", "brand_aliases", "entities", "brand", "entity_name")


def extract_dynamic_brands(text: str) -> list[str]:
    """Extract likely unseen brand tokens via CamelCase and colon-opener heuristics."""
    raw = str(text or "")
    found: list[str] = []
    seen: set[str] = set()

    for m in _CAMEL_DYNAMIC_BRAND_RE.finditer(raw):
        tok = m.group(0)
        if tok.lower() not in _NON_BRAND_CAMEL and tok.lower() not in seen:
            found.append(tok)
            seen.add(tok.lower())

    for pat in (_COLON_LATIN_BRAND_RE, _COLON_BN_BRAND_RE):
        m = pat.match(raw)
        if m:
            phrase = normalize_text(m.group(1))
            low = phrase.lower()
            if len(phrase) >= 3 and low not in _COLON_OPENER_STOPWORDS and low not in seen:
                # Strip common trailing generic words like "Notice" / "Alert" / "Desk" if paired with a brand
                cleaned = re.sub(
                    r"\s+(?:alert|notice|update|reminder|security|desk|office|support|খবর|নোটিশ|বিজ্ঞপ্তি|সতর্কতা)$",
                    "",
                    phrase,
                    flags=re.IGNORECASE,
                ).strip()
                target = cleaned if len(cleaned) >= 3 else phrase
                if target.lower() not in _COLON_OPENER_STOPWORDS and target.lower() not in seen:
                    found.append(target)
                    seen.add(target.lower())
    return found


def extract_metadata_brands(records: Iterable[dict[str, Any]]) -> list[str]:
    """Collect brand names and aliases from candidate/accepted row metadata fields."""
    collected: list[str] = []
    seen: set[str] = set()

    def _add(val: Any) -> None:
        if isinstance(val, str):
            tok = normalize_text(val)
            if tok and tok.lower() not in seen:
                collected.append(tok)
                seen.add(tok.lower())
        elif isinstance(val, (list, tuple, set)):
            for item in val:
                _add(item)
        elif isinstance(val, dict):
            for k in ("canonical", "name", "brand", "aliases"):
                if k in val:
                    _add(val[k])

    for rec in records:
        if not isinstance(rec, dict):
            continue
        for key in _BRAND_META_KEYS:
            if key in rec and rec[key] is not None:
                _add(rec[key])
    return collected


def compile_brand_regex(brands: Iterable[str] | None = None) -> re.Pattern[str]:
    """Compile a boundary-aware case-insensitive regex matching brand tokens and Bengali/English inflections."""
    tokens = sorted(
        {normalize_text(b) for b in (DEFAULT_KNOWN_BRANDS if brands is None else brands) if normalize_text(b)},
        key=len,
        reverse=True,
    )
    if not tokens:
        return re.compile(r"(?!x)x")
    parts: list[str] = []
    for tok in tokens:
        esc = re.sub(r"\\ ", r"\\s+", re.escape(tok))
        if _BN_CHAR_RE.search(tok):
            parts.append(rf"(?<![\u0980-\u09FF]){esc}(?:-?(?:ের|ে|তে|কে|র))?(?![\u0980-\u09FF])")
        else:
            parts.append(rf"(?<![A-Za-z0-9_]){esc}(?:['’]s|-?(?:ের|ে|তে|কে|র))?(?![A-Za-z0-9_])")
    return re.compile("|".join(parts), re.IGNORECASE)


_DEFAULT_BRAND_RE = compile_brand_regex(DEFAULT_KNOWN_BRANDS)


def mask_brands(
    text: str,
    brands: Iterable[str] | None = None,
    mode: str = "mask",
    dynamic_fallback: bool = True,
    brand_pattern: re.Pattern[str] | None = None,
) -> str:
    """Mask ('<BRAND>') or strip ('') known and dynamic brand tokens without leaking other tokens."""
    if mode not in ("mask", "strip"):
        raise ValueError(f"Invalid mask_brands mode={mode!r}; expected 'mask' or 'strip'.")
    if brand_pattern is not None:
        pattern = brand_pattern
    else:
        pattern = _DEFAULT_BRAND_RE if brands is None else compile_brand_regex(brands)
    repl = "<BRAND>" if mode == "mask" else " "
    out = pattern.sub(repl, str(text or ""))
    if dynamic_fallback:
        dyn_tokens = extract_dynamic_brands(out)
        if dyn_tokens:
            out = compile_brand_regex(dyn_tokens).sub(repl, out)
    return normalize_text(out)


def mask_text_views(
    text: str,
    brands: Iterable[str] | None = None,
    dynamic_fallback: bool = True,
    brand_pattern: re.Pattern[str] | None = None,
) -> dict[str, str]:
    """Return raw, canonical contact-masked, brand-masked, and brand-stripped views of a message."""
    raw = "" if text is None else str(text)
    canon_masked = mask_shortcuts(raw, mask_urls=True, mask_phones=True)
    return {
        "raw": raw,
        "canon_masked": canon_masked,
        "brand_masked": mask_brands(
            canon_masked, brands=brands, mode="mask", dynamic_fallback=dynamic_fallback, brand_pattern=brand_pattern
        ),
        "brand_stripped": mask_brands(
            canon_masked, brands=brands, mode="strip", dynamic_fallback=dynamic_fallback, brand_pattern=brand_pattern
        ),
    }


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
