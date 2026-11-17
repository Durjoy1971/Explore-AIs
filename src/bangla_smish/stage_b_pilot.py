"""Stage B Pilot isolated screening, anti-leakage checks, and surface diagnostics."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .codemix_diversity import analyze_codemix_row, collect_brand_aliases, evaluate_codemix_and_entity_diversity
from .curation import slot_template_key, slot_template_text, strong_normalize_text
from .curation_v2 import infer_scenario_domain
from .metrics import evaluate_predictions
from .schema import (
    CANONICAL_LABELS,
    DEFAULT_KNOWN_BRANDS,
    REAL_BD_BRANDS,
    _PHONE_RE as _SCHEMA_PHONE_RE,
    _URL_RE as _SCHEMA_URL_RE,
    compile_brand_regex,
    detect_script_bucket,
    extract_metadata_brands,
    extract_surface_features,
    mask_text_views,
    normalize_text,
)

TEXT_VIEW_NAMES: tuple[str, ...] = ("raw", "canon_masked", "brand_masked", "brand_stripped")
_SURFACE_8_COLS: list[str] = [
    "char_len", "word_len", "has_bn", "has_latin", "has_url", "has_phone", "has_ussd", "has_redacted_tag",
]
_LABEL_TO_IDX: dict[str, int] = {lbl: idx for idx, lbl in enumerate(CANONICAL_LABELS)}
_REAL_BRAND_SWAP_RE = compile_brand_regex(REAL_BD_BRANDS)

REQUIRED_FIELDS = (
    "id",
    "text",
    "label_3class",
    "language",
    "scenario",
    "domain",
    "length_band",
    "formality",
    "has_link",
    "has_urgency",
    "is_synthetic",
    "source_dataset",
    "source_row_ids",
    "family_id",
    "generator_model",
    "prompt_version",
    "generation_batch",
    "created_at",
    "review_status",
    "label_rationale",
)

TARGET_PILOT_QUOTAS: dict[tuple[str, str], int] = {
    ("Normal", "Bengali"): 10,
    ("Normal", "Banglish"): 8,
    ("Normal", "English"): 7,
    ("Normal", "CodeMix"): 14,
    ("Promotional", "Bengali"): 65,
    ("Promotional", "Banglish"): 115,
    ("Promotional", "English"): 21,
    ("Promotional", "CodeMix"): 60,
}

RESERVED_DOMAINS = ("example.com", "example.org", "example.net", "example.edu.bd")
CANONICAL_LENGTH_BANDS = ("short(<50)", "medium(50-120)", "long(>120)")
_LENGTH_BAND_ALIASES = {
    "short": "short(<50)",
    "short(<50)": "short(<50)",
    "medium": "medium(50-120)",
    "medium(50-120)": "medium(50-120)",
    "long": "long(>120)",
    "long(>120)": "long(>120)",
}

_REAL_BRAND_RE = re.compile(
    r"\b(?:bkash|nagad|rocket|upay|grameenphone|robi|banglalink|airtel|teletalk|skitto|"
    r"daraz|pathao|shwapno|chaldal|foodpanda|pickaboo|rokomari|evaly|dbbl|dutch.?bangla|"
    r"brac\s+bank|sonali\s+bank|islami\s+bank|gp)\b",
    re.IGNORECASE,
)
_ASCII_PHONE_RE = re.compile(
    r"(?:"
    r"(?<!\d)\+?(?:880|৮৮০)[-\s]?(?:1|১)[0-9০-৯](?:[-\s]?[0-9০-৯]{8}|[0-9০-৯]{2}[-\s]?[0-9০-৯]{6}|[0-9০-৯]{2}[-\s]?[0-9০-৯]{3}[-\s]?[0-9০-৯]{3})(?!\d)"
    r"|(?<!\d)[0০][1১][0-9০-৯](?:[-\s]?[0-9০-৯]{8}|[0-9০-৯]{2}[-\s]?[0-9০-৯]{6}|[0-9০-৯]{2}[-\s]?[0-9০-৯]{3}[-\s]?[0-9০-৯]{3})(?!\d)"
    r"|(?<![\d.])(?<!৳)(?<!৳\s)(?<!bdt)(?<!bdt\s)(?<!tk)(?<!tk\s)(?<!tk\.)(?<!tk\.\s)(?<!rs)(?<!rs\s)(?<!rs\.)(?<!rs\.\s)(?<!টাকা)(?<!টাকা\s)"
    r"(?:16|১৬)[0-9০-৯]{3}(?![\d.])(?!\s*(?:tk|taka|bdt|টাকা|/-)(?![A-Za-z0-9_\u0980-\u09FF]))"
    r")",
    re.IGNORECASE,
)
_BN_TO_ASCII_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def compute_length_band(char_len: int) -> str:
    """Return the canonical length_band label for a character length."""
    if char_len < 50:
        return "short(<50)"
    if char_len <= 120:
        return "medium(50-120)"
    return "long(>120)"


def normalize_length_band(value: Any) -> str:
    """Normalize shorthand length_band aliases ('short'/'medium'/'long') to canonical labels."""
    key = str(value or "").strip()
    return _LENGTH_BAND_ALIASES.get(key, key)

_NORMAL_OFFER_RE = re.compile(
    r"\b(?:discount|cashback|mega sale|flash sale|buy\s*1\s*get\s*1|promo\s*code|coupon)\b"
    r"|%\s*(?:off|discount|ছাড়|ছাড়|char|chhar)|ক্যাশব্যাক|ডিসকাউন্ট",
    re.IGNORECASE,
)
_EXTRACTION_RE = re.compile(
    r"(?<!never\s)(?<!not\s)(?<!don't\s)(?<!don’t\s)\b(?:send|share|enter|submit|provide|reply)\s+(?:your\s+)?(?:otp|pin|password|cvv|nid)\b"
    r"|পিন\s*(?:দিন|পাঠান)|ওটিপি\s*(?:দিন|পাঠান)|আপনার\s+পাসওয়ার্ড|লটারি\s+জিতেছেন|\blottery\s+winner\b",
    re.IGNORECASE,
)
_NEGATED_SHARE_RE = re.compile(
    r"\b(?:never|do\s+not|don['’]t)\s+(?:ever\s+)?(?:share|give|send|disclose)\b|শেয়ার\s+করবেন\s+না|কাউকে\s+(?:দিবেন|দেবেন|জানাবেন)\s+না",
    re.IGNORECASE,
)

_BANGLISH_MARKERS = {
    "apni", "apnar", "apnake", "amader", "amar", "tumi", "tomar", "tomake", "korun", "koro",
    "korte", "korben", "korle", "paben", "pan", "niben", "nao", "nin", "dekhun", "dekho",
    "shob", "sob", "ekhon", "ekhoni", "akhon", "ajke", "ajkei", "char", "chhar", "mulle", "dame",
    "theke", "jonno", "jonyo", "shathe", "sathe", "hobe", "hoyeche", "hoise", "hoy", "ache",
    "achhe", "bistarito", "janun", "jante", "kinun", "niye", "diye", "ashun", "asun", "chaile",
    "chan", "parben", "paro", "ebong", "kintu", "jodi", "tahole", "porjonto", "shudhu", "sudhu",
    "matro", "notun", "khub", "onek", "beshi", "kom", "komate", "bhalo", "valo", "shuvo", "shubho",
    "upolokkhe", "shesh", "sesh", "shuru", "suru", "thakbe", "bondho", "chalu", "cholche", "pathan",
    "pathano", "pouchabe", "dorkar", "lagbe", "lagche", "khuda", "ekta", "ei", "oi", "taratari", "druto",
    "agamee", "agami", "mash", "din", "ghonta", "bikel", "shokal", "raat", "somoy", "shomoy",
    "sobcheye", "shobcheye", "bishesh", "bisesh", "darun", "mojar", "majar", "shera", "sera",
    "khabar", "bashay", "basay", "ghore", "ghorer", "bari", "taka", "tk", "bochor", "pohela", "boishakh",
    "katun", "katate", "chhutir", "bolte", "shikhe", "niyog", "khoroch", "bidyut", "ba", "er",
}
_ENGLISH_FUNCTION_WORDS = {
    "the", "is", "are", "was", "were", "your", "you", "we", "our", "for", "with", "this",
    "that", "from", "have", "has", "will", "get", "enjoy", "shop", "visit", "order", "valid",
    "today", "now", "only", "until", "before", "after", "at", "on", "in", "to", "and", "or",
    "can", "join", "apply", "book", "receive", "grab", "save", "install", "submit", "off",
    "free", "please", "thank", "dear", "customer",
}
_URL_STRIP_RE = re.compile(r"https?://\S+|www\.\S+|\b[a-z0-9.-]+\.[a-z]{2,}\S*", re.IGNORECASE)


def _strip_urls_and_normalize_digits(text: str) -> str:
    return _URL_STRIP_RE.sub(" ", text).translate(_BN_TO_ASCII_DIGITS)


def _check_domain_and_phone(text: str) -> list[str]:
    reasons: list[str] = []
    if _REAL_BRAND_RE.search(text):
        reasons.append("real_bd_brand_mention")
    for m in _SCHEMA_URL_RE.finditer(text):
        raw_url = m.group(0)
        dom = re.sub(r"^https?://", "", raw_url, flags=re.IGNORECASE).split("/")[0].split(":")[0].lower().rstrip(".")
        if dom.startswith("www."):
            dom = dom[4:]
        if dom and not any(dom == rd or dom.endswith("." + rd) for rd in RESERVED_DOMAINS):
            reasons.append(f"non_reserved_domain:{dom}")
    ascii_text = text.translate(_BN_TO_ASCII_DIGITS)
    for m in _ASCII_PHONE_RE.finditer(ascii_text):
        digits = re.sub(r"\D", "", m.group(0).translate(_BN_TO_ASCII_DIGITS))
        if digits.startswith("880"):
            digits = "0" + digits[3:]
        if not digits.startswith("017000000"):
            reasons.append("non_fictional_phone_number")
    return reasons


def _check_language_and_boundary(text: str, label: str, language: str) -> list[str]:
    reasons: list[str] = []
    if label == "Normal":
        if _NORMAL_OFFER_RE.search(text):
            reasons.append("normal_contains_commercial_offer_cue")
        if _EXTRACTION_RE.search(text) and not _NEGATED_SHARE_RE.search(text):
            reasons.append("normal_contains_extraction_cue")
    elif label == "Promotional":
        if _EXTRACTION_RE.search(text) and not _NEGATED_SHARE_RE.search(text):
            reasons.append("promotional_contains_extraction_cue")
    else:
        reasons.append(f"disallowed_pilot_label:{label}")

    stripped = _strip_urls_and_normalize_digits(text)
    sb = detect_script_bucket(stripped)
    lat_tokens = re.findall(r"[A-Za-z]+", stripped.lower())
    bn_hits = [t for t in lat_tokens if t in _BANGLISH_MARKERS]
    en_hits = [t for t in lat_tokens if t in _ENGLISH_FUNCTION_WORDS]

    if language == "Bengali":
        if not any("\u0980" <= ch <= "\u09ff" for ch in stripped):
            reasons.append("bengali_missing_bengali_script")
    elif language == "Banglish":
        if sb != "Latin":
            reasons.append(f"banglish_unexpected_script:{sb}")
        if not bn_hits and len(en_hits) >= 2:
            reasons.append("banglish_appears_pure_english")
    elif language == "English":
        if sb != "Latin":
            reasons.append(f"english_unexpected_script:{sb}")
        if len(bn_hits) >= 2:
            reasons.append("english_contains_banglish_markers")
    elif language == "CodeMix":
        has_bn = any("\u0980" <= ch <= "\u09ff" for ch in stripped)
        if has_bn:
            if len(lat_tokens) < 2:
                reasons.append("codemix_insufficient_latin_switching")
        else:
            if not bn_hits or len(en_hits) < 1:
                reasons.append("codemix_latin_lacks_bilingual_switching")
    else:
        reasons.append(f"invalid_language:{language}")

    return reasons


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read UTF-8 JSONL file into list of dicts."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write list of dicts as UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def compute_surface_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    """Run surface-only and length-only 5-fold grouped CV diagnostics matching Stage A/B guide."""
    feats = extract_surface_features(df, "text")
    surf_cols = ["char_len", "word_len", "has_bn", "has_latin", "has_url", "has_phone", "has_ussd", "has_redacted_tag"]
    X = feats[surf_cols].to_numpy(dtype=float)
    y = df["label_3class"].astype(str).to_numpy()
    groups = df["family_id"].astype(str).to_numpy()

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    preds = cross_val_predict(clf, X, y, cv=cv, groups=groups)
    acc = round(float(accuracy_score(y, preds) * 100.0), 2)
    mf1 = round(float(f1_score(y, preds, average="macro") * 100.0), 2)

    y_bin = (df["label_3class"] == "Smishing").astype(int).to_numpy()
    len_auc = float(roc_auc_score(y_bin, feats["char_len"].to_numpy(dtype=float)))
    if len_auc < 0.5:
        len_auc = 1.0 - len_auc

    # Normal vs Promotional length/URL separability (directly relevant to pilot classes)
    np_mask = df["label_3class"].isin(["Normal", "Promotional"]).to_numpy()
    y_np = (df.loc[np_mask, "label_3class"] == "Promotional").astype(int).to_numpy()
    np_len_auc = float(roc_auc_score(y_np, feats.loc[np_mask, "char_len"].to_numpy(dtype=float)))
    if np_len_auc < 0.5:
        np_len_auc = 1.0 - np_len_auc

    per_class = {}
    work = df.reset_index(drop=True)
    feats_r = feats.reset_index(drop=True)
    for lbl in CANONICAL_LABELS:
        m = work["label_3class"] == lbl
        per_class[lbl] = {
            "count": int(m.sum()),
            "median_char_len": float(feats_r.loc[m, "char_len"].median()),
            "mean_char_len": round(float(feats_r.loc[m, "char_len"].mean()), 1),
            "pct_has_url": round(float(feats_r.loc[m, "has_url"].mean() * 100.0), 2),
            "pct_has_phone": round(float(feats_r.loc[m, "has_phone"].mean() * 100.0), 2),
        }

    return {
        "n_rows": int(len(df)),
        "surface_only_5fold_acc_pct": acc,
        "surface_only_5fold_macro_f1_pct": mf1,
        "length_only_smishing_roc_auc": round(len_auc, 4),
        "length_only_promo_vs_normal_roc_auc": round(np_len_auc, 4),
        "per_class_surface": per_class,
    }


def _make_tfidf_lr() -> Pipeline:
    """Build unified word (1-2) + char_wb (3-5) TF-IDF + balanced LogisticRegression pipeline."""
    feats = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
        ]
    )
    return Pipeline(
        [
            ("tfidf", feats),
            ("clf", LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced")),
        ]
    )


def _build_text_views_map(texts: list[str], brands: Iterable[str] | None = None) -> dict[str, list[str]]:
    """Compute all 4 text views (raw, canon_masked, brand_masked, brand_stripped) for a list of texts."""
    views: dict[str, list[str]] = {name: [] for name in TEXT_VIEW_NAMES}
    brand_pattern = compile_brand_regex(brands) if brands is not None else None
    for t in texts:
        vm = mask_text_views(t, brands=brands, brand_pattern=brand_pattern)
        for name in TEXT_VIEW_NAMES:
            views[name].append(vm[name])
    return views


def compute_3class_probe(
    df: pd.DataFrame,
    text_views: dict[str, list[str]] | None = None,
    n_splits: int = 5,
    brands: Iterable[str] | None = None,
    return_full_models: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Pipeline]]:
    """Run grouped 5-fold 3-class TF-IDF+LR probes across raw, contact-masked, brand-masked, and brand-stripped views."""
    texts_raw = df["text"].fillna("").astype(str).tolist()
    views = text_views if text_views is not None else _build_text_views_map(texts_raw, brands=brands)
    y = df["label_3class"].map(_LABEL_TO_IDX).to_numpy(dtype=int)
    groups = df["family_id"].astype(str).to_numpy()
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results: dict[str, Any] = {}
    full_models: dict[str, Pipeline] = {}
    smish_idx = _LABEL_TO_IDX["Smishing"]

    for view_name in TEXT_VIEW_NAMES:
        view_texts = views[view_name]
        oof_probs = np.zeros((len(df), len(CANONICAL_LABELS)), dtype=float)
        for tr_idx, te_idx in cv.split(view_texts, y, groups):
            clf = _make_tfidf_lr()
            clf.fit([view_texts[i] for i in tr_idx], y[tr_idx])
            oof_probs[te_idx] = clf.predict_proba([view_texts[i] for i in te_idx])

        metrics = evaluate_predictions(y, oof_probs, label_names=CANONICAL_LABELS, n_boot=200)
        full = _make_tfidf_lr().fit(view_texts, y)
        full_models[view_name] = full

        feat_names = full.named_steps["tfidf"].get_feature_names_out()
        smish_coefs = full.named_steps["clf"].coef_[smish_idx]
        neg_idx = np.argsort(smish_coefs)[:8]
        results[view_name] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "smishing_recall_argmax": metrics["smishing_recall_argmax"],
            "normal_as_smishing_fpr_argmax": metrics["normal_as_smishing_fpr_argmax"],
            "smishing_recall_at_1pct_fpr": metrics["smishing_recall_at_1pct_fpr"],
            "top_negative_smishing_features": [
                [str(feat_names[i]), round(float(smish_coefs[i]), 3)] for i in neg_idx
            ],
        }

    if return_full_models:
        return results, full_models
    return results


def _run_binary_synthetic_probe_subset(
    sub_df: pd.DataFrame,
    views: dict[str, list[str]],
    n_splits: int = 5,
    sample_weights: np.ndarray | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run grouped binary is_synthetic probes across all 4 text views and surface_8feat."""
    y = sub_df["is_synthetic_bin"].to_numpy(dtype=int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    groups = sub_df["family_id"].astype(str).to_numpy()
    pos_groups = len(set(groups[y == 1]))
    neg_groups = len(set(groups[y == 0]))
    if n_pos < 2 or n_neg < 2 or pos_groups < 2 or neg_groups < 2:
        res: dict[str, Any] = {
            "skipped": "insufficient_synthetic_or_real_rows_or_families",
            "n_synth": n_pos,
            "n_real": n_neg,
        }
        if extra_meta:
            res.update(extra_meta)
        return res

    eff_splits = min(n_splits, pos_groups, neg_groups)
    cv = StratifiedGroupKFold(n_splits=eff_splits, shuffle=True, random_state=42)
    out: dict[str, Any] = {
        "n_synth": n_pos,
        "n_real": n_neg,
        "n_splits": eff_splits,
        "sample_weighted": sample_weights is not None,
    }
    if extra_meta:
        out.update(extra_meta)

    for view_name in TEXT_VIEW_NAMES:
        view_texts = views[view_name]
        oof = np.zeros(len(sub_df), dtype=float)
        for tr_idx, te_idx in cv.split(view_texts, y, groups):
            clf = _make_tfidf_lr()
            if sample_weights is not None:
                clf.fit([view_texts[i] for i in tr_idx], y[tr_idx], clf__sample_weight=sample_weights[tr_idx])
            else:
                clf.fit([view_texts[i] for i in tr_idx], y[tr_idx])
            oof[te_idx] = clf.predict_proba([view_texts[i] for i in te_idx])[:, 1]
        preds = (oof >= 0.5).astype(int)
        full = _make_tfidf_lr()
        if sample_weights is not None:
            full.fit(view_texts, y, clf__sample_weight=sample_weights)
        else:
            full.fit(view_texts, y)
        feat_names = full.named_steps["tfidf"].get_feature_names_out()
        coefs = full.named_steps["clf"].coef_[0]
        top_idx = np.argsort(coefs)[-8:][::-1]
        out[view_name] = {
            "roc_auc": round(float(roc_auc_score(y, oof)), 4),
            "accuracy_pct": round(float(accuracy_score(y, preds) * 100.0), 2),
            "macro_f1_pct": round(float(f1_score(y, preds, average="macro") * 100.0), 2),
            "top_positive_features": [[str(feat_names[i]), round(float(coefs[i]), 3)] for i in top_idx],
        }

    surf_X = extract_surface_features(sub_df, "text")[_SURFACE_8_COLS].to_numpy(dtype=float)
    oof_s = np.zeros(len(sub_df), dtype=float)
    for tr_idx, te_idx in cv.split(surf_X, y, groups):
        clf_s = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        if sample_weights is not None:
            clf_s.fit(surf_X[tr_idx], y[tr_idx], logisticregression__sample_weight=sample_weights[tr_idx])
        else:
            clf_s.fit(surf_X[tr_idx], y[tr_idx])
        oof_s[te_idx] = clf_s.predict_proba(surf_X[te_idx])[:, 1]
    preds_s = (oof_s >= 0.5).astype(int)
    full_s = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    if sample_weights is not None:
        full_s.fit(surf_X, y, logisticregression__sample_weight=sample_weights)
    else:
        full_s.fit(surf_X, y)
    coefs_s = full_s.named_steps["logisticregression"].coef_[0]
    top_s_idx = np.argsort(coefs_s)[-8:][::-1]
    out["surface_8feat"] = {
        "roc_auc": round(float(roc_auc_score(y, oof_s)), 4),
        "accuracy_pct": round(float(accuracy_score(y, preds_s) * 100.0), 2),
        "macro_f1_pct": round(float(f1_score(y, preds_s, average="macro") * 100.0), 2),
        "top_positive_features": [[_SURFACE_8_COLS[i], round(float(coefs_s[i]), 3)] for i in top_s_idx],
    }
    return out


def _slice_views(views: dict[str, list[str]], indices: np.ndarray) -> dict[str, list[str]]:
    return {name: [views[name][int(i)] for i in indices] for name in TEXT_VIEW_NAMES}


def _run_stratum_controlled_probe(
    all_df: pd.DataFrame,
    all_views: dict[str, list[str]],
    target_label: str,
    stratum_cols: tuple[str, ...],
    stratum_type: str,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Evaluate synthetic-vs-real within a single class controlling for stratum_cols via shared strata + balanced weights."""
    cls_mask = (all_df["label_3class"] == target_label).to_numpy()
    cls_indices = np.where(cls_mask)[0]
    cls_df = all_df.iloc[cls_indices].reset_index(drop=True)
    total_real = int((cls_df["is_synthetic_bin"] == 0).sum())
    total_synth = int((cls_df["is_synthetic_bin"] == 1).sum())

    if total_real == 0 or total_synth == 0:
        return {
            "skipped": "no_synthetic_or_real_rows_for_class",
            "stratum_type": stratum_type,
            "matched_strata_count": 0,
            "matched_real_count": 0,
            "total_real_count": total_real,
            "matched_synth_count": 0,
            "total_synth_count": total_synth,
        }

    strata = [tuple(row) for row in cls_df[list(stratum_cols)].itertuples(index=False, name=None)]
    real_counts: dict[tuple[Any, ...], int] = {}
    synth_counts: dict[tuple[Any, ...], int] = {}
    for st, is_syn in zip(strata, cls_df["is_synthetic_bin"]):
        if int(is_syn) == 1:
            synth_counts[st] = synth_counts.get(st, 0) + 1
        else:
            real_counts[st] = real_counts.get(st, 0) + 1

    shared_strata = {st for st in real_counts if st in synth_counts}
    keep_local = np.array([st in shared_strata for st in strata], dtype=bool)
    matched_local_idx = np.where(keep_local)[0]
    matched_global_idx = cls_indices[matched_local_idx]

    matched_df = all_df.iloc[matched_global_idx].reset_index(drop=True)
    matched_strata = [strata[int(i)] for i in matched_local_idx]
    weights = np.array(
        [
            1.0 / (synth_counts[st] if int(is_syn) == 1 else real_counts[st])
            for st, is_syn in zip(matched_strata, matched_df["is_synthetic_bin"])
        ],
        dtype=float,
    )
    meta = {
        "stratum_type": stratum_type,
        "matched_strata_count": int(len(shared_strata)),
        "matched_real_count": int((matched_df["is_synthetic_bin"] == 0).sum()),
        "total_real_count": total_real,
        "matched_synth_count": int((matched_df["is_synthetic_bin"] == 1).sum()),
        "total_synth_count": total_synth,
    }
    return _run_binary_synthetic_probe_subset(
        matched_df,
        _slice_views(all_views, matched_global_idx),
        n_splits=n_splits,
        sample_weights=weights,
        extra_meta=meta,
    )


def _effective_domain(df: pd.DataFrame) -> pd.Series:
    texts = df["text"].fillna("").astype(str)
    if "domain" not in df.columns:
        return texts.map(infer_scenario_domain)
    doms = df["domain"].fillna("").astype(str).str.strip()
    need_infer = doms.isin(["", "unspecified", "unknown", "nan", "None"])
    if not need_infer.any():
        return doms
    inferred = texts[need_infer].map(infer_scenario_domain)
    out = doms.copy()
    out.loc[need_infer] = inferred
    return out


def compute_synthetic_origin_probes(
    eligible_df: pd.DataFrame,
    accepted_df: pd.DataFrame,
    n_splits: int = 5,
    brands: Iterable[str] | None = None,
    all_views: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Quantify synthetic-origin separability overall, vs Smishing, and class-matched (raw + stratum-controlled)."""
    if accepted_df.empty:
        return {"skipped": "no_accepted_synthetic_rows"}

    real_cols = [c for c in ("text", "label_3class", "language", "family_id", "domain") if c in eligible_df.columns]
    real_part = eligible_df[real_cols].copy()
    if "language" not in real_part.columns:
        real_part["language"] = real_part["text"].fillna("").astype(str).map(detect_script_bucket)
    real_part["domain_eff"] = _effective_domain(real_part)
    real_part["length_band_eff"] = real_part["text"].fillna("").astype(str).str.len().map(compute_length_band)
    real_part["is_synthetic_bin"] = 0

    syn_cols = [c for c in ("text", "label_3class", "language", "family_id", "domain") if c in accepted_df.columns]
    syn_part = accepted_df[syn_cols].copy()
    if "language" not in syn_part.columns:
        syn_part["language"] = syn_part["text"].fillna("").astype(str).map(detect_script_bucket)
    syn_part["domain_eff"] = _effective_domain(syn_part)
    syn_part["length_band_eff"] = syn_part["text"].fillna("").astype(str).str.len().map(compute_length_band)
    syn_part["is_synthetic_bin"] = 1

    keep_cols = ["text", "label_3class", "language", "family_id", "domain_eff", "length_band_eff", "is_synthetic_bin"]
    all_df = pd.concat([real_part[keep_cols], syn_part[keep_cols]], ignore_index=True)
    views = (
        all_views
        if all_views is not None
        else _build_text_views_map(all_df["text"].fillna("").astype(str).tolist(), brands=brands)
    )

    smish_idx = np.where(((all_df["is_synthetic_bin"] == 1) | (all_df["label_3class"] == "Smishing")).to_numpy())[0]
    norm_idx = np.where((all_df["label_3class"] == "Normal").to_numpy())[0]
    promo_idx = np.where((all_df["label_3class"] == "Promotional").to_numpy())[0]

    return {
        "all_vs_real": _run_binary_synthetic_probe_subset(all_df, views, n_splits=n_splits),
        "synth_vs_real_smishing": _run_binary_synthetic_probe_subset(
            all_df.iloc[smish_idx].reset_index(drop=True), _slice_views(views, smish_idx), n_splits=n_splits
        ),
        "synthetic_normal_vs_real_normal": _run_binary_synthetic_probe_subset(
            all_df.iloc[norm_idx].reset_index(drop=True), _slice_views(views, norm_idx), n_splits=n_splits
        ),
        "synthetic_normal_vs_real_normal_stratum_3way": _run_stratum_controlled_probe(
            all_df,
            views,
            target_label="Normal",
            stratum_cols=("language", "domain_eff", "length_band_eff"),
            stratum_type="language_domain_length_band",
            n_splits=n_splits,
        ),
        "synthetic_normal_vs_real_normal_stratum_2way": _run_stratum_controlled_probe(
            all_df,
            views,
            target_label="Normal",
            stratum_cols=("language", "length_band_eff"),
            stratum_type="language_length_band",
            n_splits=n_splits,
        ),
        "synthetic_promo_vs_real_promo": _run_binary_synthetic_probe_subset(
            all_df.iloc[promo_idx].reset_index(drop=True), _slice_views(views, promo_idx), n_splits=n_splits
        ),
        "synthetic_promo_vs_real_promo_stratum_3way": _run_stratum_controlled_probe(
            all_df,
            views,
            target_label="Promotional",
            stratum_cols=("language", "domain_eff", "length_band_eff"),
            stratum_type="language_domain_length_band",
            n_splits=n_splits,
        ),
        "synthetic_promo_vs_real_promo_stratum_2way": _run_stratum_controlled_probe(
            all_df,
            views,
            target_label="Promotional",
            stratum_cols=("language", "length_band_eff"),
            stratum_type="language_length_band",
            n_splits=n_splits,
        ),
    }


def compute_counterfactual_probes(
    eligible_df: pd.DataFrame,
    full_models: dict[str, Pipeline],
    brands: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Probe whether swapping example.com, 01700-000011, and PurbachalBank flips real Smishing predictions."""
    smish_df = eligible_df[eligible_df["label_3class"] == "Smishing"].copy()
    texts = smish_df["text"].fillna("").astype(str)
    contact_mask = texts.str.contains(_SCHEMA_URL_RE, regex=True) | texts.str.contains(_SCHEMA_PHONE_RE, regex=True)
    probed = texts[contact_mask].tolist()
    if not probed:
        return {"n_probed": 0, "skipped": "no_smishing_contact_rows"}

    smish_idx = _LABEL_TO_IDX["Smishing"]

    swap_contact: list[str] = []
    swap_contact_brand: list[str] = []
    for t in probed:
        c_txt = _SCHEMA_URL_RE.sub("https://example.com/offer", t)
        c_txt = _SCHEMA_PHONE_RE.sub("01700-000011", c_txt)
        cb_txt = _REAL_BRAND_SWAP_RE.sub("PurbachalBank", c_txt)
        swap_contact.append(c_txt)
        swap_contact_brand.append(cb_txt)

    orig_views = _build_text_views_map(probed, brands=brands)
    contact_views = _build_text_views_map(swap_contact, brands=brands)
    brand_views = _build_text_views_map(swap_contact_brand, brands=brands)

    results: dict[str, Any] = {"n_probed": int(len(probed)), "views": {}}
    for view_name in TEXT_VIEW_NAMES:
        clf = full_models[view_name]
        p_orig = clf.predict_proba(orig_views[view_name])
        p_c = clf.predict_proba(contact_views[view_name])
        p_cb = clf.predict_proba(brand_views[view_name])

        pred_orig = float((p_orig.argmax(1) == smish_idx).mean() * 100.0)
        pred_c = float((p_c.argmax(1) == smish_idx).mean() * 100.0)
        pred_cb = float((p_cb.argmax(1) == smish_idx).mean() * 100.0)

        mean_orig = float(p_orig[:, smish_idx].mean())
        mean_c = float(p_c[:, smish_idx].mean())
        mean_cb = float(p_cb[:, smish_idx].mean())

        results["views"][view_name] = {
            "smishing_pred_pct_before": round(pred_orig, 2),
            "contact_swap": {
                "smishing_pred_pct_after": round(pred_c, 2),
                "pp_drop": round(pred_c - pred_orig, 2),
                "mean_p_smishing_before": round(mean_orig, 4),
                "mean_p_smishing_after": round(mean_c, 4),
                "mean_p_smishing_delta": round(mean_c - mean_orig, 4),
            },
            "contact_and_brand_swap": {
                "smishing_pred_pct_after": round(pred_cb, 2),
                "pp_drop": round(pred_cb - pred_orig, 2),
                "mean_p_smishing_before": round(mean_orig, 4),
                "mean_p_smishing_after": round(mean_cb, 4),
                "mean_p_smishing_delta": round(mean_cb - mean_orig, 4),
            },
        }
    return results


def compute_extended_stage_b_diagnostics(
    eligible_df: pd.DataFrame,
    accepted_records: list[dict[str, Any]],
    brands: Iterable[str] | None = None,
    n_splits: int = 5,
    brand_registry: dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Run 4-view 3-class CV, class/stratum-matched is_synthetic probes, and counterfactual Smishing probes."""
    if brands is None:
        alias_dict = collect_brand_aliases(
            accepted_records,
            brand_registry=brand_registry,
            extra_brands=DEFAULT_KNOWN_BRANDS,
        )
        merged_list = sorted(
            {alias for aliases in alias_dict.values() for alias in aliases}
            | set(extract_metadata_brands(accepted_records)),
            key=len,
            reverse=True,
        )
        active_brands: Iterable[str] | None = merged_list
    else:
        active_brands = brands

    elig_views = _build_text_views_map(eligible_df["text"].fillna("").astype(str).tolist(), brands=active_brands)
    probe_3class_before = compute_3class_probe(eligible_df, text_views=elig_views, n_splits=n_splits, brands=active_brands)

    if accepted_records:
        acc_df_raw = pd.DataFrame(accepted_records)
        keep_acc_cols = [c for c in ("text", "label_3class", "language", "family_id", "domain") if c in acc_df_raw.columns]
        acc_df = acc_df_raw[keep_acc_cols].copy()
        acc_views = _build_text_views_map(acc_df["text"].fillna("").astype(str).tolist(), brands=active_brands)
        comb_views = {name: elig_views[name] + acc_views[name] for name in TEXT_VIEW_NAMES}
        combined_df = pd.concat(
            [
                eligible_df[["text", "label_3class", "language", "family_id"]],
                acc_df[["text", "label_3class", "language", "family_id"]],
            ],
            ignore_index=True,
        )
    else:
        acc_df = pd.DataFrame(columns=["text", "label_3class", "language", "family_id", "domain"])
        comb_views = elig_views
        combined_df = eligible_df[["text", "label_3class", "language", "family_id"]].copy()

    probe_3class_after, full_models = compute_3class_probe(
        combined_df,
        text_views=comb_views,
        n_splits=n_splits,
        brands=active_brands,
        return_full_models=True,
    )
    synth_probes = compute_synthetic_origin_probes(
        eligible_df,
        acc_df,
        n_splits=n_splits,
        brands=active_brands,
        all_views=comb_views if not acc_df.empty else None,
    )
    cf_probes = compute_counterfactual_probes(eligible_df, full_models, brands=active_brands)

    return {
        "three_class_before_pilot": probe_3class_before,
        "three_class_after_pilot": probe_3class_after,
        "is_synthetic_probes": synth_probes,
        "counterfactual_probes": cf_probes,
    }


def _load_review_decisions(review_paths: list[Path] | None) -> dict[str, dict[str, Any]]:
    """Load independent review logs keyed by candidate id (later files override earlier ones)."""
    reviews: dict[str, dict[str, Any]] = {}
    for rp in review_paths or []:
        for rec in load_jsonl(rp):
            rid = str(rec.get("id", "")).strip()
            if rid:
                reviews[rid] = rec
    return reviews


_DEFAULT_PRODUCTION_OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "generated" / "curation_v2"


def screen_stage_b_pilot(
    batch_paths: list[Path],
    v2_dir: str | Path,
    out_dir: str | Path,
    sim_threshold: float = 0.85,
    review_paths: list[Path] | None = None,
    include_extended_diagnostics: bool = True,
    diversity_mode: str = "audit",
    dry_run: bool = False,
    brand_registry: dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Screen Stage B pilot candidates in isolation, exporting accepted/rejected and diagnostics."""
    v2_path = Path(v2_dir)
    out_path = Path(out_dir)
    if dry_run and out_path.resolve() == _DEFAULT_PRODUCTION_OUT_DIR.resolve():
        raise ValueError(
            f"Refusing dry_run write into production output directory: {out_path}"
        )
    out_path.mkdir(parents=True, exist_ok=True)

    raw_candidates: list[dict[str, Any]] = []
    for p in batch_paths:
        raw_candidates.extend(load_jsonl(p))

    candidates: list[dict[str, Any]] = []
    input_char_lens: list[Any] = []
    for cand in raw_candidates:
        norm_cand = dict(cand)
        input_char_lens.append(norm_cand.get("char_len"))
        if "length_band" in norm_cand and norm_cand["length_band"] not in (None, ""):
            norm_cand["length_band"] = normalize_length_band(norm_cand["length_band"])
        norm_cand["char_len"] = len(str(norm_cand.get("text", "")))
        candidates.append(norm_cand)

    write_jsonl(out_path / "candidates.jsonl", candidates)

    reviews_by_id = _load_review_decisions(review_paths)

    existing_df = pd.read_parquet(
        v2_path / "existing_unique.parquet",
        columns=["text", "text_norm_strict", "eval_group", "family_id"],
    )
    quar_df = pd.read_parquet(v2_path / "quarantined.parquet", columns=["text", "text_norm_strict"])
    eligible_df = pd.read_parquet(v2_path / "eligible_train.parquet")

    ref_families = {str(f).strip() for f in existing_df["family_id"].dropna() if str(f).strip()}
    ref_records: list[tuple[str, str, str]] = []
    for t, ns, eg in zip(existing_df["text"], existing_df["text_norm_strict"], existing_df["eval_group"]):
        bucket = "train_pool" if eg == "train_pool" else "heldout_or_ood"
        ref_records.append((str(t), str(ns), bucket))
    for t, ns in zip(quar_df["text"], quar_df["text_norm_strict"]):
        ref_records.append((str(t), str(ns), "quarantined"))

    ref_norm_to_bucket: dict[str, str] = {}
    for _, ns, bucket in ref_records:
        # Strictest bucket priority if somehow duplicated
        if ns not in ref_norm_to_bucket or bucket == "heldout_or_ood":
            ref_norm_to_bucket[ns] = bucket

    ref_raw_texts = [r[1] for r in ref_records]
    ref_slot_texts = [slot_template_text(r[0]) for r in ref_records]
    ref_buckets = [r[2] for r in ref_records]

    cand_raw_texts = [strong_normalize_text(str(c.get("text", ""))) for c in candidates]
    cand_slot_texts = [slot_template_text(str(c.get("text", ""))) for c in candidates]

    # Fit isolated char-ngram TF-IDF matrices (returning ONLY booleans/reasons, never held-out text!)
    if candidates and ref_records:
        vec_raw = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        X_all_raw = vec_raw.fit_transform(ref_raw_texts + cand_raw_texts)
        sim_ref_raw = cosine_similarity(X_all_raw[len(ref_raw_texts) :], X_all_raw[: len(ref_raw_texts)])
        sim_cand_raw = cosine_similarity(X_all_raw[len(ref_raw_texts) :])

        vec_slot = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        X_all_slot = vec_slot.fit_transform(ref_slot_texts + cand_slot_texts)
        sim_ref_slot = cosine_similarity(X_all_slot[len(ref_slot_texts) :], X_all_slot[: len(ref_slot_texts)])
        sim_cand_slot = cosine_similarity(X_all_slot[len(ref_slot_texts) :])
    else:
        sim_ref_raw = np.zeros((len(candidates), len(ref_records)))
        sim_ref_slot = np.zeros((len(candidates), len(ref_records)))
        sim_cand_raw = np.zeros((len(candidates), len(candidates)))
        sim_cand_slot = np.zeros((len(candidates), len(candidates)))

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_cand_norms: set[str] = set()
    seen_cand_slots: set[str] = set()
    accepted_indices: list[int] = []

    for idx, cand in enumerate(candidates):
        reasons: list[str] = []
        cid = str(cand.get("id", "")).strip()
        text = str(cand.get("text", ""))
        lbl = str(cand.get("label_3class", ""))
        lang = str(cand.get("language", ""))
        fam = str(cand.get("family_id", "")).strip()
        norm_s = cand_raw_texts[idx]
        slot_k = slot_template_key(text)

        # 1. Schema, required fields & length_band consistency
        missing = [
            f
            for f in REQUIRED_FIELDS
            if f not in cand
            or (f not in ("source_row_ids", "has_link", "has_urgency", "is_synthetic") and cand[f] in (None, ""))
        ]
        if missing:
            reasons.append(f"missing_fields:{','.join(missing)}")
        if cand.get("is_synthetic") is not True:
            reasons.append("is_synthetic_must_be_true")
        if cand.get("source_row_ids") != []:
            reasons.append("source_row_ids_must_be_empty")
        if not norm_s or "\ufffd" in text:
            reasons.append("empty_or_corrupted_text")
        if not cid or cid in seen_ids:
            reasons.append("duplicate_or_empty_id")
        if not fam:
            reasons.append("empty_family_id")
        elif fam in ref_families:
            reasons.append("family_id_overlaps_existing")

        actual_len = len(text)
        expected_band = compute_length_band(actual_len)
        cand_band = str(cand.get("length_band", ""))
        if cand_band not in CANONICAL_LENGTH_BANDS:
            reasons.append(f"invalid_length_band:{cand_band}")
        elif cand_band != expected_band:
            reasons.append(f"length_band_mismatch:{cand_band}!={expected_band}")

        in_char_len = input_char_lens[idx]
        if in_char_len is not None and in_char_len != actual_len:
            reasons.append(f"char_len_mismatch:{in_char_len}!={actual_len}")

        # 2. Exact & near-duplicate isolated reference check
        if norm_s in ref_norm_to_bucket:
            reasons.append(f"exact_dup_{ref_norm_to_bucket[norm_s]}")
        elif norm_s in seen_cand_norms:
            reasons.append("exact_dup_intra_pilot")

        if len(ref_records) > 0:
            max_raw_idx = int(np.argmax(sim_ref_raw[idx]))
            max_raw_val = float(sim_ref_raw[idx, max_raw_idx])
            if max_raw_val >= sim_threshold:
                reasons.append(f"near_dup_{ref_buckets[max_raw_idx]}_raw(cos={max_raw_val:.3f})")

            max_slot_idx = int(np.argmax(sim_ref_slot[idx]))
            max_slot_val = float(sim_ref_slot[idx, max_slot_idx])
            if max_slot_val >= sim_threshold:
                reasons.append(f"near_dup_{ref_buckets[max_slot_idx]}_slot(cos={max_slot_val:.3f})")

        # 3. Intra-pilot shallow template / near-duplicate check against already-accepted pilot rows
        if slot_k in seen_cand_slots:
            reasons.append("intra_pilot_slot_template_clone")
        elif accepted_indices:
            max_p_raw = float(sim_cand_raw[idx, accepted_indices].max())
            max_p_slot = float(sim_cand_slot[idx, accepted_indices].max())
            if max_p_raw >= sim_threshold or max_p_slot >= sim_threshold:
                reasons.append(
                    f"intra_pilot_near_dup(raw={max_p_raw:.3f},slot={max_p_slot:.3f})"
                )

        # 4. Fictional entity, reserved domain, phone, boundary, language & CodeMix checks
        reasons.extend(_check_domain_and_phone(text))
        reasons.extend(_check_language_and_boundary(text, lbl, lang))
        if diversity_mode == "new_batch" and lbl == "Promotional" and lang == "CodeMix":
            cm_row = analyze_codemix_row(text, language=lang)
            if cm_row["is_hook_plus_pure_en"]:
                reasons.append("codemix_hook_plus_pure_en_disallowed")

        # 5. Independent review decisions (when review logs are provided)
        rev = reviews_by_id.get(cid)
        rev_decision = ""
        if rev is not None:
            rev_decision = str(rev.get("decision", "")).strip().lower()
            if rev_decision in ("reject", "rejected"):
                rev_reasons = rev.get("reasons") or ["rejected_by_independent_review"]
                for rr in rev_reasons:
                    reasons.append(f"independent_review:{rr}")
            elif rev_decision not in ("accept", "accepted"):
                reasons.append(f"invalid_review_decision:{rev_decision}")

        seen_ids.add(cid)
        if reasons:
            rej_rec = dict(cand)
            rej_rec["review_status"] = "rejected"
            rej_rec["rejection_reasons"] = sorted(set(reasons))
            if rev is not None:
                if rev.get("reviewer"):
                    rej_rec["reviewer"] = rev.get("reviewer")
                if rev.get("rationale"):
                    rej_rec["review_rationale"] = rev.get("rationale")
            rejected.append(rej_rec)
        else:
            acc_rec = dict(cand)
            if rev_decision in ("accept", "accepted"):
                acc_rec["review_status"] = "accepted"
                if rev and rev.get("reviewer"):
                    acc_rec["reviewer"] = rev.get("reviewer")
            else:
                acc_rec["review_status"] = "pending_independent_review"
            acc_rec["text_norm"] = normalize_text(text)
            acc_rec["text_norm_strict"] = norm_s
            acc_rec["char_len"] = actual_len
            accepted.append(acc_rec)
            accepted_indices.append(idx)
            seen_cand_norms.add(norm_s)
            seen_cand_slots.add(slot_k)

    write_jsonl(out_path / "accepted.jsonl", accepted)
    write_jsonl(out_path / "rejected.jsonl", rejected)

    # Check quota fulfillment per (label_3class, language)
    acc_cell_counts: dict[tuple[str, str], int] = {}
    acc_domain_counts: dict[str, int] = {}
    acc_class_domain_counts: dict[str, dict[str, int]] = {lbl: {} for lbl in CANONICAL_LABELS}
    fam_to_ids: dict[str, list[str]] = {}
    for rec in accepted:
        k = (rec["label_3class"], rec["language"])
        acc_cell_counts[k] = acc_cell_counts.get(k, 0) + 1
        dom = str(rec.get("domain", "unknown"))
        acc_domain_counts[dom] = acc_domain_counts.get(dom, 0) + 1
        acc_class_domain_counts[rec["label_3class"]][dom] = (
            acc_class_domain_counts[rec["label_3class"]].get(dom, 0) + 1
        )
        fam_to_ids.setdefault(rec["family_id"], []).append(rec["id"])

    quota_gaps: dict[str, int] = {}
    for (lbl, lang), target_n in TARGET_PILOT_QUOTAS.items():
        got = acc_cell_counts.get((lbl, lang), 0)
        if got != target_n:
            quota_gaps[f"{lbl}|{lang}"] = target_n - got

    # Surface diagnostics: eligible_train alone vs eligible_train + accepted
    diag_before = compute_surface_diagnostics(eligible_df)
    if accepted:
        acc_df = pd.DataFrame(accepted)[["text", "label_3class", "language", "family_id"]].copy()
        combined_df = pd.concat([eligible_df[["text", "label_3class", "language", "family_id"]], acc_df], ignore_index=True)
        diag_after = compute_surface_diagnostics(combined_df)
    else:
        combined_df = eligible_df
        diag_after = diag_before

    cls_before = {lbl: int((eligible_df["label_3class"] == lbl).sum()) for lbl in CANONICAL_LABELS}
    cls_after = {lbl: int((combined_df["label_3class"] == lbl).sum()) for lbl in CANONICAL_LABELS}
    target_t = max(cls_before.values())
    remaining_deficits = {lbl: int(max(0, target_t - cls_after[lbl])) for lbl in CANONICAL_LABELS}

    review_status_counts: dict[str, int] = {}
    for rec in accepted:
        st = str(rec["review_status"])
        review_status_counts[st] = review_status_counts.get(st, 0) + 1

    diversity_report = evaluate_codemix_and_entity_diversity(
        accepted,
        brand_registry=brand_registry,
        mode=diversity_mode,
    )
    extended_diagnostics = (
        compute_extended_stage_b_diagnostics(eligible_df, accepted, brand_registry=brand_registry)
        if include_extended_diagnostics
        else {"skipped": True}
    )

    summary = {
        "total_candidates": int(len(candidates)),
        "accepted_count": int(len(accepted)),
        "independently_reviewed_accepted_count": int(review_status_counts.get("accepted", 0)),
        "pending_independent_review_count": int(review_status_counts.get("pending_independent_review", 0)),
        "rejected_count": int(len(rejected)),
        "acceptance_rate_pct": round(len(accepted) / max(1, len(candidates)) * 100.0, 2),
        "review_status_counts": review_status_counts,
        "accepted_by_class": {lbl: sum(1 for r in accepted if r["label_3class"] == lbl) for lbl in CANONICAL_LABELS},
        "accepted_by_class_language": {f"{k[0]}|{k[1]}": v for k, v in sorted(acc_cell_counts.items())},
        "accepted_by_domain": dict(sorted(acc_domain_counts.items())),
        "accepted_by_class_domain": {
            lbl: dict(sorted(acc_class_domain_counts[lbl].items())) for lbl in CANONICAL_LABELS
        },
        "unique_accepted_families": int(len(fam_to_ids)),
        "shared_family_groups": {fam: ids for fam, ids in sorted(fam_to_ids.items()) if len(ids) > 1},
        "target_pilot_quotas": {f"{k[0]}|{k[1]}": v for k, v in TARGET_PILOT_QUOTAS.items()},
        "quota_gaps": quota_gaps,
        "quotas_exact_match": len(quota_gaps) == 0 and len(accepted) == sum(TARGET_PILOT_QUOTAS.values()),
        "rejection_reason_counts": (
            pd.Series([r for rec in rejected for r in rec["rejection_reasons"]]).value_counts().to_dict()
            if rejected
            else {}
        ),
        "surface_diagnostics_before_pilot": diag_before,
        "surface_diagnostics_after_pilot": diag_after,
        "codemix_and_entity_diversity": diversity_report,
        "extended_diagnostics": extended_diagnostics,
        "eligible_counts_before_pilot": cls_before,
        "eligible_counts_after_pilot": cls_after,
        "stage_b_target_T": int(target_t),
        "remaining_stage_b_deficits": remaining_deficits,
        "dry_run": bool(dry_run),
        "artifacts": {
            "candidates": str(out_path / "candidates.jsonl"),
            "accepted": str(out_path / "accepted.jsonl"),
            "rejected": str(out_path / "rejected.jsonl"),
            "summary": str(out_path / "pilot_validation_summary.json"),
        },
    }
    with open(out_path / "pilot_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
