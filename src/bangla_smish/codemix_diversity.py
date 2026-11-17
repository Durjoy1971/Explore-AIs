"""CodeMix linguistic, structural, and entity-frequency diversity metrics and gates."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

# Strip URLs, phones, USSD codes, and placeholder tags before prose-language checks
_URL_STRIP_RE = re.compile(
    r"(?:https?://|www\.)\S+|"
    r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|org|net|edu\.bd|com\.bd|gov\.bd|bd|info|xyz|top|li|ly|me|io|co)(?:/\S*)?\b",
    re.IGNORECASE,
)
_PHONE_STRIP_RE = re.compile(
    r"(?<!\d)\+?(?:880|৮৮০)[-\s]?[0-9০-৯][-\s0-9০-৯]{7,12}|"
    r"(?<!\d)[0০][1১][0-9০-৯][-\s]?[0-9০-৯]{3,4}[-\s]?[0-9০-৯]{4,6}|"
    r"(?<![\d.])(?:16|১৬)[0-9০-৯]{3}(?![\d.])",
    re.IGNORECASE,
)
_USSD_RE = re.compile(r"\*\d+(?:\*\d+)*#")
_TAG_STRIP_RE = re.compile(r"<(?:URL|PHONE|USSD|BRAND|TAG)>|\[REDACTED_[A-Z_]+\]")
_BN_CHAR_RE = re.compile(r"[\u0980-\u09ff]")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z]+")
_CAMEL_BRAND_RE = re.compile(r"\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b")
_SENT_SPLIT_RE = re.compile(r"(?<!\d)(?<!\bDr)(?<!\bMr)(?<!\bMs)(?<!\bNo)(?<=[.!?।])\s+")

# Common romanized Bengali words, particles, verbs, and pronouns (Banglish markers)
_BANGLISH_MARKERS: frozenset[str] = frozenset({
    "apni", "apnar", "apnake", "amader", "amar", "ami", "tumi", "tomar", "tomake",
    "amra", "tara", "tader", "nijer", "nijeder", "keu", "kauke",
    "aj", "ajke", "ajkei", "agamikal", "kalke", "porshu", "ekhon", "ekhoni",
    "ebar", "ebong", "ar", "ba", "na", "nei", "ache", "chilo", "hobe", "hoyeche",
    "hocche", "hoy", "holo", "korun", "koro", "korte", "korben", "korle", "korlei",
    "korchen", "koren", "korbo", "kori", "korlam", "paben", "pacchen", "paichen",
    "pao", "niben", "nin", "nao", "nile", "kinun", "kino", "kinte", "kinle",
    "dekhun", "dekho", "dekhte", "pathan", "pathate", "bachen", "bachte", "khujchen",
    "shikhe", "bolte", "bolleo", "chan", "lagche", "lagbe", "icche", "dorkar",
    "bhirer", "khoroch", "komate", "bondhuder", "bondhu", "chole", "esheche", "shera",
    "shomoy", "somoy", "basha", "bashar", "deshi", "ghore", "ghorer", "boshe",
    "shohoj", "katun", "chhutir", "chhuti", "jege", "kaaj", "kaj", "dewa", "chhara",
    "chhar", "kome", "kom", "beshi", "aro", "baki", "shuru", "cholche",
    "agei", "parben", "parbo", "pari", "paro", "diben",
    "janaben", "jante", "poucheche", "pouchabe", "jomar", "shesh", "obdi", "porjonto",
    "raat", "dupur", "shokal", "bikel", "sondha", "shondha", "masher", "mash",
    "bochor", "bochore", "shukro", "shoni", "brishtir", "brishti", "karone",
    "poriborte", "oshubidhar", "dukkhito", "porishodh", "taka", "takay", "takar",
    "matro", "shob", "sob", "shobai", "notun", "ekdom", "onek", "khub", "darun",
    "bhalo", "valo", "kintu", "jodi", "tahole", "karon", "ei", "oi", "ekta", "akta",
    "bhai", "apu", "kothay", "kobe", "kemon", "keno", "ki", "koto", "kichu", "kisu",
    "shudhu", "shudu", "proti", "prottek", "ghontay", "theke", "diye", "niye",
    "jonno", "jonyo", "sathe", "shathe", "shonge", "kache", "moddhe", "upore",
    "niche", "eder", "oder", "niyechen", "ghurte", "chailen", "ashun",
    "asun", "thakbe", "thakben", "thakun", "rakhun", "rakhen", "bhabchen", "chinta",
    "mullo", "shashroy", "upohar", "bisesh", "bishesh", "shokol", "nischinte",
    "taja", "shobji", "golper", "boi", "chhatro", "chhatri", "choshmar", "gohonar",
    "khejoor", "beshon", "chola", "sharbat", "juta", "bazar", "khuda", "pohela",
    "boishakh", "upolokkhe", "ektu", "birati", "khabar", "shajate", "shombhob",
})

# High-precision English words used to separate pure English sentences from Banglish
_ENGLISH_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "may", "might", "must", "shall", "should", "for", "from", "with", "without",
    "to", "in", "on", "at", "by", "of", "as", "into", "over", "under", "through",
    "across", "before", "after", "until", "till", "while", "during", "within",
    "and", "or", "but", "if", "not", "no", "your", "you", "we", "our", "us",
    "their", "they", "this", "that", "these", "those", "all", "any", "every",
    "each", "more", "most", "only", "just", "now", "today", "tonight", "tomorrow",
    "weekend", "friday", "saturday", "sunday", "monday", "tuesday", "wednesday",
    "thursday", "november", "december", "january", "february", "march", "april",
    "may", "june", "july", "august", "september", "october", "get", "buy", "shop",
    "save", "enjoy", "join", "apply", "register", "visit", "book", "install",
    "subscribe", "receive", "share", "earn", "meet", "please", "call", "dial",
    "grab", "order", "activate", "pay", "check", "track", "freeze", "collect",
    "submit", "learn", "open", "start", "upgrade", "partner", "sign", "offer",
    "offers", "discount", "discounts", "cashback", "free", "valid", "available",
    "off", "flat", "instant", "bonus", "deal", "deals", "sale", "items",
    "package", "packages", "bundle", "plan", "plans", "rate", "rates", "fee",
    "fees", "monthly", "weekly", "unlimited", "delivery", "pickup", "shipping",
    "first", "second", "third", "completely", "limited", "seats", "students",
    "fresh", "graduates", "apologize", "inconvenience", "rescheduling",
    "authorized", "authorize", "immediately", "anyone", "support", "staff",
})

# Canonical alias -> canonical brand key for known fictional pilot brands
_CANONICAL_BRAND_ALIASES: dict[str, str] = {
    "purbachalbank": "purbachalbank",
    "পূর্বাচলব্যাংক": "purbachalbank",
    "nodipay": "nodipay",
    "নদীপে": "nodipay",
    "bornotelecom": "bornotelecom",
    "বর্ণটেলিকম": "bornotelecom",
    "taranganet": "taranganet",
    "তরঙ্গনেট": "taranganet",
    "surovimart": "surovimart",
    "সুরভীমার্ট": "surovimart",
    "shikorparcel": "shikorparcel",
    "শিকড়পার্সেল": "shikorparcel",
    "শিকড়পার্সেল": "shikorparcel",
    "kothaedu": "kothaedu",
    "কথাএডু": "kothaedu",
    "meghnahealth": "meghnahealth",
    "মেঘনাহেলথ": "meghnahealth",
    "dighaexpress": "dighaexpress",
    "দীঘাএক্সপ্রেস": "dighaexpress",
    "aalokstore": "aalokstore",
    "আলোকস্টোর": "aalokstore",
    "chayaneer university": "chayaneer",
    "chayaneer": "chayaneer",
    "ছায়ানীড়": "chayaneer",
    "ছায়ানীড়": "chayaneer",
    "boishakhi fabrics": "boishakhifabrics",
    "বৈশাখী ফেব্রিক্স": "boishakhifabrics",
    "anondo books": "anondobooks",
    "আনন্দ বুকস": "anondobooks",
    "srijon career hub": "srijoncareerhub",
    "সৃজন ক্যারিয়ার হাব": "srijoncareerhub",
    "সৃজন ক্যারিয়ার হাব": "srijoncareerhub",
    "barendra power grid": "barendra",
    "বরেন্দ্র বিদ্যুৎ কেন্দ্র": "barendra",
    "rupnagar water supply": "rupnagarwater",
    "boshontobakery": "boshontobakery",
    "chitrobooks": "chitrobooks",
    "pallifresh": "pallifresh",
    "noyonoptics": "noyonoptics",
    "rupalithreads": "rupalithreads",
    "kallolelectronics": "kallolelectronics",
    "jhinukjewelry": "jhinukjewelry",
    "padmabazar": "padmabazar",
    "haatghor": "haatghor",
    "urmitech": "urmitech",
    "sagarikafashion": "sagarikafashion",
    "bikalpomart": "bikalpomart",
    "bonnakitchen": "bonnakitchen",
    "kapotakshocourier": "kapotakshocourier",
    "aloacademy": "aloacademy",
    "dishariskills": "dishariskills",
    "borno training hub": "bornotraininghub",
    "kironcare": "kironcare",
    "tarangagreen": "tarangagreen",
    "shondhacafe": "shondhacafe",
    "surmagym": "surmagym",
    "tistatours": "tistatours",
    "chayacinema": "chayacinema",
}

DEFAULT_THRESHOLDS: dict[str, float | int] = {
    "bn_script_en_min": 0.80,
    "hook_plus_pure_en_max_audit": 0.15,
    "hook_plus_pure_en_max_new_batch": 0.0,
    "intra_sentential_mix_min": 0.60,
    "sent1_excl_max": 0.35,
    "sent1_quest_max": 0.05,
    "max_brand_count_max": 5,
    "brandless_rate_min": 0.30,
    "ussd_1234_share_max": 0.25,
}


def _strip_contacts(text: str) -> str:
    cleaned = _URL_STRIP_RE.sub(" ", text)
    cleaned = _PHONE_STRIP_RE.sub(" ", cleaned)
    cleaned = _USSD_RE.sub(" ", cleaned)
    cleaned = _TAG_STRIP_RE.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _split_sentences(text: str) -> list[str]:
    stripped = _strip_contacts(text)
    if not stripped:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(stripped) if p.strip()]
    return parts


def _is_fragment_only(sent: str) -> bool:
    """Return True for short non-prose fragments such as 'Code: SHOP25.' or numbers."""
    if _BN_CHAR_RE.search(sent):
        return False
    toks = _LATIN_TOKEN_RE.findall(sent.lower())
    if not toks:
        return True
    if len(toks) <= 2 and toks[0] in {"code", "link", "info", "helpline", "call", "dial", "portal"}:
        return True
    return False


def _classify_segment_lang(seg: str) -> str:
    """Classify a prose segment into bn_pure, bn_en_mixed, banglish_pure, banglish_en_mixed, or en_pure."""
    has_bn = bool(_BN_CHAR_RE.search(seg))
    lat_tokens = _LATIN_TOKEN_RE.findall(seg.lower())
    if has_bn:
        return "bn_en_mixed" if lat_tokens else "bn_pure"
    if not lat_tokens:
        return "other"
    bn_hits = [t for t in lat_tokens if t in _BANGLISH_MARKERS]
    en_hits = [t for t in lat_tokens if t in _ENGLISH_WORDS]
    if not bn_hits:
        return "en_pure"
    if len(en_hits) >= 2:
        return "banglish_en_mixed"
    return "banglish_pure"


def _sent1_end_punct(sents: list[str]) -> str:
    if not sents:
        return "none"
    last = sents[0].rstrip()[-1:]
    if last in {"!", "?", "।", ".", ":"}:
        return last
    return "none"


def analyze_codemix_row(text: str, language: str | None = None) -> dict[str, Any]:
    """Classify a single SMS row's CodeMix script subtype and discourse structure."""
    raw = str(text or "")
    stripped = _strip_contacts(raw)
    has_bn = bool(_BN_CHAR_RE.search(stripped))
    lat_tokens = _LATIN_TOKEN_RE.findall(stripped.lower())
    bn_hits = [t for t in lat_tokens if t in _BANGLISH_MARKERS]
    en_hits = [t for t in lat_tokens if t in _ENGLISH_WORDS]

    if has_bn and lat_tokens:
        script_subtype = "bn_script_en"
    elif (not has_bn) and lat_tokens and (
        (bn_hits and en_hits) or (language == "CodeMix" and bool(bn_hits))
    ):
        script_subtype = "banglish_latin_en"
    elif (not has_bn) and language == "CodeMix" and lat_tokens:
        # Respect explicit CodeMix label when Banglish markers were sparse
        script_subtype = "banglish_latin_en"
    else:
        script_subtype = "other"

    sents = _split_sentences(raw)
    sentence_count = len(sents)
    sent1_punct = _sent1_end_punct(sents)
    has_excl = "!" in stripped
    has_quest = "?" in stripped

    prose_sents = [s for s in sents if not _is_fragment_only(s)] or sents
    prose_langs = [_classify_segment_lang(s) for s in prose_sents]

    discourse_pattern = "other"
    if script_subtype != "other" or language == "CodeMix":
        if len(prose_sents) >= 2:
            s1_lang = prose_langs[0]
            rest_langs = prose_langs[1:]
            all_rest_en = all(lang == "en_pure" for lang in rest_langs)

            if s1_lang != "en_pure" and all_rest_en:
                if sent1_punct == "!":
                    discourse_pattern = "hook_excl_pure_en"
                elif sent1_punct == "?":
                    discourse_pattern = "hook_quest_pure_en"
                else:
                    discourse_pattern = "stmt_colon_lead_en"
            elif s1_lang == "en_pure" and any(lang != "en_pure" for lang in rest_langs):
                discourse_pattern = "en_lead_bn_body"
            elif all(lang in {"bn_en_mixed", "banglish_en_mixed", "bn_pure", "banglish_pure"} for lang in prose_langs):
                if any(lang in {"bn_en_mixed", "banglish_en_mixed"} for lang in prose_langs):
                    discourse_pattern = "intra_sentential_mix"
                else:
                    discourse_pattern = "multi_sent_alternating"
            else:
                discourse_pattern = "multi_sent_alternating"
        elif len(prose_sents) == 1:
            s1 = prose_sents[0]
            if ":" in s1:
                left, right = s1.split(":", 1)
                left_lang = _classify_segment_lang(left)
                right_lang = _classify_segment_lang(right)
                if left_lang == "en_pure" and right_lang != "en_pure":
                    discourse_pattern = "en_lead_bn_body"
                elif left_lang != "en_pure" and right_lang == "en_pure":
                    discourse_pattern = "stmt_colon_lead_en"
                elif prose_langs[0] in {"bn_en_mixed", "banglish_en_mixed"}:
                    discourse_pattern = "intra_sentential_mix"
            elif prose_langs[0] in {"bn_en_mixed", "banglish_en_mixed"}:
                discourse_pattern = "intra_sentential_mix"
            elif script_subtype == "bn_script_en":
                discourse_pattern = "intra_sentential_mix"

    is_hook_plus_pure_en = discourse_pattern in {"hook_excl_pure_en", "hook_quest_pure_en"}

    return {
        "script_subtype": script_subtype,
        "discourse_pattern": discourse_pattern,
        "sent1_end_punct": sent1_punct,
        "has_excl": has_excl,
        "has_quest": has_quest,
        "is_hook_plus_pure_en": is_hook_plus_pure_en,
        "sentence_count": sentence_count,
    }


def _extract_brands_in_text(
    text: str,
    alias_map: dict[str, str],
) -> list[str]:
    """Return canonical brand keys mentioned in text."""
    found: list[str] = []
    seen: set[str] = set()
    lowered = text.lower()

    # Check longest aliases first to avoid partial hits
    for alias in sorted(alias_map, key=len, reverse=True):
        if not alias:
            continue
        if _BN_CHAR_RE.search(alias):
            pattern = re.escape(alias)
        else:
            pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, lowered):
            canon = alias_map[alias]
            if canon not in seen:
                found.append(canon)
                seen.add(canon)

    # Also catch unlisted CamelCase fictional brands (e.g., PadmaPay, NodiMart)
    stripped = _URL_STRIP_RE.sub(" ", text)
    for m in _CAMEL_BRAND_RE.finditer(stripped):
        tok = m.group(0)
        key = tok.lower()
        if key in {"smarttv", "whatsapp", "youtube", "facebook"}:
            continue
        canon = alias_map.get(key, key)
        if canon not in seen:
            found.append(canon)
            seen.add(canon)

    return found


def _opener_3gram(text: str) -> str:
    cleaned = _strip_contacts(text).lower()
    cleaned = re.sub(r"[^\w\u0980-\u09ff\s]", " ", cleaned)
    toks = cleaned.split()
    return " ".join(toks[:3]) if len(toks) >= 3 else " ".join(toks)


def _add_brand_entry(
    canon_to_aliases: dict[str, list[str]],
    alias_to_canon: dict[str, str],
    canonical: str,
    aliases: list[str] | tuple[str, ...] | set[str] | str | None = None,
) -> None:
    canon_clean = str(canonical or "").strip()
    if not canon_clean:
        return
    canon_key = alias_to_canon.get(canon_clean.lower(), canon_clean.lower())
    bucket = canon_to_aliases.setdefault(canon_key, [])
    existing_lower = {a.lower() for a in bucket}

    candidates: list[str] = [canon_clean]
    if isinstance(aliases, str):
        candidates.append(aliases)
    elif isinstance(aliases, (list, tuple, set)):
        candidates.extend(str(a) for a in aliases if a is not None)

    for raw_alias in candidates:
        alias_clean = raw_alias.strip()
        if not alias_clean:
            continue
        alias_low = alias_clean.lower()
        alias_to_canon.setdefault(alias_low, canon_key)
        if alias_low not in existing_lower:
            bucket.append(alias_clean)
            existing_lower.add(alias_low)


def _merge_registry_payload(
    canon_to_aliases: dict[str, list[str]],
    alias_to_canon: dict[str, str],
    payload: Any,
) -> None:
    if payload is None:
        return
    if isinstance(payload, dict):
        if set(payload.keys()) == {"brands"} and isinstance(payload["brands"], (dict, list)):
            _merge_registry_payload(canon_to_aliases, alias_to_canon, payload["brands"])
            return
        if "canonical" in payload or "name" in payload:
            canon_val = str(payload.get("canonical") or payload.get("name") or "").strip()
            alias_val = payload.get("aliases", [])
            _add_brand_entry(canon_to_aliases, alias_to_canon, canon_val, alias_val)
            return
        for canon_key, alias_val in payload.items():
            _add_brand_entry(canon_to_aliases, alias_to_canon, str(canon_key), alias_val)
    elif isinstance(payload, (list, tuple, set)):
        for item in payload:
            if isinstance(item, dict):
                _merge_registry_payload(canon_to_aliases, alias_to_canon, item)
            elif isinstance(item, str):
                _add_brand_entry(canon_to_aliases, alias_to_canon, item, [item])


def _merge_record_brand_metadata(
    canon_to_aliases: dict[str, list[str]],
    alias_to_canon: dict[str, str],
    record: dict[str, Any],
) -> None:
    anchor: str | None = None
    for anchor_key in ("brand", "entity_name"):
        val = record.get(anchor_key)
        if isinstance(val, str) and val.strip():
            anchor = val.strip()
            _add_brand_entry(canon_to_aliases, alias_to_canon, anchor, [anchor])
            break

    for field in ("brand", "entity_name", "brand_names", "brand_aliases", "entities"):
        val = record.get(field)
        if val is None or val == "":
            continue
        if isinstance(val, dict):
            _merge_registry_payload(canon_to_aliases, alias_to_canon, val)
        elif isinstance(val, (list, tuple, set)):
            for item in val:
                if isinstance(item, dict):
                    _merge_registry_payload(canon_to_aliases, alias_to_canon, item)
                elif isinstance(item, str) and item.strip():
                    target = anchor if (field == "brand_aliases" and anchor is not None) else item.strip()
                    _add_brand_entry(canon_to_aliases, alias_to_canon, target, [item.strip()])
        elif isinstance(val, str) and val.strip():
            target = anchor if (field == "brand_aliases" and anchor is not None) else val.strip()
            _add_brand_entry(canon_to_aliases, alias_to_canon, target, [val.strip()])


def collect_brand_aliases(
    records: list[dict[str, Any]] | None = None,
    *,
    brand_registry: dict[str, list[str] | str] | str | Path | None = None,
    extra_brands: list[str] | tuple[str, ...] | None = None,
) -> dict[str, list[str]]:
    """Merge baseline pilot brands, optional registry, record metadata, and extra brands."""
    canon_to_aliases: dict[str, list[str]] = {}
    alias_to_canon: dict[str, str] = {}

    for alias_str, canon_str in _CANONICAL_BRAND_ALIASES.items():
        _add_brand_entry(canon_to_aliases, alias_to_canon, canon_str, [alias_str])

    if brand_registry is not None:
        if isinstance(brand_registry, (str, Path)):
            reg_path = Path(brand_registry)
            with open(reg_path, "r", encoding="utf-8") as f:
                reg_payload = json.load(f)
            _merge_registry_payload(canon_to_aliases, alias_to_canon, reg_payload)
        else:
            _merge_registry_payload(canon_to_aliases, alias_to_canon, brand_registry)

    if records:
        for rec in records:
            if isinstance(rec, dict):
                _merge_record_brand_metadata(canon_to_aliases, alias_to_canon, rec)

    if extra_brands:
        for brand in extra_brands:
            b_clean = str(brand or "").strip()
            if b_clean:
                _add_brand_entry(canon_to_aliases, alias_to_canon, b_clean, [b_clean])

    return canon_to_aliases


def evaluate_codemix_and_entity_diversity(
    records: list[dict[str, Any]],
    *,
    known_brands: list[str] | None = None,
    brand_registry: dict[str, list[str] | str] | str | Path | None = None,
    mode: str = "audit",
) -> dict[str, Any]:
    """Compute CodeMix linguistic/structural diversity, punctuation, and entity-concentration gates."""
    if mode not in {"audit", "new_batch"}:
        raise ValueError(f"Unsupported mode: {mode!r}. Expected 'audit' or 'new_batch'.")

    merged_brands = collect_brand_aliases(
        records,
        brand_registry=brand_registry,
        extra_brands=known_brands,
    )
    alias_map: dict[str, str] = {}
    for canon_key, aliases in merged_brands.items():
        for alias in aliases:
            alias_map[alias.lower()] = canon_key

    promo_records = [r for r in records if r.get("label_3class") == "Promotional"]
    target_punct_records = promo_records if promo_records else records

    codemix_records = [r for r in records if r.get("language") == "CodeMix"]
    promo_cm_records = [r for r in codemix_records if r.get("label_3class") == "Promotional"]
    gate_cm_records = promo_cm_records if promo_cm_records else codemix_records

    cm_script_counts: Counter[str] = Counter()
    cm_discourse_counts: Counter[str] = Counter()
    for r in codemix_records:
        info = analyze_codemix_row(str(r.get("text", "")), language="CodeMix")
        cm_script_counts[info["script_subtype"]] += 1
        cm_discourse_counts[info["discourse_pattern"]] += 1

    promo_cm_script_counts: Counter[str] = Counter()
    promo_cm_discourse_counts: Counter[str] = Counter()
    for r in gate_cm_records:
        info = analyze_codemix_row(str(r.get("text", "")), language="CodeMix")
        promo_cm_script_counts[info["script_subtype"]] += 1
        promo_cm_discourse_counts[info["discourse_pattern"]] += 1

    n_cm = len(codemix_records)
    n_gate_cm = len(gate_cm_records)

    cm_script_rates = {k: round(v / n_cm, 4) for k, v in cm_script_counts.items()} if n_cm else {}
    cm_discourse_rates = {k: round(v / n_cm, 4) for k, v in cm_discourse_counts.items()} if n_cm else {}
    promo_cm_script_rates = (
        {k: round(v / n_gate_cm, 4) for k, v in promo_cm_script_counts.items()} if n_gate_cm else {}
    )
    promo_cm_discourse_rates = (
        {k: round(v / n_gate_cm, 4) for k, v in promo_cm_discourse_counts.items()} if n_gate_cm else {}
    )

    bn_script_en_rate = (
        round(promo_cm_script_counts.get("bn_script_en", 0) / n_gate_cm, 4) if n_gate_cm else 0.0
    )
    hook_plus_pure_en_count = (
        promo_cm_discourse_counts.get("hook_excl_pure_en", 0)
        + promo_cm_discourse_counts.get("hook_quest_pure_en", 0)
    )
    hook_plus_pure_en_rate = round(hook_plus_pure_en_count / n_gate_cm, 4) if n_gate_cm else 0.0
    intra_sentential_mix_rate = (
        round(promo_cm_discourse_counts.get("intra_sentential_mix", 0) / n_gate_cm, 4)
        if n_gate_cm
        else 0.0
    )

    # Punctuation and opener motif metrics on Promotional records (or all if none)
    n_target = len(target_punct_records)
    sent1_punct_counts: Counter[str] = Counter()
    opener_3grams: Counter[str] = Counter()
    brand_counts: Counter[str] = Counter()
    rows_with_brand = 0
    total_ussd_mentions = 0
    ussd_1234_count = 0

    for r in target_punct_records:
        txt = str(r.get("text", ""))
        sents = _split_sentences(txt)
        sent1_punct_counts[_sent1_end_punct(sents)] += 1
        og = _opener_3gram(txt)
        if og:
            opener_3grams[og] += 1
        brands = _extract_brands_in_text(txt, alias_map)
        if brands:
            rows_with_brand += 1
            for b in brands:
                brand_counts[b] += 1
        for u in _USSD_RE.findall(txt):
            total_ussd_mentions += 1
            if u == "*1234#" or u.startswith("*1234*"):
                ussd_1234_count += 1

    sent1_excl_rate = round(sent1_punct_counts.get("!", 0) / n_target, 4) if n_target else 0.0
    sent1_quest_rate = round(sent1_punct_counts.get("?", 0) / n_target, 4) if n_target else 0.0
    max_opener_3gram_count = opener_3grams.most_common(1)[0][1] if opener_3grams else 0
    max_opener_3gram_rate = round(max_opener_3gram_count / n_target, 4) if n_target else 0.0

    brand_mention_rate = round(rows_with_brand / n_target, 4) if n_target else 0.0
    brandless_rate = round(1.0 - brand_mention_rate, 4) if n_target else 0.0
    max_brand_count = brand_counts.most_common(1)[0][1] if brand_counts else 0
    ussd_1234_share = round(ussd_1234_count / total_ussd_mentions, 4) if total_ussd_mentions else 0.0

    hook_threshold = (
        DEFAULT_THRESHOLDS["hook_plus_pure_en_max_new_batch"]
        if mode == "new_batch"
        else DEFAULT_THRESHOLDS["hook_plus_pure_en_max_audit"]
    )

    gate_checks: dict[str, dict[str, Any]] = {
        "codemix_bn_script_en": {
            "passed": (n_gate_cm == 0) or (bn_script_en_rate >= DEFAULT_THRESHOLDS["bn_script_en_min"]),
            "actual": bn_script_en_rate,
            "threshold": DEFAULT_THRESHOLDS["bn_script_en_min"],
            "op": ">=",
            "detail": f"Bengali-script+English share on Promotional CodeMix: {bn_script_en_rate:.4f} (n={n_gate_cm}).",
        },
        "codemix_hook_plus_pure_en": {
            "passed": (n_gate_cm == 0) or (hook_plus_pure_en_rate <= hook_threshold),
            "actual": hook_plus_pure_en_rate,
            "threshold": hook_threshold,
            "op": "==" if mode == "new_batch" else "<=",
            "detail": f"[hook !/?] + [pure English] share: {hook_plus_pure_en_rate:.4f} ({hook_plus_pure_en_count}/{n_gate_cm}).",
        },
        "codemix_intra_sentential_mix": {
            "passed": (n_gate_cm == 0) or (intra_sentential_mix_rate >= DEFAULT_THRESHOLDS["intra_sentential_mix_min"]),
            "actual": intra_sentential_mix_rate,
            "threshold": DEFAULT_THRESHOLDS["intra_sentential_mix_min"],
            "op": ">=",
            "detail": f"Intra-sentential lexical mixing share: {intra_sentential_mix_rate:.4f} (n={n_gate_cm}).",
        },
        "promo_sent1_excl_rate": {
            "passed": (n_target == 0) or (sent1_excl_rate <= DEFAULT_THRESHOLDS["sent1_excl_max"]),
            "actual": sent1_excl_rate,
            "threshold": DEFAULT_THRESHOLDS["sent1_excl_max"],
            "op": "<=",
            "detail": f"Sentence-1 '!' rate on Promotional rows: {sent1_excl_rate:.4f} (n={n_target}).",
        },
        "promo_sent1_quest_rate": {
            "passed": (n_target == 0) or (sent1_quest_rate <= DEFAULT_THRESHOLDS["sent1_quest_max"]),
            "actual": sent1_quest_rate,
            "threshold": DEFAULT_THRESHOLDS["sent1_quest_max"],
            "op": "<=",
            "detail": f"Sentence-1 '?' rate on Promotional rows: {sent1_quest_rate:.4f} (n={n_target}).",
        },
        "max_brand_count": {
            "passed": max_brand_count <= DEFAULT_THRESHOLDS["max_brand_count_max"],
            "actual": max_brand_count,
            "threshold": DEFAULT_THRESHOLDS["max_brand_count_max"],
            "op": "<=",
            "detail": f"Max mentions for any single fictional brand: {max_brand_count}.",
        },
        "brandless_rate": {
            "passed": (n_target == 0) or (brandless_rate >= DEFAULT_THRESHOLDS["brandless_rate_min"]),
            "actual": brandless_rate,
            "threshold": DEFAULT_THRESHOLDS["brandless_rate_min"],
            "op": ">=",
            "detail": f"Brandless Promotional SMS rate: {brandless_rate:.4f} (n={n_target}).",
        },
        "ussd_1234_share": {
            "passed": (total_ussd_mentions == 0) or (ussd_1234_share <= DEFAULT_THRESHOLDS["ussd_1234_share_max"]),
            "actual": ussd_1234_share,
            "threshold": DEFAULT_THRESHOLDS["ussd_1234_share_max"],
            "op": "<=",
            "detail": f"*1234 USSD share: {ussd_1234_share:.4f} ({ussd_1234_count}/{total_ussd_mentions}).",
        },
    }

    all_gates_passed = all(g["passed"] for g in gate_checks.values())

    return {
        "mode": mode,
        "total_rows": len(records),
        "promotional_rows": len(promo_records),
        "codemix_rows": n_cm,
        "promotional_codemix_rows": len(promo_cm_records),
        "codemix_script_counts": dict(cm_script_counts),
        "codemix_script_rates": cm_script_rates,
        "codemix_discourse_counts": dict(cm_discourse_counts),
        "codemix_discourse_rates": cm_discourse_rates,
        "promotional_codemix_script_counts": dict(promo_cm_script_counts),
        "promotional_codemix_script_rates": promo_cm_script_rates,
        "promotional_codemix_discourse_counts": dict(promo_cm_discourse_counts),
        "promotional_codemix_discourse_rates": promo_cm_discourse_rates,
        "bn_script_en_rate": bn_script_en_rate,
        "hook_plus_pure_en_rate": hook_plus_pure_en_rate,
        "intra_sentential_mix_rate": intra_sentential_mix_rate,
        "sent1_punct_counts": dict(sent1_punct_counts),
        "sent1_excl_rate": sent1_excl_rate,
        "sent1_quest_rate": sent1_quest_rate,
        "max_opener_3gram_count": max_opener_3gram_count,
        "max_opener_3gram_rate": max_opener_3gram_rate,
        "top_opener_3grams": opener_3grams.most_common(10),
        "brand_mention_rate": brand_mention_rate,
        "brandless_rate": brandless_rate,
        "max_brand_count": max_brand_count,
        "unique_brands_count": len(brand_counts),
        "top_brands": brand_counts.most_common(10),
        "total_ussd_mentions": total_ussd_mentions,
        "ussd_1234_count": ussd_1234_count,
        "ussd_1234_share": ussd_1234_share,
        "gate_checks": gate_checks,
        "all_gates_passed": all_gates_passed,
    }
