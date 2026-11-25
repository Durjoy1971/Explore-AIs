"""Unified schema, dataset loaders, normalization, and shortcut/PII feature extractors."""

from __future__ import annotations

import csv
import re
import unicodedata
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
FICTIONAL_BATCH1_TO_4_BRANDS: tuple[str, ...] = (
    # Batch 1 (13 brands)
    "ShaplaSeba", "Shapla Seba", "শাপলাসেবা",
    "Meghdoot", "মেঘদূত",
    "RupkothaCard", "Rupkotha", "রূপকথাকার্ড",
    "KashfulPay", "কাশফুলপে",
    "NilachalStore", "Nilachal", "নীলাচলস্টোর",
    "LalbaghMart", "লালবাগমার্ট",
    "AronyokBooks", "Aronyok Books", "আরণ্যকবুকস",
    "BhorerBazar", "ভোরেরবাজার",
    "MoniharFashion", "Monihar", "মণিহারফ্যাশন",
    "SonargaoExpress", "Sonargao Express", "সোনারগাঁওএক্সপ্রেস",
    "CharukolaHealth", "Charukola Clinic", "চারুকলাহেলথ",
    "DipshikhaAcademy", "Dipshikha Academy", "দীপশিখাএকাডেমি",
    "ShitolCafe", "শীতলক্যাফে",
    # Batch 2 (8 brands)
    "DeshNet", "দেশনেট",
    "PolashPay", "পলাশপে",
    "TuragBank", "তুরাগব্যাংক",
    "ChaitiMart", "চৈতীমার্ট",
    "ProbashiCourier", "প্রবাসী কুরিয়ার", "প্রবাসীকুরিয়ার",
    "NandanBazar", "নন্দনবাজার",
    "UtshobFashion", "উৎসবফ্যাশন",
    "EkusheyTech", "একুশেটেক",
    # Batch 3 (8 brands)
    "KorotoaTel", "করতোয়াটেল", "করতোয়াটেল",
    "ShiuliPay", "শিউলিপে",
    "RupshaBank", "রূপসাব্যাংক",
    "ShimulMart", "শিমুলমার্ট",
    "BakulBazar", "বকুলবাজার",
    "MayurTech", "ময়ূরটেক", "ময়ূরটেক",
    "PalkiCourier", "পালকিকুরিয়ার", "পালকিকুরিয়ার", "পালকি কুরিয়ার", "পালকি কুরিয়ার",
    "KodomStyle", "কদমস্টাইল",
    # Batch 4 (8 brands)
    "JuiTel", "জুঁইটেল",
    "MadhobiPay", "মাধবীপে",
    "IchamotiBank", "ইছামতিব্যাংক", "ইছামতী ব্যাংক",
    "NokshiMart", "নকশিমার্ট",
    "DoyelBazar", "দোয়েলবাজার", "দোয়েলবাজার",
    "ShankhoTech", "শঙ্খটেক",
    "BoikaliCourier", "বৈকালীকুরিয়ার", "বৈকালীকুরিয়ার", "বৈকালী কুরিয়ার", "বৈকালী কুরিয়ার",
    "TitasStyle", "তিতাসস্টাইল",
)
FICTIONAL_BATCH5_BRANDS: tuple[str, ...] = (
    # Batch 5 (8 brands)
    "SurjoTel", "সূর্যটেল",
    "BeliMart", "বেলীমার্ট",
    "KajolTech", "কাজলটেক",
    "KomolPay", "কমলপে",
    "KarnaphuliBank", "কর্ণফুলীব্যাংক", "কর্ণফুলী ব্যাংক",
    "ToriCourier", "তরীকুরিয়ার", "তরীকুরিয়ার", "তরী কুরিয়ার", "তরী কুরিয়ার",
    "ChampaStyle", "চম্পাস্টাইল",
    "KanchanBazar", "কাঞ্চনবাজার",
)
FICTIONAL_BATCH6_BRANDS: tuple[str, ...] = (
    # Batch 6 (8 brands)
    "AkashNet", "আকাশনেট",
    "PiyaliMart", "পিয়ালীমার্ট", "পিয়ালীমার্ট",
    "NibirTech", "নিবিড়টেক", "নিবিড়টেক",
    "TonuPay", "তনুপে",
    "BarshaBank", "বর্ষাব্যাংক", "বর্ষা ব্যাংক",
    "UjanCourier", "উজানকুরিয়ার", "উজানকুরিয়ার", "উজান কুরিয়ার", "উজান কুরিয়ার",
    "ArshiStyle", "আরশিস্টাইল",
    "KakonBazar", "কাঁকনবাজার",
)
DEFAULT_KNOWN_BRANDS: tuple[str, ...] = tuple(
    dict.fromkeys((
        *REAL_BD_BRANDS,
        *FICTIONAL_PILOT_BRANDS,
        *FICTIONAL_BATCH1_TO_4_BRANDS,
        *FICTIONAL_BATCH5_BRANDS,
        *FICTIONAL_BATCH6_BRANDS,
    ))
)

DEFAULT_CANONICAL_BRAND_ALIASES: dict[str, str] = {
    # Real Bangladeshi brands
    "bkash": "bkash", "বিকাশ": "bkash",
    "nagad": "nagad", "নগদ": "nagad",
    "rocket": "rocket", "রকেট": "rocket",
    "upay": "upay", "উপায়": "upay", "উপায়": "upay",
    "grameenphone": "grameenphone", "gp": "grameenphone", "mygp": "grameenphone", "গ্রামীণফোন": "grameenphone",
    "robi": "robi", "রবি": "robi",
    "banglalink": "banglalink", "bl": "banglalink", "বাংলালিংক": "banglalink",
    "airtel": "airtel", "এয়ারটেল": "airtel", "এয়ারটেল": "airtel",
    "teletalk": "teletalk", "টেলিটক": "teletalk",
    "skitto": "skitto",
    "daraz": "daraz", "দারাজ": "daraz",
    "pathao": "pathao", "পাঠাও": "pathao",
    "shwapno": "shwapno", "স্বপ্ন": "shwapno",
    "chaldal": "chaldal", "চালডাল": "chaldal",
    "foodpanda": "foodpanda", "ফুডপান্ডা": "foodpanda",
    "pickaboo": "pickaboo",
    "rokomari": "rokomari",
    "evaly": "evaly",
    "dbbl": "dbbl", "dutch-bangla": "dbbl", "dutch bangla": "dbbl", "ডাচ-বাংলা": "dbbl",
    "brac bank": "brac bank", "ব্র্যাক ব্যাংক": "brac bank",
    "sonali bank": "sonali bank", "সোনালী ব্যাংক": "sonali bank",
    "islami bank": "islami bank", "ইসলামী ব্যাংক": "islami bank",
    "city bank": "city bank",
    "ebl": "ebl",
    # Fictional Pilot brands
    "purbachalbank": "purbachalbank", "পূর্বাচলব্যাংক": "purbachalbank",
    "nodipay": "nodipay", "নদীপে": "nodipay",
    "bornotelecom": "bornotelecom", "বর্ণটেলিকম": "bornotelecom",
    "taranganet": "taranganet", "তরঙ্গনেট": "taranganet",
    "surovimart": "surovimart", "সুরভীমার্ট": "surovimart",
    "shikorparcel": "shikorparcel", "শিকড়পার্সেল": "shikorparcel", "শিকড়পার্সেল": "shikorparcel",
    "kothaedu": "kothaedu", "কথাএডু": "kothaedu",
    "meghnahealth": "meghnahealth", "মেঘনাহেলথ": "meghnahealth",
    "dighaexpress": "dighaexpress", "দীঘাএক্সপ্রেস": "dighaexpress",
    "aalokstore": "aalokstore", "আলোকস্টোর": "aalokstore",
    "aloacademy": "aloacademy", "আলোঅ্যাকাডেমি": "aloacademy",
    "urmitech": "urmitech", "ঊর্মিটেক": "urmitech",
    "kapotakshocourier": "kapotakshocourier", "কপোতাক্ষকুরিয়ার": "kapotakshocourier", "কপোতাক্ষ": "kapotakshocourier",
    "bonnakitchen": "bonnakitchen", "বন্যাকিচেন": "bonnakitchen",
    "dishariskills": "dishariskills", "দিশারীস্কিলস": "dishariskills",
    "kironcare": "kironcare", "কিরণকেয়ার": "kironcare",
    "chitrobooks": "chitrobooks", "চিত্রবুকস": "chitrobooks",
    "pallifresh": "pallifresh", "পল্লীফ্রেশ": "pallifresh",
    "chayacinema": "chayacinema", "ছায়াসিনেমা": "chayacinema",
    "surmagym": "surmagym", "সুরমাজিম": "surmagym",
    "tistatours": "tistatours", "তিস্তাট্যুরস": "tistatours",
    "noyonoptics": "noyonoptics", "নয়নঅপটিক্স": "noyonoptics",
    "rupalithreads": "rupalithreads", "রূপালীথ্রেডস": "rupalithreads",
    "kallolelectronics": "kallolelectronics", "কল্লোলইলেকট্রনিক্স": "kallolelectronics",
    "jhinukjewelry": "jhinukjewelry", "ঝিনুকজুয়েলারি": "jhinukjewelry",
    "padmabazar": "padmabazar", "পদ্মাবাজার": "padmabazar",
    "haatghor": "haatghor", "হাটঘর": "haatghor",
    "sagarikafashion": "sagarikafashion", "সাগরিকাফ্যাশন": "sagarikafashion",
    "shondhacafe": "shondhacafe", "সন্ধ্যাক্যাফে": "shondhacafe",
    "modhumoticlub": "modhumoticlub", "মধুমতীক্লাব": "modhumoticlub",
    "tarangagreen": "tarangagreen", "তরঙ্গগ্রিন": "tarangagreen",
    "boshontobakery": "boshontobakery", "বসন্তবেকারি": "boshontobakery",
    "bikalpomart": "bikalpomart", "বিকল্পমার্ট": "bikalpomart",
    "chayaneer": "chayaneer", "chayaneer university": "chayaneer",
    "ছায়ানীড়": "chayaneer", "ছায়ানীড়": "chayaneer", "ছায়ানীড় বিশ্ববিদ্যালয়": "chayaneer",
    "barendra": "barendra", "barendra power grid": "barendra", "বরেন্দ্র বিদ্যুৎ কেন্দ্র": "barendra",
    "rupnagar water supply": "rupnagarwater",
    "neerhaat": "neerhaat", "নীড়হাট": "neerhaat", "নীড়হাট": "neerhaat",
    "boishakhi fabrics": "boishakhifabrics", "বৈশাখী ফেব্রিক্স": "boishakhifabrics",
    "anondo books": "anondobooks", "আনন্দ বুকস": "anondobooks",
    "srijon career hub": "srijoncareerhub", "সৃজন ক্যারিয়ার হাব": "srijoncareerhub", "সৃজন ক্যারিয়ার হাব": "srijoncareerhub",
    "borno training hub": "bornotraininghub",
    # Batch 1-4 brands
    "shaplaseba": "shaplaseba", "shapla seba": "shaplaseba", "শাপলাসেবা": "shaplaseba",
    "meghdoot": "meghdoot", "মেঘদূত": "meghdoot",
    "rupkothacard": "rupkothacard", "rupkotha": "rupkothacard", "রূপকথাকার্ড": "rupkothacard",
    "kashfulpay": "kashfulpay", "কাশফুলপে": "kashfulpay",
    "nilachalstore": "nilachalstore", "nilachal": "nilachalstore", "নীলাচলস্টোর": "nilachalstore",
    "lalbaghmart": "lalbaghmart", "লালবাগমার্ট": "lalbaghmart",
    "aronyokbooks": "aronyokbooks", "aronyok books": "aronyokbooks", "আরণ্যকবুকস": "aronyokbooks",
    "bhorerbazar": "bhorerbazar", "ভোরেরবাজার": "bhorerbazar",
    "moniharfashion": "moniharfashion", "monihar": "moniharfashion", "মণিহারফ্যাশন": "moniharfashion",
    "sonargaoexpress": "sonargaoexpress", "sonargao express": "sonargaoexpress", "সোনারগাঁওএক্সপ্রেস": "sonargaoexpress",
    "charukolahealth": "charukolahealth", "charukola clinic": "charukolahealth", "চারুকলাহেলথ": "charukolahealth",
    "dipshikhaacademy": "dipshikhaacademy", "dipshikha academy": "dipshikhaacademy", "দীপশিখাএকাডেমি": "dipshikhaacademy",
    "shitolcafe": "shitolcafe", "শীতলক্যাফে": "shitolcafe",
    "deshnet": "deshnet", "দেশনেট": "deshnet",
    "polashpay": "polashpay", "পলাশপে": "polashpay",
    "turagbank": "turagbank", "তুরাগব্যাংক": "turagbank",
    "chaitimart": "chaitimart", "চৈতীমার্ট": "chaitimart",
    "probashicourier": "probashicourier", "প্রবাসী কুরিয়ার": "probashicourier", "প্রবাসীকুরিয়ার": "probashicourier",
    "nandanbazar": "nandanbazar", "নন্দনবাজার": "nandanbazar",
    "utshobfashion": "utshobfashion", "উৎসবফ্যাশন": "utshobfashion",
    "ekusheytech": "ekusheytech", "একুশেটেক": "ekusheytech",
    "korotoatel": "korotoatel", "করতোয়াটেল": "korotoatel", "করতোয়াটেল": "korotoatel",
    "shiulipay": "shiulipay", "শিউলিপে": "shiulipay",
    "rupshabank": "rupshabank", "রূপসাব্যাংক": "rupshabank",
    "shimulmart": "shimulmart", "শিমুলমার্ট": "shimulmart",
    "bakulbazar": "bakulbazar", "বকুলবাজার": "bakulbazar",
    "mayurtech": "mayurtech", "ময়ূরটেক": "mayurtech", "ময়ূরটেক": "mayurtech",
    "palkicourier": "palkicourier", "পালকিকুরিয়ার": "palkicourier", "পালকিকুরিয়ার": "palkicourier",
    "পালকি কুরিয়ার": "palkicourier", "পালকি কুরিয়ার": "palkicourier",
    "kodomstyle": "kodomstyle", "কদমস্টাইল": "kodomstyle",
    "juitel": "juitel", "জুঁইটেল": "juitel",
    "madhobipay": "madhobipay", "মাধবীপে": "madhobipay",
    "ichamotibank": "ichamotibank", "ইছামতিব্যাংক": "ichamotibank", "ইছামতী ব্যাংক": "ichamotibank",
    "nokshimart": "nokshimart", "নকশিমার্ট": "nokshimart",
    "doyelbazar": "doyelbazar", "দোয়েলবাজার": "doyelbazar", "দোয়েলবাজার": "doyelbazar",
    "shankhotech": "shankhotech", "শঙ্খটেক": "shankhotech",
    "boikalicourier": "boikalicourier", "বৈকালীকুরিয়ার": "boikalicourier", "বৈকালীকুরিয়ার": "boikalicourier",
    "বৈকালী কুরিয়ার": "boikalicourier", "বৈকালী কুরিয়ার": "boikalicourier",
    "titasstyle": "titasstyle", "তিতাসস্টাইল": "titasstyle",
    # Batch 5 brands
    "surjotel": "surjotel", "সূর্যটেল": "surjotel",
    "belimart": "belimart", "বেলীমার্ট": "belimart",
    "kajoltech": "kajoltech", "কাজলটেক": "kajoltech",
    "komolpay": "komolpay", "কমলপে": "komolpay",
    "karnaphulibank": "karnaphulibank", "কর্ণফুলীব্যাংক": "karnaphulibank", "কর্ণফুলী ব্যাংক": "karnaphulibank",
    "toricourier": "toricourier", "তরীকুরিয়ার": "toricourier", "তরীকুরিয়ার": "toricourier",
    "তরী কুরিয়ার": "toricourier", "তরী কুরিয়ার": "toricourier",
    "champastyle": "champastyle", "চম্পাস্টাইল": "champastyle",
    "kanchanbazar": "kanchanbazar", "কাঞ্চনবাজার": "kanchanbazar",
    # Batch 6 brands
    "akashnet": "akashnet", "আকাশনেট": "akashnet",
    "piyalimart": "piyalimart", "পিয়ালীমার্ট": "piyalimart", "পিয়ালীমার্ট": "piyalimart",
    "nibirtech": "nibirtech", "নিবিড়টেক": "nibirtech", "নিবিড়টেক": "nibirtech",
    "tonupay": "tonupay", "তনুপে": "tonupay",
    "barshabank": "barshabank", "বর্ষাব্যাংক": "barshabank", "বর্ষা ব্যাংক": "barshabank",
    "ujancourier": "ujancourier", "উজানকুরিয়ার": "ujancourier", "উজানকুরিয়ার": "ujancourier",
    "উজান কুরিয়ার": "ujancourier", "উজান কুরিয়ার": "ujancourier",
    "arshistyle": "arshistyle", "আরশিস্টাইল": "arshistyle",
    "kakonbazar": "kakonbazar", "কাঁকনবাজার": "kakonbazar",
}


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
    "weekend offer", "weekend flash offer", "flash offer", "special offer", "mega offer",
    "weekend deal", "flash sale", "mega sale", "eid offer", "winter offer", "summer offer",
    "holiday offer", "bonus offer", "limited offer", "today's offer", "super offer",
    "exclusive offer",
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


_BN_INFLECTION_SUFFIX = r"(?:-?(?:ের|এ|এর|ে|তে|কে|র|য়|য়|য়ে|য়ে|দের))?"
_LAT_HYPHEN_INFLECTION_SUFFIX = r"(?:['’]s|-(?:er|e|te|ke|r|y|ye|der|ের|এ|এর|ে|তে|কে|র|য়|য়|য়ে|য়ে|দের))?"
_LAT_FULL_INFLECTION_SUFFIX = r"(?:['’]s|-(?:er|e|te|ke|r|y|ye|der|ের|এ|এর|ে|তে|কে|র|য়|য়|য়ে|য়ে|দের)|(?:er|e|te|ke|r|y|ye|der))?"


def _sorted_nfc_brand_tokens(brands: Iterable[str] | None = None) -> list[str]:
    return sorted(
        {
            unicodedata.normalize("NFC", normalize_text(b))
            for b in (DEFAULT_KNOWN_BRANDS if brands is None else brands)
            if normalize_text(b)
        },
        key=len,
        reverse=True,
    )


def compile_brand_base_regex(brands: Iterable[str] | None = None) -> re.Pattern[str]:
    """Compile a longest-first regex matching uninflected base brand aliases."""
    tokens = _sorted_nfc_brand_tokens(brands)
    if not tokens:
        return re.compile(r"(?!x)x")
    parts = [re.sub(r"\\ ", r"\\s+", re.escape(tok)) for tok in tokens]
    return re.compile("|".join(parts), re.IGNORECASE)


def compile_brand_regex(brands: Iterable[str] | None = None) -> re.Pattern[str]:
    """Compile a boundary-aware case-insensitive regex matching brand tokens and Bengali/Banglish/English inflections."""
    tokens = _sorted_nfc_brand_tokens(brands)
    if not tokens:
        return re.compile(r"(?!x)x")
    parts: list[str] = []
    for tok in tokens:
        esc = re.sub(r"\\ ", r"\\s+", re.escape(tok))
        if _BN_CHAR_RE.search(tok):
            parts.append(rf"(?<![\u0980-\u09FF]){esc}{_BN_INFLECTION_SUFFIX}(?![\u0980-\u09FF])")
        elif len(tok) >= 4 and " " not in tok:
            parts.append(rf"(?<![A-Za-z0-9_]){esc}{_LAT_FULL_INFLECTION_SUFFIX}(?![A-Za-z0-9_])")
        else:
            parts.append(rf"(?<![A-Za-z0-9_]){esc}{_LAT_HYPHEN_INFLECTION_SUFFIX}(?![A-Za-z0-9_])")
    return re.compile("|".join(parts), re.IGNORECASE)


_DEFAULT_BRAND_RE = compile_brand_regex(DEFAULT_KNOWN_BRANDS)
_DEFAULT_BRAND_BASE_RE = compile_brand_base_regex(DEFAULT_KNOWN_BRANDS)
_DEFAULT_CANONICAL_MAP_NFC: dict[str, str] = {
    unicodedata.normalize("NFC", normalize_text(k)).lower(): v
    for k, v in DEFAULT_CANONICAL_BRAND_ALIASES.items()
}


def detect_brand_mentions(
    text: str,
    *,
    alias_map: dict[str, str] | None = None,
    brands: Iterable[str] | None = None,
    dynamic_fallback: bool = True,
    brand_pattern: re.Pattern[str] | None = None,
    base_pattern: re.Pattern[str] | None = None,
    already_contact_masked: bool = False,
) -> list[str]:
    """Detect canonical brand keys using the same contact-masking, NFC normalization, and regex logic as mask_brands."""
    eff_alias_map = (
        {unicodedata.normalize("NFC", normalize_text(k)).lower(): v for k, v in alias_map.items()}
        if alias_map is not None
        else _DEFAULT_CANONICAL_MAP_NFC
    )
    if brand_pattern is not None:
        pattern = brand_pattern
        b_pat = base_pattern or compile_brand_base_regex(
            list(alias_map.keys()) if (brands is None and alias_map is not None) else brands
        )
    elif brands is not None:
        pattern = compile_brand_regex(brands)
        b_pat = base_pattern or compile_brand_base_regex(brands)
    elif alias_map is not None:
        alias_keys = list(eff_alias_map.keys())
        pattern = compile_brand_regex(alias_keys)
        b_pat = base_pattern or compile_brand_base_regex(alias_keys)
    else:
        pattern = _DEFAULT_BRAND_RE
        b_pat = base_pattern or _DEFAULT_BRAND_BASE_RE

    work = (
        str(text or "")
        if already_contact_masked
        else mask_shortcuts(str(text or ""), mask_urls=True, mask_phones=True)
    )
    work_nfc = unicodedata.normalize("NFC", work)

    found: list[str] = []
    seen: set[str] = set()
    for m in pattern.finditer(work_nfc):
        bm = b_pat.match(m.group(0))
        raw_base = bm.group(0) if bm else m.group(0)
        key = unicodedata.normalize("NFC", normalize_text(raw_base)).lower()
        canon = eff_alias_map.get(key, key)
        if canon and canon not in seen:
            found.append(canon)
            seen.add(canon)

    if dynamic_fallback:
        residual = pattern.sub("<BRAND>", work_nfc)
        for tok in extract_dynamic_brands(residual):
            key = unicodedata.normalize("NFC", normalize_text(tok)).lower()
            canon = eff_alias_map.get(key, key)
            if canon and canon not in seen:
                found.append(canon)
                seen.add(canon)

    return found


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
    out = pattern.sub(repl, unicodedata.normalize("NFC", str(text or "")))
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
    canon_masked = unicodedata.normalize("NFC", mask_shortcuts(raw, mask_urls=True, mask_phones=True))
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
