"""Stage A-followup (`curation_v2`): apply train-only label, quarantine, and language audits on top of `curation_v1`."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .schema import CANONICAL_LABELS

ALLOWED_LANGUAGES = ("Bengali", "Banglish", "English", "CodeMix")

_URGENCY_RE = re.compile(
    r"\b(?:urgent|immediately|hurry|limited|suspend(?:ed)?|block(?:ed)?|last day|expire|deadline)\b"
    r"|জরুরি|দ্রুত|অবিলম্বে|শেষ দিন|বন্ধ|স্থগিত|\b(?:druto|taratari|bondho|shesh din)\b",
    re.IGNORECASE,
)
_OTP_RE = re.compile(r"\b(?:otp|pin|verification code|verify)\b|পিন|ভেরিফাই|যাচাই", re.IGNORECASE)

# Keyword heuristics for Stage B domain/scenario coverage audit on eligible_train only
_DOMAIN_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "banking_mfs",
        re.compile(
            r"bkash|nagad|rocket|upay|dbbl|brac bank|sonali|islami bank|bank|account|atm|cvv|loan|emi|bdt|বিকাশ|নগদ|রকেট|ব্যাংক|একাউন্ট|লোন",
            re.IGNORECASE,
        ),
    ),
    (
        "telecom",
        re.compile(
            r"\bgb\b|\bmb\b|mygp|grameenphone|robi|banglalink|airtel|teletalk|skitto|sim|recharge|minute|min\b|talktime|dial \*|\*\d+\*|জিবি|এমবি|রিচার্জ|মিনিট|সিম",
            re.IGNORECASE,
        ),
    ),
    (
        "delivery_parcel",
        re.compile(
            r"parcel|courier|delivery|shipping|tracking|steadfast|pathao|redx|sundarban|পার্সেল|কুরিয়ার|ডেলিভারি",
            re.IGNORECASE,
        ),
    ),
    (
        "shopping_ecommerce",
        re.compile(
            r"daraz|chaldal|shwapno|unimart|pickaboo|rokomari|evaly|aarong|bata|kfc|mcdonald|foodpanda|discount|cashback|sale|offer|voucher|coupon|ডিসকাউন্ট|ক্যাশব্যাক|অফার|শপিং",
            re.IGNORECASE,
        ),
    ),
    (
        "education",
        re.compile(
            r"\.edu\.bd|university|college|school|admission|tuition|exam|ssc|hsc|eiin|student|scholarship|library|class|assignment|কলেজ|বিশ্ববিদ্যালয়|ভর্তি|পরীক্ষা|বৃত্তি|টিউশন",
            re.IGNORECASE,
        ),
    ),
    (
        "employment",
        re.compile(
            r"job|salary|income|earn|hiring|vacancy|interview|recruitment|work from home|part.?time|চাকরি|বেতন|আয়|ইনকাম|নিয়োগ|কাজ",
            re.IGNORECASE,
        ),
    ),
    (
        "civic_utility_health",
        re.compile(
            r"btrc|brta|passport|nid|election|govt|ministry|municipality|museum|vaccine|hospital|thalassemia|নির্বাচন|পাসপোর্ট|সরকার|হাসপাতাল|বিটিআরসি",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class AuditRuleGroup:
    """Batch specification for audited label, quarantine, or language updates."""

    ids: tuple[str, ...]
    action: str
    audit_source: str
    category: str
    rationale: str
    new_label: str | None = None
    new_language: str | None = None


# ---- Audited Label & Quarantine Rules (from auditor-src #23, strictly train_pool) ----

LABEL_AND_QUARANTINE_GROUPS: tuple[AuditRuleGroup, ...] = (
    AuditRuleGroup(
        ids=(
            "cur_0fbe4b0675d3",
            "cur_7dbf44b477df",
            "cur_fb578cc13f34",
            "cur_bbe83e1101ae",
            "cur_bfd84595bfa8",
        ),
        action="reclassify_label",
        new_label="Smishing",
        audit_source="auditor-src",
        category="emergency_family_impersonation_scam",
        rationale="Emergency family-in-danger/hospital urgent money extraction demand; multilingual sibling of legacy-conflict smishing.",
    ),
    AuditRuleGroup(
        ids=("cur_0123e9a18921", "cur_10b71143b61b", "cur_59a1b9b252c9", "cur_6d229b4ac2b4", "cur_9e1570c94434"),
        action="quarantine_ambiguous",
        audit_source="auditor-src",
        category="unattributed_account_deduction_alert",
        rationale="Unattributed account deduction alert lacking verified bank/sender identity; sibling of quarantined legacy conflict.",
    ),
    AuditRuleGroup(
        ids=("cur_19da9d42aaa3", "cur_438871060bb8", "cur_7f0a5209ec4d"),
        action="quarantine_ambiguous",
        audit_source="auditor-src",
        category="unattributed_mfs_account_check",
        rationale="Asks whether 50,000 TK account is open on a mobile number without sender context.",
    ),
    AuditRuleGroup(
        ids=("cur_1c77e92e38ec", "cur_40905f2db2c0"),
        action="quarantine_ambiguous",
        audit_source="auditor-src",
        category="unverified_utility_disconnect_threat",
        rationale="Utility disconnection demand with bare contact and no verified institutional channel.",
    ),
    AuditRuleGroup(
        ids=("cur_2d3f1d889026", "cur_8064e2d283a8", "cur_b95a75f6c81d"),
        action="quarantine_ambiguous",
        audit_source="auditor-src",
        category="unverified_servicing_callback",
        rationale="Car servicing deadline with bare mobile number lacking verified dealership identity.",
    ),
    AuditRuleGroup(
        ids=("cur_791f2a063e48", "cur_7c0822f4994b", "cur_9d75f4d2707a"),
        action="quarantine_ambiguous",
        audit_source="auditor-src",
        category="unverified_delivery_callback",
        rationale="Parcel/package delivery callback to bare mobile number without courier/order verification.",
    ),
    AuditRuleGroup(
        ids=(
            "cur_0323993bd88c",
            "cur_03b80f093ce1",
            "cur_172ca1ed7e4f",
            "cur_22be412bce99",
            "cur_2c090ec290f0",
            "cur_48d346fa4f47",
            "cur_4d2918f221b5",
            "cur_6c912611e510",
            "cur_7f46e66701c2",
            "cur_8e6c8958ea08",
            "cur_8f1761de3238",
            "cur_a6436b25d947",
            "cur_a64e0a0be848",
            "cur_a98f098cd0ab",
            "cur_b5a3aff61ac5",
            "cur_e233e8b10a34",
            "cur_fa49d6eaf474",
        ),
        action="quarantine_ambiguous",
        audit_source="auditor-src",
        category="unattributed_payment_or_bill_demand",
        rationale="Payment or electricity bill due demand without verifiable sender identity or context.",
    ),
    AuditRuleGroup(
        ids=(
            "cur_0f5a200cdcc9",
            "cur_13656fc55fbc",
            "cur_153c5329c527",
            "cur_1a083d05dd16",
            "cur_1a879c8faaa4",
            "cur_1ccc6c80e455",
            "cur_1fb625cdc75e",
            "cur_23574b6aff15",
            "cur_23b723e9b727",
            "cur_27d583cf7d72",
            "cur_30a3af4101db",
            "cur_37c84c657d8e",
            "cur_384432fdf2f5",
            "cur_3c4eecb6e461",
            "cur_41091a98e682",
            "cur_4289ca085c67",
            "cur_47b41eaf53d5",
            "cur_4ed18cd0e862",
            "cur_5111d9abdf67",
            "cur_581265acdd26",
            "cur_5db77bd1a540",
            "cur_61c603b9154a",
            "cur_62b88e3fecb4",
            "cur_62f95017d791",
            "cur_67efc224a58a",
            "cur_6bec175d959b",
            "cur_709ffb3dce96",
            "cur_760d2b28fded",
            "cur_78ed67158cf1",
            "cur_7e7f8314bbfd",
            "cur_8259d03cb502",
            "cur_8276be052b9c",
            "cur_82c2f73c2f77",
            "cur_83ab200c1aee",
            "cur_864b0dbdcf7a",
            "cur_8b1ac3235fca",
            "cur_8cff4c0559d7",
            "cur_8f0e1030eda8",
            "cur_8f5d3b93a744",
            "cur_8fd2ec15351f",
            "cur_99c25e300aab",
            "cur_9a374b9d9839",
            "cur_9e6ab5e299a8",
            "cur_a4fc231da4a8",
            "cur_a74e4d1a36e5",
            "cur_a76af96fec4e",
            "cur_a78bc15ea9f0",
            "cur_ab7adec5bf37",
            "cur_b44643d3ac91",
            "cur_b648adb2ffb2",
            "cur_b715223d2151",
            "cur_b99d52200a16",
            "cur_bf3702d1afb7",
            "cur_c0dbcd3506a0",
            "cur_c6805efca154",
            "cur_cdc471512fb1",
            "cur_cf5e3f608ac5",
            "cur_d68506273026",
            "cur_d87ae360a290",
            "cur_da9c60ed264c",
            "cur_dc1cd806d1d2",
            "cur_e25689dee95e",
            "cur_e513e73b25c8",
            "cur_ea4e766974b6",
            "cur_ed95f6c9f946",
            "cur_f2209ee0714a",
            "cur_f39d2259f30a",
            "cur_f3ff7800f923",
            "cur_f773e0ad2d13",
            "cur_f7905cfbfe0e",
            "cur_fa0e50e36eda",
            "cur_3f6778c7d4f5",
        ),
        action="quarantine_ambiguous",
        audit_source="auditor-src",
        category="bare_number_money_or_call_demand",
        rationale="Bare phone-number money/bKash/Rocket/urgent-call demand without verifiable personal relationship or sender context.",
    ),
    AuditRuleGroup(
        ids=("cur_0e9d76955d8c", "cur_1b1c796ba7cb", "cur_44daaf174c37", "cur_a4e93c47d8be", "cur_c511723d9647", "cur_fc6419283108"),
        action="reclassify_label",
        new_label="Normal",
        audit_source="auditor-src",
        category="transactional_pin_otp_notice",
        rationale="Legitimate MyGP PIN/OTP verification code; transactional service notice, not promotional marketing.",
    ),
    AuditRuleGroup(
        ids=("cur_74eb46b567ca", "cur_b0738f7890f0", "cur_daa4c76e6d94"),
        action="reclassify_label",
        new_label="Normal",
        audit_source="auditor-src",
        category="transactional_call_balance_receipt",
        rationale="Telecom call-duration and balance-charge receipt; transactional service message.",
    ),
    AuditRuleGroup(
        ids=(
            "cur_2364fa346816",
            "cur_2de14c140749",
            "cur_2e5633d63e9a",
            "cur_2fcc9eeb1c74",
            "cur_306e47b4d88c",
            "cur_45fc76484ff7",
            "cur_466e85b0d688",
            "cur_46939f35f846",
            "cur_9b3dfbd92cbf",
            "cur_ee1693809c8f",
        ),
        action="reclassify_label",
        new_label="Normal",
        audit_source="auditor-src",
        category="service_maintenance_downtime_notice",
        rationale="Scheduled system upgrade / maintenance downtime notice; informational service communication.",
    ),
    AuditRuleGroup(
        ids=("cur_0635a8ea96e0", "cur_75de46c8c101", "cur_9eb339967177", "cur_c27c3d6d8d48"),
        action="reclassify_label",
        new_label="Normal",
        audit_source="auditor-src",
        category="regulatory_btrc_handset_notice",
        rationale="Informational BTRC handset auto-registration notice with no commercial offer.",
    ),
    AuditRuleGroup(
        ids=(
            "cur_145a85c8ed77",
            "cur_5f4d5d9b60a8",
            "cur_f079a003de2e",
            "cur_8c0550618565",
            "cur_930a8ad7f857",
            "cur_9b823d48bbf3",
            "cur_3aaf25a1b96f",
            "cur_591b86e77ccc",
            "cur_a27fe25d40f2",
        ),
        action="reclassify_label",
        new_label="Normal",
        audit_source="auditor-src",
        category="noncommercial_greeting_or_tribute",
        rationale="Non-commercial institutional greeting, customer anniversary, or national tribute without commercial offer.",
    ),
    AuditRuleGroup(
        ids=(
            "cur_109cb790b307",
            "cur_616f36dee290",
            "cur_5fa13eb34638",
            "cur_fe820e5e8118",
            "cur_afcc1e5bd32e",
            "cur_27d86b9fad0f",
            "cur_d25f498841f1",
            "cur_81cfeac63996",
            "cur_92c06955d1cb",
            "cur_11d6ff427e85",
            "cur_7d138ac0737c",
            "cur_d0a9bfca5264",
            "cur_24ad8f7ff697",
            "cur_79714872405a",
            "cur_2d06a31e01ee",
            "cur_cb295bcbbece",
            "cur_e9443b84b864",
            "cur_f7e2273426dd",
            "cur_ff89442a0f54",
            "cur_85a41218878b",
            "cur_c0fa675146c6",
            "cur_122a5a49f36f",
            "cur_9db339311373",
            "cur_d02f281f01ad",
            "cur_12a9aee3ccd1",
            "cur_20a1236207ec",
            "cur_2acbcbabd998",
            "cur_83a7933c382a",
            "cur_c342ed5dd832",
            "cur_cdc750b41a6f",
            "cur_7c68e7d02691",
            "cur_082df77b377f",
            "cur_a3ccc2a3d519",
            "cur_27cf82d95cd1",
            "cur_ff5cce626417",
            "cur_0469d229336d",
            "cur_b7b4568b9d0e",
        ),
        action="reclassify_label",
        new_label="Normal",
        audit_source="auditor-src",
        category="civic_regulatory_psa_notice",
        rationale="Non-commercial government/regulatory public-service or official informational notice.",
    ),
    AuditRuleGroup(
        ids=("cur_7bfe7ba00ecc", "cur_0360bf6f8a68"),
        action="quarantine_ambiguous",
        audit_source="auditor-src",
        category="unverified_advance_fee_or_recruitment_url",
        rationale="Suspicious advance-fee loan pitch or unverified recruitment URL lacking verifiable organizational provenance.",
    ),
    AuditRuleGroup(
        ids=("cur_29a9af987b37", "cur_31e1971161e6", "cur_5ff0f3d1021d", "cur_859c1b3b5389"),
        action="reclassify_label",
        new_label="Normal",
        audit_source="auditor-src",
        category="institutional_independence_day_greeting",
        rationale="Pure institutional Independence Day greeting (Sonali Bank) with zero links, phones, offers, or extraction.",
    ),
    AuditRuleGroup(
        ids=("cur_1d314028ec6a", "cur_533309022a68", "cur_a382ce2cd72c", "cur_ce0f87d0aa0e"),
        action="reclassify_label",
        new_label="Promotional",
        audit_source="auditor-src",
        category="identifiable_college_admission_ad",
        rationale="Identifiable educational institution (Comilla City College, EIIN:134648) admission outreach with no deceptive extraction.",
    ),
    AuditRuleGroup(
        ids=(
            "cur_0d892f8376ff",
            "cur_b73baa479b0c",
            "cur_c9d7bc0f7dd0",
            "cur_f195152a06c2",
            "cur_2d90ed195928",
            "cur_545e493e74b9",
            "cur_548f9b51dff4",
            "cur_587a6a139da6",
            "cur_ea81c918fa71",
            "cur_59cf5bcfc05a",
            "cur_6179166f47c5",
            "cur_f1d8e76fbdfc",
            "cur_a3dc0f60bb54",
            "cur_5cf3213c1e3b",
            "cur_c8439c207f87",
            "cur_09fdf22761c4",
            "cur_7e30b56947a9",
            "cur_ee34d6c95ad5",
            "cur_458beb467cf6",
        ),
        action="quarantine_ambiguous",
        audit_source="auditor-src",
        category="unverified_interview_or_unbranded_discount",
        rationale="Unverified outsourced interview/exam call, garbled recruitment text, unbranded discount mobile callback, or Agora gift callback lacking sender context/extraction mechanism.",
    ),
)

# ---- Audited Language Updates (from auditor-banglaphish #22, strictly train_pool) ----

LANGUAGE_AUDIT_GROUPS: tuple[AuditRuleGroup, ...] = (
    AuditRuleGroup(
        ids=("cur_b0738f7890f0",),
        action="reclassify_language",
        new_language="English",
        audit_source="auditor-banglaphish",
        category="script_contradiction_bengali_to_english",
        rationale="Pure English telecom call receipt mislabeled Bengali.",
    ),
    AuditRuleGroup(
        ids=("cur_c2954b3fa2fa",),
        action="reclassify_language",
        new_language="Banglish",
        audit_source="auditor-banglaphish",
        category="script_contradiction_bengali_to_banglish",
        rationale="Romanized Bengali smishing text mislabeled Bengali.",
    ),
    AuditRuleGroup(
        ids=("cur_a90f09715275",),
        action="reclassify_language",
        new_language="CodeMix",
        audit_source="auditor-banglaphish",
        category="script_contradiction_banglish_to_codemix",
        rationale="Contains Bengali-script token inside Banglish/English clauses.",
    ),
    AuditRuleGroup(
        ids=("cur_3f783e706d97", "cur_535c4bec6306", "cur_d9ae00d86ddd"),
        action="reclassify_language",
        new_language="Bengali",
        audit_source="auditor-banglaphish",
        category="script_contradiction_codemix_to_bengali",
        rationale="Pure Bengali-script text with zero Latin characters mislabeled CodeMix.",
    ),
    AuditRuleGroup(
        ids=(
            "cur_00a18f4f2793",
            "cur_3d9fe2f8911b",
            "cur_525017906c19",
            "cur_6fb4ae4fa4f7",
            "cur_8adb7c671e9c",
            "cur_960826e9e75a",
            "cur_9af40874880e",
            "cur_a237b94e5b75",
            "cur_b3fe5b921e13",
            "cur_daa4c76e6d94",
            "cur_e4e9dac99c55",
            "cur_edfa9327b75b",
        ),
        action="reclassify_language",
        new_language="English",
        audit_source="auditor-banglaphish",
        category="script_contradiction_codemix_to_english",
        rationale="Pure English text in Latin script mislabeled CodeMix.",
    ),
    AuditRuleGroup(
        ids=("cur_3a0d9641bbf7", "cur_9d08242983d4", "cur_d306b7620f6e"),
        action="reclassify_language",
        new_language="Banglish",
        audit_source="auditor-banglaphish",
        category="script_contradiction_codemix_to_banglish",
        rationale="Romanized Bengali / Bangladeshi greeting mislabeled CodeMix.",
    ),
    AuditRuleGroup(
        ids=("cur_7cb463b60342", "cur_998373d40f32", "cur_73cfc82a9391", "cur_90ff75f39b50", "cur_a0d8823bf294"),
        action="reclassify_language",
        new_language="CodeMix",
        audit_source="auditor-banglaphish",
        category="multiclause_english_or_banglish_in_bengali",
        rationale="Productive multi-word English/Banglish clauses mixed with Bengali script (not isolated loanwords/URLs).",
    ),
    AuditRuleGroup(
        ids=(
            "cur_007f9a118e22",
            "cur_02a0fde5fe37",
            "cur_02d7791ebf1b",
            "cur_0460dadb0c19",
            "cur_064c60bb0bbe",
            "cur_0a28fafe9f64",
            "cur_0f1cc565646a",
            "cur_0f976c4e95cb",
            "cur_10c0d68b27b2",
            "cur_10d1f67d1a25",
            "cur_10e667800a15",
            "cur_1352b47e7be4",
            "cur_149fd1588a5f",
            "cur_174a76e0a2be",
            "cur_193223c5b871",
            "cur_1b93a4a13435",
            "cur_1bb1fa07754e",
            "cur_1c932f232fe0",
            "cur_2283678275fd",
            "cur_2c53ebea7c3f",
            "cur_2dc72d6e77f4",
            "cur_2e0ae62fa721",
            "cur_38534df297e4",
            "cur_3b66df24d183",
            "cur_3bfb594f2e46",
            "cur_3edfe510f426",
            "cur_3fc558517b42",
            "cur_49295098a3ab",
            "cur_4996fec632cd",
            "cur_4b1cdf3b160f",
            "cur_52ebde2542f6",
            "cur_578b6ad2c7c1",
            "cur_58c32396c07a",
            "cur_5912cbb87702",
            "cur_5cd43ca0ba60",
            "cur_64398246d573",
            "cur_64ed60517a2d",
            "cur_66a9dd9b4b85",
            "cur_66fbdb0caf77",
            "cur_67d9557ff8fb",
            "cur_67f468bde5a5",
            "cur_69eb1730d72c",
            "cur_6f95c716fc85",
            "cur_71cd8a05f56a",
            "cur_71f6ffa7d3bc",
            "cur_760f32459251",
            "cur_77db746f01c4",
            "cur_7ae507065fee",
            "cur_7be782e9fee5",
            "cur_8184d22ec266",
            "cur_848c8cca4d9f",
            "cur_8af6abf61965",
            "cur_8af9c088b792",
            "cur_8dceaadfbef1",
            "cur_8e76c93c3b5e",
            "cur_906c5a64d59a",
            "cur_9807cd7f421a",
            "cur_9c96187f0b55",
            "cur_a126fa3dd256",
            "cur_a2e50642d4cd",
            "cur_a4f9cceab2a0",
            "cur_a515e9da96eb",
            "cur_a6162a7cc288",
            "cur_a715ec168d7d",
            "cur_a77c25138080",
            "cur_ac01124da94f",
            "cur_b2ae8b3417a9",
            "cur_b2d166d1147f",
            "cur_b42f94e06f99",
            "cur_b71561f7fa3d",
            "cur_b880c13106a8",
            "cur_bb96830c4ef8",
            "cur_bcf3d6b8d85a",
            "cur_bd34f2133aa3",
            "cur_bdacd3768ea7",
            "cur_be386414dc1a",
            "cur_bf78e70b42c9",
            "cur_bfca11261a62",
            "cur_c03cb98fafc7",
            "cur_c481367c14bf",
            "cur_c520db850b22",
            "cur_c61263b5bbd0",
            "cur_c6fbbc55438f",
            "cur_cc524971c545",
            "cur_cc74a046b351",
            "cur_cf190ab52f50",
            "cur_d3ee66651f1b",
            "cur_d45f90d6e3c9",
            "cur_da4f40e36878",
            "cur_df0fd11d3920",
            "cur_e0043e539745",
            "cur_e10d9b1f3b3e",
            "cur_e1634d8e803a",
            "cur_e35bf3895646",
            "cur_ebd070cf7cc1",
            "cur_eed02274d1c1",
            "cur_f0d4368c200a",
            "cur_f293904dfc7a",
            "cur_f564f4225dce",
            "cur_fc931ef47083",
            "cur_fe6ba6edaa4a",
        ),
        action="reclassify_language",
        new_language="English",
        audit_source="auditor-banglaphish",
        category="pure_english_mislabeled_banglish",
        rationale="100% English syntax/lexicon (with standard BD English currency/brand terms) mislabeled Banglish.",
    ),
    AuditRuleGroup(
        ids=(
            "cur_c2f52daf6d76",
            "cur_cb7d44f8e298",
            "cur_072490350e2e",
            "cur_1c1f4e743ea4",
            "cur_fb19f4b1fc01",
            "cur_6e8208f70719",
            "cur_fb056ff1d1bf",
            "cur_7d638a1ba523",
            "cur_58d36447a259",
            "cur_8939988c60af",
            "cur_90a9c21e86b5",
            "cur_e8ea9f12a72f",
            "cur_8d4abe91e77d",
            "cur_6b3a856af954",
            "cur_7c3c36d90c3a",
            "cur_a92b63b95c9b",
            "cur_73237f0f2eb1",
            "cur_f9769c839efc",
            "cur_79d654dbe14e",
            "cur_e4f0065d210c",
            "cur_037f77f339dd",
            "cur_b1d8feadbaa3",
            "cur_f53eeccd5005",
            "cur_c2256dbe0626",
            "cur_b772bd90d000",
            "cur_92075e92bed0",
            "cur_afe7da98d8bd",
            "cur_a2d092ad6942",
            "cur_42ebaceddfdb",
            "cur_7dca7ede1cff",
            "cur_85377e019513",
            "cur_e596ed43c050",
            "cur_7aa6e54c4c32",
            "cur_c0f2d05a36be",
            "cur_f6207ef3d975",
            "cur_bc354f158aff",
            "cur_a8c5597eaa46",
            "cur_9a591e7b336c",
            "cur_d03f14975625",
            "cur_ede58817130a",
            "cur_095d19666fda",
            "cur_e0e7cb226218",
            "cur_e2564e4b0312",
            "cur_20868413073d",
            "cur_29e4dd5f2649",
            "cur_26c7ef3974ee",
            "cur_3f1056164423",
            "cur_b02b0c3649a1",
            "cur_5a13197bcdc3",
            "cur_cb67968e0715",
            "cur_d25ba5cca69d",
        ),
        action="reclassify_language",
        new_language="CodeMix",
        audit_source="auditor-banglaphish",
        category="latin_script_english_banglish_codemix",
        rationale="Productive multi-clause or syntactic switching between English and Banglish in Latin script.",
    ),
)

# Boundary exemplars kept unchanged with explicit documented rationale
KEPT_WITH_NOTE_GROUPS: tuple[AuditRuleGroup, ...] = (
    AuditRuleGroup(
        ids=("cur_15b7a3aa30d3", "cur_b576aaa18f77", "cur_1d6dbae3d37e", "cur_e8946fc1c569"),
        action="keep_with_note",
        audit_source="auditor-src",
        category="legitimate_edu_tuition_notice",
        rationale="Verified .edu.bd university tuition/service notice with URL; kept Normal per DATASET_GUIDE.md.",
    ),
    AuditRuleGroup(
        ids=("cur_fdd6be320dbf", "cur_b8bfebe787a5", "cur_74e4f7028ca9", "cur_1c8088d01b99", "cur_b9e8b474581a", "cur_d79e736d764a", "cur_f3e3562e645f"),
        action="keep_with_note",
        audit_source="auditor-src",
        category="civic_event_or_contest_promotion",
        rationale="Public event/film/contest promotion inviting active participation; kept Promotional under consistent outreach rule.",
    ),
    AuditRuleGroup(
        ids=("cur_bcfffd989eae",),
        action="keep_with_note",
        audit_source="auditor-banglaphish",
        category="latin_script_codemix_exemplar",
        rationale="Banglish+English clause switching ('Tumi onek strong, you can do it.'); kept CodeMix.",
    ),
    AuditRuleGroup(
        ids=("cur_ad3b44457ebb", "cur_7a08a02e9d98"),
        action="keep_with_note",
        audit_source="auditor-banglaphish",
        category="banglish_with_loanwords",
        rationale="Banglish matrix syntax with isolated English telecom/commerce loanwords; kept Banglish.",
    ),
)


def infer_scenario_domain(text: str) -> str:
    """Classify an SMS into a coarse Stage-B planning domain bucket."""
    for domain_name, pattern in _DOMAIN_RULES:
        if pattern.search(text):
            return domain_name
    return "personal_or_general"


def compute_eligible_coverage_profile(eligible_df: pd.DataFrame) -> dict[str, Any]:
    """Compute Stage B coverage matrix & deficit metrics strictly on audited eligible_train rows."""
    work = eligible_df.copy()
    work["len_band"] = pd.cut(
        work["char_len"], bins=[-1, 49, 120, 100000], labels=["short(<50)", "medium(50-120)", "long(>120)"]
    ).astype(str)
    work["has_contact"] = work["has_url"] | work["has_phone"] | work["has_ussd"]
    work["has_urgency"] = work["text"].str.contains(_URGENCY_RE, regex=True)
    work["has_otp_cue"] = work["text"].str.contains(_OTP_RE, regex=True)
    work["domain_bucket"] = work["text"].map(infer_scenario_domain)

    cls_counts = {lbl: int((work["label_3class"] == lbl).sum()) for lbl in CANONICAL_LABELS}
    target_t = max(cls_counts.values()) if cls_counts else 0
    deficits = {lbl: int(max(0, target_t - cnt)) for lbl, cnt in cls_counts.items()}

    def _xtab(col: str) -> dict[str, dict[str, int]]:
        ct = pd.crosstab(work["label_3class"], work[col])
        return {str(idx): {str(c): int(ct.loc[idx, c]) for c in ct.columns} for idx in ct.index}

    return {
        "eligible_train_total": int(len(work)),
        "eligible_train_families": int(work["family_id"].nunique()),
        "eligible_train_by_class": cls_counts,
        "stage_b_target_T": int(target_t),
        "stage_b_deficits": deficits,
        "stage_b_total_deficit": int(sum(deficits.values())),
        "class_by_language": _xtab("language"),
        "class_by_length_band": _xtab("len_band"),
        "class_by_domain_bucket": _xtab("domain_bucket"),
        "class_by_has_url": _xtab("has_url"),
        "class_by_has_contact": _xtab("has_contact"),
        "class_by_has_urgency": _xtab("has_urgency"),
        "class_by_has_otp_cue": _xtab("has_otp_cue"),
    }


def build_pilot_plan(coverage: dict[str, Any], max_pilot: int = 300) -> dict[str, Any]:
    """Allocate a <=300-row Stage B pilot strictly within measured deficits and language/surface gaps."""
    deficits = coverage["stage_b_deficits"]
    norm_quota = min(deficits["Normal"], 39)
    promo_quota = min(deficits["Promotional"], max_pilot - norm_quota)
    smish_quota = 0

    # Allocate Promotional pilot proportionally toward Banglish/Bengali/CodeMix shortfalls vs Smishing/Normal
    promo_by_lang = {
        "Banglish": 115,
        "Bengali": 65,
        "CodeMix": 60,
        "English": promo_quota - (115 + 65 + 60),
    }
    norm_by_lang = {
        "CodeMix": 14,
        "Bengali": 10,
        "Banglish": 8,
        "English": norm_quota - (14 + 10 + 8),
    }
    return {
        "max_pilot_rows": max_pilot,
        "pilot_total_rows": norm_quota + promo_quota + smish_quota,
        "pilot_by_class": {"Normal": norm_quota, "Promotional": promo_quota, "Smishing": smish_quota},
        "normal_by_language": norm_by_lang,
        "promotional_by_language": promo_by_lang,
    }


def _collect_audit_maps(
    v1_eligible_ids: set[str],
    label_groups: tuple[AuditRuleGroup, ...] = LABEL_AND_QUARANTINE_GROUPS,
    lang_groups: tuple[AuditRuleGroup, ...] = LANGUAGE_AUDIT_GROUPS,
) -> tuple[dict[str, tuple[str, AuditRuleGroup]], dict[str, AuditRuleGroup], dict[str, AuditRuleGroup]]:
    """Validate audit rule groups against train-only eligible IDs and flatten into lookup maps."""
    label_actions: dict[str, tuple[str, AuditRuleGroup]] = {}
    for grp in label_groups:
        for cid in grp.ids:
            if cid not in v1_eligible_ids:
                raise ValueError(f"Audit ID {cid!r} is not in curation_v1 eligible_train.parquet (train_pool guard violated).")
            if cid in label_actions:
                raise ValueError(f"Duplicate label/quarantine audit ID {cid!r}.")
            label_actions[cid] = (grp.action, grp)

    lang_actions: dict[str, AuditRuleGroup] = {}
    for grp in lang_groups:
        if grp.new_language not in ALLOWED_LANGUAGES:
            raise ValueError(f"Invalid target language {grp.new_language!r}.")
        for cid in grp.ids:
            if cid not in v1_eligible_ids:
                raise ValueError(f"Language audit ID {cid!r} is not in curation_v1 eligible_train.parquet (train_pool guard violated).")
            if cid in lang_actions:
                raise ValueError(f"Duplicate language audit ID {cid!r}.")
            lang_actions[cid] = grp

    kept_notes: dict[str, AuditRuleGroup] = {}
    for grp in KEPT_WITH_NOTE_GROUPS:
        for cid in grp.ids:
            if cid in v1_eligible_ids:
                kept_notes[cid] = grp

    return label_actions, lang_actions, kept_notes


def run_stage_a_curation_v2(
    v1_dir: str | Path,
    v2_dir: str | Path,
    label_groups: tuple[AuditRuleGroup, ...] = LABEL_AND_QUARANTINE_GROUPS,
    lang_groups: tuple[AuditRuleGroup, ...] = LANGUAGE_AUDIT_GROUPS,
) -> dict[str, Any]:
    """Produce `curation_v2` by applying verified train-only audits on top of read-only `curation_v1`."""
    v1_path = Path(v1_dir)
    v2_path = Path(v2_dir)
    v2_path.mkdir(parents=True, exist_ok=True)

    existing_v1 = pd.read_parquet(v1_path / "existing_unique.parquet")
    eligible_v1 = pd.read_parquet(v1_path / "eligible_train.parquet")
    quarantined_v1 = pd.read_parquet(v1_path / "quarantined.parquet")
    provenance_v1 = pd.read_parquet(v1_path / "provenance.parquet")

    v1_eligible_ids = set(eligible_v1["canonical_id"])
    label_actions, lang_actions, kept_notes = _collect_audit_maps(v1_eligible_ids, label_groups, lang_groups)

    # Guard: ensure no modified family has unmodified siblings in eligible_train
    modified_ids = set(label_actions)
    mod_fams = set(eligible_v1.loc[eligible_v1["canonical_id"].isin(modified_ids), "family_id"])
    unmod_sibs = eligible_v1[eligible_v1["family_id"].isin(mod_fams) & ~eligible_v1["canonical_id"].isin(modified_ids)]
    if len(unmod_sibs) > 0:
        raise ValueError(f"Found unmodified family siblings in eligible_train: {unmod_sibs['canonical_id'].tolist()}")

    existing_v2 = existing_v1.copy()
    provenance_v2 = provenance_v1.copy()

    # Build audit decision log rows and apply updates on train_pool rows only
    audit_records: list[dict[str, Any]] = []
    newly_quarantined_cids: dict[str, tuple[str, str]] = {}  # cid -> (qid, reason)

    v1_by_cid = existing_v1.set_index("canonical_id")
    all_touched_cids = sorted(set(label_actions) | set(lang_actions) | set(kept_notes))
    for cid in all_touched_cids:
        orig_row = v1_by_cid.loc[cid]
        if orig_row["eval_group"] != "train_pool" or not bool(orig_row["eligible_for_training"]):
            raise ValueError(f"Attempted to audit non-train_pool row {cid!r}.")

        orig_lbl = str(orig_row["label_3class"])
        orig_lang = str(orig_row["language"])
        new_lbl = orig_lbl
        new_lang = orig_lang
        final_status = "kept_in_eligible_train"
        action_parts: list[str] = []
        sources: list[str] = []
        categories: list[str] = []
        rationales: list[str] = []

        if cid in label_actions:
            act, grp = label_actions[cid]
            sources.append(grp.audit_source)
            categories.append(grp.category)
            rationales.append(grp.rationale)
            if act == "quarantine_ambiguous":
                action_parts.append("quarantine_ambiguous")
                final_status = "quarantined"
                new_lbl = f"Quarantined:{orig_lbl}"
                qid = f"quar_{str(orig_row['text_hash_strict'])[:12]}"
                newly_quarantined_cids[cid] = (qid, f"audit_v2_{grp.category}")
            elif act == "reclassify_label":
                assert grp.new_label in CANONICAL_LABELS
                action_parts.append(f"reclassify_label:{orig_lbl}->{grp.new_label}")
                new_lbl = grp.new_label
                existing_v2.loc[existing_v2["canonical_id"] == cid, "label_3class"] = new_lbl
                existing_v2.loc[existing_v2["canonical_id"] == cid, "label_binary_risk"] = 1 if new_lbl == "Smishing" else 0

        if cid in lang_actions:
            lgrp = lang_actions[cid]
            assert lgrp.new_language in ALLOWED_LANGUAGES
            action_parts.append(f"reclassify_language:{orig_lang}->{lgrp.new_language}")
            new_lang = lgrp.new_language
            existing_v2.loc[existing_v2["canonical_id"] == cid, "language"] = new_lang
            sources.append(lgrp.audit_source)
            categories.append(lgrp.category)
            rationales.append(lgrp.rationale)

        if cid in kept_notes and not action_parts:
            kgrp = kept_notes[cid]
            action_parts.append("keep_with_note")
            sources.append(kgrp.audit_source)
            categories.append(kgrp.category)
            rationales.append(kgrp.rationale)

        audit_records.append(
            {
                "canonical_id": cid,
                "family_id": str(orig_row["family_id"]),
                "primary_orig_id": str(orig_row["primary_orig_id"]),
                "orig_label_3class": orig_lbl,
                "audited_label_3class": new_lbl,
                "orig_language": orig_lang,
                "audited_language": new_lang,
                "final_status": final_status,
                "actions": "|".join(action_parts),
                "audit_sources": "|".join(dict.fromkeys(sources)),
                "categories": "|".join(dict.fromkeys(categories)),
                "rationale": " ; ".join(rationales),
                "text": str(orig_row["text"]),
            }
        )

    # Append newly quarantined rows to quarantined_v2 and remove from existing_v2
    new_quar_rows: list[dict[str, Any]] = []
    for cid, (qid, reason) in sorted(newly_quarantined_cids.items()):
        r = v1_by_cid.loc[cid]
        matching_prov = provenance_v1[provenance_v1["canonical_id"] == cid]
        obs_orig = "|".join(sorted(matching_prov["orig_label"].astype(str).unique())) if len(matching_prov) else str(r["label_3class"])
        obs_map = "|".join(sorted(matching_prov["mapped_label_3class"].astype(str).unique())) if len(matching_prov) else str(r["label_3class"])
        contrib_ids = ",".join(matching_prov["provenance_row_id"].tolist()) if len(matching_prov) else str(r["primary_provenance_row_id"])
        contrib_splits = ",".join(sorted(matching_prov["orig_split"].unique())) if len(matching_prov) else str(r["original_split"])

        new_quar_rows.append(
            {
                "quarantine_id": qid,
                "text": str(r["text"]),
                "text_norm": str(r["text_norm"]),
                "text_norm_strict": str(r["text_norm_strict"]),
                "text_hash_strict": str(r["text_hash_strict"]),
                "quarantine_reason": reason,
                "observed_orig_labels": obs_orig,
                "observed_mapped_labels": obs_map,
                "contributing_datasets": str(r["contributing_datasets"]),
                "contributing_orig_ids": contrib_ids,
                "contributing_splits": contrib_splits,
                "raw_row_count": int(r["contributing_count"]),
                "eligible_for_training": False,
            }
        )

        mask = provenance_v2["canonical_id"] == cid
        provenance_v2.loc[mask, "canonical_id"] = qid
        provenance_v2.loc[mask, "family_id"] = ""
        provenance_v2.loc[mask, "disposition"] = "quarantined"
        provenance_v2.loc[mask, "disposition_reason"] = reason

    quarantined_v2 = (
        pd.concat([quarantined_v1, pd.DataFrame(new_quar_rows)], ignore_index=True)
        .sort_values("quarantine_id")
        .reset_index(drop=True)
    )
    existing_v2 = existing_v2[~existing_v2["canonical_id"].isin(newly_quarantined_cids)].sort_values("canonical_id").reset_index(drop=True)

    # Recompute family_size on surviving existing_v2
    fam_sizes = existing_v2["family_id"].value_counts().to_dict()
    existing_v2["family_size"] = existing_v2["family_id"].map(fam_sizes).astype(int)

    # Export clean eligible_train_v2 with fresh leak-free family-grouped 5-fold CV
    eligible_v2 = existing_v2[existing_v2["eligible_for_training"]].copy().reset_index(drop=True)
    eligible_v2["curation_cv_fold"] = -1
    if len(eligible_v2) >= 5 and eligible_v2["family_id"].nunique() >= 5:
        n_splits = min(5, int(eligible_v2["label_3class"].value_counts().min()), int(eligible_v2["family_id"].nunique()))
        if n_splits >= 2:
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
            for fold_idx, (_, val_idx) in enumerate(
                sgkf.split(eligible_v2, eligible_v2["label_3class"].values, eligible_v2["family_id"].values)
            ):
                eligible_v2.loc[val_idx, "curation_cv_fold"] = fold_idx

    audit_cols = [
        "canonical_id",
        "family_id",
        "primary_orig_id",
        "orig_label_3class",
        "audited_label_3class",
        "orig_language",
        "audited_language",
        "final_status",
        "actions",
        "audit_sources",
        "categories",
        "rationale",
        "text",
    ]
    audit_df = pd.DataFrame(audit_records, columns=audit_cols)

    existing_path = v2_path / "existing_unique.parquet"
    eligible_path = v2_path / "eligible_train.parquet"
    quarantined_path = v2_path / "quarantined.parquet"
    provenance_path = v2_path / "provenance.parquet"
    audit_path = v2_path / "audit_decisions.parquet"
    summary_path = v2_path / "curation_summary.json"

    existing_v2.to_parquet(existing_path, index=False)
    eligible_v2.to_parquet(eligible_path, index=False)
    quarantined_v2.to_parquet(quarantined_path, index=False)
    provenance_v2.to_parquet(provenance_path, index=False)
    audit_df.to_parquet(audit_path, index=False)

    coverage = compute_eligible_coverage_profile(eligible_v2)
    pilot_plan = build_pilot_plan(coverage, max_pilot=300)

    n_reclass_lbl = sum(1 for a in label_actions.values() if a[0] == "reclassify_label")
    n_quar = len(newly_quarantined_cids)
    n_lang_kept = sum(1 for cid in lang_actions if cid not in newly_quarantined_cids)
    n_lang_total = len(lang_actions)

    summary: dict[str, Any] = {
        "base_version": "curation_v1",
        "version": "curation_v2",
        "raw_total_rows": int(len(provenance_v2)),
        "disposition_counts_raw": {k: int(v) for k, v in provenance_v2["disposition"].value_counts().items()},
        "existing_unique_total": int(len(existing_v2)),
        "existing_unique_families": int(existing_v2["family_id"].nunique()),
        "existing_unique_by_class": {lbl: int((existing_v2["label_3class"] == lbl).sum()) for lbl in CANONICAL_LABELS},
        "existing_unique_by_eval_group": {k: int(v) for k, v in existing_v2["eval_group"].value_counts().items()},
        "quarantined_groups_total": int(len(quarantined_v2)),
        "quarantined_groups_by_reason": {k: int(v) for k, v in quarantined_v2["quarantine_reason"].value_counts().items()},
        "audit_counts": {
            "total_audit_decision_rows": int(len(audit_df)),
            "label_reclassifications": int(n_reclass_lbl),
            "newly_quarantined_groups": int(n_quar),
            "language_reclassifications_total": int(n_lang_total),
            "language_reclassifications_retained_in_eligible": int(n_lang_kept),
            "kept_with_note_exemplars": int(sum(1 for r in audit_records if r["actions"] == "keep_with_note")),
        },
        "coverage_profile": coverage,
        "pilot_plan": pilot_plan,
        "artifacts": {
            "existing_unique": str(existing_path),
            "eligible_train": str(eligible_path),
            "quarantined": str(quarantined_path),
            "provenance": str(provenance_path),
            "audit_decisions": str(audit_path),
            "summary": str(summary_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
