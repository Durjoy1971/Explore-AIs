"""Unit and integration tests for Stage A dataset consolidation pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.curation import (
    curate_from_raw_dataframe,
    sha256_text,
    slot_template_key,
    strong_normalize_text,
)
from bangla_smish.schema import detect_script_bucket, normalize_text


def _make_raw_row(
    dataset: str,
    orig_id: str,
    orig_label: str,
    mapped_label: str,
    text: str,
    orig_split: str = "train",
    source_language_raw: str = "Bengali",
    source_type: str = "SMS",
) -> dict:
    norm_strict = strong_normalize_text(text)
    return {
        "dataset": dataset,
        "source_file": f"{dataset}.csv",
        "source_row_index": 0,
        "orig_id": orig_id,
        "orig_split": orig_split,
        "orig_label": orig_label,
        "mapped_label_3class": mapped_label,
        "source_language_raw": source_language_raw,
        "source_type": source_type,
        "domain_id": "unspecified",
        "domain": "unspecified",
        "text": text,
        "license": "MIT" if dataset != "legacy_2024" else "UNDOCUMENTED",
        "source_synthetic_status": "unknown" if dataset == "legacy_2024" else "augmented_multilingual",
        "provenance_row_id": f"{dataset}:{orig_id}",
        "text_norm": normalize_text(text),
        "text_norm_strict": norm_strict,
        "text_hash_strict": sha256_text(norm_strict),
        "script_bucket": detect_script_bucket(text),
        "is_unusable": (not norm_strict) or ("\ufffd" in text),
    }


class TestStageACuration(unittest.TestCase):
    def test_strong_normalize_text_handles_nfkc_case_and_invisible_chars(self) -> None:
        a = "আমি\u200d দুঃখ পে\u09af\u09bcেছি\u200c\ufeff   HELLO\u00a0World"
        b = "আমি দুঃখ পেয়েছি hello world"
        self.assertEqual(strong_normalize_text(a), strong_normalize_text(b))

    def test_quarantine_conflicts_legacy_and_unusable(self) -> None:
        rows = [
            # 1. Empty / replacement char unusable
            _make_raw_row("smishdetect_2026", "sd_0", "normal", "Normal", "   \u200d  "),
            _make_raw_row("smishdetect_2026", "sd_1", "normal", "Normal", "Bad \ufffd char"),
            # 2. 3-class conflict
            _make_raw_row("bangalabarta_2025", "bb_0", "promo", "Promotional", "সোনালী ব্যাংক শুভেচ্ছা"),
            _make_raw_row("bangalabarta_2025", "bb_1", "smish", "Smishing", "সোনালী ব্যাংক শুভেচ্ছা"),
            # 3. Cross-source legacy spam vs Normal conflict
            _make_raw_row("legacy_2024", "leg_0", "spam", "Spam_Binary", "আপনার ছেলে বিপদে টাকা পাঠান", orig_split="unspecified"),
            _make_raw_row("smishdetect_2026", "sd_2", "normal", "Normal", "আপনার ছেলে বিপদে টাকা পাঠান"),
            # 4. Legacy-only binary spam (must NOT map to Promotional/Smishing)
            _make_raw_row("legacy_2024", "leg_1", "spam", "Spam_Binary", "Win a prize now click spam", orig_split="unspecified"),
            # 5. Legacy-only binary ham (reference only)
            _make_raw_row("legacy_2024", "leg_2", "ham", "Normal", "আজকে বাসায় যাবো সন্ধ্যায়", orig_split="unspecified"),
            # 6. Clean row present in both legacy spam and smishdetect Smishing (accepted via smishdetect)
            _make_raw_row("legacy_2024", "leg_3", "spam", "Spam_Binary", "বিকাশ পিন দিন এখনই না হলে বন্ধ", orig_split="unspecified"),
            _make_raw_row("smishdetect_2026", "sd_3", "smish", "Smishing", "বিকাশ পিন দিন এখনই না হলে বন্ধ"),
        ]
        raw_df = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmp:
            summary = curate_from_raw_dataframe(raw_df, Path(tmp))
            existing = pd.read_parquet(Path(tmp) / "existing_unique.parquet")
            quar = pd.read_parquet(Path(tmp) / "quarantined.parquet")
            prov = pd.read_parquet(Path(tmp) / "provenance.parquet")

        self.assertEqual(len(existing), 1)
        self.assertEqual(existing.iloc[0]["label_3class"], "Smishing")
        self.assertIn("legacy_2024", existing.iloc[0]["contributing_datasets"])
        self.assertEqual(len(quar), 6)
        reasons = set(quar["quarantine_reason"])
        self.assertIn("empty_or_corrupted_text", reasons)
        self.assertIn("conflicting_3class_labels", reasons)
        self.assertIn("cross_source_conflict_legacy_spam_vs_normal", reasons)
        self.assertIn("legacy_binary_spam_reference_only", reasons)
        self.assertIn("legacy_binary_ham_undocumented_license_reference_only", reasons)
        self.assertEqual(len(prov), len(raw_df))
        self.assertEqual(summary["eligible_train_total"], 1)

    def test_near_dup_review_collapse_and_eval_family_exclusion(self) -> None:
        rows = [
            # Cosmetic variant (punctuation/markdown) -> collapse 1, keep 1
            _make_raw_row("smishdetect_2026", "sd_10", "promo", "Promotional", "Offer 50% off! Visit https://example.com/a", orig_split="train", source_language_raw="English"),
            _make_raw_row("smishdetect_2026", "sd_11", "promo", "Promotional", "Offer 50% off Visit [https://example.com/a](https://example.com/a)", orig_split="train", source_language_raw="English"),
            # Slot-swap clone (different phone/code) + 2-hop cosmetic variant of the collapsed slot-swap
            _make_raw_row("smishdetect_2026", "sd_12", "normal", "Normal", "01711111111 ei number e call korun", orig_split="train", source_language_raw="Banglish"),
            _make_raw_row("smishdetect_2026", "sd_12b", "normal", "Normal", "01711111111 ei number e call korun!", orig_split="train", source_language_raw="Banglish"),
            _make_raw_row("smishdetect_2026", "sd_13", "normal", "Normal", "01822222222 ei number e call korun", orig_split="test", source_language_raw="Banglish"),
            # Materially distinct near-dup sharing family with validation -> kept in existing_unique, excluded from eligible_train
            _make_raw_row("smishdetect_2026", "sd_14", "normal", "Normal", "বন্ধু, Sunday picnic plan ঠিক করেছো কি? আমরা early morning বের হবো, Let’s discuss route এবং food arrangement", orig_split="validation", source_language_raw="CodeMix"),
            _make_raw_row("smishdetect_2026", "sd_15", "normal", "Normal", "বন্ধু, Sunday picnic এর জন্য plan ঠিক করেছো কি? আমরা early morning বের হব, Let’s discuss route এবং food arrangement", orig_split="train", source_language_raw="CodeMix"),
            # Clean independent training rows across classes
            _make_raw_row("smishdetect_2026", "sd_16", "normal", "Normal", "আজকে আবহাওয়া অনেক সুন্দর এবং ঠান্ডা বাতাস বইছে।", orig_split="train"),
            _make_raw_row("smishdetect_2026", "sd_17", "smish", "Smishing", "আপনার ব্যাংক একাউন্ট স্থগিত হয়েছে যাচাই করতে লিংকে যান", orig_split="train"),
            # BanglaPhish OOD benchmark row (must be excluded from eligible_train)
            _make_raw_row("banglaphish_2026", "BP_1", "scam", "Smishing", "আপনার নগদ একাউন্ট যাচাই করতে [REDACTED_URL] দেখুন", orig_split="test", source_language_raw="Bangla"),
        ]
        raw_df = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            curate_from_raw_dataframe(raw_df, tmp_path)
            existing = pd.read_parquet(tmp_path / "existing_unique.parquet")
            elig = pd.read_parquet(tmp_path / "eligible_train.parquet")
            prov = pd.read_parquet(tmp_path / "provenance.parquet")
            with open(tmp_path / "near_duplicate_review.jsonl", "r", encoding="utf-8") as f:
                reviews = [json.loads(line) for line in f]

        # 10 raw -> 3 collapsed (sd_11 cosmetic, sd_12 slot-swap, sd_12b cosmetic->slot-swap 2-hop) -> 7 in existing_unique
        self.assertEqual(len(existing), 7)
        self.assertEqual(len(prov), 10)
        collapsed_in_prov = int((prov["disposition"] == "near_dup_variant_collapsed").sum())
        self.assertEqual(collapsed_in_prov, 3)
        # Regression check 1: every collapsed variant in provenance must appear in near_duplicate_review.jsonl
        self.assertEqual(sum(r["collapsed_count"] for r in reviews), collapsed_in_prov)
        # Regression check 2: post-collapse clusters on surviving texts never cross family_id boundaries
        from bangla_smish.audit import compute_near_dup_clusters
        from bangla_smish.curation import slot_template_text

        c_post_raw, _ = compute_near_dup_clusters(existing["text_norm_strict"].tolist(), threshold=0.85)
        c_post_slot, _ = compute_near_dup_clusters(existing["text"].map(slot_template_text).tolist(), threshold=0.85)
        self.assertEqual(pd.Series(existing["family_id"].values).groupby(c_post_raw).nunique().max(), 1)
        self.assertEqual(pd.Series(existing["family_id"].values).groupby(c_post_slot).nunique().max(), 1)
        # sd_15 shares family with sd_14 (validation), so excluded from eligible_train!
        self.assertNotIn("sd_15", set(elig["primary_orig_id"]))
        self.assertNotIn("BP_1", set(elig["primary_orig_id"]))
        self.assertNotIn("sd_13", set(elig["primary_orig_id"]))
        self.assertEqual(set(elig["primary_orig_id"]), {"sd_10", "sd_16", "sd_17"})
        # Review log recorded both collapsed variants and kept distinct family
        self.assertTrue(any(r["collapsed_count"] >= 1 for r in reviews))
        self.assertEqual(slot_template_key("01711111111 call"), slot_template_key("01822222222 call"))


if __name__ == "__main__":
    unittest.main()
