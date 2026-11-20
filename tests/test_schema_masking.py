"""Unit and regression tests for schema.py URL/phone regexes, brand masking, and text views."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.schema import (
    DEFAULT_KNOWN_BRANDS,
    detect_brand_mentions,
    extract_metadata_brands,
    extract_surface_features,
    mask_brands,
    mask_shortcuts,
    mask_text_views,
)


class TestSchemaMasking(unittest.TestCase):
    def test_url_regex_catches_bare_domains_without_decimal_or_abbrev_false_positives(self) -> None:
        cases = [
            ("Join via t.me or wa.me/8801700000011 now", True),
            ("Visit altervista.org or sub.portal.example.com today", True),
            ("Check example.edu.bd for tuition dues.", True),
            ("Price is Tk.500 or 10.5 percent, e.g. Dr.Karim said i.e. fine.", False),
            ("তারিখ ১০.৫ এবং মূল্য টাকা.৫০০ মাত্র।", False),
        ]
        for text, should_have_url in cases:
            feats = extract_surface_features(pd.DataFrame({"text": [text]}))
            self.assertEqual(bool(feats["has_url"].iloc[0]), should_have_url, msg=text)
            masked = mask_shortcuts(text)
            if should_have_url:
                self.assertIn("<URL>", masked, msg=text)
            else:
                self.assertNotIn("<URL>", masked, msg=text)

    def test_phone_regex_catches_hyphen_space_bengali_and_shortcodes(self) -> None:
        positive_cases = [
            "Call 01700-000011 for support.",
            "Helpline +880 1700-000011 or +880-1711-223344",
            "যোগাযোগ করুন ০১৭১২৩৪৫৬৭৮ নম্বরে।",
            "হেল্পলাইন ০১৭০০-০০০০১১ অথবা +৮৮০ ১৭১১-২২৩৩৪৪",
            "Dial 16216 or ১৬২১৬ for account status.",
        ]
        for text in positive_cases:
            feats = extract_surface_features(pd.DataFrame({"text": [text]}))
            self.assertTrue(bool(feats["has_phone"].iloc[0]), msg=text)
            masked = mask_shortcuts(text)
            self.assertIn("<PHONE>", masked, msg=text)

        negative_cases = [
            "Pay 16000 taka or ১৬৫০০ টাকা for the package.",
            "Offer price BDT 16000 or Tk 16000 or Tk. 16000 today.",
            "Total fee ৳16000, Rs 16000, or টাকা ১৬০০০ only.",
            "Special BDT ১৬০০০ package or 16000 tk deal.",
            "Pay 16000/- or ১৬০০০/- at the counter.",
            "Your OTP code is 614829 valid for 3 minutes.",
        ]
        for text in negative_cases:
            feats = extract_surface_features(pd.DataFrame({"text": [text]}))
            self.assertFalse(bool(feats["has_phone"].iloc[0]), msg=text)
            self.assertNotIn("<PHONE>", mask_shortcuts(text), msg=text)

        shortcode_cases = [
            "Dial 16216 or 16247 for helpline.",
            "Call 16256 now for balance info.",
            "সহায়তার জন্য ১৬২১৬ অথবা ডায়াল ১৬২৪৭ করুন।",
        ]
        for text in shortcode_cases:
            feats = extract_surface_features(pd.DataFrame({"text": [text]}))
            self.assertTrue(bool(feats["has_phone"].iloc[0]), msg=text)
            self.assertIn("<PHONE>", mask_shortcuts(text), msg=text)

    def test_brand_mask_strip_and_text_views(self) -> None:
        text = "PurbachalBank alert: Visit https://example.com/ok or call 01700-000011, সুরভীমার্টে ছাড়!"
        views = mask_text_views(text)

        self.assertEqual(views["raw"], text)
        self.assertIn("<URL>", views["canon_masked"])
        self.assertIn("<PHONE>", views["canon_masked"])
        self.assertIn("PurbachalBank", views["canon_masked"])

        self.assertIn("<BRAND>", views["brand_masked"])
        self.assertNotIn("PurbachalBank", views["brand_masked"])
        self.assertNotIn("সুরভীমার্ট", views["brand_masked"])

        self.assertNotIn("<BRAND>", views["brand_stripped"])
        self.assertNotIn("PurbachalBank", views["brand_stripped"])
        self.assertNotIn("সুরভীমার্ট", views["brand_stripped"])
        self.assertIn("<URL>", views["brand_stripped"])
        self.assertIn("<PHONE>", views["brand_stripped"])

        with self.assertRaises(ValueError):
            mask_brands(text, mode="invalid")

    def test_fictional_pilot_brand_aliases_masked_and_stripped(self) -> None:
        aliases = [
            "শিকড়পার্সেল",
            "ছায়ানীড়",
            "Boishakhi Fabrics",
            "বৈশাখী ফেব্রিক্স",
            "Anondo Books",
            "আনন্দ বুকস",
            "Srijon Career Hub",
            "সৃজন ক্যারিয়ার হাব",
            "সৃজন ক্যারিয়ার হাব",
            "Borno Training Hub",
        ]
        for brand in aliases:
            text = f"{brand} notice: Special campus update available today."
            views = mask_text_views(text)
            self.assertIn("<BRAND>", views["brand_masked"], msg=brand)
            self.assertNotIn(brand, views["brand_masked"], msg=brand)
            self.assertNotIn("<BRAND>", views["brand_stripped"], msg=brand)
            self.assertNotIn(brand, views["brand_stripped"], msg=brand)

    def test_dynamic_brand_fallback_inflections_and_metadata(self) -> None:
        # Unseen CamelCase & colon-opener brands masked/stripped with inflections, while generic openers stay intact
        txt1 = "PadmaPay's wallet: Visit https://example.com/ok now."
        v1 = mask_text_views(txt1)
        self.assertIn("<BRAND>", v1["brand_masked"])
        self.assertNotIn("PadmaPay", v1["brand_masked"])
        self.assertNotIn("PadmaPay", v1["brand_stripped"])

        txt2 = "সোনালীশপ: আজই অর্ডার করুন।"
        v2 = mask_text_views(txt2)
        self.assertIn("<BRAND>", v2["brand_masked"])
        self.assertNotIn("সোনালীশপ", v2["brand_masked"])

        txt_generic = "Alert: Check your account. জরুরি বিজ্ঞপ্তি: আজ বিদ্যুৎ বন্ধ।"
        v_gen = mask_text_views(txt_generic)
        self.assertNotIn("<BRAND>", v_gen["brand_masked"])
        self.assertIn("Alert:", v_gen["brand_masked"])

        records = [
            {"brand": "RupshaPay", "brand_aliases": ["রূপসাপে"]},
            {"entities": [{"canonical": "SurmaShop", "aliases": ["সুরমাশপ"]}]},
        ]
        meta_brands = extract_metadata_brands(records)
        self.assertEqual(set(meta_brands), {"RupshaPay", "রূপসাপে", "SurmaShop", "সুরমাশপ"})
        inflected = mask_brands("রূপসাপে-তে এবং RupshaPay's অফার", brands=meta_brands, mode="mask")
        self.assertEqual(inflected.count("<BRAND>"), 2)

    def test_bengali_and_banglish_brand_inflections_and_batch1_to_4_merged_defaults(self) -> None:
        # Batch 1-6 brands must be in DEFAULT_KNOWN_BRANDS and masked/stripped/detected identically
        self.assertIn("ShaplaSeba", DEFAULT_KNOWN_BRANDS)
        self.assertIn("শাপলাসেবা", DEFAULT_KNOWN_BRANDS)
        self.assertIn("TitasStyle", DEFAULT_KNOWN_BRANDS)
        self.assertIn("বৈকালীকুরিয়ার", DEFAULT_KNOWN_BRANDS)
        self.assertIn("SurjoTel", DEFAULT_KNOWN_BRANDS)
        self.assertIn("সূর্যটেল", DEFAULT_KNOWN_BRANDS)
        self.assertIn("KarnaphuliBank", DEFAULT_KNOWN_BRANDS)
        self.assertIn("AkashNet", DEFAULT_KNOWN_BRANDS)
        self.assertIn("পিয়ালীমার্ট", DEFAULT_KNOWN_BRANDS)

        cases = [
            ("শাপলাসেবায় ১৫০ টাকা রিচার্জে পাচ্ছেন ৫ জিবি বোনাস ডেটা!", "shaplaseba"),
            ("শাপলাসেবায় (NFD) বিশেষ অফার চলছে।", "shaplaseba"),
            ("শাপলাসেবা-য় নতুন প্যাক চালু হয়েছে।", "shaplaseba"),
            ("শাপলাসেবা ও ShaplaSebay এবং ShaplaSeba-te আজ ছাড়।", "shaplaseba"),
            ("NokshiMart-er আর NokshiMarter অফার এবং নকশিমার্টের ডিসকাউন্ট।", "nokshimart"),
            ("AalokStore-এ আর MoniharFashion-এর কালেকশন।", "aalokstore"),
            ("জুঁইটেল-এ বা করতোয়াটেলে দারুণ ডেটা প্যাক।", "juitel"),
            ("নীড়হাটে তাজা সবজিতে ২০% ছাড়, দেখুন https://example.com/neerhaat/fresh", "neerhaat"),
            ("সূর্যটেলের মাসিক কম্বো প্যাকে ২৫ জিবি ডেটা ও চম্পাস্টাইলে ছাড়।", "surjotel"),
            ("KarnaphuliBank-er credit card ar তরীকুরিয়ারে ডিসকাউন্ট।", "karnaphulibank"),
            ("পিয়ালীমার্টে শাড়ি ও AkashNet-er ডিসকাউন্ট প্যাক।", "piyalimart"),
            ("নিবিড়টেক ও বর্ষা ব্যাংকের কার্ড অফার।", "nibirtech"),
        ]
        for text, expected_canon in cases:
            views = mask_text_views(text)
            detected = detect_brand_mentions(text)
            self.assertIn(expected_canon, detected, msg=text)
            self.assertIn("<BRAND>", views["brand_masked"], msg=text)
            self.assertNotIn("<BRAND>", views["brand_stripped"], msg=text)
            self.assertNotEqual(views["brand_stripped"], views["canon_masked"], msg=text)
            # No dangling Bengali/Banglish inflection fragments like 'য়', '-এ', '-এর', '-er', '-te'
            self.assertNotIn("<BRAND>য়", views["brand_masked"], msg=text)
            self.assertNotIn("<BRAND>-এ", views["brand_masked"], msg=text)
            self.assertNotIn("<BRAND>-এর", views["brand_masked"], msg=text)
            self.assertNotIn("<BRAND>-er", views["brand_masked"], msg=text)
            self.assertNotIn("<BRAND>-te", views["brand_masked"], msg=text)

        # Boundary safety: 'Hon'ble' must NOT match 'BL', and 'রবিবার' must NOT match 'রবি'
        for non_brand_text in (
            "Hon'ble Prime Minister announced a new housing initiative.",
            "শুক্রবার থেকে রবিবার পর্যন্ত বিশেষ উইকেন্ড ছাড় চলছে।",
        ):
            views = mask_text_views(non_brand_text)
            self.assertEqual(detect_brand_mentions(non_brand_text), [], msg=non_brand_text)
            self.assertEqual(views["brand_masked"], views["canon_masked"], msg=non_brand_text)
            self.assertEqual(views["brand_stripped"], views["canon_masked"], msg=non_brand_text)

    def test_bengali_digit_ussd_masking_and_surface_feature(self) -> None:
        bn_ussd = "কম খরচে কথা বলুন, চালু করতে ডায়াল *৩২১*১৫# অথবা *১২১*৪২০৫#।"
        feats = extract_surface_features(pd.DataFrame({"text": [bn_ussd]}))
        self.assertTrue(bool(feats["has_ussd"].iloc[0]))
        masked = mask_shortcuts(bn_ussd)
        self.assertEqual(masked.count("<USSD>"), 2)
        self.assertNotIn("*৩২১*১৫#", masked)


if __name__ == "__main__":
    unittest.main()
