"""Unit and regression tests for CodeMix linguistic, structural, and entity diversity."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.codemix_diversity import (
    analyze_codemix_row,
    collect_brand_aliases,
    evaluate_codemix_and_entity_diversity,
)


class TestAnalyzeCodeMixRow(unittest.TestCase):
    def test_hook_excl_pure_en_bn_and_banglish(self) -> None:
        bn = analyze_codemix_row(
            "বইপ্রেমীদের জন্য চমৎকার খবর! Order any two novels from ChitroBooks and get one free.",
            language="CodeMix",
        )
        self.assertEqual(bn["script_subtype"], "bn_script_en")
        self.assertEqual(bn["discourse_pattern"], "hook_excl_pure_en")
        self.assertEqual(bn["sent1_end_punct"], "!")
        self.assertTrue(bn["has_excl"])
        self.assertFalse(bn["has_quest"])
        self.assertTrue(bn["is_hook_plus_pure_en"])
        self.assertEqual(bn["sentence_count"], 2)

        bl = analyze_codemix_row(
            "Notun collection chole esheche! Shop winter hoodies at 20% off today.",
            language="CodeMix",
        )
        self.assertEqual(bl["script_subtype"], "banglish_latin_en")
        self.assertEqual(bl["discourse_pattern"], "hook_excl_pure_en")
        self.assertTrue(bl["is_hook_plus_pure_en"])

    def test_hook_quest_pure_en(self) -> None:
        res = analyze_codemix_row(
            "Notun bashar furniture khujchen? HaatGhor offers 20% seasonal savings on orders above 15,000 TK.",
            language="CodeMix",
        )
        self.assertEqual(res["script_subtype"], "banglish_latin_en")
        self.assertEqual(res["discourse_pattern"], "hook_quest_pure_en")
        self.assertEqual(res["sent1_end_punct"], "?")
        self.assertTrue(res["has_quest"])
        self.assertTrue(res["is_hook_plus_pure_en"])

    def test_intra_sentential_mix_single_and_multi(self) -> None:
        single = analyze_codemix_row(
            "আজ Dhanmondi branch-এর শুভ opening-এ flat ৫০% discount।",
            language="CodeMix",
        )
        self.assertEqual(single["script_subtype"], "bn_script_en")
        self.assertEqual(single["discourse_pattern"], "intra_sentential_mix")
        self.assertEqual(single["sent1_end_punct"], "।")
        self.assertFalse(single["is_hook_plus_pure_en"])

        multi = analyze_codemix_row(
            "Online shopping-এ ২৫% cashback! Code: SHOP25. Delay না করে এখনই shopping করুন।",
            language="CodeMix",
        )
        self.assertEqual(multi["script_subtype"], "bn_script_en")
        self.assertEqual(multi["discourse_pattern"], "intra_sentential_mix")
        self.assertEqual(multi["sentence_count"], 3)
        self.assertFalse(multi["is_hook_plus_pure_en"])

    def test_en_lead_bn_body_colon_and_multi(self) -> None:
        colon = analyze_codemix_row(
            "Weekend Flash Offer: সব grocery item-এ 15% cashback, code FRESH15 ব্যবহার করুন।",
            language="CodeMix",
        )
        self.assertEqual(colon["script_subtype"], "bn_script_en")
        self.assertEqual(colon["discourse_pattern"], "en_lead_bn_body")
        self.assertFalse(colon["is_hook_plus_pure_en"])

        multi = analyze_codemix_row(
            "Mega Weekend Fest is live. আজকেই আপনার পছন্দের electronics-এ 15% cashback পান।",
            language="CodeMix",
        )
        self.assertEqual(multi["discourse_pattern"], "en_lead_bn_body")
        self.assertFalse(multi["is_hook_plus_pure_en"])

    def test_stmt_colon_lead_en(self) -> None:
        multi = analyze_codemix_row(
            "আপনার অর্ডারটি আজ ডেলিভারি হবে। Track your rider at https://example.com/track.",
            language="CodeMix",
        )
        self.assertEqual(multi["script_subtype"], "bn_script_en")
        self.assertEqual(multi["discourse_pattern"], "stmt_colon_lead_en")
        self.assertEqual(multi["sent1_end_punct"], "।")
        self.assertFalse(multi["is_hook_plus_pure_en"])

        colon = analyze_codemix_row(
            "TarangaNet er shera deal: Get 150 mins plus 4GB for 119 TK!",
            language="CodeMix",
        )
        self.assertEqual(colon["script_subtype"], "banglish_latin_en")
        self.assertEqual(colon["discourse_pattern"], "stmt_colon_lead_en")

    def test_multi_sent_alternating_and_decimals(self) -> None:
        alt = analyze_codemix_row(
            "সকল পণ্যে 20% discount। আজই order করুন। Limited time offer on 9.25% plan.",
            language="CodeMix",
        )
        self.assertEqual(alt["script_subtype"], "bn_script_en")
        self.assertEqual(alt["discourse_pattern"], "multi_sent_alternating")
        self.assertEqual(alt["sentence_count"], 3)
        self.assertFalse(alt["is_hook_plus_pure_en"])

    def test_other_monolingual_rows(self) -> None:
        bn = analyze_codemix_row("শুধু বাংলা বার্তা এখানে। কোনো ইংরেজি নেই।")
        self.assertEqual(bn["script_subtype"], "other")
        self.assertEqual(bn["discourse_pattern"], "other")

        en = analyze_codemix_row("Enjoy 20% off all items this weekend only.")
        self.assertEqual(en["script_subtype"], "other")
        self.assertEqual(en["discourse_pattern"], "other")


class TestEvaluateCodeMixAndEntityDiversity(unittest.TestCase):
    def test_balanced_diverse_new_batch_passes_all_gates(self) -> None:
        # 10 Promotional rows (5 CodeMix: 4 bn_script_en + 1 banglish_latin_en, 0 hooks, >=60% intra_sentential_mix,
        # >=30% brandless, <=35% sent1!, <=5% sent1?, max_brand_count<=5, ussd_1234_share<=0.25)
        records = [
            {
                "label_3class": "Promotional",
                "language": "CodeMix",
                "text": "সব grocery item-এ 15% cashback, code FRESH15 ব্যবহার করুন।",
            },
            {
                "label_3class": "Promotional",
                "language": "CodeMix",
                "text": "এই সপ্তাহে mobile recharge-এ 10% instant bonus পাচ্ছেন। ডায়াল *121*5#।",
            },
            {
                "label_3class": "Promotional",
                "language": "CodeMix",
                "text": "শীতের jacket ও hoodie collection-এ 25% flat discount চলছে শুক্রবার পর্যন্ত।",
            },
            {
                "label_3class": "Promotional",
                "language": "CodeMix",
                "text": "Weekend Flash Offer: বই ও stationery অর্ডারে free delivery পাচ্ছেন।",
            },
            {
                "label_3class": "Promotional",
                "language": "CodeMix",
                "text": "GP Mega Recharge: 399TK recharge e paben 35GB internet. Activate *121*399# now.",
            },
            {
                "label_3class": "Promotional",
                "language": "Bengali",
                "text": "তাজা ফল ও সবজিতে আজ ২০% ছাড়। অর্ডার করুন *5000*1# ডায়াল করে।",
            },
            {
                "label_3class": "Promotional",
                "language": "Bengali",
                "text": "নদীপে অ্যাপে বিদ্যুৎ বিল পরিশোধে ৫০ টাকা ক্যাশব্যাক!",
            },
            {
                "label_3class": "Promotional",
                "language": "Banglish",
                "text": "SuroviMart e 1,500 takar grocery kinle 200 taka char paben.",
            },
            {
                "label_3class": "Promotional",
                "language": "Banglish",
                "text": "Notun internet pack 12GB matro 149 takay, meyad 7 din, dial *167*2#.",
            },
            {
                "label_3class": "Promotional",
                "language": "English",
                "text": "Enjoy 15% off cotton kurtis at Boishakhi Fabrics until Sunday!",
            },
        ]
        res = evaluate_codemix_and_entity_diversity(records, mode="new_batch")
        self.assertTrue(res["all_gates_passed"], msg=str(res["gate_checks"]))
        self.assertEqual(res["bn_script_en_rate"], 0.8)
        self.assertEqual(res["hook_plus_pure_en_rate"], 0.0)
        self.assertEqual(res["intra_sentential_mix_rate"], 0.6)
        self.assertEqual(res["sent1_excl_rate"], 0.2)
        self.assertEqual(res["sent1_quest_rate"], 0.0)
        self.assertGreaterEqual(res["brandless_rate"], 0.3)
        self.assertLessEqual(res["max_brand_count"], 5)
        self.assertEqual(res["ussd_1234_share"], 0.0)

    def test_rejects_hook_template_question_excl_brand_and_ussd_concentration(self) -> None:
        records = [
            {
                "label_3class": "Promotional",
                "language": "CodeMix",
                "text": "Notun offer chai? Shop SuroviMart today for 20% off! Dial *1234#.",
            },
            {
                "label_3class": "Promotional",
                "language": "CodeMix",
                "text": "দারুণ সুযোগ! Buy SuroviMart items at 10% off now. Dial *1234#.",
            },
            {
                "label_3class": "Promotional",
                "language": "CodeMix",
                "text": "SuroviMart mega deal! Save 300 TK at SuroviMart! Dial *1234*1#.",
            },
            {
                "label_3class": "Promotional",
                "language": "Banglish",
                "text": "SuroviMart e offer! SuroviMart theke kinun.",
            },
            {
                "label_3class": "Promotional",
                "language": "English",
                "text": "SuroviMart sale is live! Check SuroviMart today.",
            },
            {
                "label_3class": "Promotional",
                "language": "Bengali",
                "text": "SuroviMart এ ছাড়! এখনই অর্ডার করুন।",
            },
        ]
        res = evaluate_codemix_and_entity_diversity(
            records,
            known_brands=["SuroviMart"],
            mode="new_batch",
        )
        self.assertFalse(res["all_gates_passed"])
        self.assertFalse(res["gate_checks"]["codemix_bn_script_en"]["passed"])
        self.assertFalse(res["gate_checks"]["codemix_hook_plus_pure_en"]["passed"])
        self.assertFalse(res["gate_checks"]["codemix_intra_sentential_mix"]["passed"])
        self.assertFalse(res["gate_checks"]["promo_sent1_excl_rate"]["passed"])
        self.assertFalse(res["gate_checks"]["promo_sent1_quest_rate"]["passed"])
        self.assertFalse(res["gate_checks"]["max_brand_count"]["passed"])
        self.assertFalse(res["gate_checks"]["brandless_rate"]["passed"])
        self.assertFalse(res["gate_checks"]["ussd_1234_share"]["passed"])

    def test_invalid_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_codemix_and_entity_diversity([], mode="invalid")

    def test_pilot_accepted_regression_flags_known_biases(self) -> None:
        accepted_path = ROOT / "data/generated/curation_v2/accepted.jsonl"
        if not accepted_path.exists():
            self.skipTest("Pilot accepted.jsonl not present")
        with open(accepted_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        res = evaluate_codemix_and_entity_diversity(records, mode="audit")
        self.assertEqual(res["codemix_rows"], 74)
        self.assertEqual(res["promotional_codemix_rows"], 60)
        self.assertGreaterEqual(res["hook_plus_pure_en_rate"], 0.80)
        self.assertLess(res["bn_script_en_rate"], 0.80)
        self.assertEqual(res["ussd_1234_share"], 1.0)
        self.assertFalse(res["all_gates_passed"])

    def test_unified_brand_detection_bengali_banglish_inflections_and_bn_ussd(self) -> None:
        from bangla_smish.schema import mask_text_views

        records = [
            {
                "label_3class": "Promotional",
                "language": "Bengali",
                "text": "শাপলাসেবায় ১৫০ টাকা রিচার্জে ৫ জিবি বোনাস! ডায়াল *১২১*৪২০৫#।",
            },
            {
                "label_3class": "Promotional",
                "language": "Banglish",
                "text": "ShaplaSebay ajke 200 taka recharge-e bonus, ar ShaplaSeba-r pack-e char.",
            },
            {
                "label_3class": "Promotional",
                "language": "Bengali",
                "text": "শুক্রবার থেকে রবিবার পর্যন্ত সব আইটেমে ২০% ছাড়, ডায়াল *১২৩৪#।",
            },
            {
                "label_3class": "Promotional",
                "language": "Bengali",
                "text": "তাজা আম ও লিচুতে ১৫% ছাড়, দেখতে ভিজিট করুন https://example.com/shaplaseba",
            },
        ]
        res = evaluate_codemix_and_entity_diversity(records, mode="audit")
        top = dict(res["top_brands"])
        # Rows 1 & 2 -> shaplaseba; Row 3 (রবিবার) & Row 4 (URL-only path) -> brandless
        self.assertEqual(top.get("shaplaseba"), 2)
        self.assertEqual(res["unique_brands_count"], 1)
        self.assertEqual(res["brandless_rate"], 0.5)
        self.assertEqual(res["total_ussd_mentions"], 2)
        self.assertEqual(res["ussd_1234_count"], 1)
        self.assertEqual(res["ussd_1234_share"], 0.5)
        for rec in records:
            views = mask_text_views(rec["text"])
            is_branded_mask = "<BRAND>" in views["brand_masked"]
            is_branded_strip = views["brand_stripped"] != views["canon_masked"]
            self.assertEqual(is_branded_mask, is_branded_strip, msg=rec["text"])

    def test_dynamic_metadata_brand_aliases_and_known_brands_preserve_defaults(self) -> None:
        records = [
            {
                "label_3class": "Promotional",
                "language": "Banglish",
                "text": "Tista Grocery te shob organic chal-dal e 15% char!",
                "brand": "TistaGrocery",
                "brand_aliases": ["Tista Grocery", "তিস্তা গ্রোসারি"],
            },
            {
                "label_3class": "Promotional",
                "language": "Bengali",
                "text": "তিস্তা গ্রোসারিতে আজ ২০% ছাড় চলছে।",
            },
            {
                "label_3class": "Promotional",
                "language": "Bengali",
                "text": "বর্ণটেলিকম প্যাক ৯৯ টাকায় ৫ জিবি।",
            },
            {
                "label_3class": "Promotional",
                "language": "English",
                "text": "Weekend fresh fruit savings available across Dhaka outlets.",
            },
        ]
        # Passing known_brands must merge without wiping baseline _CANONICAL_BRAND_ALIASES
        res = evaluate_codemix_and_entity_diversity(
            records,
            known_brands=["CustomOnlyBrand"],
            mode="audit",
        )
        top_brands = dict(res["top_brands"])
        self.assertEqual(top_brands.get("tistagrocery"), 2)
        self.assertEqual(top_brands.get("bornotelecom"), 1)
        self.assertEqual(res["max_brand_count"], 2)
        self.assertEqual(res["brandless_rate"], 0.25)

    def test_brand_registry_dict_and_json_file_and_collect_helper(self) -> None:
        records = [
            {
                "label_3class": "Promotional",
                "language": "CodeMix",
                "text": "সব item-এ 15% cashback দিচ্ছে আরণ্যক বুকস!",
                "entities": [{"canonical": "AranyakBooks", "aliases": ["আরণ্যক বুকস", "Aranyak Books"]}],
            },
            {
                "label_3class": "Promotional",
                "language": "Banglish",
                "text": "Aranyak Books e shob notun boi 20% kome paben.",
            },
            {
                "label_3class": "Promotional",
                "language": "Bengali",
                "text": "মেঘনা রিসোর্টে সাপ্তাহিক ছুটিতে ২৫% ছাড়!",
            },
            {
                "label_3class": "Promotional",
                "language": "Bengali",
                "text": "Meghna Resort e family package e special char.",
            },
            {
                "label_3class": "Promotional",
                "language": "Bengali",
                "text": "নদীপে অ্যাপে পেমেন্টে ৫০ টাকা ক্যাশব্যাক।",
            },
            {
                "label_3class": "Promotional",
                "language": "English",
                "text": "Flat 10% off winter sweaters in store today.",
            },
        ]
        registry = {"meghnaresort": ["Meghna Resort", "মেঘনা রিসোর্ট"]}
        collected = collect_brand_aliases(
            records,
            brand_registry=registry,
            extra_brands=["ExtraShop"],
        )
        self.assertIn("nodipay", collected)
        self.assertIn("নদীপে", collected["nodipay"])
        self.assertIn("aranyakbooks", collected)
        self.assertIn("আরণ্যক বুকস", collected["aranyakbooks"])
        self.assertIn("meghnaresort", collected)
        self.assertIn("মেঘনা রিসোর্ট", collected["meghnaresort"])
        self.assertIn("extrashop", collected)

        with tempfile.TemporaryDirectory() as tmp:
            reg_path = Path(tmp) / "brand_registry.json"
            reg_path.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")
            res = evaluate_codemix_and_entity_diversity(records, brand_registry=reg_path, mode="audit")
            top_brands = dict(res["top_brands"])
            self.assertEqual(top_brands.get("aranyakbooks"), 2)
            self.assertEqual(top_brands.get("meghnaresort"), 2)
            self.assertEqual(top_brands.get("nodipay"), 1)
            self.assertEqual(res["max_brand_count"], 2)
            self.assertAlmostEqual(res["brandless_rate"], round(1 / 6, 4))


if __name__ == "__main__":
    unittest.main()
