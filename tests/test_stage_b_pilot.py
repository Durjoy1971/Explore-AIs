"""Unit tests for Stage B pilot screening, anti-leakage, domain/brand, and boundary checks."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.stage_b_pilot import (
    TEXT_VIEW_NAMES,
    compute_extended_stage_b_diagnostics,
    compute_length_band,
    load_jsonl,
    screen_stage_b_pilot,
    write_jsonl,
)


def _cand(
    cid: str,
    text: str,
    label_3class: str = "Normal",
    language: str = "English",
    family_id: str | None = None,
    length_band: str | None = None,
    char_len: int | None = None,
) -> dict:
    rec = {
        "id": cid,
        "text": text,
        "label_3class": label_3class,
        "language": language,
        "scenario": "test_scenario",
        "domain": "banking_mfs",
        "length_band": length_band if length_band is not None else compute_length_band(len(text)),
        "formality": "formal",
        "has_link": "example." in text,
        "has_urgency": False,
        "is_synthetic": True,
        "source_dataset": "stage_b_pilot_v2",
        "source_row_ids": [],
        "family_id": family_id or f"fam_{cid}",
        "generator_model": "test",
        "prompt_version": "v1",
        "generation_batch": "test_batch",
        "created_at": "2026-11-17T00:00:00Z",
        "review_status": "pending_validation",
        "label_rationale": "unit test",
    }
    if char_len is not None:
        rec["char_len"] = char_len
    return rec


class TestStageBPilotScreening(unittest.TestCase):
    def test_rejects_brands_domains_leaks_clones_and_language_errors(self) -> None:
        v2_dir = ROOT / "data/processed/curation_v2"
        import pandas as pd

        ex = pd.read_parquet(v2_dir / "existing_unique.parquet")
        held_text = str(ex.loc[ex["eval_group"] == "smishdetect_test", "text"].iloc[0])

        cands = [
            # 1. Valid Normal English with reserved domain
            _cand("ok_1", "PurbachalBank alert: Your monthly e-statement is ready at https://portal.example.com/stmt. Never share your OTP.", "Normal", "English"),
            # 2. Valid Promotional Banglish
            _cand("ok_2", "SuroviMart e shob grocery item e 15% char cholche ajkei order korun https://shop.example.org/sale", "Promotional", "Banglish"),
            # 3. Real brand bKash -> reject
            _cand("bad_brand", "Send money via bKash easily at https://pay.example.com", "Normal", "English"),
            # 4. Non-reserved domain -> reject
            _cand("bad_dom", "Check your NodiPay parcel update at https://nodipay-bd.com/track", "Normal", "English"),
            # 5. Real phone prefix -> reject
            _cand("bad_phone", "Call ShikorParcel support at 01712345678 regarding your delivery.", "Normal", "English"),
            # 6. Held-out exact leak (must reject without leaking held-out text in reasons)
            _cand("bad_leak", held_text, "Normal", "English"),
            # 7. Intra-pilot template clone of ok_1 -> reject
            _cand("bad_clone", "PurbachalBank alert: Your monthly e-statement is ready at https://portal.example.com/stmt. Never share your OTP!", "Normal", "English"),
            # 8. Pure English mislabeled Banglish -> reject
            _cand("bad_lang", "Enjoy a flat 25% discount on all winter items at AalokStore today!", "Promotional", "Banglish"),
            # 9. Normal with commercial offer -> reject
            _cand("bad_norm_promo", "SuroviMart mega sale 30% discount on electronics today!", "Normal", "English"),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            batch_path = tmp_path / "batch.jsonl"
            write_jsonl(batch_path, cands)
            summary = screen_stage_b_pilot(
                [batch_path],
                v2_dir=v2_dir,
                out_dir=tmp_path / "out",
                include_extended_diagnostics=False,
            )

            accepted = load_jsonl(tmp_path / "out/accepted.jsonl")
            rejected = load_jsonl(tmp_path / "out/rejected.jsonl")

            self.assertEqual([r["id"] for r in accepted], ["ok_1", "ok_2"])
            self.assertEqual(len(rejected), 7)
            rej_by_id = {r["id"]: "|".join(r["rejection_reasons"]) for r in rejected}
            self.assertIn("real_bd_brand_mention", rej_by_id["bad_brand"])
            self.assertIn("non_reserved_domain", rej_by_id["bad_dom"])
            self.assertIn("non_fictional_phone_number", rej_by_id["bad_phone"])
            self.assertIn("heldout_or_ood", rej_by_id["bad_leak"])
            self.assertNotIn(held_text[:20], rej_by_id["bad_leak"])
            self.assertIn("intra_pilot", rej_by_id["bad_clone"])
            self.assertIn("banglish_appears_pure_english", rej_by_id["bad_lang"])
            self.assertIn("normal_contains_commercial_offer_cue", rej_by_id["bad_norm_promo"])
            self.assertEqual(summary["accepted_count"], 2)

    def test_phone_regex_length_band_and_review_integration(self) -> None:
        v2_dir = ROOT / "data/processed/curation_v2"
        cands = [
            # Valid fictional 5+6 Bengali-digit phone and shorthand length_band ("medium")
            _cand(
                "ok_bn_fict",
                "নদীপে সেবা: সহায়তার জন্য কল করুন ০১৭০০-০০০০১৮ নম্বরে।",
                "Normal",
                "Bengali",
                family_id="fam_shared_par",
                length_band="medium",
            ),
            # Valid parallel English sharing family_id="fam_shared_par" + price 16000 taka (not shortcode)
            _cand(
                "ok_en_par",
                "NodiPay receipt: Your 16000 taka transfer succeeded. Call 01700-000018 for support.",
                "Normal",
                "English",
                family_id="fam_shared_par",
            ),
            # Valid row with no review entry -> pending_independent_review
            _cand(
                "ok_unreviewed",
                "BornoTelecom alert: Your monthly postpaid statement is ready at https://portal.example.com/bill",
                "Normal",
                "English",
            ),
            # 5+6 hyphenated non-fictional phone -> reject
            _cand("bad_5_6_dash", "Call customer support at 01712-345678 for parcel details.", "Normal", "English"),
            # Bengali-digit non-fictional phone -> reject
            _cand("bad_bn_phone", "নদীপে সহায়তা পেতে কল করুন ০১৭১২-৩৪৫৬৭৮ নম্বরে।", "Normal", "Bengali"),
            # Bengali-digit +880 phone -> reject
            _cand("bad_bn_880", "যোগাযোগ করুন +৮৮০ ১৭১২৩৪৫৬৭৮ নম্বরে অফিস সময়ে।", "Normal", "Bengali"),
            # 5-digit shortcode 16xxx -> reject
            _cand("bad_shortcode_en", "Dial 16216 immediately to confirm your bank statement.", "Normal", "English"),
            _cand("bad_shortcode_bn", "বিস্তারিত জানতে ১৬২১৬ নম্বরে ডায়াল করুন এখনই।", "Normal", "Bengali"),
            # Mismatched length_band -> reject
            _cand(
                "bad_len_band",
                "Short notice from Chayaneer University desk.",
                "Normal",
                "English",
                length_band="long(>120)",
            ),
            # Mismatched char_len -> reject
            _cand(
                "bad_char_len",
                "Chayaneer University notice: Fall semester registration opens Monday morning.",
                "Normal",
                "English",
                char_len=999,
            ),
            # Automated-pass row rejected by independent review -> rejected.jsonl
            _cand(
                "rev_reject_row",
                "AalokStore e ajke shob winter jacket e 20% chhar cholche!",
                "Promotional",
                "Banglish",
            ),
        ]

        reviews = [
            {"id": "ok_bn_fict", "decision": "accept", "reasons": [], "reviewer": "reviewer_a"},
            {"id": "ok_en_par", "decision": "accept", "reasons": [], "reviewer": "reviewer_a"},
            {
                "id": "rev_reject_row",
                "decision": "reject",
                "reasons": ["unnatural_phrasing"],
                "reviewer": "reviewer_b",
                "rationale": "Unit test independent rejection",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            batch_path = tmp_path / "batch.jsonl"
            rev_path = tmp_path / "reviews.jsonl"
            write_jsonl(batch_path, cands)
            write_jsonl(rev_path, reviews)

            summary = screen_stage_b_pilot(
                [batch_path],
                v2_dir=v2_dir,
                out_dir=tmp_path / "out",
                review_paths=[rev_path],
                include_extended_diagnostics=False,
            )
            accepted = load_jsonl(tmp_path / "out/accepted.jsonl")
            rejected = load_jsonl(tmp_path / "out/rejected.jsonl")
            cands_out = load_jsonl(tmp_path / "out/candidates.jsonl")

            self.assertEqual([r["id"] for r in accepted], ["ok_bn_fict", "ok_en_par", "ok_unreviewed"])
            acc_by_id = {r["id"]: r for r in accepted}
            self.assertEqual(acc_by_id["ok_bn_fict"]["length_band"], "medium(50-120)")
            self.assertEqual(acc_by_id["ok_bn_fict"]["review_status"], "accepted")
            self.assertEqual(acc_by_id["ok_en_par"]["review_status"], "accepted")
            self.assertEqual(acc_by_id["ok_unreviewed"]["review_status"], "pending_independent_review")
            self.assertEqual(cands_out[0]["length_band"], "medium(50-120)")

            rej_by_id = {r["id"]: "|".join(r["rejection_reasons"]) for r in rejected}
            self.assertEqual(len(rejected), 8)
            for pid in ("bad_5_6_dash", "bad_bn_phone", "bad_bn_880", "bad_shortcode_en", "bad_shortcode_bn"):
                self.assertIn("non_fictional_phone_number", rej_by_id[pid])
            self.assertIn("length_band_mismatch", rej_by_id["bad_len_band"])
            self.assertIn("char_len_mismatch", rej_by_id["bad_char_len"])
            self.assertIn("independent_review:unnatural_phrasing", rej_by_id["rev_reject_row"])
            self.assertEqual(summary["independently_reviewed_accepted_count"], 2)
            self.assertEqual(summary["pending_independent_review_count"], 1)
            self.assertEqual(summary["rejected_count"], 8)

    def test_extended_diagnostics_codemix_diversity_and_dry_run(self) -> None:
        import hashlib
        import importlib.util
        import pandas as pd

        # 1. Extended diagnostics unit check across all 4 text views + surface_8feat + class/stratum-matched probes + counterfactuals
        elig_rows = []
        for i in range(12):
            elig_rows.append(
                {
                    "text": f"PurbachalBank monthly alert {i}: Your balance is 2,500 taka at branch {i} today.",
                    "label_3class": "Normal",
                    "language": "English",
                    "family_id": f"elig_n_{i}",
                }
            )
            elig_rows.append(
                {
                    "text": f"GrameenMart offer {i}: Save 15% on grocery packs valid until Friday evening.",
                    "label_3class": "Promotional",
                    "language": "English",
                    "family_id": f"elig_p_{i}",
                }
            )
            elig_rows.append(
                {
                    "text": f"বিকাশ একাউন্ট {i} বন্ধ হবে! যাচাই করতে ভিজিট করুন https://bad-verify.com/{i} অথবা কল 0171234567{i % 10}",
                    "label_3class": "Smishing",
                    "language": "Bengali",
                    "family_id": f"elig_s_{i}",
                }
            )
        elig_df = pd.DataFrame(elig_rows)
        synth_records = []
        for i in range(12):
            if i % 2 == 0:
                synth_records.append(
                    {
                        "id": f"syn_diag_{i}",
                        "text": f"NodiPay alert {i}: Check statement at https://portal.example.com/s{i} or call 01700-00001{i % 10}.",
                        "label_3class": "Normal",
                        "language": "English",
                        "domain": "banking_mfs",
                        "family_id": f"syn_diag_fam_{i}",
                        "brand_names": ["NodiPay"],
                    }
                )
            else:
                synth_records.append(
                    {
                        "id": f"syn_diag_{i}",
                        "text": f"RupshaShop promo {i}: Enjoy 20% off fresh groceries at https://shop.example.org/g{i} today.",
                        "label_3class": "Promotional",
                        "language": "English",
                        "domain": "shopping_ecommerce",
                        "family_id": f"syn_diag_fam_{i}",
                        "brand_names": ["RupshaShop"],
                    }
                )

        ext = compute_extended_stage_b_diagnostics(
            elig_df,
            synth_records,
            n_splits=3,
            brand_registry={"rupshashop": ["RupshaShop", "রূপসাশপ"]},
        )
        for view_name in TEXT_VIEW_NAMES:
            self.assertIn(view_name, ext["three_class_before_pilot"])
            self.assertIn(view_name, ext["three_class_after_pilot"])
            self.assertIn(view_name, ext["is_synthetic_probes"]["all_vs_real"])
            self.assertIn(view_name, ext["is_synthetic_probes"]["synth_vs_real_smishing"])
            self.assertIn(view_name, ext["is_synthetic_probes"]["synthetic_normal_vs_real_normal"])
            self.assertIn(view_name, ext["is_synthetic_probes"]["synthetic_promo_vs_real_promo"])
            self.assertIn(view_name, ext["is_synthetic_probes"]["synthetic_normal_vs_real_normal_stratum_3way"])
            self.assertIn(view_name, ext["is_synthetic_probes"]["synthetic_normal_vs_real_normal_stratum_2way"])
            self.assertIn(view_name, ext["is_synthetic_probes"]["synthetic_promo_vs_real_promo_stratum_3way"])
            self.assertIn(view_name, ext["is_synthetic_probes"]["synthetic_promo_vs_real_promo_stratum_2way"])
            self.assertIn(view_name, ext["counterfactual_probes"]["views"])
        self.assertIn("surface_8feat", ext["is_synthetic_probes"]["all_vs_real"])
        self.assertEqual(ext["counterfactual_probes"]["n_probed"], 12)
        self.assertGreaterEqual(ext["is_synthetic_probes"]["synthetic_normal_vs_real_normal_stratum_3way"]["matched_strata_count"], 1)
        self.assertGreaterEqual(ext["is_synthetic_probes"]["synthetic_promo_vs_real_promo_stratum_2way"]["matched_strata_count"], 1)

        # 2. CodeMix new_batch diversity enforcement rejects hook + pure EN
        v2_dir = ROOT / "data/processed/curation_v2"
        cm_cands = [
            _cand(
                "cm_bad_hook",
                "আজই কিনুন! Enjoy flat 30% discount on all winter items at https://example.com/sale",
                "Promotional",
                "CodeMix",
            ),
            _cand(
                "cm_good_intra",
                "PurbachalBank-এ এই মাসে cashback offer চলছে, আপনার shopping bill এখনই pay করুন।",
                "Promotional",
                "CodeMix",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bpath = tmp_path / "cm.jsonl"
            write_jsonl(bpath, cm_cands)
            cm_summary = screen_stage_b_pilot(
                [bpath],
                v2_dir=v2_dir,
                out_dir=tmp_path / "out",
                include_extended_diagnostics=False,
                diversity_mode="new_batch",
            )
            rej = load_jsonl(tmp_path / "out/rejected.jsonl")
            acc = load_jsonl(tmp_path / "out/accepted.jsonl")
            self.assertEqual([r["id"] for r in acc], ["cm_good_intra"])
            self.assertEqual([r["id"] for r in rej], ["cm_bad_hook"])
            self.assertIn("codemix_hook_plus_pure_en_disallowed", rej[0]["rejection_reasons"])
            self.assertIn("codemix_and_entity_diversity", cm_summary)

        # 3. --dry-run refuses production targets and leaves production artifacts untouched
        spec = importlib.util.spec_from_file_location("validate_stage_b_pilot", ROOT / "scripts/validate_stage_b_pilot.py")
        val_mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(val_mod)

        with self.assertRaises(ValueError):
            val_mod.run_validation(["--dry-run", "--out-dir", str(ROOT / "data/generated/curation_v2")])
        with self.assertRaises(ValueError):
            val_mod.run_validation(["--dry-run", "--report-path", str(ROOT / "reports/stage_b_pilot_v2.md")])

        prod_files = [
            ROOT / "data/generated/curation_v2/accepted.jsonl",
            ROOT / "data/generated/curation_v2/pilot_validation_summary.json",
            ROOT / "reports/stage_b_pilot_v2.md",
        ]
        before_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in prod_files}

        with tempfile.TemporaryDirectory() as tmp:
            scratch_out = Path(tmp) / "dry_out"
            dry_summary = val_mod.run_validation(
                [
                    "--dry-run",
                    "--skip-extended-diagnostics",
                    "--batches",
                    str(ROOT / "data/generated/curation_v2/accepted.jsonl"),
                    "--out-dir",
                    str(scratch_out),
                ]
            )
            self.assertTrue(dry_summary["dry_run"])
            self.assertEqual(dry_summary["accepted_count"], 300)
            self.assertEqual(dry_summary["rejected_count"], 0)
            self.assertTrue((scratch_out / "pilot_validation_summary.json").exists())
            self.assertTrue((scratch_out / "stage_b_pilot_dry_run_report.md").exists())

            # Bug fixes (1) & (3): canonical brand aliases merge (BornoTelecom+বর্ণটেলিকম=25) & Section 3e actuals render non-empty
            cm_div = dry_summary["codemix_and_entity_diversity"]
            self.assertEqual(cm_div["max_brand_count"], 25)
            self.assertEqual(cm_div["brandless_rate"], 0.023)
            report_md = (scratch_out / "stage_b_pilot_dry_run_report.md").read_text(encoding="utf-8")
            self.assertIn("| `max_brand_count` | `25` | `<= 5` | `False` |", report_md)
            self.assertIn("| `brandless_rate` | `0.023` | `>= 0.3` | `False` |", report_md)

        # Bug fix (4): _check_domain_and_phone catches bare t.me & multi-label non-reserved evil.edu.bd,
        # and allows currency BDT 16000 / 16000/- while rejecting 16216 shortcodes.
        with tempfile.TemporaryDirectory() as tmp:
            import json
            tmp_path = Path(tmp)
            dom_cands = [
                _cand("bad_tme", "NodiPay update: Visit t.me/verify_now for details.", "Normal", "English"),
                _cand("bad_edubd", "Chayaneer notice: Check portal at evil.edu.bd/login today.", "Normal", "English"),
                _cand("ok_currency", "KothaEdu fee notice: Pay BDT 16000 or 16000/- at the portal today.", "Normal", "English"),
                _cand("bad_sc", "NodiPay support: Call 16216 for account verification.", "Normal", "English"),
            ]
            dom_cands[2]["brand"] = "KothaEdu"
            b_dom = tmp_path / "dom.jsonl"
            reg_path = tmp_path / "brand_registry.json"
            reg_path.write_text(json.dumps({"kothaedu": ["KothaEdu", "কথাএডু"]}), encoding="utf-8")
            write_jsonl(b_dom, dom_cands)
            reg_summary = val_mod.run_validation(
                [
                    "--dry-run",
                    "--skip-extended-diagnostics",
                    "--batches",
                    str(b_dom),
                    "--brand-registry",
                    str(reg_path),
                    "--out-dir",
                    str(tmp_path / "out"),
                ]
            )
            acc_dom = [r["id"] for r in load_jsonl(tmp_path / "out/accepted.jsonl")]
            rej_dom = {r["id"]: "|".join(r["rejection_reasons"]) for r in load_jsonl(tmp_path / "out/rejected.jsonl")}
            self.assertEqual(acc_dom, ["ok_currency"])
            self.assertIn("non_reserved_domain:t.me", rej_dom["bad_tme"])
            self.assertIn("non_reserved_domain:evil.edu.bd", rej_dom["bad_edubd"])
            self.assertIn("non_fictional_phone_number", rej_dom["bad_sc"])
            self.assertEqual(reg_summary["accepted_count"], 1)

        after_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in prod_files}
        self.assertEqual(before_hashes, after_hashes)


if __name__ == "__main__":
    unittest.main()
