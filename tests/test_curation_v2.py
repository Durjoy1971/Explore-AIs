"""Unit and integration tests for Stage A-followup (`curation_v2`) train-only audit pipeline."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.curation_v2 import AuditRuleGroup, run_stage_a_curation_v2


class TestCurationV2(unittest.TestCase):
    def test_rejects_non_train_pool_audit_id(self) -> None:
        v1_dir = ROOT / "data" / "processed" / "curation_v1"
        ex_v1 = pd.read_parquet(v1_dir / "existing_unique.parquet")
        eval_cid = str(ex_v1.loc[ex_v1["eval_group"] == "smishdetect_test", "canonical_id"].iloc[0])
        ood_cid = str(ex_v1.loc[ex_v1["eval_group"] == "banglaphish_ood", "canonical_id"].iloc[0])

        for bad_cid in (eval_cid, ood_cid):
            bad_grp = (
                AuditRuleGroup(
                    ids=(bad_cid,),
                    action="reclassify_label",
                    new_label="Normal",
                    audit_source="test",
                    category="bad",
                    rationale="should fail",
                ),
            )
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    run_stage_a_curation_v2(v1_dir=v1_dir, v2_dir=Path(tmp), label_groups=bad_grp, lang_groups=())

    def test_curation_v2_preserves_heldout_and_reconciles_counts(self) -> None:
        v1_dir = ROOT / "data" / "processed" / "curation_v1"
        with tempfile.TemporaryDirectory() as tmp:
            v2_dir = Path(tmp)
            summary = run_stage_a_curation_v2(v1_dir=v1_dir, v2_dir=v2_dir)

            ex_v1 = pd.read_parquet(v1_dir / "existing_unique.parquet")
            ex_v2 = pd.read_parquet(v2_dir / "existing_unique.parquet")
            el_v2 = pd.read_parquet(v2_dir / "eligible_train.parquet")
            qu_v1 = pd.read_parquet(v1_dir / "quarantined.parquet")
            qu_v2 = pd.read_parquet(v2_dir / "quarantined.parquet")
            pr_v2 = pd.read_parquet(v2_dir / "provenance.parquet")
            ad_v2 = pd.read_parquet(v2_dir / "audit_decisions.parquet")

            # Held-out & OOD rows must be 100% identical to curation_v1
            cols_check = ["canonical_id", "text", "label_3class", "language", "family_id", "eval_group", "eligible_for_training"]
            ho_v1 = ex_v1[ex_v1["eval_group"] != "train_pool"][cols_check].sort_values("canonical_id").reset_index(drop=True)
            ho_v2 = ex_v2[ex_v2["eval_group"] != "train_pool"][cols_check].sort_values("canonical_id").reset_index(drop=True)
            pd.testing.assert_frame_equal(ho_v1, ho_v2)

            # Counts & isolation invariants
            self.assertEqual(len(el_v2), 3999)
            self.assertEqual(len(ex_v2), len(ex_v1) - 126)
            self.assertEqual(len(qu_v2), len(qu_v1) + 126)
            self.assertEqual(len(pr_v2), 18357)
            self.assertEqual(set(el_v2["label_3class"].unique()), {"Normal", "Promotional", "Smishing"})
            self.assertEqual(set(el_v2["language"].unique()), {"Bengali", "Banglish", "English", "CodeMix"})
            self.assertEqual(el_v2.groupby("family_id")["curation_cv_fold"].nunique().max(), 1)
            self.assertEqual(len(set(el_v2["family_id"]) & set(ex_v2.loc[~ex_v2["eligible_for_training"], "family_id"])), 0)
            self.assertEqual(summary["coverage_profile"]["eligible_train_by_class"], {"Normal": 1509, "Promotional": 942, "Smishing": 1548})
            self.assertEqual(summary["coverage_profile"]["stage_b_deficits"], {"Normal": 39, "Promotional": 606, "Smishing": 0})
            self.assertLessEqual(summary["pilot_plan"]["pilot_total_rows"], 300)
            self.assertEqual(int((ad_v2["final_status"] == "quarantined").sum()), 126)


if __name__ == "__main__":
    unittest.main()
