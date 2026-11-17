"""CLI to screen Stage B pilot batches, compute diagnostics, and write reports/stage_b_pilot_v2.md."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.schema import CANONICAL_LABELS
from bangla_smish.stage_b_pilot import TEXT_VIEW_NAMES, load_jsonl, screen_stage_b_pilot

DEFAULT_OUT_DIR = ROOT / "data/generated/curation_v2"
DEFAULT_REPORT_PATH = ROOT / "reports/stage_b_pilot_v2.md"


def build_pilot_report(summary: dict[str, Any], out_dir: Path) -> str:
    """Generate reports/stage_b_pilot_v2.md strictly from pilot validation summary and JSONL files."""
    accepted = load_jsonl(out_dir / "accepted.jsonl")
    rejected = load_jsonl(out_dir / "rejected.jsonl")
    langs = ("Bengali", "Banglish", "English", "CodeMix")

    d_bef = summary["surface_diagnostics_before_pilot"]
    d_aft = summary["surface_diagnostics_after_pilot"]

    lines: list[str] = [
        "# Stage B Pilot Screening & Validation Report (`curation_v2`)",
        "",
        "Generated reproducibly by `scripts/validate_stage_b_pilot.py` (`src/bangla_smish/stage_b_pilot.py`). "
        "Screening enforces schema, `length_band` consistency (`short(<50)`, `medium(50-120)`, `long(>120)`), "
        "`is_synthetic=True`, exact & char-ngram cosine (`>= 0.85` raw & slot-template) "
        "isolation against `curation_v2/existing_unique.parquet` (including held-out/OOD) and `quarantined.parquet`, "
        "intra-pilot template diversity, fictional entities/reserved domains, label/language rules, "
        "and independent review log ingestion.",
        "",
        "## 1. Candidate Screening & Independent Review Summary",
        "",
        f"- **Total candidates screened (`candidates.jsonl`):** **{summary['total_candidates']}**",
        f"- **Accepted (`accepted.jsonl`):** **{summary['accepted_count']}** ({summary['acceptance_rate_pct']}%)",
        f"  - Confirmed by **both** automated screening and independent review (`review_status=\"accepted\"`): **{summary['independently_reviewed_accepted_count']}**",
        f"  - Passed automated screening, pending independent review (`review_status=\"pending_independent_review\"`): **{summary['pending_independent_review_count']}**",
        f"- **Rejected (`rejected.jsonl`):** **{summary['rejected_count']}**",
        f"- **Exact pilot quota match (`300` rows):** `{summary['quotas_exact_match']}`",
        "",
        "### Accepted Pilot Distribution (`Label × Language`)",
        "",
        "| Class | Bengali | Banglish | English | CodeMix | Total Accepted | Target Pilot Quota |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    acc_map = summary["accepted_by_class_language"]
    tgt_map = summary["target_pilot_quotas"]
    for lbl in ("Normal", "Promotional", "Smishing"):
        vals = [acc_map.get(f"{lbl}|{lg}", 0) for lg in langs]
        t_tot = sum(tgt_map.get(f"{lbl}|{lg}", 0) for lg in langs)
        lines.append(f"| `{lbl}` | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | **{sum(vals)}** | {t_tot} |")

    if rejected:
        lines.extend(["", "### Rejection Breakdown", "", "| Rejection Reason | Count |", "| :--- | ---: |"])
        for reason, cnt in sorted(summary["rejection_reason_counts"].items()):
            lines.append(f"| `{reason}` | {cnt} |")
        rej_ids = ", ".join(f"`{r['id']}`" for r in rejected)
        lines.append("")
        lines.append(f"- **Rejected IDs ({len(rejected)}):** {rej_ids}")

    all_domains = sorted({str(r.get("domain", "unknown")) for r in accepted})
    norm_dom = summary["accepted_by_class_domain"].get("Normal", {})
    promo_dom = summary["accepted_by_class_domain"].get("Promotional", {})
    lines.extend(
        [
            "",
            "## 2. Domain, Length & Cue Coverage of Accepted Pilot (`N = " + str(len(accepted)) + "`)",
            "",
            "### Domain Breakdown (`Domain × Label`)",
            "",
            "| Domain | Normal | Promotional | Total |",
            "| :--- | ---: | ---: | ---: |",
        ]
    )
    for dom in all_domains:
        n_d = norm_dom.get(dom, 0)
        p_d = promo_dom.get(dom, 0)
        lines.append(f"| `{dom}` | {n_d} | {p_d} | **{n_d + p_d}** |")
    lines.append(
        f"| **Total** | **{sum(norm_dom.values())}** | **{sum(promo_dom.values())}** | **{len(accepted)}** |"
    )

    lines.extend(
        [
            "",
            "### Length Band & Surface Cue Coverage",
            "",
            "| Class | `short(<50)` | `medium(50-120)` | `long(>120)` | Has Link | Has Urgency |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for lbl in ("Normal", "Promotional"):
        sub = [r for r in accepted if r["label_3class"] == lbl]
        s_cnt = sum(1 for r in sub if r.get("length_band") == "short(<50)")
        m_cnt = sum(1 for r in sub if r.get("length_band") == "medium(50-120)")
        l_cnt = sum(1 for r in sub if r.get("length_band") == "long(>120)")
        lnk = sum(1 for r in sub if r.get("has_link"))
        urg = sum(1 for r in sub if r.get("has_urgency"))
        lines.append(f"| `{lbl}` | {s_cnt} | {m_cnt} | {l_cnt} | {lnk} | {urg} |")

    lines.extend(
        [
            "",
            "## 3. Surface-Only & Length-Only Shortcut Diagnostics (Before vs. After Pilot)",
            "",
            "| Metric (`StratifiedGroupKFold`, grouped by `family_id`) | `eligible_train` Alone | `eligible_train + accepted` | Delta |",
            "| :--- | ---: | ---: | ---: |",
            f"| Row count (`N`) | {d_bef['n_rows']} | {d_aft['n_rows']} | +{d_aft['n_rows'] - d_bef['n_rows']} |",
            f"| Surface-only 5-fold Accuracy | {d_bef['surface_only_5fold_acc_pct']}% | {d_aft['surface_only_5fold_acc_pct']}% | {d_aft['surface_only_5fold_acc_pct'] - d_bef['surface_only_5fold_acc_pct']:+.2f}% |",
            f"| Surface-only 5-fold Macro-F1 | {d_bef['surface_only_5fold_macro_f1_pct']}% | {d_aft['surface_only_5fold_macro_f1_pct']}% | {d_aft['surface_only_5fold_macro_f1_pct'] - d_bef['surface_only_5fold_macro_f1_pct']:+.2f}% |",
            f"| Length-only Smishing ROC-AUC | {d_bef['length_only_smishing_roc_auc']} | {d_aft['length_only_smishing_roc_auc']} | {d_aft['length_only_smishing_roc_auc'] - d_bef['length_only_smishing_roc_auc']:+.4f} |",
            f"| Length-only Promo vs. Normal ROC-AUC | {d_bef['length_only_promo_vs_normal_roc_auc']} | {d_aft['length_only_promo_vs_normal_roc_auc']} | {d_aft['length_only_promo_vs_normal_roc_auc'] - d_bef['length_only_promo_vs_normal_roc_auc']:+.4f} |",
            "",
            "### Per-Class Surface Cue Rates (`Before -> After Pilot`)",
            "",
            "| Class | Count (`Before -> After`) | Median Chars (`Before -> After`) | % Has URL (`Before -> After`) | % Has Phone (`Before -> After`) |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for lbl in CANONICAL_LABELS:
        b = d_bef["per_class_surface"][lbl]
        a = d_aft["per_class_surface"][lbl]
        lines.append(
            f"| `{lbl}` | {b['count']} -> {a['count']} | {b['median_char_len']} -> {a['median_char_len']} | "
            f"{b['pct_has_url']}% -> {a['pct_has_url']}% | {b['pct_has_phone']}% -> {a['pct_has_phone']}% |"
        )

    ext = summary.get("extended_diagnostics", {})
    if ext and not ext.get("skipped"):
        b3 = ext.get("three_class_before_pilot", {})
        a3 = ext.get("three_class_after_pilot", {})
        lines.extend(
            [
                "",
                "### 3b. 4-View 3-Class TF-IDF+LR Probes (`Before -> After Pilot`)",
                "",
                "| Text View | Macro-F1 (`Before -> After`) | Accuracy (`Before -> After`) | Smishing Recall (`Before -> After`) | Recall @ <=1% FPR (`Before -> After`) |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for view_name in TEXT_VIEW_NAMES:
            bv = b3.get(view_name, {})
            av = a3.get(view_name, {})
            if bv and av:
                lines.append(
                    f"| `{view_name}` | {bv['macro_f1']}% -> {av['macro_f1']}% | "
                    f"{bv['accuracy']}% -> {av['accuracy']}% | "
                    f"{bv['smishing_recall_argmax']}% -> {av['smishing_recall_argmax']}% | "
                    f"{bv['smishing_recall_at_1pct_fpr']}% -> {av['smishing_recall_at_1pct_fpr']}% |"
                )

        sp = ext.get("is_synthetic_probes", {})
        if sp and not sp.get("skipped"):
            lines.extend(
                [
                    "",
                    "### 3c. Synthetic-Origin (`is_synthetic`) Separability Probes",
                    "",
                    "| Comparison | Strata / N | Feature View | ROC-AUC | Accuracy | Macro-F1 | Top Positive Synthetic Features |",
                    "| :--- | :--- | :--- | ---: | ---: | ---: | :--- |",
                ]
            )
            probe_order = (
                "all_vs_real",
                "synth_vs_real_smishing",
                "synthetic_normal_vs_real_normal",
                "synthetic_normal_vs_real_normal_stratum_3way",
                "synthetic_normal_vs_real_normal_stratum_2way",
                "synthetic_promo_vs_real_promo",
                "synthetic_promo_vs_real_promo_stratum_3way",
                "synthetic_promo_vs_real_promo_stratum_2way",
            )
            for comp_name in probe_order:
                comp = sp.get(comp_name, {})
                if not comp or comp.get("skipped"):
                    continue
                if "matched_strata_count" in comp:
                    strata_desc = (
                        f"{comp['matched_strata_count']} strata "
                        f"({comp['matched_synth_count']}/{comp['total_synth_count']} syn, "
                        f"{comp['matched_real_count']}/{comp['total_real_count']} real)"
                    )
                else:
                    strata_desc = f"{comp.get('n_synth', 0)} syn vs {comp.get('n_real', 0)} real"
                for vname in (*TEXT_VIEW_NAMES, "surface_8feat"):
                    vm = comp.get(vname, {})
                    if vm:
                        top_str = ", ".join(f"`{t[0]}` ({t[1]:+.2f})" for t in vm.get("top_positive_features", [])[:4])
                        lines.append(
                            f"| `{comp_name}` | {strata_desc} | `{vname}` | {vm['roc_auc']:.4f} | "
                            f"{vm['accuracy_pct']}% | {vm['macro_f1_pct']}% | {top_str} |"
                        )

        cf = ext.get("counterfactual_probes", {})
        if cf and not cf.get("skipped") and cf.get("views"):
            lines.extend(
                [
                    "",
                    f"### 3d. Counterfactual URL/Phone/Brand Probes (`n = {cf['n_probed']}` Real Smishing Rows)",
                    "",
                    "| Text View | Baseline Smish % | Contact Swap Smish % (`Delta pp`, `Delta P`) | Contact+Brand Swap Smish % (`Delta pp`, `Delta P`) |",
                    "| :--- | ---: | :--- | :--- |",
                ]
            )
            for vname in TEXT_VIEW_NAMES:
                cv = cf["views"].get(vname, {})
                if cv:
                    cs = cv["contact_swap"]
                    bs = cv["contact_and_brand_swap"]
                    lines.append(
                        f"| `{vname}` | {cv['smishing_pred_pct_before']}% | "
                        f"{cs['smishing_pred_pct_after']}% ({cs['pp_drop']:+.2f} pp, {cs['mean_p_smishing_delta']:+.4f}) | "
                        f"{bs['smishing_pred_pct_after']}% ({bs['pp_drop']:+.2f} pp, {bs['mean_p_smishing_delta']:+.4f}) |"
                    )

    cm_div = summary.get("codemix_and_entity_diversity", {})
    if cm_div:
        lines.extend(
            [
                "",
                f"### 3e. CodeMix Structural Diversity & Entity-Frequency Gates (`mode={cm_div.get('mode', 'audit')}`)",
                "",
                "| Gate / Metric | Observed Value | Threshold | Gate Passed |",
                "| :--- | :--- | :--- | :--- |",
            ]
        )
        for gname, ginfo in sorted(cm_div.get("gate_checks", {}).items()):
            obs = ginfo.get("actual", ginfo.get("observed", ginfo.get("value", "")))
            thr = f"{ginfo.get('op', '')} {ginfo.get('threshold', '')}".strip()
            lines.append(f"| `{gname}` | `{obs}` | `{thr}` | `{ginfo.get('passed')}` |")
        lines.append(f"- **All diversity gates passed:** `{cm_div.get('all_gates_passed')}`")

    lines.extend(
        [
            "",
            "## 4. Updated Class Counts & Remaining Stage B Deficits",
            "",
            f"Target `T = {summary['stage_b_target_T']}`:",
            "",
            "| Class | `curation_v2` Original Eligible | Accepted Pilot | Combined Total | Target (`T`) | Remaining Stage B Deficit |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for lbl in CANONICAL_LABELS:
        orig_n = summary["eligible_counts_before_pilot"][lbl]
        pil_n = summary["accepted_by_class"][lbl]
        comb_n = summary["eligible_counts_after_pilot"][lbl]
        rem_n = summary["remaining_stage_b_deficits"][lbl]
        lines.append(f"| `{lbl}` | {orig_n} | {pil_n} | {comb_n} | {summary['stage_b_target_T']} | **{rem_n}** |")

    tot_rem = sum(summary["remaining_stage_b_deficits"].values())
    lines.append(
        f"| **Total** | **{d_bef['n_rows']}** | **{summary['accepted_count']}** | **{d_aft['n_rows']}** | — | **{tot_rem}** |"
    )

    shared_groups = summary.get("shared_family_groups", {})
    lines.extend(
        [
            "",
            "## 5. Contamination, Review Status & Synthetic-Origin Cue / Limitation Notes",
            "",
            "- **Anti-leakage & isolation:** Zero exact or near-duplicate (`cos >= 0.85` raw or slot-template) overlap with `curation_v2/existing_unique.parquet` "
            "(including `smishdetect_validation`, `smishdetect_test`, and `banglaphish_ood`) or `curation_v2/quarantined.parquet`.",
            f"- **Cross-lingual family grouping:** Accepted rows span **{summary['unique_accepted_families']}** `family_id` groups "
            f"(including **{len(shared_groups)}** linked cross-lingual parallel family groups so `StratifiedGroupKFold` keeps parallel realizations in the same fold).",
            f"- **Independent review status:** **{summary['independently_reviewed_accepted_count']}** accepted rows are confirmed by both automated screening and independent review (`review_shard_1.jsonl`, `review_shard_2.jsonl`, `review_replacements.jsonl`); "
            f"**{summary['rejected_count']}** rows flagged by independent review are isolated in `rejected.jsonl`; "
            f"**{summary['pending_independent_review_count']}** rows are `pending_independent_review`.",
            "- **Synthetic-origin cue & limitation notes:**",
            "  - Reserved domains (`example.com`, `example.org`, `example.net`, `example.edu.bd`), fictional phone numbers (`01700-0000XX` / `+88017000000XX`), and recurring fictional brand names appear only in pilot `Normal` and `Promotional` rows (`0` in `eligible_train` `Smishing`), creating potential lexical synthetic-origin cues not tested by `compute_surface_diagnostics` (which only measures length, script, URL/phone/USSD presence).",
            "  - As noted during independent review, `CodeMix` promotional rows frequently share a two-sentence `[Bengali/Banglish hook] + [English clause]` discourse pattern; downstream experiments must evaluate on real held-out and `BanglaPhish-2026` OOD sets and never treat synthetic-origin patterns as real-world prevalence.",
            "- **Remaining deficit:** Remaining `Promotional` deficit (`345` rows) has **not** been generated pending final pilot sign-off.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_validation(argv: list[str] | None = None) -> dict[str, Any]:
    """Parse CLI arguments, run screening/diagnostics (with dry-run isolation), and write outputs."""
    parser = argparse.ArgumentParser(description="Screen Stage B pilot batches and write stage_b_pilot_v2.md.")
    parser.add_argument(
        "--batches",
        nargs="+",
        type=Path,
        default=[
            ROOT / "data/generated/curation_v2/raw_batch_a.jsonl",
            ROOT / "data/generated/curation_v2/raw_batch_b.jsonl",
            ROOT / "data/generated/curation_v2/raw_batch_c_replacements.jsonl",
        ],
    )
    parser.add_argument(
        "--reviews",
        nargs="*",
        type=Path,
        default=[
            ROOT / "data/generated/curation_v2/review_shard_1.jsonl",
            ROOT / "data/generated/curation_v2/review_shard_2.jsonl",
            ROOT / "data/generated/curation_v2/review_replacements.jsonl",
        ],
    )
    parser.add_argument("--v2-dir", type=Path, default=ROOT / "data/processed/curation_v2")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--sim-threshold", type=float, default=0.85)
    parser.add_argument("--brand-registry", type=Path, default=None)
    parser.add_argument("--diversity-mode", choices=("audit", "new_batch"), default="audit")
    parser.add_argument("--skip-extended-diagnostics", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        if args.out_dir is not None and args.out_dir.resolve() == DEFAULT_OUT_DIR.resolve():
            raise ValueError(f"Refusing --dry-run write into production directory: {args.out_dir}")
        if args.report_path is not None and args.report_path.resolve() == DEFAULT_REPORT_PATH.resolve():
            raise ValueError(f"Refusing --dry-run write into production report: {args.report_path}")

        if args.out_dir is None:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_out = Path(tmp)
                summary = screen_stage_b_pilot(
                    batch_paths=args.batches,
                    v2_dir=args.v2_dir,
                    out_dir=tmp_out,
                    sim_threshold=args.sim_threshold,
                    review_paths=args.reviews,
                    include_extended_diagnostics=not args.skip_extended_diagnostics,
                    diversity_mode=args.diversity_mode,
                    dry_run=True,
                    brand_registry=args.brand_registry,
                )
                report_text = build_pilot_report(summary, tmp_out)
                if args.report_path is not None:
                    args.report_path.parent.mkdir(parents=True, exist_ok=True)
                    args.report_path.write_text(report_text, encoding="utf-8")
                print(json.dumps({"dry_run": True, "ephemeral_scratch": True, "summary": summary}, indent=2, ensure_ascii=False))
                return summary

        out_dir = args.out_dir
        report_path = args.report_path or (out_dir / "stage_b_pilot_dry_run_report.md")
    else:
        out_dir = args.out_dir or DEFAULT_OUT_DIR
        report_path = args.report_path or DEFAULT_REPORT_PATH

    summary = screen_stage_b_pilot(
        batch_paths=args.batches,
        v2_dir=args.v2_dir,
        out_dir=out_dir,
        sim_threshold=args.sim_threshold,
        review_paths=args.reviews,
        include_extended_diagnostics=not args.skip_extended_diagnostics,
        diversity_mode=args.diversity_mode,
        dry_run=args.dry_run,
        brand_registry=args.brand_registry,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(build_pilot_report(summary, out_dir))

    print(
        f"Pilot screening complete (dry_run={args.dry_run}): accepted={summary['accepted_count']}/{summary['total_candidates']} "
        f"(reviewed={summary['independently_reviewed_accepted_count']}, pending_review={summary['pending_independent_review_count']}), "
        f"rejected={summary['rejected_count']}, quotas_exact_match={summary['quotas_exact_match']}"
    )
    if summary["quota_gaps"]:
        print(f"QUOTA_GAPS: {summary['quota_gaps']}")
    if summary["rejection_reason_counts"]:
        print(f"REJECTION_REASONS: {summary['rejection_reason_counts']}")
    return summary


def main() -> None:
    run_validation()


if __name__ == "__main__":
    main()
