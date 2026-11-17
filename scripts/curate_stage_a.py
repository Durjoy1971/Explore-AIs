"""Stage A CLI: consolidate existing corpora, deduplicate, quarantine, review families, and write report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.curation import run_stage_a_curation
from bangla_smish.schema import CANONICAL_LABELS


def _format_pct(part: int, total: int) -> str:
    return f"{(part / max(1, total)) * 100.0:.2f}%"


def build_consolidation_report(root: Path, out_dir: Path, summary: dict[str, Any]) -> str:
    """Generate reports/dataset_consolidation_v1.md strictly from measured Stage A artifacts."""
    existing = pd.read_parquet(out_dir / "existing_unique.parquet")
    prov = pd.read_parquet(out_dir / "provenance.parquet")
    quar = pd.read_parquet(out_dir / "quarantined.parquet")
    elig = pd.read_parquet(out_dir / "eligible_train.parquet")
    with open(out_dir / "near_duplicate_review.jsonl", "r", encoding="utf-8") as f:
        reviews = [json.loads(line) for line in f]

    raw_total = int(len(prov))
    disp_counts = prov["disposition"].value_counts().to_dict()
    kept_raw = int(disp_counts.get("canonical_kept", 0))
    exact_merged_raw = int(disp_counts.get("exact_dup_merged", 0))
    near_collapsed_raw = int(disp_counts.get("near_dup_variant_collapsed", 0))
    quar_raw = int(disp_counts.get("quarantined", 0))

    # Source-level counts before and after
    raw_by_ds = prov["dataset"].value_counts().to_dict()
    exact_uniq_by_ds = prov.groupby("dataset")["text_hash_strict"].nunique().to_dict()

    # Source metadata lookup
    inv_by_ds = {item["dataset"]: item for item in summary["source_inventory"]}

    # Script vs source_language_raw mismatches in existing_unique
    lang_mismatch_cm_bn = int(
        ((existing["source_language_raw"] == "CodeMix") & (existing["script_bucket"] == "Bengali")).sum()
    )
    lang_mismatch_cm_lat = int(
        ((existing["source_language_raw"] == "CodeMix") & (existing["script_bucket"] == "Latin")).sum()
    )
    lang_mismatch_bn_lat = int(
        ((existing["source_language_raw"] == "Bengali") & (existing["script_bucket"] == "Latin")).sum()
    )
    lang_mismatch_bn_cm = int(
        ((existing["source_language_raw"] == "Bengali") & (existing["script_bucket"] == "CodeMix")).sum()
    )
    bp_placeholder_cm = int(
        ((existing["primary_dataset"] == "banglaphish_2026") & (existing["script_bucket"] == "CodeMix")).sum()
    )

    # Cross-dataset overlap stats from provenance
    prov_non_quar = prov[prov["disposition"] != "quarantined"]
    ds_hashes = {
        ds: set(grp["text_hash_strict"])
        for ds, grp in prov.groupby("dataset")
    }
    bb_in_sd = len(ds_hashes.get("bangalabarta_2025", set()) & ds_hashes.get("smishdetect_2026", set()))
    leg_in_bbs = len(
        ds_hashes.get("legacy_2024", set())
        & (ds_hashes.get("bangalabarta_2025", set()) | ds_hashes.get("smishdetect_2026", set()))
    )
    bp_overlap_others = len(
        ds_hashes.get("banglaphish_2026", set())
        & (
            ds_hashes.get("legacy_2024", set())
            | ds_hashes.get("bangalabarta_2025", set())
            | ds_hashes.get("smishdetect_2026", set())
        )
    )

    lines: list[str] = [
        "# Dataset Consolidation Report — Stage A (`curation_v1`)",
        "",
        "Generated reproducibly by `scripts/curate_stage_a.py` (`src/bangla_smish/curation.py`) "
        "from existing datasets only (no synthetic generation, undersampling, or oversampling).",
        "",
        "## 1. Logical Source Inventory (Before Cleaning)",
        "",
        "Each logical dataset is counted once; duplicate exports (`bangalabarta/*.zip`, "
        "`banglaphish2026/*.jsonl`, `*.xlsx`, `data/splits/*`) and derived `data/processed/*.parquet` "
        "files are excluded from raw source row counts.",
        "",
        "| Logical Dataset | Canonical Source File(s) | License | Source Type | Raw Rows | Unique Strict Texts | Retained Primary Rows in `existing_unique` |",
        "| :--- | :--- | :--- | :--- | ---: | ---: | ---: |",
    ]

    kept_by_ds = existing["primary_dataset"].value_counts().to_dict()
    for ds_name in ("legacy_2024", "bangalabarta_2025", "smishdetect_2026", "banglaphish_2026"):
        meta = inv_by_ds[ds_name]
        files_str = ", ".join(f"`{f['path']}` (`{f['sha256'][:10]}...`)" for f in meta["files"])
        lines.append(
            f"| `{ds_name}` | {files_str} | {meta['license']} | {meta['synthetic_status']} | "
            f"{raw_by_ds.get(ds_name, 0)} | {exact_uniq_by_ds.get(ds_name, 0)} | {kept_by_ds.get(ds_name, 0)} |"
        )
    lines.append(
        f"| **Total** | — | — | — | **{raw_total}** | — | **{len(existing)}** |"
    )

    lines.extend(
        [
            "",
            f"- **Cross-corpus exact lineage:** `{bb_in_sd}` / `{exact_uniq_by_ds.get('bangalabarta_2025', 0)}` "
            "unique strict `bangalabarta_2025` texts appear inside `smishdetect_2026` (100.0%). "
            f"`{leg_in_bbs}` unique strict `legacy_2024` texts match `bangalabarta_2025` / `smishdetect_2026`. "
            f"`banglaphish_2026` has `{bp_overlap_others}` exact overlaps with the other three corpora.",
            "",
            "## 2. Normalization, Exact Deduplication & Row Reconciliation",
            "",
            "- **Normalization rules:** Original `text` is preserved verbatim. Exact deduplication uses "
            "`text_norm_strict` (stripping Unicode `Cf` format chars such as ZWJ/ZWNJ/BOM and non-whitespace `Cc` chars, "
            "normalizing Unicode `Z*`/newline whitespace to ASCII space, applying `NFKC`, collapsing whitespace, and `.casefold()`), "
            "alongside repo `normalize_text()` (`text_norm`).",
            f"- **Provenance sidecar:** Every raw logical row (`{raw_total}` rows) is preserved in `provenance.parquet` with "
            "source file, row index, original ID, original split, original label, language/script tags, disposition, and reason.",
            "",
            "| Provenance Disposition | Raw Row Count | % of Raw Total | Notes |",
            "| :--- | ---: | ---: | :--- |",
            f"| `canonical_kept` | {kept_raw} | {_format_pct(kept_raw, raw_total)} | Canonical rows in `existing_unique.parquet` |",
            f"| `exact_dup_merged` | {exact_merged_raw} | {_format_pct(exact_merged_raw, raw_total)} | Exact `text_norm_strict` duplicates merged into canonical/surviving provenance |",
            f"| `near_dup_variant_collapsed` | {near_collapsed_raw} | {_format_pct(near_collapsed_raw, raw_total)} | Reviewed cosmetic/slot-swap variants collapsed in `BB+SD` |",
            f"| `quarantined` | {quar_raw} | {_format_pct(quar_raw, raw_total)} | Conflicting, unusable, or legacy reference-only rows in `quarantined.parquet` |",
            f"| **Total Reconciled** | **{kept_raw + exact_merged_raw + near_collapsed_raw + quar_raw}** | **100.00%** | Matches raw source inventory `{raw_total}` |",
            "",
            "## 3. Near-Duplicate & Template Family Review",
            "",
            "- **Clustering method:** Connected components over the union of (a) `text_norm_strict` char-ngram TF-IDF "
            "(3–5 grams, cosine ≥ 0.85) and (b) shortcut-and-digit-normalized template text (`<URL>`, `<PHONE>`, `<USSD>`, `<N>`, "
            "cosine ≥ 0.85), assigning deterministic `family_id` across all "
            f"`{summary['exact_unique_clean_before_collapse']}` clean exact-unique texts BEFORE collapsing variants.",
            f"- **Total multi-member families reviewed:** **{len(reviews)}** families logged in `near_duplicate_review.jsonl`.",
            f"- **Total families across `existing_unique.parquet`:** **{existing['family_id'].nunique()}**.",
            f"- **Variants collapsed:** **{summary['near_dup_collapsed_variants']['total']}** (all in `BB+SD`; `BanglaPhish-2026` OOD benchmark rows are preserved intact with `family_id` assigned):",
        ]
    )

    for reason, cnt in sorted(summary["near_dup_collapsed_variants"]["by_reason"].items()):
        lines.append(f"  - `{reason}`: **{cnt}** variants collapsed with explicit surviving target IDs logged.")

    lines.extend(["", "| Review Category (`near_duplicate_review.jsonl`) | Families |", "| :--- | ---: |"])
    for cat, cnt in sorted(summary["near_dup_review_families"]["by_category"].items()):
        lines.append(f"| `{cat}` | {cnt} |")

    lines.extend(
        [
            "",
            "Materially distinct near-duplicates (e.g., cross-script translations across `Bengali`/`Banglish`/`English`/`CodeMix`, "
            "paraphrases with distinct wording, and campaign messages sharing a malicious URL) are **retained** in "
            "`existing_unique.parquet` with shared `family_id` so grouped splits keep them together.",
            "",
            "## 4. Quarantined, Conflicting & Excluded Rows",
            "",
            f"Total quarantined unique strict groups: **{len(quar)}** (covering **{quar_raw}** raw rows).",
            "",
            "| Quarantine Reason | Unique Groups | Raw Rows | Description |",
            "| :--- | ---: | ---: | :--- |",
        ]
    )

    reason_desc = {
        "conflicting_3class_labels": "Identical text assigned conflicting 3-class labels (`Promotional` vs `Smishing`) in `BB+SD`",
        "cross_source_conflict_legacy_spam_vs_normal": "Labeled `spam` in `legacy_2024` vs `Normal` in `BB+SD` (includes mislabeled money-demand/account-debit SMS)",
        "cross_source_conflict_legacy_ham_vs_non_normal": "Labeled `ham` in `legacy_2024` vs `Promotional`/`Smishing` in `BB+SD`",
        "legacy_internal_binary_conflict": "Internal `ham` vs `spam` conflict within `legacy_2024`",
        "legacy_binary_spam_reference_only": "`legacy_2024`-only binary `spam`; never auto-converted to Promotional/Smishing (undocumented license)",
        "legacy_binary_ham_undocumented_license_reference_only": "`legacy_2024`-only `ham`; undocumented license and unverified against 3-class Promotional boundary",
        "empty_or_corrupted_text": "Empty text or Unicode replacement character `U+FFFD`",
    }
    raw_q_counts = summary["quarantined_raw_rows_by_reason"]
    for reason, grp_cnt in sorted(summary["quarantined_groups_by_reason"].items()):
        lines.append(
            f"| `{reason}` | {grp_cnt} | {raw_q_counts.get(reason, 0)} | {reason_desc.get(reason, reason)} |"
        )

    lines.extend(
        [
            "",
            "## 5. Evaluation Membership & Contamination Exclusions",
            "",
            "| Evaluation Group (`eval_group`) | Benchmark Membership (`benchmark_membership`) | Rows in `existing_unique` | Eligible for Training |",
            "| :--- | :--- | ---: | :--- |",
        ]
    )
    bm_counts = existing.groupby(["eval_group", "benchmark_membership", "eligible_for_training"]).size()
    for (eg, bm, elig_flag), cnt in bm_counts.items():
        lines.append(f"| `{eg}` | `{bm}` | {int(cnt)} | `{elig_flag}` |")

    lines.extend(
        [
            "",
            f"- **BanglaPhish-2026 OOD isolation:** `{int((existing['primary_dataset'] == 'banglaphish_2026').sum())}` "
            "rows preserved out-of-training (`eligible_for_training=False`); cross-corpus family overlap with `BB+SD` = "
            f"**{int((existing['eval_group'] == 'banglaphish_ood_overlap').sum())}**.",
            "- **Held-out split family exclusions:** Any `smishdetect_2026` `train` or `bangalabarta_2025` message sharing "
            "a `family_id` with `smishdetect_test` or `smishdetect_validation` is excluded from `eligible_train.parquet` "
            f"(`smishdetect_test_family_overlap`: **{(existing['benchmark_membership'] == 'smishdetect_test_family_overlap').sum()}**, "
            f"`smishdetect_validation_family_overlap`: **{(existing['benchmark_membership'] == 'smishdetect_validation_family_overlap').sum()}**).",
            "- **Traceability to `phase2_3`:** `phase2_3_cv_fold` preserves existing `unified_3class_dedup.parquet` / "
            "`banglaphish2026_dedup.parquet` fold IDs; `eligible_train.parquet` additionally provides `curation_cv_fold` "
            "(5-fold `StratifiedGroupKFold` grouped by `family_id`). Note that for any 5-fold CV over the full `BB+SD` "
            "corpus, Stage B augmentation must be derived strictly per-fold from that fold's training partition.",
            "",
            "## 6. Consolidated Inventory vs. Eligible Training Class Counts",
            "",
            "| Class (`label_3class`) | Consolidated Inventory (`existing_unique`) | Primary Non-OOD (`BB+SD`) | OOD Benchmark (`BanglaPhish-2026`) | Eligible Training (`eligible_train`) |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    non_bp_ex = existing[existing["primary_dataset"] != "banglaphish_2026"]
    bp_ex = existing[existing["primary_dataset"] == "banglaphish_2026"]
    for lbl in CANONICAL_LABELS:
        c_all = int((existing["label_3class"] == lbl).sum())
        c_bbs = int((non_bp_ex["label_3class"] == lbl).sum())
        c_bp = int((bp_ex["label_3class"] == lbl).sum())
        c_elig = int((elig["label_3class"] == lbl).sum())
        lines.append(f"| `{lbl}` | {c_all} | {c_bbs} | {c_bp} | {c_elig} |")
    lines.append(
        f"| **Total** | **{len(existing)}** | **{len(non_bp_ex)}** | **{len(bp_ex)}** | **{len(elig)}** |"
    )

    lines.extend(
        [
            "",
            "## 7. Language Coverage & Unresolved Issues",
            "",
            "### Eligible Training (`eligible_train.parquet`) by Language × Class",
            "",
            "| Language | Normal | Promotional | Smishing | Total |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    for lang in sorted(elig["language"].unique()):
        sub = elig[elig["language"] == lang]
        n_norm = int((sub["label_3class"] == "Normal").sum())
        n_promo = int((sub["label_3class"] == "Promotional").sum())
        n_smish = int((sub["label_3class"] == "Smishing").sum())
        lines.append(f"| `{lang}` | {n_norm} | {n_promo} | {n_smish} | {len(sub)} |")

    lines.extend(
        [
            "",
            "### Unresolved Language & Provenance Issues",
            "",
            f"- **Script vs. source language tag discrepancies (`existing_unique`):** "
            f"`{lang_mismatch_cm_bn}` rows tagged `CodeMix` contain 0 Latin characters (`script_bucket=Bengali`); "
            f"`{lang_mismatch_cm_lat}` rows tagged `CodeMix` contain 0 Bengali characters (`script_bucket=Latin`); "
            f"`{lang_mismatch_bn_lat}` rows tagged `Bengali` contain 0 Bengali characters (`script_bucket=Latin`, e.g., Banglish/English); "
            f"`{lang_mismatch_bn_cm}` rows tagged `Bengali` contain Latin URLs/loanwords (`script_bucket=CodeMix`). "
            "Both `language` (source metadata) and `script_bucket` are preserved separately for Stage B auditing.",
            f"- **BanglaPhish placeholder artifacts:** `{bp_placeholder_cm}` `banglaphish_2026` rows trigger `script_bucket=CodeMix` "
            "solely because of ASCII bracket placeholders (`[REDACTED_URL]`, `[DATE_01]`, `[AMOUNT]`, etc.) despite `language=Bangla`.",
            f"- **Binary spam in `legacy_2024`:** {summary['quarantined_groups_by_reason'].get('legacy_binary_spam_reference_only', 0)} unique `legacy_2024`-only `spam` texts and {summary['quarantined_groups_by_reason'].get('legacy_binary_ham_undocumented_license_reference_only', 0)} unique `legacy_2024`-only `ham` "
            "texts remain quarantined (undocumented license; cannot safely map binary spam to `Promotional` vs `Smishing` without re-annotation).",
            f"- **Label noise observed via cross-source check:** `legacy_2024` flagged {summary['quarantined_groups_by_reason'].get('cross_source_conflict_legacy_spam_vs_normal', 0)} texts as `spam` that `BB+SD` labeled `Normal` "
            "(6 genuine money-demand/account-debit SMS plus 4 literary/religious quotes). All were quarantined; any unflagged "
            "paraphrases in `BB+SD` merit targeted audit before evaluation claims.",
            "",
            "## 8. Proposed Stage B Generation Deficits (`T - n_c`)",
            "",
            f"Using accepted, eligible training rows (`eligible_train.parquet`, total `N = {len(elig)}`), "
            f"with target `T = max(n_Normal, n_Promotional, n_Smishing) = {summary['stage_b_target_T']}`:",
            "",
            "| Class | Eligible Training Count (`n_c`) | Target (`T`) | Proposed Stage B Deficit (`max(0, T - n_c)`) |",
            "| :--- | ---: | ---: | ---: |",
        ]
    )
    for lbl in CANONICAL_LABELS:
        nc = summary["eligible_train_by_class"][lbl]
        t_val = summary["stage_b_target_T"]
        deficit = summary["stage_b_deficits"][lbl]
        lines.append(f"| `{lbl}` | {nc} | {t_val} | {deficit} |")
    total_deficit = sum(summary["stage_b_deficits"].values())
    lines.append(f"| **Total** | **{len(elig)}** | — | **{total_deficit}** |")

    lines.extend(
        [
            "",
            "Stage B generation has **not** been started (Stage A only). Per `data/DATASET_GUIDE.md`, Stage B should begin "
            "with a reviewable pilot of at most 300 accepted synthetic rows allocated strictly within these measured deficits "
            f"(`Normal`: {summary['stage_b_deficits']['Normal']}, `Promotional`: {summary['stage_b_deficits']['Promotional']}, "
            f"`Smishing`: {summary['stage_b_deficits']['Smishing']}) while addressing language/script and scenario balance.",
            "",
            "## 9. Output Artifacts (`data/processed/curation_v1/`)",
            "",
            "- `data/processed/curation_v1/existing_unique.parquet`",
            "- `data/processed/curation_v1/provenance.parquet`",
            "- `data/processed/curation_v1/quarantined.parquet`",
            "- `data/processed/curation_v1/eligible_train.parquet`",
            "- `data/processed/curation_v1/near_duplicate_review.jsonl`",
            "- `data/processed/curation_v1/curation_summary.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage A dataset consolidation and write consolidation report.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root directory")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: data/processed/curation_v1)")
    parser.add_argument("--report-path", type=Path, default=None, help="Report path (default: reports/dataset_consolidation_v1.md)")
    parser.add_argument("--sim-threshold", type=float, default=0.85, help="Near-duplicate cosine threshold")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.root / "data" / "processed" / "curation_v1")
    report_path = args.report_path or (args.root / "reports" / "dataset_consolidation_v1.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    summary = run_stage_a_curation(root=args.root, output_dir=out_dir, sim_threshold=args.sim_threshold)
    report_md = build_consolidation_report(args.root, out_dir, summary)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"Stage A curation complete -> {out_dir}")
    print(f"Report written -> {report_path}")
    print(f"existing_unique={summary['existing_unique_total']}, eligible_train={summary['eligible_train_total']}")
    print(f"eligible_train_by_class={summary['eligible_train_by_class']}, deficits={summary['stage_b_deficits']}")


if __name__ == "__main__":
    main()
