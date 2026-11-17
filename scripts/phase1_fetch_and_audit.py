"""Phase 1: Verify licenses, audit duplicates, split leakage, shortcuts, and cross-corpus overlap."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.audit import audit_single_dataset, compute_cross_dataset_overlap
from bangla_smish.schema import load_all_datasets


def main() -> None:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_all_datasets(ROOT)
    audits = [audit_single_dataset(name, df) for name, df in datasets.items()]
    overlaps = compute_cross_dataset_overlap(datasets)

    payload = {"audits": audits, "cross_dataset_overlaps": overlaps}
    json_path = reports_dir / "phase1_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    lines = [
        "# Phase 1 — Dataset Verification & Shortcut Audit Report",
        "",
        "## 1. Corpus Summary, Licenses, Duplicates & Shortcut Power",
        "",
        "| Dataset | License | Provenance | Rows | Unique | Exact Dup % | Near-Dup Reduction % (cos≥0.85) | Label Conflicts | Surface-Only 5-fold Acc | Surface-Only Macro-F1 | Length-Only Risk AUC |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for a in audits:
        lines.append(
            f"| `{a['dataset']}` | {a['license']} | {a['is_synthetic']} | {a['total_rows']} | "
            f"{a['unique_norm_rows']} | {a['exact_dup_pct']}% | {a['near_dup_within_unique_pct']}% | "
            f"{a['conflicting_label_texts']} | {a['surface_only_5fold_acc_pct']}% | "
            f"{a['surface_only_5fold_macro_f1_pct']}% | {a['length_only_risk_roc_auc']} |"
        )

    lines.extend(["", "## 2. Predefined Split Leakage (Train → Val/Test Exact Normalized Overlap)", ""])
    for a in audits:
        if a["split_leakage"]:
            sl = a["split_leakage"]
            lines.append(
                f"- **`{a['dataset']}`**: train→val exact overlap = **{sl.get('train_to_validation_exact_rows', 0)}** "
                f"({sl.get('train_to_validation_pct', 0)}%), train→test exact overlap = **{sl.get('train_to_test_exact_rows', 0)}** "
                f"({sl.get('train_to_test_pct', 0)}%)"
            )
        else:
            lines.append(f"- **`{a['dataset']}`**: No predefined train/val/test split shipped.")

    lines.extend(["", "## 3. Cross-Dataset Exact Normalized-Text Overlap", ""])
    lines.append("| Dataset A | Dataset B | Shared Unique Texts | % of A | % of B |")
    lines.append("| :--- | :--- | ---: | ---: | ---: |")
    for ov in overlaps:
        lines.append(
            f"| `{ov['dataset_a']}` | `{ov['dataset_b']}` | {ov['shared_unique_texts']} | {ov['pct_of_a']}% | {ov['pct_of_b']}% |"
        )

    lines.extend(["", "## 4. Per-Class Surface & Shortcut Confounders", ""])
    for a in audits:
        lines.append(f"### `{a['dataset']}`")
        lines.append("| Class | Count | Unique | Median Chars | Mean Chars | % Has Latin | % Has URL | % Has Phone | % Has `[REDACTED_*]` |")
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for cls, st in a["class_surface_stats"].items():
            lines.append(
                f"| {cls} | {st['count']} | {st['unique_norm']} | {st['median_char_len']} | {st['mean_char_len']} | "
                f"{st['pct_has_latin']}% | {st['pct_has_url']}% | {st['pct_has_phone']}% | {st['pct_has_redacted_tag']}% |"
            )
        lines.append("")

    md_path = reports_dir / "phase1_dataset_audit.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {md_path} and {json_path}")


if __name__ == "__main__":
    main()
