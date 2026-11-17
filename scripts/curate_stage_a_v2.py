"""Stage A-followup CLI: build `curation_v2` and write audit & Stage B pilot-plan reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.curation_v2 import ALLOWED_LANGUAGES, run_stage_a_curation_v2
from bangla_smish.schema import CANONICAL_LABELS


def build_focused_audit_report(v1_dir: Path, v2_dir: Path, summary: dict[str, Any]) -> str:
    """Generate reports/stage_a_focused_audit_v2.md strictly from curation_v1 and curation_v2 files."""
    ex_v1 = pd.read_parquet(v1_dir / "existing_unique.parquet")
    el_v1 = pd.read_parquet(v1_dir / "eligible_train.parquet")
    ex_v2 = pd.read_parquet(v2_dir / "existing_unique.parquet")
    el_v2 = pd.read_parquet(v2_dir / "eligible_train.parquet")
    qu_v2 = pd.read_parquet(v2_dir / "quarantined.parquet")
    pr_v2 = pd.read_parquet(v2_dir / "provenance.parquet")
    ad_v2 = pd.read_parquet(v2_dir / "audit_decisions.parquet")

    cov = summary["coverage_profile"]
    ac = summary["audit_counts"]

    # Label transitions among retained/quarantined
    lbl_reclass = ad_v2[ad_v2["actions"].str.contains("reclassify_label", na=False)]
    lbl_quar = ad_v2[ad_v2["actions"].str.contains("quarantine_ambiguous", na=False)]
    lang_reclass = ad_v2[ad_v2["actions"].str.contains("reclassify_language", na=False)]

    lines: list[str] = [
        "# Stage A Focused Audit Report (`curation_v2`)",
        "",
        "Generated reproducibly by `scripts/curate_stage_a_v2.py` (`src/bangla_smish/curation_v2.py`) "
        "on top of read-only `data/processed/curation_v1/` (zero synthetic rows generated; zero held-out/OOD rows modified).",
        "",
        "## 1. Strict Audit Scope & Isolation Guarantees",
        "",
        f"- **Audited scope:** `curation_v1/eligible_train.parquet` (`train_pool` only, **{len(el_v1)}** rows).",
        "- **Held-out & OOD preservation:** `banglaphish_ood` (**"
        f"{int((ex_v2['eval_group'] == 'banglaphish_ood').sum())}**), `smishdetect_test` (**"
        f"{int((ex_v2['eval_group'] == 'smishdetect_test').sum())}**), and `smishdetect_validation` (**"
        f"{int((ex_v2['eval_group'] == 'smishdetect_validation').sum())}**) remain 100% identical to `curation_v1` in texts, labels, languages, and counts.",
        "- **`curation_v1` & original corpora untouched:** All outputs are written exclusively to `data/processed/curation_v2/`.",
        "",
        "## 2. Publication-Grade Decision Boundaries & Language Rules",
        "",
        "### Label Boundaries (`Normal` / `Promotional` / `Smishing` / `Quarantine`)",
        "1. **`Normal` vs. `Smishing`:** Transactional/service notices (MyGP PIN/OTP codes, telecom call/balance receipts, "
        "scheduled maintenance notices, BTRC handset auto-registration, verified `.edu.bd` university tuition notices) are "
        "`Normal` even when urgent or containing URLs/OTPs. Fake emergency/family-danger money requests and credential/fee/prize "
        "extraction lures are `Smishing`.",
        "2. **`Normal` vs. `Promotional`:** Commercial discounts, data/voice bundles, paid service offers, identifiable educational "
        "admissions (e.g., Comilla City College EIIN:134648), and event/program/contest drives inviting active participation "
        "(`Mujib` biopic, Museum Day, WTISD contest) are `Promotional`. Pure institutional holiday/anniversary greetings, national "
        "tributes, and non-commercial government/regulatory PSAs (Election Commission, BTRC consultation, e-Passport/BRTA, health/safety) are `Normal`.",
        "3. **Mandatory Quarantine (`audit_v2_*`):** Messages whose intent depends on unavailable sender identity or context "
        "— including unattributed account deductions (`1500/- taka deducted...`), bare mobile numbers demanding bKash/Rocket/money "
        "without personal context, unverified utility disconnect/car servicing/delivery callbacks, outsourced interview/exam calls "
        "without verification or extraction cues, unbranded discount mobile callbacks, garbled recruitment texts, and unverified "
        "advance-fee/recruitment URLs — are quarantined rather than guessed.",
        "",
        "### Linguistic Classification Rules (`Bengali` / `Banglish` / `English` / `CodeMix`)",
        "1. **URL/Placeholder/Loanword invariance:** URLs, USSD codes, numbers, brand names (`bKash`, `Nagad`, `GP`, `Daraz`), "
        "and isolated borrowed telecom/banking loanwords (`OTP`, `PIN`, `MB`, `GB`, `TK`, `taka`, `offer`, `bonus`, `cashback`, "
        "`recharge`, `account`, `bank`, `call`, `dial`, `verify`) in a Bengali-matrix sentence do **not** invalidate `Bengali` or `Banglish`.",
        "2. **`Bengali`:** Bengali matrix syntax in Bengali Unicode script (`[\\u0980-\\u09FF]`), allowing isolated loanwords/URLs.",
        "3. **`Banglish`:** Bengali matrix syntax in Roman/Latin script (`[A-Za-z]`), allowing isolated loanwords/URLs.",
        "4. **`English`:** English matrix syntax in Latin script (including Bangladeshi English terms like `taka`, `TK`, `lakh`, `Eid`, `Puja`).",
        "5. **`CodeMix`:** Productive intra-/inter-sentential switching across Bengali and English/Banglish clauses (in either "
        "dual Bengali+Latin script or Latin-script English+Banglish clause switching). Note: Bengali sentences in `SmishDetect` "
        "carrying substituted Latin words (`click`, `verify`, `number`, `day`, `Hello`) retain `CodeMix` for parity with the unmodified held-out `SmishDetect` test/validation schema.",
        "",
        "## 3. Summary of Applied Audit Decisions (`audit_decisions.parquet`)",
        "",
        f"- **Total decision rows logged:** **{ac['total_audit_decision_rows']}**",
        f"- **Label reclassifications (retained in `eligible_train`):** **{ac['label_reclassifications']}**",
        f"- **Newly quarantined ambiguous/unverified groups:** **{ac['newly_quarantined_groups']}**",
        f"- **Language reclassifications total:** **{ac['language_reclassifications_total']}** "
        f"(**{ac['language_reclassifications_retained_in_eligible']}** retained in `eligible_train`; "
        f"{ac['language_reclassifications_total'] - ac['language_reclassifications_retained_in_eligible']} also quarantined for missing sender context)",
        f"- **Boundary exemplars kept with documented note:** **{ac['kept_with_note_exemplars']}**",
        "",
        "### Label Reclassifications (`eligible_train`)",
        "",
        "| Transition | Category | Count |",
        "| :--- | :--- | ---: |",
    ]
    for (orig_l, new_l, cat), cnt in (
        lbl_reclass.groupby(["orig_label_3class", "audited_label_3class", "categories"]).size().items()
    ):
        lines.append(f"| `{orig_l} -> {new_l}` | `{cat}` | {int(cnt)} |")

    lines.extend(["", "### Newly Quarantined Categories (`quarantined.parquet`)", "", "| Original Class | Category | Groups Quarantined |", "| :--- | :--- | ---: |"])
    for (orig_l, cat), cnt in lbl_quar.groupby(["orig_label_3class", "categories"]).size().items():
        lines.append(f"| `{orig_l}` | `{cat}` | {int(cnt)} |")

    lines.extend(["", "### Language Reclassifications", "", "| Transition | Category | Count |", "| :--- | :--- | ---: |"])
    for (orig_lg, new_lg, cat), cnt in (
        lang_reclass.groupby(["orig_language", "audited_language", "categories"]).size().items()
    ):
        lines.append(f"| `{orig_lg} -> {new_lg}` | `{cat}` | {int(cnt)} |")

    lines.extend(
        [
            "",
            "## 4. `curation_v1` vs. `curation_v2` Counts & Reconciliation",
            "",
            "| Dataset View / Class | `curation_v1` | `curation_v2` | Delta |",
            "| :--- | ---: | ---: | ---: |",
            f"| `existing_unique.parquet` (total) | {len(ex_v1)} | {len(ex_v2)} | {len(ex_v2) - len(ex_v1)} |",
            f"| `quarantined.parquet` (unique groups) | {len(qu_v2) - ac['newly_quarantined_groups']} | {len(qu_v2)} | +{ac['newly_quarantined_groups']} |",
            f"| `provenance.parquet` (raw rows) | {len(pr_v2)} | {len(pr_v2)} | 0 |",
            f"| `eligible_train.parquet` (total) | {len(el_v1)} | {len(el_v2)} | {len(el_v2) - len(el_v1)} |",
        ]
    )
    for lbl in CANONICAL_LABELS:
        c1 = int((el_v1["label_3class"] == lbl).sum())
        c2 = int((el_v2["label_3class"] == lbl).sum())
        lines.append(f"| `eligible_train` — `{lbl}` | {c1} | {c2} | {c2 - c1:+d} |")

    lines.extend(
        [
            "",
            "### `curation_v2` Eligible Training (`N = " + str(len(el_v2)) + "`) by Class × Language",
            "",
            "| Class | Bengali | Banglish | English | CodeMix | Total |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lang_cls = cov["class_by_language"]
    for lbl in CANONICAL_LABELS:
        row_d = lang_cls.get(lbl, {})
        vals = [row_d.get(lg, 0) for lg in ALLOWED_LANGUAGES]
        lines.append(f"| `{lbl}` | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {sum(vals)} |")

    lines.extend(
        [
            "",
            "## 5. Recomputed Stage B Deficits (`T - n_c`)",
            "",
            f"From `curation_v2/eligible_train.parquet` (`N = {len(el_v2)}`), target `T = max(n_Normal, n_Promotional, n_Smishing) = {cov['stage_b_target_T']}`:",
            "",
            "| Class | `curation_v2` Eligible (`n_c`) | Target (`T`) | Stage B Deficit (`max(0, T - n_c)`) |",
            "| :--- | ---: | ---: | ---: |",
        ]
    )
    for lbl in CANONICAL_LABELS:
        nc = cov["eligible_train_by_class"][lbl]
        lines.append(f"| `{lbl}` | {nc} | {cov['stage_b_target_T']} | {cov['stage_b_deficits'][lbl]} |")
    lines.append(f"| **Total** | **{len(el_v2)}** | — | **{cov['stage_b_total_deficit']}** |")

    lines.extend(
        [
            "",
            "## 6. Unresolved Ambiguities & Remaining Audit Notes",
            "",
            "- **Civic participation campaigns vs. PSAs:** Public event/film/contest promotions (`Mujib` biopic, Museum Day, "
            "WTISD essay contest) are retained in `Promotional` under the active-participation rule, while purely informational "
            "PSAs are `Normal`. Downstream ablations can inspect both slices via `audit_decisions.parquet`.",
            "- **`CodeMix` loanword parity with held-out `SmishDetect`:** `SmishDetect` tags some Bengali messages containing "
            "isolated English substitutions (`click`, `verify`, `number`, `day`, `Hello`) as `CodeMix` across both train and "
            "held-out splits; these were preserved as `CodeMix` in `curation_v2` to avoid a train-vs-eval language-tag shift.",
            "- **Held-out splits unedited:** Held-out `smishdetect_validation` (`708`) and `smishdetect_test` (`1,698`) were "
            "not audited or altered to prevent leakage; natural-prevalence evaluation noise remains a documented limitation.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_stage_b_plan_report(v2_dir: Path, summary: dict[str, Any]) -> str:
    """Generate reports/stage_b_coverage_and_pilot_plan.md strictly from curation_v2/eligible_train.parquet."""
    cov = summary["coverage_profile"]
    pilot = summary["pilot_plan"]

    lines: list[str] = [
        "# Stage B Coverage Matrix & Pilot Generation Plan (From `curation_v2`)",
        "",
        "Derived strictly from `data/processed/curation_v2/eligible_train.parquet` "
        f"(`N = {cov['eligible_train_total']}`, `{cov['eligible_train_families']}` families). "
        "Zero held-out evaluation or `BanglaPhish-2026` OOD messages were inspected or used. **No synthetic generation has been run.**",
        "",
        "## 1. Recomputed Class Quotas (`T - n_c`)",
        "",
        f"Target `T = max(n_Normal, n_Promotional, n_Smishing) = {cov['stage_b_target_T']}`:",
        "",
        "| Class | Accepted Eligible (`n_c`) | Target (`T`) | Required Accepted New Rows (`max(0, T - n_c)`) |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for lbl in CANONICAL_LABELS:
        lines.append(
            f"| `{lbl}` | {cov['eligible_train_by_class'][lbl]} | {cov['stage_b_target_T']} | {cov['stage_b_deficits'][lbl]} |"
        )
    lines.append(f"| **Total** | **{cov['eligible_train_total']}** | — | **{cov['stage_b_total_deficit']}** |")

    lines.extend(
        [
            "",
            "## 2. Measured `curation_v2` Training Coverage Matrix",
            "",
            "### A. Class × Language (`eligible_train.parquet`)",
            "",
            "| Class | Bengali | Banglish | English | CodeMix | Total |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for lbl in CANONICAL_LABELS:
        d = cov["class_by_language"].get(lbl, {})
        vals = [d.get(lg, 0) for lg in ALLOWED_LANGUAGES]
        lines.append(f"| `{lbl}` | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {sum(vals)} |")

    lines.extend(
        [
            "",
            "### B. Class × Character-Length Band",
            "",
            "| Class | `short(<50)` | `medium(50-120)` | `long(>120)` |",
            "| :--- | ---: | ---: | ---: |",
        ]
    )
    for lbl in CANONICAL_LABELS:
        d = cov["class_by_length_band"].get(lbl, {})
        lines.append(f"| `{lbl}` | {d.get('short(<50)', 0)} | {d.get('medium(50-120)', 0)} | {d.get('long(>120)', 0)} |")

    lines.extend(
        [
            "",
            "### C. Class × Domain / Scenario Bucket (Keyword-Inferred)",
            "",
            "| Domain / Scenario | Normal | Promotional | Smishing |",
            "| :--- | ---: | ---: | ---: |",
        ]
    )
    all_domains = sorted(
        {dom for lbl in CANONICAL_LABELS for dom in cov["class_by_domain_bucket"].get(lbl, {})}
    )
    for dom in all_domains:
        n_v = cov["class_by_domain_bucket"].get("Normal", {}).get(dom, 0)
        p_v = cov["class_by_domain_bucket"].get("Promotional", {}).get(dom, 0)
        s_v = cov["class_by_domain_bucket"].get("Smishing", {}).get(dom, 0)
        lines.append(f"| `{dom}` | {n_v} | {p_v} | {s_v} |")

    lines.extend(
        [
            "",
            "### D. Shortcut Surface Cues (URL, Contact, Urgency, OTP/PIN)",
            "",
            "| Class | Has URL (`True`) | Has Any Contact (`URL/Phone/USSD`) | Has Urgency Cue | Has OTP/PIN/Verify Cue |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    for lbl in CANONICAL_LABELS:
        u_t = cov["class_by_has_url"].get(lbl, {}).get("True", 0)
        c_t = cov["class_by_has_contact"].get(lbl, {}).get("True", 0)
        urg_t = cov["class_by_has_urgency"].get(lbl, {}).get("True", 0)
        otp_t = cov["class_by_has_otp_cue"].get(lbl, {}).get("True", 0)
        tot = cov["eligible_train_by_class"][lbl]
        lines.append(
            f"| `{lbl}` | {u_t} ({u_t/tot*100:.1f}%) | {c_t} ({c_t/tot*100:.1f}%) | "
            f"{urg_t} ({urg_t/tot*100:.1f}%) | {otp_t} ({otp_t/tot*100:.1f}%) |"
        )

    lines.extend(
        [
            "",
            "## 3. Key Coverage & Shortcut Gaps Identified",
            "",
            f"1. **Severe `Promotional` `Banglish` deficit:** Only **{cov['class_by_language']['Promotional']['Banglish']}** "
            f"`Promotional` `Banglish` rows remain after removing mislabeled English/CodeMix rows (vs. "
            f"**{cov['class_by_language']['Normal']['Banglish']}** Normal and **{cov['class_by_language']['Smishing']['Banglish']}** Smishing).",
            f"2. **`Normal` URL/urgency/OTP sparsity (anti-shortcut gap):** Only **{cov['class_by_has_url']['Normal'].get('True', 0)}** "
            f"`Normal` messages contain a URL, **{cov['class_by_has_urgency']['Normal'].get('True', 0)}** contain urgency cues, and "
            f"**{cov['class_by_has_otp_cue']['Normal'].get('True', 0)}** mention OTP/PIN/verify — causing models to equate URLs, urgency, and OTPs with Smishing.",
            "3. **Length-class skew:** `Normal` is heavily `short(<50)` "
            f"(**{cov['class_by_length_band']['Normal'].get('short(<50)', 0)}**), whereas `Smishing` (`{cov['class_by_length_band']['Smishing'].get('long(>120)', 0)}` long) "
            f"and `Promotional` (`{cov['class_by_length_band']['Promotional'].get('medium(50-120)', 0)}` medium / `{cov['class_by_length_band']['Promotional'].get('long(>120)', 0)}` long) skew longer.",
            "4. **Domain gaps:** `delivery_parcel` and legitimate `banking_mfs` transactional notifications are underrepresented in `Normal`, "
            "while `education`, `delivery_parcel`, and `employment` legitimate offers are sparse in `Promotional`.",
            "",
            f"## 4. Proposed Stage B Pilot Plan (≤ {pilot['max_pilot_rows']} Accepted Rows)",
            "",
            f"Per `data/DATASET_GUIDE.md`, Stage B starts with a **{pilot['pilot_total_rows']}-row** reviewable pilot "
            "(strictly within measured deficits, `Smishing = 0`):",
            "",
            "| Class | Total Deficit | Pilot Allocation | Bengali | Banglish | CodeMix | English | Primary Coverage Targets |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
            f"| `Normal` | {cov['stage_b_deficits']['Normal']} | **{pilot['pilot_by_class']['Normal']}** | "
            f"{pilot['normal_by_language']['Bengali']} | {pilot['normal_by_language']['Banglish']} | "
            f"{pilot['normal_by_language']['CodeMix']} | {pilot['normal_by_language']['English']} | "
            "Medium/long (`50-180` chars) legitimate bank/MFS/delivery/university notices with reserved `example.com`/`.edu.bd` links, urgency, and OTP warnings |",
            f"| `Promotional` | {cov['stage_b_deficits']['Promotional']} | **{pilot['pilot_by_class']['Promotional']}** | "
            f"{pilot['promotional_by_language']['Bengali']} | {pilot['promotional_by_language']['Banglish']} | "
            f"{pilot['promotional_by_language']['CodeMix']} | {pilot['promotional_by_language']['English']} | "
            "Short/medium/long non-coercive offers across retail/e-commerce, banking cashback, telecom, delivery, education, and employment; mix link-present & link-absent |",
            f"| `Smishing` | {cov['stage_b_deficits']['Smishing']} | **0** | 0 | 0 | 0 | 0 | No deficit (`n_Smishing = T`) |",
            f"| **Total** | **{cov['stage_b_total_deficit']}** | **{pilot['pilot_total_rows']}** | "
            f"**{pilot['normal_by_language']['Bengali'] + pilot['promotional_by_language']['Bengali']}** | "
            f"**{pilot['normal_by_language']['Banglish'] + pilot['promotional_by_language']['Banglish']}** | "
            f"**{pilot['normal_by_language']['CodeMix'] + pilot['promotional_by_language']['CodeMix']}** | "
            f"**{pilot['normal_by_language']['English'] + pilot['promotional_by_language']['English']}** | — |",
            "",
            "### Pilot Execution & Validation Protocol (Before Scaling to Remaining `Promotional` Deficit)",
            "1. **Scenario-only prompting:** Generate from `(class, language, domain, length_band, formality, link_presence, urgency_presence)` "
            "specifications using reserved domains (`example.com`, `example.org`) and fictional IDs/numbers (`+88017000000XX`); never paraphrase held-out rows.",
            "2. **Automated gates:** Enforce schema, `is_synthetic=True`, exact `text_norm_strict` uniqueness, char-ngram cosine `< 0.85` "
            "against all existing sources + isolated held-out/OOD contamination filter.",
            "3. **Independent review & surface diagnostics:** Human/independent label review of all 300 pilot rows plus surface-only & "
            "length-only AUC check before generating the remaining `Promotional` balance (`606 - 261 = 345` rows).",
            "4. **Fold-safe training:** When running 5-fold CV, include only synthetic rows derived from that fold's training partition.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build curation_v2 and Stage A/B audit & pilot-plan reports.")
    parser.add_argument("--v1-dir", type=Path, default=ROOT / "data" / "processed" / "curation_v1")
    parser.add_argument("--v2-dir", type=Path, default=ROOT / "data" / "processed" / "curation_v2")
    parser.add_argument("--audit-report", type=Path, default=ROOT / "reports" / "stage_a_focused_audit_v2.md")
    parser.add_argument("--plan-report", type=Path, default=ROOT / "reports" / "stage_b_coverage_and_pilot_plan.md")
    args = parser.parse_args()

    summary = run_stage_a_curation_v2(v1_dir=args.v1_dir, v2_dir=args.v2_dir)

    args.audit_report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.audit_report, "w", encoding="utf-8") as f:
        f.write(build_focused_audit_report(args.v1_dir, args.v2_dir, summary))

    with open(args.plan_report, "w", encoding="utf-8") as f:
        f.write(build_stage_b_plan_report(args.v2_dir, summary))

    print(f"curation_v2 written -> {args.v2_dir}")
    print(f"Reports written -> {args.audit_report}, {args.plan_report}")
    print(f"eligible_train_total={summary['coverage_profile']['eligible_train_total']}, "
          f"by_class={summary['coverage_profile']['eligible_train_by_class']}, "
          f"deficits={summary['coverage_profile']['stage_b_deficits']}")


if __name__ == "__main__":
    main()
