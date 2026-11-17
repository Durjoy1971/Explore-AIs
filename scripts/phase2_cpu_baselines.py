"""Phase 2 & Phase 3 (CPU): Build leakage-free dataset, run TF-IDF & Surface baselines, and quantify shortcut drop."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.metrics import evaluate_predictions, mcnemar_pvalue
from bangla_smish.schema import CANONICAL_LABELS, load_bangalabarta_2025, load_banglaphish_2026, load_smishdetect_2026
from bangla_smish.splits import build_dedup_grouped_cv, build_length_matched_subset, perturb_zwj_homoglyph

LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(CANONICAL_LABELS)}


def make_tfidf_lr() -> Pipeline:
    feats = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
        ]
    )
    return Pipeline(
        [
            ("tfidf", feats),
            ("clf", LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced")),
        ]
    )


def make_surface_lr() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def eval_cv(df: pd.DataFrame, text_col: str, make_model_fn, label_col: str = "label_3class") -> tuple[dict, np.ndarray, np.ndarray]:
    y = df[label_col].map(LABEL_TO_IDX).to_numpy(dtype=int)
    oof_probs = np.zeros((len(df), len(CANONICAL_LABELS)), dtype=float)
    folds = df["fold"].to_numpy(dtype=int)
    texts = df[text_col].fillna("").astype(str).tolist()

    for f in sorted(np.unique(folds)):
        tr_idx = np.where(folds != f)[0]
        te_idx = np.where(folds == f)[0]
        clf = make_model_fn()
        clf.fit([texts[i] for i in tr_idx], y[tr_idx])
        oof_probs[te_idx] = clf.predict_proba([texts[i] for i in te_idx])

    metrics = evaluate_predictions(y, oof_probs, label_names=CANONICAL_LABELS)
    return metrics, y, oof_probs


def eval_surface_cv(df: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray]:
    surf_cols = ["char_len", "word_len", "has_bn", "has_latin", "has_url", "has_phone", "has_ussd", "has_redacted_tag"]
    X = df[surf_cols].to_numpy(dtype=float)
    y = df["label_3class"].map(LABEL_TO_IDX).to_numpy(dtype=int)
    folds = df["fold"].to_numpy(dtype=int)
    oof_probs = np.zeros((len(df), len(CANONICAL_LABELS)), dtype=float)

    for f in sorted(np.unique(folds)):
        tr_idx = np.where(folds != f)[0]
        te_idx = np.where(folds == f)[0]
        clf = make_surface_lr()
        clf.fit(X[tr_idx], y[tr_idx])
        oof_probs[te_idx] = clf.predict_proba(X[te_idx])

    return evaluate_predictions(y, oof_probs, label_names=CANONICAL_LABELS), y, oof_probs


def eval_lodo(df: pd.DataFrame, train_datasets: list[str], test_dataset: str, text_col: str = "text") -> dict:
    tr = df[df["dataset"].isin(train_datasets)]
    te = df[df["dataset"] == test_dataset]
    y_tr = tr["label_3class"].map(LABEL_TO_IDX).to_numpy(dtype=int)
    y_te = te["label_3class"].map(LABEL_TO_IDX).to_numpy(dtype=int)
    clf = make_tfidf_lr()
    clf.fit(tr[text_col].fillna("").astype(str).tolist(), y_tr)
    probs = clf.predict_proba(te[text_col].fillna("").astype(str).tolist())
    return evaluate_predictions(y_te, probs, label_names=CANONICAL_LABELS)


def main() -> None:
    proc_dir = ROOT / "data" / "processed"
    reports_dir = ROOT / "reports"
    proc_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load 3-class native datasets for unified benchmark (BangalaBarta + SmishDetect)
    bb = load_bangalabarta_2025(ROOT)
    sd = load_smishdetect_2026(ROOT)
    bp = load_banglaphish_2026(ROOT)

    merged_3class = pd.concat([bb, sd], ignore_index=True)
    dedup_df = build_dedup_grouped_cv(merged_3class, n_splits=5, sim_threshold=0.85)
    dedup_df["text_adv"] = dedup_df["text"].map(lambda t: perturb_zwj_homoglyph(t, seed=42))
    dedup_path = proc_dir / "unified_3class_dedup.parquet"
    dedup_df.to_parquet(dedup_path, index=False)

    # Also build dedup BanglaPhish-2026 for cross-corpus OOD evaluation
    bp_dedup = build_dedup_grouped_cv(bp, n_splits=5, sim_threshold=0.85)
    bp_dedup.to_parquet(proc_dir / "banglaphish2026_dedup.parquet", index=False)

    results = {}

    # 1. Surface-only baseline (measures pure shortcut power after dedup)
    m_surf, y_true, p_surf = eval_surface_cv(dedup_df)
    results["surface_only_dedup_cv"] = m_surf

    # 2. Raw-text TF-IDF + LR on grouped dedup CV
    m_raw_cv, _, p_raw = eval_cv(dedup_df, "text", make_tfidf_lr)
    results["tfidf_lr_dedup_cv_raw_text"] = m_raw_cv

    # 3. Shortcut-masked (URL/Phone/USSD replaced) TF-IDF + LR
    m_mask_cv, _, p_mask = eval_cv(dedup_df, "text_masked", make_tfidf_lr)
    results["tfidf_lr_dedup_cv_masked_text"] = m_mask_cv

    # 4. Evaluate raw-trained TF-IDF on Adversarial ZWJ/Homoglyph & Length-matched subsets
    y = dedup_df["label_3class"].map(LABEL_TO_IDX).to_numpy(dtype=int)
    folds = dedup_df["fold"].to_numpy(dtype=int)
    oof_adv_probs = np.zeros_like(p_raw)
    texts_raw = dedup_df["text"].fillna("").astype(str).tolist()
    texts_adv = dedup_df["text_adv"].fillna("").astype(str).tolist()
    for f in sorted(np.unique(folds)):
        tr_idx = np.where(folds != f)[0]
        te_idx = np.where(folds == f)[0]
        clf = make_tfidf_lr()
        clf.fit([texts_raw[i] for i in tr_idx], y[tr_idx])
        oof_adv_probs[te_idx] = clf.predict_proba([texts_adv[i] for i in te_idx])

    results["tfidf_lr_eval_adversarial_zwj_homoglyph"] = evaluate_predictions(y, oof_adv_probs, label_names=CANONICAL_LABELS)
    results["mcnemar_raw_vs_masked"] = mcnemar_pvalue(y_true, p_raw.argmax(1), p_mask.argmax(1))
    results["mcnemar_raw_vs_adversarial"] = mcnemar_pvalue(y_true, p_raw.argmax(1), oof_adv_probs.argmax(1))

    # Length-matched subset evaluation (40..120 chars)
    lm_mask = (dedup_df["char_len"] >= 40) & (dedup_df["char_len"] <= 120)
    results["tfidf_lr_eval_length_matched_40_120"] = evaluate_predictions(
        y[lm_mask.values], p_raw[lm_mask.values], label_names=CANONICAL_LABELS
    )

    # Banglish + CodeMix subset evaluation
    bm_mask = dedup_df["source_lang_tag"].isin(["Banglish", "CodeMix"])
    results["tfidf_lr_eval_banglish_codemix"] = evaluate_predictions(
        y[bm_mask.values], p_raw[bm_mask.values], label_names=CANONICAL_LABELS
    )

    # 5. Cross-corpus LODO: Train BangalaBarta -> Test non-overlapping SmishDetect (and vice versa)
    results["lodo_train_bangalabarta_test_smishdetect_raw"] = eval_lodo(dedup_df, ["bangalabarta_2025"], "smishdetect_2026", "text")
    results["lodo_train_smishdetect_test_bangalabarta_raw"] = eval_lodo(dedup_df, ["smishdetect_2026"], "bangalabarta_2025", "text")
    results["lodo_train_bangalabarta_test_smishdetect_masked"] = eval_lodo(dedup_df, ["bangalabarta_2025"], "smishdetect_2026", "text_masked")
    results["lodo_train_smishdetect_test_bangalabarta_masked"] = eval_lodo(dedup_df, ["smishdetect_2026"], "bangalabarta_2025", "text_masked")

    # 5b. Cross-corpus OOD: Train Unified 3-class -> Test BanglaPhish-2026 SMS-only subset
    bp_sms = bp_dedup[bp_dedup["source_lang_tag"].str.endswith("|SMS")].reset_index(drop=True)
    clf_ood = make_tfidf_lr()
    clf_ood.fit(dedup_df["text"].tolist(), y)
    bp_probs_raw = clf_ood.predict_proba(bp_sms["text"].tolist())
    results["ood_train_unified_test_banglaphish_sms_raw"] = evaluate_predictions(
        bp_sms["label_3class"].map(LABEL_TO_IDX).to_numpy(dtype=int), bp_probs_raw, label_names=CANONICAL_LABELS
    )
    clf_ood_m = make_tfidf_lr()
    clf_ood_m.fit(dedup_df["text_masked"].tolist(), y)
    bp_probs_m = clf_ood_m.predict_proba(bp_sms["text_masked"].tolist())
    results["ood_train_unified_test_banglaphish_sms_masked"] = evaluate_predictions(
        bp_sms["label_3class"].map(LABEL_TO_IDX).to_numpy(dtype=int), bp_probs_m, label_names=CANONICAL_LABELS
    )

    # 6. Predefined split comparison on SmishDetect-2026 (leaky original split vs dedup grouped CV)
    sd_raw = load_smishdetect_2026(ROOT)
    sd_tr = sd_raw[sd_raw["orig_split"] == "train"]
    sd_te = sd_raw[sd_raw["orig_split"] == "test"]
    clf_sd = make_tfidf_lr()
    clf_sd.fit(sd_tr["text"].tolist(), sd_tr["label_3class"].map(LABEL_TO_IDX).to_numpy(dtype=int))
    sd_probs = clf_sd.predict_proba(sd_te["text"].tolist())
    results["smishdetect_original_test_split"] = evaluate_predictions(
        sd_te["label_3class"].map(LABEL_TO_IDX).to_numpy(dtype=int), sd_probs, label_names=CANONICAL_LABELS
    )

    json_out = reports_dir / "phase2_3_cpu_baselines.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lines = [
        "# Phase 2 & Phase 3 (CPU Baselines) — Fair Evaluation & Shortcut-Stress Results",
        "",
        f"Unified deduplicated 3-class corpus (`BangalaBarta` + `SmishDetect-2026`): **{len(dedup_df)}** unique non-conflicting messages "
        f"(clusters: {dedup_df['dup_cluster_id'].nunique()}).",
        "",
        "| Experiment / Stress Split | N | Accuracy (95% CI) | Macro-F1 (95% CI) | Smishing Recall | Normal→Smish FPR | Recall @ ≤1% FPR | ECE |",
        "| :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for key, m in results.items():
        if not isinstance(m, dict) or "macro_f1" not in m:
            continue
        acc_str = f"{m['accuracy']}% [{m['accuracy_ci95'][0]}, {m['accuracy_ci95'][1]}]"
        f1_str = f"{m['macro_f1']}% [{m['macro_f1_ci95'][0]}, {m['macro_f1_ci95'][1]}]"
        lines.append(
            f"| `{key}` | {m['n_samples']} | {acc_str} | {f1_str} | {m['smishing_recall_argmax']}% | "
            f"{m['normal_as_smishing_fpr_argmax']}% | {m['smishing_recall_at_1pct_fpr']}% | {m['ece']} |"
        )
    lines.extend([
        "",
        "## McNemar Significance Tests (Unified Dedup CV)",
        f"- **Raw text vs. Shortcut-masked (`<URL>/<PHONE>/<USSD>`)**: p = `{results['mcnemar_raw_vs_masked']['p_value']}` "
        f"(raw-only correct: {results['mcnemar_raw_vs_masked']['n_a_only']}, masked-only correct: {results['mcnemar_raw_vs_masked']['n_b_only']})",
        f"- **Raw text vs. Adversarial ZWJ/Homoglyph**: p = `{results['mcnemar_raw_vs_adversarial']['p_value']}` "
        f"(raw-only correct: {results['mcnemar_raw_vs_adversarial']['n_a_only']}, adv-only correct: {results['mcnemar_raw_vs_adversarial']['n_b_only']})",
    ])
    md_out = reports_dir / "phase2_3_cpu_baselines.md"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved {dedup_path}, {md_out}, and {json_out}")


if __name__ == "__main__":
    main()
