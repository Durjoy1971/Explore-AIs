"""Reusable Post-Stage B GPU training, checkpointing, safe resume, evaluation, and export runner."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .metrics import evaluate_predictions, mcnemar_pvalue
from .post_stage_b_data import (
    TEXT_VIEW_COLUMNS,
    build_experiment_fold_data,
    load_heldout_eval_sets,
    verify_manifest,
)
from .schema import CANONICAL_LABELS

LANGUAGES_CANONICAL: tuple[str, ...] = ("Bengali", "Banglish", "CodeMix", "English")


@dataclass
class PostStageBConfig:
    """Configuration for GPU smoke tests and Post-Stage B Experiments A, B, C, and D."""

    run_name: str
    run_mode: str  # "gpu_smoke_test" or "experiments_abcd"
    experiments: tuple[str, ...] = ("A", "B", "C", "D")
    models: tuple[str, ...] = (
        "csebuetnlp/banglabert",
        "google/muril-base-cased",
        "xlm-roberta-base",
        "distilbert-base-multilingual-cased",
    )
    text_views: tuple[str, ...] = ("raw", "canon_masked", "brand_masked", "brand_stripped")
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 3
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 128
    seed: int = 42
    fp16: str = "auto"  # "auto", "fp16", "bf16", "none"
    experiment_b_balance_mode: str = "downsample"  # "downsample" or "class_weight"
    run_heldout_eval: bool = True
    run_adversarial_eval: bool = True
    run_counterfactual_eval: bool = True
    keep_model_weights: str = "none"  # "none", "best_fold0", "all"
    smoke_max_train_samples: int | None = None
    smoke_max_val_samples: int | None = None
    smoke_max_eval_samples: int | None = None

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_config(path: str | Path) -> PostStageBConfig:
    """Load a PostStageBConfig from a JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    for tuple_field in ("experiments", "models", "text_views", "folds"):
        if tuple_field in raw and isinstance(raw[tuple_field], list):
            raw[tuple_field] = tuple(raw[tuple_field])
    return PostStageBConfig(**raw)


def save_config(cfg: PostStageBConfig, path: str | Path) -> None:
    """Persist a PostStageBConfig atomically as JSON."""
    atomic_write_json(Path(path), asdict(cfg))


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically using a temporary file and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically using UTF-8 encoding."""
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (when installed) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed

        set_seed(seed)
    except ImportError:
        pass


def collect_environment_info() -> dict[str, Any]:
    """Collect runtime, GPU, library, and Colab environment diagnostics without credentials."""
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "in_colab": "google.colab" in sys.modules or Path("/content").exists(),
        "cuda_available": False,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "gpu_count": 0,
        "packages": {},
    }
    for pkg in ("numpy", "pandas", "scipy", "sklearn", "pyarrow", "torch", "transformers", "datasets", "accelerate"):
        try:
            mod = __import__(pkg)
            info["packages"][pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            info["packages"][pkg] = "not_installed"

    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["gpu_count"] = int(torch.cuda.device_count())
            info["gpu_name"] = str(torch.cuda.get_device_name(0))
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_gb"] = round(float(props.total_memory) / (1024**3), 2)
    except ImportError:
        pass

    return info


def unit_id(experiment_id: str, model_name: str, text_view: str, fold: int) -> str:
    """Return a filesystem-safe deterministic identifier for one (experiment, model, view, fold) unit."""
    model_slug = model_name.replace("/", "__").replace(" ", "_")
    return f"exp_{experiment_id}__{model_slug}__{text_view}__fold_{fold}"


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"post_stage_b_{log_path.parent.parent.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _find_latest_trainer_checkpoint(ckpt_dir: Path) -> str | None:
    if not ckpt_dir.exists():
        return None
    ckpts = []
    for sub in ckpt_dir.iterdir():
        if sub.is_dir() and sub.name.startswith("checkpoint-"):
            try:
                step = int(sub.name.split("-", 1)[1])
                ckpts.append((step, sub))
            except ValueError:
                continue
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x[0])
    return str(ckpts[-1][1])


def _evaluate_by_language(df: pd.DataFrame, probs: np.ndarray, *, n_boot: int = 300) -> dict[str, Any]:
    y = df["label_idx"].to_numpy(dtype=int)
    per_lang: dict[str, Any] = {}
    for lang in LANGUAGES_CANONICAL:
        mask = (df["language"] == lang).to_numpy()
        if int(mask.sum()) >= 5 and len(np.unique(y[mask])) >= 2:
            per_lang[lang] = evaluate_predictions(y[mask], probs[mask], label_names=CANONICAL_LABELS, n_boot=n_boot)
        else:
            per_lang[lang] = {"n_samples": int(mask.sum()), "skipped": "insufficient_samples_or_classes"}
    return per_lang


def _resolve_precision(fp16_mode: str) -> tuple[bool, bool]:
    import torch

    if not torch.cuda.is_available():
        return False, False
    if fp16_mode == "none":
        return False, False
    if fp16_mode == "bf16":
        return False, bool(torch.cuda.is_bf16_supported())
    if fp16_mode == "fp16":
        return True, False
    # auto: prefer bf16 on Ampere+ if supported, else fp16 on T4/V100
    if torch.cuda.is_bf16_supported():
        return False, True
    return True, False


def hf_train_and_predict_fold(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    heldout_sets: dict[str, pd.DataFrame],
    model_name: str,
    text_view: str,
    checkpoint_dir: Path,
    cfg: PostStageBConfig,
    class_weights: list[float] | None,
    seed: int,
    resume_from_checkpoint: bool = True,
    best_model_export_dir: Path | None = None,
) -> dict[str, Any]:
    """Train one HuggingFace sequence classifier fold with epoch checkpoints and evaluate all splits."""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    text_col = TEXT_VIEW_COLUMNS[text_view]
    seed_everything(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _make_ds(texts: list[str], labels: list[int] | None = None) -> Dataset:
        payload: dict[str, Any] = {"text": texts}
        if labels is not None:
            payload["label"] = labels
        ds = Dataset.from_dict(payload)
        return ds.map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=cfg.max_length),
            batched=True,
            remove_columns=["text"],
        )

    ds_tr = _make_ds(train_df[text_col].fillna("").astype(str).tolist(), train_df["label_idx"].astype(int).tolist())
    ds_val = _make_ds(val_df[text_col].fillna("").astype(str).tolist(), val_df["label_idx"].astype(int).tolist())

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(CANONICAL_LABELS))
    use_fp16, use_bf16 = _resolve_precision(cfg.fp16)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        fp16=use_fp16,
        bf16=use_bf16,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        logging_strategy="epoch",
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor.to(logits.device))
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        trainer_cls = WeightedTrainer
    else:
        trainer_cls = Trainer

    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=ds_tr,
        eval_dataset=ds_val,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    latest_ckpt = _find_latest_trainer_checkpoint(checkpoint_dir) if resume_from_checkpoint else None
    t0 = time.perf_counter()
    trainer.train(resume_from_checkpoint=latest_ckpt)
    train_seconds = round(time.perf_counter() - t0, 2)

    def _predict_probs(texts: list[str]) -> np.ndarray:
        ds_eval = _make_ds(texts)
        logits = trainer.predict(ds_eval).predictions
        return torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()

    predictions: dict[str, np.ndarray] = {
        "val_probs": _predict_probs(val_df[text_col].fillna("").astype(str).tolist()),
    }

    if cfg.run_adversarial_eval and "text_adv" in val_df.columns:
        adv_texts = val_df["text_adv"].fillna("").astype(str).tolist()
        if cfg.smoke_max_eval_samples is not None:
            adv_texts = adv_texts[: cfg.smoke_max_eval_samples]
        predictions["val_adv_probs"] = _predict_probs(adv_texts)

    if cfg.run_heldout_eval:
        for eval_key in ("smishdetect_validation", "smishdetect_test", "banglaphish_ood"):
            edf = heldout_sets[eval_key]
            if cfg.smoke_max_eval_samples is not None:
                edf = edf.iloc[: cfg.smoke_max_eval_samples]
            predictions[f"{eval_key}_probs"] = _predict_probs(edf[text_col].fillna("").astype(str).tolist())

    if cfg.run_counterfactual_eval and "counterfactual_smish_probe" in heldout_sets:
        cf_df = heldout_sets["counterfactual_smish_probe"]
        if cfg.smoke_max_eval_samples is not None:
            cf_df = cf_df.iloc[: cfg.smoke_max_eval_samples]
        predictions["cf_orig_probs"] = _predict_probs(cf_df[f"orig_{text_col}"].fillna("").astype(str).tolist())
        predictions["cf_contact_probs"] = _predict_probs(
            cf_df[f"contact_swap_{text_col}"].fillna("").astype(str).tolist()
        )
        predictions["cf_brand_probs"] = _predict_probs(
            cf_df[f"contact_brand_swap_{text_col}"].fillna("").astype(str).tolist()
        )

    if best_model_export_dir is not None:
        best_model_export_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(best_model_export_dir))
        tokenizer.save_pretrained(str(best_model_export_dir))

    return {
        "train_seconds": train_seconds,
        "resumed_from_checkpoint": latest_ckpt,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "predictions": predictions,
    }


def _compute_unit_metrics(
    *,
    val_df: pd.DataFrame,
    heldout_sets: dict[str, pd.DataFrame],
    predictions: dict[str, np.ndarray],
    cfg: PostStageBConfig,
) -> dict[str, Any]:
    n_boot = 100 if cfg.run_mode == "gpu_smoke_test" else 500
    y_val = val_df["label_idx"].to_numpy(dtype=int)
    val_probs = predictions["val_probs"]

    metrics: dict[str, Any] = {
        "val_overall": evaluate_predictions(y_val, val_probs, label_names=CANONICAL_LABELS, n_boot=n_boot),
        "val_by_language": _evaluate_by_language(val_df, val_probs, n_boot=min(n_boot, 250)),
    }

    if "val_adv_probs" in predictions:
        n_adv = len(predictions["val_adv_probs"])
        metrics["val_adv_zwj_homoglyph"] = evaluate_predictions(
            y_val[:n_adv], predictions["val_adv_probs"], label_names=CANONICAL_LABELS, n_boot=n_boot
        )

    for eval_key in ("smishdetect_validation", "smishdetect_test", "banglaphish_ood"):
        pk = f"{eval_key}_probs"
        if pk in predictions:
            edf = heldout_sets[eval_key].iloc[: len(predictions[pk])]
            ey = edf["label_idx"].to_numpy(dtype=int)
            ep = predictions[pk]
            metrics[eval_key] = evaluate_predictions(ey, ep, label_names=CANONICAL_LABELS, n_boot=n_boot)
            if eval_key in ("smishdetect_validation", "smishdetect_test"):
                metrics[f"{eval_key}_by_language"] = _evaluate_by_language(edf, ep, n_boot=min(n_boot, 250))
            if eval_key == "banglaphish_ood" and "source_type" in edf.columns:
                sms_mask = (edf["source_type"] == "SMS").to_numpy()
                other_mask = ~sms_mask
                if int(sms_mask.sum()) >= 5:
                    metrics["banglaphish_ood_sms"] = evaluate_predictions(
                        ey[sms_mask], ep[sms_mask], label_names=CANONICAL_LABELS, n_boot=n_boot
                    )
                if int(other_mask.sum()) >= 5:
                    metrics["banglaphish_ood_other"] = evaluate_predictions(
                        ey[other_mask], ep[other_mask], label_names=CANONICAL_LABELS, n_boot=n_boot
                    )

    if "cf_orig_probs" in predictions:
        smish_idx = CANONICAL_LABELS.index("Smishing")
        p_orig = predictions["cf_orig_probs"]
        p_c = predictions["cf_contact_probs"]
        p_cb = predictions["cf_brand_probs"]
        orig_pct = float((p_orig.argmax(1) == smish_idx).mean() * 100.0)
        c_pct = float((p_c.argmax(1) == smish_idx).mean() * 100.0)
        cb_pct = float((p_cb.argmax(1) == smish_idx).mean() * 100.0)
        metrics["counterfactual_probe"] = {
            "n_probed": int(len(p_orig)),
            "smishing_pred_pct_before": round(orig_pct, 2),
            "contact_swap_pct_after": round(c_pct, 2),
            "contact_swap_pp_delta": round(c_pct - orig_pct, 2),
            "contact_brand_swap_pct_after": round(cb_pct, 2),
            "contact_brand_swap_pp_delta": round(cb_pct - orig_pct, 2),
        }
    return metrics


def _should_keep_weights(keep_mode: str, fold: int) -> bool:
    if keep_mode == "all":
        return True
    if keep_mode == "best_fold0":
        return fold == 0
    return False


def run_post_stage_b_pipeline(
    *,
    bundle_dir: str | Path,
    artifacts_dir: str | Path,
    cfg: PostStageBConfig,
    resume: bool = True,
    verify_checksums: bool = True,
    train_eval_fn: Callable[..., dict[str, Any]] | None = None,
    stop_after_units: int | None = None,
) -> dict[str, Any]:
    """Execute smoke test or Experiments A/B/C/D with atomic checkpoints and safe resume."""
    bdir = Path(bundle_dir)
    adir = Path(artifacts_dir)
    adir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(adir / "logs" / "runner.log")

    manifest = verify_manifest(bdir, verify_checksums=verify_checksums)
    manifest_sha = hashlib.sha256(
        (bdir / "dataset_manifest.json").read_bytes()
    ).hexdigest()
    env_info = collect_environment_info()
    atomic_write_json(adir / "exports" / "environment_snapshot.json", env_info)
    save_config(cfg, adir / "exports" / "active_config.json")

    state_path = adir / "run_state.json"
    if resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_hash") != cfg.config_hash():
            logger.warning(
                "Config hash changed (%s -> %s); keeping existing completed units that match unit_id.",
                state.get("config_hash"),
                cfg.config_hash(),
            )
    else:
        state = {
            "run_name": cfg.run_name,
            "run_mode": cfg.run_mode,
            "manifest_sha256": manifest_sha,
            "config_hash": cfg.config_hash(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_units": [],
            "failed_units": {},
        }
    state["last_resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
    state["manifest_sha256"] = manifest_sha
    state["config_hash"] = cfg.config_hash()
    atomic_write_json(state_path, state)

    heldout_sets = load_heldout_eval_sets(bdir)
    runner_fn = train_eval_fn or hf_train_and_predict_fold

    completed_set = set(state.get("completed_units", []))
    total_units = len(cfg.experiments) * len(cfg.models) * len(cfg.text_views) * len(cfg.folds)
    logger.info(
        "Starting run_name=%s mode=%s total_units=%d completed_already=%d cuda=%s gpu=%s",
        cfg.run_name,
        cfg.run_mode,
        total_units,
        len(completed_set),
        env_info["cuda_available"],
        env_info["gpu_name"],
    )

    executed_now = 0
    for exp_id in cfg.experiments:
        for model_name in cfg.models:
            for text_view in cfg.text_views:
                for fold in cfg.folds:
                    uid = unit_id(exp_id, model_name, text_view, fold)
                    unit_dir = adir / "units" / uid
                    metrics_path = unit_dir / "metrics.json"
                    preds_path = unit_dir / "preds.npz"

                    if resume and uid in completed_set and metrics_path.exists() and preds_path.exists():
                        logger.info("SKIP completed unit: %s", uid)
                        continue

                    if stop_after_units is not None and executed_now >= stop_after_units:
                        logger.info("Reached stop_after_units=%d; stopping cleanly for resume test.", stop_after_units)
                        break

                    balance_override = cfg.experiment_b_balance_mode if exp_id == "B" else None
                    train_df, val_df, split_meta = build_experiment_fold_data(
                        bdir,
                        exp_id,
                        fold,
                        seed=cfg.seed,
                        balance_mode_override=balance_override,
                    )

                    if cfg.smoke_max_train_samples is not None:
                        train_df = train_df.iloc[: cfg.smoke_max_train_samples].reset_index(drop=True)
                    if cfg.smoke_max_val_samples is not None:
                        val_df = val_df.iloc[: cfg.smoke_max_val_samples].reset_index(drop=True)

                    ckpt_dir = adir / "checkpoints" / uid
                    best_model_dir = (
                        unit_dir / "best_model" if _should_keep_weights(cfg.keep_model_weights, fold) else None
                    )
                    logger.info(
                        "RUN unit=%s n_train=%d n_val=%d view=%s",
                        uid,
                        len(train_df),
                        len(val_df),
                        text_view,
                    )
                    try:
                        run_kwargs: dict[str, Any] = {
                            "train_df": train_df,
                            "val_df": val_df,
                            "heldout_sets": heldout_sets,
                            "model_name": model_name,
                            "text_view": text_view,
                            "checkpoint_dir": ckpt_dir,
                            "cfg": cfg,
                            "class_weights": split_meta["class_weights"],
                            "seed": cfg.seed + fold,
                            "resume_from_checkpoint": resume,
                        }
                        if runner_fn is hf_train_and_predict_fold and best_model_dir is not None:
                            run_kwargs["best_model_export_dir"] = best_model_dir
                        run_out = runner_fn(**run_kwargs)
                        predictions = run_out["predictions"]
                        unit_metrics = _compute_unit_metrics(
                            val_df=val_df,
                            heldout_sets=heldout_sets,
                            predictions=predictions,
                            cfg=cfg,
                        )
                        record = {
                            "unit_id": uid,
                            "experiment_id": exp_id,
                            "model_name": model_name,
                            "text_view": text_view,
                            "fold": int(fold),
                            "seed": int(cfg.seed + fold),
                            "split_meta": split_meta,
                            "train_seconds": run_out.get("train_seconds", 0.0),
                            "resumed_from_checkpoint": run_out.get("resumed_from_checkpoint"),
                            "metrics": unit_metrics,
                            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                        }

                        unit_dir.mkdir(parents=True, exist_ok=True)
                        tmp_npz = unit_dir / ".preds.tmp.npz"
                        np.savez_compressed(tmp_npz, val_y=val_df["label_idx"].to_numpy(dtype=int), **predictions)
                        os.replace(tmp_npz, preds_path)
                        atomic_write_json(metrics_path, record)

                        # Remove heavy optimizer checkpoints once predictions (and optional best_model) are persisted
                        if ckpt_dir.exists():
                            shutil.rmtree(ckpt_dir, ignore_errors=True)

                        if uid not in state["completed_units"]:
                            state["completed_units"].append(uid)
                        state["failed_units"].pop(uid, None)
                        atomic_write_json(state_path, state)
                        completed_set.add(uid)
                        executed_now += 1
                        logger.info(
                            "DONE unit=%s val_macro_f1=%.2f%% recall@1pctFPR=%.2f%%",
                            uid,
                            unit_metrics["val_overall"]["macro_f1"],
                            unit_metrics["val_overall"]["smishing_recall_at_1pct_fpr"],
                        )
                    except Exception as exc:
                        state["failed_units"][uid] = {
                            "error": repr(exc),
                            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                        }
                        atomic_write_json(state_path, state)
                        logger.exception("FAILED unit=%s: %s", uid, exc)
                        raise

                if stop_after_units is not None and executed_now >= stop_after_units:
                    break
            if stop_after_units is not None and executed_now >= stop_after_units:
                break
        if stop_after_units is not None and executed_now >= stop_after_units:
            break

    summary = aggregate_post_stage_b_results(bdir, adir, cfg)
    logger.info("Pipeline run finished. Executed=%d TotalCompleted=%d/%d", executed_now, len(completed_set), total_units)
    return summary


def aggregate_post_stage_b_results(
    bundle_dir: str | Path,
    artifacts_dir: str | Path,
    cfg: PostStageBConfig,
) -> dict[str, Any]:
    """Aggregate completed fold metrics, 5-fold OOF CIs, McNemar tests, and Markdown/CSV exports."""
    bdir = Path(bundle_dir)
    adir = Path(artifacts_dir)
    real_df = pd.read_parquet(bdir / "real_train_pool.parquet")
    units_dir = adir / "units"

    rows: list[dict[str, Any]] = []
    oof_groups: dict[tuple[str, str, str], dict[int, tuple[np.ndarray, np.ndarray]]] = {}

    if units_dir.exists():
        for metrics_file in sorted(units_dir.glob("*/metrics.json")):
            rec = json.loads(metrics_file.read_text(encoding="utf-8"))
            uid = rec["unit_id"]
            m_val = rec["metrics"]["val_overall"]
            sd_test = rec["metrics"].get("smishdetect_test", {})
            bp_ood = rec["metrics"].get("banglaphish_ood", {})
            cf = rec["metrics"].get("counterfactual_probe", {})
            rows.append(
                {
                    "unit_id": uid,
                    "experiment_id": rec["experiment_id"],
                    "model_name": rec["model_name"],
                    "text_view": rec["text_view"],
                    "fold": rec["fold"],
                    "n_train": rec["split_meta"]["n_train"],
                    "n_val": rec["split_meta"]["n_val"],
                    "val_macro_f1": m_val["macro_f1"],
                    "val_accuracy": m_val["accuracy"],
                    "val_recall_at_1pct_fpr": m_val["smishing_recall_at_1pct_fpr"],
                    "val_normal_as_smish_fpr": m_val["normal_as_smishing_fpr_argmax"],
                    "val_ece": m_val["ece"],
                    "sd_test_macro_f1": sd_test.get("macro_f1"),
                    "bp_ood_macro_f1": bp_ood.get("macro_f1"),
                    "cf_contact_brand_pp_delta": cf.get("contact_brand_swap_pp_delta"),
                    "train_seconds": rec.get("train_seconds", 0.0),
                }
            )
            preds_file = metrics_file.parent / "preds.npz"
            if preds_file.exists():
                with np.load(preds_file) as npz:
                    oof_groups.setdefault((rec["experiment_id"], rec["model_name"], rec["text_view"]), {})[
                        int(rec["fold"])
                    ] = (npz["val_y"], npz["val_probs"])

    oof_summaries: dict[str, Any] = {}
    oof_preds_for_mcnemar: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    n_boot = 100 if cfg.run_mode == "gpu_smoke_test" else 1000

    for (exp_id, model_name, text_view), fold_map in sorted(oof_groups.items()):
        if set(fold_map.keys()) == {0, 1, 2, 3, 4} and all(
            len(fold_map[f][0]) == int((real_df["curation_cv_fold"] == f).sum()) for f in range(5)
        ):
            ys = np.concatenate([fold_map[f][0] for f in range(5)], axis=0)
            ps = np.concatenate([fold_map[f][1] for f in range(5)], axis=0)
            key = f"exp_{exp_id}__{model_name}__{text_view}"
            oof_summaries[key] = evaluate_predictions(ys, ps, label_names=CANONICAL_LABELS, n_boot=n_boot)
            oof_preds_for_mcnemar[(exp_id, model_name, text_view)] = (ys, ps.argmax(1))

    mcnemar_vs_exp_a: dict[str, Any] = {}
    for (exp_id, model_name, text_view), (ys, preds) in sorted(oof_preds_for_mcnemar.items()):
        if exp_id == "A":
            continue
        base = oof_preds_for_mcnemar.get(("A", model_name, text_view))
        if base is not None and len(base[0]) == len(ys):
            mcnemar_vs_exp_a[f"{exp_id}_vs_A__{model_name}__{text_view}"] = mcnemar_pvalue(ys, base[1], preds)

    summary = {
        "run_name": cfg.run_name,
        "run_mode": cfg.run_mode,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_unit_count": len(rows),
        "unit_rows": rows,
        "oof_5fold_summaries": oof_summaries,
        "mcnemar_vs_exp_a": mcnemar_vs_exp_a,
    }

    exports_dir = adir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(exports_dir / "aggregated_results.json", summary)

    df_rows = pd.DataFrame(rows)
    csv_text = df_rows.to_csv(index=False) if not df_rows.empty else ""
    atomic_write_text(exports_dir / "aggregated_results.csv", csv_text)

    md_lines = [
        f"# Post-Stage B Run Report — `{cfg.run_name}` (`{cfg.run_mode}`)",
        "",
        f"- Generated UTC: `{summary['generated_at_utc']}`",
        f"- Completed units: **{len(rows)}**",
        "",
        "| Unit ID | Exp | Model | View | Fold | Val Macro-F1 | Val Acc | Recall @ <=1% FPR | Normal->Smish FPR | SD Test F1 | BP OOD F1 | CF Brand Swap dPP |",
        "| :--- | :---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md_lines.append(
            f"| `{r['unit_id']}` | {r['experiment_id']} | `{r['model_name']}` | `{r['text_view']}` | {r['fold']} | "
            f"{r['val_macro_f1']}% | {r['val_accuracy']}% | {r['val_recall_at_1pct_fpr']}% | "
            f"{r['val_normal_as_smish_fpr']}% | {r['sd_test_macro_f1']} | {r['bp_ood_macro_f1']} | "
            f"{r['cf_contact_brand_pp_delta']} |"
        )
    if oof_summaries:
        md_lines.extend(["", "## Full 5-Fold OOF Summaries", ""])
        for k, om in oof_summaries.items():
            md_lines.append(
                f"- `{k}`: Macro-F1 **{om['macro_f1']}%** {om['macro_f1_ci95']}, "
                f"Recall@<=1%FPR **{om['smishing_recall_at_1pct_fpr']}%**, ECE `{om['ece']}`"
            )
    atomic_write_text(exports_dir / "post_stage_b_report.md", "\n".join(md_lines) + "\n")
    return summary
