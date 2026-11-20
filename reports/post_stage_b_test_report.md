# Post-Stage B test and verification report

Tests ran from `/home/bs01415/Documents/Programming/ExploringAIs` on `main`. The regular suite used the repository’s Python 3.12 virtual environment. Runner integration tests used a deterministic CPU mock. Additional CPU checks used public tokenizer assets and temporary dependencies outside the repository.

| Executed command | Result |
|---|---|
| `.venv/bin/python -m unittest discover -s tests -v` | Exit 0. **63/63 passed**, zero failures or skips, 95.584 s. |
| `.venv/bin/python tests/test_pipeline.py` | Exit 0. “All tests passed!” |
| `.venv/bin/python scripts/prepare_post_stage_b_bundle.py --verify-only` | Exit 0. Recomputed frozen v1 membership and provenance checks, checked the 13 protected input digests and seven manifest files. Exposed 79 historical normalization metadata discrepancies. |
| `.venv/bin/python scripts/prepare_post_stage_b_bundle.py --verify-only --bundle-dir data/processed/post_stage_b_bundle_v2` | Exit 0. Corresponding checks on v2 passed, including zero normalization metadata discrepancies. |
| `.venv/bin/python scripts/audit_post_stage_b_model_views.py --bundle-dir data/processed/post_stage_b_bundle` | Exit 1. 128 of 560 exact transformed-input boundary comparisons failed, with one pair in each. |
| `.venv/bin/python scripts/audit_post_stage_b_model_views.py --bundle-dir data/processed/post_stage_b_bundle_v2` | Exit 1. Same 128 failures. These are confirmed data representation findings. |
| `.venv/bin/python -m unittest discover -s tests -p test_post_stage_b_metrics.py -v` | Exit 0. 6/6. |
| `.venv/bin/python -m unittest discover -s tests -p test_post_stage_b_runner.py -v` | Exit 0. 11/11. |
| `.venv/bin/python -m unittest discover -s tests -p test_post_stage_b_preprocessing.py -v` | Exit 0. 6/6. Includes notebook schema, code-cell syntax, regenerated source parity and launch/config guards. |
| `.venv/bin/python -c 'import nbformat; p="experiments/colab/post_stage_b_experiments.ipynb"; n=nbformat.read(p, as_version=4); nbformat.validate(n); print(f"Notebook schema validated ({len(n.cells)} cells)")'` | Exit 0. Validated all eight cells. |
| `git diff --check` | Exit 0. |
| SHA256 reread of the preflight inventory `/tmp/exploringais-post-stage-b-before.sha256` | Zero protected differences among 267 pre-existing data/report/legacy CSV files. Three observed experiment configuration/notebook paths were intentionally edited. |

## Critical regression mapping

| Severity | Case and implementation | Executed coverage |
|---|---|---|
| Critical | B construction and training loss · [data construction](../src/bangla_smish/post_stage_b_data.py#L772) · [trainer weighting](../src/bangla_smish/post_stage_b_runner.py#L291) | `test_experiment_abcd_fold_splits_leakage_free` checked 20 arms/folds against canonical ordered validation IDs, expected training cohorts, no shared family or row, and identical A/B ordered training IDs. `test_class_weights_from_hand_calculated_training_distribution` and `test_weighted_loss_matches_hand_calculation_only_in_train_mode` passed. Unsupported mode rejection passed. |
| Critical | Low-FPR precision and tied scores · [metric](../src/bangla_smish/metrics.py#L18) | Zero-budget float32/64, independent exhaustive cutoff search, ties, missing classes, false-positive count, limit breach after threshold transfer and exact JSON round-trip all passed. |
| Critical | Probe eligibility and leakage · [probe selector](../src/bangla_smish/post_stage_b_data.py#L384) | All five folds covered their exact expected probe IDs in each arm. Tampered fold assignment is rejected. Full-mode preflight regression passed by refusing the confirmed model-input leakage before invoking Trainer. |
| Critical | Resume and OOF identity · [archive checks](../src/bangla_smish/post_stage_b_runner.py#L703) | Configuration and manifest drift reject reuse. Corrupt ZIP and metrics retry to a new attempt. Shuffled rows are rejected even if the archive digest is updated. Committed unit count, uncommitted files, failed history and short-run resumptions were exercised. |
| High | Checkpoint integrity · [checkpoint validation](../src/bangla_smish/post_stage_b_runner.py#L230) | Tested malformed, incomplete, corrupted, and valid fixtures with older-checkpoint fallback. Native-only BF16 detection has a T4 dispatch fixture. Google Drive FUSE locking and actual CUDA checkpoint lifecycle remain unverified. |
| High | Smoke counts and preprocessing · [unit split](../src/bangla_smish/post_stage_b_runner.py#L635) | Both A and D have 64 actual training rows and 32 actual validation rows. Counts and class distributions match [the historical reconstruction](post_stage_b_implementation_report.md). Adversarial dispatch consulted the corresponding view. |
| High | Manifest and safe assembly · [bundle checker](../src/bangla_smish/post_stage_b_data.py#L629) | Physical metadata tamper, Parquet data corruption, and existing output directory rejection passed. Live v1/v2 checks independently verify frozen memberships. A temporary local kit export was checked by hashing its copied manifest and tables. Repeat export rejected the existing kit. |
| Medium | Statistical outputs · [metrics](../src/bangla_smish/metrics.py#L97) | Clustered intervals, paired accuracy deltas, fold OOF and McNemar alignment, and monotonic Holm correction were exercised. No confirmatory model result follows from mock predictions. |

## Supplementary real CPU loss calculation

The regular regression includes a NumPy-backed Torch stub. For a separate check, temporary installs provided PyTorch 2.6.0+cpu and Transformers 4.57.1 with Accelerate 1.10.1. I called the actual weighted Trainer subclass on fixed logits and checked the gradient. There was no optimizer step or training run. Measured weighted training loss was **1.380365133** and unweighted validation loss **1.151292562**, within `1e-6` of the hand calculations. The input logits and weight convention can be reproduced with the same dependencies.

```bash
PYTHONPATH=/tmp/exploringais-stage-b-cpu-deps:/tmp/exploringais-stage-b-trainer-deps:/tmp/exploringais-tokenizer-check:src .venv/bin/python - <<'PY'
from math import isclose, log
from tempfile import TemporaryDirectory
import torch
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput
from bangla_smish.post_stage_b_runner import _weighted_trainer_class

class FixedLogits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([[log(2), 0., 0.], [0., log(3), 0.]]))

    def forward(self, labels=None):
        loss = None if labels is None else torch.nn.functional.cross_entropy(self.logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=self.logits)

weights = [2/3, 1., 2.]
labels = torch.tensor([0, 2])
with TemporaryDirectory() as work:
    model = FixedLogits()
    args = TrainingArguments(work, use_cpu=True, report_to=[])
    trainer = _weighted_trainer_class(Trainer, weights)(model=model, args=args)
    model.train()
    train_loss = trainer.compute_loss(model, {'labels': labels})
    expected = ((2/3)*log(2) + 2*log(5)) / (2/3 + 2)
    assert isclose(train_loss.item(), expected, abs_tol=1e-6)
    train_loss.backward()
    assert torch.isfinite(model.logits.grad).all()
    model.eval()
    with torch.no_grad():
        eval_loss = trainer.compute_loss(model, {'labels': labels})
    assert isclose(eval_loss.item(), (log(2) + log(5))/2, abs_tol=1e-6)
PY
```

The prior T4 smoke `preds.npz` files, metrics, checkpoint files, and logs are absent from this repository and its Git history. The historical low-FPR replay therefore remains unverified. No corrected full or capped GPU model was trained. Real CPU tokenizer results for all four backbones, sequence length counts, and unmeasured Colab items are documented in the [Colab report](post_stage_b_colab_readiness_report.md).
