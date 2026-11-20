# Post-Stage B Colab readiness

**Colab gate: NOT VERIFIED. Full A/B/C/D decision: NO-GO.** The corrected notebook and the four tokenizer paths passed local checks. GPU training, CUDA memory use and recovery on a mounted Drive remain unverified for the corrected release. Exact model inputs also collide across selected training and validation/test boundaries under both brand views. The full grid must stay disabled pending owner review.

## Findings and verification

| Severity | Evidence and root cause | Correction or disposition | Regression and result |
| --- | --- | --- | --- |
| High | The initial notebook used a run-mode directory with no run identity in its path and printed an advisory when full mode was selected. It also described B as downsampled. See the initial generator at commit `282ef63`, cells 1 and 3. | Use a versioned v2 workspace and bundle, explicit artifact run ID, separate mode directories, and explicit full-run authorization. Check the mode and artifact path again at execution. The notebook now describes B as real-only class-weighted training. [Generator](../scripts/generate_post_stage_b_notebook.py#L20) | Local schema, syntax, generator equality and launch guard tests passed. It rejects missing full-run authorization and a traversal-style run ID. [Notebook tests](../tests/test_post_stage_b_preprocessing.py#L66) |
| High | Initial dependency setup checked importability and skipped existing incompatible versions. It omitted SentencePiece and the required BanglaBERT normalizer. | Validate distributions against version specifiers, pin the normalizer to upstream commit `d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9` when using BanglaBERT, and constrain installs to the existing CUDA PyTorch build. Require PyTorch 2.6+ for checkpoint loading on the selected Transformers range. [Generator](../scripts/generate_post_stage_b_notebook.py#L125) | The mocked Transformers 5.0 setup exercised the install branch and preserved the installed torch version. The public tokenizer dependencies independently resolved and loaded on CPU. Resolution of the complete Colab environment remains unverified. [Dependency regression](../tests/test_post_stage_b_preprocessing.py#L99) |
| High | The initial runner sent text directly to the tokenizer for every backbone, while the BanglaBERT model author requires its normalizer. | Normalize the selected view immediately before BanglaBERT tokenization in the shared train and inference path. The earlier view construction and original stored text are preserved. The authors’ normalization can change Unicode, whitespace and punctuation in the model input, including the raw and perturbed views. [Normalizing helper](../src/bangla_smish/post_stage_b_tokenization.py#L10) [BanglaBERT card](https://huggingface.co/csebuetnlp/banglabert) [Normalizer source](https://github.com/csebuetnlp/normalizer/blob/d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9/normalizer/normalize.py) | Routing and missing-normalizer regressions passed. The actual pinned normalizer removed U+200B, normalized curly quotes and preserved `<URL>`, `<PHONE>`, `<BRAND>` and `<USSD>` in CPU fixtures. The normalizer is research-licensed under CC BY-NC-SA 4.0. [Tokenizer tests](../tests/test_post_stage_b_preprocessing.py#L44) [Normalizer license](https://github.com/csebuetnlp/normalizer#license) |
| High | Different protected rows can become identical strings after brand masking or stripping. All four actual tokenizers reproduced the cross-boundary equality with truncation enabled. This is a property of the model representation even where row IDs and curated families are distinct. | The full-run guard refuses affected views. Resolving this requires an owner decision about the experimental protocol and protected splits. [Model-view audit](../scripts/audit_post_stage_b_model_views.py) | Each tokenizer ran 560 comparisons across A/B/C/D, all five folds, all four views, CV validation, held-out validation/test/OOD and all three counterfactual variants. Each reproduced the same 128 one-pair flags as the string audit. Raw, canonical and OOD comparisons had zero matches. |
| High | The previous T4 DistilBERT A/D completion, Drive persistence and completed-unit resume are owner-reported. The local worktree, ignored-file scan and Git history yielded the smoke configuration and no smoke prediction archive or run logs. | The historical cap and frozen row order independently reconstruct 64 train and 32 validation rows per unit. A used 64 real rows and D used 56 real plus 8 synthetic rows. Legacy achieved FPR, elapsed time, peak VRAM and checkpoint bytes require the original archive or logs. [Smoke reconstruction](post_stage_b_implementation_report.md#historical-smoke-reconciliation) | Corrected local smoke-cap and resume regressions use mocked training. The historical archive replay and current-release T4/Drive runs remain unverified. [Smoke configuration](../experiments/colab/configs/gpu_smoke_test_config.json) |
| High | The initial automatic precision check accepted the default PyTorch BF16 capability query, which includes emulation. A pre-Ampere T4 could be classified as BF16-capable through emulation. | Require native BF16 and use FP16 automatically on the T4. An explicit unsupported BF16 request raises an error. [Runner precision](../src/bangla_smish/post_stage_b_runner.py#L340) [PyTorch 2.6 implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/torch/cuda/__init__.py) | The mocked T4 native-versus-emulated precision regression passed. Physical T4 execution remains unverified. [Precision regression](../tests/test_post_stage_b_runner.py#L200) |
| Medium | Branch-derived Hub revisions and dependency ranges can resolve to different assets or package versions on different dates. Asset snapshots below record what the CPU check used. | Review the downloaded revisions and lock the Colab package set and model/tokenizer revisions before comparing completed full runs across sessions. The notebook records package versions and the dataset manifest SHA256, and the runner records the resolved model revision per completed unit. [Generator](../scripts/generate_post_stage_b_notebook.py#L151) [Runner](../src/bangla_smish/post_stage_b_runner.py#L497) | CPU snapshot recorded all four resolved revisions. Reproducibility of a future identical GPU environment remains unverified. |

## Preprocessing and tokenizer evidence

The v2 bundle applies the same stored views to A/B/C/D. `raw` preserves source text. `canon_masked` masks contact and USSD indicators and uses NFC. `brand_masked` also substitutes `<BRAND>` and `brand_stripped` removes the matched spans. Brand-aware views share the fixed vocabulary, including the static registry, all seven batch registries and accepted Pilot/Batch metadata. This is a preprocessing dependency for A/B/C even when the relevant cohort is withheld from their training rows. The independent reapplication of these rules produced zero mismatches across 12,466 non-probe rows and all four views. [View rules](../src/bangla_smish/schema.py#L530) [Bundle vocabulary](../src/bangla_smish/post_stage_b_data.py#L202)

- The full vocabulary had 364 tokens, the static vocabulary plus all batch registries had 276, and the static vocabulary alone had 258. Batch 7 supplied 8 canonical brands and 21 alias entries, corresponding to 18 NFC/casefold identities. The Batch 7 fixture passed Bengali and Banglish suffix checks and canonical-decomposition checks with dynamic fallback disabled.
- With the same detector, retaining all registries and removing accepted-row metadata changed no stored view in this corpus. Using only the static vocabulary changed 8 of the 345 accepted batch rows in each brand view. The Pilot, real, held-out validation, test and OOD stored views had zero changes in that static-vocabulary comparison. A/B/C still use the common vocabulary by design. These comparisons quantify the bundled dependency without changing it.
- Adversarial inputs perturb the raw string before applying the corresponding view and use the shared backbone tokenizer path afterward. Entity alterations may cause the existing regex to miss the perturbed form. That measures robustness of the complete view-and-model pipeline. In the 3,999-row real pool, the perturbed input differed from clean text on 3,782 raw and 3,774 per masked/stripped view. The number of changed token sequences ranged from 3,034 to 3,782 across the four backbones and views. The perturbation suite includes rows tokenized identically to their clean counterparts. [View assembly](../src/bangla_smish/post_stage_b_data.py#L229)

The CPU asset check used Python 3.12, Transformers 4.57.1, Tokenizers 0.22.2, SentencePiece 0.2.3 and the pinned normalizer. It loaded public configuration and tokenizer assets without downloading weights. All four used fast tokenizers. A long fixture was capped at 128 tokens for every backbone, and blank inputs produced the tokenizer’s usual special tokens only. Each length below includes the tokenizer’s special tokens.

| Configuration ID | Observed tokenizer | Resolved asset commit |
| --- | --- | --- |
| `csebuetnlp/banglabert` | `ElectraTokenizerFast` | `9ce791f330578f50da6bc52b54205166fb5d1c8c` |
| `google/muril-base-cased` | `BertTokenizerFast` | `afd9f36c7923d54e97903922ff1b260d091d202f` |
| `xlm-roberta-base` | `XLMRobertaTokenizerFast` | `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` |
| `distilbert-base-multilingual-cased` | `DistilBertTokenizerFast` | `45c032ab32cc946ad88a166f7cb282f58c753c2e` |

Each cell shows the measured number of rows above 128 tokens, in raw / canonical / brand-masked / brand-stripped order. Both accepted synthetic cohorts had zero rows above 128 for every backbone and view. The maximum observed length was 347 in DistilBERT’s held-out validation inputs. The existing cap and dynamic batch padding remain in use. [Runner tokenizer and collator](../src/bangla_smish/post_stage_b_runner.py#L386)

| Backbone | Real pool 3,999 | Held-out validation 708 | Test 1,698 | OOD 5,416 |
| --- | --- | --- | --- | --- |
| BanglaBERT | 19 / 11 / 11 / 11 | 6 / 2 / 2 / 2 | 21 / 5 / 5 / 5 | 0 / 0 / 0 / 0 |
| MuRIL | 9 / 5 / 5 / 5 | 2 / 2 / 2 / 2 | 10 / 2 / 2 / 2 | 0 / 0 / 0 / 0 |
| XLM-R | 14 / 9 / 9 / 9 | 5 / 5 / 5 / 5 | 2 / 2 / 3 / 2 | 0 / 0 / 0 / 0 |
| DistilBERT | 34 / 27 / 28 / 27 | 10 / 10 / 10 / 10 | 21 / 10 / 10 / 9 | 174 / 174 / 176 / 115 |

## T4 sizing and runtime evidence

The T4 has 16 GB nominal hardware memory and supports FP16. The notebook will report the actual visible device memory and CUDA/PyTorch versions for the lease. The audited runtime yielded no local CUDA or training VRAM measurement. XLM-R has the largest static memory estimate of the four. Budget additional space for activations, temporary kernels, allocator reservations and padding. [NVIDIA T4 specifications](https://www.nvidia.com/en-us/data-center/tesla-t4/) [Transformers training memory anatomy](https://huggingface.co/docs/transformers/v4.57.1/en/model_memory_anatomy)

| Backbone | Approximate parameters from base architectures | Illustrative mixed-precision AdamW static VRAM at 18 bytes/parameter | Illustrative full epoch checkpoint at 12 bytes/parameter | T4 evidence |
| --- | ---: | ---: | ---: | --- |
| BanglaBERT | 110 M | 1.98 GB | 1.32 GB | Tokenizer verified on CPU. T4 unverified. |
| MuRIL | 237 M | 4.27 GB | 2.84 GB | Tokenizer verified on CPU. T4 unverified. |
| XLM-R | 278 M | 5.00 GB | 3.34 GB | Tokenizer verified on CPU. T4 unverified. |
| DistilBERT multilingual | 135 M | 2.43 GB | 1.62 GB | Tokenizer verified on CPU. Old capped T4 run owner-reported. Corrected T4 unverified. |

These are decimal GB estimates. The checkpoint estimate counts FP32 weights plus two FP32 optimizer moment arrays. It excludes optional model exports, retry attempts, auxiliary files, size differences across package implementations and Drive overhead. With 80 units per backbone and all 3 epoch saves retained, the illustrative checkpoint total is about **2.2 TB** before retries. `keep_model_weights: none` disables the extra best-model export. Trainer epoch checkpoint retention still consumes storage. The measured v2 bundle disk footprint was 14 MiB (`du -sh`). [Runner checkpoint saves](../src/bangla_smish/post_stage_b_runner.py#L408) [MuRIL card](https://huggingface.co/google/muril-base-cased) [XLM-R card](https://huggingface.co/FacebookAI/xlm-roberta-base) [DistilBERT card](https://huggingface.co/distilbert/distilbert-base-multilingual-cased)

**Compute budget** The configuration has 320 units, 3 epochs per unit, single-device train/eval batches 32/64 and a 128-token cap. Gradient accumulation remains at the Trainer default of 1. A and B each have 300 optimizer updates per fold, C has 333 and D has 363, calculated as three times the ceiling of training rows divided by 32. Equal epochs therefore give C 11% and D 21% more updates than A/B and change absolute warmup/schedule lengths. An update-matched sensitivity run would equalize optimizer steps while changing cohort exposures per row. This audit retains the approved equal-epoch design. A smaller physical batch with accumulation would need separate authorization and validation of effective batch size, weighted-loss reduction, scheduler steps and measured memory. [Full configuration](../experiments/colab/configs/post_stage_b_abcd_config.json) [Hugging Face gradient accumulation](https://huggingface.co/docs/transformers/grad_accumulation)

**Wall time and lease** No credible training duration or compute-unit estimate is available. Record steady-state step timings, evaluation durations, model initialization, checkpoint write and checksum time in a capped authorized measurement before extrapolating per backbone. Colab availability, VM lifetime and limits vary, and mounted Drive has storage and I/O quotas. Local interruption recovery was exercised on the workspace filesystem. Mount-backed locking, rename visibility and checkpoint readback after remount require an empirical Drive check. [Runner lock](../src/bangla_smish/post_stage_b_runner.py#L609) [Colab FAQ](https://research.google.com/colaboratory/faq.html)

## Validation and owner preflight

```bash
.venv/bin/python scripts/generate_post_stage_b_notebook.py
.venv/bin/python -m unittest discover -s tests -p test_post_stage_b_preprocessing.py -v
.venv/bin/python -m unittest discover -s tests -p test_post_stage_b_runner.py -k t4 -v
.venv/bin/python scripts/prepare_post_stage_b_bundle.py --verify-only --bundle-dir data/processed/post_stage_b_bundle_v2
```

The generator validated the notebook. The focused preprocessing suite ran six tests with six passes, and the targeted precision regression passed. Local bundle verification is recorded in the integrity and test reports. CPU tokenizer, truncation and model-input equality checks completed using read-only parquet access and cached public tokenizer files. The audit used `AutoTokenizer.from_pretrained`, observed the resolved revisions in the table, and ran `prepare_tokenizer_texts` with `truncation=False` to measure original sequence lengths. The collision check used the configured `max_length=128` with `truncation=True`. No training or Drive operation was performed in this audit.

The CPU asset check’s temporary dependencies were installed and exercised with these commands. The script and cache paths were audit scratch space outside the repository.

```bash
uv --cache-dir /tmp/exploringais-uv-tokenizer-cache pip install --python .venv/bin/python \
  --target /tmp/exploringais-tokenizer-check 'transformers==4.57.1' 'sentencepiece>=0.2,<1' \
  'normalizer @ git+https://github.com/csebuetnlp/normalizer@d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9'
PYTHONPATH=/tmp/exploringais-tokenizer-check HF_HUB_CACHE=/tmp/exploringais-tokenizer-hf-cache \
  .venv/bin/python /tmp/exploringais-tokenizer-audit.py
PYTHONPATH=/tmp/exploringais-tokenizer-check HF_HUB_CACHE=/tmp/exploringais-tokenizer-hf-cache \
  .venv/bin/python /tmp/exploringais-tokenizer-audit-v2.py
PYTHONPATH=/tmp/exploringais-tokenizer-check HF_HUB_CACHE=/tmp/exploringais-tokenizer-hf-cache \
  .venv/bin/python /tmp/exploringais-token-collisions.py
PYTHONPATH=/tmp/exploringais-tokenizer-check HF_HUB_CACHE=/tmp/exploringais-tokenizer-hf-cache \
  .venv/bin/python /tmp/exploringais-token-boundaries.py
```

1. Resolve the representation-level collision gate and review storage retention and resources before considering a full run.
2. After owner authorization for the corrected smoke test, stage a new v2 kit in a fresh workspace and keep the notebook in `gpu_smoke_test`. Use a unique smoke run ID and preserve the legacy path. The notebook’s stop-after-one-unit test exercises completed-unit resume. Test recovery from an epoch checkpoint during a separate authorized capped run with at least two epochs and a fresh ID.
3. Record GPU name, visible VRAM, precision, package versions, model revisions, train/eval batches, peak VRAM, per-unit elapsed time, checkpoint sizes, Drive free space and readback results. Repeat capped checks for BanglaBERT, MuRIL and XLM-R before authorizing the four-backbone grid. Use identical configuration, code, bundle and artifact ID when resuming a specific run.
4. Keep `AUTHORIZE_FULL_TRAINING` false until a separate owner decision addresses all mandatory gates. For a later authorized fresh full run, use its own artifact ID.
