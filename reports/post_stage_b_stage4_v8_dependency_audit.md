# Post-Stage B Stage 4 v8 Dependency Closure & Verification Audit

**Date:** 2026-11-25  
**Scope:** Upstream verification and local closure testing of the Stage 4 Colab dependency specification across [`src/bangla_smish/post_stage_b_colab_deps.py`](../src/bangla_smish/post_stage_b_colab_deps.py), [`src/bangla_smish/post_stage_b_runner.py`](../src/bangla_smish/post_stage_b_runner.py), and [`pyproject.toml`](../pyproject.toml).

---

## 1. Upstream-Verified Dependency Trees

### A. `transformers==4.57.1` (PyPI Metadata `requires_dist`)
- `filelock` -> pinned `filelock>=3.0`
- `huggingface-hub<1.0,>=0.34.0` -> pinned `huggingface_hub>=0.34.0,<1.0`
- `numpy>=1.17` -> pinned `numpy>=1.26`
- `packaging>=20.0` -> pinned `packaging>=23.0`
- `pyyaml>=5.1` -> pinned `pyyaml>=5.1`
- `regex!=2019.12.17` -> pinned `regex!=2019.12.17`
- `requests` -> pinned `requests>=2.31`
- `tokenizers<=0.23.0,>=0.22.0` -> pinned `tokenizers>=0.22.0,<=0.23.0`
- `safetensors>=0.4.3` -> pinned `safetensors>=0.4.3`
- `tqdm>=4.27` -> pinned `tqdm>=4.27`

### B. `csebuetnlp/normalizer@d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9`
- Upstream `setup.py`: `install_requires = ["regex", "emoji==1.4.2", "ftfy==6.0.3"]`.
- `normalizer/normalize.py` imports Python stdlib `unicodedata` (NOT `unicodedata2`).
- `normalizer/const.py` imports `regex` and `emoji`.
- `ftfy==6.0.3` `setup.py`: `install_requires = ["wcwidth"]`.
- `emoji==1.4.2`: zero runtime dependencies.

**Root Cause of v7 P0 Failure:** Stage 4 v7 erroneously included `unicodedata2` in `_BANGLABERT_REQUIRED_PACKAGE_DISTS` and `BANGLABERT_NORMALIZER_TRANSITIVE_DEPS`, while omitting `wcwidth` and `regex` from BanglaBERT/core checks and omitting `tokenizers`/`pyyaml`/`requests`/`tqdm`/`filelock` from `STAGE4_BASE_PACKAGES`. On a fresh Colab runtime where `normalizer` never installs `unicodedata2`, Cell 6 would have deterministically raised `RuntimeError: required runtime package versions missing ... {'unicodedata2': 'not_installed'}`.

---

## 2. Reconciled Stage 4 v8 24-Distribution Closure

| Group | Distributions (`24` Total) | Checked in Cell 4 (`post_stage_b_colab_deps.py`) | Checked in Cell 6 (`post_stage_b_runner.py`) | Aligned in `pyproject.toml [gpu]` |
| :--- | :--- | :---: | :---: | :---: |
| **Core Runtime (20)** | `torch`, `transformers`, `tokenizers`, `datasets`, `accelerate`, `safetensors`, `sentencepiece`, `packaging`, `huggingface_hub`, `protobuf`, `regex`, `pyyaml`, `requests`, `tqdm`, `filelock`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `pyarrow` | Yes (`STAGE4_BASE_PACKAGES` + `torch>=2.6` + `STAGE4_CORE_IMPORTS`) | Yes (`_CORE_REQUIRED_PACKAGE_DISTS` + `_IDENTITY_PACKAGE_DISTS`) | Yes |
| **BanglaBERT Normalizer (4)** | `normalizer` (pinned VCS commit `d405944d...`), `ftfy`, `emoji`, `wcwidth` (`regex` already in Core) | Yes (`required_packages_for_models` + `required_imports_for_models`) | Yes (`_BANGLABERT_REQUIRED_PACKAGE_DISTS` + `normalizer_commit` check) | Documented VCS requirement |
| **Removed Bogus Deps** | `unicodedata2`, `urllib3` (from BanglaBERT tuple) | Removed | Removed | Never present |

Note: `post_stage_b_colab_deps.py` executes `post_stage_b_tokenization.py` standalone via `exec()` in Cell 4 before `bangla_smish` is imported; `post_stage_b_tokenization.py` remains strictly stdlib-only at module top-level (lazy-importing `transformers.utils.hub` / `huggingface_hub` inside `resolve_hf_cached_tokenizer_files`).

---

## 3. Closure Test Evidence & Remaining Live-Colab Boundary

- **Verified locally with real `importlib.metadata.PathDistribution`:** `tests/test_stage4_v8_remediation.py::test_scenario_b_dependency_closure_real_dist_info_no_unicodedata2` constructs a real temporary `site-packages` directory with `.dist-info` metadata for all 24 distributions (zero `unicodedata2`), verifies that BOTH Cell 4 (`preflight_colab_dependencies`) and Cell 6 (`collect_stable_environment_fingerprint` + `_validate_required_packages_fingerprint`) pass cleanly, and confirms that deleting `wcwidth`, `regex`, or `tokenizers` causes BOTH Cell 4 and Cell 6 to fail closed.
- **Not verified locally:** Actual `pip install` dependency resolution against live PyPI / GitHub inside Google Colab's pre-installed CUDA PyTorch wheel image (remains pending an authorized Colab runtime session).
