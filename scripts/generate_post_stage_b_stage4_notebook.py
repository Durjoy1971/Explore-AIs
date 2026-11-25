"""Stage 4 Colab notebook generator at experiments/colab/post_stage_b_stage4_experiments.ipynb."""

from __future__ import annotations

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build_stage4_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
    nb.metadata["accelerator"] = "GPU"

    cells = []
    cells.append(
        nbf.v4.new_markdown_cell(
            "# Post-Stage B Stage 4 Experiments — Google Colab Local-SSD + Safe Staging Runner\n\n"
            "Prepared Stage 4 versioned notebook calling `src/bangla_smish/`.\n\n"
            "### Mandatory Governance & Safety Gates\n"
            "- **Canonical Notebook Immutability (Working Copy Required):** NEVER edit or execute `WORKSPACE_ROOT/experiments/colab/post_stage_b_stage4_experiments.ipynb` directly in Colab. First save a working copy OUTSIDE `DRIVE_WORKSPACE_REL` (`File -> Save a copy in Drive` into another folder) so Colab autosave never mutates the kit's verified notebook SHA256.\n"
            "- **Expected First-Run Dependency Restart (Cell 4):** On a fresh Colab runtime, Cell 4 installs missing/bounded dependencies (preserving Colab CUDA PyTorch) and intentionally raises `RESTART REQUIRED`. Click `Runtime -> Restart session` in the SAME GPU runtime, then re-run from Cell 1.\n"
            "- **Stage B synthetic generation is permanently stopped.**\n"
            "- **Strict Single-GPU & Tesla T4 Hardware Gate:** Cell 2 and the runner reject multi-GPU (`gpu_count > 1`) across ALL modes. For `RUN_MODE = \"t4_benchmark\"`, both enforce exactly 1 Tesla T4 GPU (`compute_capability == 7.5`) with a verified exact NVIDIA driver version bound into `run_state.json` identity (`schema_version=5`).\n"
            "- **T4 Benchmark (`RUN_MODE = \"t4_benchmark\"`)** runs ONLY the 5-unit calibration (`Exp D Fold 0 x 4 backbones + Exp B Fold 0 XLM-R`) on dev/fixture data (**zero held-out test/OOD exposure**) and requires explicit `AUTHORIZE_T4_BENCHMARK = True`. Under Decision 2 Option A (`benchmark_workload_fixture_eval=True` with `SYNTHETIC_WORKLOAD_FIXTURE_ENABLED=False`), the frozen T4 benchmark config fails closed BEFORE lease/state initialization until an owner-approved unmodified dev-data alternative is authorized.\n"
            "- **Full Option A 160-Unit Grid (`RUN_MODE = \"experiments_abcd\"`)** remains strictly **`NO-GO`** until T4 benchmark gates pass and requires separate explicit `AUTHORIZE_FULL_TRAINING = True`.\n"
            "- **Allowed-View Tokenizer Gate (`transformers==4.57.1` Snapshot Commit Verification):** Cell 5 executes `run_allowed_view_tokenizer_collision_gate` on the 32 documented Stage 3 pairs (`raw`, `canon_masked`) at pinned HuggingFace revisions using fail-closed snapshot-path & HF cache commit verification (never falling back to requested revision), binds `bundle_manifest_sha256` and live tokenizer package versions, saves its preflight report to system temp, and passes `TOK_GATE_REPORT` into Cell 6 for post-initialization export persistence.\n"
            "- **Statistical Governance:** `CONFIRMATORY_DECLARATIONS_ENABLED = False` (`FAIL_OR_INCONCLUSIVE_CONFIRMATORY_DISABLED`); `confirmatory_mode=True` controls ONLY the Option A experimental grid and never enables superiority claims.\n"
            "- **Checkpointing, Leases & Offline Aggregation:** Local-SSD + immutable UUID Drive staging + fail-closed single-writer leases requiring explicit `EXPECTED_STALE_WRITER_IDS` and `STALE_TAKEOVER_REASON` on authorized takeover; offline CPU `aggregate_post_stage_b_results` remains supported while preserving strict config/manifest/source-code identity."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "#@title 1. Stage 4 Runtime Configuration & Explicit Authorization Gates\n"
            "RUN_MODE = 'gpu_smoke_test'  #@param ['gpu_smoke_test', 't4_benchmark', 'experiments_abcd']\n"
            "AUTHORIZE_T4_BENCHMARK = False  #@param {type:'boolean'}\n"
            "AUTHORIZE_FULL_TRAINING = False  #@param {type:'boolean'}\n"
            "REQUIRE_GPU = True  #@param {type:'boolean'}\n"
            "USE_GOOGLE_DRIVE = True  #@param {type:'boolean'}\n"
            "DRIVE_WORKSPACE_REL = 'bangla_smish_post_stage_b_stage4_v8'  #@param {type:'string'}\n"
            "ARTIFACT_RUN_ID = 'stage4-v8-smoke-01'  #@param {type:'string'}\n"
            "RESUME = True  #@param {type:'boolean'}\n"
            "ALLOW_STALE_WRITER_TAKEOVER = False  #@param {type:'boolean'}\n"
            "EXPECTED_STALE_WRITER_IDS = ''  #@param {type:'string'}\n"
            "EXPECTED_CORRUPT_LEASE_FILES = ''  #@param {type:'string'}\n"
            "STALE_TAKEOVER_REASON = ''  #@param {type:'string'}\n"
            "\n"
            "def validate_stage4_launch_settings():\n"
            "    import re\n"
            "    from pathlib import Path\n"
            "    if RUN_MODE not in {'gpu_smoke_test', 't4_benchmark', 'experiments_abcd'}:\n"
            "        raise ValueError(f'Unknown run mode {RUN_MODE!r}')\n"
            "    rel_ws = DRIVE_WORKSPACE_REL.strip()\n"
            "    if not rel_ws or Path(rel_ws).is_absolute() or '..' in Path(rel_ws).parts:\n"
            "        raise ValueError('DRIVE_WORKSPACE_REL must be a non-empty relative path without parent traversal.')\n"
            "    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', ARTIFACT_RUN_ID):\n"
            "        raise ValueError('Set ARTIFACT_RUN_ID using letters, digits, hyphens, or underscores only.')\n"
            "    if RUN_MODE == 't4_benchmark' and not AUTHORIZE_T4_BENCHMARK:\n"
            "        raise RuntimeError('5-unit T4 benchmark requires separate explicit owner authorization.')\n"
            "    if RUN_MODE == 'experiments_abcd' and not AUTHORIZE_FULL_TRAINING:\n"
            "        raise RuntimeError('Full 160-unit Option A training is NO-GO without explicit owner authorization.')\n"
            "    if ALLOW_STALE_WRITER_TAKEOVER:\n"
            "        if not STALE_TAKEOVER_REASON.strip():\n"
            "            raise ValueError('Authorized stale-writer takeover requires a non-empty STALE_TAKEOVER_REASON.')\n"
            "        if not EXPECTED_STALE_WRITER_IDS.strip() and not EXPECTED_CORRUPT_LEASE_FILES.strip():\n"
            "            raise ValueError('Authorized stale-writer takeover requires explicit EXPECTED_STALE_WRITER_IDS or EXPECTED_CORRUPT_LEASE_FILES.')\n"
            "\n"
            "# Reset prior preflight state so any Cell 1 parameter change requires re-running Cells 3 and 5.\n"
            "PORTABLE_KIT_VERIFIED = False\n"
            "VERIFIED_WORKSPACE_ROOT = None\n"
            "VERIFIED_RUN_MODE = None\n"
            "VERIFIED_ARTIFACT_RUN_ID = None\n"
            "TOKENIZER_COLLISION_GATE_PASSED = False\n"
            "TOK_GATE_REPORT = None\n"
            "TOK_GATE_PREFLIGHT_PATH = None\n"
            "validate_stage4_launch_settings()\n"
            "print(f'Configured Stage 4 {RUN_MODE=} {ARTIFACT_RUN_ID=} {USE_GOOGLE_DRIVE=} {RESUME=} {ALLOW_STALE_WRITER_TAKEOVER=}')"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "#@title 2. Verify Colab Runtime, GPU & Local SSD\n"
            "import platform, shutil, subprocess, sys\n"
            "from pathlib import Path\n"
            "import torch\n"
            "\n"
            "IN_COLAB = 'google.colab' in sys.modules or Path('/content').exists()\n"
            "print('Python:', sys.version.split()[0], '| Platform:', platform.platform(), '| IN_COLAB:', IN_COLAB)\n"
            "nvidia_driver_version = None\n"
            "if shutil.which('nvidia-smi'):\n"
            "    print(subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip())\n"
            "    _drv_lines = [ln.strip() for ln in subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).splitlines() if ln.strip()]\n"
            "    if _drv_lines:\n"
            "        _uniq_drv = sorted(set(_drv_lines))\n"
            "        nvidia_driver_version = _uniq_drv[0] if len(_uniq_drv) == 1 else ','.join(_uniq_drv)\n"
            "cuda_ok = bool(torch.cuda.is_available())\n"
            "print('PyTorch:', torch.__version__, '| CUDA available:', cuda_ok, '| NVIDIA driver:', nvidia_driver_version)\n"
            "if REQUIRE_GPU and not cuda_ok:\n"
            "    raise RuntimeError('CUDA GPU is required for Stage 4 benchmark/execution.')\n"
            "gpu_count = int(torch.cuda.device_count()) if cuda_ok else 0\n"
            "gpu_name = None\n"
            "compute_capability = None\n"
            "if cuda_ok:\n"
            "    props = torch.cuda.get_device_properties(0)\n"
            "    gpu_name = str(torch.cuda.get_device_name(0))\n"
            "    cap = torch.cuda.get_device_capability(0)\n"
            "    compute_capability = f'{int(cap[0])}.{int(cap[1])}' if isinstance(cap, (tuple, list)) and len(cap) == 2 else str(cap)\n"
            "    print('GPU count:', gpu_count, '| GPU:', gpu_name, '| Compute capability:', compute_capability, f'| VRAM: {props.total_memory / (1024**3):.2f} GB')\n"
            "if RUN_MODE == 't4_benchmark':\n"
            "    if not cuda_ok:\n"
            "        raise RuntimeError('Fail closed: t4_benchmark requires a CUDA-enabled Tesla T4 GPU.')\n"
            "    if gpu_count != 1:\n"
            "        raise RuntimeError(f'Fail closed: t4_benchmark requires exactly 1 Tesla T4 GPU, found gpu_count={gpu_count}.')\n"
            "    if 'tesla t4' not in (gpu_name or '').lower():\n"
            "        raise RuntimeError(f'Fail closed: t4_benchmark requires a Tesla T4 GPU, found {gpu_name!r}.')\n"
            "    if compute_capability != '7.5':\n"
            "        raise RuntimeError(f'Fail closed: t4_benchmark requires Tesla T4 compute capability 7.5, found {compute_capability!r}.')\n"
            "    if not nvidia_driver_version:\n"
            "        raise RuntimeError('Fail closed: t4_benchmark requires a verified NVIDIA driver version from nvidia-smi.')\n"
            "    print('Verified Tesla T4 benchmark hardware:', {'gpu_count': gpu_count, 'gpu_name': gpu_name, 'compute_capability': compute_capability, 'nvidia_driver_version': nvidia_driver_version})\n"
            "elif cuda_ok and gpu_count > 1:\n"
            "    raise RuntimeError(f'Fail closed: Stage 4 single-GPU execution requires gpu_count == 1, found gpu_count={gpu_count}.')\n"
            "HARDWARE_PREFLIGHT = {\n"
            "    'cuda_available': cuda_ok,\n"
            "    'gpu_count': gpu_count,\n"
            "    'gpu_name': gpu_name,\n"
            "    'compute_capability': compute_capability,\n"
            "    'nvidia_driver_version': nvidia_driver_version,\n"
            "    'torch_cuda_version': getattr(getattr(torch, 'version', None), 'cuda', None),\n"
            "}\n"
            "ssd_path = '/content' if Path('/content').exists() else '.'\n"
            "disk = shutil.disk_usage(ssd_path)\n"
            "print(f'Free local SSD ({ssd_path}): {disk.free / (1024**3):.2f} GB')"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "#@title 3. Mount Fresh Stage 4 Google Drive Kit & Resolve Paths\n"
            "from pathlib import Path\n"
            "import sys\n"
            "\n"
            "if USE_GOOGLE_DRIVE:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive', force_remount=False)\n"
            "    WORKSPACE_ROOT = Path('/content/drive/MyDrive') / DRIVE_WORKSPACE_REL\n"
            "else:\n"
            "    WORKSPACE_ROOT = Path('/content') / DRIVE_WORKSPACE_REL\n"
            "\n"
            "SRC_DIR = WORKSPACE_ROOT / 'src'\n"
            "BUNDLE_DIR = WORKSPACE_ROOT / 'data' / 'processed' / 'post_stage_b_bundle_v2'\n"
            "CONFIG_DIR = WORKSPACE_ROOT / 'experiments' / 'colab' / 'configs'\n"
            "ARTIFACTS_DIR = WORKSPACE_ROOT / 'artifacts' / RUN_MODE / ARTIFACT_RUN_ID\n"
            "CFG_MAP = {\n"
            "    'gpu_smoke_test': 'gpu_smoke_test_config.json',\n"
            "    't4_benchmark': 'post_stage_b_stage4_t4_benchmark_config.json',\n"
            "    'experiments_abcd': 'post_stage_b_stage4_option_a_config.json',\n"
            "}\n"
            "CONFIG_PATH = CONFIG_DIR / CFG_MAP[RUN_MODE]\n"
            "if not SRC_DIR.is_dir() or not BUNDLE_DIR.is_dir() or not CONFIG_PATH.is_file():\n"
            "    raise FileNotFoundError(f'Missing Stage 4 kit files under {WORKSPACE_ROOT}. Stage kit via prepare_stage4_colab_kit.py first.')\n"
            "if str(SRC_DIR) not in sys.path:\n"
            "    sys.path.insert(0, str(SRC_DIR))"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "#@title 4. Preflight, Install & Verify Compatible Dependencies (Preserving Colab CUDA PyTorch)\n"
            "import json\n"
            "from runpy import run_path\n"
            "\n"
            "deps_mod = run_path(str(SRC_DIR / 'bangla_smish' / 'post_stage_b_colab_deps.py'))\n"
            "cfg_models = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))['models']\n"
            "dep_status = deps_mod['preflight_colab_dependencies'](\n"
            "    cfg_models,\n"
            "    install_missing=True,\n"
            "    require_cuda=REQUIRE_GPU,\n"
            ")\n"
            "print('Verified Stage 4 dependency stack:', json.dumps(dep_status, indent=2))"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "#@title 5. Verify Portable Kit, Frozen Masks, Config & 32-Pair Allowed-View Tokenizer Collision Gate\n"
            "import json\n"
            "import tempfile\n"
            "from pathlib import Path\n"
            "from bangla_smish.post_stage_b_data import sha256_file, verify_manifest\n"
            "from bangla_smish.post_stage_b_integrity import verify_portable_colab_kit\n"
            "from bangla_smish.post_stage_b_runner import collect_environment_info, load_config, seed_everything\n"
            "from bangla_smish.post_stage_b_stats import CONFIRMATORY_DECLARATIONS_ENABLED, STATISTICAL_VALIDATION_STATUS\n"
            "from bangla_smish.post_stage_b_tokenizer_gate import run_allowed_view_tokenizer_collision_gate\n"
            "\n"
            "PORTABLE_KIT_VERIFIED = False\n"
            "VERIFIED_WORKSPACE_ROOT = None\n"
            "VERIFIED_RUN_MODE = None\n"
            "VERIFIED_ARTIFACT_RUN_ID = None\n"
            "TOKENIZER_COLLISION_GATE_PASSED = False\n"
            "TOK_GATE_REPORT = None\n"
            "TOK_GATE_PREFLIGHT_PATH = None\n"
            "kit_summary = verify_portable_colab_kit(WORKSPACE_ROOT)\n"
            "print('Verified portable kit:', json.dumps(kit_summary, indent=2))\n"
            "manifest = verify_manifest(BUNDLE_DIR, verify_checksums=True)\n"
            "print('Verified bundle:', manifest['manifest_version'], '| SHA256:', sha256_file(BUNDLE_DIR / 'dataset_manifest.json'))\n"
            "cfg = load_config(CONFIG_PATH)\n"
            "if cfg.run_mode != RUN_MODE:\n"
            "    raise ValueError(f'Config run_mode {cfg.run_mode} != {RUN_MODE}')\n"
            "seed_everything(cfg.seed)\n"
            "if cfg.tokenizer_revisions:\n"
            "    TOK_GATE_PREFLIGHT_PATH = (\n"
            "        Path(tempfile.gettempdir())\n"
            "        / f'stage4_tokenizer_collision_gate_{RUN_MODE}_{ARTIFACT_RUN_ID}_{cfg.config_hash()[:12]}.json'\n"
            "    )\n"
            "    TOK_GATE_REPORT = run_allowed_view_tokenizer_collision_gate(\n"
            "        BUNDLE_DIR,\n"
            "        cfg,\n"
            "        allowed_views=tuple(v for v in cfg.text_views if v in {'raw', 'canon_masked'}),\n"
            "        export_path=TOK_GATE_PREFLIGHT_PATH,\n"
            "    )\n"
            "    TOKENIZER_COLLISION_GATE_PASSED = bool(TOK_GATE_REPORT['passed'])\n"
            "    print(\n"
            "        'Tokenizer collision gate PASSED:',\n"
            "        TOK_GATE_REPORT['total_checks'],\n"
            "        'checks, collisions=',\n"
            "        TOK_GATE_REPORT['collision_count'],\n"
            "        '| preflight_report=',\n"
            "        TOK_GATE_PREFLIGHT_PATH,\n"
            "    )\n"
            "elif RUN_MODE != 'gpu_smoke_test':\n"
            "    raise RuntimeError('Fail closed: tokenizer_revisions must be pinned for benchmark/confirmatory runs.')\n"
            "PORTABLE_KIT_VERIFIED = True\n"
            "VERIFIED_WORKSPACE_ROOT = WORKSPACE_ROOT\n"
            "VERIFIED_RUN_MODE = RUN_MODE\n"
            "VERIFIED_ARTIFACT_RUN_ID = ARTIFACT_RUN_ID\n"
            "print('Statistical governance:', STATISTICAL_VALIDATION_STATUS, '| confirmatory_declarations_enabled=', CONFIRMATORY_DECLARATIONS_ENABLED)\n"
            "print('Loaded config:', cfg.run_name, '| config_hash:', cfg.config_hash())\n"
            "print('Environment:', json.dumps(collect_environment_info(), indent=2))"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "#@title 6. Execute Stage 4 Pipeline (Local-SSD + Immutable Drive Staging + Safe Leases)\n"
            "from bangla_smish.post_stage_b_runner import run_post_stage_b_pipeline\n"
            "\n"
            "validate_stage4_launch_settings()\n"
            "_required_globals = ('WORKSPACE_ROOT', 'BUNDLE_DIR', 'ARTIFACTS_DIR', 'CONFIG_PATH', 'CFG_MAP', 'cfg')\n"
            "_missing_globals = [name for name in _required_globals if name not in globals()]\n"
            "if _missing_globals:\n"
            "    raise RuntimeError(f'Missing prerequisite cell state {_missing_globals}; run Cells 1 through 5 in order before Cell 6.')\n"
            "if (\n"
            "    not globals().get('PORTABLE_KIT_VERIFIED', False)\n"
            "    or globals().get('VERIFIED_WORKSPACE_ROOT') != WORKSPACE_ROOT\n"
            "    or globals().get('VERIFIED_RUN_MODE') != RUN_MODE\n"
            "    or globals().get('VERIFIED_ARTIFACT_RUN_ID') != ARTIFACT_RUN_ID\n"
            "    or ARTIFACTS_DIR != WORKSPACE_ROOT / 'artifacts' / RUN_MODE / ARTIFACT_RUN_ID\n"
            "    or CONFIG_PATH != WORKSPACE_ROOT / 'experiments' / 'colab' / 'configs' / CFG_MAP[RUN_MODE]\n"
            "    or cfg.run_mode != RUN_MODE\n"
            "):\n"
            "    raise RuntimeError('Stale or unverified preflight state detected; re-run Cells 3 and 5 after any Cell 1 change.')\n"
            "if cfg.tokenizer_revisions and (\n"
            "    not globals().get('TOKENIZER_COLLISION_GATE_PASSED', False)\n"
            "    or not isinstance(globals().get('TOK_GATE_REPORT'), dict)\n"
            "):\n"
            "    raise RuntimeError('Pre-training 32-pair allowed-view tokenizer collision gate has not passed.')\n"
            "stale_ids = [s.strip() for s in EXPECTED_STALE_WRITER_IDS.split(',') if s.strip()] or None\n"
            "corrupt_files = [s.strip() for s in EXPECTED_CORRUPT_LEASE_FILES.split(',') if s.strip()] or None\n"
            "summary = run_post_stage_b_pipeline(\n"
            "    bundle_dir=BUNDLE_DIR,\n"
            "    artifacts_dir=ARTIFACTS_DIR,\n"
            "    cfg=cfg,\n"
            "    resume=RESUME,\n"
            "    verify_checksums=True,\n"
            "    allow_stale_takeover=ALLOW_STALE_WRITER_TAKEOVER,\n"
            "    expected_stale_writer_ids=stale_ids,\n"
            "    expected_corrupt_files=corrupt_files,\n"
            "    takeover_reason=STALE_TAKEOVER_REASON or None,\n"
            "    tokenizer_gate_report=globals().get('TOK_GATE_REPORT'),\n"
            ")\n"
            "print('Completed units total:', summary['completed_unit_count'], '/', summary['expected_unit_count'])"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "#@title 7. Inspect Exported Telemetry, Optimizer Steps, Retraining Metadata & Results\n"
            "import json, pandas as pd\n"
            "exports_dir = ARTIFACTS_DIR / 'exports'\n"
            "if not exports_dir.is_dir():\n"
            "    print('Exports directory not found yet:', exports_dir)\n"
            "else:\n"
            "    print('Exported files:', [p.name for p in sorted(exports_dir.glob('*'))])\n"
            "    tok_gate_path = exports_dir / 'tokenizer_collision_gate.json'\n"
            "    if tok_gate_path.exists():\n"
            "        tgate = json.loads(tok_gate_path.read_text(encoding='utf-8'))\n"
            "        print('Tokenizer gate export:', 'PASSED' if tgate.get('passed') else 'FAILED', '| checks=', tgate.get('total_checks'), '| collisions=', tgate.get('collision_count'))\n"
            "    retrain_path = exports_dir / 'retraining_metadata.json'\n"
            "    if retrain_path.exists():\n"
            "        rmeta = json.loads(retrain_path.read_text(encoding='utf-8'))\n"
            "        print('Retraining metadata:', rmeta['unit_count'], 'units | status=', rmeta['statistical_validation_status'])\n"
            "    csv_path = exports_dir / 'aggregated_results.csv'\n"
            "    if csv_path.exists() and csv_path.stat().st_size > 0:\n"
            "        print(pd.read_csv(csv_path).to_string(index=False))"
        )
    )

    cell_ids = ("s4md0001", "s4cfg002", "s4gpu003", "s4mnt004", "s4dep005", "s4ver006", "s4run007", "s4out008")
    for cell, cid in zip(cells, cell_ids, strict=True):
        cell["id"] = cid
    nb["cells"] = cells
    return nb


def main() -> None:
    out_path = ROOT / "experiments" / "colab" / "post_stage_b_stage4_experiments.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nb = build_stage4_notebook()
    nbf.validate(nb)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated and validated {out_path}")


if __name__ == "__main__":
    main()
