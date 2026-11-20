"""Offline regressions for Post-Stage B preprocessing and Colab launch."""

import importlib.util
import json
import sys
import unicodedata
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangla_smish.post_stage_b_data import collect_post_stage_b_brand_vocabulary
from bangla_smish.post_stage_b_tokenization import BANGLABERT_MODEL, prepare_tokenizer_texts
from bangla_smish.schema import compile_brand_regex, mask_text_views


class TestPostStageBPreprocessing(unittest.TestCase):
    def test_batch7_registry_covers_aliases_and_inflections(self):
        registry_file = ROOT / "data/generated/curation_v2/batch7_45/brand_registry.json"
        registry = json.loads(registry_file.read_text(encoding="utf-8"))["brands"]
        vocabulary = collect_post_stage_b_brand_vocabulary(ROOT)
        aliases = {unicodedata.normalize("NFC", alias).casefold() for alias in vocabulary}
        pattern = compile_brand_regex(vocabulary)
        for group in registry.values():
            for alias in group:
                with self.subTest(alias=alias):
                    self.assertIn(unicodedata.normalize("NFC", alias).casefold(), aliases)
                    views = mask_text_views(alias, brand_pattern=pattern, dynamic_fallback=False)
                    self.assertEqual(views["brand_masked"], "<BRAND>")
                    self.assertEqual(views["brand_stripped"], "")

        for alias in ("KheyaPayer", "ShoilpikStyle-te", "খেয়াপেতে", "বেলা কুরিয়ারে"):
            with self.subTest(inflected=alias):
                decomposed = unicodedata.normalize("NFD", alias)
                views = mask_text_views(decomposed, brand_pattern=pattern, dynamic_fallback=False)
                self.assertEqual(views["raw"], decomposed)
                self.assertTrue(unicodedata.is_normalized("NFC", views["canon_masked"]))
                self.assertEqual(views["brand_masked"], "<BRAND>")
                self.assertEqual(views["brand_stripped"], "")

    def test_banglabert_normalization_preserves_order_and_routes_all_views(self):
        texts = ["বাংলা ABC", " <URL> <PHONE> ", "<BRAND>", "", "বাংলা\u200b"]
        normalize = Mock(side_effect=lambda text: text.replace("\u200b", "").strip())
        with patch.dict(sys.modules, {"normalizer": SimpleNamespace(normalize=normalize)}):
            self.assertEqual(
                prepare_tokenizer_texts(texts, BANGLABERT_MODEL),
                ["বাংলা ABC", "<URL> <PHONE>", "<BRAND>", "", "বাংলা"],
            )
            self.assertEqual([call.args[0] for call in normalize.call_args_list], texts)
            normalize.reset_mock()
            for model in ("google/muril-base-cased", "xlm-roberta-base", "distilbert-base-multilingual-cased"):
                self.assertEqual(prepare_tokenizer_texts(texts, model), texts)
            normalize.assert_not_called()

    def test_banglabert_missing_normalizer_fails_closed(self):
        with patch.dict(sys.modules, {"normalizer": None}):
            with self.assertRaisesRegex(RuntimeError, "BanglaBERT requires"):
                prepare_tokenizer_texts(["বাংলা"], BANGLABERT_MODEL)


class TestPostStageBNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "experiments/colab/post_stage_b_experiments.ipynb"
        cls.notebook = nbformat.read(path, as_version=4)

    def test_schema_code_syntax_and_generator_match(self):
        nbformat.validate(self.notebook)
        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, f"<notebook cell {cell.id}>", "exec")
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])
        path = ROOT / "scripts/generate_post_stage_b_notebook.py"
        spec = importlib.util.spec_from_file_location("generate_post_stage_b_notebook", path)
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        self.assertEqual(generator.build_notebook(), self.notebook)

    def test_full_training_requires_authorization_and_artifact_name_is_safe(self):
        settings = {}
        exec(self.notebook.cells[1].source, settings)
        self.assertEqual(settings["RUN_MODE"], "gpu_smoke_test")
        self.assertFalse(settings["AUTHORIZE_FULL_TRAINING"])
        settings["RUN_MODE"] = "experiments_abcd"
        with self.assertRaisesRegex(RuntimeError, "explicit project owner authorization"):
            settings["validate_launch_settings"]()
        settings["AUTHORIZE_FULL_TRAINING"] = True
        settings["ARTIFACT_RUN_ID"] = "../legacy"
        with self.assertRaisesRegex(ValueError, "unique directory name"):
            settings["validate_launch_settings"]()
        settings["ARTIFACT_RUN_ID"] = "reviewed-v2-abcd-01"
        settings["validate_launch_settings"]()

    def test_dependency_setup_fixes_incompatible_installed_package_without_replacing_torch(self):
        settings = {
            "CONFIG_PATH": ROOT / "experiments/colab/configs/gpu_smoke_test_config.json",
            "Path": Path,
            "SRC_DIR": ROOT / "src",
        }
        compatible_versions = {
            "datasets": "4.0", "accelerate": "1.0", "safetensors": "0.6", "sentencepiece": "0.2",
            "numpy": "2.0", "pandas": "2.1", "scipy": "1.11", "scikit-learn": "1.4", "pyarrow": "15.0",
        }

        def installed_distribution(name):
            return SimpleNamespace(version="5.0" if name == "transformers" else compatible_versions[name])

        def inspect_install(command):
            constraint = Path(command[command.index("-c") + 1])
            self.assertEqual(constraint.read_text(encoding="utf-8"), "torch==2.6.0+cu124\n")
            self.assertIn("transformers>=4.57.1,<5", command)
            self.assertIn("sentencepiece>=0.2,<1", command)
            self.assertFalse(any(arg.startswith("torch") for arg in command))

        with (
            patch("importlib.metadata.distribution", side_effect=installed_distribution),
            patch("importlib.metadata.version", return_value="2.6.0+cu124"),
            patch("subprocess.check_call", side_effect=inspect_install) as install,
            self.assertRaisesRegex(RuntimeError, "Dependencies installed"),
        ):
            exec(self.notebook.cells[4].source, settings)
        install.assert_called_once()


if __name__ == "__main__":
    unittest.main()
