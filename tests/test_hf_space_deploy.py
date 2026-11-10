from __future__ import annotations

import ast
import unittest
from pathlib import Path

from scripts.publish_hf_space import space_files, validate_files


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "deploy/huggingface/app.py"


class HuggingFaceSpaceDeployTest(unittest.TestCase):
    def test_publish_manifest_is_explicit_and_complete(self):
        files = space_files()
        validate_files(files)
        self.assertIn("app.py", files)
        self.assertIn("README.md", files)
        self.assertIn("weights/id_impression_avatar_sdxl_v2.safetensors", files)
        self.assertNotIn("data/dataset.jsonl", files)
        self.assertEqual(sum(name.endswith(".safetensors") for name in files), 1)

    def test_space_app_uses_zerogpu_and_full_size_sdxl(self):
        source = APP.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIn("@spaces.GPU(duration=50)", source)
        self.assertIn("width=1024", source)
        self.assertIn("height=1024", source)
        self.assertIn("num_inference_steps=30", source)
        self.assertTrue(any(isinstance(node, ast.FunctionDef) and node.name == "generate_avatar" for node in ast.walk(tree)))

    def test_space_card_selects_gradio(self):
        card = (ROOT / "deploy/huggingface/README.md").read_text(encoding="utf-8")
        self.assertIn("sdk: gradio", card)
        self.assertIn("app_file: app.py", card)
        self.assertIn("python_version: 3.10.13", card)


if __name__ == "__main__":
    unittest.main()
