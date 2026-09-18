from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TrainingPreparationTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            json.loads(line)
            for line in (ROOT / "data/dataset.example.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.captions = json.loads(
            (ROOT / "configs/training/captions_en.example.json").read_text(encoding="utf-8")
        )
        self.masks = json.loads(
            (ROOT / "configs/training/header_masks.example.json").read_text(encoding="utf-8")
        )
        self.split = json.loads(
            (ROOT / "configs/training/split_v1.example.json").read_text(encoding="utf-8")
        )

    def test_caption_and_mask_coverage_is_exact(self):
        ids = {record["id"] for record in self.records}
        names = {Path(record["image_path"]).name for record in self.records}
        self.assertEqual(set(self.captions), ids)
        self.assertEqual(set(self.masks), names)

    def test_split_is_nonempty_unique_and_known(self):
        ids = {record["id"] for record in self.records}
        validation = self.split["validation_ids"]
        self.assertGreater(len(validation), 0)
        self.assertEqual(len(set(validation)), len(validation))
        self.assertTrue(set(validation) <= ids)

    def test_captions_do_not_embed_ids(self):
        for user_id, caption in self.captions.items():
            self.assertTrue(caption.strip())
            self.assertNotIn(user_id, caption)

    def test_backend_is_commit_pinned(self):
        lock = json.loads(
            (ROOT / "configs/training/backend.lock.json").read_text(encoding="utf-8")
        )
        self.assertRegex(lock["trainer"]["commit"], r"^[0-9a-f]{40}$")
        self.assertRegex(lock["base_model"]["revision"], r"^[0-9a-f]{40}$")
        self.assertRegex(lock["handwriting_font"]["commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(lock["handwriting_font"]["license"], "SIL-OFL-1.1")

    def test_square_avatar_training_config_is_conservative(self):
        config = (ROOT / "configs/training/sdxl_avatar_v2_dataset.toml").read_text(encoding="utf-8")
        self.assertIn("resolution = [1024, 1024]", config)
        self.assertIn("batch_size = 2", config)
        self.assertIn("enable_bucket = false", config)


if __name__ == "__main__":
    unittest.main()
