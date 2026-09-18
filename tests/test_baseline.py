from __future__ import annotations

import unittest

from src.prompting.prompt_builder import build_prompt
from src.semantic.concept_generator import generate_concept
from src.semantic.id_parser import tokenize_id


class BaselineTest(unittest.TestCase):
    def test_known_concept_uses_synthetic_example(self):
        concept = generate_concept("示例春花")
        self.assertEqual(concept.id, "示例春花")
        self.assertEqual(concept.expression_type, "symbolic")
        self.assertIn("花", concept.main_subjects)

    def test_arbitrary_flower_is_not_a_hidden_known_case(self):
        concept = generate_concept("普通春花")
        self.assertNotEqual(concept.main_concept, "从春天联想到春雨、阳光和花")

    def test_arbitrary_id_has_valid_concept(self):
        concept = generate_concept("蓝色小龙_7")
        concept.validate()
        self.assertEqual(concept.expression_type, "fantasy_hybrid")

    def test_generation_prompt_is_short_english_description(self):
        for user_id in ("示例春花", "样例黑猫_928", "蓝色小龙_7", "完全陌生昵称"):
            image_prompt = generate_concept(user_id).image_prompt
            self.assertTrue(image_prompt)
            self.assertTrue(image_prompt.isascii())
            self.assertLess(len(image_prompt.split()), 24)

    def test_prompt_preserves_sparse_doodle_style(self):
        positive, negative = build_prompt(generate_concept("虚构玫瑰"))
        self.assertIn("large empty space", positive)
        self.assertIn("professional illustration", negative)

    def test_empty_id_is_rejected(self):
        with self.assertRaises(ValueError):
            tokenize_id("   ")


if __name__ == "__main__":
    unittest.main()
