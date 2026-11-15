import unittest
import torch
from src.config import NirnayConfig
from src.dataset import format_decision_prompt, extract_distribution

class TestDatasetAndFormatting(unittest.TestCase):
    def test_prompt_formatting(self):
        prompt = format_decision_prompt(
            state="CPU load 99%",
            question="Scale up?",
            options=["No", "Yes"],
            kind="choice"
        )
        self.assertIn("A: No", prompt)
        self.assertIn("B: Yes", prompt)
        self.assertIn("<|im_start|>system", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_extract_distribution_list(self):
        record = {"target": [0.3, 0.7]}
        targets, masks = extract_distribution(record, options=["No", "Yes"], num_total_letters=6)
        self.assertEqual(len(targets), 6)
        self.assertEqual(len(masks), 6)
        self.assertAlmostEqual(targets[0], 0.3)
        self.assertAlmostEqual(targets[1], 0.7)
        self.assertEqual(targets[2:], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(masks, [True, True, False, False, False, False])

    def test_extract_distribution_dict(self):
        record = {"distribution": {"false": 0.1, "true": 0.9}}
        targets, masks = extract_distribution(record, options=["false", "true"], num_total_letters=6)
        self.assertAlmostEqual(targets[0], 0.1)
        self.assertAlmostEqual(targets[1], 0.9)
        self.assertEqual(masks[:2], [True, True])
        self.assertEqual(masks[2:], [False, False, False, False])

if __name__ == "__main__":
    unittest.main()
