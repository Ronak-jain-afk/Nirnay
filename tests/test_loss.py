import unittest
import torch
import torch.nn.functional as F
from src.loss import DistillationLoss

class TestDistillationLoss(unittest.TestCase):
    def setUp(self):
        self.loss_fn = DistillationLoss(rps_weight=0.5)

    def test_exact_match_zero_loss(self):
        """When logits match target distribution, KL loss should be near zero."""
        targets = torch.tensor([[0.2, 0.8, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        # Logits that yield exactly the target probabilities
        logits = torch.tensor([[np_val for np_val in [torch.log(torch.tensor(0.2)).item(), torch.log(torch.tensor(0.8)).item(), -100.0, -100.0, -100.0, -100.0]]], dtype=torch.float32)
        active_mask = torch.tensor([[True, True, False, False, False, False]])

        loss = self.loss_fn(logits, targets, active_mask=active_mask, kinds=["choice"])
        self.assertAlmostEqual(loss.item(), 0.0, places=3)

    def test_masked_options_ignored(self):
        """Inactive/padded options should not affect the probability distribution."""
        targets = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        # Large logits in inactive positions should be masked out
        logits_clean = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        logits_with_garbage = torch.tensor([[1.0, 1.0, 999.0, 999.0, 999.0, 999.0]], dtype=torch.float32)
        active_mask = torch.tensor([[True, True, False, False, False, False]])

        loss1 = self.loss_fn(logits_clean, targets, active_mask=active_mask)
        loss2 = self.loss_fn(logits_with_garbage, targets, active_mask=active_mask)
        self.assertAlmostEqual(loss1.item(), loss2.item(), places=3)

    def test_rps_ordinal_ordering(self):
        """Ranked Probability Score should penalize predictions far from true ordinal rank."""
        # True rating is 0 (first option)
        targets = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        active_mask = torch.ones((1, 6), dtype=torch.bool)

        # Prediction A predicts option 1 (close to 0)
        logits_close = torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        # Prediction B predicts option 5 (far from 0)
        logits_far = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 5.0]], dtype=torch.float32)

        loss_close = self.loss_fn(logits_close, targets, active_mask=active_mask, kinds=["score"])
        loss_far = self.loss_fn(logits_far, targets, active_mask=active_mask, kinds=["score"])

        # Distant ordinal prediction should incur strictly higher loss due to RPS penalty
        self.assertGreater(loss_far.item(), loss_close.item())

if __name__ == "__main__":
    unittest.main()
