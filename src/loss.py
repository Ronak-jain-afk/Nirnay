import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DistillationLoss(nn.Module):
    """
    Distributional distillation loss:
    Computes KL divergence between student logits and teacher probability distributions,
    with an optional Ranked Probability Score (RPS) penalty for ordinal score primitives.
    """
    def __init__(self, rps_weight: float = 0.5):
        super().__init__()
        self.rps_weight = rps_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        kinds: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: Unnormalized logits (batch_size, num_options)
            targets: Teacher probability targets (batch_size, num_options) summing to 1.0 per active row
            active_mask: Boolean mask (batch_size, num_options) where True indicates a valid option
            kinds: List of kind strings (e.g. ['choice', 'score', ...]) for each sample in batch
        """
        if active_mask is not None:
            # Mask inactive options with very large negative value before softmax
            masked_logits = logits.masked_fill(~active_mask, -1e9)
        else:
            masked_logits = logits

        log_probs = F.log_softmax(masked_logits, dim=-1)
        
        # Soft cross-entropy / KL divergence:
        # KL(targets || probs) = sum(targets * (log(targets) - log_probs))
        # When targets have zeros, 0 * log(0) is 0.
        # - targets * log_probs provides exact gradients with zero NaN risk:
        ce_loss = -torch.sum(targets * log_probs, dim=-1)
        
        # If target has entropy, optionally compute true KL:
        with torch.no_grad():
            target_entropy = -torch.sum(targets * torch.log(targets.clamp(min=1e-12)), dim=-1)
        kl_loss = torch.mean(ce_loss - target_entropy)

        # Ranked Probability Score for ordinal 'score' types
        if self.rps_weight > 0 and kinds is not None:
            score_indices = [i for i, k in enumerate(kinds) if k == "score"]
            if score_indices:
                idx = torch.tensor(score_indices, device=logits.device, dtype=torch.long)
                sub_logits = masked_logits[idx]
                sub_targets = targets[idx]
                
                probs = F.softmax(sub_logits, dim=-1)
                cum_pred = torch.cumsum(probs, dim=-1)
                cum_target = torch.cumsum(sub_targets, dim=-1)
                
                # Exclude final CDF step since sum(p) == sum(y) == 1.0 (difference is 0)
                rps = torch.mean((cum_pred[:, :-1] - cum_target[:, :-1]) ** 2)
                return kl_loss + self.rps_weight * rps

        return kl_loss
