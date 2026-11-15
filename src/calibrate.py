import json
import os
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader

from src.model import NirnayDecisionModel

def compute_ece_and_brier(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float]:
    """
    Computes Expected Calibration Error (ECE) and Brier Score.
    """
    # Confidences and predicted classes
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    ground_truth = np.argmax(targets, axis=1)
    accuracies = (predictions == ground_truth).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            avg_acc = np.mean(accuracies[in_bin])
            avg_conf = np.mean(confidences[in_bin])
            ece += (bin_size / total_samples) * np.abs(avg_acc - avg_conf)

    # Brier score: mean squared error across probability distributions
    brier = float(np.mean(np.sum((probs - targets) ** 2, axis=1)))
    return float(ece), brier

class TemperatureCalibrator:
    """Calibrates output probabilities using post-hoc temperature scaling."""
    def __init__(self, model: NirnayDecisionModel):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def collect_logits_and_targets(
        self, dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        all_logits = []
        all_targets = []
        all_masks = []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["targets"].cpu().numpy()
            active_mask = batch["active_mask"].cpu().numpy()

            logits = self.model.extract_option_logits(input_ids, attention_mask)
            all_logits.append(logits.float().cpu().numpy())
            all_targets.append(targets)
            all_masks.append(active_mask)

        return (
            np.concatenate(all_logits, axis=0),
            np.concatenate(all_targets, axis=0),
            np.concatenate(all_masks, axis=0),
        )

    def fit_temperature(
        self, dataloader: DataLoader, output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        print("Collecting calibration logits and targets...")
        logits, targets, masks = self.collect_logits_and_targets(dataloader)

        # Apply active option masking
        masked_logits = np.where(masks, logits, -1e9)

        # Softmax helper
        def softmax(z, tau=1.0):
            scaled = z / tau
            e_x = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
            # Zero out masked options
            e_x = np.where(masks, e_x, 0.0)
            return e_x / np.sum(e_x, axis=1, keepdims=True)

        # Baseline metrics (tau = 1.0)
        baseline_probs = softmax(masked_logits, tau=1.0)
        base_ece, base_brier = compute_ece_and_brier(baseline_probs, targets)

        # Cross-entropy objective for temperature optimization
        def nll_objective(tau):
            p = softmax(masked_logits, tau=tau)
            # Clip for numerical stability
            log_p = np.log(np.clip(p, 1e-12, 1.0))
            return -np.mean(np.sum(targets * log_p, axis=1))

        res = minimize_scalar(nll_objective, bounds=(0.05, 5.0), method="bounded")
        best_tau = float(res.x)

        # Calibrated metrics
        calibrated_probs = softmax(masked_logits, tau=best_tau)
        cal_ece, cal_brier = compute_ece_and_brier(calibrated_probs, targets)

        results = {
            "optimal_temperature": best_tau,
            "baseline_ece": base_ece,
            "calibrated_ece": cal_ece,
            "ece_reduction_pct": ((base_ece - cal_ece) / max(base_ece, 1e-6)) * 100,
            "baseline_brier": base_brier,
            "calibrated_brier": cal_brier,
        }

        self.model.set_temperature(best_tau)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Calibration results saved to {output_path}")

        print(f"\n--- Temperature Calibration Summary ---")
        print(f"Optimal Temperature (tau): {best_tau:.4f}")
        print(f"ECE:   {base_ece:.4f} -> {cal_ece:.4f} (Reduction: {results['ece_reduction_pct']:.1f}%)")
        print(f"Brier: {base_brier:.4f} -> {cal_brier:.4f}")
        return results
