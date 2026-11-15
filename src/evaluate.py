import time
import json
import torch
import numpy as np
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader

from src.model import NirnayDecisionModel
from src.calibrate import compute_ece_and_brier

class NirnayEvaluator:
    """Evaluates the model on test splits, reporting Brier score, ECE, Top-1 accuracy, and latency."""
    def __init__(self, model: NirnayDecisionModel):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, name: str = "Test Set") -> Dict[str, Any]:
        self.model.eval()
        
        all_probs = []
        all_targets = []
        latencies = []
        correct_top1 = 0
        total_samples = 0

        # Warmup GPU
        for _ in range(5):
            dummy_in = torch.randint(0, 1000, (1, 32), device=self.device)
            _ = self.model.extract_option_logits(dummy_in)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["targets"].cpu().numpy()
            active_mask = batch["active_mask"].to(self.device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.amp.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            ):
                logits, probs = self.model(input_ids, attention_mask, active_mask=active_mask)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_latency_ms = ((t1 - t0) * 1000.0) / input_ids.size(0)
            latencies.append(batch_latency_ms)

            probs_np = probs.float().cpu().numpy()
            all_probs.append(probs_np)
            all_targets.append(targets)

            pred_choice = np.argmax(probs_np, axis=1)
            target_choice = np.argmax(targets, axis=1)
            correct_top1 += int(np.sum(pred_choice == target_choice))
            total_samples += input_ids.size(0)

        probs_matrix = np.concatenate(all_probs, axis=0)
        targets_matrix = np.concatenate(all_targets, axis=0)

        ece, brier = compute_ece_and_brier(probs_matrix, targets_matrix)
        accuracy = correct_top1 / max(total_samples, 1)
        mean_latency = float(np.mean(latencies))
        p95_latency = float(np.percentile(latencies, 95))

        metrics = {
            "dataset": name,
            "total_samples": total_samples,
            "accuracy_top1": float(accuracy),
            "ece": float(ece),
            "brier_score": float(brier),
            "mean_latency_ms": mean_latency,
            "p95_latency_ms": p95_latency,
            "throughput_decisions_per_sec": 1000.0 / mean_latency if mean_latency > 0 else 0.0,
        }

        print(f"\n================ Evaluation Report: {name} ================")
        print(f"Total Samples:     {total_samples}")
        print(f"Top-1 Accuracy:    {accuracy * 100:.2f}%")
        print(f"Brier Score:       {brier:.4f}")
        print(f"ECE (10-bin):      {ece:.4f}")
        print(f"Mean Latency:      {mean_latency:.2f} ms/decision")
        print(f"P95 Latency:       {p95_latency:.2f} ms/decision")
        print(f"Throughput:        {metrics['throughput_decisions_per_sec']:.1f} decisions/sec")
        print(f"===========================================================")

        return metrics
