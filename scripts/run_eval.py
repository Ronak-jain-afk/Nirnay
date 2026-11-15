import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from src.config import NirnayConfig
from src.model import NirnayDecisionModel
from src.dataset import NirnayDataCollator
from src.evaluate import NirnayEvaluator

def run_benchmarks(
    model_path: str = "checkpoints/nirnay-0.5b-pilot",
    num_eval_samples: int = 2000,
):
    print("=" * 65)
    print("        NIRNAY: Benchmark Evaluation on Official Splits        ")
    print("=" * 65)

    config = NirnayConfig()
    print(f"Loading checkpoint from: {model_path}")
    model = NirnayDecisionModel.from_pretrained(
        config=config,
        is_training=False,
        adapter_path=model_path,
    )

    # Load calibration parameters if present
    cal_file = os.path.join(model_path, "calibration_params.json")
    if os.path.exists(cal_file):
        with open(cal_file, "r") as f:
            data = json.load(f)
            model.set_temperature(float(data.get("optimal_temperature", 1.0)))

    collator = NirnayDataCollator(tokenizer=model.tokenizer, config=config)
    evaluator = NirnayEvaluator(model)

    results = {}

    # Evaluate on test_set_30k
    print(f"\nLoading test_set_30k ({num_eval_samples} samples)...")
    test_stream = load_dataset(config.dataset_name, split="test_set_30k", streaming=True)
    test_samples = list(test_stream.take(num_eval_samples))
    test_loader = DataLoader(
        test_samples,
        batch_size=config.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    results["test_set_30k"] = evaluator.evaluate(test_loader, name="test_set_30k")

    # Evaluate on ood (out of distribution)
    print(f"\nLoading ood ({num_eval_samples} samples)...")
    ood_stream = load_dataset(config.dataset_name, split="ood", streaming=True)
    ood_samples = list(ood_stream.take(num_eval_samples))
    ood_loader = DataLoader(
        ood_samples,
        batch_size=config.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    results["ood"] = evaluator.evaluate(ood_loader, name="ood (Out of Distribution)")

    summary_file = os.path.join(model_path, "benchmark_summary.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark summary saved to: {summary_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Nirnay Benchmark Evaluation")
    parser.add_argument("--model_path", type=str, default="checkpoints/nirnay-0.5b-pilot")
    parser.add_argument("--num_samples", type=int, default=2000)
    args = parser.parse_args()

    run_benchmarks(model_path=args.model_path, num_eval_samples=args.num_samples)
