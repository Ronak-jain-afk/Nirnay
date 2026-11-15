import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from src.config import NirnayConfig
from src.model import NirnayDecisionModel
from src.dataset import NirnayDataCollator
from src.trainer import NirnayTrainer
import argparse
from typing import Optional
from tqdm import tqdm
from src.calibrate import TemperatureCalibrator
from src.evaluate import NirnayEvaluator

def run_pilot_distillation(
    num_train_samples: int = 10000,
    num_val_samples: int = 1000,
    output_dir: str = "checkpoints/nirnay-0.5b-pilot",
    resume_from: Optional[str] = None,
):
    print("=" * 65)
    print(f"      NIRNAY: Pilot Distillation Run ({num_train_samples} Stratified Rows)      ")
    if resume_from:
        print(f"      [RESUMING FROM: {resume_from}]")
    print("=" * 65)

    config = NirnayConfig(output_dir=output_dir)
    print(f"Target Device:  {'CUDA (BF16)' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU Model:      {torch.cuda.get_device_name(0)}")
        print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    print(f"\n[1/5] Streaming {num_train_samples} samples from {config.dataset_name}...")
    dataset_train_stream = load_dataset(config.dataset_name, split="train", streaming=True)
    train_samples = []
    for item in tqdm(dataset_train_stream.take(num_train_samples), total=num_train_samples, desc="Loading train"):
        train_samples.append(item)
    print(f"Successfully loaded {len(train_samples)} training samples.")

    print(f"Loading {num_val_samples} validation samples...")
    dataset_val_stream = load_dataset(config.dataset_name, split="calibration", streaming=True)
    val_samples = []
    for item in tqdm(dataset_val_stream.take(num_val_samples), total=num_val_samples, desc="Loading val"):
        val_samples.append(item)
    print(f"Successfully loaded {len(val_samples)} validation samples.")

    print(f"\n[2/5] Initializing Qwen2.5-0.5B decision backbone and LoRA adapters...")
    model = NirnayDecisionModel.from_pretrained(
        config,
        is_training=True,
        adapter_path=resume_from,
    )

    collator = NirnayDataCollator(tokenizer=model.tokenizer, config=config)
    train_loader = DataLoader(
        train_samples,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_samples,
        batch_size=config.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    print(f"\n[3/5] Starting Pilot Distillation Training (1 Epoch)...")
    trainer = NirnayTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
    )

    if resume_from:
        trainer.load_checkpoint(resume_from)

    # Initial pre-training baseline evaluation
    pre_eval = trainer.evaluate(val_loader)
    print(f"Initial Zero-Shot Val Loss: {pre_eval['eval_loss']:.4f} | Accuracy: {pre_eval['eval_accuracy']*100:.2f}%")

    trainer.train_epoch(epoch=0)
    post_eval = trainer.evaluate(val_loader)
    print(f"Post-Pilot Val Loss:        {post_eval['eval_loss']:.4f} | Accuracy: {post_eval['eval_accuracy']*100:.2f}%")

    print(f"\n[4/5] Saving pilot checkpoint to {output_dir}...")
    trainer.save_checkpoint(output_dir)

    print(f"\n[5/5] Optimizing post-hoc temperature calibration...")
    calibrator = TemperatureCalibrator(model)
    cal_file = os.path.join(output_dir, "calibration_params.json")
    cal_results = calibrator.fit_temperature(val_loader, output_path=cal_file)

    evaluator = NirnayEvaluator(model)
    evaluator.evaluate(val_loader, name="Pilot Calibration Set")

    print("\n" + "=" * 65)
    print(f"Pilot Distillation Complete! Checkpoint saved in: {output_dir}")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Nirnay Pilot Distillation")
    parser.add_argument("--samples", type=int, default=10000, help="Number of pilot training rows (default: 10,000)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/nirnay-0.5b-pilot")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint directory to resume from")
    args = parser.parse_args()

    run_pilot_distillation(
        num_train_samples=args.samples,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
    )
