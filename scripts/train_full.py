import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from typing import Optional
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from src.config import NirnayConfig
from src.model import NirnayDecisionModel
from src.dataset import NirnayDataCollator
from src.trainer import NirnayTrainer
from src.calibrate import TemperatureCalibrator
from src.evaluate import NirnayEvaluator

def run_full_distillation(
    num_train_samples: int = -1,
    epochs: int = 1,
    output_dir: str = "checkpoints/nirnay-0.5b",
    resume_from: Optional[str] = None,
):
    print("=" * 65)
    print("           NIRNAY: Production Distillation Run                 ")
    if resume_from:
        print(f"           [RESUMING FROM: {resume_from}]")
    print("=" * 65)

    config = NirnayConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
    )
    print(f"Device: {'CUDA (BF16)' if torch.cuda.is_available() else 'CPU'}")

    print(f"\n[1/5] Loading training dataset: {config.dataset_name}...")
    if num_train_samples > 0:
        stream = load_dataset(config.dataset_name, split="train", streaming=True)
        train_samples = list(stream.take(num_train_samples))
    else:
        dataset = load_dataset(config.dataset_name, split="train")
        train_samples = dataset

    print(f"Total training samples: {len(train_samples)}")

    print("Loading calibration and validation sets...")
    cal_stream = load_dataset(config.dataset_name, split="calibration", streaming=True)
    cal_samples = list(cal_stream.take(5000))

    print("\n[2/5] Initializing Qwen2.5-0.5B model with LoRA...")
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
        cal_samples,
        batch_size=config.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    print(f"\n[3/5] Starting Distillation Training ({epochs} epoch(s))...")
    trainer = NirnayTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
    )

    if resume_from:
        trainer.load_checkpoint(resume_from)

    start_ep = trainer.current_epoch
    for ep in range(start_ep, epochs):
        trainer.train_epoch(epoch=ep)
        metrics = trainer.evaluate(val_loader)
        print(f"Epoch {ep+1} Evaluation: Loss: {metrics['eval_loss']:.4f} | Accuracy: {metrics['eval_accuracy']*100:.2f}%")

    print(f"\n[4/5] Saving final model checkpoint...")
    trainer.save_checkpoint(output_dir)

    print(f"\n[5/5] Temperature Calibration...")
    calibrator = TemperatureCalibrator(model)
    cal_file = os.path.join(output_dir, "calibration_params.json")
    calibrator.fit_temperature(val_loader, output_path=cal_file)

    evaluator = NirnayEvaluator(model)
    evaluator.evaluate(val_loader, name="Full Calibration Split")

    print("\n" + "=" * 65)
    print(f"Distillation Finished! Checkpoint: {output_dir}")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Nirnay Full Distillation")
    parser.add_argument("--num_samples", type=int, default=-1, help="-1 for entire dataset")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="checkpoints/nirnay-0.5b")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint directory to resume from")
    args = parser.parse_args()

    run_full_distillation(
        num_train_samples=args.num_samples,
        epochs=args.epochs,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
    )
