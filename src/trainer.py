import os
import sys
import glob
import shutil
import time
import math
import torch
from typing import Optional, Dict, Any, List
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from src.config import NirnayConfig
from src.model import NirnayDecisionModel
from src.loss import DistillationLoss

class NirnayTrainer:
    """
    Custom PyTorch trainer with:
    - Mixed precision (BF16) and gradient accumulation
    - Full training state persistence (optimizer, scheduler, step/epoch counters, RNG)
    - Periodic step checkpoints (every N steps) and automatic checkpoint rotation
    - Graceful Ctrl+C pause (emergency checkpoint-latest flush)
    - Seamless resume from any checkpoint directory
    """
    def __init__(
        self,
        model: NirnayDecisionModel,
        config: NirnayConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = DistillationLoss(rps_weight=config.rps_weight)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Track global progress
        self.global_step = 0
        self.current_epoch = 0
        self.resume_batch_idx = 0

        # Setup optimizer (only parameters requiring gradients - LoRA)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_steps = (len(train_dataloader) // config.gradient_accumulation_steps) * config.num_train_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(total_steps, 1),
        )

    def save_checkpoint(self, path: Optional[str] = None, is_emergency: bool = False):
        """
        Saves full checkpoint:
        1. LoRA adapter weights (adapter_model.safetensors, adapter_config.json)
        2. Tokenizer files
        3. Full trainer state (optimizer, scheduler, global_step, epoch, RNG)
        4. Updates checkpoint-latest
        """
        save_dir = path or os.path.join(self.config.output_dir, f"checkpoint-step-{self.global_step}")
        os.makedirs(save_dir, exist_ok=True)

        # 1. Save LoRA adapter weights
        if hasattr(self.model, "model") and hasattr(self.model.model, "save_pretrained"):
            self.model.model.save_pretrained(save_dir)
        elif hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_dir)

        # 2. Save tokenizer
        if self.model.tokenizer is not None:
            self.model.tokenizer.save_pretrained(save_dir)

        # 3. Save full optimizer, scheduler, and step state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        state_file = os.path.join(save_dir, "trainer_state.pt")
        torch.save(training_state, state_file)

        # 4. Synchronize to checkpoint-latest
        latest_dir = os.path.join(self.config.output_dir, "checkpoint-latest")
        if os.path.abspath(save_dir) != os.path.abspath(latest_dir):
            os.makedirs(latest_dir, exist_ok=True)
            if hasattr(self.model, "model") and hasattr(self.model.model, "save_pretrained"):
                self.model.model.save_pretrained(latest_dir)
            if self.model.tokenizer is not None:
                self.model.tokenizer.save_pretrained(latest_dir)
            torch.save(training_state, os.path.join(latest_dir, "trainer_state.pt"))

        # 5. Prune old step checkpoints based on save_total_limit
        self._prune_step_checkpoints()

        if is_emergency:
            print(f"\n[EMERGENCY CHECKPOINT] Successfully flushed training state to:")
            print(f"  -> {save_dir}")
            print(f"  -> {latest_dir}")
        else:
            print(f"\n[CHECKPOINT] Step {self.global_step} saved to: {save_dir}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Restores full training state (optimizer, scheduler, global_step, epoch, RNG) from a checkpoint directory.
        """
        state_file = os.path.join(checkpoint_path, "trainer_state.pt")
        if not os.path.exists(state_file):
            print(f"[WARN] No trainer_state.pt found in {checkpoint_path}. Starting optimizer from scratch.")
            return

        print(f"[RESUME] Restoring optimizer, scheduler, and step state from: {state_file}")
        state = torch.load(state_file, map_location=self.device)

        # Restore optimizer and scheduler states
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])

        # Restore counters
        self.global_step = state.get("global_step", 0)
        self.current_epoch = state.get("epoch", 0)

        # Restore RNG states safely (set_rng_state expects CPU ByteTensor)
        if "torch_rng_state" in state and state["torch_rng_state"] is not None:
            try:
                torch.set_rng_state(state["torch_rng_state"].cpu())
            except Exception as e:
                print(f"[WARN] Could not restore CPU RNG state: {e}")

        if "cuda_rng_state" in state and state["cuda_rng_state"] is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state(state["cuda_rng_state"].cpu())
            except Exception as e:
                print(f"[WARN] Could not restore CUDA RNG state: {e}")

        # Compute batch offset to skip already processed batches in current epoch
        batches_per_epoch = len(self.train_dataloader)
        accum_steps = self.config.gradient_accumulation_steps
        self.resume_batch_idx = (self.global_step * accum_steps) % batches_per_epoch

        print(f"[RESUME] State Restored: Epoch {self.current_epoch + 1}, Global Step {self.global_step}, Batch Offset {self.resume_batch_idx}/{batches_per_epoch}")

    def _prune_step_checkpoints(self):
        """Maintains only the most recent N step checkpoints to conserve disk space."""
        if self.config.save_total_limit is None or self.config.save_total_limit <= 0:
            return

        step_dirs = []
        for entry in glob.glob(os.path.join(self.config.output_dir, "checkpoint-step-*")):
            if os.path.isdir(entry):
                try:
                    step_num = int(entry.split("-step-")[-1])
                    step_dirs.append((step_num, entry))
                except ValueError:
                    pass

        step_dirs.sort(key=lambda x: x[0])
        while len(step_dirs) > self.config.save_total_limit:
            oldest_step, oldest_dir = step_dirs.pop(0)
            try:
                shutil.rmtree(oldest_dir)
                print(f"[PRUNE] Removed old checkpoint: {oldest_dir}")
            except Exception as e:
                print(f"[WARN] Failed to prune {oldest_dir}: {e}")

    def train_epoch(self, epoch: int) -> float:
        self.current_epoch = epoch
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()

        start_time = time.time()
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_train_epochs}")

        if self.resume_batch_idx > 0:
            print(f"[RESUME] Fast-forwarding {self.resume_batch_idx} batches already processed...")

        try:
            for step, batch in enumerate(pbar):
                # Fast forward if resuming mid-epoch
                if step < self.resume_batch_idx:
                    continue
                if self.resume_batch_idx > 0 and step == self.resume_batch_idx:
                    self.resume_batch_idx = 0  # Fast-forward complete

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["targets"].to(self.device)
                active_mask = batch["active_mask"].to(self.device)
                kinds = batch.get("kinds", None)

                # Mixed precision forward
                with torch.amp.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu",
                    dtype=torch.bfloat16 if self.config.bf16 and torch.cuda.is_available() else torch.float32,
                ):
                    logits = self.model.extract_option_logits(input_ids, attention_mask)
                    loss = self.loss_fn(logits, targets, active_mask=active_mask, kinds=kinds)
                    loss = loss / self.config.gradient_accumulation_steps

                loss.backward()

                step_loss = loss.item() * self.config.gradient_accumulation_steps
                total_loss += step_loss
                num_batches += 1

                # Optimizer step on gradient accumulation boundary
                if (step + 1) % self.config.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_dataloader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Periodic step checkpointing
                    if self.config.save_every_n_steps > 0 and self.global_step % self.config.save_every_n_steps == 0:
                        step_path = os.path.join(self.config.output_dir, f"checkpoint-step-{self.global_step}")
                        self.save_checkpoint(path=step_path)

                if (step + 1) % self.config.log_every_n_steps == 0:
                    vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                    current_lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        "step": self.global_step,
                        "loss": f"{step_loss:.4f}",
                        "lr": f"{current_lr:.2e}",
                        "vram": f"{vram_gb:.2f}GB",
                    })

        except KeyboardInterrupt:
            print("\n" + "!" * 70)
            print("  [GRACEFUL PAUSE] Caught Ctrl+C (KeyboardInterrupt)!")
            print("  Flushing gradients and saving emergency checkpoint-latest...")
            print("!" * 70)
            emergency_dir = os.path.join(self.config.output_dir, "checkpoint-latest")
            self.save_checkpoint(path=emergency_dir, is_emergency=True)
            print(f"\n[INFO] You can resume this training run at any time using:")
            print(f"       python main.py train --resume_from {emergency_dir}\n")
            print("!" * 70)
            sys.exit(0)

        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - start_time
        print(f"\nEpoch {epoch+1} Complete | Average Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
        return avg_loss

    @torch.no_grad()
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        eval_loader = dataloader or self.val_dataloader
        if eval_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        correct_top1 = 0
        total_samples = 0

        for batch in eval_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["targets"].to(self.device)
            active_mask = batch["active_mask"].to(self.device)
            kinds = batch.get("kinds", None)

            with torch.amp.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.bfloat16 if self.config.bf16 and torch.cuda.is_available() else torch.float32,
            ):
                logits, probs = self.model(input_ids, attention_mask, active_mask=active_mask)
                loss = self.loss_fn(logits, targets, active_mask=active_mask, kinds=kinds)

            total_loss += loss.item()
            num_batches += 1

            # Top-1 accuracy check
            pred_choice = torch.argmax(probs, dim=-1)
            target_choice = torch.argmax(targets, dim=-1)
            correct_top1 += (pred_choice == target_choice).sum().item()
            total_samples += input_ids.size(0)

        metrics = {
            "eval_loss": total_loss / max(num_batches, 1),
            "eval_accuracy": correct_top1 / max(total_samples, 1),
        }
        return metrics
