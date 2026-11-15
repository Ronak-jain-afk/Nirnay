from dataclasses import dataclass, field
from typing import Tuple, List

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F"]

@dataclass
class NirnayConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    
    # Dataset
    dataset_name: str = "SargeDev/jev-distill-corpus-v3"
    max_seq_len: int = 512
    option_letters: List[str] = field(default_factory=lambda: list(OPTION_LETTERS))
    
    # Training (Optimized for RTX 4050 6 GB VRAM)
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4   # Effective batch size = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    num_train_epochs: int = 1
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42
    
    # Loss
    rps_weight: float = 0.5  # Weight of Ranked Probability Score for ordinal scores
    
    # Output & Checkpointing
    output_dir: str = "checkpoints/nirnay-0.5b"
    calibration_file: str = "checkpoints/nirnay-0.5b/calibration_params.json"
    log_every_n_steps: int = 50
    eval_every_n_steps: int = 500
    save_every_n_steps: int = 500
    save_total_limit: int = 2
