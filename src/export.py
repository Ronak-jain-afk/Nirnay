import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import shutil
import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.config import NirnayConfig

def export_merged_model(
    base_model_name: str,
    adapter_dir: str,
    output_dir: str,
    calibration_file: Optional[str] = None,
    save_dtype: torch.dtype = torch.bfloat16,
):
    """
    Merges LoRA adapter weights into base Qwen2.5-0.5B model and exports standalone safetensors.
    """
    print(f"Loading base model: {base_model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=save_dtype,
        device_map="cpu",
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        local_files_only=True,
    )

    print(f"Loading LoRA adapters from: {adapter_dir}...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("Merging adapter weights into base model...")
    merged_model = peft_model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving merged standalone model to: {output_dir}...")
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # Copy calibration params if present
    if calibration_file and os.path.exists(calibration_file):
        dest_cal = os.path.join(output_dir, "calibration_params.json")
        shutil.copyfile(calibration_file, dest_cal)
        print(f"Copied calibration parameters to {dest_cal}")

    print(f"Standalone model export completed successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export Nirnay model")
    parser.add_argument("--adapter_dir", type=str, default="checkpoints/nirnay-0.5b")
    parser.add_argument("--output_dir", type=str, default="checkpoints/nirnay-0.5b-merged")
    args = parser.parse_args()

    export_merged_model(
        base_model_name="Qwen/Qwen2.5-0.5B",
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        calibration_file=os.path.join(args.adapter_dir, "calibration_params.json"),
    )
