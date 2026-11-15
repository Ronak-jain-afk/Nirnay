import os
import time
import json
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.config import NirnayConfig, OPTION_LETTERS
from src.schema import DecisionResult, DecisionInput, DecisionKind
from src.dataset import format_decision_prompt

class NirnayEngine:
    """
    High-performance runtime engine for single-pass calibrated System One decisions.
    """
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        temperature: float = 1.0,
        option_letters: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = max(temperature, 1e-4)
        self.option_letters = option_letters or OPTION_LETTERS
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Cache option token IDs
        self.option_token_ids = [
            self.tokenizer.encode(letter, add_special_tokens=False)[-1]
            for letter in self.option_letters
        ]
        self.option_token_ids_tensor = torch.tensor(
            self.option_token_ids, dtype=torch.long, device=self.device
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        base_model_name: str = "Qwen/Qwen2.5-0.5B",
        device: Optional[str] = None,
    ) -> "NirnayEngine":
        dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

        # Check for calibration params
        cal_path = os.path.join(model_path, "calibration_params.json")
        temp = 1.0
        if os.path.exists(cal_path):
            with open(cal_path, "r") as f:
                cal_data = json.load(f)
                temp = float(cal_data.get("optimal_temperature", 1.0))
                print(f"Loaded calibrated temperature: tau = {temp:.4f}")

        # Check if model_path is an adapter directory or merged model
        adapter_config = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config):
            print(f"Loading base model {base_model_name} with LoRA adapter {model_path}...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=dtype,
                device_map="auto" if dev.type == "cuda" else None,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base, model_path)
        else:
            print(f"Loading standalone model from {model_path}...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto" if dev.type == "cuda" else None,
                trust_remote_code=True,
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(model=model, tokenizer=tokenizer, temperature=temp, device=dev)

    @torch.no_grad()
    def decide(
        self,
        state: str,
        question: str,
        options: Optional[List[str]] = None,
        kind: DecisionKind = "choice",
    ) -> DecisionResult:
        """
        Executes a single forward pass decision and returns calibrated probabilities.
        """
        # Resolve options based on primitive kind
        if options is None or len(options) == 0:
            if kind == "noul":
                options = ["false", "true"]
            elif kind == "score":
                options = ["0", "1", "2", "3", "4", "5"]
            else:
                raise ValueError("Must provide options for 'choice' primitive.")

        # Safeguard: Bound state length to ~1500 tokens so long documents (like legal contracts in ContractNLI)
        # do not cause quadratic attention VRAM spikes or PCIe paging thrashing.
        max_state_chars = 6000
        if len(state) > max_state_chars:
            state = state[:max_state_chars] + "... [truncated]"

        prompt = format_decision_prompt(
            state=state,
            question=question,
            options=options,
            kind=kind,
            option_letters=self.option_letters,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model") and hasattr(self.model.base_model.model, "model"):
            backbone = self.model.base_model.model.model
        elif hasattr(self.model, "model") and hasattr(self.model.model, "model"):
            backbone = self.model.model.model
        elif hasattr(self.model, "model"):
            backbone = self.model.model
        else:
            backbone = self.model

        lm_head = self.model.get_output_embeddings() if hasattr(self.model, "get_output_embeddings") else getattr(self.model, "lm_head")

        out = backbone(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )
        last_hidden = out.last_hidden_state[0, -1, :]  # (896,)
        sub_weight = lm_head.weight[self.option_token_ids_tensor[:len(options)]]
        candidate_logits = F.linear(last_hidden, sub_weight)

        # Apply temperature calibration
        scaled_logits = candidate_logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        probs_list = probs.float().cpu().tolist()
        logits_list = candidate_logits.float().cpu().tolist()

        prob_dict = {opt: round(p, 4) for opt, p in zip(options, probs_list)}
        logit_dict = {opt: round(l, 4) for opt, l in zip(options, logits_list)}
        best_decision = options[int(torch.argmax(probs).item())]

        return DecisionResult(
            decision=best_decision,
            probabilities=prob_dict,
            logits=logit_dict,
            temperature=self.temperature,
            latency_ms=round(latency_ms, 2),
            kind=kind,
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nirnay Interactive Decision CLI")
    parser.add_argument("--model_path", type=str, default="checkpoints/nirnay-0.5b")
    args = parser.parse_args()

    engine = NirnayEngine.from_pretrained(args.model_path)
    print("\n--- Nirnay Interactive Decision CLI ---")
    print("Engine ready. Evaluating sample incident scenario...\n")

    res = engine.decide(
        state="Pod cpu_usage_ratio is 0.94. Request queue depth increased 4x in 60s. Memory head-room is 350MB.",
        question="Should auto-scaling horizontal pod autoscaler (HPA) trigger replica addition?",
        options=["false", "true"],
        kind="noul",
    )
    print(f"Decision:      {res.decision}")
    print(f"Probabilities: {res.probabilities}")
    print(f"Latency:       {res.latency_ms:.2f} ms")
