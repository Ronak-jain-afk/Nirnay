import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

from src.config import NirnayConfig, OPTION_LETTERS

class NirnayDecisionModel(nn.Module):
    """
    Qwen2.5-0.5B wrapper for single-forward-pass calibrated decision making.
    Extracts candidate option logits directly from the final token position.
    """
    def __init__(
        self,
        config: NirnayConfig,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        option_token_ids: Optional[List[int]] = None,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        
        # Temperature parameter for post-hoc calibration
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))

        # Register option token IDs
        if option_token_ids is not None:
            self.register_buffer(
                "option_token_ids",
                torch.tensor(option_token_ids, dtype=torch.long)
            )
        else:
            self.option_token_ids = None

    @classmethod
    def from_pretrained(
        cls,
        config: NirnayConfig,
        is_training: bool = True,
        adapter_path: Optional[str] = None,
    ) -> "NirnayDecisionModel":
        """Instantiate Qwen2.5-0.5B with LoRA adapters."""
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Resolve option token IDs (e.g. for "A", "B", "C", "D", "E", "F")
        option_token_ids = [
            tokenizer.encode(letter, add_special_tokens=False)[-1]
            for letter in config.option_letters
        ]

        # Use BF16 on CUDA if available
        dtype = torch.bfloat16 if (config.bf16 and torch.cuda.is_available()) else torch.float32

        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        if adapter_path is not None and os.path.exists(adapter_path):
            print(f"[MODEL] Loading existing LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=is_training)
            if is_training:
                if config.gradient_checkpointing:
                    base_model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                    base_model.enable_input_require_grads()
                model.print_trainable_parameters()
        elif is_training:
            if config.gradient_checkpointing:
                base_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                base_model.enable_input_require_grads()

            peft_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(base_model, peft_config)
            model.print_trainable_parameters()
        else:
            model = base_model

        instance = cls(
            config=config,
            model=model,
            tokenizer=tokenizer,
            option_token_ids=option_token_ids,
        )
        return instance

    @property
    def backbone(self):
        """Retrieve the underlying transformer backbone (Qwen2Model)."""
        if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model") and hasattr(self.model.base_model.model, "model"):
            return self.model.base_model.model.model
        elif hasattr(self.model, "model") and hasattr(self.model.model, "model"):
            return self.model.model.model
        elif hasattr(self.model, "model"):
            return self.model.model
        return self.model

    @property
    def lm_head(self):
        """Retrieve LM output head."""
        if hasattr(self.model, "get_output_embeddings"):
            return self.model.get_output_embeddings()
        return getattr(self.model, "lm_head")

    def extract_option_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Runs one forward pass through transformer backbone and extracts only candidate
        option logits at the final position without materializing the full 152k vocabulary.
        """
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = out.last_hidden_state  # (batch_size, seq_len, 896)

        # Retrieve hidden state at the last position for each sequence
        if attention_mask is not None and getattr(self.tokenizer, "padding_side", "left") != "left":
            last_indices = attention_mask.sum(dim=-1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_hidden = hidden_states[batch_indices, last_indices, :]
        else:
            last_hidden = hidden_states[:, -1, :]  # (batch_size, 896)

        # Project only the candidate option tokens (e.g. A, B, C, D, E, F)
        lm_head = self.lm_head
        sub_weight = lm_head.weight[self.option_token_ids]
        bias = lm_head.bias[self.option_token_ids] if getattr(lm_head, "bias", None) is not None else None

        option_logits = F.linear(last_hidden, sub_weight, bias)
        return option_logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning unnormalized option logits and calibrated probabilities.
        
        Returns:
            (option_logits, probabilities)
        """
        logits = self.extract_option_logits(input_ids, attention_mask)

        if active_mask is not None:
            masked_logits = logits.masked_fill(~active_mask, -1e9)
        else:
            masked_logits = logits

        # Scale by calibrated temperature
        scaled_logits = masked_logits / self.temperature
        probabilities = F.softmax(scaled_logits, dim=-1)

        return logits, probabilities

    def set_temperature(self, tau: float):
        """Set post-hoc calibration temperature."""
        self.temperature.fill_(max(tau, 1e-4))
