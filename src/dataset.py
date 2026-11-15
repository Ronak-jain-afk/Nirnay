import torch
from typing import Dict, Any, List, Optional, Tuple
from transformers import PreTrainedTokenizerBase
from src.config import NirnayConfig, OPTION_LETTERS

def format_decision_prompt(
    state: str,
    question: str,
    options: List[str],
    kind: str = "choice",
    option_letters: Optional[List[str]] = None,
) -> str:
    """
    Formats the state, question, and candidate options into a standard chat prompt.
    The prompt ends right before the decision token.
    """
    if option_letters is None:
        option_letters = OPTION_LETTERS

    # Format options with letter prefixes (e.g. "A: Option 1\nB: Option 2")
    formatted_options = []
    for i, opt in enumerate(options):
        if i < len(option_letters):
            formatted_options.append(f"{option_letters[i]}: {opt}")
    options_text = "\n".join(formatted_options)

    # Qwen-compatible chat prompt
    prompt = (
        f"<|im_start|>system\n"
        f"You are Nirnay, a fast calibrated decision model. Output the single letter corresponding to the best decision.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"State:\n{state}\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Decision:<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt

def extract_distribution(
    record: Dict[str, Any],
    options: List[str],
    num_total_letters: int = 6,
) -> Tuple[List[float], List[bool]]:
    """
    Extracts normalized probability distribution and active mask aligned with option letters.
    Handles dict, list, or winner-only annotations.
    """
    num_opts = len(options)
    targets = [0.0] * num_total_letters
    active_mask = [False] * num_total_letters

    for i in range(min(num_opts, num_total_letters)):
        active_mask[i] = True

    raw_dist = record.get("target") if record.get("target") is not None else record.get("distribution")
    
    if isinstance(raw_dist, list):
        for i, val in enumerate(raw_dist):
            if i < num_total_letters:
                targets[i] = float(val)
    elif isinstance(raw_dist, dict):
        # Could be keyed by option text, index, or letter ('A', 'B', etc.)
        for i, opt in enumerate(options):
            if i >= num_total_letters:
                break
            letter = OPTION_LETTERS[i]
            if opt in raw_dist:
                targets[i] = float(raw_dist[opt])
            elif letter in raw_dist:
                targets[i] = float(raw_dist[letter])
            elif str(i) in raw_dist:
                targets[i] = float(raw_dist[str(i)])
    elif "winner" in record and record["winner"] is not None:
        winner = record["winner"]
        for i, opt in enumerate(options):
            if i >= num_total_letters:
                break
            if opt == winner or OPTION_LETTERS[i] == winner:
                targets[i] = 1.0
                break

    # Normalize targets over active options
    active_sum = sum(targets[i] for i in range(num_total_letters) if active_mask[i])
    if active_sum > 0:
        for i in range(num_total_letters):
            if active_mask[i]:
                targets[i] /= active_sum
    else:
        # Uniform fallback over active options if missing or zero sum
        active_count = sum(active_mask)
        if active_count > 0:
            for i in range(num_total_letters):
                if active_mask[i]:
                    targets[i] = 1.0 / active_count

    return targets, active_mask

class NirnayDataCollator:
    """
    Collates raw dataset rows into tokenized batches with dynamic padding.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: NirnayConfig,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.num_total_letters = len(config.option_letters)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts = []
        all_targets = []
        all_masks = []
        kinds = []

        for row in batch:
            state = row.get("state", "")
            question = row.get("question", "")
            kind = row.get("kind", "choice")
            options = row.get("options", [])
            
            # Canonicalize options if empty
            if not options:
                if kind == "noul":
                    options = ["false", "true"]
                elif kind == "score":
                    options = ["0", "1", "2", "3", "4", "5"]
                else:
                    options = ["Option A", "Option B"]

            prompt = format_decision_prompt(
                state=state,
                question=question,
                options=options,
                kind=kind,
                option_letters=self.config.option_letters,
            )
            prompts.append(prompt)

            targets, active_mask = extract_distribution(
                record=row,
                options=options,
                num_total_letters=self.num_total_letters,
            )
            all_targets.append(targets)
            all_masks.append(active_mask)
            kinds.append(kind)

        # Tokenize with left padding to align the final decision token position
        encoding = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "targets": torch.tensor(all_targets, dtype=torch.float32),
            "active_mask": torch.tensor(all_masks, dtype=torch.bool),
            "kinds": kinds,
        }
