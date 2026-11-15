# Nirnay (निर्णय) - 0.5B (v0-beta)

> **A fast, calibrated "System One" decision model powered by Qwen2.5-0.5B.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Model Size: 494M](https://img.shields.io/badge/Parameters-494M-green.svg)](https://huggingface.co/Qwen/Qwen2.5-0.5B)
[![Inference Latency: <28ms](https://img.shields.io/badge/Latency-%3C28ms-brightgreen.svg)]()
[![S1MB Evaluated: 137 Tasks](https://img.shields.io/badge/S1MB%20English%20v1-Validated-orange.svg)](https://huggingface.co/blog/hotchpotch/system-one-mosaic-benchmark)

Nirnay is a non-autoregressive decision model distilled from large reasoning teacher distributions (`SargeDev/jev-distill-corpus-v3`). Unlike conventional LLMs that generate verbose token-by-token chains of thought, Nirnay operates on the **"System One" cognitive paradigm**: delivering instant, calibrated probabilities in a **single forward pass** with native BF16 acceleration and $<28\text{ ms}$ latency on consumer laptop GPUs (e.g., RTX 4050 6GB).

---

## What is a System One Decision Model?

In cognitive science, **System 1** describes fast, instinctive, and automatic judgment, while **System 2** refers to slow, deliberative reasoning. 

Standard LLMs act like System 2: to make a simple routing or verification decision, they generate dozens of tokens autoregressively, taking hundreds of milliseconds. Nirnay bypasses token generation entirely:

```
State (Context / Text) ──┐
Question               ──┼──> [Qwen2.5-0.5B Backbone (BF16)] ──> Option Linear Projection ──> Calibrated Distribution
Options [A, B, ...]    ──┘    (Last Token Hidden State 896)      (Direct Option Weights)       (P(A), P(B), ...)
```

By extracting the final token hidden state and projecting directly onto candidate option weights, Nirnay eliminates the 152k-vocabulary LM head projection and KV-cache overhead, achieving **35+ decisions per second** at just **1.4 GB VRAM**.

---

## Core Decision Primitives

1. **`noul`**: Binary verifications (`[P(false), P(true)]`) for content moderation, guardrails, and assertions.
2. **`choice`**: Multi-class categorical routing across dynamic candidate options (e.g., intent triage, tool selection).
3. **`score`**: Ordinal ratings on a discrete `0..5` scale trained with Ranked Probability Score (RPS) to penalize distance from target distributions.
4. **Calibrated Probabilities**: Soft-target distillation ensures output confidence reflects true ground truth likelihoods rather than overconfident one-hot predictions.

---

## Official Benchmark Results

### 1. In-Distribution & Out-of-Distribution Benchmarks

Evaluated on the official test splits of the distillation corpus:

| Benchmark Split | Samples | Top-1 Accuracy | Brier Score | Mean Latency | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`test_set_30k`** (In-Distribution) | 1,000 | **99.60%** | **0.0001** | **27.45 ms** | **36.4 dec/sec** |
| **`ood`** (Out-of-Distribution) | 1,000 | **73.00%** | 0.3359 | **40.27 ms** | **24.8 dec/sec** |

*Hardware: Single NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) in BF16.*

---

### 2. System One Mosaic Benchmark (S1MB English v1)

Nirnay was evaluated across the complete, official **137-benchmark suite (14,009 cases)** of the [System One Mosaic Benchmark (S1MB)](https://huggingface.co/blog/hotchpotch/system-one-mosaic-benchmark) created by Yuichi Tateno (`hotchpotch`):

```text
Validated 137 files; 137 unique results
```

| Decision Type | Benchmarks | Nirnay-0.5B (`v0-beta` @ 48% steps) | Evaluation Focus |
| :--- | :---: | :---: | :--- |
| **Noul** (Binary Judgments) | 59 tasks | **10.77** | Yes/No classifications, safety filters, NLI |
| **Choice** (Candidate Selection) | 57 tasks | **10.26** | Multi-choice routing, QA, intent classification |
| **Score** (Ordinal Ratings) | 21 tasks | **0.48** | Fine-grained criteria scoring (0–5 scale) |
| **Official S1MB Task Avg** | **137 tasks** | **7.17** | Equal-weighted mean of baseline-adjusted scores |

#### Comparison with Published S1MB Leaderboard Models

| Model | Parameters | S1MB Task Avg | Noul | Choice | Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `laya` | 421M | 13.36 | 20.19 | 16.31 | 3.58 |
| `laya-multilingual` | 322M | 9.28 | 14.01 | 13.03 | 0.79 |
| `minojev` | 1.72B | 7.93 | 8.57 | 14.56 | 0.67 |
| **Nirnay-0.5B-v0-beta (Ours @ 48%)** | **494M** | **7.17** | **10.77** | **10.26** | **0.48** |
| `JevForge-0.8B` | 854M | 1.92 | 0.00 | 5.75 | 0.00 |

> *Note: S1MB Task Avg measures baseline-adjusted skill above constant/random baselines. At only 48% training progress (316,560 samples), Nirnay-0.5B already surpasses published 0.8B models and rivals 1.7B parameter architectures.*

#### Highlight Benchmarks (Baseline-Adjusted Skill)
* **`open-jev--customer-control-v1` (Noul)**: **100.00%**
* **`open-jev--vizdoom-basic-v1` (Noul)**: **94.12%**
* **`open-jev--customer-control-v1` (Choice)**: **82.61%**
* **`laya--ag-news` (Choice)**: **76.12%** (84.0% raw accuracy)
* **`open-jev--mailroom-control-v1` (Choice)**: **58.33%**
* **`s1mb-generalization-contextual-choice`**: **52.36%**
* **`arc-choice` (Science Reasoning)**: **8.82%** (38.0% raw accuracy vs 25% random)

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Nirnay.git
cd Nirnay

# Install dependencies via uv
uv sync
```

### 2. Python Inference Engine

```python
from src.infer import NirnayEngine

# Load standalone merged model or LoRA checkpoint
engine = NirnayEngine.from_pretrained("checkpoints/nirnay-0.5b-v0-beta")

# 1. Binary Gate Check (noul)
result = engine.decide(
    state="Database replica lag has spiked to 45 seconds during peak traffic.",
    question="Should read traffic be diverted to the primary cluster immediately?",
    kind="noul",
)
print(f"Decision: {result.decision} ({result.probabilities}) in {result.latency_ms:.2f} ms")

# 2. Multi-Class Routing (choice)
result = engine.decide(
    state="Customer query: 'Where can I download my annual billing receipt for 2025?'",
    question="Which department should handle this request?",
    options=["Technical Support", "Billing & Invoices", "Sales & Upgrades", "Account Security"],
    kind="choice",
)
print(f"Routed to: {result.decision} ({result.probabilities}) in {result.latency_ms:.2f} ms")

# 3. Quality Rating (score)
result = engine.decide(
    state="Customer review: 'Delivery was fast, but the packaging was torn and the product slightly scuffed.'",
    question="Rate the satisfaction score from 0 (very poor) to 5 (excellent):",
    kind="score",
)
print(f"Predicted Score: {result.decision} ({result.probabilities}) in {result.latency_ms:.2f} ms")
```

---

### 3. Serving via TypeSafe / S1MB `/v1/systemone` API

Nirnay includes a built-in, lightweight HTTP server that conforms to the standard TypeSafe / Jev System One protocol:

```bash
uv run main.py serve --model_path checkpoints/nirnay-0.5b-v0-beta --port 8080
```

Query the server with `curl`:
```bash
curl -X POST http://127.0.0.1:8080/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "state": "System memory usage at 91%. Inbound requests up 200%.",
    "questions": {
      "q1": {
        "type": "noul",
        "instructions": "Should auto-scaling worker nodes be provisioned immediately?"
      }
    }
  }'
```

---

### 4. Running Benchmarks

```bash
# Evaluate on test_set_30k and ood
uv run main.py evaluate --model_path checkpoints/nirnay-0.5b-v0-beta --num_samples 1000

# Run official S1MB harness (from S1MB checkout)
uv run s1mb run --adapter llama-cpp --model nirnay-0.5b --revision v0-beta --category smoke-v1 --limit 50
```

---

## Training & Distillation Methodology

Nirnay is distilled from `SargeDev/jev-distill-corpus-v3` (655,806 samples) with a specialized training pipeline:

1. **Architecture**: Base `Qwen/Qwen2.5-0.5B` adapted with 16-bit LoRA ($r=16, \alpha=32$) across all linear projection layers (`q, k, v, o, gate, up, down`).
2. **Sub-Vocabulary Projection**: The full 152k LM-head is bypassed; only the candidate option token embeddings are linearly projected from the final hidden state.
3. **Compound Objective**:
   $$\mathcal{L} = \mathcal{L}_{\text{CE}}(\hat{p}, p_{\text{teacher}}) + \lambda_{\text{RPS}} \cdot \mathcal{L}_{\text{RPS}}(\hat{p}, p_{\text{teacher}})$$
   Ranked Probability Score (RPS) penalizes the cumulative distribution error on ordinal `score` judgments.
4. **Long-Document Safeguard**: Documents are bounded to ~1,500 tokens with full preservation of dynamic candidate options, preventing attention memory thrashing.
5. **Fault-Tolerant Checkpointing**: Catches `Ctrl+C` (`KeyboardInterrupt`) to flush latest weights, scheduler steps, AdamW momentum & variance, and RNG states.

---

## Release Roadmap

- [x] **v0-pilot**: 5,000-sample feasibility study establishing sub-30ms latency and linear option heads.
- [x] **v0-beta (Current)**: 316,560 samples (Step 19,785), standalone 988 MB `.safetensors` release, verified across 137 S1MB benchmarks.
- [ ] **v1.0 (Upcoming)**: Full 655,806-sample epoch completion, post-hoc temperature scaling calibration, and GGUF / ONNX export for edge deployment.

---

## Citation & Acknowledgments

If you use Nirnay in your research or applications, please cite:

```bibtex
@software{nirnay2026,
  author = {Ronak & Contributors},
  title = {Nirnay: Fast, Calibrated System One Decision Model},
  year = {2026},
  url = {https://github.com/<your-username>/Nirnay}
}
```

Special thanks to:
* **Yuichi Tateno (`hotchpotch`)** for the [System One Mosaic Benchmark (S1MB)](https://github.com/hotchpotch/S1MB).
* **SargeDev** for open-sourcing the [Jev Distill Corpus v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3).
* **Qwen Team (Alibaba Cloud)** for the foundational [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) architecture.
