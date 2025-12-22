# Sentence Boundary Detection with LLM Prompting Strategies

## Problem Statement

Given a tokenized text sequence, the task is to determine **whether each token corresponds to a sentence boundary**.
Formally, for every token ( t_i ), we predict a binary label:

* `0` → token does **not** end a sentence
* `1` → token **ends** a sentence

This is a **token-level sentence segmentation** problem, where punctuation alone is insufficient due to abbreviations, quotations, ellipses, and literary constructs. The goal is to evaluate whether **prompt-based Large Language Models (LLMs)** can reliably perform this task without fine-tuning.

---

## Dataset Overview (EDA Summary)

The dataset consists of Italian literary text, provided in tokenized form with ground-truth sentence boundary labels.

* **Train / Dev**

  * Format: CSV with columns `token`, `label`
  * Highly imbalanced: ~3–4% sentence boundaries
  * Tokens include words and punctuation as separate units

* **Out-of-Domain (OOD)**

  * Format: semicolon-separated `token;label`
  * Sourced from a different literary work
  * Used to evaluate generalization

Key characteristics observed during EDA:

* Strong class imbalance (`label=1` is rare)
* Presence of noisy rows and formatting inconsistencies
* Sentence boundaries often depend on **context**, not punctuation alone

(Full EDA details are documented separately.) 

---

## Encoder / Prompting Strategies

We evaluate **seven LLM-based strategies**, each framing sentence boundary detection differently:

1. **Sliding Window Binary Classification**
   Local context window + independent YES/NO decision per punctuation.

2. **Next-Token Probability Analysis**
   Uses next-token probabilities to infer sentence starts (local models only).

3. **Marker Insertion**
   Model rewrites text inserting explicit `<EOS>` markers.

4. **Structured JSON Output**
   Model outputs sentence boundary indices or aligned binary labels.

5. **Few-Shot Learning with Hard Examples**
   Carefully curated edge cases guide the model without fine-tuning.

6. **Chain-of-Thought Reasoning**
   Explicit reasoning over punctuation, context, and syntax before prediction.

7. **Iterative Refinement**
   Two-pass approach that corrects uncertain or systematic errors.

All strategies share a common infrastructure for chunking, inference, parsing, and evaluation.

---

## Repository Structure

```text
DECODER/
├── model_calls/          # Local + OpenRouter model interfaces
├── prompts/              # Prompt templates
├── strategies_src/       # Strategy implementations and runner
├── visualization/        # Metrics plots and analysis scripts
├── results/              # CSV outputs and summaries
├── run_all.sh            # Experiment launcher
├── requirements.txt
```

---

## How to Run

### Quick Examples

```bash
# Quick test (Strategy 5, dev set)
./run_all.sh quick

# Run a single strategy
./run_all.sh single 5 llama-3.1-1b dev

# Run all strategies with local models
./run_all.sh all-local

# Run fast strategies only (3, 4, 5)
./run_all.sh fast

# Run everything
./run_all.sh all
```

### Run via Python Directly

```bash
python strategies_src/run_strategies.py --strategy 5 --model llama-3.1-1b --dataset dev
python strategies_src/run_strategies.py --all --local-only --all-datasets
```

---

