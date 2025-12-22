"""
Strategy 6: Chain-of-Thought Reasoning

Ask model to reason step by step about each potential boundary.
More expensive but potentially more accurate for hard cases.
"""

import os
import re
from typing import List, Dict
from tqdm import tqdm

from stratagies_src.utils import (
    load_manzoni_data,
    load_ood_data,
    evaluate_predictions,
    save_predictions,
    print_metrics,
    get_data_paths,
)

STRATEGY_ID = 6
STRATEGY_NAME = "chain_of_thought"


def get_punctuation_indices(tokens: List[str]) -> List[int]:
    """Find indices of punctuation marks."""
    punctuation = {'.', '!', '?', ';', ':', '...', '…'}
    indices = []
    for i, token in enumerate(tokens):
        if token in punctuation:
            indices.append(i)
    return indices


def create_cot_prompt(tokens: List[str], punct_idx: int, context_size: int = 20) -> str:
    """Create chain-of-thought prompt for a single punctuation mark."""
    start = max(0, punct_idx - context_size)
    end = min(len(tokens), punct_idx + context_size + 1)
    
    before = " ".join(tokens[start:punct_idx])
    punct = tokens[punct_idx]
    after = " ".join(tokens[punct_idx + 1:end]) if punct_idx + 1 < end else ""
    
    # Get next word if available
    next_word = tokens[punct_idx + 1] if punct_idx + 1 < len(tokens) else "END"
    
    prompt = f"""Analyze if this punctuation mark ends a sentence in Italian text.

Context before: "{before}"
Punctuation: [{punct}]
Context after: "{after}"
Next word: "{next_word}"

Think step by step:
1. What type of punctuation is this?
2. Is the text before a complete thought?
3. Does the next word start a new sentence (capital letter, new subject)?
4. Could this be part of an abbreviation or special construction?

After reasoning, answer with FINAL: YES or FINAL: NO

Your analysis:"""
    return prompt


def parse_cot_response(response: str) -> int:
    """Parse CoT response to extract final answer."""
    response = response.upper()
    
    # Look for explicit FINAL answer
    final_match = re.search(r'FINAL:\s*(YES|NO)', response)
    if final_match:
        return 1 if final_match.group(1) == "YES" else 0
    
    # Look for answer at end
    if response.strip().endswith("YES"):
        return 1
    if response.strip().endswith("NO"):
        return 0
    
    # Count YES/NO occurrences in conclusion
    last_100 = response[-100:]
    yes_count = last_100.count("YES")
    no_count = last_100.count("NO")
    
    if yes_count > no_count:
        return 1
    return 0


def run_strategy_local(
    model_key: str,
    tokens: List[str],
    labels: List[int],
    batch_size: int = 8,
) -> List[int]:
    """Run strategy 6 with local model."""
    from model_calls.local_models import LocalModelInference

    predictions = [0] * len(tokens)
    punct_indices = get_punctuation_indices(tokens)
    print(f"  Found {len(punct_indices)} punctuation marks for CoT analysis")
    
    if len(punct_indices) == 0:
        return predictions
    
    prompts = []
    for idx in punct_indices:
        prompt = create_cot_prompt(tokens, idx)
        prompts.append(prompt)
    
    model = LocalModelInference(model_key=model_key)
    
    try:
        responses = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="CoT Reasoning"):
            batch = prompts[i:i + batch_size]
            # More tokens for reasoning
            batch_responses = model.generate_batch(batch, max_new_tokens=200, batch_size=len(batch))
            responses.extend(batch_responses)
    finally:
        model.cleanup()
    
    for idx, response in zip(punct_indices, responses):
        predictions[idx] = parse_cot_response(response)
    
    return predictions


def run_strategy_openrouter(
    model_key: str,
    tokens: List[str],
    labels: List[int],
) -> List[int]:
    """Run strategy 6 with OpenRouter API."""
    from model_calls.openrouter_api import OpenRouterClient
    
    predictions = [0] * len(tokens)
    punct_indices = get_punctuation_indices(tokens)
    print(f"  Found {len(punct_indices)} punctuation marks for CoT analysis")
    
    if len(punct_indices) == 0:
        return predictions
    
    client = OpenRouterClient()
    
    for idx in tqdm(punct_indices, desc="CoT Reasoning"):
        prompt = create_cot_prompt(tokens, idx)
        
        try:
            response = client.generate(
                model=model_key,
                prompt=prompt,
                max_tokens=250,
                temperature=0.1
            )
            predictions[idx] = parse_cot_response(response)
        except Exception as e:
            print(f"Error at index {idx}: {e}")
    
    return predictions


def run_experiment(
    model_key: str,
    dataset: str = "dev",
    api: str = "auto",
    output_base_dir: str = "results",
) -> Dict:
    """Run strategy 6 experiment."""
    paths = get_data_paths()
    
    if dataset == "ood":
        tokens, labels = load_ood_data(paths['ood'])
    elif dataset == "train":
        tokens, labels = load_manzoni_data(paths['train'])
    else:
        tokens, labels = load_manzoni_data(paths['dev'])
    
    print(f"\nStrategy 6: Chain-of-Thought Reasoning")
    print(f"Model: {model_key}")
    print(f"Dataset: {dataset} ({len(tokens)} tokens)")
    
    if api == "auto":
        local_models = ["llama-3.1-1b", "llama-3.1-3b", "llama-3.1-8b", "qwen3-8b"]
        api = "local" if model_key in local_models else "openrouter"
    
    if api == "local":
        predictions = run_strategy_local(model_key, tokens, labels)
    else:
        predictions = run_strategy_openrouter(model_key, tokens, labels)
    
    metrics = evaluate_predictions(labels, predictions)
    print_metrics(metrics, model_key, f"strategy_{STRATEGY_ID}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, output_base_dir, dataset, f"{STRATEGY_NAME}_{STRATEGY_ID}")
    os.makedirs(output_dir, exist_ok=True)
    
    model_name_safe = model_key.replace("/", "_").replace(":", "_").replace("-", "_")
    output_path = os.path.join(output_dir, f"{model_name_safe}.csv")
    save_predictions(tokens, labels, predictions, output_path)
    
    return {"model": model_key, "dataset": dataset, "metrics": metrics}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama-3.1-1b")
    parser.add_argument("--dataset", type=str, default="dev", choices=["dev", "train", "ood"])
    parser.add_argument("--api", type=str, default="auto", choices=["auto", "local", "openrouter"])
    
    args = parser.parse_args()
    run_experiment(args.model, args.dataset, args.api)
