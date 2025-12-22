"""
Strategy 1: Sliding Window Binary Classification

For each punctuation mark, extract context window and ask YES/NO if it's a boundary.
"""

import os
import sys
from typing import List, Tuple, Dict
from tqdm import tqdm

from stratagies_src.utils import (
    load_manzoni_data,
    load_ood_data,
    evaluate_predictions,
    save_predictions,
    print_metrics,
    get_data_paths,
)

STRATEGY_ID = 1
STRATEGY_NAME = "sliding_window"


def get_punctuation_indices(tokens: List[str]) -> List[int]:
    """Find indices of punctuation marks that could be sentence boundaries."""
    punctuation = {'.', '!', '?', ';', ':', '...', '…'}
    indices = []
    for i, token in enumerate(tokens):
        if token in punctuation or (len(token) == 1 and token in '.!?;:'):
            indices.append(i)
    return indices


def create_context_window(tokens: List[str], idx: int, window_size: int = 15) -> str:
    """Create context window around a token."""
    start = max(0, idx - window_size)
    end = min(len(tokens), idx + window_size + 1)
    
    context_tokens = tokens[start:end]
    target_pos = idx - start
    
    # Mark the target token
    context_str = " ".join(context_tokens[:target_pos])
    context_str += f" [{tokens[idx]}] "
    context_str += " ".join(context_tokens[target_pos + 1:])
    
    return context_str.strip()


def create_binary_prompt(context: str, token: str) -> str:
    """Create YES/NO prompt for boundary detection."""
    prompt = f"""You are analyzing Italian text for sentence boundaries.

Context: "{context}"

The token in brackets [{token}] is a punctuation mark.

Question: Does this punctuation mark end a sentence? 

Rules:
- Period (.) usually ends sentences, but not in abbreviations like "S." or "dott."
- Semicolon (;) rarely ends sentences in Italian literary text
- Question mark (?) and exclamation mark (!) end sentences
- Consider if the next word starts a new thought

Answer with ONLY "YES" or "NO":"""
    return prompt


def parse_binary_response(response: str) -> int:
    """Parse YES/NO response to 0/1."""
    response = response.strip().upper()
    if response.startswith("YES"):
        return 1
    elif response.startswith("NO"):
        return 0
    if "YES" in response:
        return 1
    return 0


def run_strategy_local(
    model_key: str,
    tokens: List[str],
    labels: List[int],
    batch_size: int = 16,
) -> List[int]:
    """Run strategy 1 with local model."""
    from model_calls.local_models import LocalModelInference
    
    predictions = [0] * len(tokens)
    punct_indices = get_punctuation_indices(tokens)
    print(f"  Found {len(punct_indices)} punctuation marks to classify")
    
    if len(punct_indices) == 0:
        return predictions
    
    prompts = []
    for idx in punct_indices:
        context = create_context_window(tokens, idx)
        prompt = create_binary_prompt(context, tokens[idx])
        prompts.append(prompt)
    
    model = LocalModelInference(model_key=model_key)
    
    try:
        responses = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Classifying"):
            batch = prompts[i:i + batch_size]
            batch_responses = model.generate_batch(batch, max_new_tokens=10, batch_size=len(batch))
            responses.extend(batch_responses)
    finally:
        model.cleanup()
    
    for idx, response in zip(punct_indices, responses):
        predictions[idx] = parse_binary_response(response)
    
    return predictions


def run_strategy_openrouter(
    model_key: str,
    tokens: List[str],
    labels: List[int],
) -> List[int]:
    """Run strategy 1 with OpenRouter API."""
    from model_calls.openrouter_api import OpenRouterClient
    
    predictions = [0] * len(tokens)
    punct_indices = get_punctuation_indices(tokens)
    print(f"  Found {len(punct_indices)} punctuation marks to classify")
    
    if len(punct_indices) == 0:
        return predictions
    
    client = OpenRouterClient()
    
    for idx in tqdm(punct_indices, desc="Classifying"):
        context = create_context_window(tokens, idx)
        prompt = create_binary_prompt(context, tokens[idx])
        
        try:
            response = client.generate(
                model=model_key,
                prompt=prompt,
                max_tokens=10,
                temperature=0.1
            )
            predictions[idx] = parse_binary_response(response)
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            predictions[idx] = 0
    
    return predictions


def run_experiment(
    model_key: str,
    dataset: str = "dev",
    api: str = "auto",
    output_base_dir: str = "results",
) -> Dict:
    """Run strategy 1 experiment."""
    paths = get_data_paths()
    
    if dataset == "ood":
        tokens, labels = load_ood_data(paths['ood'])
    elif dataset == "train":
        tokens, labels = load_manzoni_data(paths['train'])
    else:
        tokens, labels = load_manzoni_data(paths['dev'])
    
    print(f"\nStrategy 1: Sliding Window Binary Classification")
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
