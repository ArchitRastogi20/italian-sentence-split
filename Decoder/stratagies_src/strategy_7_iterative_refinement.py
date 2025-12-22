"""
Strategy 7: Iterative Refinement / Self-Correction

Two-pass approach:
1. First pass: Make initial predictions
2. Second pass: Verify and correct predictions with additional context
"""

import os
from typing import List, Dict, Tuple
from tqdm import tqdm

from stratagies_src.utils import (
    load_manzoni_data,
    load_ood_data,
    evaluate_predictions,
    save_predictions,
    print_metrics,
    get_data_paths,
)
from archieved_prompts.prompts import parse_model_output

STRATEGY_ID = 7
STRATEGY_NAME = "iterative_refinement"


def create_initial_prompt(tokens: List[str]) -> str:
    """Create prompt for first pass prediction."""
    token_str = " ".join(tokens)
    
    prompt = f"""Italian sentence boundary detection.

Tokens: {token_str}

For each token, output 1 if it ENDS a sentence, else 0.
Output comma-separated values ({len(tokens)} total):"""
    return prompt


def create_verification_prompt(
    tokens: List[str],
    predictions: List[int],
    focus_indices: List[int],
    context_size: int = 10,
) -> str:
    """Create prompt for verification pass."""
    
    # Build verification examples
    examples = []
    for idx in focus_indices[:5]:  # Limit to 5 examples
        start = max(0, idx - context_size)
        end = min(len(tokens), idx + context_size + 1)
        
        context = " ".join(tokens[start:end])
        token = tokens[idx]
        pred = predictions[idx]
        pred_str = "BOUNDARY" if pred == 1 else "NOT boundary"
        
        examples.append(f"  Token '{token}' at position {idx}: predicted {pred_str}")
        examples.append(f"  Context: \"{context}\"")
        examples.append("")
    
    examples_str = "\n".join(examples)
    
    prompt = f"""Verify these sentence boundary predictions for Italian text.

Predictions to verify:
{examples_str}

For each position listed above, is the prediction CORRECT or INCORRECT?

Rules:
- Period (.) usually ends sentences except in abbreviations
- Semicolon (;) rarely ends sentences in Italian
- Question mark (?) and exclamation mark (!) end sentences
- Check if next word starts a new thought

Output format: For each position, output the index and corrected label.
Example: "idx:label" like "45:1" or "45:0"

Your corrections (only list positions that need correction):"""
    return prompt


def parse_verification_response(response: str, predictions: List[int]) -> List[int]:
    """Parse verification response and update predictions."""
    import re
    
    # Look for patterns like "45:1" or "position 45: 1"
    pattern = r'(\d+)\s*[:=]\s*([01])'
    matches = re.findall(pattern, response)
    
    corrected = predictions.copy()
    for idx_str, label_str in matches:
        idx = int(idx_str)
        label = int(label_str)
        if 0 <= idx < len(corrected):
            corrected[idx] = label
    
    return corrected


def get_uncertain_indices(tokens: List[str], predictions: List[int]) -> List[int]:
    """Find indices that might need verification."""
    uncertain = []
    
    for i, (token, pred) in enumerate(zip(tokens, predictions)):
        # Check punctuation predictions
        if token in {'.', '!', '?', ';', ':'}:
            uncertain.append(i)
        # Check if prediction seems inconsistent with token
        elif pred == 1 and token not in {'.', '!', '?', '...', '…'}:
            uncertain.append(i)
    
    return uncertain


def run_strategy_local(
    model_key: str,
    tokens: List[str],
    labels: List[int],
    chunk_size: int = 100,
    batch_size: int = 8,
) -> List[int]:
    """Run strategy 7 with local model."""
    from model_calls.local_models import LocalModelInference
    
    predictions = [0] * len(tokens)
    
    # Pass 1: Initial predictions
    print("  Pass 1: Initial predictions")
    num_chunks = (len(tokens) + chunk_size - 1) // chunk_size
    
    prompts = []
    chunk_ranges = []
    for i in range(0, len(tokens), chunk_size):
        end = min(i + chunk_size, len(tokens))
        chunk_tokens = tokens[i:end]
        prompts.append(create_initial_prompt(chunk_tokens))
        chunk_ranges.append((i, end))
    
    model = LocalModelInference(model_key=model_key)
    
    try:
        # Pass 1
        responses = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Pass 1"):
            batch = prompts[i:i + batch_size]
            batch_responses = model.generate_batch(batch, max_new_tokens=250, batch_size=len(batch))
            responses.extend(batch_responses)
        
        for (start, end), response in zip(chunk_ranges, responses):
            chunk_len = end - start
            chunk_preds = parse_model_output(response, chunk_len)
            predictions[start:end] = chunk_preds
        
        # Pass 2: Verification
        print("  Pass 2: Verification")
        uncertain = get_uncertain_indices(tokens, predictions)
        print(f"    Found {len(uncertain)} positions to verify")
        
        if uncertain:
            # Process verification in batches
            verify_prompts = []
            verify_ranges = []
            
            batch_indices = []
            for i in range(0, len(uncertain), 10):
                batch_idx = uncertain[i:i + 10]
                batch_indices.append(batch_idx)
                
                # Determine chunk for this batch
                min_idx = min(batch_idx)
                max_idx = max(batch_idx)
                start = max(0, min_idx - 20)
                end = min(len(tokens), max_idx + 20)
                
                prompt = create_verification_prompt(tokens, predictions, batch_idx)
                verify_prompts.append(prompt)
                verify_ranges.append(batch_idx)
            
            verify_responses = []
            for i in tqdm(range(0, len(verify_prompts), batch_size), desc="Pass 2"):
                batch = verify_prompts[i:i + batch_size]
                batch_responses = model.generate_batch(batch, max_new_tokens=200, batch_size=len(batch))
                verify_responses.extend(batch_responses)
            
            for response in verify_responses:
                predictions = parse_verification_response(response, predictions)
    
    finally:
        model.cleanup()
    
    return predictions


def run_strategy_openrouter(
    model_key: str,
    tokens: List[str],
    labels: List[int],
    chunk_size: int = 100,
) -> List[int]:
    """Run strategy 7 with OpenRouter API."""
    from model_calls.openrouter_api import OpenRouterClient
    
    predictions = [0] * len(tokens)
    client = OpenRouterClient()
    
    # Pass 1: Initial predictions
    print("  Pass 1: Initial predictions")
    num_chunks = (len(tokens) + chunk_size - 1) // chunk_size
    
    for i in tqdm(range(0, len(tokens), chunk_size), desc="Pass 1", total=num_chunks):
        end = min(i + chunk_size, len(tokens))
        chunk_tokens = tokens[i:end]
        prompt = create_initial_prompt(chunk_tokens)
        
        try:
            response = client.generate(
                model=model_key,
                prompt=prompt,
                max_tokens=250,
                temperature=0.1
            )
            chunk_len = end - i
            chunk_preds = parse_model_output(response, chunk_len)
            predictions[i:end] = chunk_preds
        except Exception as e:
            print(f"Error at chunk {i}: {e}")
    
    # Pass 2: Verification
    print("  Pass 2: Verification")
    uncertain = get_uncertain_indices(tokens, predictions)
    print(f"    Found {len(uncertain)} positions to verify")
    
    for i in tqdm(range(0, len(uncertain), 10), desc="Pass 2"):
        batch_idx = uncertain[i:i + 10]
        prompt = create_verification_prompt(tokens, predictions, batch_idx)
        
        try:
            response = client.generate(
                model=model_key,
                prompt=prompt,
                max_tokens=200,
                temperature=0.1
            )
            predictions = parse_verification_response(response, predictions)
        except Exception as e:
            print(f"Verification error: {e}")
    
    return predictions


def run_experiment(
    model_key: str,
    dataset: str = "dev",
    api: str = "auto",
    output_base_dir: str = "results",
) -> Dict:
    """Run strategy 7 experiment."""
    paths = get_data_paths()
    
    if dataset == "ood":
        tokens, labels = load_ood_data(paths['ood'])
    elif dataset == "train":
        tokens, labels = load_manzoni_data(paths['train'])
    else:
        tokens, labels = load_manzoni_data(paths['dev'])
    
    print(f"\nStrategy 7: Iterative Refinement")
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
