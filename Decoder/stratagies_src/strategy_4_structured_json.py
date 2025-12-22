"""
Strategy 4: Structured JSON/List Output Generation

Ask LLM to output structured format - list of 0/1 labels aligned to tokens.
"""

import os
import json
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

STRATEGY_ID = 4
STRATEGY_NAME = "structured_json"


def create_json_prompt(tokens: List[str]) -> str:
    """Create prompt asking for JSON output with boundary indices."""
    # Number the tokens
    numbered = [f"{i}:{t}" for i, t in enumerate(tokens)]
    token_str = " ".join(numbered)
    
    prompt = f"""You are a sentence boundary detector for Italian text.

Task: Identify which tokens END a sentence.

Tokens (numbered):
{token_str}

Output a JSON object with a single key "boundaries" containing a list of token indices that end sentences.

Rules:
- Include index of period (.) that ends a sentence (not abbreviations)
- Include index of question mark (?) and exclamation mark (!)
- Do NOT include commas (,) or semicolons (;)
- Only include indices from 0 to {len(tokens) - 1}

Output ONLY valid JSON, nothing else:"""
    return prompt


def create_list_prompt(tokens: List[str]) -> str:
    """Create prompt asking for comma-separated list of 0/1."""
    token_str = " ".join(tokens)
    
    prompt = f"""Sentence boundary detection for Italian text.

Tokens: {token_str}

Output: For each of the {len(tokens)} tokens, output 1 if it ENDS a sentence, else 0.
Return ONLY a comma-separated list of {len(tokens)} values (0 or 1).

Example format: 0,0,0,1,0,0,1,0,0,1

Your output:"""
    return prompt


def parse_json_response(response: str, num_tokens: int) -> List[int]:
    """Parse JSON response to list of predictions."""
    predictions = [0] * num_tokens
    
    # Try to extract JSON
    response = response.strip()
    
    # Look for JSON object
    json_match = re.search(r'\{[^}]+\}', response)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "boundaries" in data:
                for idx in data["boundaries"]:
                    if isinstance(idx, int) and 0 <= idx < num_tokens:
                        predictions[idx] = 1
                return predictions
        except json.JSONDecodeError:
            pass
    
    # Try to find array of numbers
    array_match = re.search(r'\[[\d,\s]+\]', response)
    if array_match:
        try:
            indices = json.loads(array_match.group())
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < num_tokens:
                    predictions[idx] = 1
            return predictions
        except json.JSONDecodeError:
            pass
    
    # Try to find numbers directly
    numbers = re.findall(r'\b(\d+)\b', response)
    for num_str in numbers:
        idx = int(num_str)
        if 0 <= idx < num_tokens:
            predictions[idx] = 1
    
    return predictions


def parse_list_response(response: str, num_tokens: int) -> List[int]:
    """Parse comma-separated list response."""
    predictions = [0] * num_tokens
    
    response = response.strip()
    
    # Remove any non-digit, non-comma characters at start/end
    response = re.sub(r'^[^01]+', '', response)
    response = re.sub(r'[^01]+$', '', response)
    
    # Split by comma or space
    parts = re.split(r'[,\s]+', response)
    
    for i, part in enumerate(parts):
        if i >= num_tokens:
            break
        part = part.strip()
        if part == '1':
            predictions[i] = 1
        elif part == '0':
            predictions[i] = 0
    
    return predictions


def run_strategy_local(
    model_key: str,
    tokens: List[str],
    labels: List[int],
    chunk_size: int = 150,
    batch_size: int = 8,
    use_json: bool = True,
) -> List[int]:
    """Run strategy 4 with local model."""
    from model_calls.local_models import LocalModelInference
    
    predictions = [0] * len(tokens)
    
    num_chunks = (len(tokens) + chunk_size - 1) // chunk_size
    print(f"  Processing {num_chunks} chunks (JSON mode: {use_json})")
    
    prompts = []
    chunk_ranges = []
    for i in range(0, len(tokens), chunk_size):
        end = min(i + chunk_size, len(tokens))
        chunk_tokens = tokens[i:end]
        if use_json:
            prompts.append(create_json_prompt(chunk_tokens))
        else:
            prompts.append(create_list_prompt(chunk_tokens))
        chunk_ranges.append((i, end))
    
    model = LocalModelInference(model_key=model_key)
    
    try:
        responses = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch = prompts[i:i + batch_size]
            batch_responses = model.generate_batch(batch, max_new_tokens=400, batch_size=len(batch))
            responses.extend(batch_responses)
    finally:
        model.cleanup()
    
    for (start, end), response in zip(chunk_ranges, responses):
        chunk_len = end - start
        if use_json:
            chunk_preds = parse_json_response(response, chunk_len)
        else:
            chunk_preds = parse_list_response(response, chunk_len)
        predictions[start:end] = chunk_preds
    
    return predictions


def run_strategy_openrouter(
    model_key: str,
    tokens: List[str],
    labels: List[int],
    chunk_size: int = 150,
    use_json: bool = True,
) -> List[int]:
    """Run strategy 4 with OpenRouter API."""
    from model_calls.openrouter_api import OpenRouterClient
    
    predictions = [0] * len(tokens)
    client = OpenRouterClient()
    
    num_chunks = (len(tokens) + chunk_size - 1) // chunk_size
    
    for i in tqdm(range(0, len(tokens), chunk_size), desc="Generating", total=num_chunks):
        end = min(i + chunk_size, len(tokens))
        chunk_tokens = tokens[i:end]
        
        if use_json:
            prompt = create_json_prompt(chunk_tokens)
        else:
            prompt = create_list_prompt(chunk_tokens)
        
        try:
            response = client.generate(
                model=model_key,
                prompt=prompt,
                max_tokens=400,
                temperature=0.1
            )
            
            chunk_len = end - i
            if use_json:
                chunk_preds = parse_json_response(response, chunk_len)
            else:
                chunk_preds = parse_list_response(response, chunk_len)
            predictions[i:end] = chunk_preds
            
        except Exception as e:
            print(f"Error at chunk {i}: {e}")
    
    return predictions


def run_experiment(
    model_key: str,
    dataset: str = "dev",
    api: str = "auto",
    output_base_dir: str = "results",
    use_json: bool = True,
) -> Dict:
    """Run strategy 4 experiment."""
    paths = get_data_paths()
    
    if dataset == "ood":
        tokens, labels = load_ood_data(paths['ood'])
    elif dataset == "train":
        tokens, labels = load_manzoni_data(paths['train'])
    else:
        tokens, labels = load_manzoni_data(paths['dev'])
    
    print(f"\nStrategy 4: Structured JSON Output")
    print(f"Model: {model_key}")
    print(f"Dataset: {dataset} ({len(tokens)} tokens)")
    print(f"Output format: {'JSON' if use_json else 'List'}")
    
    if api == "auto":
        local_models = ["llama-3.1-1b", "llama-3.1-3b", "llama-3.1-8b", "qwen3-8b"]
        api = "local" if model_key in local_models else "openrouter"
    
    if api == "local":
        predictions = run_strategy_local(model_key, tokens, labels, use_json=use_json)
    else:
        predictions = run_strategy_openrouter(model_key, tokens, labels, use_json=use_json)
    
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
    parser.add_argument("--use-list", action="store_true", help="Use list format instead of JSON")
    
    args = parser.parse_args()
    run_experiment(args.model, args.dataset, args.api, use_json=not args.use_list)
