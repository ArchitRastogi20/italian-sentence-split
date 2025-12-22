"""
Strategy 2: Next-Token Probability Analysis

Analyze probability distribution over next tokens after punctuation.
High probability for sentence starters (capital letters) indicates boundary.
Only works with local models (requires logit access).
"""

import os
import torch
from typing import List, Dict
from tqdm import tqdm

from utils import (
    load_manzoni_data,
    load_ood_data,
    evaluate_predictions,
    save_predictions,
    print_metrics,
    get_data_paths,
)

STRATEGY_ID = 2
STRATEGY_NAME = "next_token_prob"

# Common Italian sentence starters (capital letters)
SENTENCE_STARTERS = [
    "Il", "La", "Lo", "Le", "Li", "I", "Gli",
    "Un", "Una", "Uno",
    "E", "Ma", "Se", "Non", "Per", "Con", "Da", "Di", "A", "In",
    "Quel", "Quella", "Questo", "Questa", "Chi", "Che", "Come", "Quando", "Dove",
    "Era", "Fu", "Si", "Ne", "Ci",
    "Al", "Alla", "Allo", "Ai", "Alle",
]


def get_punctuation_indices(tokens: List[str]) -> List[int]:
    """Find indices of punctuation marks that could be sentence boundaries."""
    punctuation = {'.', '!', '?', ';', ':', '...', '…'}
    indices = []
    for i, token in enumerate(tokens):
        if token in punctuation:
            indices.append(i)
    return indices


def analyze_next_token_probs(
    model,
    tokenizer,
    tokens: List[str],
    punct_idx: int,
    context_size: int = 50,
) -> float:
    """
    Analyze probability of sentence-starting tokens after punctuation.
    Returns a score indicating likelihood of sentence boundary.
    """
    # Build context up to and including punctuation
    start = max(0, punct_idx - context_size)
    context_tokens = tokens[start:punct_idx + 1]
    context_text = " ".join(context_tokens)
    
    # Tokenize
    inputs = tokenizer(context_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last token logits
        probs = torch.softmax(logits, dim=-1)
    
    # Get probabilities for sentence starters
    total_starter_prob = 0.0
    for starter in SENTENCE_STARTERS:
        starter_ids = tokenizer.encode(starter, add_special_tokens=False)
        if starter_ids:
            first_token_id = starter_ids[0]
            if first_token_id < len(probs):
                total_starter_prob += probs[first_token_id].item()
    
    # Also check for space + capital letter pattern
    space_ids = tokenizer.encode(" ", add_special_tokens=False)
    if space_ids:
        space_prob = probs[space_ids[0]].item() if space_ids[0] < len(probs) else 0
        total_starter_prob += space_prob * 0.5  # Weight space probability
    
    return total_starter_prob


def run_strategy_local(
    model_key: str,
    tokens: List[str],
    labels: List[int],
    threshold: float = 0.15,
) -> List[int]:
    """Run strategy 2 with local model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    predictions = [0] * len(tokens)
    punct_indices = get_punctuation_indices(tokens)
    print(f"  Found {len(punct_indices)} punctuation marks to analyze")
    
    if len(punct_indices) == 0:
        return predictions
    
    # Model mapping
    model_map = {
        "llama-3.1-1b": "meta-llama/Llama-3.2-1B-Instruct",
        "llama-3.1-3b": "meta-llama/Llama-3.2-3B-Instruct",
        "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "qwen3-8b": "Qwen/Qwen2.5-7B-Instruct",
    }
    
    model_name = model_map.get(model_key, model_key)
    print(f"  Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    try:
        scores = []
        for idx in tqdm(punct_indices, desc="Analyzing probabilities"):
            score = analyze_next_token_probs(model, tokenizer, tokens, idx)
            scores.append(score)
            
            if score > threshold:
                predictions[idx] = 1
        
        # Print score statistics
        if scores:
            print(f"  Score stats: min={min(scores):.4f}, max={max(scores):.4f}, mean={sum(scores)/len(scores):.4f}")
    
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return predictions


def run_strategy_openrouter(
    model_key: str,
    tokens: List[str],
    labels: List[int],
) -> List[int]:
    """
    Strategy 2 requires logit access - not available via OpenRouter.
    Fallback to heuristic-based approach.
    """
    print("  Warning: Strategy 2 requires logit access. Using heuristic fallback.")
    
    predictions = [0] * len(tokens)
    punct_indices = get_punctuation_indices(tokens)
    
    for idx in punct_indices:
        token = tokens[idx]
        # Simple heuristic: period, !, ? are likely boundaries
        if token in {'.', '!', '?'}:
            # Check if next token exists and starts with capital
            if idx + 1 < len(tokens):
                next_token = tokens[idx + 1]
                if next_token and next_token[0].isupper():
                    predictions[idx] = 1
                elif next_token in {'-', '–', '«', '"'}:
                    # Dialogue markers often follow sentence ends
                    predictions[idx] = 1
    
    return predictions


def run_experiment(
    model_key: str,
    dataset: str = "dev",
    api: str = "auto",
    output_base_dir: str = "results",
    threshold: float = 0.15,
) -> Dict:
    """Run strategy 2 experiment."""
    paths = get_data_paths()
    
    if dataset == "ood":
        tokens, labels = load_ood_data(paths['ood'])
    elif dataset == "train":
        tokens, labels = load_manzoni_data(paths['train'])
    else:
        tokens, labels = load_manzoni_data(paths['dev'])
    
    print(f"\nStrategy 2: Next-Token Probability Analysis")
    print(f"Model: {model_key}")
    print(f"Dataset: {dataset} ({len(tokens)} tokens)")
    print(f"Threshold: {threshold}")
    
    if api == "auto":
        local_models = ["llama-3.1-1b", "llama-3.1-3b", "llama-3.1-8b", "qwen3-8b"]
        api = "local" if model_key in local_models else "openrouter"
    
    if api == "local":
        predictions = run_strategy_local(model_key, tokens, labels, threshold)
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
    parser.add_argument("--threshold", type=float, default=0.15)
    
    args = parser.parse_args()
    run_experiment(args.model, args.dataset, args.api, threshold=args.threshold)
