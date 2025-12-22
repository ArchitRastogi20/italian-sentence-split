"""
Strategy 3: Full Sequence Marker Insertion

Ask LLM to rewrite text with <EOS> markers at sentence boundaries.
Then align markers back to original tokens.
"""

import os
import re
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

STRATEGY_ID = 3
STRATEGY_NAME = "marker_insertion"

EOS_MARKER = "<EOS>"


def create_marker_prompt(tokens: List[str]) -> str:
    """Create prompt asking model to insert EOS markers."""
    text = " ".join(tokens)
    prompt = f"""You are a sentence boundary detector for Italian text.

Task: Rewrite the following text, inserting {EOS_MARKER} marker immediately after each sentence-ending token.

Rules:
- Insert {EOS_MARKER} after periods that end sentences (not abbreviations)
- Insert {EOS_MARKER} after question marks and exclamation marks
- Do NOT insert {EOS_MARKER} after commas or semicolons
- Do NOT add or remove any words
- Keep all original punctuation

Input text:
{text}

Output (same text with {EOS_MARKER} markers):"""
    return prompt


def align_markers_to_tokens(
    original_tokens: List[str],
    marked_text: str,
) -> List[int]:
    """
    Align EOS markers in output back to original token positions.
    Returns list of 0/1 labels.
    """
    predictions = [0] * len(original_tokens)
    
    # Clean the marked text
    marked_text = marked_text.strip()
    
    # Find all EOS marker positions
    marker_pattern = re.compile(re.escape(EOS_MARKER), re.IGNORECASE)
    
    # Remove markers and track positions
    text_parts = marker_pattern.split(marked_text)
    
    # Reconstruct and find which tokens precede markers
    current_pos = 0
    token_idx = 0
    
    for i, part in enumerate(text_parts):
        part = part.strip()
        if not part:
            continue
        
        # Tokenize this part (simple whitespace split)
        part_tokens = part.split()
        
        # Match tokens
        for pt in part_tokens:
            # Find matching original token
            while token_idx < len(original_tokens):
                orig = original_tokens[token_idx]
                # Check if tokens match (allowing for some flexibility)
                if pt == orig or pt.strip('.,;:!?"\'-') == orig.strip('.,;:!?"\'-'):
                    token_idx += 1
                    break
                token_idx += 1
        
        # If there's a marker after this part (not the last part)
        if i < len(text_parts) - 1 and token_idx > 0:
            # Mark the previous token as boundary
            predictions[token_idx - 1] = 1
    
    return predictions


def align_markers_simple(
    original_tokens: List[str],
    marked_text: str,
) -> List[int]:
    """
    Simpler alignment: find tokens followed by EOS marker.
    """
    predictions = [0] * len(original_tokens)
    marked_text = marked_text.strip()
    
    # Look for patterns like "token <EOS>" or "token<EOS>"
    for i, token in enumerate(original_tokens):
        # Check if this token appears followed by EOS marker
        patterns = [
            f"{token} {EOS_MARKER}",
            f"{token}{EOS_MARKER}",
            f"{token} {EOS_MARKER.lower()}",
            f"{token}{EOS_MARKER.lower()}",
        ]
        
        for pattern in patterns:
            if pattern in marked_text:
                predictions[i] = 1
                break
    
    return predictions


def run_strategy_local(
    model_key: str,
    tokens: List[str],
    labels: List[int],
    chunk_size: int = 200,
    batch_size: int = 8,
) -> List[int]:
    """Run strategy 3 with local model."""
    from model_calls.local_models import LocalModelInference
    
    predictions = [0] * len(tokens)
    
    # Process in chunks
    num_chunks = (len(tokens) + chunk_size - 1) // chunk_size
    print(f"  Processing {num_chunks} chunks")
    
    # Create prompts for each chunk
    prompts = []
    chunk_ranges = []
    for i in range(0, len(tokens), chunk_size):
        end = min(i + chunk_size, len(tokens))
        chunk_tokens = tokens[i:end]
        prompts.append(create_marker_prompt(chunk_tokens))
        chunk_ranges.append((i, end))
    
    model = LocalModelInference(model_key=model_key)
    
    try:
        responses = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch = prompts[i:i + batch_size]
            batch_responses = model.generate_batch(batch, max_new_tokens=500, batch_size=len(batch))
            responses.extend(batch_responses)
    finally:
        model.cleanup()
    
    # Align markers for each chunk
    for (start, end), response in zip(chunk_ranges, responses):
        chunk_tokens = tokens[start:end]
        chunk_preds = align_markers_simple(chunk_tokens, response)
        predictions[start:end] = chunk_preds
    
    return predictions


def run_strategy_openrouter(
    model_key: str,
    tokens: List[str],
    labels: List[int],
    chunk_size: int = 200,
) -> List[int]:
    """Run strategy 3 with OpenRouter API."""
    from model_calls.openrouter_api import OpenRouterClient
    
    predictions = [0] * len(tokens)
    client = OpenRouterClient()
    
    num_chunks = (len(tokens) + chunk_size - 1) // chunk_size
    
    for i in tqdm(range(0, len(tokens), chunk_size), desc="Generating", total=num_chunks):
        end = min(i + chunk_size, len(tokens))
        chunk_tokens = tokens[i:end]
        prompt = create_marker_prompt(chunk_tokens)
        
        try:
            response = client.generate(
                model=model_key,
                prompt=prompt,
                max_tokens=600,
                temperature=0.1
            )
            chunk_preds = align_markers_simple(chunk_tokens, response)
            predictions[i:end] = chunk_preds
        except Exception as e:
            print(f"Error at chunk {i}: {e}")
    
    return predictions


def run_experiment(
    model_key: str,
    dataset: str = "dev",
    api: str = "auto",
    output_base_dir: str = "results",
) -> Dict:
    """Run strategy 3 experiment."""
    paths = get_data_paths()
    
    if dataset == "ood":
        tokens, labels = load_ood_data(paths['ood'])
    elif dataset == "train":
        tokens, labels = load_manzoni_data(paths['train'])
    else:
        tokens, labels = load_manzoni_data(paths['dev'])
    
    print(f"\nStrategy 3: Marker Insertion")
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
