"""
Strategy 5: Few-Shot Learning with Hard Examples

Provide carefully curated examples covering edge cases.
Model learns pattern from examples.
"""

import os
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
from archieved_prompts.prompts import parse_model_output

STRATEGY_ID = 5
STRATEGY_NAME = "few_shot_hard"

# Hard examples covering edge cases
FEW_SHOT_EXAMPLES = """
### Example 1: Abbreviation (NOT a boundary) ###
Tokens: S. Maria della Stella era
Labels: 0,0,0,0,0,0
Explanation: "S." is an abbreviation for "Santa", not a sentence end.

### Example 2: True sentence boundary ###
Tokens: in nuovi seni . La costiera
Labels: 0,0,0,1,0,0
Explanation: Period after "seni" ends the sentence, "La" starts a new one.

### Example 3: Colon before speech (NOT a boundary) ###
Tokens: disse : – Non mi
Labels: 0,0,0,0,0
Explanation: Colon introduces speech, not a sentence end.

### Example 4: Semicolon in literary text (NOT a boundary) ###
Tokens: era schermito ; però con
Labels: 0,0,0,0,0
Explanation: Semicolon separates clauses but not sentences in Italian literature.

### Example 5: Question mark (IS a boundary) ###
Tokens: Al sagrestano gli crede ? – Perche
Labels: 0,0,0,0,1,0,0
Explanation: Question mark ends the interrogative sentence.

### Example 6: Exclamation in dialogue (IS a boundary) ###
Tokens: – Un re ! – diranno
Labels: 0,0,0,1,0,0
Explanation: Exclamation ends the exclamatory sentence.

### Example 7: Ellipsis (context dependent) ###
Tokens: C' era una volta ... –
Labels: 0,0,0,0,0,0
Explanation: Ellipsis here indicates pause, not sentence end.

### Example 8: Period after short word (check context) ###
Tokens: dott. Azzecca - garbugli disse
Labels: 0,0,0,0,0
Explanation: "dott." is abbreviation for "dottore".

### Example 9: Multiple sentences ###
Tokens: la forza . In marzo ,
Labels: 0,0,1,0,0,0
Explanation: Period ends first sentence, "In" starts new one.

### Example 10: Comma (NEVER a boundary) ###
Tokens: dal deposito , scende appoggiata
Labels: 0,0,0,0,0
Explanation: Commas never end sentences.
"""


def create_few_shot_prompt(tokens: List[str]) -> str:
    """Create few-shot prompt with hard examples."""
    token_str = " ".join(tokens)
    
    prompt = f"""You are an expert Italian sentence boundary detector.

Learn from these examples:
{FEW_SHOT_EXAMPLES}

Now classify the following tokens. For each token, output 1 if it ENDS a sentence, otherwise 0.

Tokens: {token_str}

Output ONLY comma-separated 0s and 1s ({len(tokens)} values total):"""
    return prompt


def run_strategy_local(
    model_key: str,
    tokens: List[str],
    labels: List[int],
    chunk_size: int = 100,
    batch_size: int = 8,
) -> List[int]:
    """Run strategy 5 with local model."""
    from model_calls.local_models import LocalModelInference
    
    predictions = [0] * len(tokens)
    
    num_chunks = (len(tokens) + chunk_size - 1) // chunk_size
    print(f"  Processing {num_chunks} chunks with few-shot examples")
    
    prompts = []
    chunk_ranges = []
    for i in range(0, len(tokens), chunk_size):
        end = min(i + chunk_size, len(tokens))
        chunk_tokens = tokens[i:end]
        prompts.append(create_few_shot_prompt(chunk_tokens))
        chunk_ranges.append((i, end))
    
    model = LocalModelInference(model_key=model_key)
    
    try:
        responses = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch = prompts[i:i + batch_size]
            batch_responses = model.generate_batch(batch, max_new_tokens=300, batch_size=len(batch))
            responses.extend(batch_responses)
    finally:
        model.cleanup()
    
    for (start, end), response in zip(chunk_ranges, responses):
        chunk_len = end - start
        chunk_preds = parse_model_output(response, chunk_len)
        predictions[start:end] = chunk_preds
    
    return predictions


def run_strategy_openrouter(
    model_key: str,
    tokens: List[str],
    labels: List[int],
    chunk_size: int = 100,
) -> List[int]:
    """Run strategy 5 with OpenRouter API."""
    from model_calls.openrouter_api import OpenRouterClient

    
    predictions = [0] * len(tokens)
    client = OpenRouterClient()
    
    num_chunks = (len(tokens) + chunk_size - 1) // chunk_size
    
    for i in tqdm(range(0, len(tokens), chunk_size), desc="Generating", total=num_chunks):
        end = min(i + chunk_size, len(tokens))
        chunk_tokens = tokens[i:end]
        prompt = create_few_shot_prompt(chunk_tokens)
        
        try:
            response = client.generate(
                model=model_key,
                prompt=prompt,
                max_tokens=300,
                temperature=0.1
            )
            chunk_len = end - i
            chunk_preds = parse_model_output(response, chunk_len)
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
    """Run strategy 5 experiment."""
    paths = get_data_paths()
    
    if dataset == "ood":
        tokens, labels = load_ood_data(paths['ood'])
    elif dataset == "train":
        tokens, labels = load_manzoni_data(paths['train'])
    else:
        tokens, labels = load_manzoni_data(paths['dev'])
    
    print(f"\nStrategy 5: Few-Shot Learning with Hard Examples")
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
