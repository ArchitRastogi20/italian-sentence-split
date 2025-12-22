"""
Run all strategies for sentence splitting.

Usage:
    python run_strategies.py --strategy 1 --model llama-3.1-1b --dataset dev
    python run_strategies.py --all --dataset dev
    python run_strategies.py --all --all-datasets
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict

# Strategy modules
STRATEGIES = {
    1: ("sliding_window", "strategy_1_sliding_window"),
    2: ("next_token_prob", "strategy_2_next_token_prob"),
    3: ("marker_insertion", "strategy_3_marker_insertion"),
    4: ("structured_json", "strategy_4_structured_json"),
    5: ("few_shot_hard", "strategy_5_few_shot_hard"),
    6: ("chain_of_thought", "strategy_6_chain_of_thought"),
    7: ("iterative_refinement", "strategy_7_iterative_refinement"),
}

# LOCAL_MODELS = ["llama-3.1-1b", "llama-3.1-3b"]
LOCAL_MODELS = ["llama-3.1-8b", "qwen3-8b"]
OPENROUTER_MODELS = ["gpt-oss-20b", "kimi-k2"]
ALL_MODELS = LOCAL_MODELS + OPENROUTER_MODELS

DATASETS = ["dev", "ood"]


def run_strategy(strategy_id: int, model: str, dataset: str, api: str = "auto") -> Dict:
    """Run a single strategy."""
    if strategy_id not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_id}")
    
    name, module_name = STRATEGIES[strategy_id]
    
    # Import strategy module
    module = __import__(module_name)
    
    # Run experiment
    result = module.run_experiment(
        model_key=model,
        dataset=dataset,
        api=api,
    )
    
    return result


def run_all_strategies(
    models: List[str] = None,
    datasets: List[str] = None,
    strategies: List[int] = None,
) -> List[Dict]:
    """Run all combinations of strategies, models, and datasets."""
    if models is None:
        models = LOCAL_MODELS  # Default to local models only
    if datasets is None:
        datasets = DATASETS
    if strategies is None:
        strategies = list(STRATEGIES.keys())
    
    results = []
    total = len(strategies) * len(models) * len(datasets)
    current = 0
    
    for strategy_id in strategies:
        for model in models:
            for dataset in datasets:
                current += 1
                name, _ = STRATEGIES[strategy_id]
                print(f"\n{'='*60}")
                print(f"[{current}/{total}] Strategy {strategy_id} ({name}) - {model} - {dataset}")
                print(f"{'='*60}")
                
                try:
                    result = run_strategy(strategy_id, model, dataset)
                    result["strategy_id"] = strategy_id
                    result["strategy_name"] = name
                    results.append(result)
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    results.append({
                        "strategy_id": strategy_id,
                        "strategy_name": name,
                        "model": model,
                        "dataset": dataset,
                        "error": str(e),
                    })
    
    return results


def print_summary(results: List[Dict]):
    """Print summary table of results."""
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    print(f"{'Strategy':<25} {'Model':<18} {'Dataset':<8} {'F1':<10} {'Prec':<10} {'Recall':<10}")
    print("-"*100)
    
    for r in results:
        if "error" in r:
            print(f"{r.get('strategy_name', 'unknown'):<25} {r['model']:<18} {r['dataset']:<8} ERROR")
        else:
            m = r["metrics"]
            print(f"{r['strategy_name']:<25} {r['model']:<18} {r['dataset']:<8} {m['f1']:.4f}    {m['precision']:.4f}    {m['recall']:.4f}")
    
    print("="*100)


def save_summary(results: List[Dict], output_dir: str = "results"):
    """Save summary to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"strategies_summary_{timestamp}.json")
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nSummary saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Run sentence splitting strategies")
    
    parser.add_argument("--strategy", type=int, choices=list(STRATEGIES.keys()),
                       help="Strategy ID (1-7)")
    parser.add_argument("--model", type=str, default="llama-3.1-1b",
                       help="Model to use")
    parser.add_argument("--dataset", type=str, default="dev",
                       choices=["dev", "train", "ood"],
                       help="Dataset to use")
    parser.add_argument("--api", type=str, default="auto",
                       choices=["auto", "local", "openrouter"],
                       help="API to use")
    parser.add_argument("--all", action="store_true",
                       help="Run all strategies")
    parser.add_argument("--all-datasets", action="store_true",
                       help="Run on all datasets (dev, ood)")
    parser.add_argument("--all-models", action="store_true",
                       help="Run all models (local + openrouter)")
    parser.add_argument("--local-only", action="store_true",
                       help="Run only local models")
    parser.add_argument("--strategies", type=int, nargs="+",
                       help="List of strategy IDs to run")
    parser.add_argument("--models", type=str, nargs="+",
                       help="List of models to run")
    
    args = parser.parse_args()
    
    if args.all:
        # Determine which models to use
        if args.all_models:
            models = ALL_MODELS
        elif args.local_only:
            models = LOCAL_MODELS
        elif args.models:
            models = args.models
        else:
            models = LOCAL_MODELS
        
        # Determine datasets
        datasets = DATASETS if args.all_datasets else [args.dataset]
        
        # Determine strategies
        strategies = args.strategies if args.strategies else list(STRATEGIES.keys())
        
        print(f"Running strategies: {strategies}")
        print(f"Models: {models}")
        print(f"Datasets: {datasets}")
        
        results = run_all_strategies(
            models=models,
            datasets=datasets,
            strategies=strategies,
        )
        
        print_summary(results)
        save_summary(results)
    
    elif args.strategy:
        result = run_strategy(args.strategy, args.model, args.dataset, args.api)
        print(f"\nResult: {json.dumps(result, indent=2, default=str)}")
    
    else:
        parser.print_help()
        print("\nAvailable strategies:")
        for sid, (name, _) in STRATEGIES.items():
            print(f"  {sid}: {name}")


if __name__ == "__main__":
    main()
