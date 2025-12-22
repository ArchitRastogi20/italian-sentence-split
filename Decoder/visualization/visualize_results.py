"""
Visualization and analysis script for sentence splitting results.
Generates figures and combined CSV with all metrics.

Usage:
    python visualization/visualize_results.py
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stratagies_src.utils import evaluate_predictions

# Configuration
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# Strategy mapping
STRATEGY_NAMES = {
    1: "Sliding Window",
    2: "Next-Token Prob",
    3: "Marker Insertion",
    4: "Structured JSON",
    5: "Few-Shot Hard",
    6: "Chain-of-Thought",
    7: "Iterative Refinement",
}

STRATEGY_FOLDERS = {
    "sliding_window_1": 1,
    "next_token_prob_2": 2,
    "marker_insertion_3": 3,
    "structured_json_4": 4,
    "few_shot_hard_5": 5,
    "chain_of_thought_6": 6,
    "iterative_refinement_7": 7,
}


def load_all_results() -> pd.DataFrame:
    """Load all CSV results and compute metrics."""
    results = []
    
    for dataset in ["dev", "ood"]:
        dataset_dir = os.path.join(RESULTS_DIR, dataset)
        if not os.path.exists(dataset_dir):
            print(f"Warning: {dataset_dir} not found")
            continue
        
        # Find all strategy folders
        for strategy_folder in os.listdir(dataset_dir):
            strategy_path = os.path.join(dataset_dir, strategy_folder)
            if not os.path.isdir(strategy_path):
                continue
            
            # Get strategy ID
            strategy_id = STRATEGY_FOLDERS.get(strategy_folder)
            if strategy_id is None:
                # Try to extract from folder name
                for key, val in STRATEGY_FOLDERS.items():
                    if key in strategy_folder:
                        strategy_id = val
                        break
            
            if strategy_id is None:
                print(f"Warning: Unknown strategy folder {strategy_folder}")
                continue
            
            strategy_name = STRATEGY_NAMES.get(strategy_id, strategy_folder)
            
            # Find all model CSV files
            csv_files = glob.glob(os.path.join(strategy_path, "*.csv"))
            
            for csv_path in csv_files:
                filename = os.path.basename(csv_path)
                model_name = filename.replace(".csv", "")
                
                try:
                    df = pd.read_csv(csv_path)
                    
                    if "label_gt" not in df.columns or "prediction" not in df.columns:
                        print(f"Warning: Invalid columns in {csv_path}")
                        continue
                    
                    y_true = df["label_gt"].tolist()
                    y_pred = df["prediction"].tolist()
                    
                    metrics = evaluate_predictions(y_true, y_pred)
                    
                    results.append({
                        "model": model_name,
                        "strategy_id": strategy_id,
                        "strategy": strategy_name,
                        "dataset": dataset,
                        "accuracy": metrics["accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "file": csv_path,
                    })
                    
                except Exception as e:
                    print(f"Error processing {csv_path}: {e}")
    
    return pd.DataFrame(results)


def save_combined_csv(df: pd.DataFrame):
    """Save combined results to CSV."""
    output_path = os.path.join(OUTPUT_DIR, "combined_results.csv")
    
    # Select and order columns
    columns = ["model", "strategy", "strategy_id", "dataset", "f1", "accuracy", "precision", "recall"]
    df_out = df[columns].copy()
    
    # Sort by strategy, dataset, then f1 descending
    df_out = df_out.sort_values(["strategy_id", "dataset", "f1"], ascending=[True, True, False])
    
    df_out.to_csv(output_path, index=False)
    print(f"Saved combined results to: {output_path}")
    
    return output_path


def plot_f1_by_strategy(df: pd.DataFrame):
    """Bar plot of F1 scores grouped by strategy."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, dataset in enumerate(["dev", "ood"]):
        ax = axes[idx]
        data = df[df["dataset"] == dataset]
        
        if len(data) == 0:
            continue
        
        # Pivot for grouped bar chart
        pivot = data.pivot(index="strategy", columns="model", values="f1")
        
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title(f"F1 Score by Strategy - {dataset.upper()}", fontsize=14)
        ax.set_xlabel("Strategy", fontsize=12)
        ax.set_ylabel("F1 Score", fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, "f1_by_strategy.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_f1_by_model(df: pd.DataFrame):
    """Bar plot of F1 scores grouped by model."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, dataset in enumerate(["dev", "ood"]):
        ax = axes[idx]
        data = df[df["dataset"] == dataset]
        
        if len(data) == 0:
            continue
        
        pivot = data.pivot(index="model", columns="strategy", values="f1")
        
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title(f"F1 Score by Model - {dataset.upper()}", fontsize=14)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("F1 Score", fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, "f1_by_model.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_heatmap(df: pd.DataFrame):
    """Heatmap of F1 scores (model x strategy)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, dataset in enumerate(["dev", "ood"]):
        ax = axes[idx]
        data = df[df["dataset"] == dataset]
        
        if len(data) == 0:
            continue
        
        pivot = data.pivot(index="model", columns="strategy", values="f1")
        
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"label": "F1 Score"}
        )
        ax.set_title(f"F1 Score Heatmap - {dataset.upper()}", fontsize=14)
        ax.set_xlabel("Strategy", fontsize=12)
        ax.set_ylabel("Model", fontsize=12)
        ax.tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, "f1_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_precision_recall(df: pd.DataFrame):
    """Scatter plot of precision vs recall."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, dataset in enumerate(["dev", "ood"]):
        ax = axes[idx]
        data = df[df["dataset"] == dataset]
        
        if len(data) == 0:
            continue
        
        # Color by strategy
        strategies = data["strategy"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
        
        for strategy, color in zip(strategies, colors):
            strat_data = data[data["strategy"] == strategy]
            ax.scatter(
                strat_data["recall"],
                strat_data["precision"],
                c=[color],
                label=strategy,
                s=100,
                alpha=0.7
            )
            
            # Add model labels
            for _, row in strat_data.iterrows():
                ax.annotate(
                    row["model"][:10],
                    (row["recall"], row["precision"]),
                    fontsize=6,
                    alpha=0.7
                )
        
        ax.set_title(f"Precision vs Recall - {dataset.upper()}", fontsize=14)
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
        
        # Add F1 iso-lines
        for f1 in [0.2, 0.4, 0.6, 0.8]:
            recall_range = np.linspace(0.01, 1, 100)
            precision_line = (f1 * recall_range) / (2 * recall_range - f1)
            valid = (precision_line > 0) & (precision_line <= 1)
            ax.plot(recall_range[valid], precision_line[valid], "--", color="gray", alpha=0.3)
            ax.annotate(f"F1={f1}", (0.95, f1 / (2 - f1)), fontsize=8, color="gray")
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, "precision_recall.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_dev_vs_ood(df: pd.DataFrame):
    """Compare dev vs OOD performance."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Pivot to get dev and ood F1 for each model-strategy combo
    dev_data = df[df["dataset"] == "dev"].set_index(["model", "strategy"])["f1"]
    ood_data = df[df["dataset"] == "ood"].set_index(["model", "strategy"])["f1"]
    
    common_idx = dev_data.index.intersection(ood_data.index)
    
    if len(common_idx) == 0:
        print("Warning: No common model-strategy combinations between dev and ood")
        return
    
    dev_f1 = dev_data.loc[common_idx]
    ood_f1 = ood_data.loc[common_idx]
    
    # Color by strategy
    strategies = [idx[1] for idx in common_idx]
    unique_strategies = list(set(strategies))
    colors = {s: plt.cm.tab10(i / len(unique_strategies)) for i, s in enumerate(unique_strategies)}
    
    for (model, strategy), dev_val in dev_f1.items():
        ood_val = ood_f1[(model, strategy)]
        ax.scatter(dev_val, ood_val, c=[colors[strategy]], s=100, alpha=0.7)
        ax.annotate(f"{model[:8]}", (dev_val, ood_val), fontsize=7, alpha=0.7)
    
    # Add diagonal line (perfect generalization)
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Perfect generalization")
    
    # Create legend
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[s], markersize=10, label=s) 
               for s in unique_strategies]
    ax.legend(handles=handles, title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    
    ax.set_title("Dev vs OOD F1 Score", fontsize=14)
    ax.set_xlabel("Dev F1 Score", fontsize=12)
    ax.set_ylabel("OOD F1 Score", fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, "dev_vs_ood.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_strategy_comparison(df: pd.DataFrame):
    """Box plot comparing strategies across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, dataset in enumerate(["dev", "ood"]):
        ax = axes[idx]
        data = df[df["dataset"] == dataset]
        
        if len(data) == 0:
            continue
        
        # Order strategies by median F1
        strategy_order = data.groupby("strategy")["f1"].median().sort_values(ascending=False).index
        
        sns.boxplot(
            data=data,
            x="strategy",
            y="f1",
            order=strategy_order,
            ax=ax,
            palette="Set2"
        )
        sns.stripplot(
            data=data,
            x="strategy",
            y="f1",
            order=strategy_order,
            ax=ax,
            color="black",
            alpha=0.5,
            size=8
        )
        
        ax.set_title(f"Strategy Comparison - {dataset.upper()}", fontsize=14)
        ax.set_xlabel("Strategy", fontsize=12)
        ax.set_ylabel("F1 Score", fontsize=12)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, "strategy_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_model_comparison(df: pd.DataFrame):
    """Box plot comparing models across strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, dataset in enumerate(["dev", "ood"]):
        ax = axes[idx]
        data = df[df["dataset"] == dataset]
        
        if len(data) == 0:
            continue
        
        # Order models by median F1
        model_order = data.groupby("model")["f1"].median().sort_values(ascending=False).index
        
        sns.boxplot(
            data=data,
            x="model",
            y="f1",
            order=model_order,
            ax=ax,
            palette="Set3"
        )
        sns.stripplot(
            data=data,
            x="model",
            y="f1",
            order=model_order,
            ax=ax,
            color="black",
            alpha=0.5,
            size=8
        )
        
        ax.set_title(f"Model Comparison - {dataset.upper()}", fontsize=14)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("F1 Score", fontsize=12)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, "model_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_all_metrics(df: pd.DataFrame):
    """Grouped bar chart showing all metrics."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    
    for dataset in ["dev", "ood"]:
        data = df[df["dataset"] == dataset]
        
        if len(data) == 0:
            continue
        
        # Average metrics by strategy
        avg_by_strategy = data.groupby("strategy")[metrics].mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(avg_by_strategy))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i * width, avg_by_strategy[metric], width, label=metric.capitalize())
        
        ax.set_title(f"Average Metrics by Strategy - {dataset.upper()}", fontsize=14)
        ax.set_xlabel("Strategy", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(avg_by_strategy.index, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(FIGURES_DIR, f"all_metrics_{dataset}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")


def plot_best_results(df: pd.DataFrame):
    """Bar chart of best F1 for each strategy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, dataset in enumerate(["dev", "ood"]):
        ax = axes[idx]
        data = df[df["dataset"] == dataset]
        
        if len(data) == 0:
            continue
        
        # Best F1 per strategy
        best = data.loc[data.groupby("strategy")["f1"].idxmax()]
        best = best.sort_values("f1", ascending=True)
        
        colors = plt.cm.RdYlGn(best["f1"])
        bars = ax.barh(best["strategy"], best["f1"], color=colors)
        
        # Add model labels
        for bar, model in zip(bars, best["model"]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   model[:15], va="center", fontsize=9)
        
        ax.set_title(f"Best F1 per Strategy - {dataset.upper()}", fontsize=14)
        ax.set_xlabel("F1 Score", fontsize=12)
        ax.set_xlim(0, 1.15)
        ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, "best_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def print_summary_tables(df: pd.DataFrame):
    """Print summary tables to console."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    for dataset in ["dev", "ood"]:
        data = df[df["dataset"] == dataset]
        if len(data) == 0:
            continue
        
        print(f"\n--- {dataset.upper()} Dataset ---")
        print(f"\nTop 10 by F1:")
        top10 = data.nlargest(10, "f1")[["model", "strategy", "f1", "precision", "recall", "accuracy"]]
        print(top10.to_string(index=False))
        
        print(f"\nBest model per strategy:")
        best_per_strategy = data.loc[data.groupby("strategy")["f1"].idxmax()][["strategy", "model", "f1"]]
        print(best_per_strategy.to_string(index=False))
        
        print(f"\nBest strategy per model:")
        best_per_model = data.loc[data.groupby("model")["f1"].idxmax()][["model", "strategy", "f1"]]
        print(best_per_model.to_string(index=False))
    
    print("\n" + "="*80)


def main():
    """Main function to generate all visualizations."""
    # Create output directories
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("Loading results...")
    df = load_all_results()
    
    if len(df) == 0:
        print("No results found. Make sure results are in the 'results' directory.")
        return
    
    print(f"Found {len(df)} result files")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Strategies: {df['strategy'].unique().tolist()}")
    print(f"Datasets: {df['dataset'].unique().tolist()}")
    
    # Save combined CSV
    print("\nSaving combined CSV...")
    save_combined_csv(df)
    
    # Generate all plots
    print("\nGenerating visualizations...")
    
    try:
        plot_f1_by_strategy(df)
    except Exception as e:
        print(f"Error in plot_f1_by_strategy: {e}")
    
    try:
        plot_f1_by_model(df)
    except Exception as e:
        print(f"Error in plot_f1_by_model: {e}")
    
    try:
        plot_heatmap(df)
    except Exception as e:
        print(f"Error in plot_heatmap: {e}")
    
    try:
        plot_precision_recall(df)
    except Exception as e:
        print(f"Error in plot_precision_recall: {e}")
    
    try:
        plot_dev_vs_ood(df)
    except Exception as e:
        print(f"Error in plot_dev_vs_ood: {e}")
    
    try:
        plot_strategy_comparison(df)
    except Exception as e:
        print(f"Error in plot_strategy_comparison: {e}")
    
    try:
        plot_model_comparison(df)
    except Exception as e:
        print(f"Error in plot_model_comparison: {e}")
    
    try:
        plot_all_metrics(df)
    except Exception as e:
        print(f"Error in plot_all_metrics: {e}")
    
    try:
        plot_best_results(df)
    except Exception as e:
        print(f"Error in plot_best_results: {e}")
    
    # Print summary
    print_summary_tables(df)
    
    print(f"\nDone! Figures saved to: {FIGURES_DIR}")
    print(f"Combined CSV saved to: {os.path.join(OUTPUT_DIR, 'combined_results.csv')}")


if __name__ == "__main__":
    main()