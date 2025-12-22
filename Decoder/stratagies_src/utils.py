"""
Utility functions for sentence splitting task.
Handles data loading, preprocessing, and evaluation.
"""

import os
import csv
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)


def load_manzoni_data(filepath: str) -> Tuple[List[str], List[int]]:
    """
    Load Manzoni dataset (train or dev).
    Returns list of tokens and list of labels.
    """
    tokens = []
    labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 2:
                tokens.append(row[0])
                labels.append(int(row[1]))
            elif len(row) == 1:
                # Handle edge cases where comma is the token
                tokens.append(',')
                labels.append(0)
    
    return tokens, labels


def load_ood_data(filepath: str) -> Tuple[List[str], List[int]]:
    """
    Load OOD (Pinocchio) test data.
    Uses semicolon as delimiter with special handling for edge cases.
    """
    tokens = []
    labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip first two lines (header and column names)
    for line in lines[2:]:
        # Handle Windows line endings
        line = line.strip().replace('\r', '')
        if not line:
            continue
        
        # Handle quoted semicolons: ";";0 or ";";1
        if line.startswith('";"'):
            # The token is a semicolon itself
            tokens.append(';')
            # Get the label after the quoted semicolon
            parts = line.split(';')
            if len(parts) >= 3:
                try:
                    labels.append(int(parts[2]))
                except ValueError:
                    labels.append(0)
            else:
                labels.append(0)
            continue
        
        # Split by semicolon for normal cases
        parts = line.split(';')
        if len(parts) >= 2:
            token = parts[0].strip('"')  # Remove any quotes
            try:
                label = int(parts[1])
                tokens.append(token)
                labels.append(label)
            except ValueError:
                # Skip malformed lines
                continue
        elif len(parts) == 1 and parts[0]:
            # Handle edge cases
            tokens.append(parts[0])
            labels.append(0)
    
    return tokens, labels


def chunk_tokens(tokens: List[str], labels: List[int], chunk_size: int = 100, overlap: int = 10) -> List[Tuple[List[str], List[int], int]]:
    """
    Chunk tokens into manageable sizes for LLM processing.
    Returns list of (tokens, labels, start_index) tuples.
    """
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_labels = labels[start:end]
        chunks.append((chunk_tokens, chunk_labels, start))
        
        # Move start, accounting for overlap at end
        if end >= len(tokens):
            break
        start = end - overlap
    
    return chunks


def evaluate_predictions(y_true: List[int], y_pred: List[int]) -> Dict:
    """
    Evaluate predictions and return metrics dictionary.
    """
    # Ensure same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def save_predictions(
    tokens: List[str], 
    labels_gt: List[int], 
    predictions: List[int],
    output_path: str
):
    """
    Save predictions to CSV file.
    """
    # Ensure all lists have same length
    min_len = min(len(tokens), len(labels_gt), len(predictions))
    
    df = pd.DataFrame({
        'token': tokens[:min_len],
        'label_gt': labels_gt[:min_len],
        'prediction': predictions[:min_len]
    })
    
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")
    
    return df


def print_metrics(metrics: Dict, model_name: str, prompt_id: str):
    """Print evaluation metrics in a nice format."""
    print(f"\n{'='*60}")
    print(f"Results for {model_name} with {prompt_id}")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  Predicted:  0      1")
    print(f"  Actual 0: {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"  Actual 1: {cm[1][0]:5d}  {cm[1][1]:5d}")
    print(f"{'='*60}\n")


def get_data_paths() -> Dict[str, str]:
    """Get paths to data files."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    return {
        'train': os.path.join(data_dir, 'manzoni_train_tokens.csv'),
        'dev': os.path.join(data_dir, 'manzoni_dev_tokens.csv'),
        'ood': os.path.join(data_dir, 'OOD_test.csv'),
    }


def get_output_dir() -> str:
    """Get output directory path."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


if __name__ == "__main__":
    # Test data loading
    paths = get_data_paths()
    
    print("Testing data loading...")
    
    # Test train data
    tokens, labels = load_manzoni_data(paths['train'])
    print(f"\nTrain data: {len(tokens)} tokens")
    print(f"Label distribution: 0={labels.count(0)}, 1={labels.count(1)}")
    print(f"First 10 tokens: {tokens[:10]}")
    print(f"First 10 labels: {labels[:10]}")
    
    # Test dev data
    tokens, labels = load_manzoni_data(paths['dev'])
    print(f"\nDev data: {len(tokens)} tokens")
    print(f"Label distribution: 0={labels.count(0)}, 1={labels.count(1)}")
    
    # Test OOD data
    tokens, labels = load_ood_data(paths['ood'])
    print(f"\nOOD data: {len(tokens)} tokens")
    print(f"Label distribution: 0={labels.count(0)}, 1={labels.count(1)}")
    print(f"First 10 tokens: {tokens[:10]}")
    print(f"First 10 labels: {labels[:10]}")
