#!/usr/bin/env python3
"""
XGBoost testing script for sentence splitting
Loads trained model and evaluates on test data
Saves results to both TXT and CSV files
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    precision_recall_fscore_support, 
    accuracy_score,
    classification_report,
    confusion_matrix
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
class Config:
    # Test data files
    DEV_FILE = "manzoni_dev_tokens.csv"
    OOD_FILE = "OOD_test.csv"
    
    # Model and embedding
    MODEL_FILE = "outputs/xgboost_model.json"
    EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    
    # Output
    OUTPUT_DIR = "outputs"
    RESULTS_TXT = "test_results.txt"
    RESULTS_CSV = "test_results.csv"
    PREDICTIONS_CSV = "test_predictions.csv"

cfg = Config()

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data(filepath):
    """Load token data with proper handling for different formats"""
    print(f"Loading {filepath}...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Detect file format
    if 'OOD' in filepath or 'pinocchio' in filepath.lower():
        # OOD file uses semicolon separator
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            
            # Check if first line is header
            if first_line.lower().startswith('pinocchio') or first_line == 'token;label':
                # Skip header, read rest
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(';')
                    if len(parts) == 2:
                        token, label = parts
                        # Skip if it's still a header-like line
                        if token == 'token' and label == 'label':
                            continue
                        try:
                            data.append([token, int(label)])
                        except ValueError:
                            continue
            else:
                # First line is data
                parts = first_line.split(';')
                if len(parts) == 2:
                    token, label = parts
                    try:
                        data.append([token, int(label)])
                    except ValueError:
                        pass
                
                # Read rest
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(';')
                    if len(parts) == 2:
                        token, label = parts
                        try:
                            data.append([token, int(label)])
                        except ValueError:
                            continue
        
        df = pd.DataFrame(data, columns=['token', 'label'])
    else:
        # Dev file uses comma separator
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            header = f.readline()  # Skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split by comma from right, only once
                parts = line.rsplit(',', 1)
                if len(parts) == 2:
                    token, label = parts
                    try:
                        data.append([token, int(label)])
                    except ValueError:
                        continue
        
        df = pd.DataFrame(data, columns=['token', 'label'])
    
    print(f"  Loaded {len(df)} tokens")
    
    # Group tokens into sentences
    sentences = []
    labels = []
    current_sent = []
    current_labels = []
    
    for _, row in df.iterrows():
        token = str(row['token'])
        label = int(row['label'])
        
        current_sent.append(token)
        current_labels.append(label)
        
        if label == 1:
            sentences.append(current_sent)
            labels.append(current_labels)
            current_sent = []
            current_labels = []
    
    if current_sent:
        sentences.append(current_sent)
        labels.append(current_labels)
    
    print(f"  Grouped into {len(sentences)} sentences")
    return sentences, labels

# ============================================================================
# FEATURE CREATION
# ============================================================================
def create_xgboost_features(sentences, labels, embedding_model):
    """Create features for XGBoost"""
    X = []
    y = []
    tokens_list = []  # Keep track of tokens for later analysis
    
    for sent, labs in tqdm(zip(sentences, labels), desc="Creating features", total=len(sentences)):
        for i, (token, label) in enumerate(zip(sent, labs)):
            # Context window
            context_start = max(0, i - 2)
            context_end = min(len(sent), i + 3)
            context = ' '.join(sent[context_start:context_end])
            
            # Get embedding
            emb = embedding_model.encode(context, show_progress_bar=False)
            
            # Hand-crafted features
            features = [
                1 if token == '.' else 0,
                1 if token == '!' else 0,
                1 if token == '?' else 0,
                1 if token == ',' else 0,
                1 if token == ';' else 0,
                1 if token == ':' else 0,
                1 if i + 1 < len(sent) and len(sent[i+1]) > 0 and sent[i + 1][0].isupper() else 0,
                1 if i > 0 and len(sent[i-1]) > 0 and sent[i - 1][0].isupper() else 0,
                len(token),
                i / len(sent),
            ]
            
            X.append(np.concatenate([emb, features]))
            y.append(label)
            tokens_list.append(token)
    
    return np.array(X), np.array(y), tokens_list

# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_and_save(model, X, y, tokens, dataset_name, output_file):
    """Evaluate model and save detailed results"""
    
    # Make predictions using Booster
    dmatrix = xgb.DMatrix(X)
    y_pred_proba = model.predict(dmatrix)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average='binary', zero_division=0
    )
    acc = accuracy_score(y, y_pred)
    
    # Get detailed classification report
    class_report = classification_report(
        y, y_pred, 
        target_names=['No Split (0)', 'Split (1)'],
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Write to text file
    with open(output_file, 'a') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{dataset_name} Evaluation Results\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"  Accuracy:  {acc:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall:    {recall:.4f}\n")
        f.write(f"  F1 Score:  {f1:.4f}\n\n")
        
        f.write("Classification Report:\n")
        f.write(class_report + "\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(f"                 Predicted\n")
        f.write(f"                 0      1\n")
        f.write(f"Actual  0      {cm[0][0]:5d}  {cm[0][1]:5d}\n")
        f.write(f"        1      {cm[1][0]:5d}  {cm[1][1]:5d}\n\n")
        
        # Error analysis - show some misclassified examples
        f.write("Sample Misclassifications (first 10):\n")
        f.write("-" * 80 + "\n")
        misclassified_idx = np.where(y != y_pred)[0]
        for idx in misclassified_idx[:10]:
            f.write(f"Token: '{tokens[idx]}'\n")
            f.write(f"  True label: {y[idx]}, Predicted: {y_pred[idx]}, ")
            f.write(f"Probability: {y_pred_proba[idx]:.4f}\n")
        f.write("\n\n")
    
    print(f"\n{dataset_name} Results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'token': tokens,
        'true_label': y,
        'predicted_label': y_pred,
        'prediction_probability': y_pred_proba
    })
    
    predictions_file = os.path.join(
        cfg.OUTPUT_DIR, 
        f"{dataset_name.lower().replace(' ', '_')}_predictions.csv"
    )
    predictions_df.to_csv(predictions_file, index=False)
    print(f"  Predictions saved to: {predictions_file}")
    
    return {
        'dataset': dataset_name,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_samples': len(y),
        'num_misclassified': len(misclassified_idx)
    }

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 80)
    print("XGBoost Sentence Splitting - Testing")
    print("=" * 80)
    
    # Check if model exists
    if not os.path.exists(cfg.MODEL_FILE):
        print(f"\nError: Model file not found: {cfg.MODEL_FILE}")
        print("Please run the training script first!")
        return
    
    # Load model (use Booster directly to avoid XGBoost 3.1.2 sklearn wrapper issues)
    print(f"\n[1/5] Loading model from: {cfg.MODEL_FILE}")
    model = xgb.Booster()
    model.load_model(cfg.MODEL_FILE)
    print("  Model loaded successfully")
    
    # Load embedding model
    print(f"\n[2/5] Loading embedding model: {cfg.EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    embedding_model = embedding_model.to('cuda')
    print("  Embedding model loaded on GPU")
    
    # Prepare output files
    txt_output = os.path.join(cfg.OUTPUT_DIR, cfg.RESULTS_TXT)
    csv_output = os.path.join(cfg.OUTPUT_DIR, cfg.RESULTS_CSV)
    
    # Clear previous results
    with open(txt_output, 'w') as f:
        f.write("XGBoost Sentence Splitting - Test Results\n")
        f.write(f"Model: {cfg.MODEL_FILE}\n")
        f.write("=" * 80 + "\n\n")
    
    all_results = []
    
    # Test on Dev set
    print("\n[3/5] Testing on Dev set...")
    dev_sentences, dev_labels = load_data(cfg.DEV_FILE)
    X_dev, y_dev, tokens_dev = create_xgboost_features(dev_sentences, dev_labels, embedding_model)
    
    dev_results = evaluate_and_save(model, X_dev, y_dev, tokens_dev, "Dev Set", txt_output)
    all_results.append(dev_results)
    
    # Test on OOD set
    print("\n[4/5] Testing on OOD set...")
    ood_sentences, ood_labels = load_data(cfg.OOD_FILE)
    X_ood, y_ood, tokens_ood = create_xgboost_features(ood_sentences, ood_labels, embedding_model)
    
    ood_results = evaluate_and_save(model, X_ood, y_ood, tokens_ood, "OOD Set", txt_output)
    all_results.append(ood_results)
    
    # Save summary CSV
    print("\n[5/5] Saving summary...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(csv_output, index=False)
    
    print(f"\nResults saved to:")
    print(f"   - {txt_output}")
    print(f"   - {csv_output}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()