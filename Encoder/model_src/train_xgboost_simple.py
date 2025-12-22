#!/usr/bin/env python3
"""
Simple XGBoost training script for sentence splitting
No wandb, no hyperparameter tuning, just clean training
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
class Config:
    # Data files
    TRAIN_FILE = "manzoni_train_tokens.csv"
    DEV_FILE = "manzoni_dev_tokens.csv"
    
    # XGBoost hyperparameters (using good defaults)
    MAX_DEPTH = 10
    LEARNING_RATE = 0.01
    N_ESTIMATORS = 500
    MIN_CHILD_WEIGHT = 3
    SUBSAMPLE = 0.8
    COLSAMPLE_BYTREE = 0.8
    
    # Embedding model
    EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    
    # Output
    OUTPUT_DIR = "outputs"
    MODEL_FILE = "xgboost_model.json"

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data(filepath):
    """Load token data with proper handling for different formats"""
    print(f"Loading {filepath}...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Train/Dev files use comma separator
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline()  # Skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split by comma from right, only once (handles comma as token)
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
        
        # Label 1 marks end of sentence
        if label == 1:
            sentences.append(current_sent)
            labels.append(current_labels)
            current_sent = []
            current_labels = []
    
    # Add last sentence if exists
    if current_sent:
        sentences.append(current_sent)
        labels.append(current_labels)
    
    print(f"  Grouped into {len(sentences)} sentences")
    return sentences, labels

# ============================================================================
# FEATURE CREATION
# ============================================================================
def create_xgboost_features(sentences, labels, embedding_model):
    """Create features for XGBoost using embeddings and hand-crafted features"""
    X = []
    y = []
    
    for sent, labs in tqdm(zip(sentences, labels), desc="Creating features", total=len(sentences)):
        for i, (token, label) in enumerate(zip(sent, labs)):
            # Context window: 2 tokens before, current token, 2 tokens after
            context_start = max(0, i - 2)
            context_end = min(len(sent), i + 3)
            context = ' '.join(sent[context_start:context_end])
            
            # Get sentence embedding for context
            emb = embedding_model.encode(context, show_progress_bar=False)
            
            # Hand-crafted features
            features = [
                1 if token == '.' else 0,                    # Is period
                1 if token == '!' else 0,                    # Is exclamation
                1 if token == '?' else 0,                    # Is question mark
                1 if token == ',' else 0,                    # Is comma
                1 if token == ';' else 0,                    # Is semicolon
                1 if token == ':' else 0,                    # Is colon
                1 if i + 1 < len(sent) and len(sent[i+1]) > 0 and sent[i + 1][0].isupper() else 0,  # Next token starts with capital
                1 if i > 0 and len(sent[i-1]) > 0 and sent[i - 1][0].isupper() else 0,              # Prev token starts with capital
                len(token),                                   # Token length
                i / len(sent),                                # Relative position in sentence
            ]
            
            # Concatenate embedding and hand-crafted features
            X.append(np.concatenate([emb, features]))
            y.append(label)
    
    return np.array(X), np.array(y)

# ============================================================================
# TRAINING
# ============================================================================
def train_xgboost(X_train, y_train, X_eval, y_eval):
    """Train XGBoost model with fixed hyperparameters"""
    
    print("\nTraining XGBoost model...")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_eval)}")
    print(f"Hyperparameters:")
    print(f"  max_depth: {cfg.MAX_DEPTH}")
    print(f"  learning_rate: {cfg.LEARNING_RATE}")
    print(f"  n_estimators: {cfg.N_ESTIMATORS}")
    print(f"  min_child_weight: {cfg.MIN_CHILD_WEIGHT}")
    print(f"  subsample: {cfg.SUBSAMPLE}")
    print(f"  colsample_bytree: {cfg.COLSAMPLE_BYTREE}")
    
    # Create XGBoost classifier
    # IMPORTANT: Use 'device' instead of 'gpu_id' for XGBoost 3.1+
    model = xgb.XGBClassifier(
        max_depth=cfg.MAX_DEPTH,
        learning_rate=cfg.LEARNING_RATE,
        n_estimators=cfg.N_ESTIMATORS,
        min_child_weight=cfg.MIN_CHILD_WEIGHT,
        subsample=cfg.SUBSAMPLE,
        colsample_bytree=cfg.COLSAMPLE_BYTREE,
        device='cuda:0',  # Use 'device' instead of 'gpu_id' for XGBoost 3.1+
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    
    # Train with evaluation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_eval, y_eval)],
        verbose=True
    )
    
    return model

def evaluate_model(model, X, y, dataset_name):
    """Evaluate model and print metrics"""
    y_pred = model.predict(X)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average='binary', zero_division=0
    )
    acc = accuracy_score(y, y_pred)
    
    print(f"\n{dataset_name} Results:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': acc}

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 80)
    print("XGBoost Sentence Splitting - Simple Training")
    print("=" * 80)
    
    # Load data
    print("\n[1/5] Loading data...")
    train_sentences, train_labels = load_data(cfg.TRAIN_FILE)
    dev_sentences, dev_labels = load_data(cfg.DEV_FILE)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_sentences)} sentences")
    print(f"  Dev:   {len(dev_sentences)} sentences")
    
    # Load embedding model
    print(f"\n[2/5] Loading embedding model: {cfg.EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    embedding_model = embedding_model.to('cuda')
    print("  Model loaded on GPU")
    
    # Create features
    print("\n[3/5] Creating features...")
    X_train, y_train = create_xgboost_features(train_sentences, train_labels, embedding_model)
    X_dev, y_dev = create_xgboost_features(dev_sentences, dev_labels, embedding_model)
    
    print(f"  Feature dimensions: {X_train.shape[1]}")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Dev samples:   {len(X_dev)}")
    
    # Train model
    print("\n[4/5] Training model...")
    model = train_xgboost(X_train, y_train, X_dev, y_dev)
    
    # Evaluate
    print("\n[5/5] Evaluating model...")
    train_results = evaluate_model(model, X_train, y_train, "Training Set")
    dev_results = evaluate_model(model, X_dev, y_dev, "Validation Set")
    
    # Save model (using booster to avoid XGBoost 3.1.2 bug)
    # XGBoost 3.1.2 has a bug with sklearn wrapper's save_model()
    # Solution: save the underlying booster object directly
    model_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_FILE)
    model.get_booster().save_model(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save results summary
    results_df = pd.DataFrame([
        {'Dataset': 'Train', **train_results},
        {'Dataset': 'Dev', **dev_results}
    ])
    results_path = os.path.join(cfg.OUTPUT_DIR, "training_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()