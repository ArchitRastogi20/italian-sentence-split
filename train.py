# train.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import xgboost as xgb
from sentence_transformers import SentenceTransformer
import optuna
from tqdm import tqdm
import wandb
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
class Config:
    # Data
    TRAIN_FILE = "manzoni_train_tokens.csv"
    DEV_FILE = "manzoni_dev_tokens.csv"
    OOD_FILE = "OOD_test.csv"
    
    # Models to train (only 2 as requested)
    ENCODER_MODELS = [
        "dbmdz/bert-base-italian-xxl-cased",
        "answerdotai/ModernBERT-base",
    ]
    
    # Training
    MAX_LENGTH = 512
    STRIDE = 64
    BATCH_SIZE = 16
    GRAD_ACCUM = 4  # Effective batch size = 64
    LEARNING_RATE = 5e-5
    EPOCHS = 10
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    FP16 = True
    DATALOADER_WORKERS = 12
    
    # Optimization
    N_TRIALS = 15
    
    # Paths
    OUTPUT_DIR = "outputs"
    CACHE_DIR = "cache"

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(cfg.CACHE_DIR, exist_ok=True)

# ============================================================================
# UTILS
# ============================================================================
def load_data(filepath):
    """Load token data with proper handling for different formats"""
    print(f"Loading {filepath}...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Detect file format
    if 'OOD' in filepath or 'pinocchio' in filepath.lower():
        # OOD file uses semicolon separator
        # Read line by line to properly handle header
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
        # Train/Dev files use comma separator
        # Read line by line to handle commas in tokens
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            header = f.readline()  # Skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split by comma, but handle cases where token is comma
                parts = line.rsplit(',', 1)  # Split from right, only once
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

def compute_metrics(pred):
    """Compute metrics for Trainer"""
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    predictions = predictions[labels != -100].flatten()
    labels = labels[labels != -100].flatten()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': acc
    }

# ============================================================================
# DATASET
# ============================================================================
class SentenceSplitDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=512, stride=64):
        self.encodings = []
        self.labels_aligned = []
        
        for sent, labs in tqdm(zip(sentences, labels), desc="Tokenizing", total=len(sentences)):
            # Join tokens with spaces
            text = ' '.join(sent)
            
            # Tokenize with overflow
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding='max_length'
            )
            
            # Align labels for each chunk
            for i in range(len(encoding['input_ids'])):
                self.encodings.append({
                    'input_ids': encoding['input_ids'][i],
                    'attention_mask': encoding['attention_mask'][i]
                })
                
                # Align labels to first subword only
                word_ids = encoding.word_ids(batch_index=i)
                label_ids = []
                previous_word_idx = None
                
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(labs[word_idx] if word_idx < len(labs) else 0)
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                
                self.labels_aligned.append(label_ids)
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.encodings[idx].items()}
        item['labels'] = torch.tensor(self.labels_aligned[idx])
        return item

# ============================================================================
# ENCODER TRAINING
# ============================================================================
def train_encoder(model_name, train_dataset, eval_dataset, trial=None):
    """Train a single encoder model"""
    
    # Hyperparameters
    if trial:
        lr = trial.suggest_float('lr', 1e-5, 1e-4, log=True)
        weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
        warmup_ratio = trial.suggest_float('warmup_ratio', 0.0, 0.3)
    else:
        lr = cfg.LEARNING_RATE
        weight_decay = cfg.WEIGHT_DECAY
        warmup_ratio = cfg.WARMUP_RATIO
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cfg.CACHE_DIR)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, 
        num_labels=2,
        cache_dir=cfg.CACHE_DIR
    )
    
    model_short_name = model_name.split('/')[-1]
    output_dir = f"{cfg.OUTPUT_DIR}/{model_short_name}"
    
    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.BATCH_SIZE * 2,
        gradient_accumulation_steps=cfg.GRAD_ACCUM,
        num_train_epochs=cfg.EPOCHS,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        fp16=cfg.FP16,
        dataloader_num_workers=cfg.DATALOADER_WORKERS,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        report_to="wandb" if not trial else "none",
        run_name=f"{model_short_name}" if not trial else None,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    
    return trainer, eval_results['eval_f1']

# ============================================================================
# XGBOOST
# ============================================================================
def create_xgboost_features(sentences, labels, embedding_model):
    """Create features for XGBoost"""
    X = []
    y = []
    
    for sent, labs in tqdm(zip(sentences, labels), desc="Creating XGB features", total=len(sentences)):
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
                i / len(sent),  # Relative position
            ]
            
            X.append(np.concatenate([emb, features]))
            y.append(label)
    
    return np.array(X), np.array(y)

def train_xgboost(train_sentences, train_labels, eval_sentences, eval_labels, trial=None):
    """Train XGBoost model"""
    
    # Load embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embedding_model = embedding_model.to('cuda')
    
    # Create features
    print("Creating train features...")
    X_train, y_train = create_xgboost_features(train_sentences, train_labels, embedding_model)
    print("Creating eval features...")
    X_eval, y_eval = create_xgboost_features(eval_sentences, eval_labels, embedding_model)
    
    # Hyperparameters
    if trial:
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }
    else:
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    
    # Train
    model = xgb.XGBClassifier(
        **params,
        tree_method='gpu_hist',
        gpu_id=0,
        eval_metric='logloss',
        early_stopping_rounds=50,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_eval, y_eval)],
        verbose=False
    )
    
    # Evaluate
    y_pred = model.predict(X_eval)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_eval, y_pred, average='binary', zero_division=0
    )
    acc = accuracy_score(y_eval, y_pred)
    
    print(f"XGBoost - P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}, Acc: {acc:.4f}")
    
    return model, f1

# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================
def objective_encoder(trial, model_name, train_dataset, eval_dataset):
    """Optuna objective for encoder"""
    _, f1 = train_encoder(model_name, train_dataset, eval_dataset, trial)
    return f1

def objective_xgboost(trial, train_sentences, train_labels, eval_sentences, eval_labels):
    """Optuna objective for XGBoost"""
    _, f1 = train_xgboost(train_sentences, train_labels, eval_sentences, eval_labels, trial)
    return f1

# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_on_test(trainer, test_dataset, model_name, dataset_name):
    """Evaluate model on test set"""
    results = trainer.evaluate(test_dataset)
    
    print(f"\n{model_name} on {dataset_name}:")
    print(f"Precision: {results['eval_precision']:.4f}")
    print(f"Recall: {results['eval_recall']:.4f}")
    print(f"F1: {results['eval_f1']:.4f}")
    print(f"Accuracy: {results['eval_accuracy']:.4f}")
    
    return results

def evaluate_xgboost_on_test(model, test_sentences, test_labels, embedding_model, dataset_name):
    """Evaluate XGBoost on test set"""
    X_test, y_test = create_xgboost_features(test_sentences, test_labels, embedding_model)
    y_pred = model.predict(X_test)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', zero_division=0
    )
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nXGBoost on {dataset_name}:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': acc}

# ============================================================================
# MAIN
# ============================================================================
def main():
    # Initialize wandb
    wandb.init(project="sentence-splitting", name="full-pipeline")
    
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    train_sentences, train_labels = load_data(cfg.TRAIN_FILE)
    dev_sentences, dev_labels = load_data(cfg.DEV_FILE)
    ood_sentences, ood_labels = load_data(cfg.OOD_FILE)
    
    print(f"\nTrain: {len(train_sentences)} sentences")
    print(f"Dev: {len(dev_sentences)} sentences")
    print(f"OOD: {len(ood_sentences)} sentences")
    
    all_results = {}
    
    # ========================================================================
    # TRAIN ENCODERS
    # ========================================================================
    for model_name in cfg.ENCODER_MODELS:
        print("\n" + "=" * 80)
        print(f"TRAINING ENCODER: {model_name}")
        print("=" * 80)
        
        model_short_name = model_name.split('/')[-1]
        
        # Prepare datasets
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cfg.CACHE_DIR)
        
        print("Preparing train dataset...")
        train_dataset = SentenceSplitDataset(
            train_sentences, train_labels, tokenizer, 
            cfg.MAX_LENGTH, cfg.STRIDE
        )
        
        print("Preparing dev dataset...")
        dev_dataset = SentenceSplitDataset(
            dev_sentences, dev_labels, tokenizer,
            cfg.MAX_LENGTH, cfg.STRIDE
        )
        
        print("Preparing OOD dataset...")
        ood_dataset = SentenceSplitDataset(
            ood_sentences, ood_labels, tokenizer,
            cfg.MAX_LENGTH, cfg.STRIDE
        )
        
        # Hyperparameter tuning
        print(f"\nHyperparameter tuning for {model_short_name}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective_encoder(trial, model_name, train_dataset, dev_dataset),
            n_trials=cfg.N_TRIALS,
            show_progress_bar=True
        )
        
        print(f"\nBest hyperparameters: {study.best_params}")
        print(f"Best F1: {study.best_value:.4f}")
        
        # Train final model with best params
        print(f"\nTraining final model for {model_short_name}...")
        trainer, _ = train_encoder(model_name, train_dataset, dev_dataset)
        
        # Evaluate
        dev_results = evaluate_on_test(trainer, dev_dataset, model_short_name, "DEV")
        ood_results = evaluate_on_test(trainer, ood_dataset, model_short_name, "OOD")
        
        all_results[model_short_name] = {
            'dev': dev_results,
            'ood': ood_results,
            'best_params': study.best_params
        }
        
        # Save model
        trainer.save_model(f"{cfg.OUTPUT_DIR}/{model_short_name}/final")
        
        # Clear memory
        del trainer, train_dataset, dev_dataset, ood_dataset, tokenizer
        torch.cuda.empty_cache()
    
    # ========================================================================
    # TRAIN XGBOOST
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING XGBOOST")
    print("=" * 80)
    
    # Hyperparameter tuning
    print("\nHyperparameter tuning for XGBoost...")
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective_xgboost(
            trial, train_sentences, train_labels, dev_sentences, dev_labels
        ),
        n_trials=cfg.N_TRIALS,
        show_progress_bar=True
    )
    
    print(f"\nBest hyperparameters: {study.best_params}")
    print(f"Best F1: {study.best_value:.4f}")
    
    # Train final model
    print("\nTraining final XGBoost model...")
    xgb_model, _ = train_xgboost(train_sentences, train_labels, dev_sentences, dev_labels)
    
    # Evaluate
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embedding_model = embedding_model.to('cuda')
    
    dev_results = evaluate_xgboost_on_test(xgb_model, dev_sentences, dev_labels, embedding_model, "DEV")
    ood_results = evaluate_xgboost_on_test(xgb_model, ood_sentences, ood_labels, embedding_model, "OOD")
    
    all_results['XGBoost'] = {
        'dev': dev_results,
        'ood': ood_results,
        'best_params': study.best_params
    }
    
    # Save model
    xgb_model.save_model(f"{cfg.OUTPUT_DIR}/xgboost_final.json")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    results_df = []
    for model_name, results in all_results.items():
        results_df.append({
            'Model': model_name,
            'Dev F1': results['dev']['eval_f1'] if 'eval_f1' in results['dev'] else results['dev']['f1'],
            'Dev Acc': results['dev']['eval_accuracy'] if 'eval_accuracy' in results['dev'] else results['dev']['accuracy'],
            'OOD F1': results['ood']['eval_f1'] if 'eval_f1' in results['ood'] else results['ood']['f1'],
            'OOD Acc': results['ood']['eval_accuracy'] if 'eval_accuracy' in results['ood'] else results['ood']['accuracy'],
        })
    
    results_df = pd.DataFrame(results_df)
    print(results_df.to_string(index=False))
    results_df.to_csv(f"{cfg.OUTPUT_DIR}/final_results.csv", index=False)
    
    wandb.log({"final_results": wandb.Table(dataframe=results_df)})
    wandb.finish()
    
    print(f"\n✅ All done! Results saved to {cfg.OUTPUT_DIR}")

if __name__ == "__main__":
    main()