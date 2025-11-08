# train_encoders.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import optuna
from tqdm import tqdm
import wandb
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
class Config:
    TRAIN_FILE = "manzoni_train_tokens.csv"
    DEV_FILE = "manzoni_dev_tokens.csv"
    OOD_FILE = "OOD_test.csv"
    
    ENCODER_MODELS = [
        "dbmdz/bert-base-italian-xxl-cased",
        "answerdotai/ModernBERT-base",
    ]
    
    MAX_LENGTH = 512
    STRIDE = 64
    BATCH_SIZE = 8
    GRAD_ACCUM = 4
    EPOCHS = 10
    N_TRIALS = 20
    
    OUTPUT_DIR = "outputs"
    CACHE_DIR = "cache"
    WANDB_PROJECT = "sentence_splitting_encoder"

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(cfg.CACHE_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data(filepath):
    print(f"Loading {filepath}...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if 'OOD' in filepath or 'pinocchio' in filepath.lower():
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line.lower().startswith('pinocchio') or first_line == 'token;label':
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(';')
                    if len(parts) == 2:
                        token, label = parts
                        if token == 'token' and label == 'label':
                            continue
                        try:
                            data.append([token, int(label)])
                        except ValueError:
                            continue
            else:
                parts = first_line.split(';')
                if len(parts) == 2:
                    token, label = parts
                    try:
                        data.append([token, int(label)])
                    except ValueError:
                        pass
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
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            header = f.readline()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(',', 1)
                if len(parts) == 2:
                    token, label = parts
                    try:
                        data.append([token, int(label)])
                    except ValueError:
                        continue
        df = pd.DataFrame(data, columns=['token', 'label'])
    
    print(f"  Loaded {len(df)} tokens")
    
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
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)
    predictions = predictions[labels != -100].flatten()
    labels = labels[labels != -100].flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': acc}

# ============================================================================
# DATASET
# ============================================================================
class SentenceSplitDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=512, stride=64):
        self.encodings = []
        self.labels_aligned = []
        
        for sent, labs in tqdm(zip(sentences, labels), desc="Tokenizing", total=len(sentences)):
            text = ' '.join(sent)
            encoding = tokenizer(
                text, truncation=True, max_length=max_length, stride=stride,
                return_overflowing_tokens=True, return_offsets_mapping=True, padding='max_length'
            )
            
            for i in range(len(encoding['input_ids'])):
                self.encodings.append({
                    'input_ids': encoding['input_ids'][i],
                    'attention_mask': encoding['attention_mask'][i]
                })
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
# TRAINING
# ============================================================================
def train_encoder(model_name, train_dataset, eval_dataset, ood_dataset, trial=None, final=False):
    if trial:
        lr = trial.suggest_float('lr', 1e-5, 1e-4, log=True)
        weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
        warmup_ratio = trial.suggest_float('warmup_ratio', 0.0, 0.3)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    else:
        lr = 5e-5
        weight_decay = 0.01
        warmup_ratio = 0.1
        batch_size = cfg.BATCH_SIZE
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cfg.CACHE_DIR)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=2, cache_dir=cfg.CACHE_DIR
    )
    
    model_short_name = model_name.split('/')[-1]
    output_dir = f"{cfg.OUTPUT_DIR}/{model_short_name}"
    
    run_name = f"{model_short_name}_final" if final else f"{model_short_name}_trial"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=cfg.GRAD_ACCUM,
        num_train_epochs=cfg.EPOCHS,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        fp16=True,
        dataloader_num_workers=12,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=3,
        logging_steps=50,
        report_to="wandb" if final else "none",
        run_name=run_name,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        logging_first_step=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    trainer.train()
    
    eval_results = trainer.evaluate(eval_dataset)
    
    if final and ood_dataset:
        ood_results = trainer.evaluate(ood_dataset)
        wandb.log({
            f"{model_short_name}/ood_f1": ood_results['eval_f1'],
            f"{model_short_name}/ood_accuracy": ood_results['eval_accuracy'],
            f"{model_short_name}/ood_precision": ood_results['eval_precision'],
            f"{model_short_name}/ood_recall": ood_results['eval_recall'],
        })
        return trainer, eval_results['eval_f1'], ood_results
    
    return trainer, eval_results['eval_f1'], None

def objective(trial, model_name, train_dataset, eval_dataset, ood_dataset):
    _, f1, _ = train_encoder(model_name, train_dataset, eval_dataset, ood_dataset, trial=trial, final=False)
    return f1

# ============================================================================
# MAIN
# ============================================================================
def main():
    wandb.login()
    
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
    
    for model_name in cfg.ENCODER_MODELS:
        print("\n" + "=" * 80)
        print(f"TRAINING: {model_name}")
        print("=" * 80)
        
        model_short_name = model_name.split('/')[-1]
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cfg.CACHE_DIR)
        
        print("Preparing datasets...")
        train_dataset = SentenceSplitDataset(train_sentences, train_labels, tokenizer, cfg.MAX_LENGTH, cfg.STRIDE)
        dev_dataset = SentenceSplitDataset(dev_sentences, dev_labels, tokenizer, cfg.MAX_LENGTH, cfg.STRIDE)
        ood_dataset = SentenceSplitDataset(ood_sentences, ood_labels, tokenizer, cfg.MAX_LENGTH, cfg.STRIDE)
        
        print(f"\nHyperparameter tuning for {model_short_name}...")
        study = optuna.create_study(direction='maximize', study_name=f"{model_short_name}_study")
        study.optimize(
            lambda trial: objective(trial, model_name, train_dataset, dev_dataset, ood_dataset),
            n_trials=cfg.N_TRIALS,
            show_progress_bar=True
        )
        
        print(f"\nBest params: {study.best_params}")
        print(f"Best F1: {study.best_value:.4f}")
        
        best_params = study.best_params
        
        print(f"\nTraining final model for {model_short_name}...")
        run = wandb.init(
            project=cfg.WANDB_PROJECT,
            name=f"{model_short_name}_final",
            config={
                "model": model_name,
                "best_params": best_params,
                "train_sentences": len(train_sentences),
                "dev_sentences": len(dev_sentences),
                "ood_sentences": len(ood_sentences),
            },
            reinit=True
        )
        
        trainer, dev_f1, ood_results = train_encoder(
            model_name, train_dataset, dev_dataset, ood_dataset, trial=None, final=True
        )
        
        dev_results = trainer.evaluate(dev_dataset)
        
        wandb.log({
            f"{model_short_name}/best_lr": best_params.get('lr', 5e-5),
            f"{model_short_name}/best_weight_decay": best_params.get('weight_decay', 0.01),
            f"{model_short_name}/best_warmup_ratio": best_params.get('warmup_ratio', 0.1),
            f"{model_short_name}/best_batch_size": best_params.get('batch_size', 16),
            f"{model_short_name}/dev_f1": dev_results['eval_f1'],
            f"{model_short_name}/dev_accuracy": dev_results['eval_accuracy'],
            f"{model_short_name}/dev_precision": dev_results['eval_precision'],
            f"{model_short_name}/dev_recall": dev_results['eval_recall'],
        })
        
        all_results[model_short_name] = {
            'dev': dev_results,
            'ood': ood_results,
            'best_params': best_params
        }
        
        trainer.save_model(f"{cfg.OUTPUT_DIR}/{model_short_name}/best_model")
        
        print(f"\n{model_short_name} Results:")
        print(f"  Dev F1: {dev_results['eval_f1']:.4f}")
        print(f"  Dev Acc: {dev_results['eval_accuracy']:.4f}")
        print(f"  OOD F1: {ood_results['eval_f1']:.4f}")
        print(f"  OOD Acc: {ood_results['eval_accuracy']:.4f}")
        
        wandb.finish()
        
        del trainer, train_dataset, dev_dataset, ood_dataset, tokenizer
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    results_df = []
    for model_name, results in all_results.items():
        results_df.append({
            'Model': model_name,
            'Dev F1': results['dev']['eval_f1'],
            'Dev Acc': results['dev']['eval_accuracy'],
            'OOD F1': results['ood']['eval_f1'],
            'OOD Acc': results['ood']['eval_accuracy'],
            'Best LR': results['best_params'].get('lr', 5e-5),
            'Best WD': results['best_params'].get('weight_decay', 0.01),
        })
    
    results_df = pd.DataFrame(results_df)
    print(results_df.to_string(index=False))
    results_df.to_csv(f"{cfg.OUTPUT_DIR}/final_results.csv", index=False)
    
    final_run = wandb.init(project=cfg.WANDB_PROJECT, name="final_summary", reinit=True)
    wandb.log({"final_results": wandb.Table(dataframe=results_df)})
    wandb.finish()
    
    print(f"\n✅ Complete! Results in {cfg.OUTPUT_DIR}")

if __name__ == "__main__":
    main()