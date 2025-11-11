# train_encoders_crf.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModel,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from torchcrf import CRF
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import optuna
from tqdm import tqdm
import wandb
import random
import string
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
class Config:
    TRAIN_FILE = "manzoni_train_tokens.csv"
    DEV_FILE = "manzoni_dev_tokens.csv"
    OOD_FILE = "OOD_test.csv"
    
    # Just run remaining models
    ENCODER_MODELS = [
        "answerdotai/ModernBERT-base"
    ]
    
    MAX_LENGTH = 512
    STRIDE = 64
    BATCH_SIZE = 64           # Up from 16 - you have 24GB VRAM
    GRAD_ACCUM = 1            # Down from 4 - effective batch still 48
    EPOCHS = 10
    N_TRIALS = 10             # Up from 10 - faster with your CPU for XLM we used 15 trials
    
    OUTPUT_DIR = "outputs_crf"
    CACHE_DIR = "cache"
    WANDB_PROJECT = "sentance-splitting-encoder-crf"

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(cfg.CACHE_DIR, exist_ok=True)

def generate_run_name():
    """Generate random 6-character name"""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

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

# ============================================================================
# MODEL WITH CRF
# ============================================================================
class BERTWithCRF(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, cache_dir=cfg.CACHE_DIR)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the encoder"""
        if hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the encoder"""
        if hasattr(self.encoder, 'gradient_checkpointing_disable'):
            self.encoder.gradient_checkpointing_disable()
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)
        
        mask = attention_mask.bool()
        
        if labels is not None:
            labels_crf = labels.clone()
            valid_labels_mask = (labels != -100)
            labels_crf[labels_crf == -100] = 0
            
            log_likelihood = self.crf(emissions, labels_crf, mask=mask, reduction='none')
            
            batch_size = input_ids.size(0)
            masked_log_likelihood = []
            for i in range(batch_size):
                n_valid = valid_labels_mask[i].sum()
                if n_valid > 0:
                    masked_log_likelihood.append(log_likelihood[i])
            
            if len(masked_log_likelihood) > 0:
                loss = -torch.stack(masked_log_likelihood).mean()
            else:
                loss = -log_likelihood.mean()
            
            predictions = self.crf.decode(emissions, mask=mask)
            
            return {'loss': loss, 'logits': emissions, 'predictions': predictions}
        else:
            predictions = self.crf.decode(emissions, mask=mask)
            return {'logits': emissions, 'predictions': predictions}

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
# CUSTOM TRAINER FOR CRF
# ============================================================================
class CRFTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override compute_loss to handle CRF model outputs"""
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.pop("labels")
        
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            loss = outputs['loss']
            predictions = outputs['predictions']
        
        max_len = inputs['input_ids'].shape[1]
        predictions_padded = []
        for pred in predictions:
            pred_tensor = torch.tensor(pred, device=inputs['input_ids'].device)
            if len(pred_tensor) < max_len:
                padding = torch.zeros(max_len - len(pred_tensor), dtype=torch.long, device=pred_tensor.device)
                pred_tensor = torch.cat([pred_tensor, padding])
            predictions_padded.append(pred_tensor)
        
        predictions_tensor = torch.stack(predictions_padded)
        
        return (loss, predictions_tensor, labels)

def compute_metrics(pred):
    predictions, labels = pred
    
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    predictions_flat = predictions.flatten()
    labels_flat = labels.flatten()
    
    mask = labels_flat != -100
    predictions_filtered = predictions_flat[mask]
    labels_filtered = labels_flat[mask]
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_filtered, predictions_filtered, average='binary', zero_division=0
    )
    acc = accuracy_score(labels_filtered, predictions_filtered)
    
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': acc}

# ============================================================================
# WANDB CALLBACK FOR OPTUNA
# ============================================================================
class WandbOptunaCallback:
    def __init__(self, model_name):
        self.model_name = model_name
        
    def __call__(self, study, trial):
        # Log each trial to wandb
        wandb.log({
            f"{self.model_name}/trial_number": trial.number,
            f"{self.model_name}/trial_value": trial.value if trial.value is not None else 0,
            f"{self.model_name}/trial_lr": trial.params.get('lr', 0),
            f"{self.model_name}/trial_weight_decay": trial.params.get('weight_decay', 0),
            f"{self.model_name}/trial_warmup_ratio": trial.params.get('warmup_ratio', 0),
            f"{self.model_name}/trial_batch_size": trial.params.get('batch_size', 0),
            f"{self.model_name}/best_value_so_far": study.best_value,
        })

# ============================================================================
# TRAINING
# ============================================================================
def train_encoder_crf(model_name, train_dataset, eval_dataset, ood_dataset, trial=None, final=False, run_name=None):
    if trial:
        lr = trial.suggest_float('lr', 1e-5, 1e-4, log=True)
        weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
        warmup_ratio = trial.suggest_float('warmup_ratio', 0.0, 0.3)
        batch_size = trial.suggest_categorical('batch_size', [8, 16])
    else:
        lr = 5e-5
        weight_decay = 0.01
        warmup_ratio = 0.1
        batch_size = cfg.BATCH_SIZE
    
    model_short_name = model_name.split('/')[-1]
    output_dir = f"{cfg.OUTPUT_DIR}/{model_short_name}"
    
    model = BERTWithCRF(model_name, num_labels=2)
    
    if final and run_name:
        run_display_name = f"{model_short_name}_{run_name}"
    else:
        run_display_name = f"{model_short_name}_trial"
    
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
        dataloader_num_workers=8,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",  # Don't log individual trials
        run_name=run_display_name,
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        logging_first_step=True,
    )
    
    trainer = CRFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    trainer.train()
    
    eval_results = trainer.evaluate(eval_dataset)
    
    if final:
        wandb.log({
            "train/final_loss": trainer.state.log_history[-1].get('loss', 0),
            "train/learning_rate": lr,
            "train/weight_decay": weight_decay,
            "train/warmup_ratio": warmup_ratio,
            "train/batch_size": batch_size,
            "dev/precision": eval_results['eval_precision'],
            "dev/recall": eval_results['eval_recall'],
            "dev/f1": eval_results['eval_f1'],
            "dev/accuracy": eval_results['eval_accuracy'],
        })
        
        if ood_dataset:
            ood_results = trainer.evaluate(ood_dataset, metric_key_prefix="ood")
            wandb.log({
                "ood/precision": ood_results['ood_precision'],
                "ood/recall": ood_results['ood_recall'],
                "ood/f1": ood_results['ood_f1'],
                "ood/accuracy": ood_results['ood_accuracy'],
            })
            return trainer, eval_results['eval_f1'], eval_results, ood_results
    
    return trainer, eval_results['eval_f1'], eval_results, None

def objective(trial, model_name, train_dataset, eval_dataset, ood_dataset):
    _, f1, _, _ = train_encoder_crf(model_name, train_dataset, eval_dataset, ood_dataset, trial=trial, final=False)
    return f1

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 80)
    print("CRF-BASED SENTENCE SPLITTING")
    print("=" * 80)
    
    wandb.login()
    
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    train_sentences, train_labels = load_data(cfg.TRAIN_FILE)
    dev_sentences, dev_labels = load_data(cfg.DEV_FILE)
    ood_sentences, ood_labels = load_data(cfg.OOD_FILE)
    
    print(f"\nTrain: {len(train_sentences)} sentences")
    print(f"Dev: {len(dev_sentences)} sentences")
    print(f"OOD: {len(ood_sentences)} sentences")
    
    all_results = []
    
    for model_name in cfg.ENCODER_MODELS:
        print("\n" + "=" * 80)
        print(f"TRAINING: {model_name}")
        print("=" * 80)
        
        model_short_name = model_name.split('/')[-1]
        
        # Initialize wandb for hyperparameter search
        tuning_run = wandb.init(
            project=cfg.WANDB_PROJECT,
            name=f"{model_short_name}_hyperparameter_search",
            config={
                "model": model_name,
                "phase": "hyperparameter_tuning",
                "n_trials": cfg.N_TRIALS,
            },
            reinit=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cfg.CACHE_DIR)
        
        print("Preparing datasets...")
        train_dataset = SentenceSplitDataset(train_sentences, train_labels, tokenizer, cfg.MAX_LENGTH, cfg.STRIDE)
        dev_dataset = SentenceSplitDataset(dev_sentences, dev_labels, tokenizer, cfg.MAX_LENGTH, cfg.STRIDE)
        ood_dataset = SentenceSplitDataset(ood_sentences, ood_labels, tokenizer, cfg.MAX_LENGTH, cfg.STRIDE)
        
        print(f"\nHyperparameter tuning for {model_short_name}...")
        study = optuna.create_study(direction='maximize', study_name=f"{model_short_name}_crf_study")
        
        # Add wandb callback
        wandb_callback = WandbOptunaCallback(model_short_name)
        
        study.optimize(
            lambda trial: objective(trial, model_name, train_dataset, dev_dataset, ood_dataset),
            n_trials=cfg.N_TRIALS,
            show_progress_bar=True,
            catch=(Exception,),
            callbacks=[wandb_callback]
        )
        
        print(f"\nBest params: {study.best_params}")
        print(f"Best F1: {study.best_value:.4f}")
        
        # Log final hyperparameter search results
        wandb.log({
            f"{model_short_name}/best_trial_value": study.best_value,
            f"{model_short_name}/best_lr": study.best_params.get('lr', 5e-5),
            f"{model_short_name}/best_weight_decay": study.best_params.get('weight_decay', 0.01),
            f"{model_short_name}/best_warmup_ratio": study.best_params.get('warmup_ratio', 0.1),
            f"{model_short_name}/best_batch_size": study.best_params.get('batch_size', 16),
        })
        
        wandb.finish()
        
        best_params = study.best_params
        
        # Generate random run name
        run_suffix = generate_run_name()
        
        print(f"\nTraining final model for {model_short_name}...")
        final_run = wandb.init(
            project=cfg.WANDB_PROJECT,
            name=f"{model_short_name}_{run_suffix}",
            config={
                "model": model_name,
                "architecture": "Encoder + CRF",
                "phase": "final_training",
                "best_params": best_params,
                "train_sentences": len(train_sentences),
                "dev_sentences": len(dev_sentences),
                "ood_sentences": len(ood_sentences),
                "max_length": cfg.MAX_LENGTH,
                "epochs": cfg.EPOCHS,
            },
            reinit=True
        )
        
        trainer, dev_f1, dev_results, ood_results = train_encoder_crf(
            model_name, train_dataset, dev_dataset, ood_dataset, 
            trial=None, final=True, run_name=run_suffix
        )
        
        trainer.save_model(f"{cfg.OUTPUT_DIR}/{model_short_name}/best_model")
        tokenizer.save_pretrained(f"{cfg.OUTPUT_DIR}/{model_short_name}/best_model")
        
        print(f"\n{model_short_name} Results:")
        print(f"  Dev F1: {dev_results['eval_f1']:.4f}")
        print(f"  Dev Acc: {dev_results['eval_accuracy']:.4f}")
        print(f"  OOD F1: {ood_results['ood_f1']:.4f}")
        print(f"  OOD Acc: {ood_results['ood_accuracy']:.4f}")
        
        all_results.append({
            'Model': model_short_name,
            'Architecture': 'Encoder + CRF',
            'Dev F1': dev_results['eval_f1'],
            'Dev Accuracy': dev_results['eval_accuracy'],
            'Dev Precision': dev_results['eval_precision'],
            'Dev Recall': dev_results['eval_recall'],
            'OOD F1': ood_results['ood_f1'],
            'OOD Accuracy': ood_results['ood_accuracy'],
            'OOD Precision': ood_results['ood_precision'],
            'OOD Recall': ood_results['ood_recall'],
            'Best LR': best_params.get('lr', 5e-5),
            'Best Weight Decay': best_params.get('weight_decay', 0.01),
            'Best Warmup Ratio': best_params.get('warmup_ratio', 0.1),
            'Best Batch Size': best_params.get('batch_size', 16),
        })
        
        wandb.log({
            "summary/dev_f1": dev_results['eval_f1'],
            "summary/ood_f1": ood_results['ood_f1'],
        })
        
        wandb.finish()
        
        del trainer, train_dataset, dev_dataset, ood_dataset, tokenizer, model
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    results_df.to_csv(f"{cfg.OUTPUT_DIR}/final_results_crf.csv", index=False)
    
    final_summary = wandb.init(project=cfg.WANDB_PROJECT, name="final_summary", reinit=True)
    wandb.log({"final_results": wandb.Table(dataframe=results_df)})
    
    best_dev_f1 = results_df.loc[results_df['Dev F1'].idxmax()]
    best_ood_f1 = results_df.loc[results_df['OOD F1'].idxmax()]
    
    wandb.log({
        "best_dev_f1_model": best_dev_f1['Model'],
        "best_dev_f1_score": best_dev_f1['Dev F1'],
        "best_ood_f1_model": best_ood_f1['Model'],
        "best_ood_f1_score": best_ood_f1['OOD F1'],
    })
    
    wandb.finish()
    
    print(f"\n✅ Complete! Results saved to {cfg.OUTPUT_DIR}/final_results_crf.csv")

if __name__ == "__main__":
    main()