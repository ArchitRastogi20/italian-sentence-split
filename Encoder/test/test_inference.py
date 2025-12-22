# test_inference.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
from torchcrf import CRF
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from tqdm import tqdm
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    TEST_FILE = "OOD_test.csv"
    
    MODELS = {
        "bert-italian-crf": {
            "model_path": "ArchitRastogi/bert-base-italian-xxl-cased-sentence-splitter-CRF",
            "base_model": "dbmdz/bert-base-italian-xxl-cased",
            "is_crf": True
        },
        "modernbert-crf": {
            "model_path": "ArchitRastogi/ModernBERT-italian-sentence-splitter-CRF",
            "base_model": "answerdotai/ModernBERT-base",
            "is_crf": True
        },
        "xlm-roberta-crf": {
            "model_path": "ArchitRastogi/xlm-roberta-base-italian-sentence-splitter-CRF",
            "base_model": "FacebookAI/xlm-roberta-base",
            "is_crf": True
        },
        "bert-italian-base": {
            "model_path": "ArchitRastogi/bert-base-italian-xxl-cased-sentence-splitter-base",
            "base_model": "dbmdz/bert-base-italian-xxl-cased",
            "is_crf": False
        },
        "modernbert-base": {
            "model_path": "ArchitRastogi/ModernBERT-italian-sentence-splitter-base",
            "base_model": "answerdotai/ModernBERT-base",
            "is_crf": False
        },
    }
    
    MAX_LENGTH = 512
    STRIDE = 64
    BATCH_SIZE = 16
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUTPUT_DIR = "inference_results"
    CACHE_DIR = "cache"

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(cfg.CACHE_DIR, exist_ok=True)

logger.info(f"Using device: {cfg.DEVICE}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

class BERTWithCRF(nn.Module):
    def __init__(self, base_model, num_labels=2):
        super().__init__()
        self.encoder = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        
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

def load_test_data(filepath):
    logger.info(f"Loading test data from {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
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
    logger.info(f"Loaded {len(df)} tokens")
    
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
    
    logger.info(f"Grouped into {len(sentences)} sentences")
    return sentences, labels, df

class InferenceDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=512, stride=64, use_stride=True):
        self.encodings = []
        self.labels_aligned = []
        self.original_indices = []
        
        for idx, (sent, labs) in enumerate(tqdm(zip(sentences, labels), desc="Tokenizing", total=len(sentences))):
            text = ' '.join(sent)
            
            if use_stride:
                encoding = tokenizer(
                    text, 
                    truncation=True, 
                    max_length=max_length,
                    stride=stride,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    padding='max_length'
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
                    self.original_indices.append(idx)
            else:
                encoding = tokenizer(
                    text, 
                    truncation=True, 
                    max_length=max_length,
                    padding='max_length',
                    return_offsets_mapping=True
                )
                
                self.encodings.append({
                    'input_ids': encoding['input_ids'],
                    'attention_mask': encoding['attention_mask']
                })
                
                word_ids = encoding.word_ids()
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
                self.original_indices.append(idx)
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.encodings[idx].items()}
        item['labels'] = torch.tensor(self.labels_aligned[idx])
        item['idx'] = self.original_indices[idx]
        return item

def run_inference(model, dataloader, is_crf=False):
    model.eval()
    all_predictions = []
    all_labels = []
    
    logger.info("Running inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            input_ids = batch['input_ids'].to(cfg.DEVICE)
            attention_mask = batch['attention_mask'].to(cfg.DEVICE)
            labels = batch['labels'].to(cfg.DEVICE)
            
            if is_crf:
                outputs = model(input_ids, attention_mask)
                predictions = outputs['predictions']
                
                for i, pred_seq in enumerate(predictions):
                    mask = (labels[i] != -100).cpu().numpy()
                    pred_seq_padded = pred_seq + [0] * (len(mask) - len(pred_seq))
                    pred_masked = np.array(pred_seq_padded)[mask]
                    label_masked = labels[i].cpu().numpy()[mask]
                    all_predictions.extend(pred_masked.tolist())
                    all_labels.extend(label_masked.tolist())
            else:
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                predictions = torch.argmax(logits, dim=-1)
                
                for i in range(predictions.shape[0]):
                    mask = (labels[i] != -100).cpu().numpy()
                    pred_masked = predictions[i].cpu().numpy()[mask]
                    label_masked = labels[i].cpu().numpy()[mask]
                    all_predictions.extend(pred_masked.tolist())
                    all_labels.extend(label_masked.tolist())
    
    return np.array(all_predictions), np.array(all_labels)

def compute_metrics(predictions, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    
    class_report = classification_report(
        labels, predictions, 
        target_names=['No Split (0)', 'Split (1)'],
        digits=4,
        zero_division=0
    )
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': class_report
    }

def test_model(model_name, model_config, sentences, labels):
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing model: {model_name}")
    logger.info(f"Model path: {model_config['model_path']}")
    logger.info(f"Base model: {model_config['base_model']}")
    logger.info(f"Model type: {'CRF' if model_config['is_crf'] else 'Base Encoder'}")
    logger.info(f"{'='*80}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config['base_model'],
            cache_dir=cfg.CACHE_DIR,
            trust_remote_code=True
        )
        logger.info(f"Tokenizer loaded from {model_config['base_model']}")
        
        if model_config['is_crf']:
            from huggingface_hub import hf_hub_download
            
            base_encoder = AutoModel.from_pretrained(
                model_config['base_model'],
                cache_dir=cfg.CACHE_DIR,
                trust_remote_code=True
            )
            logger.info(f"Base encoder loaded")
            
            model = BERTWithCRF(base_encoder, num_labels=2)
            logger.info(f"CRF wrapper created")
            
            try:
                model_file = hf_hub_download(
                    repo_id=model_config['model_path'],
                    filename="model.safetensors",
                    cache_dir=cfg.CACHE_DIR
                )
                logger.info(f"Loading weights from safetensors")
                from safetensors.torch import load_file
                state_dict = load_file(model_file)
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Weights loaded successfully")
            except Exception as e:
                logger.info(f"Safetensors not found, trying pytorch_model.bin: {e}")
                try:
                    files = ["pytorch_model.bin", "model.bin"]
                    loaded = False
                    for filename in files:
                        try:
                            model_file = hf_hub_download(
                                repo_id=model_config['model_path'],
                                filename=filename,
                                cache_dir=cfg.CACHE_DIR
                            )
                            logger.info(f"Loading weights from {filename}")
                            state_dict = torch.load(model_file, map_location=cfg.DEVICE)
                            model.load_state_dict(state_dict, strict=False)
                            logger.info(f"Weights loaded successfully from {filename}")
                            loaded = True
                            break
                        except Exception as e2:
                            logger.warning(f"Could not load {filename}: {e2}")
                            continue
                    
                    if not loaded:
                        logger.error("Could not load model weights from any source")
                        return None
                except Exception as e3:
                    logger.error(f"Failed to load any weights: {e3}")
                    return None
        else:
            model = AutoModelForTokenClassification.from_pretrained(
                model_config['model_path'],
                cache_dir=cfg.CACHE_DIR,
                num_labels=2,
                trust_remote_code=True
            )
            logger.info(f"Base model loaded from {model_config['model_path']}")
        
        model = model.to(cfg.DEVICE)
        logger.info(f"Model moved to {cfg.DEVICE}")
        
        use_stride = True
        logger.info(f"Using stride for tokenization: {use_stride} (all models trained with stride)")
        
        test_dataset = InferenceDataset(
            sentences, labels, tokenizer, 
            cfg.MAX_LENGTH, cfg.STRIDE, 
            use_stride=use_stride
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=cfg.BATCH_SIZE, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        predictions, true_labels = run_inference(model, test_loader, is_crf=model_config['is_crf'])
        
        metrics = compute_metrics(predictions, true_labels)
        
        logger.info(f"\nResults for {model_name}:")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"\nClassification Report:\n{metrics['classification_report']}")
        
        with open(f"{cfg.OUTPUT_DIR}/{model_name}_classification_report.txt", 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Model Path: {model_config['model_path']}\n")
            f.write(f"Base Model: {model_config['base_model']}\n")
            f.write(f"Model Type: {'CRF' if model_config['is_crf'] else 'Base Encoder'}\n")
            f.write(f"HuggingFace Link: https://huggingface.co/{model_config['model_path']}\n")
            f.write(f"\nMetrics:\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1 Score:  {metrics['f1']:.4f}\n")
            f.write(f"\nClassification Report:\n{metrics['classification_report']}\n")
        
        del model, tokenizer, test_dataset, test_loader
        torch.cuda.empty_cache()
        
        return {
            'model_name': model_name,
            'model_type': 'CRF' if model_config['is_crf'] else 'Base',
            'huggingface_link': f"https://huggingface.co/{model_config['model_path']}",
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        }
        
    except Exception as e:
        logger.error(f"Error testing model {model_name}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    logger.info("Starting inference pipeline")
    logger.info(f"Test file: {cfg.TEST_FILE}")
    
    sentences, labels, raw_df = load_test_data(cfg.TEST_FILE)
    
    logger.info(f"\nDataset Statistics:")
    logger.info(f"Total tokens: {len(raw_df)}")
    logger.info(f"Total sentences: {len(sentences)}")
    logger.info(f"Label distribution:\n{raw_df['label'].value_counts()}")
    
    results = []
    
    for model_name, model_config in cfg.MODELS.items():
        result = test_model(model_name, model_config, sentences, labels)
        if result:
            results.append(result)
    
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('f1', ascending=False)
        
        logger.info("\n" + "="*80)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info("="*80)
        logger.info("\n" + results_df.to_string(index=False))
        
        output_csv = f"{cfg.OUTPUT_DIR}/inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_csv, index=False)
        logger.info(f"\nResults saved to: {output_csv}")
        
        logger.info("\nBest performing model:")
        best_model = results_df.iloc[0]
        logger.info(f"Model: {best_model['model_name']}")
        logger.info(f"F1 Score: {best_model['f1']:.4f}")
        logger.info(f"Accuracy: {best_model['accuracy']:.4f}")
        logger.info(f"Precision: {best_model['precision']:.4f}")
        logger.info(f"Recall: {best_model['recall']:.4f}")
        logger.info(f"HuggingFace: {best_model['huggingface_link']}")
    else:
        logger.error("No results generated. Please check the errors above.")
    
    logger.info("\nInference pipeline completed successfully")

if __name__ == "__main__":
    main()