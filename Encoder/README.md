# Encoder-Based Sentence Boundary Detection

This module implements and evaluates **encoder-based approaches** for sentence boundary detection on Italian literary text.
The task is formulated as **token-level binary classification**, where each token is labeled as either ending a sentence (`1`) or not (`0`).

The encoder-based experiments complement the decoder-based prompting strategies and serve as strong supervised baselines.

---

## Problem Definition

Given a sequence of tokens
[
t_1, t_2, \dots, t_n
]
the objective is to predict a label for each token:

* `0` → not a sentence boundary
* `1` → sentence boundary

This is a challenging task due to:

* Severe class imbalance (~3–4% positive labels)
* Ambiguity of punctuation (abbreviations, dialogue, ellipses)
* Long-range contextual dependencies in literary text

---

## Dataset

The experiments use Italian literary text from *I Promessi Sposi* (Manzoni), with an additional **out-of-domain (OOD)** evaluation set from *Pinocchio*.

* **Train / Validation**

  * Token-level CSV (`token`, `label`)
  * ~75k train tokens, ~9k validation tokens
  * ~96.7% non-boundary tokens

* **OOD Test**

  * Different literary style
  * Evaluates cross-domain generalization

Dataset statistics, preprocessing details, and error handling are documented in the accompanying report .

---

## Models Implemented

### Transformer Encoders

Fine-tuned encoder models for binary token classification:

* Italian BERT (`dbmdz/bert-base-italian-xxl-cased`)
* ModernBERT
* XLM-RoBERTa

Each model is trained in two variants:

* **Base classifier**
* **CRF-enhanced classifier** (structured prediction over label sequences)

### Embedding + XGBoost Models

Feature-based models using:

* Contextual embeddings (Italian BERT, MiniLM)
* Hand-crafted features (punctuation, capitalization, position)
* Gradient-boosted decision trees (XGBoost)

These models trade some expressiveness for **strong OOD robustness and efficiency**.

---

## Available Pretrained Models

The following pretrained encoder models are supported:

```python
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
    }
}
```

---

## Repository Structure

```text
Encoder/
├── data/                   # Tokenized datasets
├── eda/                    # Exploratory data analysis
├── model_src/              # Training scripts
│   ├── train_encoders.py
│   ├── train_encoders_crf.py
│   ├── train_xgboost_simple.py
│   └── train_tfidf_xgboost.py
├── models/                 # Saved checkpoints
├── results/                # Validation and OOD results
├── test/                   # Inference and evaluation scripts
├── sentence_splitting_inference.ipynb
└── README.md
```

---

## Training Models

### Train Encoder Models

```bash
python model_src/train_encoders.py
python model_src/train_encoders_crf.py
```

### Train XGBoost Models

```bash
python model_src/train_xgboost_simple.py
python model_src/train_tfidf_xgboost.py
```

---

## Evaluation & Inference

Run inference on trained models:

```bash
python test/test_inference.py
python test/test_xgboost.py
```

All results are saved under:

```text
results/
├── encoder_valid_results/
├── encoder_test_results/
└── xgboost/
```

Metrics include **precision, recall, F1**, and accuracy (reported with caution due to class imbalance).

---

## Summary

Encoder-based models provide strong supervised baselines for sentence boundary detection.
While transformer encoders excel in-domain, embedding-based XGBoost models show **superior OOD robustness**, making them a strong alternative when generalization is critical.




