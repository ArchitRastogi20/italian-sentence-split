#!/usr/bin/env python3
"""
XGBoost Sentence Splitter with TF-IDF features (GPU)
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report
)
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
class Config:
    TRAIN_FILE = "manzoni_train_tokens.csv"
    DEV_FILE   = "manzoni_dev_tokens.csv"
    TEST_FILE  = "OOD_test.csv"

    OUTPUT_DIR = "outputs_xgb_tfidf"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # TF-IDF
    MAX_FEATURES = 50_000
    WORD_NGRAMS = (1, 2)
    CHAR_NGRAMS = (3, 5)

    # XGBoost
    MAX_DEPTH = 10
    LEARNING_RATE = 0.05
    N_ESTIMATORS = 500
    SUBSAMPLE = 0.8
    COLSAMPLE = 0.8
    MIN_CHILD_WEIGHT = 3

cfg = Config()

# =============================================================================
# DATA LOADING
# =============================================================================
def load_data(filepath):
    data = []

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Remove empty lines
    lines = [l.strip() for l in lines if l.strip()]

    # Detect separator
    sep = ";" if ";" in lines[1] else ","

    # Skip possible extra title line (Pinocchio OOD)
    start_idx = 0
    if not lines[0].startswith("token"):
        start_idx += 1
    if not lines[start_idx].startswith("token"):
        start_idx += 1

    for line in lines[start_idx + 1:]:
        parts = line.split(sep)
        if len(parts) != 2:
            continue

        token, label = parts
        try:
            data.append((token, int(label)))
        except ValueError:
            continue

    # Group tokens into sentences
    sentences, labels = [], []
    cur_sent, cur_lab = [], []

    for tok, lab in data:
        cur_sent.append(tok)
        cur_lab.append(lab)
        if lab == 1:
            sentences.append(cur_sent)
            labels.append(cur_lab)
            cur_sent, cur_lab = [], []

    if cur_sent:
        sentences.append(cur_sent)
        labels.append(cur_lab)

    return sentences, labels

# =============================================================================
# FEATURE CREATION
# =============================================================================
def build_examples(sentences, labels):
    texts, y, meta = [], [], []

    for sent, labs in zip(sentences, labels):
        for i, (tok, lab) in enumerate(zip(sent, labs)):
            ctx = sent[max(0, i-2): min(len(sent), i+3)]
            text = " ".join(ctx)

            texts.append(text)
            y.append(lab)
            meta.append(tok)

    return texts, np.array(y), meta

# =============================================================================
# TRAIN
# =============================================================================
def train_xgb(X_tr, y_tr, X_dev, y_dev):
    model = xgb.XGBClassifier(
        max_depth=cfg.MAX_DEPTH,
        learning_rate=cfg.LEARNING_RATE,
        n_estimators=cfg.N_ESTIMATORS,
        subsample=cfg.SUBSAMPLE,
        colsample_bytree=cfg.COLSAMPLE,
        min_child_weight=cfg.MIN_CHILD_WEIGHT,
        tree_method="hist",
        device="cuda",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_dev, y_dev)],
        verbose=True
    )

    return model

# =============================================================================
# EVALUATION
# =============================================================================
def evaluate(model, X, y, name):
    preds = model.predict(X)

    p, r, f1, _ = precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(y, preds)

    report = classification_report(y, preds, digits=4)

    return {
        "dataset": name,
        "precision": p,
        "recall": r,
        "f1": f1,
        "accuracy": acc,
        "report": report,
        "preds": preds
    }

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\nLoading data...")
    tr_s, tr_l = load_data(cfg.TRAIN_FILE)
    dv_s, dv_l = load_data(cfg.DEV_FILE)
    ts_s, ts_l = load_data(cfg.TEST_FILE)

    print("Building TF-IDF examples...")
    Xtr_txt, y_tr, tr_tok = build_examples(tr_s, tr_l)
    Xdv_txt, y_dv, dv_tok = build_examples(dv_s, dv_l)
    Xts_txt, y_ts, ts_tok = build_examples(ts_s, ts_l)

    print("Vectorizing...")
    vectorizer = TfidfVectorizer(
        max_features=cfg.MAX_FEATURES,
        ngram_range=cfg.WORD_NGRAMS,
        analyzer="word"
    )

    X_tr = vectorizer.fit_transform(Xtr_txt)
    X_dv = vectorizer.transform(Xdv_txt)
    X_ts = vectorizer.transform(Xts_txt)

    print("Training XGBoost (GPU)...")
    model = train_xgb(X_tr, y_tr, X_dv, y_dv)

    print("Evaluating...")
    res_dev = evaluate(model, X_dv, y_dv, "dev")
    res_test = evaluate(model, X_ts, y_ts, "test")

    # -------------------------------------------------------------------------
    # SAVE METRICS
    # -------------------------------------------------------------------------
    metrics = pd.DataFrame([
        {k: v for k, v in res_dev.items() if k not in ("report", "preds")},
        {k: v for k, v in res_test.items() if k not in ("report", "preds")},
    ])
    metrics.to_csv(f"{cfg.OUTPUT_DIR}/metrics.csv", index=False)

    # -------------------------------------------------------------------------
    # SAVE REPORTS
    # -------------------------------------------------------------------------
    with open(f"{cfg.OUTPUT_DIR}/classification_report.txt", "w") as f:
        f.write("=== DEV SET ===\n")
        f.write(res_dev["report"])
        f.write("\n\n=== TEST SET ===\n")
        f.write(res_test["report"])

    # -------------------------------------------------------------------------
    # SAVE PREDICTIONS
    # -------------------------------------------------------------------------
    pd.DataFrame({
        "token": dv_tok,
        "label": y_dv,
        "prediction": res_dev["preds"]
    }).to_csv(f"{cfg.OUTPUT_DIR}/dev_predictions.csv", index=False)

    pd.DataFrame({
        "token": ts_tok,
        "label": y_ts,
        "prediction": res_test["preds"]
    }).to_csv(f"{cfg.OUTPUT_DIR}/test_predictions.csv", index=False)

    print("\nDone.")
    print(f"Results saved to: {cfg.OUTPUT_DIR}")

if __name__ == "__main__":
    main()
