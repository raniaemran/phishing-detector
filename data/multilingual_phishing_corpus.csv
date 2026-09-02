"""
End-to-end pipeline execution script for stylometric phishing detection.
Integrates dataset loading, stylometric feature extraction, and LightGBM training.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from lightgbm import LGBMClassifier

from stylometric_features import StylometricTransformer


def run_pipeline(corpus_path: str):
    print(f"Loading corpus from {corpus_path}...")
    df = pd.read_csv(corpus_path, encoding="utf-8-sig")

    # Filter to real SpaPhish data for robust training
    df_real = df[df["source"] == "spaphish"].dropna(subset=["text", "label"])
    
    texts = df_real["text"].tolist()
    langs = df_real["lang"].tolist()
    labels = df_real["label"].astype(int).values

    X = list(zip(texts, langs))
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Extracting stylometric features and training LightGBM...")
    transformer = StylometricTransformer()
    X_train_feats = transformer.fit_transform(X_train)
    X_test_feats = transformer.transform(X_test)

    clf = LGBMClassifier(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train_feats, y_train)

    preds = clf.predict(X_test_feats)
    probs = clf.predict_proba(X_test_feats)[:, 1]

    print("\n=== Evaluation Results ===")
    print(classification_report(y_test, preds, target_names=["Legitimate", "Phishing"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="unified_phishing_corpus.csv")
    args = parser.parse_args()
    run_pipeline(args.corpus)
