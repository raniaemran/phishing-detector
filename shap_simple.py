"""
Simple SHAP analysis using TF-IDF features (no spacy required).
Runs in seconds and saves bar, beeswarm, and summary plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def run_simple_shap(
    texts: list[str],
    labels: list[int],
    top_n: int = 20,
    output_prefix: str = "shap_simple",
):
    # Build pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=500,
            stop_words='english',
            min_df=1,
        )),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')),
    ])

    print(f"Fitting model on {len(texts)} samples...")
    pipe.fit(texts, labels)

    tfidf = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()
    X_transformed = tfidf.transform(texts).toarray()

    print("Computing SHAP values...")
    explainer = shap.LinearExplainer(clf, X_transformed, feature_names=feature_names)
    shap_values = explainer.shap_values(X_transformed)

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    print("Top features:")
    print(importance.head(10).to_string(index=False))

    # --- BAR PLOT ---
    plt.figure(figsize=(10, 8))
    top = importance.head(top_n)
    plt.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
    plt.xlabel("mean |SHAP value| (impact on phishing-risk output)")
    plt.title(f"Top {top_n} discriminative TF-IDF features")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_prefix}_bar.png")

    # --- BEESWARM PLOT ---
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        max_display=top_n,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_beeswarm.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_prefix}_beeswarm.png")

    # --- SUMMARY PLOT (all features) ---
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_prefix}_summary.png")

    return importance


if __name__ == "__main__":
    df = pd.read_csv("data/Spaphish_dataset_-_DiB.csv")

    print(f"Total emails: {len(df)}")
    print(f"Phishing: {df['label'].sum()}, Benign: {len(df) - df['label'].sum()}")

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    run_simple_shap(
        texts=texts,
        labels=labels,
        top_n=20,
        output_prefix="shap_simple",
    )
