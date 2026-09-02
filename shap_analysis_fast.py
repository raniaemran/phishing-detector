"""
Fast SHAP analysis on phishing corpus.
Samples 300 rows, uses limited features, runs in <60 seconds.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from stylometric_features import build_stylometric_pipeline

def run_fast_shap(
    texts: list[str],
    langs: list[str],
    labels: list[int],
    top_n: int = 20,
    output_prefix: str = "shap_fast",
):
    X = list(zip(texts, langs))
    y = np.array(labels)

    from lightgbm import LGBMClassifier
    from sklearn.pipeline import Pipeline
    from stylometric_features import StylometricTransformer

    pipe = Pipeline([
        ("stylo", StylometricTransformer(top_k_pos_bigrams=80, top_k_dep=30, top_k_morph=50)),
        ("clf", LGBMClassifier(
            n_estimators=80,
            num_leaves=12,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )),
    ])

    print("Fitting model on", len(X), "samples...")
    pipe.fit(X, y)

    stylo = pipe.named_steps["stylo"]
    clf = pipe.named_steps["clf"]
    feature_names = stylo.get_feature_names_out()
    X_transformed = stylo.transform(X)

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    plt.figure(figsize=(8, 6))
    top = importance.head(top_n)
    plt.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
    plt.xlabel("mean |SHAP value| (impact on phishing-risk output)")
    plt.title(f"Top {top_n} discriminative stylometric features")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_bar.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        max_display=top_n,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_beeswarm.png", dpi=150)
    plt.close()

    print(f"Saved {output_prefix}_bar.png and {output_prefix}_beeswarm.png")
    print("\nTop 10 features:\n", importance.head(10).to_string(index=False))

    return importance, pipe


if __name__ == "__main__":
    df = pd.read_csv("unified_phishing_corpus.csv")
    df = df[df["source"] == "spaphish"]

    print(f"Total real emails: {len(df)}")
    print(f"Phishing: {df['label'].sum()}, Benign: {len(df) - df['label'].sum()}")

    df_sample = df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(min(len(x), 150), random_state=42)
    ).reset_index(drop=True)

    print(f"Sampled: {len(df_sample)} rows")
    print(f"Phishing: {df_sample['label'].sum()}, Benign: {len(df_sample) - df_sample['label'].sum()}")

    texts = df_sample["text"].tolist()
    langs = df_sample["lang"].tolist()
    labels = df_sample["label"].tolist()

    run_fast_shap(
        texts=texts,
        langs=langs,
        labels=labels,
        top_n=20,
        output_prefix="shap_fast",
    )
