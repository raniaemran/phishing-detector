import pandas as pd
import hashlib
import csv
from pathlib import Path

def merge_data(spaphish_path, multilingual_path, sender_history_path, outdir="."):
    # Load Spaphish
    df_spa = pd.read_csv(spaphish_path, encoding="utf-8-sig")
    text = (df_spa["subject"].fillna("") + "\n" + df_spa["body"].fillna("")).str.strip()
    
    df1 = pd.DataFrame({
        "id": df_spa["hash"],
        "text": text,
        "lang": "es",
        "label": df_spa["Label"].astype(int),
        "source": "spaphish",
        "is_synthetic": False,
        "url_count": df_spa["url_count"],
        "attachments_count": df_spa["attachments_count"],
        "hops_count": df_spa["hops_count"],
        "persuasion_authority": df_spa["authority"],
        "persuasion_social_proof": df_spa["social_proof"],
        "persuasion_liking_similarity_deception": df_spa["liking_similarity_deception"],
        "persuasion_commitment_integrity_reciprocation": df_spa["commitment_integrity_reciprocation"],
        "persuasion_distraction": df_spa["distraction"],
        "threat_vector_note": None,
    })
    
    # Load multilingual
    df_multi = pd.read_csv(multilingual_path, encoding="utf-8-sig")
    ids = [hashlib.sha256(t.encode()).hexdigest()[:16] for t in df_multi["text"]]
    df2 = pd.DataFrame({
        "id": ids,
        "text": df_multi["text"],
        "lang": df_multi["language_code"].str.lower(),
        "label": df_multi["is_phishing"].astype(int),
        "source": "multilingual_smoke_test",
        "is_synthetic": True,
        "url_count": None,
        "attachments_count": None,
        "hops_count": None,
        "persuasion_authority": None,
        "persuasion_social_proof": None,
        "persuasion_liking_similarity_deception": None,
        "persuasion_commitment_integrity_reciprocation": None,
        "persuasion_distraction": None,
        "threat_vector_note": df_multi["threat_vector"],
    })
    
    # Merge
    merged = pd.concat([df1, df2], ignore_index=True)
    
    # Summary
    print("=== Merge summary ===")
    print(f"Total rows: {len(merged)}")
    print("\nBy source:")
    print(merged.groupby("source").size())
    print("\nBy lang:")
    print(merged.groupby("lang").size())
    print("\nLabel balance (real data):")
    real = merged[merged["source"] == "spaphish"]
    print(real["label"].value_counts(normalize=True).round(3))
    
    # Save
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "unified_phishing_corpus.csv"
    merged.to_csv(outpath, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"\nWrote {outpath} ({len(merged)} rows)")
    
    # Load sender history
    sender_df = pd.read_csv(sender_history_path, encoding="utf-8-sig")
    sender_path = outdir / "sender_history_schema.csv"
    sender_df.to_csv(sender_path, index=False)
    print(f"Wrote {sender_path} ({len(sender_df)} rows)")
    
    return merged

if __name__ == "__main__":
    merge_data(
        spaphish_path="data/Spaphish_dataset_-_DiB.csv",
        multilingual_path="data/multilingual_phishing_corpus.csv",
        sender_history_path="data/sender_history_baseline.csv",
        outdir="."
    )
