import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
import umap
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
OUTPUT_PATH = "final_load.parquet"

# Backup directory and backup frequency
BACKUP_DIR = "backups"
BACKUP_EVERY_N_CALLS = 50   # write a backup every N API calls


def preprocess_text(text: str) -> str:
    """
    Preprocess text by:
      - Lowercasing
      - Removing partial Markdown symbols (*, #, `, _)
      - Collapsing multiple spaces to a single space
    """
    text = text.lower()
    text = re.sub(r'[\*\#\`\_]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def save_backup(records: List[Dict[str, Any]], label: str) -> None:
    """Write a backup CSV for results (including metadata_json)."""
    if not records:
        return

    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{label}_{ts}"

    backup_main = os.path.join(BACKUP_DIR, f"{base}.parquet")
    pd.DataFrame(records).to_parquet(backup_main)

    print(f"[backup] Saved backup to {backup_main}")


def get_umap_reducer(n_components: int, n_neighbors: int, min_dist: float, random_state: int):
    """
    Returns a UMAP reducer with the specified parameters.
    """
    return umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )


def main():
    all_records: List[Dict[str, Any]] = []
    # Final save
    if all_records:
        pd.DataFrame(all_records).to_parquet(
            OUTPUT_PATH,
        )
        print(f"\nSaved final results (including metadata_json) to {OUTPUT_PATH}")
    else:
        print("\nNo records were generated.")


    # ----------------------------------------
    # 2) Embeddings
    # ----------------------------------------
    print("Loading model and embedding texts...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    sentences = ["This is an example sentence", "Each sentence is converted", "This is a test", "Very simple"]
    embeddings = model.encode(sentences)
    #print(embeddings)
    umap_reducer = get_umap_reducer(min_dist=0.1,n_components=2,random_state=42,n_neighbors=2)
    umap_result = umap_reducer.fit_transform(embeddings)
    print(umap_result)
    sentence_dict = {}
    plt.figure(figsize=(24, 16))
    plt.title(f"Semantic Similarity Visualization TEST")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    # Plot all sentence coordinates.
    plt.scatter(
        umap_result[:, 0],
        umap_result[:, 1],
        s=100,
    )
    
    # Add each sentence as the label for its point.
    for sentence, (x, y) in zip(sentences, umap_result):
        plt.annotate(
            sentence,
            xy=(x, y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "alpha": 0.7,
            },
            arrowprops={
                "arrowstyle": "-",
                "alpha": 0.5,
            },
        )
    plt.tight_layout()
    plt.show()
                
    

            
            
if __name__ == "__main__":
    main()

    """
                if overlaps:
                    plt.scatter(
                        points_2d[overlaps, 0],
                        points_2d[overlaps, 1],
                        color="black",
                        marker="x",
                        s=150,
                        label=f"{tag} Overlap"
                    )
                """