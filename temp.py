from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pacmap
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
import hdbscan

MOCK_DATA_PATH = "test-data.parquet"
ENRICHED_DATA_PATH = "test-data-enriched.parquet"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_STATE = 42
from mockUpData import MOCK_ARXIV_RECORDS
print(len(MOCK_ARXIV_RECORDS))
ReducerName = Literal["umap", "pacmap"]


def create_mock_parquet(path: str = MOCK_DATA_PATH) -> pd.DataFrame:
    """Create the example arXiv entries as a parquet file."""
    df = pd.DataFrame(MOCK_ARXIV_RECORDS)
    df.to_parquet(path, index=False)
    return df


def load_arxiv_parquet(path: str = MOCK_DATA_PATH) -> pd.DataFrame:
    """Load arXiv parquet data and keep DOI, title, and abstract columns."""
    df = pd.read_parquet(path)
    return (
        df.rename(columns={"summary": "abstract"})
        .loc[:, ["id", "title", "abstract", "published"]]
        .copy()
    )


def preprocess_text(text: str) -> str:
    """Clean text before embedding."""
    text = text.lower()
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"[\*\#\`\_{}^\\]+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9.,;:()/%+\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def add_clean_text_and_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """Add cleaned abstracts and sentence-transformer embeddings."""
    model = SentenceTransformer(MODEL_NAME)
    result = df.copy()
    result["abstract_cleaned"] = result["abstract"].map(preprocess_text)
    embeddings = model.encode(result["abstract_cleaned"].tolist(), show_progress_bar=False)
    result["embedding"] = [embedding.astype(float).tolist() for embedding in embeddings]
    return result


def fit_reducers(
    embeddings: np.ndarray, reducer_names: Sequence[ReducerName]
) -> dict[str, np.ndarray]:
    """Fit the selected reducers and return two-dimensional coordinates."""
    n_samples = len(embeddings)
    n_neighbors = max(2, min(3, n_samples - 1))
    reductions = {}

    if "pacmap" in reducer_names:
        reductions["pacmap"] = pacmap.PaCMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            MN_ratio=0.5,
            FP_ratio=1.0,
            random_state=RANDOM_STATE,
        ).fit_transform(embeddings)

    if "umap" in reducer_names:
        reductions["umap"] = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            random_state=RANDOM_STATE,
        ).fit_transform(embeddings)

    return reductions


def cluster_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Cluster embeddings with HDBSCAN."""
    min_cluster_size = 2 if len(embeddings) < 10 else 5
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1)
    return clusterer.fit_predict(embeddings)


def relative_timeline(df: pd.DataFrame) -> pd.Series:
    """Scale publication dates from earliest 0.0 to latest 1.0."""
    published = pd.to_datetime(df["published"], utc=True)
    elapsed = (published - published.min()).dt.total_seconds()
    total = elapsed.max()
    return elapsed / total if total else elapsed


def add_predictions(
    df: pd.DataFrame, reductions: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Add labels, timeline, and reducer coordinates without changing row order."""
    result = df.copy()
    embeddings = np.vstack(result["embedding"].to_numpy())

    result["predicted_label"] = cluster_embeddings(embeddings)
    result["relative_timeline"] = relative_timeline(result).to_numpy()

    for reducer_name, coordinates in reductions.items():
        result[f"{reducer_name}_x"] = coordinates[:, 0]
        result[f"{reducer_name}_y"] = coordinates[:, 1]

    return result


def plot_reductions(df: pd.DataFrame, reductions: dict[str, np.ndarray]) -> None:
    """Show selected reductions colored by cluster and relative timeline."""
    labels = df["predicted_label"]
    print("labels are:")
    print(labels.to_numpy())
    timeline = df["relative_timeline"]

    fig, axes = plt.subplots(
        len(reductions),
        2,
        figsize=(14, 5 * len(reductions)),
        constrained_layout=True,
        squeeze=False,
    )
    plot_specs = []
    for reducer_name in reductions:
        display_name = reducer_name.upper() if reducer_name == "umap" else "PaCMAP"
        plot_specs.extend(
            [
                (reducer_name, f"{display_name} HDBSCAN", labels, "tab10"),
                (
                    reducer_name,
                    f"{display_name} relative timeline",
                    timeline,
                    "viridis",
                ),
            ]
        )

    for ax, (reducer_name, title, colors, cmap) in zip(axes.flat, plot_specs):
        coords = reductions[reducer_name]
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, cmap=cmap, s=110)
        ax.set_title(title)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")

        for row, (x, y) in zip(df.itertuples(), coords):
            ax.annotate(
                row.title[:42],
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        if "timeline" in title:
            fig.colorbar(scatter, ax=ax, label="Relative publication date")

    plt.show()


def main(
    reducer_names: Sequence[ReducerName],
    output_path: str = ENRICHED_DATA_PATH,
) -> None:
    if not reducer_names:
        raise ValueError("Select at least one reducer: 'umap' or 'pacmap'.")

    create_mock_parquet(MOCK_DATA_PATH)
    print("start")
    arxiv_df = load_arxiv_parquet(MOCK_DATA_PATH)
    original_id_order = arxiv_df["id"].tolist()
    print("data has been loaded")
    arxiv_df = add_clean_text_and_embeddings(arxiv_df)
    print("data has been cleaned")
    embeddings = np.vstack(arxiv_df["embedding"].to_numpy())
    print("embeddings have been done")
    reductions = fit_reducers(embeddings, reducer_names)
    arxiv_df = add_predictions(arxiv_df, reductions)

    if arxiv_df["id"].tolist() != original_id_order:
        raise RuntimeError("Row order changed while enriching the data.")

    arxiv_df.to_parquet(output_path, index=False)
    print(f"Saved enriched data to {output_path} with row order preserved.")

    print(f"Created {MOCK_DATA_PATH} with {len(MOCK_ARXIV_RECORDS)} rows.")
    print(arxiv_df[["id", "title", "abstract_cleaned", "predicted_label"]])

    plot_reductions(arxiv_df, reductions)


if __name__ == "__main__":
    # Select ("umap",), ("pacmap",), or ("umap", "pacmap").
    main(("umap",), output_path="test-data-enriched.parquet")
