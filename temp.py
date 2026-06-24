from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pacmap
import pandas as pd
import plotly.express as px
import umap
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan

MOCK_DATA_PATH = "test-data.parquet"
REAL_DATA_PATH = "result.parquet"
ENRICHED_DATA_PATH = "test-data-enriched.parquet"
CLUSTER_KEYWORDS_PATH = "cluster-keywords.parquet"
INTERACTIVE_PLOTS_DIR = "interactive-plots"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_STATE = 42
OLDER_OUTLIER_COLOR = "#B5485D"
from mockUpData import MOCK_ARXIV_RECORDS
print(len(MOCK_ARXIV_RECORDS))
ReducerName = Literal["umap", "pacmap"]


def create_mock_parquet(path: str = MOCK_DATA_PATH) -> pd.DataFrame:
    """Create the example arXiv entries as a parquet file."""
    df = pd.DataFrame(MOCK_ARXIV_RECORDS)
    df.to_parquet(path, index=False)
    return df


def load_arxiv_parquet(path: str = MOCK_DATA_PATH) -> pd.DataFrame:
    """Load arXiv parquet data and drop rows with missing or empty required fields."""
    df: pd.DataFrame = pd.read_parquet(path)

    df = (
        df.rename(columns={"summary": "abstract"})
        .loc[:, ["id", "title", "abstract", "published"]]
        .copy()
    )

    required_columns = ["id", "title", "abstract", "published"]

    non_empty_mask = df[required_columns].notna().all(axis=1)

    for column in required_columns:
        if pd.api.types.is_string_dtype(df[column]) or df[column].dtype == object:
            non_empty_mask &= df[column].astype("string").str.strip().ne("")

    dropped_rows = len(df) - int(non_empty_mask.sum())
    print(f"Dropped {dropped_rows} rows with missing or empty required fields.")

    return df.loc[non_empty_mask].copy()

def preprocess_text(text: str) -> str:
    """Clean text before embedding."""
    text = str(text).lower()
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


def add_reduced_embeddings(
    df: pd.DataFrame,
    n_components: int,
    source_column: str = "embedding",
    output_column: str = "reduced_embedding",
) -> pd.DataFrame:
    """Apply UMAP to stored embeddings and save the reduced vectors."""
    result = df.copy()
    embeddings = np.vstack(result[source_column].to_numpy())
    n_samples, n_features = embeddings.shape
    maximum_components = min(n_samples - 2, n_features - 1)

    if not 1 <= n_components <= maximum_components:
        raise ValueError(
            "UMAP dimensions must be at least 1, lower than the original "
            f"embedding size, and at least two below the sample count; received "
            f"{n_components}, maximum allowed is {maximum_components}."
        )

    n_neighbors = max(2, min(15, n_samples - 1))
    reduced_embeddings = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric="cosine",
        random_state=RANDOM_STATE,
    ).fit_transform(embeddings)
    result[output_column] = [
        embedding.astype(float).tolist() for embedding in reduced_embeddings
    ]
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


def publication_timeline_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Grade dates from the 5th percentile through newest and flag older records."""
    published = pd.to_datetime(df["published"], utc=True, errors="coerce")
    valid_published = published.dropna()

    timeline_grade = pd.Series(np.nan, index=df.index, dtype=float)
    older_outlier = pd.Series(False, index=df.index, dtype=bool)
    timeline_status = pd.Series(
        "Publication date unavailable", index=df.index, dtype="string"
    )

    if not valid_published.empty:
        gradient_start = valid_published.quantile(0.05)
        newest_publication = valid_published.max()
        grading_range_seconds = (
            newest_publication - gradient_start
        ).total_seconds()

        if grading_range_seconds:
            timeline_grade = (
                (published - gradient_start).dt.total_seconds()
                / grading_range_seconds
            ).clip(0.0, 1.0)
        else:
            timeline_grade.loc[published.notna()] = 0.5

        older_outlier = published.lt(gradient_start).fillna(False)
        timeline_status.loc[published.notna()] = "Within timeline grading"
        timeline_status.loc[older_outlier] = (
            "Older outlier (lowest 5% excluded from gradient)"
        )

    return pd.DataFrame(
        {
            "relative_timeline": timeline_grade,
            "timeline_outlier": older_outlier,
            "timeline_status": timeline_status,
        },
        index=df.index,
    )


def relative_timeline(df: pd.DataFrame) -> pd.Series:
    """Scale publication dates from the 5th percentile to newest."""
    return publication_timeline_metadata(df)["relative_timeline"]


def add_predictions(
    df: pd.DataFrame,
    reductions: dict[str, np.ndarray],
    embedding_column: str = "embedding",
) -> pd.DataFrame:
    """Add labels, timeline, and reducer coordinates without changing row order."""
    result = df.copy()
    embeddings = np.vstack(result[embedding_column].to_numpy())

    result["predicted_label"] = cluster_embeddings(embeddings)
    timeline_metadata = publication_timeline_metadata(result)
    for column in timeline_metadata.columns:
        result[column] = timeline_metadata[column].to_numpy()

    for reducer_name, coordinates in reductions.items():
        result[f"{reducer_name}_x"] = coordinates[:, 0]
        result[f"{reducer_name}_y"] = coordinates[:, 1]

    return result


def extract_cluster_keywords(
    df: pd.DataFrame, top_n: int = 5
) -> pd.DataFrame:
    """Find each cluster's top TF-IDF words and documents containing them."""
    records = []

    for cluster_label, cluster_df in df.groupby(
        "predicted_label", sort=False, dropna=False
    ):
        documents = cluster_df["abstract_cleaned"].fillna("").tolist()
        vectorizer = TfidfVectorizer(stop_words="english")

        try:
            tfidf_matrix = vectorizer.fit_transform(documents)
        except ValueError:
            continue

        terms = vectorizer.get_feature_names_out()
        cluster_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
        top_indices = cluster_scores.argsort()[::-1][:top_n]

        for rank, term_index in enumerate(top_indices, start=1):
            if cluster_scores[term_index] <= 0:
                continue

            matching_rows = tfidf_matrix[:, term_index].nonzero()[0]
            document_ids = cluster_df.iloc[matching_rows]["id"].tolist()

            records.append(
                {
                    "predicted_label": int(cluster_label),
                    "rank": rank,
                    "word": terms[term_index],
                    "mean_tfidf_score": float(cluster_scores[term_index]),
                    "document_count": len(document_ids),
                    "document_ids": document_ids,
                }
            )

    return pd.DataFrame(
        records,
        columns=[
            "predicted_label",
            "rank",
            "word",
            "mean_tfidf_score",
            "document_count",
            "document_ids",
        ],
    )


def save_interactive_reductions(
    df: pd.DataFrame,
    reductions: dict[str, np.ndarray],
    output_dir: str | Path = INTERACTIVE_PLOTS_DIR,
) -> list[Path]:
    """Save uncluttered reductions as individual HTML files with hover details."""
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    hover_data = {
        "id": True,
        "published": True,
        "predicted_label": True,
        "relative_timeline": ":.3f",
        "Dimension 1": ":.3f",
        "Dimension 2": ":.3f",
    }
    timeline_hover_data = {
        **hover_data,
        "timeline_status": True,
        "timeline_outlier": False,
    }

    for reducer_name, coordinates in reductions.items():
        display_name = reducer_name.upper() if reducer_name == "umap" else "PaCMAP"
        plot_df = df.copy()
        plot_df["Dimension 1"] = coordinates[:, 0]
        plot_df["Dimension 2"] = coordinates[:, 1]
        # Treat cluster IDs as categories, including HDBSCAN's -1 noise label.
        plot_df["cluster"] = plot_df["predicted_label"].map(
            lambda label: "Noise" if label == -1 else f"Cluster {label}"
        )

        cluster_figure = px.scatter(
            plot_df,
            x="Dimension 1",
            y="Dimension 2",
            color="cluster",
            hover_name="title",
            hover_data=hover_data,
            title=f"{display_name} HDBSCAN clusters",
            labels={"cluster": "Cluster"},
            template="plotly_white",
        )
        cluster_figure.update_traces(marker={"size": 10, "opacity": 0.82})

        timeline_figure = px.scatter(
            plot_df.loc[~plot_df["timeline_outlier"]],
            x="Dimension 1",
            y="Dimension 2",
            color="relative_timeline",
            color_continuous_scale="Viridis",
            range_color=(0.0, 1.0),
            hover_name="title",
            hover_data=timeline_hover_data,
            title=f"{display_name} publication timeline (5th percentile to newest)",
            labels={
                "relative_timeline": "Publication date (5th percentile to newest)",
                "timeline_status": "Timeline status",
            },
            template="plotly_white",
        )
        timeline_figure.update_traces(marker={"size": 10, "opacity": 0.82})

        older_outliers = plot_df.loc[plot_df["timeline_outlier"]]
        if not older_outliers.empty:
            outlier_figure = px.scatter(
                older_outliers,
                x="Dimension 1",
                y="Dimension 2",
                hover_name="title",
                hover_data=timeline_hover_data,
                labels={"timeline_status": "Timeline status"},
            )
            outlier_figure.update_traces(
                marker={
                    "size": 10,
                    "opacity": 0.9,
                    "color": OLDER_OUTLIER_COLOR,
                    "line": {"color": "white", "width": 0.7},
                },
                name="Older outlier",
                showlegend=True,
            )
            timeline_figure.add_traces(outlier_figure.data)

        for suffix, figure in (
            ("clusters", cluster_figure),
            ("timeline", timeline_figure),
        ):
            figure.update_layout(hoverlabel={"namelength": -1})
            output_path = output_directory / f"{reducer_name}-{suffix}.html"
            figure.write_html(
                output_path,
                include_plotlyjs=True,
                full_html=True,
                config={"displaylogo": False, "scrollZoom": True},
            )
            saved_paths.append(output_path)

    return saved_paths


def main(
    reducer_names: Sequence[ReducerName],
    output_path: str = ENRICHED_DATA_PATH,
    cluster_keywords_path: str = CLUSTER_KEYWORDS_PATH,
    interactive_plots_dir: str | Path = INTERACTIVE_PLOTS_DIR,
    embedding_reduction_dimensions: int | None = None,
) -> None:
    if not reducer_names:
        raise ValueError("Select at least one reducer: 'umap' or 'pacmap'.")

    # create_mock_parquet(MOCK_DATA_PATH)
    print("start")
    arxiv_df = load_arxiv_parquet(REAL_DATA_PATH)
    original_id_order = arxiv_df["id"].tolist()
    print("data has been loaded")
    arxiv_df = add_clean_text_and_embeddings(arxiv_df)
    print("data has been cleaned")
    print("embeddings have been done")

    embedding_column = "embedding"
    if embedding_reduction_dimensions is not None:
        arxiv_df = add_reduced_embeddings(
            arxiv_df,
            n_components=embedding_reduction_dimensions,
        )
        embedding_column = "reduced_embedding"
        print(
            "Reduced embeddings with UMAP to "
            f"{embedding_reduction_dimensions} dimensions."
        )

    embeddings = np.vstack(arxiv_df[embedding_column].to_numpy())
    reductions = fit_reducers(embeddings, reducer_names)
    arxiv_df = add_predictions(
        arxiv_df,
        reductions,
        embedding_column=embedding_column,
    )
    cluster_keywords_df = extract_cluster_keywords(arxiv_df)

    if arxiv_df["id"].tolist() != original_id_order:
        raise RuntimeError("Row order changed while enriching the data.")

    arxiv_df.to_parquet(output_path, index=False)
    cluster_keywords_df.to_parquet(cluster_keywords_path, index=False)
    print(f"Saved enriched data to {output_path} with row order preserved.")
    print(f"Saved cluster TF-IDF keywords to {cluster_keywords_path}.")

    print(f"Created {MOCK_DATA_PATH} with {len(MOCK_ARXIV_RECORDS)} rows.")
    print(arxiv_df[["id", "title", "abstract_cleaned", "predicted_label"]])

    plot_paths = save_interactive_reductions(
        arxiv_df,
        reductions,
        output_dir=interactive_plots_dir,
    )
    for plot_path in plot_paths:
        print(f"Saved interactive plot to {plot_path}.")


if __name__ == "__main__":
    # Select ("umap",), ("pacmap",), or ("umap", "pacmap").
    # Set embedding_reduction_dimensions to an integer (for example, 50)
    # to apply UMAP before clustering and visualization; use None to disable it.
    main(
        ("umap",),
        output_path="test-data-enriched.parquet",
        cluster_keywords_path="cluster-keywords.parquet",
        embedding_reduction_dimensions=8,
    )
