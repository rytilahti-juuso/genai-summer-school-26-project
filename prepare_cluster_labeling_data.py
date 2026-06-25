"""Prepare cluster keywords and paper examples for LLM-based cluster labeling.

Example:
    Configure the parameters in the ``main(...)`` call at the bottom, then run:
    python prepare_cluster_labeling_data.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd

SamplingStrategy = Literal["representative", "random", "first"]

ENRICHED_FILENAMES = (
    "final-data-enriched.parquet",
    "final-data-enriched-data.parquet",
)
KEYWORDS_FILENAME = "final-data-cluster-keywords.parquet"


def _find_enriched_file(folder: Path) -> Path:
    for filename in ENRICHED_FILENAMES:
        path = folder / filename
        if path.is_file():
            return path
    expected = " or ".join(ENRICHED_FILENAMES)
    raise FileNotFoundError(f"Could not find {expected} in {folder}")


def load_cluster_files(folder: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the enriched papers and cluster-keyword parquet files."""
    folder = Path(folder)
    enriched_path = _find_enriched_file(folder)
    keywords_path = folder / KEYWORDS_FILENAME

    if not keywords_path.is_file():
        raise FileNotFoundError(f"Could not find {KEYWORDS_FILENAME} in {folder}")

    enriched_df = pd.read_parquet(enriched_path)
    keywords_df = pd.read_parquet(keywords_path)

    enriched_required = {"predicted_label", "title", "abstract"}
    keywords_required = {"predicted_label", "rank", "word"}
    missing_enriched = enriched_required.difference(enriched_df.columns)
    missing_keywords = keywords_required.difference(keywords_df.columns)

    if missing_enriched:
        raise ValueError(
            f"{enriched_path.name} is missing columns: "
            f"{', '.join(sorted(missing_enriched))}"
        )
    if missing_keywords:
        raise ValueError(
            f"{keywords_path.name} is missing columns: "
            f"{', '.join(sorted(missing_keywords))}"
        )

    return enriched_df, keywords_df


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _choose_examples(
    cluster_papers: pd.DataFrame,
    keywords: list[str],
    sample_count: int,
    strategy: SamplingStrategy,
    random_seed: int,
) -> pd.DataFrame:
    papers = cluster_papers.copy()
    papers["title"] = papers["title"].map(_clean_text)
    papers["abstract"] = papers["abstract"].map(_clean_text)
    papers = papers.loc[
        papers["title"].ne("") & papers["abstract"].ne("")
    ].drop_duplicates(subset=["title", "abstract"])

    if strategy == "representative":
        keyword_patterns = [
            re.compile(
                rf"(?<!\w){re.escape(keyword.lower())}(?!\w)"
            ) for keyword in keywords
        ]

        def relevance(row: pd.Series) -> int:
            text = f"{row['title']} {row['abstract']}".lower()
            return sum(bool(pattern.search(text)) for pattern in keyword_patterns)

        papers["_relevance"] = papers.apply(relevance, axis=1)
        papers = papers.sort_values(
            ["_relevance", "title"],
            ascending=[False, True],
            kind="stable",
        )
    elif strategy == "random":
        papers = papers.sample(
            frac=1,
            random_state=random.Random(random_seed).randrange(2**32),
        )
    elif strategy != "first":
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    return papers.head(sample_count).loc[:, ["title", "abstract"]]


def build_cluster_labeling_data(
    folder: str | Path,
    keyword_count: int = 8,
    sample_count: int = 5,
    sampling_strategy: SamplingStrategy = "representative",
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Build DataFrame, tuple dictionary, and structured LLM input records.

    Returns:
        cluster_df:
            One row per cluster. ``keywords`` and ``sample_titles`` are lists,
            and ``title_abstract_pairs`` is a list of tuples.
        cluster_dict:
            Requested compact mapping:
            ``{cluster_id: {"keywords": "word1, ...", "examples": [(title,
            abstract), ...]}}``.
        llm_records:
            JSON-friendly records with explicit field names. This is generally
            the easiest structure to pass to an LLM one cluster at a time.
    """
    if keyword_count < 1 or sample_count < 1:
        raise ValueError("keyword_count and sample_count must both be positive.")

    enriched_df, keywords_df = load_cluster_files(folder)
    ranked_keywords = (
        keywords_df.sort_values(["predicted_label", "rank"], kind="stable")
        .dropna(subset=["predicted_label", "word"])
        .groupby("predicted_label", sort=True)["word"]
        .apply(lambda words: list(dict.fromkeys(map(_clean_text, words)))[:keyword_count])
    )

    rows: list[dict[str, Any]] = []
    cluster_dict: dict[int, dict[str, Any]] = {}
    llm_records: list[dict[str, Any]] = []

    for raw_label, keywords in ranked_keywords.items():
        cluster_id = int(raw_label)
        cluster_papers = enriched_df.loc[
            enriched_df["predicted_label"].eq(raw_label),
            ["title", "abstract"],
        ]
        examples_df = _choose_examples(
            cluster_papers,
            keywords,
            sample_count,
            sampling_strategy,
            random_seed + cluster_id,
        )
        pairs = list(examples_df.itertuples(index=False, name=None))
        sample_titles = examples_df["title"].tolist()
        examples = [
            {"title": title, "abstract": abstract} for title, abstract in pairs
        ]

        rows.append(
            {
                "cluster_id": cluster_id,
                "keywords": keywords,
                "keywords_text": ", ".join(keywords),
                "sample_titles": sample_titles,
                "title_abstract_pairs": pairs,
            }
        )
        cluster_dict[cluster_id] = {
            "keywords": ", ".join(keywords),
            "examples": pairs,
        }
        llm_records.append(
            {
                "cluster_id": cluster_id,
                "keywords": keywords,
                "examples": examples,
            }
        )

    cluster_df = pd.DataFrame(
        rows,
        columns=[
            "cluster_id",
            "keywords",
            "keywords_text",
            "sample_titles",
            "title_abstract_pairs",
        ],
    )
    return cluster_df, cluster_dict, llm_records


def save_outputs(
    cluster_df: pd.DataFrame,
    llm_records: list[dict[str, Any]],
    output_folder: str | Path,
) -> tuple[Path, Path]:
    """Save the analysis DataFrame and JSON-friendly LLM records."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    dataframe_path = output_folder / "cluster-labeling-data.parquet"
    json_path = output_folder / "cluster-labeling-records.json"

    cluster_df.to_parquet(dataframe_path, index=False)
    json_path.write_text(
        json.dumps(llm_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dataframe_path, json_path


def main(
    folder: str | Path,
    keyword_count: int = 8,
    sample_count: int = 5,
    sampling_strategy: SamplingStrategy = "representative",
    random_seed: int = 42,
    output_folder: str | Path | None = None,
) -> None:
    """Prepare and save cluster-labeling data using function parameters."""
    cluster_df, cluster_dict, llm_records = build_cluster_labeling_data(
        folder,
        keyword_count=keyword_count,
        sample_count=sample_count,
        sampling_strategy=sampling_strategy,
        random_seed=random_seed,
    )
    dataframe_path, json_path = save_outputs(
        cluster_df,
        llm_records,
        output_folder or folder,
    )

    print(f"Prepared {len(cluster_df)} clusters.")
    print(f"DataFrame: {dataframe_path}")
    print(f"LLM records: {json_path}")
    print(f"Dictionary keys: {list(cluster_dict)}")


if __name__ == "__main__":
    main(
        folder="interactive-plots-with-100-clusters",
        keyword_count=8,
        sample_count=5,
        sampling_strategy="representative",
        random_seed=42,
        output_folder=None,
    )
