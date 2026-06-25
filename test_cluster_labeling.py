import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from llm_connection import FALLBACK_LABEL, label_clusters
from prepare_cluster_labeling_data import build_cluster_labeling_data
from temp import (
    add_generated_labels,
    create_label_distribution_figure,
)


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(json.loads(prompt))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class ClusterLabelingTests(unittest.TestCase):
    def test_label_clusters_keeps_input_ids_and_checkpoints_failures(self):
        clusters = [
            {"cluster_id": 7, "keywords": ["alpha"], "examples": []},
            {"cluster_id": 8, "keywords": ["beta"], "examples": []},
            {"cluster_id": 9, "keywords": ["gamma"], "examples": []},
        ]
        llm = FakeLLM(
            [
                "<CLUSTER>999</CLUSTER><LABEL>Alpha research</LABEL>"
                "<REASONING>Shared alpha methods.</REASONING>",
                "<LABEL>Missing reasoning</LABEL>",
                RuntimeError("offline"),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "results.json"
            results = label_clusters(llm, clusters, checkpoint)
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))

        self.assertEqual([item["cluster_id"] for item in results], [7, 8, 9])
        self.assertEqual(results[0]["generated_label"], "Alpha research")
        self.assertEqual(results[1]["generated_label"], FALLBACK_LABEL)
        self.assertIn("missing REASONING", results[1]["justification"])
        self.assertEqual(results[2]["generated_label"], FALLBACK_LABEL)
        self.assertIn("RuntimeError", results[2]["justification"])
        self.assertEqual(saved, results)

    def test_labeling_input_includes_noise_without_keywords(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            pd.DataFrame(
                {
                    "predicted_label": [-1, 0],
                    "title": ["Noise paper", "Cluster paper"],
                    "abstract": ["Noise abstract", "Cluster abstract"],
                }
            ).to_parquet(folder / "final-data-enriched.parquet", index=False)
            pd.DataFrame(
                {
                    "predicted_label": [0],
                    "rank": [1],
                    "word": ["cluster"],
                }
            ).to_parquet(
                folder / "final-data-cluster-keywords.parquet",
                index=False,
            )

            _, _, records = build_cluster_labeling_data(folder)

        self.assertEqual([record["cluster_id"] for record in records], [-1, 0])
        self.assertEqual(records[0]["keywords"], [])
        self.assertEqual(records[0]["examples"][0]["title"], "Noise paper")

    def test_distribution_includes_noise_and_uses_full_denominator(self):
        source = pd.DataFrame(
            {"predicted_label": [-1, 0, 0, 1]}
        )
        labeled = add_generated_labels(
            source,
            [
                {"cluster_id": -1, "generated_label": "Noise"},
                {"cluster_id": 0, "generated_label": "Zero"},
                {"cluster_id": 1, "generated_label": "One"},
            ],
        )

        figure = create_label_distribution_figure(labeled)
        percentages = list(figure.data[0].customdata[:, 2])

        self.assertEqual(list(figure.data[0].y), [1, 2, 1])
        self.assertEqual(percentages, [25.0, 50.0, 25.0])
        self.assertIn("Cluster -1 — Noise", list(figure.data[0].x))

    def test_missing_results_use_not_generated(self):
        labeled = add_generated_labels(
            pd.DataFrame({"predicted_label": [-1, 2]}),
            [],
        )

        self.assertEqual(
            labeled["generated_label"].tolist(),
            [FALLBACK_LABEL, FALLBACK_LABEL],
        )

    def test_unavailable_llm_saves_fallback_results(self):
        clusters = [{"cluster_id": -1, "keywords": [], "examples": []}]

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "results.json"
            results = label_clusters(None, clusters, checkpoint)

        self.assertEqual(results[0]["generated_label"], FALLBACK_LABEL)
        self.assertIn("unavailable", results[0]["justification"])


if __name__ == "__main__":
    unittest.main()
