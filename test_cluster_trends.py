import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from temp import (
    calculate_monthly_cluster_trends,
    create_cluster_trends_figure,
    load_arxiv_parquet,
    save_cluster_trends,
)


class MonthlyClusterTrendTests(unittest.TestCase):
    def test_load_arxiv_parquet_concatenates_result_files_numerically(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            for suffix in (10000, 2000, 1000):
                pd.DataFrame(
                    {
                        "id": [f"id-{suffix}"],
                        "title": [f"title-{suffix}"],
                        "summary": [f"summary-{suffix}"],
                        "published": ["2024-01-01"],
                    }
                ).to_parquet(
                    directory_path / f"result{suffix}.parquet",
                    index=False,
                )

            loaded = load_arxiv_parquet(
                str(directory_path / "result*.parquet")
            )

            self.assertEqual(
                loaded["id"].tolist(),
                ["id-1000", "id-2000", "id-10000"],
            )
            self.assertEqual(
                loaded.columns.tolist(),
                ["id", "title", "abstract", "published"],
            )

    def test_aggregation_fills_months_and_excludes_noise(self):
        source = pd.DataFrame(
            {
                "published": [
                    "2024-01-05",
                    "2024-01-20",
                    "2024-03-01",
                    "2024-02-10",
                    "invalid",
                ],
                "predicted_label": [0, 0, 0, -1, 1],
            }
        )

        monthly, statistics = calculate_monthly_cluster_trends(source)

        self.assertEqual(monthly["predicted_label"].unique().tolist(), [0])
        self.assertEqual(
            monthly["month"].dt.strftime("%Y-%m").tolist(),
            ["2024-01", "2024-02", "2024-03"],
        )
        self.assertEqual(monthly["item_count"].tolist(), [2, 0, 1])
        self.assertEqual(statistics["month_count"].tolist(), [3])

    def test_growth_rates_use_linear_slope_and_mean(self):
        source = pd.DataFrame(
            {
                "published": [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-02-02",
                    "2024-03-01",
                    "2024-03-02",
                    "2024-03-03",
                ],
                "predicted_label": [2, 2, 2, 2, 2, 2],
            }
        )

        monthly, statistics = calculate_monthly_cluster_trends(source)
        cluster_statistics = statistics.iloc[0]

        np.testing.assert_allclose(monthly["regression_count"], [1, 2, 3])
        self.assertAlmostEqual(
            cluster_statistics["growth_items_per_month"], 1.0
        )
        self.assertAlmostEqual(
            cluster_statistics["growth_percent_per_month"], 50.0
        )

    def test_global_range_is_shared_by_all_clusters(self):
        source = pd.DataFrame(
            {
                "published": ["2024-01-01", "2024-03-01"],
                "predicted_label": [0, 1],
            }
        )

        monthly, _ = calculate_monthly_cluster_trends(source)

        cluster_zero = monthly.loc[monthly["predicted_label"].eq(0)]
        cluster_one = monthly.loc[monthly["predicted_label"].eq(1)]
        self.assertEqual(cluster_zero["item_count"].tolist(), [1, 0, 0])
        self.assertEqual(cluster_one["item_count"].tolist(), [0, 0, 1])

    def test_single_month_has_zero_growth(self):
        source = pd.DataFrame(
            {
                "published": ["2024-06-01", "2024-06-02"],
                "predicted_label": [3, 3],
            }
        )

        monthly, statistics = calculate_monthly_cluster_trends(source)

        self.assertEqual(monthly["regression_count"].tolist(), [2.0])
        self.assertEqual(statistics["growth_items_per_month"].tolist(), [0.0])
        self.assertEqual(
            statistics["growth_percent_per_month"].tolist(), [0.0]
        )

    def test_invalid_dates_or_only_noise_produce_empty_results(self):
        source = pd.DataFrame(
            {
                "published": ["invalid", "2024-01-01"],
                "predicted_label": [0, -1],
            }
        )

        monthly, statistics = calculate_monthly_cluster_trends(source)

        self.assertTrue(monthly.empty)
        self.assertTrue(statistics.empty)

    def test_figure_has_observed_and_regression_trace_per_cluster(self):
        source = pd.DataFrame(
            {
                "published": [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-01-03",
                    "2024-02-03",
                ],
                "predicted_label": [0, 0, 1, 1],
            }
        )
        monthly, statistics = calculate_monthly_cluster_trends(source)

        figure = create_cluster_trends_figure(monthly, statistics)

        self.assertEqual(len(figure.data), 4)
        self.assertEqual(
            [trace.line.dash for trace in figure.data],
            [None, "dash", None, "dash"],
        )
        self.assertIn("items/month", figure.data[1].name)
        self.assertIn("%/month", figure.data[1].name)

    def test_save_cluster_trends_writes_requested_filename(self):
        source = pd.DataFrame(
            {
                "published": ["2024-01-01"],
                "predicted_label": [0],
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = save_cluster_trends(source, directory)

            self.assertEqual(output_path, Path(directory) / "cluster-trends.html")
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
