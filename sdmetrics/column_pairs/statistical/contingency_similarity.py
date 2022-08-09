"""Contingency Similarity Metric."""

import pandas as pd

from sdmetrics.column_pairs.base import ColumnPairsMetric
from sdmetrics.goal import Goal


class ContingencySimilarity(ColumnPairsMetric):
    """Contingency similarity metric.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
    """

    name = 'ContingencySimilarity'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def compute(cls, real_data, synthetic_data):
        """Compare the contingency similarity of two discrete columns.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset.

        Returns:
            float:
                The contingency similarity of the two columns.
        """
        columns = real_data.columns[:2]
        real = real_data[columns]
        synthetic = synthetic_data[columns]
        contingency_real = pd.crosstab(
            index=real[columns[0]].astype(str),
            columns=real[columns[1]].astype(str),
            normalize=True,
        )
        contingency_synthetic = pd.crosstab(
            index=synthetic[columns[0]].astype(str),
            columns=synthetic[columns[1]].astype(str),
            normalize=True,
        )

        for col in set(contingency_synthetic.columns) - set(contingency_real.columns):
            contingency_real[col] = [0 for _ in range(len(contingency_real))]
        for col in set(contingency_real.columns) - set(contingency_synthetic.columns):
            contingency_synthetic[col] = [0 for _ in range(len(contingency_synthetic))]
        for row in set(contingency_synthetic.index) - set(contingency_real.index):
            contingency_real.loc[row] = [0 for _ in range(len(contingency_real.columns))]
        for row in set(contingency_real.index) - set(contingency_synthetic.index):
            contingency_synthetic.loc[row] = [0 for _ in range(len(contingency_synthetic.columns))]

        variation = abs(contingency_real - contingency_synthetic) / 2
        return 1 - variation.sum().sum()

    @classmethod
    def normalize(cls, raw_score):
        """Return the `raw_score` as is, since it is already normalized.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                The normalized value of the metric
        """
        return super().normalize(raw_score)
