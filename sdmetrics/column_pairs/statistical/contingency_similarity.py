"""Contingency Similarity Metric."""

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
        contingency_real = real.groupby(list(columns), dropna=False).size() / len(real)
        contingency_synthetic = synthetic.groupby(list(columns), dropna=False).size() / len(
            synthetic
        )
        combined_index = contingency_real.index.union(contingency_synthetic.index)
        contingency_synthetic = contingency_synthetic.reindex(combined_index, fill_value=0)
        contingency_real = contingency_real.reindex(combined_index, fill_value=0)
        diff = abs(contingency_real - contingency_synthetic).fillna(0)
        variation = diff / 2
        return 1 - variation.sum()

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
