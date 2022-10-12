"""Category Coverage Metric."""

import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.single_column.base import SingleColumnMetric


class CategoryCoverage(SingleColumnMetric):
    """Category coverage metric.

    Compute the fraction of real data categories that are present in the synthetic data.

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

    name = 'CategoryCoverage'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def compute(cls, real_data, synthetic_data):
        """Compare the category coverage of two continuous columns.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset.

        Returns:
            float:
                The category coverage ratio of the two columns.
        """
        results = cls.compute_breakdown(real_data, synthetic_data)
        return results['score']

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data):
        """Compare the category coverage of two continuous columns.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset.

        Returns:
            dict:
                A mapping of the category coverage results.
        """
        real_data = pd.Series(real_data).dropna()
        synthetic_data = pd.Series(synthetic_data).dropna()

        real_data_values = set(real_data.value_counts().index)
        synthetic_data_values = set(synthetic_data.value_counts().index)
        synthetic_coverage = synthetic_data_values.intersection(real_data_values)

        return {
            'score': len(synthetic_coverage) / len(real_data_values),
            'real': len(real_data_values),
            'synthetic': len(synthetic_coverage),
        }

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
