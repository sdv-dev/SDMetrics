"""Missing Value Similarity Metric."""

import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.single_column.base import SingleColumnMetric


class MissingValueSimilarity(SingleColumnMetric):
    """Missing value similarity metric.

    Compute the percentage of missing values between the real and synthetic data.

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

    name = 'MissingValueSimilarity'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data):
        """Compare the missing value similarity of two continuous columns.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset.

        Returns:
            dict:
                A mapping of the missing value similarity results.
        """
        real_data = pd.Series(real_data)
        synthetic_data = pd.Series(synthetic_data)

        real_data_value = real_data.isna().sum() / len(real_data)
        synthetic_data_value = synthetic_data.isna().sum() / len(synthetic_data)

        return {
            'score': 1 - abs(real_data_value - synthetic_data_value),
            'real': real_data_value,
            'synthetic': synthetic_data_value,
        }

    @classmethod
    def compute(cls, real_data, synthetic_data):
        """Compare the missing value similarity of two continuous columns.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset.

        Returns:
            float:
                The missing value similarity of the two columns.
        """
        results = cls.compute_breakdown(real_data, synthetic_data)
        return results['score']

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
