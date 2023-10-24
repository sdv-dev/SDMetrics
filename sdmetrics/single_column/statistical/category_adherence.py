"""Category Adherence Metric."""

import numpy as np

from sdmetrics.goal import Goal
from sdmetrics.single_column.base import SingleColumnMetric


class CategoryAdherence(SingleColumnMetric):
    """Category adherence metric.

    The proportion of synthetic data points that match an existing category from the real data.

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

    name = 'CategoryAdherence'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data):
        """Compute the score breakdown of the category adherence metric.

        Args:
            real_data (pandas.Series):
                The real data.
            synthetic_data (pandas.Series):
                The synthetic data.

        Returns:
            dict:
                The score breakdown of the category adherence metric.
        """
        real_data = real_data.fillna(np.nan)
        synthetic_data = synthetic_data.fillna(np.nan)
        score = synthetic_data.isin(real_data).mean()

        return {'score': score}

    @classmethod
    def compute(cls, real_data, synthetic_data):
        """Compute the category adherence of two columns.

        Args:
            real_data (pandas.Series):
                The real data.
            synthetic_data (pandas.Series):
                The synthetic data.

        Returns:
            float:
                The category adherence metric score.
        """
        return cls.compute_breakdown(real_data, synthetic_data)['score']
