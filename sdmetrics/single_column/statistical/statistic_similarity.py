"""Statistic Similarity Metric."""

import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.single_column.base import SingleColumnMetric
from sdmetrics.utils import is_datetime


class StatisticSimilarity(SingleColumnMetric):
    """Statistic similarity metric.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
        statistic (str):
            The statistic to compute the metric on (mean, std, or median). Defaults to mean.
    """

    name = 'StatisticSimilarity'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0
    statistic = 'mean'

    def __init__(self, statistic='mean'):
        self.statistic = statistic

    def compute(self, real_data, synthetic_data):
        """Compare the statistic similarity of two continuous columns.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset.

        Returns:
            float:
                The statistical similarity of the two columns.
        """
        real_data = pd.Series(real_data).dropna()
        synthetic_data = pd.Series(synthetic_data).dropna()

        if is_datetime(real_data):
            real_data = pd.to_numeric(real_data)
            synthetic_data = pd.to_numeric(synthetic_data)

        if self.statistic == 'mean':
            score_real = real_data.mean()
            score_synthetic = synthetic_data.mean()
        elif self.statistic == 'std':
            score_real = real_data.std()
            score_synthetic = synthetic_data.std()
        elif self.statistic == 'median':
            score_real = real_data.median()
            score_synthetic = synthetic_data.median()
        else:
            raise ValueError(f'requested statistic {self.statistic} is not valid. '
                             'Please choose either mean, std, or median.')

        score = 1 - abs(score_real - score_synthetic) / (real_data.max() - real_data.min())
        return max(score, 0)

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
