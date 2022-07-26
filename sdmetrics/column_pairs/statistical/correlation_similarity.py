"""Correlation Similarity Metric."""

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from sdmetrics.column_pairs.base import ColumnPairsMetric
from sdmetrics.goal import Goal
from sdmetrics.utils import is_datetime


class CorrelationSimilarity(ColumnPairsMetric):
    """Correlation similarity metric.

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

    name = 'CorrelationSimilarity'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data, coefficient='Pearson'):
        """Compare the breakdown of correlation similarity of two continuous columns.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset.

        Returns:
            dict:
                A dict containing the score, and the real and synthetic metric values.
        """
        real_data[pd.isna(real_data)] = 0.0
        synthetic_data[pd.isna(synthetic_data)] = 0.0
        column1, column2 = real_data.columns[:2]

        if is_datetime(real_data):
            real_data = pd.to_numeric(real_data)
            synthetic_data = pd.to_numeric(synthetic_data)

        correlation_fn = None
        if coefficient == 'Pearson':
            correlation_fn = pearsonr
        elif coefficient == 'Spearman':
            correlation_fn = spearmanr
        else:
            raise ValueError(f'requested coefficient {coefficient} is not valid. '
                             'Please choose either Pearson or Spearman.')

        correlation_real = correlation_fn(real_data[column1], real_data[column2])
        correlation_synthetic = correlation_fn(synthetic_data[column1], synthetic_data[column2])
        return {
            'score': 1 - abs(correlation_real - correlation_synthetic) / 2,
            'real': correlation_real,
            'synthetic': correlation_synthetic,
        }

    @classmethod
    def compute(cls, real_data, synthetic_data, coefficient='Pearson'):
        """Compare the correlation similarity of two continuous columns.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset.

        Returns:
            float:
                The correlation similarity of the two columns.
        """
        return cls.compute_breakdown(real_data, synthetic_data, coefficient)['score']

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