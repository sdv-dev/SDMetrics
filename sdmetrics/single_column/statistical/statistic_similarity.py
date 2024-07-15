"""Statistic Similarity Metric."""

import warnings

import numpy as np
import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.single_column.base import SingleColumnMetric
from sdmetrics.utils import is_datetime
from sdmetrics.warnings import ConstantInputWarning


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
    """

    name = 'StatisticSimilarity'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def compute(cls, real_data, synthetic_data, statistic='mean'):
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
        return cls.compute_breakdown(real_data, synthetic_data, statistic)['score']

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data, statistic='mean'):
        """Compare the breakdown of statistic similarity of two continuous columns.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset.

        Returns:
            dict:
                A dict containing the score, and the real and synthetic metric values.
        """
        real_data = pd.Series(real_data).dropna()
        synthetic_data = pd.Series(synthetic_data).dropna()

        if real_data.nunique() == 1:
            msg = (
                'The real data input array is constant. '
                'The StatisticSimilarity metric is either undefined or infinite.'
            )
            warnings.warn(ConstantInputWarning(msg))
            return {'score': np.nan}

        if is_datetime(real_data):
            real_data = pd.to_numeric(real_data)
            synthetic_data = pd.to_numeric(synthetic_data)

        if statistic == 'mean':
            score_real = real_data.mean()
            score_synthetic = synthetic_data.mean()
        elif statistic == 'std':
            score_real = real_data.std()
            score_synthetic = synthetic_data.std()
        elif statistic == 'median':
            score_real = real_data.median()
            score_synthetic = synthetic_data.median()
        else:
            raise ValueError(
                f'requested statistic {statistic} is not valid. '
                'Please choose either mean, std, or median.'
            )

        score = 1 - abs(score_real - score_synthetic) / (real_data.max() - real_data.min())
        return {'real': score_real, 'synthetic': score_synthetic, 'score': max(score, 0)}

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
