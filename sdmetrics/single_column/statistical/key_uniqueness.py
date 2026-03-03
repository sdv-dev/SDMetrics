"""Key Uniqueness Metric."""

import logging

import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.single_column.base import SingleColumnMetric

LOGGER = logging.getLogger(__name__)


class KeyUniqueness(SingleColumnMetric):
    """Key uniqueness metric.

    The proportion of data points in the synthetic data that are unique.

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

    name = 'KeyUniqueness'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data):
        """Compute the score breakdown of the key uniqueness metric.

        Args:
            real_data (pandas.DataFrame or pd.Series):
                The real data key.
            synthetic_data (pandas.DataFrame or pd.Series):
                The synthetic data key.

        Returns:
            dict:
                The score breakdown of the key uniqueness metric.
        """
        if isinstance(real_data, pd.Series):
            real_data = real_data.to_frame()
        if isinstance(synthetic_data, pd.Series):
            synthetic_data = synthetic_data.to_frame()

        has_duplicates = real_data.duplicated().any()
        has_nans = real_data.isna().all(axis=1).any()
        if has_duplicates or has_nans:
            LOGGER.info('The real data contains NA or duplicate values.')

        is_nan_synthetic = synthetic_data.isna().all(axis=1)
        nans_or_duplicates_synthetic = synthetic_data.duplicated() | is_nan_synthetic
        score = 1 - nans_or_duplicates_synthetic.sum() / len(synthetic_data)

        return {'score': score}

    @classmethod
    def compute(cls, real_data, synthetic_data):
        """Compute the key uniqueness metric.

        Args:
            real_data (pd.DataFrame or pd.Series):
                The real data key.
            synthetic_data (pd.DataFrame or pd.Series):
                The synthetic data key.

        Returns:
            float:
                The proportion of data points in the synthetic data that are unique.
        """
        return cls.compute_breakdown(real_data, synthetic_data)['score']
