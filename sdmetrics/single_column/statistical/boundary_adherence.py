"""Boundary Adherence Metric."""

import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.single_column.base import SingleColumnMetric
from sdmetrics.utils import is_datetime


class BoundaryAdherence(SingleColumnMetric):
    """Boundary adherence metric.

    Compute the fraction of rows in the synthetic data that are within the min and max
    bounds of the real data

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

    name = 'BoundaryAdherence'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def compute(cls, real_data, synthetic_data):
        """Compute the boundary adherence of two continuous columns.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset.

        Returns:
            float:
                The boundary adherence of the two columns.
        """
        real_data = pd.Series(real_data)
        synthetic_data = pd.Series(synthetic_data)
        if any(pd.isna(real_data)):
            real_data = real_data.dropna()
            synthetic_data = synthetic_data.dropna()

        if is_datetime(real_data):
            real_data = pd.to_numeric(real_data)
            synthetic_data = pd.to_numeric(synthetic_data)

        valid = synthetic_data.between(real_data.min(), real_data.max())

        return valid.sum() / len(synthetic_data)

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
