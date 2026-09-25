"""Boundary Adherence Metric."""

import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.single_column.base import SingleColumnMetric
from sdmetrics.utils import is_datetime


class BoundaryAdherence(SingleColumnMetric):
    """Boundary adherence metric.

    Compute the fraction of rows in the synthetic data that are within the min and max
    bounds of the real data. If any of ``range_min``, ``range_max`` or ``range_is_nullable``
    are provided, they are used instead of the values computed from the real data.

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
    def compute(
        cls, real_data, synthetic_data, range_min=None, range_max=None, range_is_nullable=None
    ):
        """Compute the boundary adherence of two continuous columns.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset.
            range_min (float or datetime, optional):
                The minimum value the column is allowed to take. If ``None``, the minimum
                value of the real data is used instead. Defaults to ``None``.
            range_max (float or datetime, optional):
                The maximum value the column is allowed to take. If ``None``, the maximum
                value of the real data is used instead. Defaults to ``None``.
            range_is_nullable (bool, optional):
                Whether the column is allowed to contain missing values.
                Defaults to ``None``.

        Returns:
            float:
                The boundary adherence of the two columns.
        """
        real_data = pd.Series(real_data)
        synthetic_data = pd.Series(synthetic_data)

        range_min = real_data.min() if range_min is None else range_min
        range_max = real_data.max() if range_max is None else range_max

        if range_is_nullable is None:
            range_is_nullable = any(pd.isna(real_data))

        real_data = real_data.dropna()
        if range_is_nullable:
            synthetic_data = synthetic_data.dropna()

        if is_datetime(real_data):
            bounds = pd.to_datetime([range_min, range_max]).astype(real_data.dtype)
            real_data = pd.to_numeric(real_data)
            synthetic_data = pd.to_numeric(synthetic_data)
            range_min, range_max = pd.to_numeric(bounds)

        valid = synthetic_data.between(range_min, range_max)

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
