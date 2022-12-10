"""Range Coverage Metric."""

import numpy as np
import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.single_column.base import SingleColumnMetric


class RangeCoverage(SingleColumnMetric):
    """Range coverage metric.

    Compute whether a synthetic column covers the full range of values that are
    present in a real column

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

    name = 'RangeCoverage'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def compute(cls, real_data, synthetic_data):
        """Compute the range coverage of synthetic columns over the real column.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset.

        Returns:
            float:
                The range coverage of the synthetic data over the real data.
        """
        if not isinstance(real_data, pd.Series):
            real_data = pd.Series(real_data)

        if not isinstance(synthetic_data, pd.Series):
            synthetic_data = pd.Series(synthetic_data)

        min_r = real_data.min()
        max_r = real_data.max()
        min_s = synthetic_data.min()
        max_s = synthetic_data.max()

        if min_r == max_r:
            return np.nan

        normalized_min = max((min_s - min_r) / (max_r - min_r), 0)
        normalized_max = max((max_r - max_s) / (max_r - min_r), 0)
        return max(1 - (normalized_min + normalized_max), 0)

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
