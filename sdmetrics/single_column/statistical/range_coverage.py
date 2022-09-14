"""Range Coverage Metric."""

import numpy as np

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
        min_r = min(real_data)
        max_r = max(real_data)
        min_s = min(synthetic_data)
        max_s = max(synthetic_data)

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
