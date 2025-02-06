"""Base class for metrics that compare pairs of columns."""

from sdmetrics.base import BaseMetric
from time import process_time
import numpy as np
DEFAULT_NUM_ROWS = None
DEFAULT_NUM_TRY = None

class ColumnPairsMetric(BaseMetric):
    """Base class for metrics that compare pairs of columns.

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

    name = None
    goal = None
    min_value = None
    max_value = None

    @staticmethod
    def compute(real_data, synthetic_data):
        """Compute this metric.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset, passed as pandas.DataFrame
                with 2 columns.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset, passed as a
                pandas.DataFrame with 2 columns.

        Returns:
            float:
                Metric output.
        """
        raise NotImplementedError()

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data):
        """Compute the breakdown of this metric."""
        start = process_time()
        num_try = 1 if DEFAULT_NUM_TRY is None else DEFAULT_NUM_TRY
        result = np.zeros(num_try)
        for i in range(num_try):
            if DEFAULT_NUM_ROWS is not None:
                real_to_subsample = min(DEFAULT_NUM_ROWS, len(real_data))
                real_data_to_compute = real_data.sample(n=real_to_subsample)
                synthetic_data_to_compute = synthetic_data.sample(n=real_to_subsample)

            result[i] = cls.compute(real_data_to_compute, synthetic_data_to_compute)

        score = np.mean(result)
        end = process_time()

        return {'score': score, 'time': end - start, 'num_rows': DEFAULT_NUM_ROWS}