"""Total Variation Complement Metric."""

import pandas as pd

from sdmetrics.errors import IncomputableMetricError
from sdmetrics.goal import Goal
from sdmetrics.single_column.base import SingleColumnMetric
from sdmetrics.utils import get_frequencies


class TVComplement(SingleColumnMetric):
    """Total Variation Complement metric.

    The complement of the total variation distance.

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

    name = 'TVComplement'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def compute(cls, real_data, synthetic_data):
        """Compute the complement of the total variation distance of two discrete columns.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset.

        Returns:
            float:
                The complement of the total variation distance.
        """
        real_data = pd.Series(real_data).dropna()
        synthetic_data = pd.Series(synthetic_data).dropna()

        if len(synthetic_data) == 0 or len(real_data) == 0:
            raise IncomputableMetricError(
                'The TVComplement metric must have 1 or more non-null values.'
            )

        f_obs, f_exp = get_frequencies(real_data, synthetic_data)
        total_variation = 0
        for i in range(len(f_obs)):
            total_variation += abs(f_obs[i] - f_exp[i])

        return 1 - 0.5 * total_variation

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
