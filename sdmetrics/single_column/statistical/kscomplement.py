"""Kolmogorov-Smirnov test based Metric."""

import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from sdmetrics.goal import Goal
from sdmetrics.single_column.base import SingleColumnMetric
from sdmetrics.utils import is_datetime

MAX_DECIMALS = sys.float_info.dig - 1


class KSComplement(SingleColumnMetric):
    """Kolmogorov-Smirnov statistic based metric.

    This function uses the two-sample Kolmogorov–Smirnov test to compare
    the distributions of the two continuous columns using the empirical CDF.
    It returns 1 minus the KS Test D statistic, which indicates the maximum
    distance between the expected CDF and the observed CDF values.

    As a result, the output value is 1.0 if the distributions are identical
    and 0.0 if they are completely different.

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

    name = 'Inverted Kolmogorov-Smirnov D statistic'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def compute(real_data, synthetic_data):
        """Compare two continuous columns using a Kolmogorov–Smirnov test.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset.

        Returns:
            float:
                1 minus the Kolmogorov–Smirnov D statistic.
        """
        real_data = pd.Series(real_data).dropna()
        synthetic_data = pd.Series(synthetic_data).dropna()

        if is_datetime(real_data):
            real_data = pd.to_numeric(real_data)
            synthetic_data = pd.to_numeric(synthetic_data)

        real_data = real_data.round(MAX_DECIMALS)
        synthetic_data = synthetic_data.round(MAX_DECIMALS)

        try:
            statistic, _ = ks_2samp(real_data, synthetic_data)
        except ValueError as e:
            if str(e) == 'Data passed to ks_2samp must not be empty':
                return np.nan
            else:
                raise ValueError(e)

        return 1 - statistic

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
