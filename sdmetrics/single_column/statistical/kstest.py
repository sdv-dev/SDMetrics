"""Kolmogorov-Smirnov test based Metric."""

import pandas as pd
from scipy.stats import ks_2samp

from sdmetrics.goal import Goal
from sdmetrics.single_column.base import SingleColumnMetric


class KSTest(SingleColumnMetric):
    """Kolmogorov-Smirnov test based metric.

    This function uses the two-sample Kolmogorov–Smirnov test to compare
    the distributions of the two continuous columns using the empirical CDF.
    It returns the resulting p-value so that a small value indicates that we
    can reject the null hypothesis (i.e. and suggests that the distributions
    are different).

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
        dtypes (tuple[str]):
            The data types which this metric works on (i.e. ``('float', 'str')``).
    """

    name = 'Kolmogorov-Smirnov'
    dtypes = ('float', 'int')
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def compute(real_data, synthetic_data):
        """Compare two discrete columns using a Kolmogorov–Smirnov test.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset.

        Returns:
            float:
                The Kolmogorov–Smirnov test p-value
        """
        real_data = pd.Series(real_data).fillna(0)
        synthetic_data = pd.Series(synthetic_data).fillna(0)
        _, pvalue = ks_2samp(real_data, synthetic_data)

        return pvalue
