"""Chi-Squared test based metric."""

import warnings
from collections import Counter

from scipy.stats import chisquare

from sdmetrics.goal import Goal
from sdmetrics.single_column.base import SingleColumnMetric


def get_frequencies(real, synthetic):
    """Get percentual frequencies for each possible real categorical value.

    Given two iterators containing categorical data, this transforms it into
    observed/expected frequencies which can be used for statistical tests. It
    adds a regularization term to handle cases where the synthetic data contains
    values that don't exist in the real data.

    Args:
        real (list):
            A list of hashable objects.
        synthetic (list):
            A list of hashable objects.

    Yields:
        tuble[list, list]:
            The observed and expected frequencies (as a percent).
    """
    f_obs, f_exp = [], []
    real, synthetic = Counter(real), Counter(synthetic)
    for value in synthetic:
        if value not in real:
            warnings.warn(f'Unexpected value {value} in synthetic data.')
            real[value] += 1e-6  # Regularization to prevent NaN.

    for value in real:
        f_obs.append(synthetic[value] / sum(synthetic.values()))
        f_exp.append(real[value] / sum(real.values()))

    return f_obs, f_exp


class CSTest(SingleColumnMetric):
    """Chi-Squared test based metric.

    This metric uses the Chi-Squared test to compare the distributions
    of the two categorical columns. It returns the resulting p-value so that
    a small value indicates that we can reject the null hypothesis (i.e. and
    suggests that the distributions are different).

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

    name = 'Chi-Squared'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0
    dtypes = ('object', 'bool')

    @staticmethod
    def compute(real_data, synthetic_data):
        """Compare two discrete columns using a Chi-Squared test.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset.

        Returns:
            float:
                The Chi-Squared test p-value
        """
        f_obs, f_exp = get_frequencies(real_data, synthetic_data)
        if len(f_obs) == len(f_exp) == 1:
            pvalue = 1.0
        else:
            _, pvalue = chisquare(f_obs, f_exp)

        return pvalue
