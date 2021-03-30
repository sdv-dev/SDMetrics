"""Chi-Squared test based metric."""

from scipy.stats import chisquare

from sdmetrics.goal import Goal
from sdmetrics.single_column.base import SingleColumnMetric
from sdmetrics.utils import get_frequencies


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
    """

    name = 'Chi-Squared'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

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

    @classmethod
    def normalize(cls, raw_score):
        """Returns the `raw_score` as is, since it is already normalized.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                The normalized value of the metric
        """
        return super().normalize(raw_score)
