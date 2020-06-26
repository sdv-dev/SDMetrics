
from scipy.stats import chisquare

from sdmetrics.report import Goal
from sdmetrics.statistical.univariate.base import UnivariateMetric
from sdmetrics.statistical.utils import frequencies


class CSTest(UnivariateMetric):

    name = "chisquare"
    dtypes = ["object", "bool"]

    @staticmethod
    def metric(real_column, synthetic_column):
        """This function uses the Chi-squared test to compare the distributions
        of the two categorical columns. It returns the resulting p-value so that
        a small value indicates that we can reject the null hypothesis (i.e. and
        suggests that the distributions are different).

        Arguments:
            real_column (np.ndarray): The values from the real database.
            synthetic_column (np.ndarray): The values from the synthetic database.

        Returns:
            (str, Goal, str, tuple): A tuple containing (value, goal, unit, domain)
            which corresponds to the fields in a Metric object.
        """
        f_obs, f_exp = frequencies(real_column, synthetic_column)
        if len(f_obs) == len(f_exp) == 1:
            pvalue = 1.0
        else:
            _, pvalue = chisquare(f_obs, f_exp)

        return pvalue, Goal.MAXIMIZE, "p-value", (0.0, 1.0)
