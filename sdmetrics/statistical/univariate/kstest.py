import numpy as np
from scipy.stats import ks_2samp

from sdmetrics.report import Goal
from sdmetrics.statistical.univariate.base import UnivariateMetric


class KSTest(UnivariateMetric):

    name = "kstest"
    dtypes = ["float", "int"]

    @staticmethod
    def metric(real_column, synthetic_column):
        """This function uses the two-sample Kolmogorov–Smirnov test to compare
        the distributions of the two continuous columns using the empirical CDF.
        It returns the resulting p-value so that a small value indicates that we
        can reject the null hypothesis (i.e. and suggests that the distributions
        are different).

        Arguments:
            real_column (np.ndarray): The values from the real database.
            synthetic_column (np.ndarray): The values from the synthetic database.

        Returns:
            (str, Goal, str, tuple): A tuple containing (value, goal, unit, domain)
            which corresponds to the fields in a Metric object.
        """
        real_column = real_column.copy()
        synthetic_column = synthetic_column.copy()
        real_column[np.isnan(real_column)] = 0.0
        synthetic_column[np.isnan(synthetic_column)] = 0.0
        statistic, pvalue = ks_2samp(real_column, synthetic_column)
        return pvalue, Goal.MAXIMIZE, "p-value", (0.0, 1.0)
