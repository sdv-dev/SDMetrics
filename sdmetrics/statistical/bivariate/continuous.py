
import numpy as np
from scipy.special import rel_entr

from sdmetrics.report import Goal
from sdmetrics.statistical.bivariate.base import BivariateMetric


class ContinuousDivergence(BivariateMetric):

    name = "continuous-kl"
    dtypes = [
        ("float", "float"),
        ("float", "int"),
        ("int", "int")
    ]

    @staticmethod
    def metric(real, synthetic):
        """
        This approximates the KL divergence by binning the continuous values
        to turn them into categorical values and then computing the relative
        entropy.

        TODO:
            * Investigate a KDE-based approach.

        Arguments:
            real (np.ndarray): The values from the real database.
            synthetic (np.ndarray): The values from the synthetic database.

        Returns:
            (str, Goal, str, tuple): A tuple containing (value, goal, unit, domain)
            which corresponds to the fields in a Metric object.
        """
        real[np.isnan(real)] = 0.0
        synthetic[np.isnan(synthetic)] = 0.0

        real, xedges, yedges = np.histogram2d(real[:, 0], real[:, 1])
        synthetic, _, _ = np.histogram2d(
            synthetic[:, 0], synthetic[:, 1], bins=[xedges, yedges])

        f_obs, f_exp = synthetic.flatten() + 1e-5, real.flatten() + 1e-5
        f_obs, f_exp = f_obs / np.sum(f_obs), f_exp / np.sum(f_exp)

        value = np.sum(rel_entr(f_obs, f_exp))
        return value, Goal.MINIMIZE, "entropy", (0.0, float("inf"))
