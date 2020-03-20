
import numpy as np
from scipy.special import rel_entr

from sdmetrics import Goal

from .base import BivariateMetric


class ContinuousDivergence(BivariateMetric):

    name = "continuous-kl"
    dtypes = [
        ("float", "float"),
        ("float", "int"),
        ("int", "int")
    ]

    @staticmethod
    def metric(real, fake):
        """
        This approximates the KL divergence by binning the continuous values
        to turn them into categorical values and then computing the relative
        entropy.

        TODO:
            * Investigate a KDE-based approach.

        Arguments:
            real_column (np.ndarray): The values from the real database.
            fake_column (np.ndarray): The values from the fake database.

        Returns:
            (str, Goal, str, tuple): A tuple containing (value, goal, unit, domain)
            which corresponds to the fields in a Metric object.
        """
        real, xedges, yedges = np.histogram2d(real[:, 0], real[:, 1])
        fake, _, _ = np.histogram2d(
            fake[:, 0], fake[:, 1], bins=[xedges, yedges])

        f_obs, f_exp = fake.flatten() + 1e-5, real.flatten() + 1e-5
        f_obs, f_exp = f_obs / np.sum(f_obs), f_exp / np.sum(f_exp)

        value = np.sum(rel_entr(f_obs, f_exp))
        return value, Goal.MINIMIZE, "entropy", (0.0, float("inf"))
