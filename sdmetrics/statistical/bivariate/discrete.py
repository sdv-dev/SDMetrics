
import numpy as np
from scipy.special import rel_entr

from sdmetrics import Goal
from sdmetrics.statistical.utils import frequencies

from .base import BivariateMetric


class DiscreteDivergence(BivariateMetric):

    name = "discrete-kl"
    dtypes = [
        ("object", "object"),
        ("object", "bool"),
        ("bool", "bool")
    ]

    @staticmethod
    def metric(real, fake):
        assert real.shape[1] == 2, "Expected 2d data."
        assert fake.shape[1] == 2, "Expected 2d data."
        real = [(x[0], x[1]) for x in real]
        fake = [(x[0], x[1]) for x in fake]
        f_obs, f_exp = frequencies(real, fake)
        value = np.sum(rel_entr(f_obs, f_exp))
        return value, Goal.MINIMIZE, "entropy", (0.0, float("inf"))
