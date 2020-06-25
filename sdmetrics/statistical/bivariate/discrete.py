
import numpy as np
from scipy.special import rel_entr

from sdmetrics.report import Goal
from sdmetrics.statistical.bivariate.base import BivariateMetric
from sdmetrics.statistical.utils import frequencies


class DiscreteDivergence(BivariateMetric):

    name = "discrete-kl"
    dtypes = [
        ("object", "object"),
        ("object", "bool"),
        ("bool", "bool")
    ]

    @staticmethod
    def metric(real, synthetic):
        assert real.shape[1] == 2, "Expected 2d data."
        assert synthetic.shape[1] == 2, "Expected 2d data."
        real = [(x[0], x[1]) for x in real]
        synthetic = [(x[0], x[1]) for x in synthetic]
        f_obs, f_exp = frequencies(real, synthetic)
        value = np.sum(rel_entr(f_obs, f_exp))
        return value, Goal.MINIMIZE, "entropy", (0.0, float("inf"))
