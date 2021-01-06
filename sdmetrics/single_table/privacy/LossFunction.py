import numpy as np
import pandas as pd

from copulas.univariate.base import Univariate
from sdmetrics.single_table.privacy.base import LossFunction

class CdfInvLp(LossFunction):
    """This loss function first applies the fitted cdfs to every single entry (i.e. turning
    the numerical values into their respective percentiles) and then measures the Lp distance
    to the pth power, between the predicted value and the real value.
    """
    def __init__(self, p = 2):
        """
        Args:
            p (float):
                The p parameter in L_p metric. Must be positive.
        """
        self.p = p
        self.cdfs = []

    def fit(self, data, cols):
        for col in cols:
            data = np.array(data[col])
            dist_model = Univariate()
            dist_model.fit(data)
            self.cdfs.append(dist_model)

    def measure(self, pred, real):
        dist = 0
        for idx in range(len(pred)):
            percentiles = self.cdfs[idx].cdf(np.array([pred[idx], real[idx]]))
            dist += abs(percentiles[0] - percentiles[1])**self.p
        return dist

class Lp(LossFunction):
    """pth power of the Lp distance.
    """
    def __init__(self, p = 2):
        """
        Args:
            p (float):
                The p parameter in L_p metric. Must be positive.
        """
        self.p = p

    def measure(self, pred, real):
        dist = 0
        for idx in range(len(pred)):
            dist += abs(pred[idx] - real[1])**self.p
        return dist