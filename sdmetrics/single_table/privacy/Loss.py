import numpy as np
import pandas as pd

from copulas.univariate.base import Univariate

class LossFunction():
    def fit(self, data, cols):
        """Learn the metric on the value space.

        Args:
            real_data (pandas.DataFrame):
                The real data data table.
            cols (list[str]):
                The names for the target columns (usually the sensitive cols).
        """
        pass

    def measure(self, pred, real):
        """Calculate the loss of a single prediction.

        Args:
            pred (tuple):
                The predicted value.
            real (tuple):
                The actual value.
        """
        raise NotImplementedError("Please implement the loss measuring algorithm!")

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
            col_data = np.array(data[col])
            dist_model = Univariate()
            dist_model.fit(col_data)
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