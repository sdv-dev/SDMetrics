import numpy as np
from copulas.univariate.base import Univariate


class LossFunction():
    """Base class for a loss function."""

    def fit(self, data, cols):
        """Learn the metric on the value space.

        Args:
            real_data (pandas.DataFrame):
                The real data table.
            cols (list[str]):
                The names for the target columns (usually the sensitive cols).
        """

    def measure(self, pred, real):
        """Calculate the loss of a single prediction.

        Args:
            pred (tuple):
                The predicted value.
            real (tuple):
                The actual value.
        """
        raise NotImplementedError('Please implement the loss measuring algorithm!')


class InverseCDFDistance(LossFunction):
    """Measure the distance between continuous key fields.

    This loss function first applies the fitted cdfs to every single entry (i.e. turning
    the numerical values into their respective percentiles) and then measures the Lp distance
    to the pth power, between the predicted value and the real value.
    """

    def __init__(self, p=2):
        """
        Args:
            p (float):
                The p parameter in L_p metric. Must be positive.
        """
        self.p = p
        self.cdfs = []

    def fit(self, data, cols):
        """Fits univariate distributions (automatically selected).

        Args:
            data (DataFrame):
                Data, where each column in `cols` is a continuous column.
            cols (list[str]):
                Column names.
        """
        for col in cols:
            col_data = np.array(data[col])
            dist_model = Univariate()
            dist_model.fit(col_data)
            self.cdfs.append(dist_model)

    def measure(self, pred, real):
        """Compute the distance (L_p norm) between the pred and real values.

        This uses the probability integral transform to map the pred/real values
        to a CDF value (between 0.0 and 1.0). Then, it computes the L_p norm
        between the CDF(pred) and CDF(real).

        Args:
            pred (tuple):
                Predicted value(s) corresponding to the columns specified in fit.
            real (tuple):
                Real value(s) corresponding to the columns specified in fit.

        Returns:
            float:
                The L_p norm of the CDF value.
        """
        assert len(pred) == len(real)

        dist = 0
        for idx in range(len(real)):
            percentiles = self.cdfs[idx].cdf(np.array([pred[idx], real[idx]]))
            dist += abs(percentiles[0] - percentiles[1])**self.p

        return dist
