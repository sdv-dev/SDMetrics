"""Regression Efficacy based metrics."""

import numpy as np
from sklearn import linear_model, neural_network
from sklearn.metrics import r2_score

from sdmetrics.goal import Goal
from sdmetrics.single_table.efficacy.base import MLEfficacyMetric


class RegressionEfficacyMetric(MLEfficacyMetric):
    """RegressionEfficacy base class."""

    name = None
    goal = Goal.MAXIMIZE
    min_value = -np.inf
    max_value = 1
    SCORER = r2_score

    @classmethod
    def normalize(cls, raw_score):
        """Returns a normalized version of the R^2 score.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                The normalized value of the metric
        """
        return super().normalize(raw_score)


class LinearRegression(RegressionEfficacyMetric):
    """LinearRegression Efficacy based metric.

    This fits a LinearRegression to the synthetic data and
    then evaluates it making predictions on the real data.
    """

    MODEL = linear_model.LinearRegression


class MLPRegressor(RegressionEfficacyMetric):
    """MLPRegressor Efficacy based metric.

    This fits a MLPRegressor to the synthetic data and
    then evaluates it making predictions on the real data.
    """

    MODEL = neural_network.MLPRegressor
    MODEL_KWARGS = {
        'hidden_layer_sizes': (100, ),
        'max_iter': 50
    }
