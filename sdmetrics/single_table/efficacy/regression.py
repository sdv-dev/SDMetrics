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

    @staticmethod
    def _compute_scores(real_target, predictions):
        return r2_score(real_target, predictions)


class LinearRegression(RegressionEfficacyMetric):

    model = linear_model.LinearRegression


class MLPRegressor(RegressionEfficacyMetric):

    model = neural_network.MLPRegressor
    model_kwargs = {
        'hidden_layer_sizes': (100, ),
        'max_iter': 50
    }
