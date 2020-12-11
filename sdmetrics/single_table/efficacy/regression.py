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


class LinearRegression(RegressionEfficacyMetric):

    MODEL = linear_model.LinearRegression


class MLPRegressor(RegressionEfficacyMetric):

    MODEL = neural_network.MLPRegressor
    MODEL_KWARGS = {
        'hidden_layer_sizes': (100, ),
        'max_iter': 50
    }
