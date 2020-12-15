"""Base class for Efficacy metrics for single table datasets."""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sdmetrics.goal import Goal
from sdmetrics.single_table.efficacy.base import MLEfficacyMetric


class BinaryEfficacyMetric(MLEfficacyMetric):
    """Base class for Binary Classification Efficacy metrics."""

    name = None
    goal = Goal.MAXIMIZE
    min_value = 0
    max_value = 1
    SCORER = f1_score


class BinaryDecisionTreeClassifier(BinaryEfficacyMetric):

    MODEL = DecisionTreeClassifier
    MODEL_KWARGS = {
        'max_depth': 15,
        'class_weight': 'balanced'
    }


class BinaryAdaBoostClassifier(BinaryEfficacyMetric):

    MODEL = AdaBoostClassifier


class BinaryLogisticRegression(BinaryEfficacyMetric):

    MODEL = LogisticRegression
    MODEL_KWARGS = {
        'solver': 'lbfgs',
        'n_jobs': 2,
        'class_weight': 'balanced',
        'max_iter': 50
    }


class BinaryMLPClassifier(BinaryEfficacyMetric):

    MODEL = MLPClassifier
    MODEL_KWARGS = {
        'hidden_layer_sizes': (50, ),
        'max_iter': 50
    }
