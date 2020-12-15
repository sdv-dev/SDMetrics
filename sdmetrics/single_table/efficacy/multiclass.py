"""Base class for Efficacy metrics for single table datasets."""

from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sdmetrics.goal import Goal
from sdmetrics.single_table.efficacy.base import MLEfficacyMetric


def f1_macro(real_target, predictions):
    return f1_score(real_target, predictions, average='macro'),


class MulticlassEfficacyMetric(MLEfficacyMetric):
    """Base class for Multiclass Classification Efficacy Metrics."""

    name = None
    goal = Goal.MAXIMIZE
    min_value = (0, 0, 0)
    max_value = (1, 1, 1)
    SCORER = f1_macro


class MulticlassDecisionTreeClassifier(MulticlassEfficacyMetric):

    MODEL = DecisionTreeClassifier
    MODEL_KWARGS = {
        'max_depth': 30,
        'class_weight': 'balanced',
    }


class MulticlassMLPClassifier(MulticlassEfficacyMetric):

    MODEL = MLPClassifier
    MODEL_KWARGS = {
        'hidden_layer_sizes': (100, ),
        'max_iter': 50
    }
