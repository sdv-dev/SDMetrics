"""Base class for Efficacy metrics for single table datasets."""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sdmetrics.goal import Goal
from sdmetrics.single_table.efficacy.base import MLEfficacyMetric


class BinaryClassificationEfficacyMetric(MLEfficacyMetric):
    """Base class for Binary Classification Efficacy metrics."""

    name = None
    goal = Goal.MAXIMIZE
    min_value = (0, 0)
    max_value = (1, 1)

    @staticmethod
    def _compute_scores(real_target, predictions):
        return (
            accuracy_score(real_target, predictions),
            f1_score(real_target, predictions, average='binary')
        )


class BinaryDecisionTreeClassifier(BinaryClassificationEfficacyMetric):

    model = DecisionTreeClassifier
    model_kwargs = {
        'max_depth': 15,
        'class_weight': 'balanced'
    }


class BinaryAdaBoostClassifier(BinaryClassificationEfficacyMetric):

    model = AdaBoostClassifier


class BinaryLogisticRegression(BinaryClassificationEfficacyMetric):

    model = LogisticRegression
    model_kwargs = {
        'solver': 'lbfgs',
        'n_jobs': 2,
        'class_weight': 'balanced',
        'max_iter': 50
    }


class BinaryMLPClassifier(BinaryClassificationEfficacyMetric):

    model = MLPClassifier
    model_kwargs = {
        'hidden_layer_sizes': (50, ),
        'max_iter': 50
    }
