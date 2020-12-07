"""Base class for Efficacy metrics for single table datasets."""

from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sdmetrics.goal import Goal
from sdmetrics.single_table.efficacy.base import MLEfficacyMetric


class MulticlassClassificationEfficacyMetric(MLEfficacyMetric):
    """Base class for Multiclass Classification Efficacy Metrics."""

    name = None
    goal = Goal.MAXIMIZE
    min_value = (0, 0, 0)
    max_value = (1, 1, 1)

    @staticmethod
    def _compute_scores(real_target, predictions):
        return (
            accuracy_score(real_target, predictions),
            f1_score(real_target, predictions, average='macro'),
            f1_score(real_target, predictions, average='micro'),
        )


class MulticlassDecisionTreeClassifier(MulticlassClassificationEfficacyMetric):

    model = DecisionTreeClassifier
    model_kwargs = {
        'max_depth': 30,
        'class_weight': 'balanced',
    }


class MulticlassMLPClassifier(MulticlassClassificationEfficacyMetric):

    model = MLPClassifier
    model_kwargs = {
        'hidden_layer_sizes': (100, ),
        'max_iter': 50
    }
