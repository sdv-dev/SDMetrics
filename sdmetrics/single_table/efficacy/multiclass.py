"""Base class for Efficacy metrics for single table datasets."""

from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sdmetrics.goal import Goal
from sdmetrics.single_table.efficacy.base import MLEfficacyMetric


def f1_macro(real_target, predictions):
    return f1_score(real_target, predictions, average='macro')


class MulticlassEfficacyMetric(MLEfficacyMetric):
    """Base class for Multiclass Classification Efficacy Metrics."""

    name = None
    goal = Goal.MAXIMIZE
    min_value = 0
    max_value = 1
    SCORER = f1_macro

    @classmethod
    def normalize(cls, raw_score):
        """Returns the `raw_score` as is, since it is already normalized.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                The normalized value of the metric
        """
        return super().normalize(raw_score)


class MulticlassDecisionTreeClassifier(MulticlassEfficacyMetric):
    """Multiclass DecisionTreeClassifier Efficacy based metric.

    This fits a DecisionTreeClassifier to the synthetic data and
    then evaluates it making predictions on the real data.
    """

    MODEL = DecisionTreeClassifier
    MODEL_KWARGS = {
        'max_depth': 30,
        'class_weight': 'balanced',
    }


class MulticlassMLPClassifier(MulticlassEfficacyMetric):
    """Multiclass MLPClassifier Efficacy based metric.

    This fits a MLPClassifier to the synthetic data and
    then evaluates it making predictions on the real data.
    """

    MODEL = MLPClassifier
    MODEL_KWARGS = {
        'hidden_layer_sizes': (100, ),
        'max_iter': 50
    }
