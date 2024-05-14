"""Base class for Binary Efficacy metrics for single table datasets."""

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

    @classmethod
    def _score(cls, scorer, test_target, predictions):
        if test_target.dtype == 'object':
            first_label = test_target.unique()[0]
            test_target = test_target == first_label
            if predictions.dtype == 'object':
                predictions = predictions == first_label

        return super()._score(scorer, test_target, predictions)

    @classmethod
    def _fit_predict(cls, train_data, train_target, test_data, test_target):
        if test_target.dtype == 'object':
            train_target = train_target == test_target.unique()[0]

        return super()._fit_predict(train_data, train_target, test_data, test_target)

    @classmethod
    def normalize(cls, raw_score):
        """Return the `raw_score` as is, since it is already normalized.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                The normalized value of the metric
        """
        return super().normalize(raw_score)


class BinaryDecisionTreeClassifier(BinaryEfficacyMetric):
    """Binary DecisionTreeClassifier Efficacy based metric.

    This fits a DecisionTreeClassifier to the training data and
    then evaluates it making predictions on the test data.
    """

    MODEL = DecisionTreeClassifier
    MODEL_KWARGS = {'max_depth': 15, 'class_weight': 'balanced'}


class BinaryAdaBoostClassifier(BinaryEfficacyMetric):
    """Binary AdaBoostClassifier Efficacy based metric.

    This fits an AdaBoostClassifier to the training data and
    then evaluates it making predictions on the test data.
    """

    MODEL = AdaBoostClassifier


class BinaryLogisticRegression(BinaryEfficacyMetric):
    """Binary LogisticRegression Efficacy based metric.

    This fits a LogisticRegression to the training data and
    then evaluates it making predictions on the test data.
    """

    MODEL = LogisticRegression
    MODEL_KWARGS = {'solver': 'lbfgs', 'n_jobs': 2, 'class_weight': 'balanced', 'max_iter': 50}


class BinaryMLPClassifier(BinaryEfficacyMetric):
    """Binary MLPClassifier Efficacy based metric.

    This fits a MLPClassifier to the training data and
    then evaluates it making predictions on the test data.
    """

    MODEL = MLPClassifier
    MODEL_KWARGS = {'hidden_layer_sizes': (50,), 'max_iter': 50}
