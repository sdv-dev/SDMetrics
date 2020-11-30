"""Base class for Machine Learning Detection metrics for single table datasets."""

import numpy as np
from rdt import HyperTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from sdmetrics.single_table.base import SingleTableMetric


class DetectionMetric(SingleTableMetric):
    """Base class for Machine Learning Detection based metrics on single tables.

    These metrics build a Machine Learning Classifier that learns to tell the synthetic
    data apart from the real data, which later on is evaluated using Cross Validation.

    The output of the metric is one minus the average ROC AUC score obtained.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
    """

    name = None
    goal = None
    min_value = None
    max_value = None

    @staticmethod
    def fit_predict(X_train, y_train, X_test):
        """Fit a classifier and then use it to predict."""
        raise NotImplementedError()

    @classmethod
    def compute(cls, real_data, synthetic_data):
        """Compute this metric.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.

        Returns:
            float:
                One minus the ROC AUC Score obtained by the classifier.
        """
        transformer = HyperTransformer()
        real_data = transformer.fit_transform(real_data).values
        synthetic_data = transformer.transform(synthetic_data).values

        X = np.concatenate([real_data, synthetic_data])
        y = np.hstack([np.ones(len(real_data)), np.zeros(len(synthetic_data))])
        X[np.isin(X, [np.inf, -np.inf])] = None
        X = SimpleImputer().fit_transform(X)

        scores = []
        kf = StratifiedKFold(n_splits=3, shuffle=True)
        for train_index, test_index in kf.split(X, y):
            y_pred = cls.fit_predict(X[train_index], y[train_index], X[test_index])
            auroc = roc_auc_score(y[test_index], y_pred)
            if auroc < 0.5:
                auroc = 1.0 - auroc

            scores.append(auroc)

        return 1 - np.mean(scores)
