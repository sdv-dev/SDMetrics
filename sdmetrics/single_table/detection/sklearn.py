"""scikit-learn based DetectionMetrics for single table datasets."""

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from sdmetrics.single_table.detection.base import DetectionMetric


class ScikitLearnClassifierDetectionMetric(DetectionMetric):
    """Base class for Detection metrics build using Scikit Learn Classifiers.

    The base class for these metrics makes a prediction using a scikit-learn
    pipeline which contains a SimpleImputer, a RobustScaler and finally
    the classifier, which is defined in the subclasses.
    """

    name = 'Scikit-Learn Detection'

    @staticmethod
    def _get_classifier():
        """Build and return an instance of a scikit-learn Classifier."""
        raise NotImplementedError()

    @classmethod
    def _fit_predict(cls, X_train, y_train, X_test):
        """Fit a pipeline to train data and then use it to make prediction on test data."""
        model = Pipeline([
            ('imputer', SimpleImputer()),
            ('scalar', RobustScaler()),
            ('classifier', cls._get_classifier()),
        ])
        model.fit(X_train, y_train)

        return model.predict_proba(X_test)[:, 1]


class LogisticDetection(ScikitLearnClassifierDetectionMetric):
    """ScikitLearnClassifierDetectionMetric based on a LogisticRegression.

    This metric builds a LogisticRegression Classifier that learns to tell the synthetic
    data apart from the real data, which later on is evaluated using Cross Validation.

    The output of the metric is one minus the average ROC AUC score obtained.
    """

    name = "LogisticRegression Detection"

    @staticmethod
    def _get_classifier():
        return LogisticRegression(solver="lbfgs")


class SVCDetection(ScikitLearnClassifierDetectionMetric):
    """ScikitLearnClassifierDetectionMetric based on a SVC.

    This metric builds a SVC Classifier that learns to tell the synthetic
    data apart from the real data, which later on is evaluated using Cross Validation.

    The output of the metric is one minus the average ROC AUC score obtained.
    """

    name = "SVC Detection"

    @staticmethod
    def _get_classifier():
        return SVC(probability=True, gamma='scale')
