"""scikit-learn based TabularDetectors."""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from sdmetrics.detection.tabular.base import TabularDetector


class ScikitLearnDetector(TabularDetector):

    def _get_classifier(self):
        """Build and return an instance of a scikit-learn Classifier."""
        raise NotImplementedError()

    def fit(self, X, y):
        """This function trains a sklearn pipeline with a robust scalar
        and a logistic regression classifier.

        Arguments:
            X (np.ndarray): The numerical features (i.e. transformed rows).
            y (np.ndarray): The binary classification target.
        """
        X[np.isin(X, [np.inf, -np.inf])] = None
        self.model = Pipeline([
            ('imputer', SimpleImputer()),
            ('scalar', RobustScaler()),
            ('classifier', self._get_classifier()),
        ])
        self.model.fit(X, y)

    def predict_proba(self, X):
        X[np.isin(X, [np.inf, -np.inf])] = None
        return self.model.predict_proba(X)[:, 1]


class LogisticDetector(ScikitLearnDetector):

    name = "logistic"

    def _get_classifier(self):
        return LogisticRegression(solver="lbfgs")


class SVCDetector(ScikitLearnDetector):

    name = "svc"

    def _get_classifier(self):
        return SVC(probability=True, gamma='scale')
