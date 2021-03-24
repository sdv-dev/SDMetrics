import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR as svr

from sdmetrics.single_table.privacy.base import NumericalPrivacyMetric, PrivacyAttackerModel


class NumericalSklearnAttacker(PrivacyAttackerModel):
    """Base class for numerical attacker based on sklearn models.

    It is used to train a model to predict sensitive attributes from key attributes
    using the synthetic data. Then, evaluate the privacy of the model by
    trying to predict the sensitive attributes of the real data.

    Attributes:
        skl_learner (Class):
            A (wrapped) sklearn classifier class that can be called with no arguments.
    """
    SKL_LEARNER = None

    def __init__(self):
        self.predictor = self.SKL_LEARNER()

    def fit(self, synthetic_data, key, sensitive):
        """Fit the NumericalSklearnAttacker on the synthetic data.

        Args:
            synthetic_data(pandas.DataFrame):
                The synthetic data table used for adverserial learning.
            key_fields(list[str]):
                The names of the key columns.
            sensitive_fields(list[str]):
                The names of the sensitive columns.
        """
        key_table = np.array(synthetic_data[key])
        sensitive_table = np.array(synthetic_data[sensitive])

        self.predictor.fit(key_table, sensitive_table)

    def predict(self, key_data):
        """Make a prediction of the sensitive data given keys.

        Args:
            key_data(tuple):
                The key data.

        Returns:
            tuple:
                The predicted sensitive data.
        """
        sensitive_pred = self.predictor.predict([key_data])
        if len(np.array(sensitive_pred).shape) == 1:
            sensitive_pred = [sensitive_pred]

        return tuple(sensitive_pred[0])


class SVRWrapper():
    """A wrapper arround `sklearn.svm.SVR` to support multidimensional y."""

    def __init__(self):
        self.predictors = []

    def fit(self, X, Y):
        """Fit the classifier to training data X and lables Y.

        Arguments:
            X (np.array):
                training data matrix of shape (n_samples, n_features).
            Y (np.array):
                label matrix of shape (n_samples, n_labels).
        """
        n_labels = Y.shape[1]
        for idx in range(n_labels):
            Y_col = Y[:, idx]
            predictor = svr()
            predictor.fit(X, Y_col)
            self.predictors.append(predictor)

    def predict(self, X):
        """Predict the labels corresponding to data X.

        Arguments:
            X (np.array): training data matrix of shape (n_samples, n_features)

        Returns:
            np.array: label matrix of shape (n_samples, n_labels)
        """
        Y = []
        for predictor in self.predictors:
            Y.append(predictor.predict(X))

        Y = np.array(Y).T
        return Y


class LRAttacker(NumericalSklearnAttacker):
    """The privacy attaker based on the Linear Regression model."""

    SKL_LEARNER = LinearRegression


class NumericalLR(NumericalPrivacyMetric):
    """The Numerical Linear Regression privacy metric. Scored based on the LRAttacker."""

    name = 'Numerical Linear Regression'
    MODEL = LRAttacker


class MLPAttacker(NumericalSklearnAttacker):
    """The privacy attaker based on the MLP (Multi-layer Perceptron) regression model."""

    SKL_LEARNER = MLPRegressor


class NumericalMLP(NumericalPrivacyMetric):
    """The Multi-layer Perceptron regression privacy metric. Scored based on the MLPAttacker."""

    name = 'Multi-layer Perceptron Regression'
    MODEL = MLPAttacker


class SVRAttacker(NumericalSklearnAttacker):
    """The privacy attaker based on the SVR (Support-vector Regression) model."""

    SKL_LEARNER = SVRWrapper


class NumericalSVR(NumericalPrivacyMetric):
    """The Numerical Support-vector Regression privacy metric. Scored based on the SVRAttacker."""

    name = 'Numerical Support-vector Regression'
    MODEL = SVRAttacker
