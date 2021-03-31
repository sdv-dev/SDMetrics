import numpy as np
import sklearn.naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVC

from sdmetrics.single_table.privacy.base import (
    CategoricalPrivacyMetric, CategoricalType, PrivacyAttackerModel)
from sdmetrics.single_table.privacy.util import allow_nan, allow_nan_array


class CategoricalSklearnAttacker(PrivacyAttackerModel):
    """Base class for the categorical attackers based on sklearn models.

    It is used to train a model to predict sensitive attributes from key attributes
    using the synthetic data. Then, evaluate the privacy of the model by
    trying to predict the sensitive attributes of the real data.

    Attributes:
        key_type (CategoricalType):
            Required key attribute type (class_num or one_hot) by the learner.
        sensitive_type (CategoricalType):
            Required sensitive attribute type (class_num or one_hot) by the learner.
        skl_learner (Class):
            A (wrapped) sklearn classifier class that can be called with no arguments.
    """

    KEY_TYPE = None
    SENSITIVE_TYPE = None
    SKL_LEARNER = None

    def __init__(self):
        self.predictor = self.SKL_LEARNER()
        self.key_processor = (
            OrdinalEncoder() if self.KEY_TYPE == CategoricalType.CLASS_NUM
            else OneHotEncoder()
        )
        self.sensitive_processor = (
            OrdinalEncoder() if
            self.SENSITIVE_TYPE == CategoricalType.CLASS_NUM else OneHotEncoder()
        )

    def fit(self, synthetic_data, key_fields, sensitive_fields):
        """Fit the CategoricalSklearnAttacker on the synthetic data.

        Args:
            synthetic_data(pandas.DataFrame):
                The synthetic data table used for adverserial learning.
            key_fields(list[str]):
                The names of the key columns.
            sensitive_fields(list[str]):
                The names of the sensitive columns.
        """
        key_table = allow_nan(synthetic_data[key_fields])
        sensitive_table = allow_nan(synthetic_data[sensitive_fields])
        self.key_processor.fit(key_table)
        self.sensitive_processor.fit(sensitive_table)

        key_train = self.key_processor.transform(key_table)
        sensitive_train = self.sensitive_processor.transform(sensitive_table)
        self.predictor.fit(key_train, sensitive_train)

    def predict(self, key_data):
        """Make a prediction of the sensitive data given keys.

        Args:
            key_data(tuple):
                The key data.

        Returns:
            tuple:
                The predicted sensitive data.
        """
        keys = allow_nan_array(key_data)  # de-nan key attributes
        try:
            # key attributes in ML ready format
            keys_transform = self.key_processor.transform([keys])
        except ValueError:  # Some attributes of the input haven't appeared in synthetic tables
            return None

        sensitive_pred = self.predictor.predict(keys_transform)
        if len(np.array(sensitive_pred).shape) == 1:
            sensitive_pred = [sensitive_pred]

        # predicted sensitive attributes in original format
        sensitives = self.sensitive_processor.inverse_transform(sensitive_pred)
        return tuple(sensitives[0])


class SVCWrapper():
    """A wrapper arround `sklearn.svm.SVC` to support multidimensional y."""

    def __init__(self):
        self.predictors = []

    def fit(self, X, Y):
        """Fit the classifier to training data X and lables Y.

        Arguments:
            X (np.array):
                training data matrix of shape (n_samples, n_features)
            Y (np.array):
                label matrix of shape (n_samples, n_labels)
        """
        n_labels = Y.shape[1]
        for idx in range(n_labels):
            Y_col = Y[:, idx]
            predictor = SVC()
            predictor.fit(X, Y_col)
            self.predictors.append(predictor)

    def predict(self, X):
        """Predict the labels corresponding to data X.

        Arguments:
            X (np.array):
                training data matrix of shape (n_samples, n_features)

        Returns:
            np.array: label matrix of shape (n_samples, n_labels)
        """
        Y = []
        for predictor in self.predictors:
            Y.append(predictor.predict(X))

        Y = np.array(Y).T
        return Y


class NBWrapper():
    """A wrapper arround `sklearn.naive_bayes.CategoricalNB` to support multidimensional y."""

    def __init__(self):
        self.predictors = []

    def fit(self, X, Y):
        """Fit the classifier to training data X and lables Y.

        Arguments:
            X (np.array):
                training data matrix of shape (n_samples, n_features)
            Y (np.array):
                label matrix of shape (n_samples, n_labels)
        """
        n_labels = Y.shape[1]
        for idx in range(n_labels):
            Y_col = Y[:, idx]
            predictor = sklearn.naive_bayes.CategoricalNB()
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


class CategoricalNBAttacker(CategoricalSklearnAttacker):
    """The Categorical NaiveBaysian privacy attaker.

    Uses a naive bayesian classifier to calculate the score based on prediction accuracy.
    """

    KEY_TYPE = CategoricalType.CLASS_NUM
    SENSITIVE_TYPE = CategoricalType.CLASS_NUM
    SKL_LEARNER = NBWrapper


class CategoricalNB(CategoricalPrivacyMetric):
    """The Categorical NaiveBaysian privacy metric. Scored based on the CategoricalNBAttacker."""

    name = 'Categorical NaiveBayesian'
    MODEL = CategoricalNBAttacker
    ACCURACY_BASE = True


class CategoricalKNNAttacker(CategoricalSklearnAttacker):
    """The Categorical KNN (k nearest neighbors) privacy attaker.

    Uses a KNN classifier to calculate the score based on prediction accuracy.
    """

    KEY_TYPE = CategoricalType.ONE_HOT
    SENSITIVE_TYPE = CategoricalType.CLASS_NUM
    SKL_LEARNER = KNeighborsClassifier


class CategoricalKNN(CategoricalPrivacyMetric):
    """The Categorical KNN privacy metric. Scored based on the KNNAttacker."""

    name = 'K-Nearest Neighbors'
    MODEL = CategoricalKNNAttacker
    ACCURACY_BASE = True


class CategoricalRFAttacker(CategoricalSklearnAttacker):
    """The Categorical RF (Random Forest) privacy attaker.

    Uses a RF classifier to calculate the score based on prediction accuracy.
    """

    KEY_TYPE = CategoricalType.ONE_HOT
    SENSITIVE_TYPE = CategoricalType.CLASS_NUM
    SKL_LEARNER = RandomForestClassifier


class CategoricalRF(CategoricalPrivacyMetric):
    """The Categorical RF privacy metric. Scored based on the CategoricalRFAttacker."""

    name = 'Categorical Random Forest'
    MODEL = CategoricalRFAttacker
    ACCURACY_BASE = True


class CategoricalSVMAttacker(CategoricalSklearnAttacker):
    """The Categorical SVM (Support Vector Machine) privacy attaker.

    Uses a SVM classifier to calculate the score based on prediction accuracy.
    """

    KEY_TYPE = CategoricalType.ONE_HOT
    SENSITIVE_TYPE = CategoricalType.CLASS_NUM
    SKL_LEARNER = SVCWrapper


class CategoricalSVM(CategoricalPrivacyMetric):
    """The Categorical SVM privacy metric. Scored based on the CategoricalSVMAttacker."""

    name = 'Support Vector Classifier'
    MODEL = CategoricalSVMAttacker
    ACCURACY_BASE = True
