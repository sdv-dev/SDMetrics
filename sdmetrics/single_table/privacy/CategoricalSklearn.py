import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVC

from sdmetrics.single_table.privacy.base import CategoricalType, CatPrivacyMetric,\
    PrivacyAttackerModel
from sdmetrics.single_table.privacy.util import allow_nan, allow_nan_array

class CatSklearnAttacker(PrivacyAttackerModel):
    """Base class for categorical attacker based on sklearn models.

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
        self.predictors = self.SKL_LEARNER()
        self.key_processor = OrdinalEncoder() if self.KEY_TYPE == CategoricalType.CLASS_NUM \
            else OneHotEncoder()
        self.sensitive_processor = OrdinalEncoder() if \
            self.SENSITIVE_TYPE == CategoricalType.CLASS_NUM else OneHotEncoder()

    def fit(self, synthetic_data, key, sensitive):
        key_table = allow_nan(synthetic_data[key])
        sensitive_table = allow_nan(synthetic_data[sensitive])
        self.key_processor.fit(key_table)
        self.sensitive_processor.fit(sensitive_table)

        key_train = self.key_processor.transform(key_table)
        sensitive_train = self.sensitive_processor.transform(sensitive_table)
        self.predictor.fit(key_train, sensitive_train)

    def predict(self, key_data):
        keys = allow_nan_array(key_data) #de-nan key attributes
        try:
            #key attributes in ML ready format
            keys_transform = self.key_processor.transform([keys])
        except: #Some attributes of the input haven't appeared in synthetic tables
            return None
        sensitive_pred = self.predictor.predict(keys_transform) 
        if len(np.array(sensitive_pred).shape) == 1:
            sensitive_pred = [sensitive_pred]

        #predicted sensitive attributes in original format
        sensitives = self.sensitive_processor.inverse_transform(sensitive_pred) 
        return tuple(sensitives[0])

class SvcWrapper():
    """This class provides an wrapper arround sklearn.svm.SVC so that it can support 
    multidimensional y.
    """

    def __init__(self):
        self.predictors = []
    
    def fit(self, X, Y):
        """
        Fit the classifier to training data X and lables Y.

        Arguments:
            X (np.array): training data matrix of shape (n_samples, n_features)
            Y (np.array): label matrix of shape (n_samples, n_labels)
        """
        n_labels = Y.shape[1]
        for idx in range(n_labels):
            Y_col = Y[:, idx]
            predictor = SVC()
            predictor.fit(X, Y_col)
            self.predictors.append(predictor)

    def predict(self, X):
        """
        Predict the labels corresponding to data X.

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

class CatNBAttacker(CatSklearnAttacker):
    """The Categorical NaiveBaysian privacy attaker uses a naive bayesian classifier
    and the score is calculated based on prediction accuracy.
    """
    KEY_TYPE = CategoricalType.CLASS_NUM
    SENSITIVE_TYPE = CategoricalType.CLASS_NUM
    SKL_LEARNER = CategoricalNB

class CatNB(CatPrivacyMetric):
    """The Categorical NaiveBaysian privacy metric. Scored based on the CatNBAttacker.
    """

    name = 'Categorical NaiveBayesian'
    MODEL = CatNBAttacker
    ACCURACY_BASE = True

class KNNAttacker(CatSklearnAttacker):
    """The KNN (k nearest neighbors) privacy attaker uses a KNN classifier
    and the score is calculated based on prediction accuracy.
    """
    KEY_TYPE = CategoricalType.ONE_HOT
    SENSITIVE_TYPE = CategoricalType.CLASS_NUM
    SKL_LEARNER = KNeighborsClassifier

class KNN(CatPrivacyMetric):
    """The KNN privacy metric. Scored based on the KNNAttacker.
    """

    name = 'K Nearest Neighbors'
    MODEL = KNNAttacker
    ACCURACY_BASE = True

class CatRFAttacker(CatSklearnAttacker):
    """The Categorical RF (Random Forest) privacy attaker uses a RF classifier
    and the score is calculated based on prediction accuracy.
    """
    KEY_TYPE = CategoricalType.ONE_HOT
    SENSITIVE_TYPE = CategoricalType.CLASS_NUM
    SKL_LEARNER = RandomForestClassifier

class CatRF(CatPrivacyMetric):
    """The Categorical RF privacy metric. Scored based on the CatRFAttacker.
    """

    name = 'Categorical Random Forest'
    MODEL = CatRFAttacker
    ACCURACY_BASE = True

class CatSVMAttacker(CatSklearnAttacker):
    """The Categorical SVM (Support Vector Machine) privacy attaker uses a SVM classifier
    and the score is calculated based on prediction accuracy.
    """
    KEY_TYPE = CategoricalType.ONE_HOT
    SENSITIVE_TYPE = CategoricalType.CLASS_NUM
    SKL_LEARNER = SvcWrapper

class CatSVM(CatPrivacyMetric):
    """The Categorical SVM privacy metric. Scored based on the CatSVMAttacker.
    """

    name = 'Support Vector Classifier'
    MODEL = CatSVMAttacker
    ACCURACY_BASE = True