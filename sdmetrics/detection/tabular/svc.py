from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from sdmetrics.detection.tabular.base import TabularDetector


class SVCDetector(TabularDetector):

    name = "svc"

    def fit(self, X, y):
        """This function trains a sklearn pipeline with a robust scalar
        and a support vector classifier.

        Arguments:
            X (np.ndarray): The numerical features (i.e. transformed rows).
            y (np.ndarray): The binary classification target.
        """
        self.model = Pipeline([
            ('scalar', RobustScaler()),
            ('classifier', SVC(probability=True, gamma='scale')),
        ])
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
