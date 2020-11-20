import numpy as np

from sdmetrics.detection.tabular.logistic import LogisticDetector


def test_logistic_nan_inf():
    """Make sure that NaN and Inf inputs are handled without crashes."""
    ld = LogisticDetector()

    X = np.array([[1, 2, 3, np.inf, None]]).T
    y = np.array([1, 0, 0, 1, 1])
    ld.fit(X, y)

    ld.predict_proba(X)
