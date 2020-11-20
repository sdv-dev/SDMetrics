import numpy as np

from sdmetrics.detection.tabular.sklearn import LogisticDetector, SVCDetector


def test_logistic_nan_inf():
    """Make sure that NaN and Inf inputs are handled without crashes."""
    detector = LogisticDetector()

    X = np.array([[1, 2, 3, np.inf, None]]).T
    y = np.array([1, 0, 0, 1, 1])
    detector.fit(X, y)

    detector.predict_proba(X)


def test_svc_nan_inf():
    """Make sure that NaN and Inf inputs are handled without crashes."""
    detector = SVCDetector()

    X = np.array([[1, 2, 3, np.inf, None]]).T
    y = np.array([1, 0, 0, 1, 1])
    detector.fit(X, y)

    detector.predict_proba(X)
