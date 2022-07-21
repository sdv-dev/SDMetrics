"""Machine Learning Classification Efficacy based metrics for Time Series."""

from sdmetrics.timeseries import ml_scorers
from sdmetrics.timeseries.efficacy.base import TimeSeriesEfficacyMetric


class TimeSeriesClassificationEfficacyMetric(TimeSeriesEfficacyMetric):
    """TimeSeriesEfficacy metrics for Time Series Classification problems."""


class LSTMClassifierEfficacy(TimeSeriesClassificationEfficacyMetric):
    """TimeSeriesEfficacy metric based on an LSTM Classifier."""

    _scorer = ml_scorers.lstm_classifier
