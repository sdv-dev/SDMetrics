"""Machine Learning Detection based metrics for Time Series."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sdmetrics.goal import Goal
from sdmetrics.timeseries import ml_scorers
from sdmetrics.timeseries.base import TimeSeriesMetric
from sdmetrics.utils import HyperTransformer


class TimeSeriesDetectionMetric(TimeSeriesMetric):
    """Base class for Machine Learning Detection based metrics on time series.

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

    name = 'TimeSeries Detection'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def _build_x(data, hypertransformer, sequence_key):
        X = pd.DataFrame()
        for entity_id, entity_data in data.groupby(sequence_key):
            entity_data = entity_data.drop(sequence_key, axis=1)
            entity_data = hypertransformer.transform(entity_data)
            entity_data = pd.Series(
                {column: entity_data[column].to_numpy() for column in entity_data.columns},
                name=entity_id,
            )

            X = pd.concat([X, pd.DataFrame(entity_data).T], ignore_index=True)

        return X

    @staticmethod
    def _compute_score(X_train, X_test, y_train, y_test):
        """Fit a classifier and then use it to predict."""
        raise NotImplementedError()

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, sequence_key=None):
        """Compute this metric.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset, passed as a pandas.DataFrame.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset, passed as a pandas.DataFrame.
            metadata (dict):
                TimeSeries metadata dict. If not passed, it is build based on the
                real_data fields and dtypes.
            sequence_key (list[str]):
                Names of the columns which identify different time series
                sequences.

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        real_data, synthetic_data = real_data.copy(), synthetic_data.copy()
        _, sequence_key = cls._validate_inputs(real_data, synthetic_data, metadata, sequence_key)

        ht = HyperTransformer()
        ht.fit(real_data.drop(sequence_key, axis=1))

        real_x = cls._build_x(real_data, ht, sequence_key)
        synt_x = cls._build_x(synthetic_data, ht, sequence_key)

        X = pd.concat([real_x, synt_x])
        y = pd.Series(np.array([0] * len(real_x) + [1] * len(synt_x)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y)

        return 1 - cls._compute_score(X_train, X_test, y_train, y_test)

    @classmethod
    def normalize(cls, raw_score):
        """Return the `raw_score` as is, since it is already normalized.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                The normalized value of the metric
        """
        assert cls.min_value == 0.0
        return super().normalize(raw_score)


class LSTMDetection(TimeSeriesDetectionMetric):
    """TimeSeriesDetection metric based on an LSTM Classifier."""

    _compute_score = ml_scorers.lstm_classifier
