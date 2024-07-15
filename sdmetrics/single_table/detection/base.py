"""Base class for Machine Learning Detection metrics for single table datasets."""

import logging

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from sdmetrics.errors import IncomputableMetricError
from sdmetrics.goal import Goal
from sdmetrics.single_table.base import SingleTableMetric
from sdmetrics.utils import HyperTransformer, get_alternate_keys

LOGGER = logging.getLogger(__name__)


class DetectionMetric(SingleTableMetric):
    """Base class for Machine Learning Detection based metrics on single tables.

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

    name = 'SingleTable Detection'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def _fit_predict(X_train, y_train, X_test):
        """Fit a classifier and then use it to predict."""
        raise NotImplementedError()

    @staticmethod
    def _drop_non_compute_columns(real_data, synthetic_data, metadata):
        """Drop all columns that cannot be statistically modeled."""
        transformed_real_data = real_data
        transformed_synthetic_data = synthetic_data

        if metadata is not None:
            drop_columns = []
            drop_columns.extend(get_alternate_keys(metadata))
            for column in metadata.get('columns', []):
                if 'primary_key' in metadata and (
                    column == metadata['primary_key'] or column in metadata['primary_key']
                ):
                    drop_columns.append(column)

                column_info = metadata['columns'].get(column, {})
                sdtype = column_info.get('sdtype')
                pii = column_info.get('pii')
                if sdtype not in ['numerical', 'datetime', 'categorical'] or pii:
                    drop_columns.append(column)

            if drop_columns:
                transformed_real_data = real_data.drop(drop_columns, axis=1)
                transformed_synthetic_data = synthetic_data.drop(drop_columns, axis=1)
        return transformed_real_data, transformed_synthetic_data

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None):
        """Compute this metric.

        This builds a Machine Learning Classifier that learns to tell the synthetic
        data apart from the real data, which later on is evaluated using Cross Validation.

        The output of the metric is one minus the average ROC AUC score obtained.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict. If not passed, it is build based on the
                real_data fields and dtypes.

        Returns:
            float:
                One minus the ROC AUC Cross Validation Score obtained by the classifier.
        """
        real_data, synthetic_data, metadata = cls._validate_inputs(
            real_data, synthetic_data, metadata
        )

        transformed_real_data, transformed_synthetic_data = cls._drop_non_compute_columns(
            real_data, synthetic_data, metadata
        )

        ht = HyperTransformer()
        transformed_real_data = ht.fit_transform(transformed_real_data).to_numpy()
        transformed_synthetic_data = ht.transform(transformed_synthetic_data).to_numpy()
        X = np.concatenate([transformed_real_data, transformed_synthetic_data])
        y = np.hstack([
            np.ones(len(transformed_real_data)),
            np.zeros(len(transformed_synthetic_data)),
        ])
        if np.isin(X, [np.inf, -np.inf]).any():
            X[np.isin(X, [np.inf, -np.inf])] = np.nan

        try:
            scores = []
            kf = StratifiedKFold(n_splits=3, shuffle=True)
            for train_index, test_index in kf.split(X, y):
                y_pred = cls._fit_predict(X[train_index], y[train_index], X[test_index])
                roc_auc = roc_auc_score(y[test_index], y_pred)

                scores.append(max(0.5, roc_auc) * 2 - 1)

            return 1 - np.mean(scores)
        except ValueError as err:
            raise IncomputableMetricError(f'DetectionMetric: Unable to be fit with error {err}')

    @classmethod
    def normalize(cls, raw_score):
        """Return the `raw_score` as is, since it is already normalized.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                Simply returns `raw_score`.
        """
        return super().normalize(raw_score)
