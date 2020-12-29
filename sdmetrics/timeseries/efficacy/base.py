"""Base class for Machine Learning Efficacy based metrics for Time Series."""

import numpy as np
import pandas as pd
import rdt
from sklearn.model_selection import train_test_split

from sdmetrics.goal import Goal
from sdmetrics.timeseries.base import TimeSeriesMetric


class TimeSeriesEfficacyMetric(TimeSeriesMetric):
    """Base class for Machine Learning Efficacy based metrics on time series.

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

    name = 'TimeSeries Efficacy'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = np.inf

    @classmethod
    def _validate_inputs(cls, real_data, synthetic_data, metadata, entity_columns, target):
        metadata, entity_columns = super()._validate_inputs(
            real_data, synthetic_data, metadata, entity_columns)

        if 'target' in metadata:
            target = metadata['target']
        elif target is None:
            raise TypeError('`target` must be passed either directly or inside `metadata`')

        return entity_columns, target

    @staticmethod
    def _build_xy(transformer, data, entity_columns, target_column):
        X = pd.DataFrame()
        y = pd.Series()
        for entity_id, group in data.groupby(entity_columns):
            y = y.append(pd.Series({entity_id: group.pop(target_column).iloc[0]}))
            entity_data = group.drop(entity_columns, axis=1)
            entity_data = transformer.transform(entity_data)
            entity_data = pd.Series({
                column: entity_data[column].to_numpy()
                for column in entity_data.columns
            }, name=entity_id)
            X = X.append(entity_data)

        return X, y

    @classmethod
    def _compute_score(cls, real_data, synthetic_data, entity_columns, target):
        transformer = rdt.HyperTransformer(dtype_transformers={
            'O': 'one_hot_encoding',
            'M': rdt.transformers.DatetimeTransformer(strip_constant=True),
        })
        transformer.fit(real_data.drop(entity_columns + [target], axis=1))

        real_x, real_y = cls._build_xy(transformer, real_data, entity_columns, target)
        synt_x, synt_y = cls._build_xy(transformer, synthetic_data, entity_columns, target)

        train, test = train_test_split(real_x.index, shuffle=True)
        real_x_train, real_x_test = real_x.loc[train], real_x.loc[test]
        real_y_train, real_y_test = real_y.loc[train], real_y.loc[test]

        real_acc = cls._scorer(real_x_train, real_x_test, real_y_train, real_y_test)
        synt_acc = cls._scorer(synt_x, real_x_test, synt_y, real_y_test)

        return synt_acc / real_acc

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, entity_columns=None, target=None):
        """Compute this metric.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset, passed as a pandas.DataFrame.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset, passed as a pandas.DataFrame.
            metadata (dict):
                TimeSeries metadata dict. If not passed, it is build based on the
                real_data fields and dtypes.
            entity_columns (list[str]):
                Names of the columns which identify different time series
                sequences.
            target (str):
                Name of the column to use as the target.

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        entity_columns, target = cls._validate_inputs(
            real_data, synthetic_data, metadata, entity_columns, target)

        return cls._compute_score(real_data, synthetic_data, entity_columns, target)
