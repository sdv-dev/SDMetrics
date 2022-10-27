"""MLEfficacy metric for single table datasets."""

import logging

import numpy as np

from sdmetrics.goal import Goal
from sdmetrics.single_table.efficacy.base import MLEfficacyMetric
from sdmetrics.single_table.efficacy.binary import BinaryEfficacyMetric
from sdmetrics.single_table.efficacy.multiclass import MulticlassEfficacyMetric
from sdmetrics.single_table.efficacy.regression import RegressionEfficacyMetric
from sdmetrics.utils import get_columns_from_metadata, get_type_from_column_meta

LOGGER = logging.getLogger(__name__)


class MLEfficacy(MLEfficacyMetric):
    """Problem and ML Model agnostic efficacy metric.

    This metric analyzes the target column and applies all the Regression, Binary
    Classification or Multiclass Classification metrics to the table depending
    on the type of column that needs to be predicted.

    The output is the average score obtained by the different metrics of the
    chosen type.
    """

    name = 'Machine Learning Efficacy'
    goal = Goal.MAXIMIZE
    min_value = -np.inf
    max_value = np.inf

    @classmethod
    def compute(cls, test_data, train_data, metadata=None, target=None):
        """Compute this metric.

        A ``target`` column name must be given, either directly or as a first level
        entry in the ``metadata`` dict, which will be used as the target column for the
        Machine Learning prediction.

        This analyzes the target column and applies all the Regression, Binary
        Classification or Multiclass Classification metrics to the table depending
        on the type of column that needs to be predicted.

        The output is the average score obtained by the different metrics of the
        chosen type.

        Args:
            test_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the test dataset.
            train_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the training dataset.
            target (str):
                Name of the column to use as the target.
            scorer (Union[callable, list[callable], NoneType]):
                Scorer (or list of scorers) to apply. If not passed, use the default
                one for the type of metric.

        Returns:
            union[float, tuple[float]]:
                Scores obtained by the models when evaluated on the test data.
        """
        target = cls._validate_inputs(test_data, train_data, metadata, target)
        target_type = get_type_from_column_meta(get_columns_from_metadata(metadata)[target])
        target_data = test_data[target]
        uniques = target_data.unique()
        if len(uniques) == 2:
            LOGGER.info('MLEfficacy: Selecting Binary Classification metrics')
            metrics = BinaryEfficacyMetric.get_subclasses()
        elif target_type == 'numerical':
            LOGGER.info('MLEfficacy: Selecting Regression metrics')
            metrics = RegressionEfficacyMetric.get_subclasses()
        elif target_type == 'categorical':
            LOGGER.info('MLEfficacy: Selecting Multiclass Classification metrics')
            metrics = MulticlassEfficacyMetric.get_subclasses()
        else:
            raise ValueError(f'Unsupported target type: {target_type}')

        scores = []
        for name, metric in metrics.items():
            LOGGER.info('MLEfficacy: Computing %s', name)
            scores.append(metric.compute(test_data, train_data, metadata, target))

    @classmethod
    def normalize(cls, raw_score):
        """Return a normalized version of the `raw_score`.

        This normalizes the raw score by applying the sigmoid function.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                The normalized value of the metric
        """
        return super().normalize(raw_score)
