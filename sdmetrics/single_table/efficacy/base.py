"""Base class for Efficacy metrics for single table datasets."""

import numpy as np
from rdt import HyperTransformer

from sdmetrics.single_table.base import SingleTableMetric


class MLEfficacyMetric(SingleTableMetric):
    """Base class for Machine Learning Efficacy metrics on single tables.

    These metrics fit a Machine Learning model on the synthetic data and
    then evaluate it making predictions on the real data.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
        model:
            Model class to use for the prediction.
        model_kwargs:
            Keyword arguments to use to create the model instance.
    """

    name = None
    goal = None
    min_value = None
    max_value = None
    MODEL = None
    MODEL_KWARGS = None
    METRICS = None

    @classmethod
    def _fit_predict(cls, synthetic_data, synthetic_target, real_data):
        """Fit a model in the synthetic data and make predictions for the real data."""
        unique_labels = np.unique(synthetic_target)
        if len(unique_labels) == 1:
            predictions = np.full(len(real_data), unique_labels[0])
        else:
            model_kwargs = cls.model_kwargs.copy() if cls.model_kwargs else {}
            model = cls.model(**model_kwargs)
            model.fit(synthetic_data, synthetic_target)
            predictions = model.predict(real_data)

        return predictions

    @staticmethod
    def _compute_scores(real_target, predictions):
        """Compute scores comparing the real targets and the predictions."""
        raise NotImplementedError()

    @classmethod
    def compute(cls, real_data, synthetic_data, target, scorer=None):
        """Compute this metric.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            target (str):
                Name of the column to use as the target.
            scorer (Union[callable, list[callable], NoneType]):
                Scorer (or list of scorers) to apply. If not passed, use the default
                one for the type of metric.

        Returns:
            union[float, tuple[float]]:
                Scores obtained by the models when evaluated on the real data.
        """
        real_data = real_data.copy()
        synthetic_data = synthetic_data.copy()
        real_target = real_data.pop(target)
        synthetic_target = synthetic_data.pop(target)

        transformer = HyperTransformer()
        real_data = transformer.fit_transform(real_data)
        synthetic_data = transformer.transform(synthetic_data)

        predictions = cls._fit_predict(synthetic_data, synthetic_target, real_data)

        scorer = scorer or cls.SCORER
        if isinstance(scorer, (list, tuple)):
            scorers = scorer
            return tuple((scorer(real_target, predictions) for scorer in scorers))
        else:
            return scorer(real_target, predictions)
