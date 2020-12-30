"""Base class for Efficacy metrics for single table datasets."""

import numpy as np
import rdt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

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
    def _fit_predict(cls, synthetic_data, synthetic_target, real_data, real_target):
        """Fit a model in the synthetic data and make predictions for the real data."""
        del real_target  # delete argument which subclasses use but this method does not.
        unique_labels = np.unique(synthetic_target)
        if len(unique_labels) == 1:
            predictions = np.full(len(real_data), unique_labels[0])
        else:
            transformer = rdt.HyperTransformer(dtype_transformers={'O': 'one_hot_encoding'})
            real_data = transformer.fit_transform(real_data)
            synthetic_data = transformer.transform(synthetic_data)

            real_data[np.isin(real_data, [np.inf, -np.inf])] = None
            synthetic_data[np.isin(synthetic_data, [np.inf, -np.inf])] = None

            model_kwargs = cls.MODEL_KWARGS.copy() if cls.MODEL_KWARGS else {}
            model = cls.MODEL(**model_kwargs)

            pipeline = Pipeline([
                ('imputer', SimpleImputer()),
                ('scaler', RobustScaler()),
                ('model', model)
            ])

            pipeline.fit(synthetic_data, synthetic_target)

            predictions = pipeline.predict(real_data)

        return predictions

    @classmethod
    def _validate_inputs(cls, real_data, synthetic_data, metadata, target):
        metadata = super()._validate_inputs(real_data, synthetic_data, metadata)
        if 'target' in metadata:
            target = metadata['target']
        elif target is None:
            raise TypeError('`target` must be passed either directly or inside `metadata`')

        return target

    @classmethod
    def _score(cls, scorer, real_target, predictions):
        scorer = scorer or cls.SCORER
        if isinstance(scorer, (list, tuple)):
            scorers = scorer
            return tuple((scorer(real_target, predictions) for scorer in scorers))
        else:
            return scorer(real_target, predictions)

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, target=None, scorer=None):
        """Compute this metric.

        This fits a Machine Learning model on the synthetic data and
        then evaluates it making predictions on the real data.

        A ``target`` column name must be given, either directly or as a first level
        entry in the ``metadata`` dict, which will be used as the target column for the
        Machine Learning prediction.

        Optionally, a list of ML scorer functions can be given. Otherwise, the default
        one for the type of problem is used.

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
        target = cls._validate_inputs(real_data, synthetic_data, metadata, target)

        real_data = real_data.copy()
        synthetic_data = synthetic_data.copy()
        real_target = real_data.pop(target)
        synthetic_target = synthetic_data.pop(target)

        predictions = cls._fit_predict(synthetic_data, synthetic_target, real_data, real_target)

        return cls._score(scorer, real_target, predictions)
