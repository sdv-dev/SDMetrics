"""Base class for Efficacy metrics for single table datasets."""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from sdmetrics.single_table.base import SingleTableMetric
from sdmetrics.utils import HyperTransformer


class MLEfficacyMetric(SingleTableMetric):
    """Base class for Machine Learning Efficacy metrics on single tables.

    These metrics fit a Machine Learning model on the training data and
    then evaluate it making predictions on the test data.

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
    def _fit_predict(cls, train_data, train_target, test_data, test_target):
        """Fit a model to the training data and make predictions for the test data."""
        del test_target  # delete argument which subclasses use but this method does not.
        unique_labels = np.unique(train_target)
        if len(unique_labels) == 1:
            predictions = np.full(len(test_data), unique_labels[0])
        else:
            ht = HyperTransformer()
            test_data = ht.fit_transform(test_data)
            train_data = ht.transform(train_data)

            test_data[np.isin(test_data, [np.inf, -np.inf])] = None
            train_data[np.isin(train_data, [np.inf, -np.inf])] = None

            model_kwargs = cls.MODEL_KWARGS.copy() if cls.MODEL_KWARGS else {}
            model = cls.MODEL(**model_kwargs)

            pipeline = Pipeline([
                ('imputer', SimpleImputer()),
                ('scaler', RobustScaler()),
                ('model', model)
            ])

            pipeline.fit(train_data, train_target)

            predictions = pipeline.predict(test_data)

        return predictions

    @classmethod
    def _validate_inputs(cls, test_data, train_data, metadata, target):
        test_data, train_data, metadata = super()._validate_inputs(
            test_data, train_data, metadata)
        if 'target' in metadata:
            target = metadata['target']
        elif target is None:
            raise TypeError('`target` must be passed either directly or inside `metadata`')

        return target

    @classmethod
    def _score(cls, scorer, test_target, predictions):
        scorer = scorer or cls.SCORER
        if isinstance(scorer, (list, tuple)):
            scorers = scorer
            return tuple((scorer(test_target, predictions) for scorer in scorers))
        else:
            return scorer(test_target, predictions)

    @classmethod
    def compute(cls, test_data, train_data, metadata=None, target=None, scorer=None):
        """Compute this metric.

        This fits a Machine Learning model on the training data and
        then evaluates it making predictions on the test data.

        A ``target`` column name must be given, either directly or as a first level
        entry in the ``metadata`` dict, which will be used as the target column for the
        Machine Learning prediction.

        Optionally, a list of ML scorer functions can be given. Otherwise, the default
        one for the type of problem is used.

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

        test_data = test_data.copy()
        train_data = train_data.copy()
        test_target = test_data.pop(target)
        train_target = train_data.pop(target)

        predictions = cls._fit_predict(train_data, train_target, test_data, test_target)

        return cls._score(scorer, test_target, predictions)
