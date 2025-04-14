"""BayesianNetwork based metrics for single table."""

import logging

import numpy as np
import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.single_table.base import SingleTableMetric

try:
    import torch
except ModuleNotFoundError:
    torch = None

LOGGER = logging.getLogger(__name__)


class BNLikelihoodBase(SingleTableMetric):
    """BayesianNetwork Likelihood Single Table base metric."""

    @classmethod
    def _likelihoods(cls, real_data, synthetic_data, metadata=None, structure=None):
        try:
            from pomegranate.bayesian_network import BayesianNetwork

            if torch is None:
                raise ImportError
        except ImportError:
            raise ImportError(
                'Please install pomegranate with `pip install sdmetrics[pomegranate]`.'
            )

        real_data, synthetic_data, metadata = cls._validate_inputs(
            real_data, synthetic_data, metadata
        )
        structure = metadata.get('structure', structure)
        fields = cls._select_fields(metadata, ('categorical', 'boolean'))

        if not fields:
            return np.full(len(real_data), np.nan)

        LOGGER.debug('Fitting the BayesianNetwork to the real data')
        bn = BayesianNetwork(structure=structure if structure else None, algorithm='chow-liu')
        category_to_integer = {
            column: {
                category: i
                for i, category in enumerate(
                    pd.concat([real_data[column], synthetic_data[column]]).unique()
                )
            }
            for column in fields
        }
        real_data[fields] = real_data[fields].replace(category_to_integer).astype('int64')
        synthetic_data[fields] = synthetic_data[fields].replace(category_to_integer).astype('int64')

        bn.fit(torch.tensor(real_data[fields].to_numpy()))
        LOGGER.debug('Evaluating likelihood of the synthetic data')
        probabilities = []
        for _, row in synthetic_data[fields].iterrows():
            try:
                probabilities.append(bn.probability([row.to_numpy()]).item())
            except ValueError:
                probabilities.append(0)

        return np.asarray(probabilities)


class BNLikelihood(BNLikelihoodBase):
    """BayesianNetwork Likelihood Single Table metric.

    This metric fits a BayesianNetwork to the real data and then evaluates how
    likely it is that the synthetic data belongs to the same distribution.

    The output is the average probability across all the synthetic rows.

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

    name = 'BayesianNetwork Likelihood'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, structure=None):
        """Compute this metric.

        This fits a BayesianNetwork to the real data and then evaluates how
        likely it is that the synthetic data belongs to the same distribution.

        Real data and synthetic data must be passed as ``pandas.DataFrame`` instances
        and ``metadata`` as a ``Table`` metadata ``dict`` representation.

        If no ``metadata`` is given, one will be built from the values observed
        in the ``real_data``.

        If a ``structure`` is given, either directly or as a ``structure`` first level
        entry within the ``metadata`` dict, it is passed to the underlying BayesianNetwork
        for fitting. Otherwise, the structure is learned from the data using the ``chow-liu``
        algorithm.

        ``structure`` can be passed as either a tuple of tuples representing only the
        network structure or as a ``dict`` representing a full serialization of a previously
        fitted ``BayesianNetwork``. In the later scenario, only the ``structure`` will be
        extracted from the ``BayesianNetwork`` instance, and then a new one will be fitted
        to the given data.

        The output is the average probability across all the synthetic rows.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict. If not passed, it is build based on the
                real_data fields and dtypes. Optionally, the metadata can include
                a ``structure`` entry with the structure of the Bayesian Network.
            structure (dict):
                Optional. BayesianNetwork structure to use when fitting
                to the real data. If not passed, learn it from the data
                using the ``chow-liu`` algorith. This is ignored if ``metadata``
                is passed and it contains a ``structure`` entry in it.

        Returns:
            float:
                Mean of the probabilities returned by the Bayesian Network.
        """
        return np.mean(cls._likelihoods(real_data, synthetic_data, metadata, structure))


class BNLogLikelihood(BNLikelihoodBase):
    """BayesianNetwork Log Likelihood Single Table metric.

    This metric fits a BayesianNetwork to the real data and then evaluates how
    likely it is that the synthetic data belongs to the same distribution.

    The output is the average log probability across all the synthetic rows.

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

    name = 'BayesianNetwork Log Likelihood'
    goal = Goal.MAXIMIZE
    min_value = -np.inf
    max_value = 0

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, structure=None):
        """Compute this metric.

        This fits a BayesianNetwork to the real data and then evaluates how
        likely it is that the synthetic data belongs to the same distribution.

        Real data and synthetic data must be passed as ``pandas.DataFrame`` instances
        and ``metadata`` as a ``Table`` metadata ``dict`` representation.

        If no ``metadata`` is given, one will be built from the values observed
        in the ``real_data``.

        If a ``structure`` is given, either directly or as a ``structure`` first level
        entry within the ``metadata`` dict, it is passed to the underlying BayesianNetwork
        for fitting. Otherwise, the structure is learned from the data using the ``chow-liu``
        algorithm.

        ``structure`` can be passed as either a tuple of tuples representing only the
        network structure or as a ``dict`` representing a full serialization of a previously
        fitted ``BayesianNetwork``. In the later scenario, only the ``structure`` will be
        extracted from the ``BayesianNetwork`` instance, and then a new one will be fitted
        to the given data.

        The output is the average log probability across all the synthetic rows.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict. If not passed, it is build based on the
                real_data fields and dtypes. Optionally, the metadata can include
                a ``structure`` entry with the structure of the Bayesian Network.
            structure (dict):
                Optional. BayesianNetwork structure to use when fitting
                to the real data. If not passed, learn it from the data
                using the ``chow-liu`` algorith. This is ignored if ``metadata``
                is passed and it contains a ``structure`` entry in it.

        Returns:
            float:
                Mean of the log probabilities returned by the Bayesian Network.
        """
        likelihoods = cls._likelihoods(real_data, synthetic_data, metadata, structure)
        likelihoods[np.where(likelihoods == 0)] = 1e-8
        return np.mean(np.log(likelihoods))

    @classmethod
    def normalize(cls, raw_score):
        """Normalize the log-likelihood value.

        Note that this is not the mean likelihood but rather the exponentiation
        of the mean log-likelihood.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                The normalized value of the metric
        """
        return super().normalize(raw_score)
