"""BayesianNetwork based metrics for single table."""

import numpy as np
from pomegranate import BayesianNetwork

from sdmetrics.goal import Goal
from sdmetrics.single_table.base import SingleTableMetric


def _bayesian_likelihoods(real_data, synthetic_data):
    columns = real_data.select_dtypes(('object', 'bool')).columns
    bn = BayesianNetwork.from_samples(real_data[columns].to_numpy())

    probabilities = []
    for _, row in synthetic_data[columns].iterrows():
        try:
            probabilities.append(bn.probability([row.to_numpy()]))
        except ValueError:
            probabilities.append(0)

    return np.asarray(probabilities)


class BNLikelihood(SingleTableMetric):
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

    @staticmethod
    def compute(real_data, synthetic_data):
        """Compute this metric.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.

        Returns:
            float:
                Mean of the probabilities returned by the Bayesian Network.
        """
        return np.mean(_bayesian_likelihoods(real_data, synthetic_data))


class BNLogLikelihood(SingleTableMetric):
    """BayesianNetwork Log Likelihood Single Table metric.

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
    min_value = -np.inf
    max_value = np.inf

    @staticmethod
    def compute(real_data, synthetic_data):
        """Compute this metric.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.

        Returns:
            float:
                Mean of the log probabilities returned by the Bayesian Network.
        """
        likelihoods = _bayesian_likelihoods(real_data, synthetic_data)
        likelihoods[np.where(likelihoods == 0)] = 1e-8
        return np.mean(np.log(likelihoods))
