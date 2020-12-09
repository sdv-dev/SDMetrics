"""GaussianMixture based metrics for single table."""

import numpy as np
from sklearn.mixture import GaussianMixture

from sdmetrics.goal import Goal
from sdmetrics.single_table.base import SingleTableMetric


class GMLogLikelihood(SingleTableMetric):
    """GaussianMixture Single Table metric.

    This metric fits multiple GaussianMixture models to the real data and then
    evaluates how likely it is that the synthetic data belongs to the same
    distribution as the real data.

    The output is the average log likelihood across all the GMMs.

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

    name = 'GaussianMixture Log Likelihood'
    goal = Goal.MAXIMIZE
    min_value = -np.inf
    max_value = np.inf

    @staticmethod
    def compute(real_data, synthetic_data, n_components=(10, 20, 30), iterations=3):
        """Compute this metric.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            n_components (tuple[int]):
                Tuple indicating the number of components to use
                for the tests. Defaults to (10, 20, 30)
            iterations (int):
                Number of times that each number of components should
                be evaluated.

        Returns:
            float:
                Average score returned by the GaussianMixtures.
        """
        columns = real_data.select_dtypes('number').columns
        scores = []
        for _ in range(iterations):
            for nc in n_components:
                gmm = GaussianMixture(nc, covariance_type='diag')
                gmm.fit(real_data[columns])
                scores.append(gmm.score(synthetic_data[columns]))

        return np.mean(scores)
