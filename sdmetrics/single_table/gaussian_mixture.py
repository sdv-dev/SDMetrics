"""GaussianMixture based metrics for single table."""
import itertools
import logging

import numpy as np
from sklearn.mixture import GaussianMixture

from sdmetrics.goal import Goal
from sdmetrics.single_table.base import SingleTableMetric

LOGGER = logging.getLogger(__name__)


class GMLogLikelihood(SingleTableMetric):
    """GaussianMixture Single Table metric.

    This metric fits multiple GaussianMixture models to the real data and then
    evaluates how likely it is that the synthetic data belongs to the same
    distribution as the real data.

    By default, GaussianMixture models with 10, 20 and 30 components are
    fitted a total of 3 times.

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

    @classmethod
    def _select_gmm(cls, real_data, n_components, covariance_type):
        if isinstance(n_components, int):
            min_comp = max_comp = n_components
        else:
            min_comp, max_comp = n_components

        if isinstance(covariance_type, str):
            covariance_type = (covariance_type, )

        combinations = list(itertools.product(range(min_comp, max_comp + 1), covariance_type))
        if len(combinations) == 1:
            return combinations[0]

        lowest_bic = np.inf
        best = None
        for n_components, covariance_type in combinations:
            gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
            try:
                gmm.fit(real_data)
                bic = gmm.bic(real_data)
                LOGGER.debug('%s, %s: %s', n_components, covariance_type, bic)
                if bic < lowest_bic:
                    lowest_bic = bic
                    best = (n_components, covariance_type)

            except ValueError:
                pass

        return best

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, n_components=(1, 30),
                covariance_type='diag', iterations=3, retries=3):
        """Compute this metric.

        This fits multiple GaussianMixture models to the real data and then
        evaluates how likely it is that the synthetic data belongs to the same
        distribution as the real data.

        By default, GaussianMixture models will search for the optimal number of
        components and covariance type using the real data and then evaluate
        the likelihood of the synthetic data using those arguments 3 times.

        Real data and synthetic data must be passed as ``pandas.DataFrame`` instances
        and ``metadata`` as a ``Table`` metadata ``dict`` representation.

        If no ``metadata`` is given, one will be built from the values observed
        in the ``real_data``.

        The output is the average log likelihood across all the GMMs evaluated.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict.
            n_components (Union[int, tuple[int]]):
                Number of components to use for the GMM. If a tuple with
                2 integers is passed, the optimal number of components within
                the range will be searched. Defaults to (1, 30)
            covariance_type (Union[str, tuple[str]]):
                Covariange type to use for the GMM. If multiple values are
                passed, the best one will be searched. Defaults to ``'diag'``.
            iterations (int):
                Number of times that each number of components should
                be evaluated before averaging the scores. Defaults to 3.
            retries (int):
                Number of times that each iteration will be retried if the
                GMM model crashes during fit. Defaults to 3.

        Returns:
            float:
                Average score returned by the GaussianMixtures.
        """
        metadata = cls._validate_inputs(real_data, synthetic_data, metadata)
        fields = cls._select_fields(metadata, 'numerical')
        if not fields:
            LOGGER.debug('No numerical fields found. Returning NaN.')
            return np.nan

        real_data = real_data[fields]
        synthetic_data = synthetic_data[fields]
        real_data = real_data.fillna(real_data.mean())
        synthetic_data = synthetic_data.fillna(synthetic_data.mean())

        if not isinstance(n_components, int) or not isinstance(covariance_type, str):
            LOGGER.debug('Selecting best GMM parameters')
            best_gmm = cls._select_gmm(real_data, n_components, covariance_type)
            if best_gmm is None:
                return np.nan

            n_components, covariance_type = best_gmm
            LOGGER.debug('n_components=%s and covariance_type=%s selected',
                         n_components, covariance_type)

        scores = []
        for _ in range(iterations * retries):
            try:
                gmm = GaussianMixture(n_components, covariance_type=covariance_type)
                gmm.fit(real_data)
                scores.append(gmm.score(synthetic_data))
                if len(scores) >= iterations:
                    break
            except ValueError:
                pass

        return np.mean(scores)

    @classmethod
    def normalize(cls, raw_score):
        """Normalize the log-likelihood value.

        Notice that this is not the mean likelihood.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                The normalized value of the metric
        """
        return super().normalize(raw_score)
