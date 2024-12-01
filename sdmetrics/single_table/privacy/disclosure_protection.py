"""DisclosureProtection metrics."""

import numpy as np
import pandas as pd

from sdmetrics.single_table.base import SingleTableMetric
from sdmetrics.single_table.privacy.cap import (
    CategoricalCAP,
    CategoricalGeneralizedCAP,
    CategoricalZeroCAP,
)

CAP_METHODS = {
    'cap': CategoricalCAP,
    'zero_cap': CategoricalZeroCAP,
    'generalized_cap': CategoricalGeneralizedCAP,
}


class DisclosureProtection(SingleTableMetric):
    """The DisclosureProtection metric."""

    @classmethod
    def _validate_inputs(
        cls,
        real_data,
        synthetic_data,
        known_column_names,
        sensitive_column_names,
        computation_method,
        continuous_column_names,
        num_discrete_bins,
    ):
        assert isinstance(real_data, pd.DataFrame)
        assert isinstance(synthetic_data, pd.DataFrame)
        assert set(synthetic_data.columns).issuperset(set(real_data.columns))
        assert all(col in real_data.columns for col in known_column_names)
        assert all(col in real_data.columns for col in sensitive_column_names)
        assert computation_method in CAP_METHODS.keys()
        if continuous_column_names is not None:
            assert all(col in real_data.columns for col in continuous_column_names)
        assert isinstance(num_discrete_bins, int)

    @classmethod
    def _get_null_categories(cls, real_data, synthetic_data, columns):
        base_null_value = '__NULL_VALUE__'
        null_category_map = {}
        for col in columns:
            null_value = base_null_value
            categories = set(real_data[col].unique()).union(set(synthetic_data[col].unique()))
            while null_value in categories:
                null_value += '_'

            null_category_map[col] = null_value

        return null_category_map

    @classmethod
    def _discretize_column(cls, real_column, synthetic_column, num_bins):
        bin_labels = [str(x) for x in range(num_bins)]
        real_binned, bins = pd.cut(real_column, num_bins, labels=bin_labels, retbins=True)
        bins[0], bins[-1] = -np.inf, np.inf
        synthetic_binned = pd.cut(synthetic_column, bins, labels=bin_labels)

        return real_binned, synthetic_binned

    @classmethod
    def _compute_baseline(cls, real_data, synthetic_data, sensitive_column_names):
        unique_categories_prod = np.prod([
            real_data[col].nunique(dropna=False) for col in sensitive_column_names
        ])
        return 1 - float(1 / unique_categories_prod)

    @classmethod
    def compute_breakdown(
        cls,
        real_data,
        synthetic_data,
        known_column_names,
        sensitive_column_names,
        computation_method='cap',
        continuous_column_names=None,
        num_discrete_bins=10,
    ):
        """Compute this metric breakdown.

        Args:
            real_data (pd.DataFrame):
                A pd.DataFrame with the real data.
            synthetic_data (pd.DataFrame):
                A pd.DataFrame with the synthetic data.
            known_column_names (list[str]):
                A list with the string names of the columns that an attacker may already know.
            sensitive_column_names (list[str]):
                A list with the string names of the columns that an attacker wants to guess
                (but does not already know).
            computation_method (str, optional):
                The type of computation we'll use to simulate the attack. Options are:
                    - 'cap':  Use the CAP method described in the original paper.
                    - 'generalized_cap': Use the generalized CAP method.
                    - 'zero_cap': Use the zero cap method.
                Defaults to 'cap'.
            continuous_column_names (list[str], optional):
                A list of column names that represent continuous values (as opposed to discrete
                values). These columns will be discretized. Defaults to None.
            num_discrete_bins (int, optional):
                Number of bins to discretize continous columns in to. Defaults to 10.

        Returns:
            dict
                Mapping of the metric output. Must include the key 'score'.
        """
        cls._validate_inputs(
            real_data,
            synthetic_data,
            known_column_names,
            sensitive_column_names,
            computation_method,
            continuous_column_names,
            num_discrete_bins,
        )
        real_data = real_data.copy()
        synthetic_data = synthetic_data.copy()

        # Discretize continous columns
        if continuous_column_names is not None:
            for col_name in continuous_column_names:
                real_data[col_name], synthetic_data[col_name] = cls._discretize_column(
                    real_data[col_name], synthetic_data[col_name], num_discrete_bins
                )

        # Convert null values to own category
        null_category_map = cls._get_null_categories(
            real_data, synthetic_data, known_column_names + sensitive_column_names
        )
        real_data = real_data.fillna(null_category_map)
        synthetic_data = synthetic_data.fillna(null_category_map)

        # Compute baseline
        baseline_protection = cls._compute_baseline(
            real_data, synthetic_data, sensitive_column_names
        )

        # Compute CAP metric
        cap_metric = CAP_METHODS.get(computation_method)
        cap_protection = cap_metric.compute(
            real_data,
            synthetic_data,
            key_fields=known_column_names,
            sensitive_fields=sensitive_column_names,
        )

        score = min(cap_protection / baseline_protection, 1)

        return {
            'score': score,
            'cap_protection': cap_protection,
            'baseline_protection': baseline_protection,
        }

    @classmethod
    def compute(
        cls,
        real_data,
        synthetic_data,
        known_column_names,
        sensitive_column_names,
        computation_method='cap',
        continuous_column_names=None,
        num_discrete_bins=10,
    ):
        """Compute the DisclosureProtection metric.

        Args:
            real_data (pd.DataFrame):
                A pd.DataFrame with the real data.
            synthetic_data (pd.DataFrame):
                A pd.DataFrame with the synthetic data.
            known_column_names (list[str]):
                A list with the string names of the columns that an attacker may already know.
            sensitive_column_names (list[str]):
                A list with the string names of the columns that an attacker wants to guess
                (but does not know).
            computation_method (str, optional):
                The type of computation we'll use to simulate the attack. Options are:
                    - 'cap':  Use the CAP method described in the original paper.
                    - 'generalized_cap': Use the generalized CAP method.
                    - 'zero_cap': Use the zero cap method.
                Defaults to 'cap'.
            continuous_column_names (list[str], optional):
                A list of column names that represent continuous values (as opposed to discrete
                values). These columns will be discretized. Defaults to None.
            num_discrete_bins (int, optional):
                Number of bins to discretize continous columns in to. Defaults to 10.

        Returns:
            float:
                The score for the DisclosureProtection metric.
        """
        score_breakdown = cls.compute_breakdown(
            real_data,
            synthetic_data,
            known_column_names,
            sensitive_column_names,
            computation_method,
            continuous_column_names,
            num_discrete_bins,
        )
        return score_breakdown['score']
