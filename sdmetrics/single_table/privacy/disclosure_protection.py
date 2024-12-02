"""DisclosureProtection metrics."""

import numpy as np
import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.single_table.base import SingleTableMetric
from sdmetrics.single_table.privacy.cap import (
    CategoricalCAP,
    CategoricalGeneralizedCAP,
    CategoricalZeroCAP,
)

CAP_METHODS = {
    'CAP': CategoricalCAP,
    'ZERO_CAP': CategoricalZeroCAP,
    'GENERALIZED_CAP': CategoricalGeneralizedCAP,
}


class DisclosureProtection(SingleTableMetric):
    """The DisclosureProtection metric."""

    goal = Goal.MAXIMIZE
    min_value = 0
    max_value = 1

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
        if not isinstance(real_data, pd.DataFrame) or not isinstance(real_data, pd.DataFrame):
            raise ValueError('Real and synthetic data must be pandas DataFrames.')

        if len(known_column_names) == 0:
            raise ValueError('Must provide at least 1 known column name.')
        elif not set(real_data.columns).issuperset(set(known_column_names)):
            missing = "', '".join(set(known_column_names) - set(real_data.columns))
            raise ValueError(f"Known column(s) '{missing}' are missing from the real data.")

        if len(sensitive_column_names) == 0:
            raise ValueError('Must provide at least 1 sensitive column name.')
        elif not set(real_data.columns).issuperset(set(sensitive_column_names)):
            missing = "', '".join(set(sensitive_column_names) - set(real_data.columns))
            raise ValueError(f"Sensitive column(s) '{missing}' are missing from the real data.")

        if computation_method.upper() not in CAP_METHODS.keys():
            raise ValueError(
                f"Unknown computation method '{computation_method}'. "
                f"Please use one of 'cap', 'zero_cap', or 'generalized_cap'."
            )

        if continuous_column_names is not None and not set(real_data.columns).issuperset(
            set(continuous_column_names)
        ):
            missing = "', '".join(set(continuous_column_names) - set(real_data.columns))
            raise ValueError(f"Continous column(s) '{missing}' are missing from the real data.")

        if not isinstance(num_discrete_bins, int) or num_discrete_bins <= 0:
            raise ValueError('`num_discrete_bins` must be an integer greater than zero.')

        super()._validate_inputs(real_data, synthetic_data)

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

        return real_binned.to_numpy(), synthetic_binned.to_numpy()

    @classmethod
    def _compute_baseline(cls, real_data, sensitive_column_names):
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
        computation_method = computation_method.upper()
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
        baseline_protection = cls._compute_baseline(real_data, sensitive_column_names)

        # Compute CAP metric
        cap_metric = CAP_METHODS.get(computation_method)
        cap_protection = cap_metric.compute(
            real_data,
            synthetic_data,
            key_fields=known_column_names,
            sensitive_fields=sensitive_column_names,
        )

        if baseline_protection == 0:
            score = 0 if cap_protection == 0 else 1
        else:
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
