"""Disclosure protection metrics."""

import warnings

import numpy as np
import pandas as pd
import tqdm

from sdmetrics.goal import Goal
from sdmetrics.single_table.base import SingleTableMetric
from sdmetrics.single_table.privacy.cap import (
    CategoricalCAP,
    CategoricalGeneralizedCAP,
    CategoricalZeroCAP,
)

MAX_NUM_ROWS = 10000

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
        real_binned, bins = pd.cut(
            pd.to_numeric(real_column.to_numpy()), num_bins, labels=bin_labels, retbins=True
        )
        bins[0], bins[-1] = -np.inf, np.inf
        synthetic_binned = pd.cut(
            pd.to_numeric(synthetic_column.to_numpy()), bins, labels=bin_labels
        )

        return real_binned.to_numpy(), synthetic_binned.to_numpy()

    @classmethod
    def _discretize_and_fillna(
        cls,
        real_data,
        synthetic_data,
        known_column_names,
        sensitive_column_names,
        continuous_column_names,
        num_discrete_bins,
    ):
        """Helper to discretize continous columns and convert null values to categories.

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
            continuous_column_names (list[str]):
                A list of column names that represent continuous values (as opposed to discrete
                values). These columns will be discretized. Defaults to None.
            num_discrete_bins (int):
                Number of bins to discretize continous columns in to. Defaults to 10.

        Returns:
            tuple(pd.DataFrame, pd.DataFrame):
                The pre-processed real and synthetic data.
        """
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
        return real_data, synthetic_data

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
                Mapping of the metric output with the keys:
                    - 'score': The overall score for the metric.
                    - 'cap_protection': The protection score from the selected computation method.
                    - 'baseline_protection': The baseline protection for the columns.
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
        if len(real_data) > MAX_NUM_ROWS or len(synthetic_data) > MAX_NUM_ROWS:
            warnings.warn(
                f'Data exceeds {MAX_NUM_ROWS} rows, perfomance may be slow. '
                'Consider using the `DisclosureProtectionEstimate` for faster computation.'
            )

        real_data, synthetic_data = cls._discretize_and_fillna(
            real_data,
            synthetic_data,
            known_column_names,
            sensitive_column_names,
            continuous_column_names,
            num_discrete_bins,
        )

        # Compute baseline
        baseline_protection = cls._compute_baseline(real_data, sensitive_column_names)

        # Compute CAP metric
        cap_metric = CAP_METHODS.get(computation_method)
        cap_protection = cap_metric._compute(
            real_data,
            synthetic_data,
            key_fields=known_column_names,
            sensitive_fields=sensitive_column_names,
        )

        if baseline_protection == 0:
            score = np.nan
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


class DisclosureProtectionEstimate(DisclosureProtection):
    """DisclosureProtectionEstimate metric."""

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
        num_rows_subsample,
        num_iterations,
    ):
        super()._validate_inputs(
            real_data,
            synthetic_data,
            known_column_names,
            sensitive_column_names,
            computation_method,
            continuous_column_names,
            num_discrete_bins,
        )
        if not isinstance(num_rows_subsample, int) or num_rows_subsample <= 0:
            raise ValueError('`num_rows_subsample` must be an integer greater than zero.')

        if not isinstance(num_iterations, int) or num_iterations <= 0:
            raise ValueError('`num_iterations` must be an integer greater than zero.')

    @classmethod
    def _compute_estimated_cap_metric(
        cls,
        real_data,
        synthetic_data,
        baseline_protection,
        known_column_names,
        sensitive_column_names,
        computation_method,
        num_rows_subsample,
        num_iterations,
        verbose,
    ):
        estimation_iterator = tqdm.tqdm(range(num_iterations), disable=(not verbose))
        if verbose:
            description = 'Estimating Disclosure Protection (Score={score:.3f})'
            estimation_iterator.set_description(description.format(score=0))

        cap_metric = CAP_METHODS.get(computation_method)
        estimated_score_sum = 0
        for i in estimation_iterator:
            real_data_samp = real_data.sample(min(num_rows_subsample, len(real_data)))
            synth_data_samp = synthetic_data.sample(min(num_rows_subsample, len(synthetic_data)))

            estimated_cap_protection = cap_metric._compute(
                real_data_samp,
                synth_data_samp,
                key_fields=known_column_names,
                sensitive_fields=sensitive_column_names,
            )
            estimated_score_sum += estimated_cap_protection
            average_computed_score = estimated_score_sum / (i + 1.0)
            if baseline_protection == 0:
                average_score = np.nan
            else:
                average_score = min(average_computed_score / baseline_protection, 1)

            if verbose:
                estimation_iterator.set_description(description.format(score=average_score))

        return average_score, average_computed_score

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
        num_rows_subsample=1000,
        num_iterations=10,
        verbose=True,
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
            num_rows_subsample (int, optional):
                The number of rows to subsample in each of the real and synthetic datasets per
                iteration. Defaults to 1000 rows.
            num_iterations (int, optional):
                The number of iterations to do for different subsample. Defaults to 10.
            verbose (bool, optional):
                Whether to show the progress bar. Defaults to True.

        Returns:
            dict
                Mapping of the metric output with the keys:
                    - 'score': The overall score for the metric.
                    - 'cap_protection': The protection score from the selected computation method.
                    - 'baseline_protection': The baseline protection for the columns.
        """
        cls._validate_inputs(
            real_data,
            synthetic_data,
            known_column_names,
            sensitive_column_names,
            computation_method,
            continuous_column_names,
            num_discrete_bins,
            num_rows_subsample,
            num_iterations,
        )
        computation_method = computation_method.upper()
        real_data, synthetic_data = cls._discretize_and_fillna(
            real_data,
            synthetic_data,
            known_column_names,
            sensitive_column_names,
            continuous_column_names,
            num_discrete_bins,
        )

        # Compute baseline
        baseline_protection = cls._compute_baseline(real_data, sensitive_column_names)

        # Compute estimated CAP metric
        average_score, average_computed_score = cls._compute_estimated_cap_metric(
            real_data,
            synthetic_data,
            baseline_protection=baseline_protection,
            known_column_names=known_column_names,
            sensitive_column_names=sensitive_column_names,
            computation_method=computation_method,
            num_rows_subsample=num_rows_subsample,
            num_iterations=num_iterations,
            verbose=verbose,
        )

        return {
            'score': average_score,
            'cap_protection': average_computed_score,
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
        num_rows_subsample=1000,
        num_iterations=10,
        verbose=True,
    ):
        """Compute the DisclosureProtectionEstimate metric.

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
            num_rows_subsample (int, optional):
                The number of rows to subsample in each of the real and synthetic datasets per
                iteration. Defaults to 1000 rows.
            num_iterations (int, optional):
                The number of iterations to do for different subsample. Defaults to 10.
            verbose (bool, optional):
                Whether to show the progress bar. Defaults to True.

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
