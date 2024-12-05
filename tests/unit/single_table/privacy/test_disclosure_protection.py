"""Test for the disclosure metrics."""

import re
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table.privacy.disclosure_protection import (
    DisclosureProtection,
    DisclosureProtectionEstimate,
)
from tests.utils import DataFrameMatcher


class TestDisclosureProtection:
    def test__validate_inputs(self):
        """Test input validation."""
        # Setup
        default_kwargs = {
            'real_data': pd.DataFrame({'col1': range(5), 'col2': range(5)}),
            'synthetic_data': pd.DataFrame({'col1': range(10), 'col2': range(10)}),
            'known_column_names': ['col1'],
            'sensitive_column_names': ['col2'],
            'computation_method': 'cap',
            'continuous_column_names': ['col2'],
            'num_discrete_bins': 10,
        }
        bad_data = ['a', 'b', 'c']
        missing_known_column = ['col1', 'missing_col']
        missing_sensitive_column = ['col2', 'missing_col']
        bad_computation = 'unknown computation'
        missing_continous_column = ['missing_col']
        bad_bin = '10'

        # Run and Assert
        DisclosureProtection._validate_inputs(**default_kwargs)

        bad_data_error = re.escape('Real and synthetic data must be pandas DataFrames.')
        with pytest.raises(ValueError, match=bad_data_error):
            DisclosureProtection._validate_inputs(**{**default_kwargs, 'real_data': bad_data})

        empty_known_error = re.escape('Must provide at least 1 known column name.')
        with pytest.raises(ValueError, match=empty_known_error):
            DisclosureProtection._validate_inputs(**{**default_kwargs, 'known_column_names': []})

        missing_known_error = re.escape(
            "Known column(s) 'missing_col' are missing from the real data."
        )
        with pytest.raises(ValueError, match=missing_known_error):
            DisclosureProtection._validate_inputs(**{
                **default_kwargs,
                'known_column_names': missing_known_column,
            })

        empty_sensitive_error = re.escape('Must provide at least 1 sensitive column name.')
        with pytest.raises(ValueError, match=empty_sensitive_error):
            DisclosureProtection._validate_inputs(**{
                **default_kwargs,
                'sensitive_column_names': [],
            })

        missing_sensitive_error = re.escape(
            "Sensitive column(s) 'missing_col' are missing from the real data."
        )
        with pytest.raises(ValueError, match=missing_sensitive_error):
            DisclosureProtection._validate_inputs(**{
                **default_kwargs,
                'sensitive_column_names': missing_sensitive_column,
            })

        bad_computation_error = re.escape(
            "Unknown computation method 'unknown computation'. "
            "Please use one of 'cap', 'zero_cap', or 'generalized_cap'."
        )
        with pytest.raises(ValueError, match=bad_computation_error):
            DisclosureProtection._validate_inputs(**{
                **default_kwargs,
                'computation_method': bad_computation,
            })

        missing_continous_error = re.escape(
            "Continous column(s) 'missing_col' are missing from the real data."
        )
        with pytest.raises(ValueError, match=missing_continous_error):
            DisclosureProtection._validate_inputs(**{
                **default_kwargs,
                'continuous_column_names': missing_continous_column,
            })

        bad_bin_error = re.escape('`num_discrete_bins` must be an integer greater than zero.')
        with pytest.raises(ValueError, match=bad_bin_error):
            DisclosureProtection._validate_inputs(**{
                **default_kwargs,
                'num_discrete_bins': bad_bin,
            })

    def test__get_null_categories(self):
        """Test the method creates a unique category for null values in each column."""
        # Setup
        real_data = pd.DataFrame({
            'col1': ['A', 'B', 'C'],
            'col2': ['__NULL_VALUE__', np.nan, 'A'],
            'col3': [np.nan, None, pd.NA],
        })
        synthetic_data = pd.DataFrame({
            'col1': ['A', 'B', '__NULL_VALUE__'],
            'col2': ['D', np.nan, 'A'],
            'col3': [np.nan, None, pd.NA],
        })
        column_names = ['col1', 'col2']

        # Run
        null_category_map = DisclosureProtection._get_null_categories(
            real_data, synthetic_data, column_names
        )

        # Assert
        assert null_category_map == {'col1': '__NULL_VALUE___', 'col2': '__NULL_VALUE___'}

    @pytest.mark.parametrize('dtype', ['int32', 'int64', 'Int32', 'Int64'])
    def test__discretize_column_int_dtypes(self, dtype):
        """Test discretizing a continous column"""
        # Setup
        real_column = pd.Series([0, 2, 6, 8, 10], dtype=dtype)
        synthetic_column = pd.Series([-10, 1, 3, 5, 7, 9, 20], dtype=dtype)

        # Run
        binned_real, binned_synthetic = DisclosureProtection._discretize_column(
            real_column, synthetic_column, 5
        )

        # Assert
        expected_real = pd.Series(pd.Categorical(['0', '0', '2', '3', '4']))
        np.testing.assert_array_equal(binned_real, expected_real)
        expected_synthetic = pd.Series(pd.Categorical(['0', '0', '1', '2', '3', '4', '4']))
        np.testing.assert_array_equal(binned_synthetic, expected_synthetic)

    @pytest.mark.parametrize('dtype', ['float32', 'float64', 'Float32', 'Float64'])
    def test__discretize_column_float_dtypes(self, dtype):
        """Test discretizing a continous column"""
        # Setup
        real_column = pd.Series([0, 0.2, 6.99, np.nan, 10.02], dtype=dtype)
        synthetic_column = pd.Series([-10.0, 0.1, 3.77, np.nan, 7.89, np.nan, 20.99], dtype=dtype)

        # Run
        binned_real, binned_synthetic = DisclosureProtection._discretize_column(
            real_column, synthetic_column, 5
        )

        # Assert
        expected_real = np.array(['0', '0', '3', np.nan, '4'], dtype='object')
        assert list(binned_real) == list(expected_real)
        expected_synthetic = np.array(['0', '0', '1', np.nan, '3', np.nan, '4'], dtype='object')
        assert list(binned_synthetic) == list(expected_synthetic)

    def test__discretize_and_fillna(self):
        """Test helper method to discretize continous columns and fill nan values."""
        # Setup
        real_data = pd.DataFrame({
            'known': ['A', 'A', pd.NA, 'B', 'B'],
            'continous': [0, 1, 3, 8, 10],
            'continous_nan': [0, 7, 2, np.nan, 10],
            'extra': [None, pd.NA, 0, 10, 100],
        })
        synthetic_data = pd.DataFrame({
            'known': ['A', 'A', 'B', 'B', None],
            'continous': [-1, 0, 3, 5, 11],
            'continous_nan': [0, 1, 2, np.nan, 100],
            'extra': [None, pd.NA, 0, 10, 100],
        })
        known_column_names = ['known']
        sensitive_column_names = ['continous', 'continous_nan']
        continuous_column_names = ['continous', 'continous_nan']
        num_discrete_bins = 5

        # Run
        processed_real, processed_synthetic = DisclosureProtection._discretize_and_fillna(
            real_data,
            synthetic_data,
            known_column_names,
            sensitive_column_names,
            continuous_column_names,
            num_discrete_bins,
        )

        # Assert
        expected_real = pd.DataFrame({
            'known': ['A', 'A', '__NULL_VALUE__', 'B', 'B'],
            'continous': ['0', '0', '1', '3', '4'],
            'continous_nan': ['0', '3', '0', '__NULL_VALUE__', '4'],
            'extra': real_data['extra'],
        })
        expected_synthetic = pd.DataFrame({
            'known': ['A', 'A', 'B', 'B', '__NULL_VALUE__'],
            'continous': ['0', '0', '1', '2', '4'],
            'continous_nan': ['0', '0', '0', '__NULL_VALUE__', '4'],
            'extra': synthetic_data['extra'],
        })
        pd.testing.assert_frame_equal(expected_real, processed_real)
        pd.testing.assert_frame_equal(expected_synthetic, processed_synthetic)

    def test__compute_baseline(self):
        """Test computing the baseline score for random data."""
        # Setup
        real_data = pd.DataFrame({
            'col1': ['A', 'A', 'A', 'A', 'A'],
            'col2': ['A', 'B', 'A', 'B', 'A'],
            'col3': range(5),
        })
        sensitive_column_names = ['col1', 'col2']

        # Run
        baseline_score = DisclosureProtection._compute_baseline(real_data, sensitive_column_names)

        # Assert
        assert baseline_score == 0.5

    @patch('sdmetrics.single_table.privacy.disclosure_protection.CAP_METHODS')
    def test_compute_breakdown(self, CAPMethodsMock):
        """Test the ``compute_breakdown`` method."""
        # Setup
        real_data = pd.DataFrame({
            'col1': np.random.choice(['A', 'B', 'C', 'D'], size=10),
            'col2': ['X', 'Y', 'Z', None, np.nan, 'X', 'Y', 'Z', 'X', 'X'],
            'col3': range(10),
        })
        synthetic_data = pd.DataFrame({
            'col1': np.random.choice(['A', 'B', 'C', 'D'], size=10),
            'col2': np.random.choice(['X', 'Y', 'Z', np.nan, None], size=10),
            'col3': range(-2, 8),
        })
        CAPMock = Mock()
        CAPMock.compute.return_value = 0.9
        CAPMethodsMock.keys.return_value = ['CAP', 'ZERO_CAP', 'GENERALIZED_CAP']
        CAPMethodsMock.get.return_value = CAPMock

        # Run
        score_breakdown = DisclosureProtection.compute_breakdown(
            real_data=real_data,
            synthetic_data=synthetic_data,
            known_column_names=['col1'],
            sensitive_column_names=['col2', 'col3'],
            continuous_column_names=['col3'],
            num_discrete_bins=2,
        )

        # Assert
        assert score_breakdown == {
            'score': 1,
            'baseline_protection': 1 - (1.0 / 8.0),
            'cap_protection': 0.9,
        }

    @patch('sdmetrics.single_table.privacy.disclosure_protection.CAP_METHODS')
    def test_compute_breakdown_zero_baseline(self, CAPMethodsMock):
        """Test the ``compute_breakdown`` method when baseline score is 0."""
        # Setup
        real_data = pd.DataFrame({
            'col1': np.random.choice(['A', 'B', 'C', 'D'], size=10),
            'col2': ['A'] * 10,
        })
        synthetic_data = pd.DataFrame({
            'col1': np.random.choice(['A', 'B', 'C', 'D'], size=10),
            'col2': ['A'] * 10,
        })
        CAPMock = Mock()
        CAPMock.compute.return_value = 0.5
        CAPMethodsMock.keys.return_value = ['CAP', 'ZERO_CAP', 'GENERALIZED_CAP']
        CAPMethodsMock.get.return_value = CAPMock

        # Run
        score_breakdown_with_cap = DisclosureProtection.compute_breakdown(
            real_data=real_data,
            synthetic_data=synthetic_data,
            known_column_names=['col1'],
            sensitive_column_names=['col2'],
        )

        CAPMock.compute.return_value = 0
        score_breakdown_no_cap = DisclosureProtection.compute_breakdown(
            real_data=real_data,
            synthetic_data=synthetic_data,
            known_column_names=['col1'],
            sensitive_column_names=['col2'],
        )

        # Assert
        assert score_breakdown_with_cap == {
            'score': 1,
            'baseline_protection': 0,
            'cap_protection': 0.5,
        }
        assert score_breakdown_no_cap == {'score': 0, 'baseline_protection': 0, 'cap_protection': 0}

    @patch(
        'sdmetrics.single_table.privacy.disclosure_protection.DisclosureProtection.compute_breakdown'
    )
    def test_compute(self, compute_breakdown_mock):
        """Test the ``compute`` method."""
        # Setup
        real_data = pd.DataFrame({
            'col1': np.random.choice(['A', 'B', 'C', 'D'], size=10),
            'col2': ['A'] * 10,
        })
        synthetic_data = pd.DataFrame({
            'col1': np.random.choice(['A', 'B', 'C', 'D'], size=10),
            'col2': ['A'] * 10,
        })
        compute_breakdown_mock.return_value = {
            'score': 0.8,
            'baseline_protection': 0.6,
            'cap_protection': 0.64,
        }

        # Run
        score = DisclosureProtection.compute(
            real_data, synthetic_data, known_column_names=['col1'], sensitive_column_names=['col2']
        )

        # Assert
        assert score == 0.8


class TestDisclosureProtectionEstimate:
    def test__validate_inputs(self):
        """Test input validation."""
        # Setup
        default_kwargs = {
            'real_data': pd.DataFrame({'col1': range(5), 'col2': range(5)}),
            'synthetic_data': pd.DataFrame({'col1': range(10), 'col2': range(10)}),
            'known_column_names': ['col1'],
            'sensitive_column_names': ['col2'],
            'computation_method': 'cap',
            'continuous_column_names': ['col2'],
            'num_discrete_bins': 10,
            'num_rows_subsample': 1000,
            'num_iterations': 10,
        }
        bad_rows_subsample = 0
        bad_num_iterations = 0

        # Run and Assert
        DisclosureProtectionEstimate._validate_inputs(**default_kwargs)

        bad_rows_subsample_error = re.escape(
            '`num_rows_subsample` must be an integer greater than zero.'
        )
        with pytest.raises(ValueError, match=bad_rows_subsample_error):
            DisclosureProtectionEstimate._validate_inputs(**{
                **default_kwargs,
                'num_rows_subsample': bad_rows_subsample,
            })

        bad_num_iterations_error = re.escape(
            '`num_iterations` must be an integer greater than zero.'
        )
        with pytest.raises(ValueError, match=bad_num_iterations_error):
            DisclosureProtectionEstimate._validate_inputs(**{
                **default_kwargs,
                'num_iterations': bad_num_iterations,
            })

    @patch('sdmetrics.single_table.privacy.disclosure_protection.tqdm')
    @patch('sdmetrics.single_table.privacy.disclosure_protection.CAP_METHODS')
    def test__compute_estimated_cap_metric(self, CAPMethodsMock, mock_tqdm):
        """Test the ``_compute_estimated_cap_metric`` method."""
        # Setup
        real_data = pd.DataFrame({
            'col1': np.random.choice(['A', 'B', 'C', 'D'], size=5),
            'col2': np.random.choice(['X', 'Y'], size=5),
        })
        synthetic_data = pd.DataFrame({
            'col1': np.random.choice(['A', 'B', 'C', 'D'], size=100),
            'col2': np.random.choice(['X', 'Y'], size=100),
        })
        CAPMock = Mock()
        CAPMock.compute.side_effect = [0.4, 0.5, 0.2, 0.6, 0.2]
        CAPMethodsMock.keys.return_value = ['CAP', 'ZERO_CAP', 'GENERALIZED_CAP']
        CAPMethodsMock.get.return_value = CAPMock
        progress_bar = MagicMock()
        progress_bar.__iter__.return_value = range(5)
        mock_tqdm.tqdm.return_value = progress_bar

        # Run
        avg_score, avg_computed_score = DisclosureProtectionEstimate._compute_estimated_cap_metric(
            real_data,
            synthetic_data,
            baseline_protection=0.5,
            known_column_names=['col1'],
            sensitive_column_names=['col2'],
            computation_method='CAP',
            num_rows_subsample=10,
            num_iterations=5,
            verbose=True,
        )

        # Assert
        assert avg_score == 0.76
        assert avg_computed_score == 0.38
        progress_bar.set_description.assert_has_calls([
            call('Estimating Disclosure Protection (Score=0.000)'),
            call('Estimating Disclosure Protection (Score=0.800)'),
            call('Estimating Disclosure Protection (Score=0.900)'),
            call('Estimating Disclosure Protection (Score=0.733)'),
            call('Estimating Disclosure Protection (Score=0.850)'),
            call('Estimating Disclosure Protection (Score=0.760)'),
        ])

    @patch('sdmetrics.single_table.privacy.disclosure_protection.CAP_METHODS')
    def test__compute_estimated_cap_metric_zero_baseline(self, CAPMethodsMock):
        """Test the ``_compute_estimated_cap_metric`` method with a zero baseline."""
        # Setup
        real_data = pd.DataFrame({
            'col1': np.random.choice(['A', 'B', 'C', 'D'], size=5),
            'col2': ['A'] * 5,
        })
        synthetic_data = pd.DataFrame({
            'col1': np.random.choice(['A', 'B', 'C', 'D'], size=100),
            'col2': ['A'] * 100,
        })
        CAPMock = Mock()
        CAPMock.compute.side_effect = [0.4, 0.5, 0.2, 0.6, 0.2]
        CAPMethodsMock.keys.return_value = ['CAP', 'ZERO_CAP', 'GENERALIZED_CAP']
        CAPMethodsMock.get.return_value = CAPMock

        # Run
        avg_score, avg_computed_score = DisclosureProtectionEstimate._compute_estimated_cap_metric(
            real_data,
            synthetic_data,
            baseline_protection=0,
            known_column_names=['col1'],
            sensitive_column_names=['col2'],
            computation_method='CAP',
            num_rows_subsample=10,
            num_iterations=5,
            verbose=False,
        )

        # Assert
        assert avg_score == 1
        assert avg_computed_score == 0.38

    @patch(
        'sdmetrics.single_table.privacy.disclosure_protection.DisclosureProtectionEstimate._compute_estimated_cap_metric'
    )
    def test_compute_breakdown(self, mock__compute_estimated_cap_metric):
        """Test computing the breakdown."""
        # Setup
        real_data = pd.DataFrame({
            'col1': np.random.choice(['A', 'B', 'C', 'D'], size=10),
            'col2': ['X', 'Y', 'Z', 'Y', 'X', 'X', 'Y', 'Z', 'X', 'A'],
            'col3': ['A', 'B'] * 5,
        })
        synthetic_data = pd.DataFrame({
            'col1': np.random.choice(['A', 'B', 'C', 'D'], size=10),
            'col2': np.random.choice(['X', 'Y', 'Z', 'X', 'X'], size=10),
            'col3': ['A'] * 10,
        })
        mock__compute_estimated_cap_metric.return_value = (0.8, 0.6)

        # Run
        score_breakdown = DisclosureProtectionEstimate.compute_breakdown(
            real_data=real_data,
            synthetic_data=synthetic_data,
            known_column_names=['col1'],
            sensitive_column_names=['col2', 'col3'],
            num_discrete_bins=2,
        )

        # Assert
        assert score_breakdown == {
            'score': 0.8,
            'baseline_protection': 0.875,
            'cap_protection': 0.6,
        }
        mock__compute_estimated_cap_metric.assert_called_once_with(
            DataFrameMatcher(real_data),
            DataFrameMatcher(synthetic_data),
            baseline_protection=0.875,
            known_column_names=['col1'],
            sensitive_column_names=['col2', 'col3'],
            computation_method='CAP',
            num_rows_subsample=1000,
            num_iterations=10,
            verbose=True,
        )

    @patch(
        'sdmetrics.single_table.privacy.disclosure_protection.DisclosureProtectionEstimate.compute_breakdown'
    )
    def test_compute(self, compute_breakdown_mock):
        """Test the ``compute`` method."""
        # Setup
        real_data = pd.DataFrame({
            'col1': np.random.choice(['A', 'B', 'C', 'D'], size=10),
            'col2': ['A'] * 10,
        })
        synthetic_data = pd.DataFrame({
            'col1': np.random.choice(['A', 'B', 'C', 'D'], size=10),
            'col2': ['A'] * 10,
        })
        compute_breakdown_mock.return_value = {
            'score': 0.8,
            'baseline_protection': 0.6,
            'cap_protection': 0.64,
        }

        # Run
        score = DisclosureProtectionEstimate.compute(
            real_data, synthetic_data, known_column_names=['col1'], sensitive_column_names=['col2']
        )

        # Assert
        assert score == 0.8
