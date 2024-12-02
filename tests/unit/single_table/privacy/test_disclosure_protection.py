"""Test for the disclosure protection metrics."""

import re
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table.privacy.disclosure_protection import DisclosureProtection


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

    def test__discreteize_column(self):
        """Test discretizing a continous column"""
        # Setup
        real_column = pd.Series([0, 2, 6, 8, 10])
        synthetic_column = pd.Series([-10, 1, 3, 5, 7, 9, 20])

        # Run
        binned_real, binned_synthetic = DisclosureProtection._discretize_column(
            real_column, synthetic_column, 5
        )

        # Assert
        expected_real = pd.Series(pd.Categorical(['0', '0', '2', '3', '4']))
        pd.testing.assert_series_equal(binned_real, expected_real, check_categorical=False)
        expected_synthetic = pd.Series(pd.Categorical(['0', '0', '1', '2', '3', '4', '4']))
        pd.testing.assert_series_equal(
            binned_synthetic, expected_synthetic, check_categorical=False
        )

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
