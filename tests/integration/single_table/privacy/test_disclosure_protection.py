import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table.privacy.disclosure_protection import DisclosureProtection


@pytest.fixture
def real_data():
    return pd.DataFrame({
        'key1': ['a', 'b', 'c', 'd', 'e'] * 20,
        'key2': range(100),
        'sensitive1': ['a', 'b', 'c', 'd', 'e'] * 20,
        'sensitive2': range(100),
    })


@pytest.fixture
def perfect_synthetic_data():
    random_state = np.random.RandomState(42)

    return pd.DataFrame({
        'key1': random_state.choice(['a', 'b', 'c', 'd', 'e'], 20),
        'key2': range(20),
        'sensitive1': random_state.choice(['f', 'g', 'h', 'i', 'j'], 20),
        'sensitive2': random_state.randint(5, 10, size=20),
    })


@pytest.fixture
def good_synthetic_data():
    random_state = np.random.RandomState(42)
    return pd.DataFrame({
        'key1': random_state.choice(['a', 'b', 'c', 'd', 'e'], 20),
        'key2': random_state.randint(0, 5, size=20),
        'sensitive1': random_state.choice(['a', 'b', 'c', 'd', 'e'], 20),
        'sensitive2': random_state.randint(0, 5, size=20),
    })


@pytest.fixture
def bad_synthetic_data():
    return pd.DataFrame({
        'key1': ['a', 'b', 'c', 'd', 'e'] * 20,
        'key2': range(100),
        'sensitive1': ['a', 'b', 'c', 'e', 'd'] * 20,
        'sensitive2': range(100),
    })


class TestDisclosureProtection:
    def test_end_to_end_perfect(self, real_data, perfect_synthetic_data):
        """Test DisclosureProtection metric end to end with perfect synthetic data."""
        # Setup
        sensitive_columns = ['sensitive1', 'sensitive2']
        known_columns = ['key1', 'key2']
        continous_columns = ['key2', 'sensitive2']

        # Run
        score_breakdown = DisclosureProtection.compute_breakdown(
            real_data,
            perfect_synthetic_data,
            sensitive_column_names=sensitive_columns,
            known_column_names=known_columns,
            continuous_column_names=continous_columns,
            num_discrete_bins=10,
        )

        # Assert
        assert score_breakdown == {'score': 1, 'cap_protection': 1, 'baseline_protection': 0.98}

    def test_end_to_end_good(self, real_data, good_synthetic_data):
        """Test DisclosureProtection metric end to end with good synthetic data."""
        # Setup
        sensitive_columns = ['sensitive1', 'sensitive2']
        known_columns = ['key1', 'key2']
        continuous_columns = ['key2', 'sensitive2']

        # Run
        score_breakdown = DisclosureProtection.compute_breakdown(
            real_data,
            good_synthetic_data,
            sensitive_column_names=sensitive_columns,
            known_column_names=known_columns,
            continuous_column_names=continuous_columns,
            num_discrete_bins=10,
        )

        # Assert
        assert score_breakdown == {
            'score': 0.8979591836734694,
            'cap_protection': 0.88,
            'baseline_protection': 0.98,
        }

    def test_end_to_end_bad(self, real_data, bad_synthetic_data):
        """Test DisclosureProtection metric end to end with bad synthetic data."""
        # Setup
        sensitive_columns = ['sensitive1', 'sensitive2']
        known_columns = ['key1', 'key2']
        continuous_columns = ['key2', 'sensitive2']

        # Run
        score_breakdown = DisclosureProtection.compute_breakdown(
            real_data,
            bad_synthetic_data,
            sensitive_column_names=sensitive_columns,
            known_column_names=known_columns,
            continuous_column_names=continuous_columns,
            num_discrete_bins=10,
        )

        # Assert
        assert score_breakdown == {
            'score': 0.40816326530612246,
            'cap_protection': 0.4,
            'baseline_protection': 0.98,
        }

    @pytest.mark.parametrize('cap_method', ['cap', 'zero_cap', 'generalized_cap'])
    def test_all_cap_methods(self, cap_method, real_data, perfect_synthetic_data):
        """Test DisclosureProtection metric with all possible CAP methods."""
        # Setup
        sensitive_columns = ['sensitive1', 'sensitive2']
        known_columns = ['key1', 'key2']
        continuous_columns = ['key2', 'sensitive2']

        # Run
        score_breakdown = DisclosureProtection.compute_breakdown(
            real_data,
            perfect_synthetic_data,
            sensitive_column_names=sensitive_columns,
            known_column_names=known_columns,
            continuous_column_names=continuous_columns,
            computation_method=cap_method,
            num_discrete_bins=10,
        )

        # Assert
        assert score_breakdown == {
            'score': 1.0,
            'cap_protection': 1.0,
            'baseline_protection': 0.98,
        }
