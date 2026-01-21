import re
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sdmetrics.column_pairs.statistical import ContingencySimilarity


class TestContingencySimilarity:
    def test__validate_inputs(self):
        """Test the ``_validate_inputs`` method."""
        # Setup
        bad_data = pd.Series(range(5))
        real_data = pd.DataFrame({'col1': range(10), 'col2': range(10, 20)})
        bad_synthetic_data = pd.DataFrame({'bad_column': range(10), 'col2': range(10)})
        synthetic_data = pd.DataFrame({'col1': range(5), 'col2': range(5)})
        bad_continous_columns = ['col1', 'missing_col']
        bad_num_discrete_bins = -1
        bad_num_rows_subsample = -1

        # Run and Assert
        ContingencySimilarity._validate_inputs(
            real_data=real_data,
            synthetic_data=synthetic_data,
            continuous_column_names=None,
            num_discrete_bins=10,
            num_rows_subsample=3,
            real_association_threshold=0,
        )
        expected_bad_data = re.escape('The data must be a pandas DataFrame with two columns.')
        with pytest.raises(ValueError, match=expected_bad_data):
            ContingencySimilarity._validate_inputs(
                real_data=bad_data,
                synthetic_data=bad_data,
                continuous_column_names=None,
                num_discrete_bins=10,
                num_rows_subsample=3,
                real_association_threshold=0,
            )

        expected_mismatch_columns_error = re.escape(
            'The columns in the real and synthetic data must match.'
        )
        with pytest.raises(ValueError, match=expected_mismatch_columns_error):
            ContingencySimilarity._validate_inputs(
                real_data=real_data,
                synthetic_data=bad_synthetic_data,
                continuous_column_names=None,
                num_discrete_bins=10,
                num_rows_subsample=3,
                real_association_threshold=0,
            )

        expected_bad_continous_column_error = re.escape(
            "Continuous column(s) 'missing_col' not found in the data."
        )
        with pytest.raises(ValueError, match=expected_bad_continous_column_error):
            ContingencySimilarity._validate_inputs(
                real_data=real_data,
                synthetic_data=synthetic_data,
                continuous_column_names=bad_continous_columns,
                num_discrete_bins=10,
                num_rows_subsample=3,
                real_association_threshold=0,
            )

        expected_bad_num_discrete_bins_error = re.escape(
            '`num_discrete_bins` must be an integer greater than zero.'
        )
        with pytest.raises(ValueError, match=expected_bad_num_discrete_bins_error):
            ContingencySimilarity._validate_inputs(
                real_data=real_data,
                synthetic_data=synthetic_data,
                continuous_column_names=['col1'],
                num_discrete_bins=bad_num_discrete_bins,
                num_rows_subsample=3,
                real_association_threshold=0,
            )
        expected_bad_num_rows_subsample_error = re.escape(
            '`num_rows_subsample` must be an integer greater than zero.'
        )
        with pytest.raises(ValueError, match=expected_bad_num_rows_subsample_error):
            ContingencySimilarity._validate_inputs(
                real_data=real_data,
                synthetic_data=synthetic_data,
                continuous_column_names=['col1'],
                num_discrete_bins=10,
                num_rows_subsample=bad_num_rows_subsample,
                real_association_threshold=0,
            )

        expected_bad_threshold_error = re.escape(
            'real_association_threshold must be a number between 0 and 1.'
        )
        with pytest.raises(ValueError, match=expected_bad_threshold_error):
            ContingencySimilarity._validate_inputs(
                real_data=real_data,
                synthetic_data=synthetic_data,
                continuous_column_names=['col1'],
                num_discrete_bins=10,
                num_rows_subsample=3,
                real_association_threshold=-0.1,
            )

    @patch(
        'sdmetrics.column_pairs.statistical.contingency_similarity.ContingencySimilarity.compute_breakdown'
    )
    def test_compute_mock(self, compute_breakdown_mock):
        """Test that the ``compute`` method calls the ``compute_breakdown`` method."""
        # Setup
        real_data = pd.DataFrame({'col1': [1.0, 2.4, 2.6, 0.8], 'col2': [1, 2, 3, 4]})
        synthetic_data = pd.DataFrame({'col1': [1.0, 1.8, 2.6, 1.0], 'col2': [2, 3, 7, -10]})
        compute_breakdown_mock.return_value = {'score': 0.25}

        # Run
        score = ContingencySimilarity.compute(real_data, synthetic_data)

        # Assert
        compute_breakdown_mock.assert_called_once_with(real_data, synthetic_data, None, 10, None, 0)
        assert score == 0.25

    @patch(
        'sdmetrics.column_pairs.statistical.contingency_similarity.ContingencySimilarity._validate_inputs'
    )
    def test_compute_breakdown(self, validate_inputs_mock):
        """Test the ``compute`` method.

        Expect that the total variation distance of the two contingency matricies
        is computed.

        Input:
        - Real data.
        - Synthetic data.

        Output:
        - The metric result.
        """
        # Setup
        real_data = pd.DataFrame({'col1': [1.0, 2.4, 2.6, 0.8], 'col2': [1, 2, 3, 5]})
        synthetic_data = pd.DataFrame({'col1': [1.0, 1.8, 2.6, 1.0], 'col2': [2, 3, 4, 1]})
        expected_score = 0.25

        # Run
        metric = ContingencySimilarity()
        result = metric.compute_breakdown(real_data, synthetic_data)

        # Assert
        validate_inputs_mock.assert_called_once_with(
            real_data,
            synthetic_data,
            None,
            10,
            None,
            0,
        )
        assert result['score'] == expected_score
        assert np.isnan(result['real_association'])

    @patch('sdmetrics.column_pairs.statistical.contingency_similarity.discretize_column')
    def test_compute_with_num_rows_subsample(self, discretize_column_mock):
        """Test the ``compute`` method with ``num_rows_subsample``."""
        # Setup
        np.random.seed(0)
        real_data = pd.DataFrame({'col1': [1.0, 2.4, 2.6, 0.8], 'col2': [1, 2, 3, 4]})
        synthetic_data = pd.DataFrame({'col1': [1.0, 1.8], 'col2': [2, 3]})
        discretize_column_mock.return_value = (
            pd.DataFrame({'col2': [1, 2, 3]}),
            pd.DataFrame({'col2': [2, 3]}),
        )
        expected_score = 0.0

        # Run
        metric = ContingencySimilarity()
        result = metric.compute(
            real_data,
            synthetic_data,
            continuous_column_names=['col2'],
            num_discrete_bins=4,
            num_rows_subsample=3,
        )

        # Assert
        arg_mock = discretize_column_mock.call_args
        expected_real = pd.Series([3, 4, 2], name='col2', index=[2, 3, 1])
        expected_synthetic = pd.Series([2, 3], name='col2', index=[0, 1])
        pd.testing.assert_series_equal(arg_mock[0][0], expected_real)
        pd.testing.assert_series_equal(arg_mock[0][1], expected_synthetic)
        assert result == expected_score

    def test_compute_with_discretization(self):
        """Test the ``compute`` method with continuous columns."""
        # Setup
        real_data = pd.DataFrame({'col1': [1.0, 2.4, 2.6, 0.8], 'col2': [1, 2, 3, 4]})
        synthetic_data = pd.DataFrame({'col1': [1.0, 1.8, 2.6, 1.0], 'col2': [2, 3, 7, -10]})
        expected_score = 0.25

        # Run
        metric = ContingencySimilarity()
        result = metric.compute(
            real_data, synthetic_data, continuous_column_names=['col2'], num_discrete_bins=4
        )

        # Assert
        assert result == expected_score

    @patch('sdmetrics.column_pairs.statistical.contingency_similarity.ColumnPairsMetric.normalize')
    def test_normalize(self, normalize_mock):
        """Test the ``normalize`` method.

        Expect that the inherited ``normalize`` method is called.

        Input:
        - Raw score

        Output:
        - The output of the inherited ``normalize`` method.
        """
        # Setup
        metric = ContingencySimilarity()
        raw_score = 0.9

        # Run
        result = metric.normalize(raw_score)

        # Assert
        normalize_mock.assert_called_once_with(raw_score)
        assert result == normalize_mock.return_value

    @pytest.mark.filterwarnings('error:.*The values in the array are unorderable.*:RuntimeWarning')
    def test_no_runtime_warning_raised(self):
        """Test that no RuntimeWarning warning is raised when the metric is computed."""
        # Setup
        real_data = pd.DataFrame(data={'A': ['value'] * 4, 'B': ['1', '2', '3', pd.NA]})
        synthetic_data = pd.DataFrame(data={'A': ['value'] * 3, 'B': ['1', '2', pd.NA]})

        # Run and Assert
        ContingencySimilarity.compute(
            real_data=real_data[['A', 'B']], synthetic_data=synthetic_data[['A', 'B']]
        )

    def test_real_association_threshold_returns_nan(self):
        """Test that NaN is returned when real association is below threshold."""
        # Setup
        real_data = pd.DataFrame({
            'col1': np.random.choice(['A', 'B', 'C'], size=100),
            'col2': np.random.choice(['X', 'Y', 'Z'], size=100),
        })
        synthetic_data = pd.DataFrame({
            'col1': np.random.choice(['A', 'B', 'C'], size=100),
            'col2': np.random.choice(['X', 'Y', 'Z'], size=100),
        })

        # Run
        result = ContingencySimilarity.compute(
            real_data=real_data,
            synthetic_data=synthetic_data,
            real_association_threshold=0.3,
        )

        # Assert
        assert np.isnan(result)

    def test_real_association_threshold_computes_normally(self):
        """Test that metric computes normally when real association exceeds threshold."""
        # Setup
        real_data = pd.DataFrame({
            'col1': ['A'] * 50 + ['B'] * 50,
            'col2': ['X'] * 48 + ['Y'] * 2 + ['Y'] * 48 + ['X'] * 2,
        })
        synthetic_data = pd.DataFrame({
            'col1': ['A'] * 50 + ['B'] * 50,
            'col2': ['X'] * 45 + ['Y'] * 5 + ['Y'] * 45 + ['X'] * 5,
        })

        # Run
        result = ContingencySimilarity.compute(
            real_data=real_data,
            synthetic_data=synthetic_data,
            real_association_threshold=0.3,
        )

        # Assert
        assert 0 <= result <= 1
