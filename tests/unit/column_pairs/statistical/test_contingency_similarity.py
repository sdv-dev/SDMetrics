import re
from unittest.mock import patch

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

        # Run and Assert
        expected_bad_data = re.escape('The data must be a pandas DataFrame with two columns.')
        with pytest.raises(ValueError, match=expected_bad_data):
            ContingencySimilarity._validate_inputs(
                real_data=bad_data,
                synthetic_data=bad_data,
                continuous_column_names=None,
                num_discrete_bins=10,
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
            )

    def test_compute(self):
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
        result = metric.compute(real_data, synthetic_data)

        # Assert
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
