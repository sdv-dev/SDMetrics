import random
import re
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table.privacy import DCROverfittingProtection


@pytest.fixture()
def test_data():
    train_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(50)]})
    holdout_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(50)]})
    synthetic_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(50)]})
    metadata = {'columns': {'num_col': {'sdtype': 'numerical'}}}
    return (train_data, holdout_data, synthetic_data, metadata)


class TestDCROverfittingProtection:
    def test__validate_inputs(self, test_data):
        """Test that we properly validate inputs to our DCROverfittingProtection."""
        # Setup
        train_data, holdout_data, synthetic_data, metadata = test_data

        # Run and Assert
        num_subsample_error_post = re.escape(
            f'must be an integer greater than 1 and less than num of rows in the data ({len(synthetic_data)}).'
        )
        with pytest.raises(ValueError, match=num_subsample_error_post):
            DCROverfittingProtection.compute_breakdown(
                train_data, synthetic_data, holdout_data, metadata, 0
            )

        with pytest.raises(ValueError, match=num_subsample_error_post):
            DCROverfittingProtection.compute_breakdown(
                train_data, synthetic_data, holdout_data, metadata, len(synthetic_data) * 2
            )

        subsample_none_msg = re.escape(
            'num_iterations should not be greater than 1 if there is no subsampling.'
        )
        with pytest.raises(ValueError, match=subsample_none_msg):
            DCROverfittingProtection.compute_breakdown(
                train_data, synthetic_data, holdout_data, metadata, None, 10
            )

        zero_iteration_msg = re.escape('num_iterations (0) must be an integer greater than 1.')
        with pytest.raises(ValueError, match=zero_iteration_msg):
            DCROverfittingProtection.compute_breakdown(
                train_data, synthetic_data, holdout_data, metadata, 1, 0
            )

        no_dcr_metadata = {'columns': {'bad_col': {'sdtype': 'unknown'}}}
        no_dcr_data = pd.DataFrame({'bad_col': [1.0]})

        missing_metric = 'There are no overlapping statistical columns to measure.'
        with pytest.raises(ValueError, match=missing_metric):
            DCROverfittingProtection.compute_breakdown(
                no_dcr_data, no_dcr_data, no_dcr_data, no_dcr_metadata
            )

        small_holdout_data = holdout_data.sample(frac=0.2)
        small_validation_msg = (
            f'Your real_validation_data contains {len(small_holdout_data)} rows while your '
            f'real_training_data contains {len(holdout_data)} rows. For most accurate '
            'results, we recommend that the validation data at least half the size of the training data.'
        )
        with pytest.warns(UserWarning, match=small_validation_msg):
            DCROverfittingProtection.compute_breakdown(
                train_data, synthetic_data, small_holdout_data, metadata
            )

    @patch('numpy.where')
    @patch('sdmetrics.single_table.privacy.dcr_overfitting_protection.calculate_dcr')
    def test_compute_breakdown(self, mock_calculate_dcr, mock_numpy_where, test_data):
        """Test that compute breakdown correctly measures the fraction of data overfitted."""
        # Setup
        train_data, holdout_data, synthetic_data, metadata = test_data
        num_iterations = 2
        num_rows_subsample = 2
        mock_calculate_dcr_array = np.array([0.0] * len(train_data))
        mock_calculate_dcr.return_value = pd.Series(mock_calculate_dcr_array)
        data = np.array([1] * (len(train_data) // 2) + [0] * (len(train_data) // 2))
        mock_numpy_where.return_value = pd.Series(data)

        # Run
        result = DCROverfittingProtection.compute_breakdown(
            train_data, synthetic_data, holdout_data, metadata, num_rows_subsample, num_iterations
        )

        # Assert
        assert mock_calculate_dcr.call_count == 2 * num_iterations
        assert result['score'] == 1.0
        assert result['synthetic_data_percentages']['closer_to_training'] == 0.5
        assert result['synthetic_data_percentages']['closer_to_holdout'] == 0.5

    @patch(
        'sdmetrics.single_table.privacy.dcr_overfitting_protection.DCROverfittingProtection.compute_breakdown'
    )
    def test_compute(self, mock_compute_breakdown, test_data):
        """Test that compute makes a call to compute_breakdown."""
        # Setup
        train_data, holdout_data, synthetic_data, metadata = test_data
        num_iterations = 2
        num_rows_subsample = 2

        # Run
        DCROverfittingProtection.compute(
            train_data, synthetic_data, holdout_data, metadata, num_rows_subsample, num_iterations
        )

        # Assert
        mock_compute_breakdown.assert_called_once_with(
            train_data, synthetic_data, holdout_data, metadata, num_rows_subsample, num_iterations
        )
