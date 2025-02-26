import random
import re

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from sdmetrics.demos import load_single_table_demo
from sdmetrics.single_table.privacy import DCROverfittingProtection


class TestDCROverfittingProtection:
    def test_end_to_end_with_demo(self):
        """Test end to end for DCROverfittingProtection metric against the demo dataset.

        In this end to end test, test against demo dataset. Use subsampling to speed
        up the test. Make sure that if hold two datasets to be the same we get expected
        values even with subsampling. Note that if synthetic data is equally distant from
        the training data and the holdout data, it is labeled as closer to holdout data.
        """
        # Setup
        real_data, synthetic_data, metadata = load_single_table_demo()
        train_df, holdout_df = train_test_split(real_data, test_size=0.2)

        # Run
        num_rows_subsample = 50
        compute_breakdown_result = DCROverfittingProtection.compute_breakdown(
            train_df, synthetic_data, holdout_df, metadata
        )
        compute_result = DCROverfittingProtection.compute(
            train_df, synthetic_data, holdout_df, metadata
        )
        compute_holdout_same = DCROverfittingProtection.compute_breakdown(
            train_df, synthetic_data, synthetic_data, metadata, num_rows_subsample
        )
        compute_train_same = DCROverfittingProtection.compute_breakdown(
            synthetic_data, synthetic_data, holdout_df, metadata, num_rows_subsample
        )
        compute_all_same = DCROverfittingProtection.compute_breakdown(
            synthetic_data,
            synthetic_data,
            synthetic_data,
            metadata,
            num_rows_subsample,
        )

        synth_percentages_key = 'synthetic_data_percentages'
        synth_train_key = 'closer_to_training'
        synth_holdout_key = 'closer_to_holdout'
        score_key = 'score'

        # Assert
        assert compute_result == compute_breakdown_result[score_key]
        assert compute_holdout_same[score_key] == 1.0
        assert compute_holdout_same[synth_percentages_key][synth_train_key] == 0.0
        assert compute_holdout_same[synth_percentages_key][synth_holdout_key] == 1.0
        assert compute_train_same[score_key] == 0.0
        assert compute_train_same[synth_percentages_key][synth_train_key] == 1.0
        assert compute_train_same[synth_percentages_key][synth_holdout_key] == 0.0
        assert compute_all_same[score_key] == 1.0
        assert compute_all_same[synth_percentages_key][synth_train_key] == 0.0
        assert compute_all_same[synth_percentages_key][synth_holdout_key] == 1.0

    def test_compute_breakdown_drop_all_columns(self):
        """Testing invalid sdtypes and ensure only appropriate columns are measured."""
        # Setup
        train_data = pd.DataFrame({'bad_col': [10.0, 15.0], 'num_col': [1.0, 2.0]})
        synth_data = pd.DataFrame({'bad_col': [2.0, 1.0], 'num_col': [1.0, 2.0]})
        holdout_data = pd.DataFrame({'bad_col': [2.0, 1.0], 'num_col': [3.0, 4.0]})
        metadata = {
            'columns': {
                'bad_col': {'sdtype': 'unknown'},
                'num_col': {'sdtype': 'numerical'},
            }
        }

        # Run
        result = DCROverfittingProtection.compute_breakdown(
            train_data, synth_data, holdout_data, metadata
        )

        # Assert
        assert result['score'] == 0.0
        assert result['synthetic_data_percentages']['closer_to_training'] == 1.0
        assert result['synthetic_data_percentages']['closer_to_holdout'] == 0.0

    def test_compute_breakdown_subsampling(self):
        """Test subsampling produces different values."""
        # Setup
        train_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(50)]})
        holdout_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(50)]})
        synthetic_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(50)]})
        metadata = {'columns': {'num_col': {'sdtype': 'numerical'}}}
        num_rows_subsample = 4
        large_num_subsample = len(synthetic_data) * 2

        # Run
        compute_subsample = DCROverfittingProtection.compute_breakdown(
            train_data, synthetic_data, holdout_data, metadata, num_rows_subsample
        )

        large_subsample_msg = re.escape('Ignoring the num_rows_subsample and num_iterations args.')
        with pytest.warns(UserWarning, match=large_subsample_msg):
            compute_large_subsample = DCROverfittingProtection.compute_breakdown(
                train_data, synthetic_data, holdout_data, metadata, large_num_subsample
            )

        compute_full_1 = DCROverfittingProtection.compute_breakdown(
            train_data, synthetic_data, holdout_data, metadata
        )
        compute_full_2 = DCROverfittingProtection.compute_breakdown(
            train_data, synthetic_data, holdout_data, metadata
        )

        # Assert that subsampling provides different values if smaller than data length.
        assert compute_subsample != compute_full_1
        assert compute_full_1 == compute_full_2
        assert compute_large_subsample == compute_full_1

    def test_compute_breakdown_iterations(self):
        """Test that number iterations for subsampling works as expected."""
        # Setup
        train_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(10)]})
        holdout_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(10)]})
        synthetic_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(10)]})
        metadata = {'columns': {'num_col': {'sdtype': 'numerical'}}}
        num_rows_subsample = 3
        num_iterations = 1000

        # Run
        compute_num_iteration_1 = DCROverfittingProtection.compute_breakdown(
            train_data, synthetic_data, holdout_data, metadata, num_rows_subsample, 1
        )
        compute_num_iteration_1000 = DCROverfittingProtection.compute_breakdown(
            train_data, synthetic_data, holdout_data, metadata, num_rows_subsample, num_iterations
        )
        compute_train_same = DCROverfittingProtection.compute_breakdown(
            synthetic_data,
            synthetic_data,
            holdout_data,
            metadata,
            num_rows_subsample,
            num_iterations,
        )

        # Assert
        assert compute_num_iteration_1 != compute_num_iteration_1000
        assert compute_train_same['score'] == 0.0
