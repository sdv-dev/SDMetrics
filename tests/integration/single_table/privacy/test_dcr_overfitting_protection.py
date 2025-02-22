import numpy as np
import pandas as pd
import pytest

from sklearn.model_selection import train_test_split
from sdmetrics.demos import load_single_table_demo
from sdmetrics.single_table.privacy.dcr_overfitting_protection import DCROverfittingProtection


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
        """Testing invalid sdtypes and if there are no columns to measure."""
        train_data = pd.DataFrame({'bad_col': [1.0, 2.0]})
        bad_metadata = {'columns': {'bad_col': {'sdtype': 'unknown'}}}

        error_msg = (
            'There are no valid sdtypes in the dataframes to run the DCRBaselineProtection metric.'
        )
        with pytest.raises(ValueError, match=error_msg):
            DCROverfittingProtection.compute_breakdown(
                train_data, train_data, train_data, bad_metadata
            )

        with pytest.raises(ValueError, match=error_msg):
            DCROverfittingProtection.compute(train_data, train_data, train_data, bad_metadata)

    # def test_compute_breakdown_subsampling(self):
    #     """Test subsampling produces different values."""
    #     # Setup
    #     train_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(20)]})
    #     holdout_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(20)]})
    #     synthetic_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(20)]})
    #     metadata = {'columns': {'num_col': {'sdtype': 'numerical'}}}

    #     error_msg = re.escape('num_rows_subsample (0) must be greater than 1.')
    #     num_rows_subsample = 4

    #     # Run
    #     with pytest.raises(ValueError, match=error_msg):
    #         DCRBaselineProtection.compute_breakdown(
    #             train_data, synthetic_data, holdout_data, metadata, 0
    #         )

    #     compute_subsample_1 = DCRBaselineProtection.compute_breakdown(
    #         train_data, synthetic_data, holdout_data, metadata, num_rows_subsample
    #     )
    #     compute_subsample_2 = DCRBaselineProtection.compute_breakdown(
    #         train_data, synthetic_data, holdout_data, metadata, num_rows_subsample
    #     )

    #     compute_full_1 = DCRBaselineProtection.compute_breakdown(
    #         train_data, synthetic_data, holdout_data, metadata
    #     )
    #     compute_full_2 = DCRBaselineProtection.compute_breakdown(
    #         train_data, synthetic_data, holdout_data, metadata
    #     )

    #     # Assert that subsampling provides different values.
    #     assert compute_subsample_1 != compute_subsample_2
    #     assert compute_full_1 == compute_full_2

    # def test_compute_breakdown_iterations(self):
    #     """Test that number iterations for subsampling affect results."""
    #     # Setup
    #     train_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(10)]})
    #     holdout_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(10)]})
    #     synthetic_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(10)]})
    #     metadata = {'columns': {'num_col': {'sdtype': 'numerical'}}}
    #     zero_error_msg = re.escape('num_iterations (0) must be greater than 1.')
    #     subsample_none_msg = re.escape(
    #         'num_iterations should not be greater than 1 if there is not subsampling.'
    #     )
    #     num_rows_subsample = 3

    #     # Run
    #     with pytest.raises(ValueError, match=zero_error_msg):
    #         DCRBaselineProtection.compute_breakdown(
    #             train_data, synthetic_data, holdout_data, metadata, num_rows_subsample, 0
    #         )

    #     with pytest.raises(ValueError, match=subsample_none_msg):
    #         DCRBaselineProtection.compute_breakdown(
    #             train_data, synthetic_data, holdout_data, metadata, None, 10
    #         )

    #     compute_num_iteration_1 = DCRBaselineProtection.compute_breakdown(
    #         train_data, synthetic_data, holdout_data, metadata, num_rows_subsample, 1
    #     )
    #     compute_num_iteration_1000 = DCRBaselineProtection.compute_breakdown(
    #         train_data, synthetic_data, holdout_data, metadata, num_rows_subsample, 1000
    #     )
    #     compute_train_same = DCRBaselineProtection.compute_breakdown(
    #         synthetic_data, synthetic_data, holdout_data, metadata, num_rows_subsample, 1000
    #     )

    #     assert compute_num_iteration_1 != compute_num_iteration_1000
    #     assert compute_train_same['score'] == 0.0
