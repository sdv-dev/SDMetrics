import numpy as np
import pandas as pd
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
