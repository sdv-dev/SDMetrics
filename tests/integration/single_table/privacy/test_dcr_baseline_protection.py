import random
import re
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from sdmetrics.demos import load_single_table_demo
from sdmetrics.single_table.privacy.dcr_baseline_protection import DCRBaselineProtection


class TestDisclosureProtection:
    def test_end_to_end_with_demo(self):
        # Setup
        real_data, synthetic_data, metadata = load_single_table_demo()
        train_df, holdout_df = train_test_split(real_data, test_size=0.2)

        # Run
        num_rows_subsample = 50
        compute_breakdown_result = DCRBaselineProtection.compute_breakdown(
            train_df, synthetic_data, holdout_df, metadata)
        compute_result = DCRBaselineProtection.compute(
            train_df, synthetic_data, holdout_df, metadata)
        compute_baseline_same = DCRBaselineProtection.compute_breakdown(
            train_df, synthetic_data, synthetic_data, metadata, num_rows_subsample)
        compute_train_same = DCRBaselineProtection.compute_breakdown(
            synthetic_data, synthetic_data, holdout_df, metadata, num_rows_subsample)
        compute_all_same = DCRBaselineProtection.compute_breakdown(
            synthetic_data, synthetic_data, synthetic_data, metadata, num_rows_subsample)

        median_key = 'median_DCR_to_real_data'
        synth_median_key = 'synthetic_data'
        baseline_key = 'random_data_baseline'
        score_key = 'score'

        # Assert
        assert compute_result == compute_breakdown_result[score_key]
        assert compute_baseline_same[score_key] == 1.0
        assert compute_baseline_same[median_key][baseline_key] == 0.0
        assert compute_train_same[score_key] == 0.0
        assert compute_train_same[median_key][synth_median_key] == 0.0
        assert np.isnan(compute_all_same[score_key])
        assert compute_all_same[median_key][synth_median_key] == 0.0
        assert compute_all_same[median_key][baseline_key] == 0.0

    def test_compute_breakdown_drop_all_columns(self):
        train_data = pd.DataFrame({
            'bad_col': [1.0, 2.0]
        })
        bad_metadata = {
            'columns': {
                'bad_col': {
                    'sdtype': 'unknown'
                }
            }
        }

        error_msg = 'There are no valid sdtypes in the dataframes to run the DCRBaselineProtection metric.'
        with pytest.raises(ValueError, match=error_msg):
            DCRBaselineProtection.compute_breakdown(train_data, train_data, train_data, bad_metadata)

        with pytest.raises(ValueError, match=error_msg):
            DCRBaselineProtection.compute(train_data, train_data, train_data, bad_metadata)

    def test_compute_breakdown_subsampling(self):
        # Setup
        train_data = pd.DataFrame({
            'num_col': [random.randint(1, 1000) for _ in range(20)]
        })
        holdout_data = pd.DataFrame({
            'num_col': [random.randint(1, 1000) for _ in range(20)]
        })
        synthetic_data = pd.DataFrame({
            'num_col': [random.randint(1, 1000) for _ in range(20)]
        })
        metadata = {
            'columns': {
                'num_col': {
                    'sdtype': 'numerical'
                }
            }
        }

        error_msg = re.escape('num_rows_subsample (0) must be greater than 1.')
        with pytest.raises(ValueError, match=error_msg):
            DCRBaselineProtection.compute_breakdown(
                train_data, synthetic_data, holdout_data, metadata, 0)

        # Run
        num_rows_subsample = 4
        compute_subsample_1 = DCRBaselineProtection.compute_breakdown(
            train_data, synthetic_data, holdout_data, metadata, num_rows_subsample)
        compute_subsample_2 = DCRBaselineProtection.compute_breakdown(
            train_data, synthetic_data, holdout_data, metadata, num_rows_subsample)

        compute_full_1 = DCRBaselineProtection.compute_breakdown(
            train_data, synthetic_data, holdout_data, metadata)
        compute_full_2 = DCRBaselineProtection.compute_breakdown(
            train_data, synthetic_data, holdout_data, metadata)

        # Assert that subsampling provides different values.
        assert compute_subsample_1 != compute_subsample_2
        assert compute_full_1 == compute_full_2

    def test_compute_breakdown_iterations(self):
        # Setup
        data = [random.randint(1, 100) for _ in range(20)]
        train_data = pd.DataFrame({
            'num_col': random.sample(data, len(data))
        })
        holdout_data = pd.DataFrame({
            'num_col': random.sample(data, len(data))
        })
        synthetic_data = pd.DataFrame({
            'num_col': random.sample(data, len(data))
        })
        metadata = {
            'columns': {
                'num_col': {
                    'sdtype': 'numerical'
                }
            }
        }

        error_msg = re.escape('num_iterations (0) must be greater than 1.')
        num_rows_subsample = 5
        with pytest.raises(ValueError, match=error_msg):
            DCRBaselineProtection.compute_breakdown(
                train_data, synthetic_data, holdout_data, metadata, num_rows_subsample, 0)

        # Run
        compute_subsample_1 = DCRBaselineProtection.compute_breakdown(
            train_data, synthetic_data, holdout_data, metadata, num_rows_subsample)
        compute_subsample_2 = DCRBaselineProtection.compute_breakdown(
            train_data, synthetic_data, holdout_data, metadata, num_rows_subsample)

        compute_full_1 = DCRBaselineProtection.compute_breakdown(
            train_data, synthetic_data, holdout_data, metadata)
        compute_full_2 = DCRBaselineProtection.compute_breakdown(
            train_data, synthetic_data, holdout_data, metadata)

        # Assert that subsampling provides different values.
        assert compute_subsample_1 != compute_subsample_2
        assert compute_full_1 == compute_full_2
