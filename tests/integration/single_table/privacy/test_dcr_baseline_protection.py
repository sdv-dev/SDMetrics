import random
import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from sdmetrics.demos import load_single_table_demo
from sdmetrics.single_table.privacy.dcr_baseline_protection import DCRBaselineProtection
from tests.utils import check_if_value_in_threshold


@pytest.fixture()
def training_data():
    return pd.DataFrame({
        'num_col': [0, 0, np.nan, np.nan, 10, 10],
        'cat_col': ['A', 'B', 'A', None, 'B', None],
        'bool_col': [True, False, True, False, None, False],
        'unknown_column': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'datetime_col': [
            datetime(2025, 1, 1),
            datetime(2025, 1, 1),
            datetime(2025, 1, 11),
            datetime(2025, 1, 10),
            pd.NaT,
            datetime(2025, 1, 11),
        ],
    })


@pytest.fixture()
def validation_data():
    return pd.DataFrame({
        'num_col': [10, 0, np.nan, 10, 0, 10],
        'cat_col': [None, 'B', 'A', None, 'B', 'A'],
        'bool_col': [True, True, True, False, False, False],
        'unknown_column': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'datetime_col': [
            datetime(2025, 1, 1),
            datetime(2025, 1, 10),
            datetime(2025, 1, 11),
            datetime(2025, 1, 10),
            pd.NaT,
            pd.NaT,
        ],
    })


@pytest.fixture()
def synthetic_data():
    return pd.DataFrame({
        'num_col': [2, 2, np.nan, np.nan, 8, 8],
        'cat_col': ['B', 'A', 'B', None, 'A', None],
        'bool_col': [False, True, False, True, None, True],
        'unknown_column': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'datetime_col': [
            datetime(2025, 1, 2),
            datetime(2025, 1, 2),
            datetime(2025, 1, 8),
            datetime(2025, 1, 9),
            pd.NaT,
            datetime(2025, 1, 8),
        ],
    })


@pytest.fixture()
def test_metadata():
    return {
        'primary_key': 'student_id',
        'columns': {
            'num_col': {
                'sdtype': 'numerical',
            },
            'cat_col': {
                'sdtype': 'categorical',
            },
            'bool_col': {
                'sdtype': 'boolean',
            },
            'datetime_col': {
                'sdtype': 'datetime',
            },
            'unknown_column': {
                'sdtype': 'invalid',
            },
        },
    }


@pytest.fixture
def expected_scores():
    return {
        'score': 0.958333,
        'median_DCR_to_real_data': {'synthetic_data': 0.2875, 'random_data_baseline': 0.3},
    }


class TestDisclosureProtection:
    def test_end_to_end(
        self, synthetic_data, training_data, validation_data, test_metadata, expected_scores
    ):
        """Test end to end for DCRBaslineProtection metric against a small dataset.

        Synthesized data here should be considered pretty private as the synthetic data maps
        the value of training data to different values. Test that invalid sdtypes do not affect the
        score.
        """
        # Setup
        train_data_diff_unknown_col = training_data.copy()
        train_data_diff_unknown_col['unknown_column'] = [100, 100, 100, 100, 100, 100]

        # Run
        compute_breakdown_result = DCRBaselineProtection.compute_breakdown(
            training_data, synthetic_data, validation_data, test_metadata
        )
        compute_breakdown_diff_result = DCRBaselineProtection.compute_breakdown(
            train_data_diff_unknown_col, synthetic_data, validation_data, test_metadata
        )
        compute_result = DCRBaselineProtection.compute(
            training_data, synthetic_data, validation_data, test_metadata
        )

        # Assert
        median_key = 'median_DCR_to_real_data'
        assert expected_scores[median_key] == compute_breakdown_result[median_key]
        assert compute_breakdown_diff_result == compute_breakdown_diff_result
        check_if_value_in_threshold(expected_scores['score'], compute_result, threshold=0.001)

    def test_end_to_end_with_demo(self):
        """Test end to end for DCRBaslineProtection metric against the demo dataset.

        In this end to end test, test against demo dataset. Use subsampling to speed
        up the test.
        Make sure that if hold two datasets to be the same we get expected v
        alues even with subsampling.
        """
        # Setup
        real_data, synthetic_data, metadata = load_single_table_demo()
        train_df, holdout_df = train_test_split(real_data, test_size=0.2)

        # Run
        num_rows_subsample = 50
        compute_breakdown_result = DCRBaselineProtection.compute_breakdown(
            train_df, synthetic_data, holdout_df, metadata
        )
        compute_result = DCRBaselineProtection.compute(
            train_df, synthetic_data, holdout_df, metadata
        )
        compute_baseline_same = DCRBaselineProtection.compute_breakdown(
            train_df, synthetic_data, synthetic_data, metadata, num_rows_subsample
        )
        compute_train_same = DCRBaselineProtection.compute_breakdown(
            synthetic_data, synthetic_data, holdout_df, metadata, num_rows_subsample
        )
        compute_all_same = DCRBaselineProtection.compute_breakdown(
            synthetic_data,
            synthetic_data,
            synthetic_data,
            metadata,
            num_rows_subsample,
        )

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
        """Testing invalid sdtypes and if there are no columns to measure."""
        train_data = pd.DataFrame({'bad_col': [1.0, 2.0]})
        bad_metadata = {'columns': {'bad_col': {'sdtype': 'unknown'}}}

        error_msg = (
            'There are no valid sdtypes in the dataframes to run the DCRBaselineProtection metric.'
        )
        with pytest.raises(ValueError, match=error_msg):
            DCRBaselineProtection.compute_breakdown(
                train_data, train_data, train_data, bad_metadata
            )

        with pytest.raises(ValueError, match=error_msg):
            DCRBaselineProtection.compute(train_data, train_data, train_data, bad_metadata)

    def test_compute_breakdown_subsampling(self):
        """Test subsampling produces different values."""
        # Setup
        train_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(20)]})
        holdout_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(20)]})
        synthetic_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(20)]})
        metadata = {'columns': {'num_col': {'sdtype': 'numerical'}}}

        error_msg = re.escape('num_rows_subsample (0) must be greater than 1.')
        num_rows_subsample = 4

        # Run
        with pytest.raises(ValueError, match=error_msg):
            DCRBaselineProtection.compute_breakdown(
                train_data, synthetic_data, holdout_data, metadata, 0
            )

        compute_subsample_1 = DCRBaselineProtection.compute_breakdown(
            train_data, synthetic_data, holdout_data, metadata, num_rows_subsample
        )
        compute_subsample_2 = DCRBaselineProtection.compute_breakdown(
            train_data, synthetic_data, holdout_data, metadata, num_rows_subsample
        )

        compute_full_1 = DCRBaselineProtection.compute_breakdown(
            train_data, synthetic_data, holdout_data, metadata
        )
        compute_full_2 = DCRBaselineProtection.compute_breakdown(
            train_data, synthetic_data, holdout_data, metadata
        )

        # Assert that subsampling provides different values.
        assert compute_subsample_1 != compute_subsample_2
        assert compute_full_1 == compute_full_2

    def test_compute_breakdown_iterations(self):
        """Test that number iterations for subsampling affect results."""
        # Setup
        train_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(10)]})
        holdout_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(10)]})
        synthetic_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(10)]})
        metadata = {'columns': {'num_col': {'sdtype': 'numerical'}}}
        zero_error_msg = re.escape('num_iterations (0) must be greater than 1.')
        subsample_none_msg = re.escape(
            'num_iterations should not be greater than 1 if there is not subsampling.'
        )
        num_rows_subsample = 3

        # Run
        with pytest.raises(ValueError, match=zero_error_msg):
            DCRBaselineProtection.compute_breakdown(
                train_data, synthetic_data, holdout_data, metadata, num_rows_subsample, 0
            )

        with pytest.raises(ValueError, match=subsample_none_msg):
            DCRBaselineProtection.compute_breakdown(
                train_data, synthetic_data, holdout_data, metadata, None, 10
            )

        compute_num_iteration_1 = DCRBaselineProtection.compute_breakdown(
            train_data, synthetic_data, holdout_data, metadata, num_rows_subsample, 1
        )
        compute_num_iteration_1000 = DCRBaselineProtection.compute_breakdown(
            train_data, synthetic_data, holdout_data, metadata, num_rows_subsample, 1000
        )
        compute_train_same = DCRBaselineProtection.compute_breakdown(
            synthetic_data, synthetic_data, holdout_data, metadata, num_rows_subsample, 1000
        )

        assert compute_num_iteration_1 != compute_num_iteration_1000
        assert compute_train_same['score'] == 0.0
