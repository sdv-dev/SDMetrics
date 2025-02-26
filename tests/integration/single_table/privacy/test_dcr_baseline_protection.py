import random
import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

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


class TestDCRBaselineProtection:
    def test_end_to_end_with_demo(self):
        """Test end to end for DCRBaslineProtection metric against the demo dataset.

        In this end to end test, test against demo dataset. Use subsampling to speed
        up the test. Make sure that if hold two datasets to be the same we get expected
        values even with subsampling.
        """
        # Setup
        real_data, synthetic_data, metadata = load_single_table_demo()

        # Run
        compute_breakdown_result = DCRBaselineProtection.compute_breakdown(
            real_data, synthetic_data, metadata
        )
        compute_same_data = DCRBaselineProtection.compute_breakdown(
            synthetic_data, synthetic_data, metadata
        )

        median_key = 'median_DCR_to_real_data'
        synth_median_key = 'synthetic_data'
        baseline_key = 'random_data_baseline'
        score_key = 'score'

        # Assert
        assert compute_same_data[median_key][synth_median_key] == 0.0
        assert compute_same_data[median_key][baseline_key] > 0.0
        assert compute_same_data[score_key] == 0.0
        assert compute_breakdown_result[score_key] > compute_same_data[score_key]

    def test_compute_breakdown_drop_all_columns(self):
        """Testing invalid sdtypes and ensure only appropriate columns are measured."""
        real_data = pd.DataFrame({'diff_col_1': [10.0, 15.0], 'num_col': [1.0, 2.0]})
        synth_data = pd.DataFrame({'diff_col_2': [2.0, 1.0], 'num_col': [1.0, 2.0]})
        metadata = {
            'columns': {
                'diff_col': {'sdtype': 'unknown'},
                'num_col': {'sdtype': 'numerical'},
            }
        }

        result = DCRBaselineProtection.compute_breakdown(
            real_data, synth_data, metadata
        )
        assert result['score'] == 0.0
        assert result['median_DCR_to_real_data']['random_data_baseline'] > 0

    def test_compute_breakdown_subsampling(self):
        """Test subsampling produces different values."""
        # Setup
        real_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(20)]})
        synthetic_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(20)]})
        metadata = {'columns': {'num_col': {'sdtype': 'numerical'}}}

        num_rows_subsample = 4

        # Run
        compute_subsample = DCRBaselineProtection.compute_breakdown(
            real_data, synthetic_data, metadata, num_rows_subsample
        )
        compute_full_1 = DCRBaselineProtection.compute_breakdown(
            real_data, synthetic_data, metadata
        )
        compute_full_2 = DCRBaselineProtection.compute_breakdown(
            real_data, synthetic_data, metadata
        )

        # Assert that subsampling provides different values.
        assert (
            compute_subsample['median_DCR_to_real_data']['synthetic_data'] !=
            compute_full_1['median_DCR_to_real_data']['synthetic_data']
        )
        assert (
            compute_full_1['median_DCR_to_real_data']['synthetic_data'] ==
            compute_full_2['median_DCR_to_real_data']['synthetic_data']
        )

    def test_compute_breakdown_iterations(self):
        """Test that number iterations for subsampling works as expected."""
        # Setup
        real_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(10)]})
        synthetic_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(10)]})
        metadata = {'columns': {'num_col': {'sdtype': 'numerical'}}}
        num_rows_subsample = 3
        num_iterations = 1000

        # Run
        compute_num_iteration_1 = DCRBaselineProtection.compute_breakdown(
            real_data, synthetic_data, metadata, num_rows_subsample, 1
        )
        compute_num_iteration_1000 = DCRBaselineProtection.compute_breakdown(
            real_data, synthetic_data, metadata, num_rows_subsample, num_iterations
        )
        compute_train_same = DCRBaselineProtection.compute_breakdown(
            synthetic_data, synthetic_data, metadata, num_rows_subsample, num_iterations
        )

        # Assert
        assert compute_num_iteration_1 != compute_num_iteration_1000
        assert compute_train_same['score'] == 0.0
