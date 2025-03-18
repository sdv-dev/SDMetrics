import random
import re

import numpy as np
import pandas as pd
import pytest

from datetime import datetime

from sdmetrics.demos import load_single_table_demo
from sdmetrics.single_table.privacy import DCRBaselineProtection


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

        result = DCRBaselineProtection.compute_breakdown(real_data, synth_data, metadata)
        assert result['score'] == 0.0
        assert result['median_DCR_to_real_data']['random_data_baseline'] > 0

    def test_compute_breakdown_subsampling(self):
        """Test subsampling produces different values."""
        # Setup
        real_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(20)]})
        synthetic_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(20)]})
        metadata = {'columns': {'num_col': {'sdtype': 'numerical'}}}
        large_num_subsample = len(synthetic_data) * 2
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

        large_subsample_msg = re.escape('Ignoring the num_rows_subsample and num_iterations args.')
        with pytest.warns(UserWarning, match=large_subsample_msg):
            compute_large_subsample = DCRBaselineProtection.compute_breakdown(
                real_data, synthetic_data, metadata, large_num_subsample
            )

        # Assert that subsampling provides different values.
        assert (
            compute_subsample['median_DCR_to_real_data']['synthetic_data']
            != compute_full_1['median_DCR_to_real_data']['synthetic_data']
        )
        assert (
            compute_full_1['median_DCR_to_real_data']['synthetic_data']
            == compute_full_2['median_DCR_to_real_data']['synthetic_data']
        )

        assert (
            compute_large_subsample['median_DCR_to_real_data']['synthetic_data']
            == compute_full_2['median_DCR_to_real_data']['synthetic_data']
        )

    def test_compute_breakdown_max_dcr(self):
        """Test that the score is 1.0 if synthetic data has no overlapping values."""
        # Setup
        real_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(10)]})
        synthetic_data = pd.DataFrame({'num_col': [random.randint(2000, 3000) for _ in range(10)]})
        metadata = {'columns': {'num_col': {'sdtype': 'numerical'}}}

        # Run should not have any iteration that gives us an unexpected value.
        result = DCRBaselineProtection.compute_breakdown(
            real_data,
            synthetic_data,
            metadata,
        )

        # Assert
        assert result['score'] == 1.0
        assert result['median_DCR_to_real_data']['synthetic_data'] == 1.0

    def test_end_to_end_with_single_value(self):
        """Test end to end with a simple single synthetic value."""
        # Setup
        real_data = pd.DataFrame(data={'A': [0, 10, 3, 4, 1]})
        synthetic_data = pd.DataFrame(data={'A': [5]})
        metadata = {'columns': {'A': {'sdtype': 'numerical'}}}

        # Run
        result = DCRBaselineProtection.compute_breakdown(
            real_data=real_data, synthetic_data=synthetic_data, metadata=metadata
        )

        # Assert
        assert result['median_DCR_to_real_data']['synthetic_data'] == 0.1

    def test_end_to_end_with_nan_value(self):
        """Test end to end with a nan value."""
        # Setup
        real_data = pd.DataFrame(data={'A': [0, 10, 3, 4, 1, np.nan]})
        synthetic_data = pd.DataFrame(data={'A': [np.nan]})
        metadata = {'columns': {'A': {'sdtype': 'numerical'}}}

        # Run
        result = DCRBaselineProtection.compute_breakdown(
            real_data=real_data, synthetic_data=synthetic_data, metadata=metadata
        )

        # Assert
        assert result['median_DCR_to_real_data']['synthetic_data'] == 0.0
        if result['median_DCR_to_real_data']['random_data_baseline'] != 0.0:
            assert result['score'] == 0.0

    def test_end_to_end_with_zero_col_range(self):
        """Test end to end with a nan value."""
        # Setup
        real_data = pd.DataFrame(data={'A': [1.0]})
        synthetic_data = pd.DataFrame(data={'A': [2.0, np.nan]})
        metadata = {'columns': {'A': {'sdtype': 'numerical'}}}

        # Run
        result = DCRBaselineProtection.compute_breakdown(
            real_data=real_data, synthetic_data=synthetic_data, metadata=metadata
        )

        # Assert
        assert result['median_DCR_to_real_data']['synthetic_data'] == 1.0
        assert result['median_DCR_to_real_data']['random_data_baseline'] == 0.0
        assert np.isnan(result['score'])

    def test_end_to_end_sample_random_median(self):
        """Test end to end with a simple single synthetic value."""
        # Setup
        real_data = pd.DataFrame(data={'A': [2, 6, 3, 4, 1]})
        synthetic_data = pd.DataFrame(data={'A': [5, 5, 5, 5, 5]})
        metadata = {'columns': {'A': {'sdtype': 'numerical'}}}
        num_rows_sample = 1
        num_iterations = 5

        # Run
        result = DCRBaselineProtection.compute_breakdown(
            real_data=real_data,
            synthetic_data=synthetic_data,
            metadata=metadata,
            num_rows_subsample=num_rows_sample,
            num_iterations=num_iterations,
        )

        # Assert
        assert result['median_DCR_to_real_data']['synthetic_data'] == 0.2
        if result['median_DCR_to_real_data']['random_data_baseline'] == 0.0:
            assert np.isnan(result['score'])

    def test_end_to_end_with_datetimes(self):
        """Test end to end with a simple single synthetic value."""
        # Setup
        real_data = pd.DataFrame(
            data={
                'datetime': [datetime(2025, 1, 1), datetime(2025, 1, 1)],
                'datetime_str': ['2025-01-01', '2025-01-02'],
                'datetime_str_no_fmt': ['2025-01-01', '2025-01-02']
            })
        synthetic_data = pd.DataFrame(data={'A': [5, 5, 5, 5, 5]})
        metadata = {'columns': {'A': {'sdtype': 'numerical'}}}
        num_rows_sample = 1
        num_iterations = 5

        # Run
        result = DCRBaselineProtection.compute_breakdown(
            real_data=real_data,
            synthetic_data=synthetic_data,
            metadata=metadata,
            num_rows_subsample=num_rows_sample,
            num_iterations=num_iterations,
        )

        # Assert
        assert result['median_DCR_to_real_data']['synthetic_data'] == 0.2
        if result['median_DCR_to_real_data']['random_data_baseline'] == 0.0:
            assert np.isnan(result['score'])
