from unittest.mock import patch

import numpy as np
import pandas as pd

from sdmetrics.single_table import NewRowSynthesis


class TestNewRowSynthesis:

    def test_compute(self):
        """Test the ``compute`` method and expect that the new row synthesis score is returned."""
        # Setup
        real_data = pd.DataFrame({
            'pk': [0, 1, 2, 3, 4],
            'col1': [0, 1, 2, 3, 4],
            'col2': [1, 2, 1, 3, 4],
            'col3': ['a', 'b', 'c', 'd', 'b'],
            'col4': [1.32, np.nan, 1.43, np.nan, 2.0],
            'col5': [51, 52, 53, 54, 55],
            'col6': ['2020-01-02', '2021-01-04', '2021-05-03', '2022-10-11', '2022-11-13'],
        })
        synthetic_data = pd.DataFrame({
            'pk': [5, 6, 7, 8, 9],
            'col1': [0, 1, 2, 3, 4],
            'col2': [1, 3, 4, 2, 2],
            'col3': ['a', 'b', 'c', 'b', 'e'],
            'col4': [1.32, 1.56, 1.21, np.nan, 1.90],
            'col5': [51, 51, 54, 55, 53],
            'col6': ['2020-01-02', '2022-11-24', '2022-06-01', '2021-04-12', '2020-12-11'],
        })
        metadata = {
            'primary_key': 'pk',
            'columns': {
                'pk': {'sdtype': 'id'},
                'col1': {'sdtype': 'id'},
                'col2': {'sdtype': 'numerical'},
                'col3': {'sdtype': 'categorical'},
                'col4': {'sdtype': 'numerical'},
                'col5': {'sdtype': 'categorical'},
                'col6': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            },
        }
        metric = NewRowSynthesis()

        # Run
        score = metric.compute(real_data, synthetic_data, metadata)

        # Assert
        assert score == 0.8

    def test_compute_breakdown_multi_line(self):
        """Test the ``compute_breakdown`` method with a multi-line value.

        Expect that the match is made correctly."""
        # Setup
        real_data = pd.DataFrame({
            'col1': ['PSC 0481, Box 5945\nAPO AP 37588', 'Unit 9759 Box 8761\nDPO AE 97614'],
        })
        synthetic_data = pd.DataFrame({
            'col1': ['PSC 0481, Box 5945\nAPO AP 37588', '9759 8761\nDPO AE 97614'],
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'categorical'},
            },
        }
        metric = NewRowSynthesis()

        # Run
        score = metric.compute(real_data, synthetic_data, metadata)

        # Assert
        assert score == 0.5

    def test_compute_with_sample_size(self):
        """Test the ``compute`` method with a sample size.

        Expect that the new row synthesis score is returned.
        """
        # Setup
        real_data = pd.DataFrame({
            'col1': [1, 2, 1, 3, 4],
            'col2': ['a', 'b', 'c', 'd', 'b'],
            'col3': [1.32, np.nan, 1.43, np.nan, 2.0],
        })
        synthetic_data = pd.DataFrame({
            'col1': [1, 3, 4, 2, 2],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [1.46, 1.56, 1.21, np.nan, 1.92],
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'categorical'},
                'col3': {'sdtype': 'numerical'},
            },
        }
        sample_size = 2
        metric = NewRowSynthesis()

        # Run
        score = metric.compute(
            real_data, synthetic_data, metadata, synthetic_sample_size=sample_size)

        # Assert
        assert score == 1

    @patch('sdmetrics.single_table.new_row_synthesis.warnings')
    def test_compute_with_sample_size_too_large(self, warnings_mock):
        """Test the ``compute`` method with a sample size larger than the number of rows.

        Expect that the new row synthesis is returned. Expect a warning to be raised.
        """
        # Setup
        real_data = pd.DataFrame({
            'col1': [1, 2, 1, 3, 4],
            'col2': ['a', 'b', 'c', 'd', 'b'],
            'col3': [1.32, np.nan, 1.43, np.nan, 2.0],
        })
        synthetic_data = pd.DataFrame({
            'col1': [1, 3, 4, 2, 2],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [1.35, 1.56, 1.21, np.nan, 1.92],
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'categorical'},
                'col3': {'sdtype': 'numerical'},
            },
        }
        sample_size = 15
        metric = NewRowSynthesis()

        # Run
        score = metric.compute(
            real_data, synthetic_data, metadata, synthetic_sample_size=sample_size)

        # Assert
        assert score == 1
        warnings_mock.warn.assert_called_once_with(
            'The provided `synthetic_sample_size` of 15 is larger than the number of '
            'synthetic data rows (5). Proceeding without sampling.'
        )

    def test_compute_with_many_columns(self):
        """Test the ``compute`` method with more than 32 columns.

        Expect that the new row synthesis is returned.
        """
        # Setup
        num_cols = 32
        real_data = pd.DataFrame({
            f'col{i}': list(np.random.uniform(low=0, high=10, size=100)) for i in range(num_cols)
        })
        synthetic_data = pd.DataFrame({
            f'col{i}': list(np.random.uniform(low=0, high=10, size=100)) for i in range(num_cols)
        })
        metadata = {
            'columns': {
                f'col{i}': {'sdtype': 'numerical'} for i in range(num_cols)
            },
        }
        metric = NewRowSynthesis()

        # Run
        score = metric.compute(real_data, synthetic_data, metadata)

        # Assert
        assert score == 1

    @patch('sdmetrics.single_table.new_row_synthesis.SingleTableMetric.normalize')
    def test_normalize(self, normalize_mock):
        """Test the ``normalize`` method.

        Expect that the inherited ``normalize`` method is called.
        """
        # Setup
        metric = NewRowSynthesis()
        raw_score = 0.9

        # Run
        result = metric.normalize(raw_score)

        # Assert
        normalize_mock.assert_called_once_with(raw_score)
        assert result == normalize_mock.return_value
