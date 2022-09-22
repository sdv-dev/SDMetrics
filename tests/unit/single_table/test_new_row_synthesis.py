from unittest.mock import patch

import numpy as np
import pandas as pd

from sdmetrics.single_table import NewRowSynthesis


class TestNewRowSynthesis:

    def test_compute(self):
        """Test the ``compute`` method and expect that the new row synthesis score is returned."""
        # Setup
        real_data = pd.DataFrame({
            'col1': [1, 2, 1, 3, 4],
            'col2': ['a', 'b', 'c', 'd', 'b'],
            'col3': [1.32, np.nan, 1.43, np.nan, 2.0],
        })
        synthetic_data = pd.DataFrame({
            'col1': [1, 3, 4, 2, 2],
            'col2': ['a', 'b', 'c', 'b', 'e'],
            'col3': [1.32, 1.56, 1.21, np.nan, 1.90],
        })
        metadata = {
            'fields': {
                'col1': {'type': 'numerical', 'subtype': 'int'},
                'col2': {'type': 'categorical'},
                'col3': {'type': 'numerical', 'subtype': 'float'},
            },
        }
        metric = NewRowSynthesis()

        # Run
        score = metric.compute(real_data, synthetic_data, metadata)

        # Assert
        assert score == 0.6

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
            'fields': {
                'col1': {'type': 'numerical', 'subtype': 'int'},
                'col2': {'type': 'categorical'},
                'col3': {'type': 'numerical', 'subtype': 'float'},
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
            'fields': {
                'col1': {'type': 'numerical', 'subtype': 'int'},
                'col2': {'type': 'categorical'},
                'col3': {'type': 'numerical', 'subtype': 'float'},
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
