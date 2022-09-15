from unittest.mock import patch

import numpy as np
import pandas as pd

from sdmetrics.single_table import SyntheticUniqueness


class TestSyntheticUniqueness:

    def test_compute(self):
        """Test the ``compute`` method.

        Expect that the synthetic uniqueness is returned.

        Input:
        - real data
        - synthetic data

        Output:
        - the evaluated metric
        """
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

        # Run
        metric = SyntheticUniqueness()
        score = metric.compute(real_data, synthetic_data, metadata)

        # Assert
        assert score == 0.6

    def test_compute_with_sample_size(self):
        """Test the ``compute`` method with a sample size.

        Expect that the synthetic uniqueness is returned.

        Input:
        - real data
        - synthetic data

        Output:
        - the evaluated metric
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
            'col3': [1.33, 1.56, 1.21, np.nan, 1.92],
        })
        metadata = {
            'fields': {
                'col1': {'type': 'numerical', 'subtype': 'int'},
                'col2': {'type': 'categorical'},
                'col3': {'type': 'numerical', 'subtype': 'float'},
            },
        }
        sample_size = 2

        # Run
        metric = SyntheticUniqueness()
        score = metric.compute(
            real_data, synthetic_data, metadata, synthetic_sample_size=sample_size)

        # Assert
        assert score == 1

    @patch('sdmetrics.single_table.synthetic_uniqueness.SingleTableMetric.normalize')
    def test_normalize(self, normalize_mock):
        """Test the ``normalize`` method.

        Expect that the inherited ``normalize`` method is called.

        Input:
        - raw score

        Output:
        - the output of the inherited ``normalize`` method.
        """
        # Setup
        metric = SyntheticUniqueness()
        raw_score = 0.9

        # Run
        result = metric.normalize(raw_score)

        # Assert
        normalize_mock.assert_called_once_with(raw_score)
        assert result == normalize_mock.return_value
