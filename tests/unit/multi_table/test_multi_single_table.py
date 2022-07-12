from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from sdmetrics.multi_table import MultiSingleTableMetric


class TestMultiSingleTableMetric:

    def test__compute(self):
        """Test the ``_compute`` method.

        Expect that a nested mapping of table name to column breakdown value is returned.

        Setup:
        - Mock the single table metric's ``compute_breakdown`` method.

        Input:
        - real data
        - synthetic data

        Output:
        - A nested mapping of table name to column breakdown is returned, if the column breakdown
          is available.
        """
        # Setup
        table_a_breakdown = {'a': 1.0, 'b': 2.0, 'c': 1.1, 'd': np.nan}
        table_b_breakdown = {'col1': 1.2, 'col2': 1.1, 'col3': 1.5}

        metric_mock = Mock()
        metric_mock.single_table_metric.compute_breakdown.side_effect = [
            table_a_breakdown, table_b_breakdown
        ]

        data = {
            'tableA': pd.DataFrame({
                'a': [0, 1, 2, 3],
                'b': [100, 200, 300, 400],
                'c': ['one', 'two', 'three', 'four'],
                'd': ['a', 'b', 'c', 'd'],
            }),
            'tableB': pd.DataFrame({
                'col1': [0, 1, 2, 3, 4],
                'col2': ['a', 'b', 'c', 'd', 'e'],
                'col3': [4.3, 2.1, 4.1, 1.2, 3.2],
            })
        }

        # Run
        result = MultiSingleTableMetric._compute(metric_mock, data, data)

        # Assert
        assert result == {'tableA': table_a_breakdown, 'tableB': table_b_breakdown}

    def test_compute(self):
        """Test the ``compute`` method.

        Expect that an average of the column metric scores across all tables is returned.

        Setup:
        - ``_compute`` helper method should return a nested mapping of table name to metric
          breakdown.

        Input:
        - real data
        - synthetic data

        Output:
        - The average metric score.
        """
        # Setup
        metric_breakdown = {
            'tableA': {
                'a': {'score': 1.0},
                'b': {'score': 2.0},
                'c': {'score': 1.0},
                'd': {'score': np.nan},
            },
            'tableB': {
                'col1': {'score': 2.0},
                'col2': {'score': np.nan},
                'col3': {'error': ValueError('test error')},
            },
        }

        # Run
        with patch.object(MultiSingleTableMetric, '_compute', return_value=metric_breakdown):
            result = MultiSingleTableMetric.compute(real_data=Mock(), synthetic_data=Mock())

        # Assert
        assert result == 6.0 / 4.0

    def test_compute_breakdown(self):
        """Test the ``compute_breakdown`` method.

        Expect that a nested breakdown of the column metric scores by table and column is returned.

        Setup:
        - ``_compute`` helper method should return a nested mapping of table name to metric
          breakdown.

        Input:
        - real data
        - synthetic data

        Output:
        - A nested mapping of table name to the metric scores broken down by column.
        """
        # Setup
        metric_breakdown = {
            'tableA': {'a': 1.0, 'b': 2.0, 'c': 1.0, 'd': np.nan},
            'tableB': {'col1': 2.0, 'col2': np.nan, 'col3': ValueError('test error')},
        }

        # Run
        with patch.object(MultiSingleTableMetric, '_compute', return_value=metric_breakdown):
            result = MultiSingleTableMetric.compute_breakdown(Mock(), Mock())

        # Assert
        assert result == metric_breakdown
