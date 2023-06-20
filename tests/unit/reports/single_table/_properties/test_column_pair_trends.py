import json
import re
from unittest.mock import call, patch

import pandas as pd

from sdmetrics.reports.single_table._properties.column_pair_trends import ColumnPairTrends


class TestColumnPairTrends:

    def test__datetime_to_numeric():
        pass



    @patch('sdmetrics.reports.single_table._properties.column_pair_trends.CorrelationSimilarity.compute')
    @patch('sdmetrics.reports.single_table._properties.column_pair_trends.ContingencySimilarity.compute')
    def test__generate_details(self, contingency_compute_mock, correlation_compute_mock):
        """Test the ``_generate_details`` method."""
        # Setup
        real_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [False, True, True],
            'col3': ['a', 'b', 'c'],
            'col4': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
        })
        synthetic_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [False, True, True],
            'col3': ['a', 'b', 'c'],
            'col4': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'boolean'},
                'col3': {'sdtype': 'categorical'},
                'col4': {'sdtype': 'datetime'}
            }
        }

        # Run
        column_shape_property = ColumnPairTrends()
        column_shape_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        expected_calls_contingency = [
            call(real_data['col1', 'col2'], synthetic_data['col1', 'col2']),
            call(real_data[['col1', 'col3']], synthetic_data[['col1', 'col3']]),
            call(real_data[['col2', 'col3']], synthetic_data[['col2', 'col3']]),
            call(real_data[['col2', 'col4']], synthetic_data[['col2', 'col4']]),
            call(real_data[['col3', 'col4']], synthetic_data[['col3', 'col4']]),
        ]

        #correlation_compute_mock.assert_called_once_with(real_data[['col1', 'col4']], synthetic_data[['col1', 'col4']])
        contingency_compute_mock.assert_has_calls(expected_calls_contingency)