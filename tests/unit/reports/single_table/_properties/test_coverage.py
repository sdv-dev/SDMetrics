from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd

from sdmetrics.reports.single_table._properties.coverage import Coverage


class TestCoverage:

    @patch('sdmetrics.reports.single_table._properties.coverage.RangeCoverage.compute')
    @patch('sdmetrics.reports.single_table._properties.coverage.CategoryCoverage.compute')
    def test__generate_details(self, category_coverage_mock, range_coverage_mock):
        """Test the ``_generate_details`` method."""
        # Setup
        real_data = pd.DataFrame({
            'col1': [1, 2, np.nan],
            'col2': [False, True, True],
            'col3': [None, 'b', 'c'],
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
        coverage_property = Coverage()
        coverage_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        expected_calls_range = [
            call(real_data['col1'], synthetic_data['col1']),
            call(real_data['col4'], synthetic_data['col4']),
        ]
        expected_calls_category = [
            call(real_data['col2'], synthetic_data['col2']),
            call(real_data['col3'], synthetic_data['col3']),
        ]

        range_coverage_mock.assert_has_calls(expected_calls_range)
        category_coverage_mock.assert_has_calls(expected_calls_category)

    @patch('sdmetrics.reports.single_table._properties.coverage.px')
    def test_get_visualization(self, mock_px):
        """Test the ``get_visualization`` method."""
        # Setup
        coverage_property = Coverage()

        mock_df = pd.DataFrame({
            'Column': ['Column1', 'Column2', 'Column3'],
            'Score': [0.7, 0.3, np.nan],
            'Metric': ['RangeCoverage', 'CategoryCoverage', 'CategoryCoverage']
        })
        coverage_property.details = mock_df

        mock__compute_average = Mock(return_value=0.5)
        coverage_property._compute_average = mock__compute_average

        mock_fig = Mock()
        mock_px.bar.return_value = mock_fig

        # Run
        coverage_property.get_visualization()

        # Assert
        mock__compute_average.assert_called_once()
        expected_df = pd.DataFrame({
            'Column': ['Column1', 'Column2'],
            'Score': [0.7, 0.3],
            'Metric': ['RangeCoverage', 'CategoryCoverage']
        })

        expected_kwargs = {
            'data_frame': expected_df,
            'x': 'Column',
            'y': 'Score',
            'title': f'Data Diagnostics: Column Coverage (Average Score={0.5})',
            'category_orders': {'group': mock_df['Column'].tolist()},
            'color': 'Metric',
            'color_discrete_map': {
                'RangeCoverage': '#000036',
                'CategoryCoverage': '#03AFF1',
            },
            'pattern_shape': 'Metric',
            'pattern_shape_sequence': ['', '/'],
            'hover_name': 'Column',
            'hover_data': {
                'Column': False,
                'Metric': True,
                'Score': True,
            },
        }

        _, kwargs = mock_px.bar.call_args

        assert kwargs.pop('data_frame').equals(expected_kwargs.pop('data_frame'))
        assert kwargs == expected_kwargs

        mock_fig.update_yaxes.assert_called_once_with(range=[0, 1], title_text='Diagnostic Score')
        mock_fig.update_layout.assert_called_once_with(
            xaxis_categoryorder='total ascending', plot_bgcolor='#F5F5F8', margin={'t': 150}
        )
