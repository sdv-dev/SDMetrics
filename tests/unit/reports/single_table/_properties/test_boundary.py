from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd

from sdmetrics.reports.single_table._properties.boundary import Boundary


class TestBoundary:

    @patch('sdmetrics.reports.single_table._properties.boundary.BoundaryAdherence.compute')
    def test__generate_details(self, boundary_adherence_mock):
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
        boundary_property = Boundary()
        boundary_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        expected_calls_boundary = [
            call(real_data['col1'], synthetic_data['col1']),
            call(real_data['col4'], synthetic_data['col4']),
        ]

        boundary_adherence_mock.assert_has_calls(expected_calls_boundary)

    @patch('sdmetrics.reports.single_table._properties.boundary.BoundaryAdherence.compute')
    def test__generate_details_error(self, boundary_adherence_mock):
        """Test the ``_generate_details`` method when the metric raises an error."""
        # Setup

        boundary_adherence_mock.side_effect = ValueError('Mock Error')
        real_data = pd.DataFrame({
            'col1': [1, 2, np.nan],
        })
        synthetic_data = pd.DataFrame({
            'col1': [1, 2, 3]
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'}
            }
        }

        # Run
        boundary_property = Boundary()
        details = boundary_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        expected_calls_boundary = [
            call(real_data['col1'], synthetic_data['col1']),
        ]

        boundary_adherence_mock.assert_has_calls(expected_calls_boundary)
        expected_details = pd.DataFrame({
            'Column': ['col1'],
            'Metric': ['BoundaryAdherence'],
            'Score': [np.nan],
            'Error': ['ValueError: Mock Error']
        })

        pd.testing.assert_frame_equal(details, expected_details)

    @patch('sdmetrics.reports.single_table._properties.boundary.BoundaryAdherence.compute')
    def test__generate_details_all_nans(self, boundary_adherence_mock):
        """Test the ``_generate_details`` method when the columns contain only ``NaN`` values."""
        # Setup
        real_data = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan],
        })
        synthetic_data = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan]
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'}
            }
        }

        # Run
        boundary_property = Boundary()
        details = boundary_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        boundary_adherence_mock.assert_not_called()
        expected_details = pd.DataFrame({
            'Column': ['col1'],
            'Metric': ['BoundaryAdherence'],
            'Score': [np.nan],
            'Error': ['InvalidDataError: All NaN values in both real and synthetic data.']
        })

        pd.testing.assert_frame_equal(details, expected_details)

    @patch('sdmetrics.reports.single_table._properties.boundary.BoundaryAdherence.compute')
    def test__generate_details_all_nans_real_column(self, boundary_adherence_mock):
        """Test the ``_generate_details`` method when the real column contains only ``NaNs``."""
        # Setup
        real_data = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan],
        })
        synthetic_data = pd.DataFrame({
            'col1': [1, 2, 3]
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'}
            }
        }

        # Run
        boundary_property = Boundary()
        details = boundary_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        boundary_adherence_mock.assert_not_called()
        expected_details = pd.DataFrame({
            'Column': ['col1'],
            'Metric': ['BoundaryAdherence'],
            'Score': [np.nan],
            'Error': ['InvalidDataError: All NaN values in real data.']
        })

        pd.testing.assert_frame_equal(details, expected_details)

    @patch('sdmetrics.reports.single_table._properties.boundary.BoundaryAdherence.compute')
    def test__generate_details_all_nans_synthetic_column(self, boundary_adherence_mock):
        """Test the ``_generate_details`` method when synthetic column contains only ``NaNs``."""
        # Setup
        real_data = pd.DataFrame({
            'col1': [1, 2, 3],
        })
        synthetic_data = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan]
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'}
            }
        }

        # Run
        boundary_property = Boundary()
        details = boundary_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        boundary_adherence_mock.assert_not_called()
        expected_details = pd.DataFrame({
            'Column': ['col1'],
            'Metric': ['BoundaryAdherence'],
            'Score': [np.nan],
            'Error': ['InvalidDataError: All NaN values in synthetic data.']
        })

        pd.testing.assert_frame_equal(details, expected_details)

    @patch('sdmetrics.reports.single_table._properties.boundary.px')
    def test_get_visualization(self, mock_px):
        """Test the ``get_visualization`` method."""
        # Setup
        boundary_property = Boundary()

        mock_df = pd.DataFrame({
            'Column': ['Column1', 'Column2'],
            'Score': [0.7, 0.3],
            'Metric': ['Rangeboundary', 'Categoryboundary']
        })
        boundary_property.details = mock_df

        mock__compute_average = Mock(return_value=0.5)
        boundary_property._compute_average = mock__compute_average

        mock_fig = Mock()
        mock_px.bar.return_value = mock_fig

        # Run
        boundary_property.get_visualization()

        # Assert
        mock__compute_average.assert_called_once()

        expected_kwargs = {
            'data_frame': mock_df,
            'x': 'Column',
            'y': 'Score',
            'title': f'Data Diagnostics: Column Boundary (Average Score={0.5})',
            'category_orders': {'group': list(mock_df['Column'])},
            'color': 'Metric',
            'color_discrete_map': {
                'BoundaryAdherence': '#000036',
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
