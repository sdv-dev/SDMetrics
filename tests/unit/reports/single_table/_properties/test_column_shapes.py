from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd

from sdmetrics.reports.single_table._properties.column_shapes import ColumnShapes


class TestColumnShapes:

    @patch('sdmetrics.reports.single_table._properties.column_shapes.KSComplement.compute')
    @patch('sdmetrics.reports.single_table._properties.column_shapes.TVComplement.compute')
    def test__generate_details(self, tv_complement_compute_mock, ks_complement_compute_mock):
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
        column_shape_property = ColumnShapes()
        column_shape_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        expected_calls_ksc = [
            call(real_data['col1'], synthetic_data['col1']),
            call(real_data['col4'], synthetic_data['col4']),
        ]
        expected_calls_tvc = [
            call(real_data['col2'], synthetic_data['col2']),
            call(real_data['col3'], synthetic_data['col3']),
        ]

        ks_complement_compute_mock.assert_has_calls(expected_calls_ksc)
        tv_complement_compute_mock.assert_has_calls(expected_calls_tvc)

    @patch('sdmetrics.reports.single_table._properties.column_shapes.KSComplement.compute')
    @patch('sdmetrics.reports.single_table._properties.column_shapes.TVComplement.compute')
    def test__generate_details_with_nans(
        self, tv_complement_compute_mock, ks_complement_compute_mock
    ):
        """Test the ``_generate_details`` method when there are NaNs in the data."""
        # Setup
        real_data = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': [False, True, np.nan],
            'col3': [None, 'b', 'c'],
            'col4': pd.to_datetime(['2020-01-01', np.nan, '2020-01-03'])
        })
        synthetic_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [False, True, True],
            'col3': ['a', None, 'c'],
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
        column_shape_property = ColumnShapes()
        column_shape_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        expected_calls_ksc = [
            call(real_data['col1'], synthetic_data['col1']),
            call(real_data['col4'], synthetic_data['col4']),
        ]
        expected_calls_tvc = [
            call(real_data['col2'], synthetic_data['col2']),
            call(real_data['col3'], synthetic_data['col3']),
        ]

        ks_complement_compute_mock.assert_has_calls(expected_calls_ksc)
        tv_complement_compute_mock.assert_has_calls(expected_calls_tvc)

    def test__generate_details_error(self):
        """Test the ``_generate_details`` method with the error column."""
        # Setup
        real_data = pd.DataFrame({'col1': [1, '2', 3]})
        synthetic_data = pd.DataFrame({'col1': [4, 5, 6]})
        metadata = {'columns': {'col1': {'sdtype': 'numerical'}}}

        column_shape_property = ColumnShapes()

        # Run
        result = column_shape_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        expected_message = (
            "TypeError: '<' not supported between instances of 'str' and 'int'"
        )
        result_nan = result.loc[pd.isna(result['Score'])]
        column_names_nan = result_nan['Column'].tolist()
        error_message = result_nan['Error'].tolist()

        assert column_names_nan == ['col1']
        assert error_message == [expected_message]

        @patch('sdmetrics.reports.single_table._properties.column_shapes.px')
        def test_get_visualization(self, mock_px):
            """Test the ``get_visualization`` method."""
            # Setup
            column_shape_property = ColumnShapes()

            mock_df = pd.DataFrame({
                'Column': ['Column1', 'Column2'],
                'Score': [0.7, 0.3],
                'Metric': ['KSComplement', 'TVComplement']
            })
            column_shape_property.details = mock_df

            mock__compute_average = Mock(return_value=0.5)
            column_shape_property._compute_average = mock__compute_average

            mock_bar = Mock()
            mock_px.bar.return_value = mock_bar

            # Run
            column_shape_property.get_visualization()

            # Assert
            mock__compute_average.assert_called_once()

            # Expected call
            expected_kwargs = {
                'data_frame': mock_df,
                'x': 'Column',
                'y': 'Score',
                'title': (
                    'Data Quality: Column Shapes (Average'
                    f'Score={mock__compute_average.return_value})'
                ),
                'category_orders': {'group': mock_df['Column'].tolist()},
                'color': 'Metric',
                'color_discrete_map': {
                    'KSComplement': '#000036',
                    'TVComplement': '#03AFF1',
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

            # Check call_args of mock_px.bar
            _, kwargs = mock_px.bar.call_args

            # Check DataFrame separately
            assert kwargs.pop('data_frame').equals(expected_kwargs.pop('data_frame'))

            # Check other arguments
            assert kwargs == expected_kwargs

            mock_bar.update_yaxes.assert_called_once_with(range=[0, 1])
            mock_bar.update_layout.assert_called_once_with(
                xaxis_categoryorder='total ascending',
                plot_bgcolor='#F5F5F8',
                margin={'t': 150},
                font={'size': 18}
            )
