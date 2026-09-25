from unittest.mock import Mock, call, patch

import pandas as pd
import pytest

from sdmetrics.reports.single_table._properties.data_validity import (
    BoundaryAdherence,
    DataValidity,
    DatetimeFormatAdherence,
    KeyUniqueness,
    RegexFormatAdherence,
)


class TestDataValidity:
    @pytest.mark.parametrize(
        ('metric', 'column_name', 'expected_result'),
        [
            (DatetimeFormatAdherence, 'start_date', True),
            (DatetimeFormatAdherence, 'end_date', False),
            (RegexFormatAdherence, 'user_id', True),
            (RegexFormatAdherence, 'session_id', False),
            (BoundaryAdherence, 'end_date', True),
            (KeyUniqueness, 'session_id', True),
        ],
    )
    def test__has_required_argument(self, metric, column_name, expected_result):
        """Test the ``_has_required_argument`` method."""
        # Setup
        columns_meta = {
            'start_date': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'end_date': {'sdtype': 'datetime'},
            'user_id': {'sdtype': 'id', 'regex_format': r'\d{1,30}'},
            'session_id': {'sdtype': 'id'},
        }

        # Run
        result = DataValidity._has_required_argument(metric, column_name, columns_meta)

        # Assert
        assert result is expected_result

    @patch('sdmetrics.reports.single_table._properties.data_validity.BoundaryAdherence.compute')
    @patch('sdmetrics.reports.single_table._properties.data_validity.CategoryAdherence.compute')
    def test__generate_details_with_ranges_in_metadata(
        self, category_a_compute_mock, boundary_a_compute_mock
    ):
        """Test that the range information of the metadata is passed down to the metrics."""
        # Setup
        real_data = pd.DataFrame({
            'col1': [10, 20, 30],
            'col2': ['a', 'a', 'b'],
            'col3': [True, False, True],
            'col4': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
        })
        synthetic_data = pd.DataFrame({
            'col1': [15, 25, 35],
            'col2': ['a', 'c', 'a'],
            'col3': [True, True, False],
            'col4': pd.to_datetime(['2020-01-04', '2020-01-05', '2020-01-06']),
        })
        metadata = {
            'columns': {
                'col1': {
                    'sdtype': 'numerical',
                    'range_min': 0,
                    'range_max': 80,
                    'decimal_places': 0,
                },
                'col2': {
                    'sdtype': 'categorical',
                    'range_values': ['a', 'b', 'c'],
                    'range_is_nullable': True,
                },
                'col3': {'sdtype': 'boolean'},
                'col4': {
                    'sdtype': 'datetime',
                    'datetime_format': '%Y-%m-%d',
                    'range_min': '2019-01-01',
                },
            },
        }

        # Run
        data_validity_property = DataValidity()
        data_validity_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        expected_calls_ba = [
            call(real_data['col1'], synthetic_data['col1'], range_min=0, range_max=80),
            call(
                real_data['col4'],
                synthetic_data['col4'],
                range_min='2019-01-01',
            ),
        ]
        expected_calls_ca = [
            call(
                real_data['col2'],
                synthetic_data['col2'],
                range_values=['a', 'b', 'c'],
                range_is_nullable=True,
            ),
            call(real_data['col3'], synthetic_data['col3']),
        ]
        boundary_a_compute_mock.assert_has_calls(expected_calls_ba)
        category_a_compute_mock.assert_has_calls(expected_calls_ca)

    @patch('sdmetrics.reports.single_table._properties.data_validity.BoundaryAdherence.compute')
    @patch('sdmetrics.reports.single_table._properties.data_validity.CategoryAdherence.compute')
    @patch('sdmetrics.reports.single_table._properties.data_validity.KeyUniqueness.compute')
    def test__generate_details(
        self, key_uniqueness_mock, category_a_compute_mock, boundary_a_compute_mock
    ):
        """Test the ``_generate_details`` method."""
        # Setup
        real_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [False, True, True],
            'col3': ['a', 'b', 'c'],
            'col4': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
            'col5': ['ID_1', 'ID_2', 'ID_3'],
            'col6': ['A', 'B', 'C'],
        })
        synthetic_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [False, True, True],
            'col3': ['a', 'b', 'c'],
            'col4': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
            'col5': ['ID_4', 'ID_5', 'ID_6'],
            'col6': ['D', 'E', 'F'],
        })
        metadata = {
            'primary_key': 'col5',
            'alternate_keys': ['col6'],
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'boolean'},
                'col3': {'sdtype': 'categorical'},
                'col4': {'sdtype': 'datetime'},
                'col5': {'sdtype': 'id'},
                'col6': {'sdtype': 'other'},
            },
        }

        # Run
        data_validity_property = DataValidity()
        data_validity_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        expected_calls_ba = [
            call(real_data['col1'], synthetic_data['col1']),
            call(real_data['col4'], synthetic_data['col4']),
        ]
        expected_calls_ca = [
            call(real_data['col2'], synthetic_data['col2']),
            call(real_data['col3'], synthetic_data['col3']),
        ]
        expected_calls_key = [
            call(real_data['col5'], synthetic_data['col5']),
            call(real_data['col6'], synthetic_data['col6']),
        ]
        boundary_a_compute_mock.assert_has_calls(expected_calls_ba)
        category_a_compute_mock.assert_has_calls(expected_calls_ca)
        key_uniqueness_mock.assert_has_calls(expected_calls_key)

    @patch(
        'sdmetrics.reports.single_table._properties.data_validity.DatetimeFormatAdherence.compute'
    )
    @patch('sdmetrics.reports.single_table._properties.data_validity.RegexFormatAdherence.compute')
    @patch('sdmetrics.reports.single_table._properties.data_validity.BoundaryAdherence.compute')
    @patch('sdmetrics.reports.single_table._properties.data_validity.CategoryAdherence.compute')
    @patch('sdmetrics.reports.single_table._properties.data_validity.KeyUniqueness.compute')
    def test__generate_detail_with_specified_formats(
        self,
        key_uniqueness_mock,
        category_a_compute_mock,
        boundary_a_compute_mock,
        regex_a_compute_mock,
        datetime_a_compute_mock,
    ):
        """Test the ``_generate_details`` method with specified formats."""
        # Setup
        real_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [False, True, True],
            'col3': ['a', 'b', 'c'],
            'col4': ['2020-01-01', '2020-01-02', '2020-01-03'],
            'col5': ['ID_1', 'ID_2', 'ID_3'],
            'col6': ['A', 'B', 'C'],
        })
        synthetic_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [False, True, True],
            'col3': ['a', 'b', 'c'],
            'col4': ['2020-01-01', '2020-01-02', '2020-01-03'],
            'col5': ['ID_4', 'ID_5', 'ID_6'],
            'col6': ['D', 'E', 'F'],
        })
        metadata = {
            'primary_key': 'col5',
            'alternate_keys': ['col6'],
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'boolean'},
                'col3': {'sdtype': 'categorical'},
                'col4': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                'col5': {'sdtype': 'id', 'regex_format': r'ID_\d+'},
                'col6': {'sdtype': 'other'},
            },
        }

        # Run
        data_validity_property = DataValidity()
        data_validity_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        expected_calls = [
            (real_data['col4'], synthetic_data['col4'], '%Y-%m-%d'),
            (real_data['col5'], synthetic_data['col5'], r'ID_\d+'),
        ]
        expected_calls_ba = [
            call(real_data['col1'], synthetic_data['col1']),
            call(real_data['col4'], synthetic_data['col4']),
        ]
        expected_calls_ca = [
            call(real_data['col2'], synthetic_data['col2']),
            call(real_data['col3'], synthetic_data['col3']),
        ]
        expected_calls_key = [
            call(real_data['col5'], synthetic_data['col5']),
            call(real_data['col6'], synthetic_data['col6']),
        ]
        datetime_a_compute_mock.assert_called_once()
        regex_a_compute_mock.assert_called_once()
        boundary_a_compute_mock.assert_has_calls(expected_calls_ba)
        category_a_compute_mock.assert_has_calls(expected_calls_ca)
        key_uniqueness_mock.assert_has_calls(expected_calls_key)

        for i, mock in enumerate([datetime_a_compute_mock, regex_a_compute_mock]):
            mock_call_args = mock.call_args.args
            for actual, expected in zip(mock_call_args, expected_calls[i]):
                if isinstance(expected, pd.Series):
                    pd.testing.assert_series_equal(actual, expected)
                else:
                    assert actual == expected

    def test__generate_details_error(self):
        """Test the ``_generate_details`` method with the error column."""
        # Setup
        real_data = pd.DataFrame({'col1': [1, '2', 3]})
        synthetic_data = pd.DataFrame({'col1': [4, 5, 6]})
        metadata = {'columns': {'col1': {'sdtype': 'numerical'}}}

        data_validity_property = DataValidity()

        # Run
        result = data_validity_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        expected_message = "TypeError: '<=' not supported between instances of 'int' and 'str'"
        result_nan = result.loc[pd.isna(result['Score'])]
        column_names_nan = result_nan['Column'].tolist()
        error_message = result_nan['Error'].tolist()

        assert column_names_nan == ['col1']
        assert error_message == [expected_message]

    @patch('sdmetrics.reports.single_table._properties.data_validity.px')
    def test_get_visualization(self, mock_px):
        """Test the ``get_visualization`` method."""
        # Setup
        data_validity_property = DataValidity()

        mock_df = pd.DataFrame({
            'Column': ['Column1', 'Column2', 'Column3'],
            'Score': [0.7, 0.3, 0.5],
            'Metric': ['BoundaryAdherence', 'CategoryAdherence', 'KeyUniqueness'],
        })
        data_validity_property.details = mock_df

        mock__compute_average = Mock(return_value=0.5)
        data_validity_property._compute_average = mock__compute_average

        mock_bar = Mock()
        mock_px.bar.return_value = mock_bar

        # Run
        data_validity_property.get_visualization()

        # Assert
        mock__compute_average.assert_called_once()

        # Expected call
        expected_kwargs = {
            'data_frame': mock_df,
            'x': 'Column',
            'y': 'Score',
            'title': (
                'Data Diagnostic: Data Validity (Average '
                f'Score={mock__compute_average.return_value})'
            ),
            'category_orders': {'group': mock_df['Column'].tolist()},
            'color': 'Metric',
            'color_discrete_map': {
                'BoundaryAdherence': '#000036',
                'CategoryAdherence': '#03AFF1',
                'KeyUniqueness': '#01E0C9',
            },
            'pattern_shape': 'Metric',
            'pattern_shape_sequence': ['', '/', '.'],
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
            font={'size': 18},
        )

    @patch('sdmetrics.reports.single_table._properties.data_validity.BoundaryAdherence.compute')
    @patch('sdmetrics.reports.single_table._properties.data_validity.CategoryAdherence.compute')
    @patch('sdmetrics.reports.single_table._properties.data_validity.KeyUniqueness.compute')
    def test__generate_details_skip_sequence_index_boundary_adherence(
        self, key_uniqueness_mock, category_a_compute_mock, boundary_a_compute_mock
    ):
        """Test that sequence_index columns are excluded from BoundaryAdherence checks."""
        # Setup
        real_data = pd.DataFrame({
            'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
            'value': [1, 2, 3],
            'category': ['a', 'b', 'c'],
        })
        synthetic_data = pd.DataFrame({
            'date': pd.to_datetime(['2020-01-04', '2020-01-05', '2020-01-06']),
            'value': [4, 5, 6],
            'category': ['d', 'e', 'f'],
        })
        metadata = {
            'sequence_index': 'date',  # This should skip BoundaryAdherence
            'columns': {
                'date': {'sdtype': 'datetime'},
                'value': {'sdtype': 'numerical'},
                'category': {'sdtype': 'categorical'},
            },
        }

        boundary_a_compute_mock.return_value = 0.8
        category_a_compute_mock.return_value = 0.9

        # Run
        data_validity_property = DataValidity()
        result = data_validity_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        expected_calls_ba = [call(real_data['value'], synthetic_data['value'])]
        boundary_a_compute_mock.assert_has_calls(expected_calls_ba)
        assert boundary_a_compute_mock.call_count == 1

        expected_calls_ca = [call(real_data['category'], synthetic_data['category'])]
        category_a_compute_mock.assert_has_calls(expected_calls_ca)
        assert category_a_compute_mock.call_count == 1

        key_uniqueness_mock.assert_not_called()

        expected_columns = ['value', 'category']
        assert list(result['Column']) == expected_columns
        expected_metrics = ['BoundaryAdherence', 'CategoryAdherence']
        assert list(result['Metric']) == expected_metrics
