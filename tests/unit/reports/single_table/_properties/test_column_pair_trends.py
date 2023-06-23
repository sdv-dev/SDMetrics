import itertools
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from sdmetrics.reports.single_table._properties.column_pair_trends import ColumnPairTrends


class TestColumnPairTrends:

    def test__convert_datetime_columns_to_numeric(self):
        """Test the ``_convert_datetime_columns_to_numeric`` method."""
        # Setup
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [False, True, True],
            'col3': ['a', 'b', 'c'],
            'col4': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-01']),
            'col5': [None, '2020-01-02', '2020-01-03'],
            'col6': ['error', '2020-01-02', '2020-01-03']
        })

        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'boolean'},
                'col3': {'sdtype': 'categorical'},
                'col4': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                'col5': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                'col6': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'}
            }
        }
        cpt_property = ColumnPairTrends()

        # Run
        cpt_property._convert_datetime_columns_to_numeric(data, metadata)

        # Assert
        expected_error = (
            'Error: ValueError time data "error" doesn\'t match format "%Y-%m-%d", at '
            'position 0. You might want to try:\n    - passing `format` if your strings'
            " have a consistent format;\n    - passing `format=\'ISO8601\'` if your"
            ' strings are all ISO8601 but not necessarily in exactly the same format;\n'
            "    - passing `format=\'mixed\'`, and the format will be inferred for each"
            ' element individually. You might want to use `dayfirst` alongside this.'
        )

        assert data['col4'].dtype == np.int64
        assert data['col5'].dtype == np.float64
        assert 'col6' in list(cpt_property._columns_datetime_conversion_failed.keys())
        assert cpt_property._columns_datetime_conversion_failed['col6'] == expected_error

    def test__discretize_column(self):
        """Test the ``_discretize_column`` method."""
        # Setup
        data = pd.DataFrame({
            'err_col': ['a', 'b', 'c', 'd', 'e'],
            'int_col': [1, 2, 3, None, 5],
            'float_col': [1.1, np.nan, 3.3, 4.4, 5.5]
        })
        bin_edges = None
        cpt_property = ColumnPairTrends()

        # Run
        col_err, bin_edges = cpt_property._discretize_column('err_col', data['err_col'])
        col_int, bin_edges = cpt_property._discretize_column('int_col', data['int_col'])
        col_float, bin_edges = cpt_property._discretize_column(
            'float_col', data['float_col'], bin_edges
        )

        # Assert
        assert 'err_col' in list(cpt_property._columns_discretization_failed.keys())
        assert list(col_int) == [1, 3, 6, 11, 11]
        assert list(col_float) == [1, 11, 6, 9, 11]

    def test__get_processed_data(self):
        """Test the ``_get_processed_data`` method."""
        # Setup
        data = pd.DataFrame({
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
        cpt_property = ColumnPairTrends()
        processed_data = cpt_property._get_processed_data(data, metadata)

        # Assert
        expected_datetime = pd.to_numeric(
            pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
        )
        expected_processed_real = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [False, True, True],
            'col3': ['a', 'b', 'c'],
            'col4': expected_datetime,
            'col1_discrete': [1, 6, 11],
            'col4_discrete': [1, 6, 11],
        })

        pd.testing.assert_frame_equal(processed_data, expected_processed_real)

    def test__get_processed_data_with_nans(self):
        """Test the ``_get_processed_data`` method."""
        # Setup
        data = pd.DataFrame({
            'col1': [None, 2, 3],
            'col2': [False, np.nan, True],
            'col3': ['a', 'b', 'c'],
            'col4': pd.to_datetime(['2020-01-01', None, '2020-01-03'])
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
        cpt_property = ColumnPairTrends()
        processed_data = cpt_property._get_processed_data(data, metadata)

        # Assert
        expected_datetime = pd.to_numeric(pd.to_datetime(['2020-01-01', None, '2020-01-03']))
        expected_datetime = pd.Series(expected_datetime)
        expected_datetime = expected_datetime.replace(-9223372036854775808, np.nan)

        expected_processed_real = pd.DataFrame({
            'col1': [None, 2, 3],
            'col2': [False, np.nan, True],
            'col3': ['a', 'b', 'c'],
            'col4': expected_datetime,
            'col1_discrete': [11, 1, 11],
            'col4_discrete': [1, 11, 11],
        })

        pd.testing.assert_frame_equal(processed_data, expected_processed_real)

    def test__get_metric(self):
        """Test the ``_get_metric`` method.

        The method should return the correct metric for each combination of column types.
        """
        # Setup
        cpt = ColumnPairTrends()

        # Run and Assert
        cpt._get_metric('datetime', 'datetime').__name__ == 'CorrelationSimilarity'
        cpt._get_metric('numerical', 'numerical').__name__ == 'CorrelationSimilarity'
        cpt._get_metric('datetime', 'numerical').__name__ == 'CorrelationSimilarity'
        cpt._get_metric('datetime', 'categorical').__name__ == 'ContingencySimilarity'
        cpt._get_metric('datetime', 'boolean').__name__ == 'ContingencySimilarity'
        cpt._get_metric('numerical', 'categorical').__name__ == 'ContingencySimilarity'
        cpt._get_metric('numerical', 'boolean').__name__ == 'ContingencySimilarity'
        cpt._get_metric('categorical', 'boolean').__name__ == 'ContingencySimilarity'
        cpt._get_metric('categorical', 'categorical').__name__ == 'ContingencySimilarity'
        cpt._get_metric('boolean', 'boolean').__name__ == 'ContingencySimilarity'

    def get_columns_data(self):
        """Test the ``_get_columns_data`` method.

        The method should return the correct data for each combination of column types.
        """
        # Setup
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [False, True, True],
            'col3': ['a', 'b', 'c'],
            'col4': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
            'col1_discrete': [1, 6, 11],
            'col4_discrete': [1, 6, 11],
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'boolean'},
                'col3': {'sdtype': 'categorical'},
                'col4': {'sdtype': 'datetime'},
            }
        }

        cpt_property = ColumnPairTrends()

        # Run and Assert
        expected_return = [
            data[['col1_discrete', 'col2']],
            data[['col1_discrete', 'col3']],
            data[['col1', 'col4']],
            data[['col2', 'col3']],
            data[['col2', 'col4_discrete']],
            data[['col3', 'col4_discrete']],
        ]
        for idx, col1, col2 in enumerate(itertools.combinations(data.columns, 2)):
            columns_data = cpt_property._get_columns_data(data, metadata)
            pd.testing.assert_frame_equal(columns_data, expected_return[idx])

    def test_required_preprocessing(self):
        """Test the ``_required_preprocessing`` method.

        The method should return the correct boolean for each combination of column types.
        The output is True if one of the column has been preprocessed.
        """
        # Setup
        sdtype_pairs = [
            ('datetime', 'datetime'),
            ('numerical', 'numerical'),
            ('datetime', 'numerical'),
            ('datetime', 'categorical'),
            ('datetime', 'boolean'),
            ('numerical', 'categorical'),
            ('numerical', 'boolean'),
            ('categorical', 'boolean'),
            ('categorical', 'categorical'),
            ('boolean', 'boolean'),
        ]

        cpt_property = ColumnPairTrends()

        # Run and Assert
        expected_return = [
            True, False, True, True, True, True, True, False, False, False
        ]
        for idx, sdtype_pair in enumerate(sdtype_pairs):
            sdtype_1 = sdtype_pair[0]
            sdtype_2 = sdtype_pair[1]
            result = cpt_property._required_preprocessing(sdtype_1, sdtype_2)
            assert result == expected_return[idx]

    def test_preprocessing_failed(self):
        """Test the ``_preprocessing_failed`` method."""
        # Setup
        cpt_property = ColumnPairTrends()
        cpt_property._columns_datetime_conversion_failed = {'col1': 'Error1'}
        cpt_property._columns_discretization_failed = {'col3': 'Error3'}

        # Run
        result_1 = cpt_property._preprocessing_failed('col1', 'col2', 'datetime', 'datetime')
        result_2 = cpt_property._preprocessing_failed('col2', 'col1', 'datetime', 'datetime')
        result_3 = cpt_property._preprocessing_failed('col3', 'col4', 'numerical', 'boolean')
        result_4 = cpt_property._preprocessing_failed('col2', 'col4', 'datetime', 'boolean')

        # Assert
        assert result_1 == 'Error1'
        assert result_2 == 'Error1'
        assert result_3 == 'Error3'
        assert result_4 is None

    @patch('sdmetrics.reports.single_table._properties.column_pair_trends.'
           'ContingencySimilarity.compute_breakdown')
    @patch('sdmetrics.reports.single_table._properties.column_pair_trends.'
           'CorrelationSimilarity.compute_breakdown')
    def test__generate_details(self, correlation_compute_mock, contingency_compute_mock):
        """Test the ``_generate_details`` method."""
        # Setup
        real_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [False, True, True],
            'col3': ['a', 'b', 'c'],
            'col4': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
        })

        synthetic_data = pd.DataFrame({
            'col1': [3, 1, 2],
            'col2': [False, False, True],
            'col3': ['a', 'b', 'b'],
            'col4': pd.to_datetime(['2020-01-03', '2020-01-01', '2020-01-02']),
        })

        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'boolean'},
                'col3': {'sdtype': 'categorical'},
                'col4': {'sdtype': 'datetime'},
            }
        }

        processed_real = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [False, True, True],
            'col3': ['a', 'b', 'c'],
            'col4': [1577836800000000000, 1577923200000000000, 1578009600000000000],
            'col1_discrete': [1, 6, 11],
            'col4_discrete': [1, 6, 11],
        })

        processed_synthetic = pd.DataFrame({
            'col1': [3, 1, 2],
            'col2': [False, False, True],
            'col3': ['a', 'b', 'b'],
            'col4': [1578009600000000000, 1577836800000000000, 1577923200000000000],
            'col1_discrete': [11, 1, 6],
            'col4_discrete': [11, 1, 6],
        })

        cpt_property = ColumnPairTrends()

        # Run
        cpt_property._generate_details(real_data, synthetic_data, metadata, None)

        # Assert
        _, correlation_kwargs = correlation_compute_mock.call_args
        assert correlation_kwargs['real_data'].equals(processed_real[['col1', 'col4']])
        assert correlation_kwargs['synthetic_data'].equals(processed_synthetic[['col1', 'col4']])

        expected_real_data = [
            processed_real[['col1_discrete', 'col2']],
            processed_real[['col1_discrete', 'col3']],
            processed_real[['col2', 'col3']],
            processed_real[['col2', 'col4_discrete']],
            processed_real[['col3', 'col4_discrete']],
        ]
        expected_synthetic_data = [
            processed_synthetic[['col1_discrete', 'col2']],
            processed_synthetic[['col1_discrete', 'col3']],
            processed_synthetic[['col2', 'col3']],
            processed_synthetic[['col2', 'col4_discrete']],
            processed_synthetic[['col3', 'col4_discrete']],
        ]
        for idx, call1 in enumerate(contingency_compute_mock.call_args_list):
            _, contingency_kwargs = call1
            assert contingency_kwargs['real_data'].equals(expected_real_data[idx])
            assert contingency_kwargs['synthetic_data'].equals(expected_synthetic_data[idx])

    def test__get_correlation_matrix_score(self):
        """Test the ``_get_correlation_matrix`` method to generate the ``Score`` heatmap."""
        # Setup
        cpt_property = ColumnPairTrends()
        cpt_property._details = pd.DataFrame({
            'Column 1': ['col1', 'col1', 'col2'],
            'Column 2': ['col2', 'col3', 'col3'],
            'metric': ['CorrelationSimilarity', 'ContingencySimilarity', 'ContingencySimilarity'],
            'Score': [0.5, 0.6, 0.7],
        })

        # Run
        heatmap = cpt_property._get_correlation_matrix('Score')

        # Assert
        expected_heatmap = pd.DataFrame({
            'col1': [1, 0.5, 0.6],
            'col2': [0.5, 1, 0.7],
            'col3': [0.6, 0.7, 1],
        }, index=['col1', 'col2', 'col3'])

        pd.testing.assert_frame_equal(heatmap, expected_heatmap)

    def test__get_correlation_matrix_correlation(self):
        """Test the ``_get_correlation_matrix`` method to generate the ``Correlation`` heatmap."""
        # Setup
        cpt_property = ColumnPairTrends()
        cpt_property._details = pd.DataFrame({
            'Column 1': ['col1', 'col1', 'col2'],
            'Column 2': ['col2', 'col3', 'col3'],
            'metric': ['CorrelationSimilarity', 'ContingencySimilarity', 'ContingencySimilarity'],
            'Score': [0.5, 0.6, 0.7],
            'Real Correlation': [0.3, None, None],
            'Synthetic Correlation': [0.4, None, None],
        })

        # Run
        heatmap_real = cpt_property._get_correlation_matrix('Real Correlation')
        heatmap_synthetic = cpt_property._get_correlation_matrix('Synthetic Correlation')

        # Assert
        expected_real_heatmap = pd.DataFrame({
            'col1': [1, 0.3],
            'col2': [0.3, 1],
        }, index=['col1', 'col2'])

        expected_synthetic_heatmap = pd.DataFrame({
            'col1': [1, 0.4],
            'col2': [0.4, 1],
        }, index=['col1', 'col2'])

        pd.testing.assert_frame_equal(heatmap_real, expected_real_heatmap)
        pd.testing.assert_frame_equal(heatmap_synthetic, expected_synthetic_heatmap)

    @patch('sdmetrics.reports.single_table._properties.column_pair_trends.make_subplots')
    def test_get_visualization(self, mock_make_subplots):
        """Test the ``get_visualization`` method."""
        # Setup
        cpt_property = ColumnPairTrends()

        fig_mock = Mock()
        fig_mock.add_trace = Mock()
        mock_make_subplots.return_value = fig_mock

        mock__get_correlation_matrix = Mock()
        cpt_property._get_correlation_matrix = mock__get_correlation_matrix

        mock_heatmap = Mock()
        cpt_property._get_heatmap = Mock(return_value=mock_heatmap)

        mock__update_layout = Mock()
        cpt_property._update_layout = mock__update_layout

        # Run
        result = cpt_property.get_visualization()

        # Assert
        assert mock__get_correlation_matrix.call_count == 3
        mock_make_subplots.assert_called()
        assert cpt_property._get_heatmap.call_count == 3
        assert fig_mock.add_trace.call_count == 3
        cpt_property._update_layout.assert_called_once_with(fig_mock)
        assert result == fig_mock
