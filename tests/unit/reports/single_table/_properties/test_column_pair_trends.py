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
            'col6': ['error', '2020-01-02', '2020-01-03'],
        })

        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'boolean'},
                'col3': {'sdtype': 'categorical'},
                'col4': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                'col5': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                'col6': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            }
        }
        cpt_property = ColumnPairTrends()

        # Run
        cpt_property._convert_datetime_columns_to_numeric(data, metadata)

        # Assert
        assert data['col4'].dtype == np.int64
        assert data['col5'].dtype == np.float64
        assert 'col6' in list(cpt_property._columns_datetime_conversion_failed.keys())

    def test__discretize_column(self):
        """Test the ``_discretize_column`` method."""
        # Setup
        data = pd.DataFrame({
            'err_col': ['a', 'b', 'c', 'd', 'e'],
            'int_col': [1, 2, 3, None, 5],
            'float_col': [1.1, np.nan, 3.3, 4.4, 5.5],
        })
        bin_edges = None
        cpt_property = ColumnPairTrends()

        # Run
        _, bin_edges = cpt_property._discretize_column('err_col', data['err_col'])
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
        data = pd.DataFrame(
            {
                'col1': [1, 2, 3],
                'col2': [False, True, True],
                'col3': ['a', 'b', 'c'],
                'col4': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
            },
            index=[4, 5, 6],
        )
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'boolean'},
                'col3': {'sdtype': 'categorical'},
                'col4': {'sdtype': 'datetime'},
            }
        }

        # Run
        cpt_property = ColumnPairTrends()
        processed_data, discrete_data = cpt_property._get_processed_data(data, metadata)

        # Assert
        expected_datetime = pd.to_numeric(
            pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
        )
        expected_processed_data = pd.DataFrame(
            {
                'col1': [1, 2, 3],
                'col2': [False, True, True],
                'col3': ['a', 'b', 'c'],
                'col4': expected_datetime,
            },
            index=[4, 5, 6],
        )

        expected_discrete_data = pd.DataFrame(
            {
                'col1': [1, 6, 11],
                'col4': [1, 6, 11],
            },
            index=[4, 5, 6],
        )

        pd.testing.assert_frame_equal(processed_data, expected_processed_data)
        pd.testing.assert_frame_equal(discrete_data, expected_discrete_data)

    def test__get_processed_data_with_nans(self):
        """Test the ``_get_processed_data`` method."""
        # Setup
        data = pd.DataFrame({
            'col1': [None, 2, 3],
            'col2': [False, np.nan, True],
            'col3': ['a', 'b', 'c'],
            'col4': pd.to_datetime(['2020-01-01', None, '2020-01-03']),
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'boolean'},
                'col3': {'sdtype': 'categorical'},
                'col4': {'sdtype': 'datetime'},
            }
        }

        # Run
        cpt_property = ColumnPairTrends()
        processed_data, discrete_data = cpt_property._get_processed_data(data, metadata)

        # Assert
        expected_datetime = pd.to_numeric(pd.to_datetime(['2020-01-01', None, '2020-01-03']))
        expected_datetime = pd.Series(expected_datetime)
        expected_datetime = expected_datetime.replace(-9223372036854775808, np.nan)

        expected_processed_data = pd.DataFrame({
            'col1': [None, 2, 3],
            'col2': [False, np.nan, True],
            'col3': ['a', 'b', 'c'],
            'col4': expected_datetime,
        })

        expected_discrete_data = pd.DataFrame({
            'col1': [11, 1, 11],
            'col4': [1, 11, 11],
        })

        pd.testing.assert_frame_equal(processed_data, expected_processed_data)
        pd.testing.assert_frame_equal(discrete_data, expected_discrete_data)

    def test__get_columns_data_and_metric(self):
        """Test the ``_get_columns_data_and_metric`` method.

        The method should return the correct data for each combination of column types.
        """
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

        discrete_real = pd.DataFrame({
            'col1': [1, 6, 11],
            'col4': [1, 6, 11],
        })

        discrete_synthetic = pd.DataFrame({
            'col1': [11, 1, 6],
            'col4': [11, 1, 6],
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
        expected_real_data_return = [
            pd.concat([discrete_real['col1'], real_data['col2']], axis=1),
            pd.concat([discrete_real['col1'], real_data['col3']], axis=1),
            real_data[['col1', 'col4']],
            real_data[['col2', 'col3']],
            pd.concat([real_data['col2'], discrete_real['col4']], axis=1),
            pd.concat([real_data['col3'], discrete_real['col4']], axis=1),
        ]
        expected_synthetic_data_return = [
            pd.concat([discrete_synthetic['col1'], synthetic_data['col2']], axis=1),
            pd.concat([discrete_synthetic['col1'], synthetic_data['col3']], axis=1),
            synthetic_data[['col1', 'col4']],
            synthetic_data[['col2', 'col3']],
            pd.concat([synthetic_data['col2'], discrete_synthetic['col4']], axis=1),
            pd.concat([synthetic_data['col3'], discrete_synthetic['col4']], axis=1),
        ]
        expected_metric_return = [
            'ContingencySimilarity',
            'ContingencySimilarity',
            'CorrelationSimilarity',
            'ContingencySimilarity',
            'ContingencySimilarity',
            'ContingencySimilarity',
        ]
        for idx, (col1, col2) in enumerate(itertools.combinations(metadata['columns'], 2)):
            cols_real, cols_synthetic, metric = cpt_property._get_columns_data_and_metric(
                col1, col2, real_data, discrete_real, synthetic_data, discrete_synthetic, metadata
            )
            pd.testing.assert_frame_equal(cols_real, expected_real_data_return[idx])
            pd.testing.assert_frame_equal(cols_synthetic, expected_synthetic_data_return[idx])
            assert metric.__name__ == expected_metric_return[idx]

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

    @patch(
        'sdmetrics.reports.single_table._properties.column_pair_trends.'
        'ContingencySimilarity.compute_breakdown'
    )
    @patch(
        'sdmetrics.reports.single_table._properties.column_pair_trends.'
        'CorrelationSimilarity.compute_breakdown'
    )
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
        })

        discrete_real = pd.DataFrame({
            'col1': [1, 6, 11],
            'col4': [1, 6, 11],
        })

        processed_synthetic = pd.DataFrame({
            'col1': [3, 1, 2],
            'col2': [False, False, True],
            'col3': ['a', 'b', 'b'],
            'col4': [1578009600000000000, 1577836800000000000, 1577923200000000000],
        })

        discrete_synthetic = pd.DataFrame({
            'col1': [11, 1, 6],
            'col4': [11, 1, 6],
        })

        cpt_property = ColumnPairTrends()

        # Run
        cpt_property._generate_details(real_data, synthetic_data, metadata, None)

        # Assert
        _, correlation_kwargs = correlation_compute_mock.call_args
        assert correlation_kwargs['real_data'].equals(processed_real[['col1', 'col4']])
        assert correlation_kwargs['synthetic_data'].equals(processed_synthetic[['col1', 'col4']])

        expected_real_data = [
            pd.concat([discrete_real['col1'], processed_real['col2']], axis=1),
            pd.concat([discrete_real['col1'], processed_real['col3']], axis=1),
            processed_real[['col2', 'col3']],
            pd.concat([processed_real['col2'], discrete_real['col4']], axis=1),
            pd.concat([processed_real['col3'], discrete_real['col4']], axis=1),
        ]
        expected_synthetic_data = [
            pd.concat([discrete_synthetic['col1'], processed_synthetic['col2']], axis=1),
            pd.concat([discrete_synthetic['col1'], processed_synthetic['col3']], axis=1),
            processed_synthetic[['col2', 'col3']],
            pd.concat([processed_synthetic['col2'], discrete_synthetic['col4']], axis=1),
            pd.concat([processed_synthetic['col3'], discrete_synthetic['col4']], axis=1),
        ]
        for idx, call1 in enumerate(contingency_compute_mock.call_args_list):
            _, contingency_kwargs = call1
            assert contingency_kwargs.keys() == {'real_data', 'synthetic_data'}
            assert contingency_kwargs['real_data'].equals(expected_real_data[idx])
            assert contingency_kwargs['synthetic_data'].equals(expected_synthetic_data[idx])

    @patch(
        'sdmetrics.reports.single_table._properties.column_pair_trends.'
        'ContingencySimilarity.compute_breakdown'
    )
    def test__generate_details_large_dataset(self, contingency_compute_mock):
        """Test the ``_generate_details`` for data with more than 50000 rows."""
        # Setup
        real_data = pd.DataFrame({
            'col1': ['a', 'b', 'c'] * 20000,
            'col2': [False, True, True] * 20000,
        })
        synthetic_data = pd.DataFrame({
            'col1': ['c', 'a', 'b'] * 20000,
            'col2': [False, False, True] * 20000,
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'categorical'},
                'col2': {'sdtype': 'boolean'},
            }
        }

        cpt_property = ColumnPairTrends()

        # Run
        cpt_property._generate_details(real_data, synthetic_data, metadata, None)

        # Assert
        contingency_kwargs = contingency_compute_mock.call_args_list[0][1]
        pd.testing.assert_frame_equal(contingency_kwargs['real_data'], real_data)
        pd.testing.assert_frame_equal(contingency_kwargs['synthetic_data'], synthetic_data)
        assert contingency_kwargs['num_rows_subsample'] == 50000

    def test__get_correlation_matrix_score(self):
        """Test the ``_get_correlation_matrix`` method to generate the ``Score`` heatmap."""
        # Setup
        cpt_property = ColumnPairTrends()
        cpt_property.details = pd.DataFrame({
            'Column 1': ['col1', 'col1', 'col2'],
            'Column 2': ['col2', 'col3', 'col3'],
            'Metric': ['CorrelationSimilarity', 'ContingencySimilarity', 'ContingencySimilarity'],
            'Score': [0.5, 0.6, 0.7],
        })

        # Run
        heatmap = cpt_property._get_correlation_matrix('Score')

        # Assert
        expected_heatmap = pd.DataFrame(
            {
                'col1': [1, 0.5, 0.6],
                'col2': [0.5, 1, 0.7],
                'col3': [0.6, 0.7, 1],
            },
            index=['col1', 'col2', 'col3'],
        )

        pd.testing.assert_frame_equal(heatmap, expected_heatmap)

    def test__get_correlation_matrix_correlation(self):
        """Test the ``_get_correlation_matrix`` method to generate the ``Correlation`` heatmap."""
        # Setup
        cpt_property = ColumnPairTrends()
        cpt_property.details = pd.DataFrame({
            'Column 1': ['col1', 'col1', 'col2'],
            'Column 2': ['col2', 'col3', 'col3'],
            'Metric': ['CorrelationSimilarity', 'ContingencySimilarity', 'ContingencySimilarity'],
            'Score': [0.5, 0.6, 0.7],
            'Real Correlation': [0.3, None, None],
            'Synthetic Correlation': [0.4, None, None],
        })

        # Run
        heatmap_real = cpt_property._get_correlation_matrix('Real Correlation')
        heatmap_synthetic = cpt_property._get_correlation_matrix('Synthetic Correlation')

        # Assert
        expected_real_heatmap = pd.DataFrame(
            {
                'col1': [1, 0.3],
                'col2': [0.3, 1],
            },
            index=['col1', 'col2'],
        )

        expected_synthetic_heatmap = pd.DataFrame(
            {
                'col1': [1, 0.4],
                'col2': [0.4, 1],
            },
            index=['col1', 'col2'],
        )

        pd.testing.assert_frame_equal(heatmap_real, expected_real_heatmap)
        pd.testing.assert_frame_equal(heatmap_synthetic, expected_synthetic_heatmap)

    def test__get_correlation_matrix_score_drops_empty_columns(self):
        """Test that empty score columns are removed from the score heatmap."""
        # Setup
        cpt_property = ColumnPairTrends()
        cpt_property.details = pd.DataFrame({
            'Column 1': ['col1', 'col2'],
            'Column 2': ['col2', 'col3'],
            'Metric': ['CorrelationSimilarity', 'CorrelationSimilarity'],
            'Score': [0.5, np.nan],
            'Error': [None, None],
        })

        # Run
        heatmap = cpt_property._get_correlation_matrix('Score')

        # Assert
        expected_heatmap = pd.DataFrame(
            {
                'col1': [1, 0.5],
                'col2': [0.5, 1],
            },
            index=['col1', 'col2'],
        )
        pd.testing.assert_frame_equal(heatmap, expected_heatmap)

    @patch(
        'sdmetrics.reports.single_table._properties.column_pair_trends.'
        'CorrelationSimilarity.compute_breakdown'
    )
    def test__generate_details_real_correlation_threshold(self, correlation_compute_mock):
        """Test that real correlation thresholds set contribution flags."""
        # Setup
        real_data = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [1, 2, 3, 4]})
        synthetic_data = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [1, 2, 3, 4]})
        metadata = {'columns': {'col1': {'sdtype': 'numerical'}, 'col2': {'sdtype': 'numerical'}}}
        correlation_compute_mock.return_value = {
            'score': 0.8,
            'real': 0.2,
            'synthetic': 0.1,
        }
        cpt_property = ColumnPairTrends()
        cpt_property.real_correlation_threshold = 0.3

        # Run
        details = cpt_property._generate_details(real_data, synthetic_data, metadata, None)

        # Assert
        expected_details = pd.DataFrame({
            'Column 1': ['col1'],
            'Column 2': ['col2'],
            'Metric': ['CorrelationSimilarity'],
            'Score': [0.8],
            'Real Correlation': [0.2],
            'Synthetic Correlation': [0.1],
            'Real Association': [np.nan],
            'Meets Threshold?': pd.Series([False], dtype='boolean'),
        })
        pd.testing.assert_frame_equal(details, expected_details)
        correlation_compute_mock.assert_called_once()

    @patch(
        'sdmetrics.reports.single_table._properties.column_pair_trends.'
        'ContingencySimilarity.compute_breakdown'
    )
    def test__generate_details_real_association_threshold(self, contingency_compute_mock):
        """Test that real association thresholds set contribution flags."""
        # Setup
        real_data = pd.DataFrame({'col1': ['A', 'A', 'B', 'B'], 'col2': ['X', 'Y', 'X', 'Y']})
        metadata = {
            'columns': {'col1': {'sdtype': 'categorical'}, 'col2': {'sdtype': 'categorical'}}
        }
        contingency_compute_mock.return_value = {
            'score': np.nan,
            'real_association': 0.2,
        }
        cpt_property = ColumnPairTrends()
        cpt_property.real_association_threshold = 0.3

        # Run
        details = cpt_property._generate_details(real_data, real_data, metadata, None)

        # Assert
        expected_details = pd.DataFrame({
            'Column 1': ['col1'],
            'Column 2': ['col2'],
            'Metric': ['ContingencySimilarity'],
            'Score': [np.nan],
            'Real Correlation': [np.nan],
            'Synthetic Correlation': [np.nan],
            'Real Association': [0.2],
            'Meets Threshold?': pd.Series([False], dtype='boolean'),
        })
        pd.testing.assert_frame_equal(details, expected_details)
        _, contingency_kwargs = contingency_compute_mock.call_args
        assert contingency_kwargs['real_association_threshold'] == 0.3

    @patch('sdmetrics.reports.single_table._properties.column_pair_trends.make_subplots')
    def test_get_visualization(self, mock_make_subplots):
        """Test the ``get_visualization`` method."""
        # Setup
        cpt_property = ColumnPairTrends()

        fig_mock = Mock()
        fig_mock.add_trace = Mock()
        fig_mock.update_xaxes = Mock()
        mock_make_subplots.return_value = fig_mock

        similarity_correlation = pd.DataFrame(
            {'col1': [1, 0.5], 'col2': [0.5, 1]}, index=['col1', 'col2']
        )
        real_correlation = pd.DataFrame(
            {'col1': [1, 0.2], 'col2': [0.2, 1]}, index=['col1', 'col2']
        )
        synthetic_correlation = pd.DataFrame(
            {'col1': [1, 0.3], 'col2': [0.3, 1]}, index=['col1', 'col2']
        )
        cpt_property._get_correlation_matrix = Mock(
            side_effect=[similarity_correlation, real_correlation, synthetic_correlation]
        )

        mock_heatmap = Mock()
        cpt_property._get_heatmap = Mock(return_value=[mock_heatmap])

        mock__update_layout = Mock()
        cpt_property._update_layout = mock__update_layout

        # Run
        result = cpt_property.get_visualization()

        # Assert
        assert cpt_property._get_correlation_matrix.call_count == 3
        mock_make_subplots.assert_called()
        assert cpt_property._get_heatmap.call_count == 3
        assert fig_mock.add_trace.call_count == 3
        cpt_property._update_layout.assert_called_once_with(fig_mock)
        assert result == fig_mock

    @patch('sdmetrics.reports.single_table._properties.column_pair_trends.make_subplots')
    def test_get_visualization_without_correlations(self, mock_make_subplots):
        """Test the ``get_visualization`` method without numerical correlations."""
        # Setup
        cpt_property = ColumnPairTrends()
        cpt_property.details = pd.DataFrame({
            'Column 1': ['col1'],
            'Column 2': ['col2'],
            'Metric': ['ContingencySimilarity'],
            'Score': [0.75],
        })

        fig_mock = Mock()
        fig_mock.add_trace = Mock()
        fig_mock.update_xaxes = Mock()
        mock_make_subplots.return_value = fig_mock

        mock_heatmap = Mock()
        cpt_property._get_heatmap = Mock(return_value=[mock_heatmap])

        mock__update_layout = Mock()
        cpt_property._update_layout = mock__update_layout

        # Run
        result = cpt_property.get_visualization()

        # Assert
        mock_make_subplots.assert_called_with(
            rows=1, cols=1, subplot_titles=['Real vs. Synthetic Similarity']
        )
        assert fig_mock.add_trace.call_count == 1
        cpt_property._update_layout.assert_called_once_with(fig_mock, show_correlations=False)
        assert result == fig_mock

    def test_get_visualization_empty(self):
        """Test the ``get_visualization`` method when no scores are available."""
        # Setup
        cpt_property = ColumnPairTrends()
        cpt_property.details = pd.DataFrame({
            'Column 1': ['col1'],
            'Column 2': ['col2'],
            'Metric': ['CorrelationSimilarity'],
            'Score': [np.nan],
        })

        # Run
        fig = cpt_property.get_visualization()

        # Assert
        assert fig.data == ()

    def test_get_visualization_layout_alignment(self):
        """Test layout settings that keep subplots aligned."""
        # Setup
        cpt_property = ColumnPairTrends()
        cpt_property.details = pd.DataFrame({
            'Column 1': ['col1'],
            'Column 2': ['col2'],
            'Metric': ['CorrelationSimilarity'],
            'Score': [0.75],
            'Real Correlation': [0.5],
            'Synthetic Correlation': [0.6],
            'Real Association': [0.2],
            'Meets Threshold?': pd.Series([True], dtype='boolean'),
        })

        # Run
        fig = cpt_property.get_visualization()

        # Assert
        assert fig.layout.height == 900
        assert fig.layout.width == 900
        assert fig.layout.xaxis3.matches == 'x2'
        assert fig.layout.yaxis3.matches == 'y2'
        assert fig.layout.yaxis3.visible is False
        assert fig.layout.coloraxis.cmin == 0
        assert fig.layout.coloraxis.cmax == 1
        assert fig.layout.coloraxis.colorbar.x == 0.8
        assert fig.layout.coloraxis.colorbar.y == 0.8
        assert fig.layout.coloraxis2.cmin == -1
        assert fig.layout.coloraxis2.cmax == 1
        assert fig.layout.coloraxis2.colorbar.y == 0.2
