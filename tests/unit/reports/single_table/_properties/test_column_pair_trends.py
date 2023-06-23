from unittest.mock import Mock, patch, call
import itertools

import pandas as pd
import numpy as np

from sdmetrics.reports.single_table._properties.column_pair_trends import ColumnPairTrends


class TestColumnPairTrends:

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
        expected_datetime = pd.to_numeric(pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']))
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
        """Test the ``_get_metric`` method."""
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

    @patch('sdmetrics.reports.single_table._properties.column_pair_trends'
           '.CorrelationSimilarity.compute_breakdown')
    @patch('sdmetrics.reports.single_table._properties.column_pair_trends'
           '.ContingencySimilarity.compute_breakdown')
    def test__generate_details(self, contingency_compute_mock, correlation_compute_mock):
        """Test the ``_generate_details`` method."""
        # Setup
        real_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [False, True, True],
            'col3': ['a', 'b', 'c'],
            'col4': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
        })

        synthetic_data = pd.DataFrame({
            'col1': [4, 5, 6],
            'col2': [False, False, True],
            'col3': ['a', 'b', 'b'],
            'col4': pd.to_datetime(['2020-01-04', '2020-01-05', '2020-01-06']),
        })

        metadata = {'columns': {
            'col1': {'sdtype': 'numerical'},
            'col2': {'sdtype': 'boolean'},
            'col3': {'sdtype': 'categorical'},
            'col4': {'sdtype': 'datetime'},
        }}

        processed_real = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [False, True, True],
            'col3': ['a', 'b', 'c'],
            'col4': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
            'col1_discrete': [1, 2, 3],
            'col4_discrete': [0, 1, 2],
        })

        processed_synthetic = pd.DataFrame({
            'col1': [4, 5, 6],
            'col2': [False, False, True],
            'col3': ['a', 'b', 'b'],
            'col4': pd.to_datetime(['2020-01-04', '2020-01-05', '2020-01-06']),
            'col1_discrete': [4, 5, 6],
            'col4_discrete': [3, 4, 5],
        })

        cpt_property = ColumnPairTrends()

        mock_processed_data = Mock()
        cpt_property._get_processed_data = mock_processed_data

        mock_get_columns_data = Mock()
        cpt_property._get_columns_data = mock_get_columns_data


        # Run
        cpt_property._generate_details(real_data, synthetic_data, metadata, None)

        # Assert
        mock_processed_data.assert_has_calls(
            [call(real_data, metadata), call(synthetic_data, metadata)]
        )

        columns = ['col1', 'col2', 'col3', 'col4']
        list_calls = [
            call(col_name_1, col_name_2, processed_real, processed_synthetic, metadata)
            for col_name_1, col_name_2 in itertools.combinations(columns, 2)
        ]
        mock_get_columns_data.assert_has_calls(list_calls)

        
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
        cpt_property.get_visualization()

        # Assert
        assert mock__get_correlation_matrix.call_count == 3
        mock_make_subplots.assert_called()
        assert cpt_property._get_heatmap.call_count == 3
        assert fig_mock.add_trace.call_count == 3
        cpt_property._update_layout.assert_called()
