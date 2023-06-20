from unittest.mock import Mock, patch

import pandas as pd

from sdmetrics.reports.single_table._properties.column_pair_trends import ColumnPairTrends


class TestColumnPairTrends:

    def test__get_processed_data(self):
        """Test the ``_get_processed_data`` method."""
        # Setup
        real_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [False, True, True],
            'col3': ['a', 'b', 'c'],
            'col4': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
        })
        synthetic_data = pd.DataFrame({
            'col1': [4, 5, 6],
            'col2': [False, True, True],
            'col3': ['a', 'b', 'c'],
            'col4': pd.to_datetime(['2020-01-04', '2020-01-05', '2020-01-06'])
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
        processed_real, processed_synthetic = cpt_property._get_processed_data(
            real_data, synthetic_data, metadata
        )

        # Assert
        expected_processed_real = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [False, True, True],
            'col3': ['a', 'b', 'c'],
            'col4': pd.to_numeric(pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])),
            'col1_discrete': [0, 1, 3],
            'col4_discrete': [0, 1, 3],
        })
        expected_processed_synthetic = pd.DataFrame({
            'col1': [4, 5, 6],
            'col2': [False, True, True],
            'col3': ['a', 'b', 'c'],
            'col4': pd.to_numeric(
                pd.to_datetime(['2020-01-04', '2020-01-05', '2020-01-06']).to_numpy()
            ),
            'col1_discrete': [5, 7, 9],
            'col4_discrete': [5, 7, 9],
        })

        pd.testing.assert_frame_equal(processed_real, expected_processed_real)
        pd.testing.assert_frame_equal(processed_synthetic, expected_processed_synthetic)

    def test__get_metric_and_columns_continous(self):
        """Test the ``_get_metric_and_columns`` method for continuous columns."""
        # Setup
        real_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
        })
        synthetic_data = pd.DataFrame({
            'col1': [4, 5, 6],
            'col2': pd.to_datetime(['2020-01-04', '2020-01-05', '2020-01-06'])
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'datetime'}
            }
        }
        cpt_property = ColumnPairTrends()

        # Run
        metric, columns_real, columns_synthetic = cpt_property._get_metric_and_columns(
            'col1', 'col2', real_data, synthetic_data, metadata
        )

        # Assert
        assert metric.__name__ == 'CorrelationSimilarity'
        pd.testing.assert_frame_equal(columns_real, real_data)
        pd.testing.assert_frame_equal(columns_synthetic, synthetic_data)

    def test__get_metric_and_columns_discrete(self):
        """Test the ``_get_metric_and_columns`` method for discrete columns."""
        # Setup
        real_data = pd.DataFrame({
            'col1': [True, False, False],
            'col2': ['A', 'B', 'C']
        })
        synthetic_data = pd.DataFrame({
            'col1': [True, True, False],
            'col2': ['B', 'B', 'C']
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'boolean'},
                'col2': {'sdtype': 'categorical'}
            }
        }
        cpt_property = ColumnPairTrends()

        # Run
        metric, columns_real, columns_synthetic = cpt_property._get_metric_and_columns(
            'col1', 'col2', real_data, synthetic_data, metadata
        )

        # Assert
        assert metric.__name__ == 'ContingencySimilarity'
        pd.testing.assert_frame_equal(columns_real, real_data)
        pd.testing.assert_frame_equal(columns_synthetic, synthetic_data)

    def test__get_metric_and_columns_mixed(self):
        """Test the ``_get_metric_and_columns`` method for mixed columns."""
        # Setup
        real_data = pd.DataFrame({
            'col1': [True, False, False],
            'col2': [1, 2, 3],
            'col2_discrete': [1, 2, 3]
        })
        synthetic_data = pd.DataFrame({
            'col1': [True, True, False],
            'col2': [2, 2, 3],
            'col2_discrete': [2, 2, 3]
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'boolean'},
                'col2': {'sdtype': 'numerical'}
            }
        }
        cpt_property = ColumnPairTrends()

        # Run
        metric, columns_real, columns_synthetic = cpt_property._get_metric_and_columns(
            'col1', 'col2', real_data, synthetic_data, metadata
        )

        # Assert
        assert metric.__name__ == 'ContingencySimilarity'
        pd.testing.assert_frame_equal(columns_real, real_data[['col1', 'col2_discrete']])
        pd.testing.assert_frame_equal(columns_synthetic, synthetic_data[['col1', 'col2_discrete']])

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
            'col4': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
        })
        synthetic_data = pd.DataFrame({
            'col1': [4, 5, 6],
            'col2': [False, False, True],
            'col3': ['a', 'b', 'b'],
            'col4': pd.to_datetime(['2020-01-04', '2020-01-05', '2020-01-06'])
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'boolean'},
                'col3': {'sdtype': 'categorical'},
                'col4': {'sdtype': 'datetime'}
            }
        }

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

        column_shape_cpt_property = ColumnPairTrends()

        mock_processed_data = Mock(return_value=(processed_real, processed_synthetic))
        column_shape_cpt_property._get_processed_data = mock_processed_data

        # Run
        column_shape_cpt_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        mock_processed_data.assert_called_once_with(real_data, synthetic_data, metadata)

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
        for idx, call in enumerate(contingency_compute_mock.call_args_list):
            _, contingency_kwargs = call
            assert contingency_kwargs['real_data'].equals(expected_real_data[idx])
            assert contingency_kwargs['synthetic_data'].equals(expected_synthetic_data[idx])

    def test__get_correlation_matrix(self):
        pass

    def test__get_visualization(self):
        pass
