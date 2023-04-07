from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from sdmetrics.multi_table.statistical import CardinalityShapeSimilarity


class TestCardinalityShapeSimilarity:

    def test_compute(self):
        """Test the ``compute`` method.

        Expect that an average of the column metric scores across all tables is returned.

        Setup:
        - ``compute_breakdown`` should return a nested mapping of table name to metric
          breakdown.

        Input:
        - real data
        - synthetic data

        Output:
        - The average metric score.
        """
        # Setup
        metric_breakdown = {
            ('tableA', 'tableB'): {'score': 0.9},
            ('tableB', 'tableC'): {'score': 0.7},
            ('tableA', 'tableD'): {'score': 0.7},
        }

        # Run
        with patch.object(
            CardinalityShapeSimilarity,
            'compute_breakdown',
            return_value=metric_breakdown,
        ):
            result = CardinalityShapeSimilarity.compute(
                real_data=Mock(),
                synthetic_data=Mock(),
                metadata=Mock(),
            )

        # Assert
        assert result == 2.3 / 3

    def test_compute_no_relationships(self):
        """Test the ``compute`` method when there are no relationships.

        Expect that a score of `numpy.nan` is returned, as the metric is not applicable.

        Setup:
        - ``compute_breakdown`` should return a score of `numpy.nan`.

        Input:
        - real data
        - synthetic data

        Output:
        - `numpy.nan`
        """
        # Setup
        metric_breakdown = {'score': np.nan}

        # Run
        with patch.object(
            CardinalityShapeSimilarity,
            'compute_breakdown',
            return_value=metric_breakdown,
        ):
            result = CardinalityShapeSimilarity.compute(
                real_data=Mock(),
                synthetic_data=Mock(),
                metadata=Mock(),
            )

        # Assert
        assert np.isnan(result)

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
        metadata = {
            'tables': {
                'tableA': {'columns': {'col1': {}}},
                'tableB': {
                    'columns': {
                        'col1': {},
                        'col2': {},
                    },
                },
                'tableC': {'columns': {'col2': {}}},
            },
            'relationships': [
                {
                    'parent_table_name': 'tableA',
                    'parent_primary_key': 'col1',
                    'child_table_name': 'tableB',
                    'child_foreign_key': 'col1'
                },
                {
                    'parent_table_name': 'tableB',
                    'parent_primary_key': 'col2',
                    'child_table_name': 'tableC',
                    'child_foreign_key': 'col2'
                }
            ]
        }
        real_data = {
            'tableA': pd.DataFrame({'col1': [1, 2, 3, 4, 5]}),
            'tableB': pd.DataFrame(
                {'col1': [1, 1, 2, 3, 3, 5], 'col2': ['a', 'b', 'c', 'd', 'e', 'f']}),
            'tableC': pd.DataFrame({'col2': ['a', 'b', 'c']}),
        }
        synthetic_data = {
            'tableA': pd.DataFrame({'col1': [1, 2, 3, 4, 5]}),
            'tableB': pd.DataFrame(
                {'col1': [1, 2, 4, 4, 3, 5], 'col2': ['a', 'b', 'c', 'd', 'e', 'f']}),
            'tableC': pd.DataFrame({'col2': ['a', 'b', 'd']}),
        }
        expected_metric_breakdown = {
            ('tableA', 'tableB'): {'score': 0.8},
            ('tableB', 'tableC'): {'score': 1.0},
        }

        # Run
        result = CardinalityShapeSimilarity.compute_breakdown(real_data, synthetic_data, metadata)

        # Assert
        assert result == expected_metric_breakdown

    def test_compute_breakdown_no_relationships(self):
        """Test the ``compute_breakdown`` method when there are no relationships.

        Expect that a breakdown score of `{'score': numpy.nan}` is returned, because the metric
        is not applicable.

        Setup:
        - ``metadata`` should contain no relationships.

        Input:
        - real data
        - synthetic data

        Output:
        - A breakdown with a score of `numpy.nan`
        """
        # Setup
        metadata = {
            'tables': {
                'tableA': {'columns': {'col1': {}}},
                'tableB': {'columns': {'col1': {}, 'col2': {}}},
                'tableC': {'columns': {'col2': {}}},
            },
        }
        real_data = {
            'tableA': pd.DataFrame({'col1': [1, 2, 3, 4, 5]}),
            'tableB': pd.DataFrame(
                {'col1': [1, 1, 2, 3, 3, 5], 'col2': ['a', 'b', 'c', 'd', 'e', 'f']}),
            'tableC': pd.DataFrame({'col2': ['a', 'b', 'c']}),
        }
        synthetic_data = {
            'tableA': pd.DataFrame({'col1': [1, 2, 3, 4, 5]}),
            'tableB': pd.DataFrame(
                {'col1': [1, 2, 4, 4, 3, 5], 'col2': ['a', 'b', 'c', 'd', 'e', 'f']}),
            'tableC': pd.DataFrame({'col2': ['a', 'b', 'd']}),
        }
        expected_metric_breakdown = {'score': np.nan}

        # Run
        result = CardinalityShapeSimilarity.compute_breakdown(real_data, synthetic_data, metadata)

        # Assert
        assert result == expected_metric_breakdown

    @patch('sdmetrics.multi_table.statistical.cardinality_shape_similarity.MultiTableMetric.'
           'normalize')
    def test_normalize(self, normalize_mock):
        """Test the ``normalize`` method.

        Expect that the inherited ``normalize`` method is called.

        Input:
        - Raw score

        Output:
        - The output of the inherited ``normalize`` method.
        """
        # Setup
        raw_score = 0.9

        # Run
        result = CardinalityShapeSimilarity.normalize(raw_score)

        # Assert
        normalize_mock.assert_called_once_with(raw_score)
        assert result == normalize_mock.return_value
