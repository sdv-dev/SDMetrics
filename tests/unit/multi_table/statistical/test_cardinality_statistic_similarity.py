from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from sdmetrics.multi_table.statistical import CardinalityStatisticSimilarity
from sdmetrics.warnings import ConstantInputWarning


class TestCardinalityStatisticSimilarity:

    def test__compute_statistic(self):
        """Test the ``_compute_statistic`` method.

        Expect that a metric breakdown using the desired statistic is returned.

        Input:
        - real distribution
        - synthetic distribution

        Output:
        - A metric breakdown, containing 'score'.
        """
        # Setup
        real_distribution = pd.Series([1, 2, 2, 5, 1])
        synthetic_distribution = pd.Series([2, 2, 3, 1, 4])

        # Run
        result = CardinalityStatisticSimilarity._compute_statistic(
            real_distribution, synthetic_distribution, 'mean')

        # Assert
        assert result == {'real': 2.2, 'synthetic': 2.4, 'score': 0.9500000000000001}

    def test__compute_statistic_constant_input(self):
        """Test the ``_compute_statistic`` method with constant input.

        Expect that a warning is returned.

        Input:
        - real distribution
        - synthetic distribution

        Output:
        - A metric breakdown, containing 'score'.

        Side Effects:
        - A ``ConstantInputWarning`` is thrown.
        """
        # Setup
        real_distribution = pd.Series([1, 1, 1, 1])
        synthetic_distribution = pd.Series([2, 2, 3, 1])
        expected_warn_msg = (
            'One or more columns of the real data input is constant. '
            'The CardinalityStatisticSimilarity metric is either undefined or infinite '
            'for those columns.'
        )

        # Run
        with np.testing.assert_warns(ConstantInputWarning, match=expected_warn_msg):
            result = CardinalityStatisticSimilarity._compute_statistic(
                real_distribution, synthetic_distribution, 'mean')

        # Assert
        assert result == {'score': np.nan}

    def test_compute(self):
        """Test the ``compute`` method.

        Expect that an average of the column metric scores across all tables is returned.

        Setup:
        - ``_compute`` helper method should return a nested mapping of table name to metric
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
            CardinalityStatisticSimilarity,
            'compute_breakdown',
            return_value=metric_breakdown,
        ):
            result = CardinalityStatisticSimilarity.compute(
                real_data=Mock(),
                synthetic_data=Mock(),
                metadata=Mock(),
                statistic='mean',
            )

        # Assert
        assert result == 2.3 / 3

    def test_compute_no_relationships(self):
        """Test the ``compute`` method when there are no relationships.

        Expect that a score of `numpy.nan` is returned, as the metric is not applicable.

        Setup:
        - ``_compute`` helper method should return a score of `numpy.nan`.

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
            CardinalityStatisticSimilarity,
            'compute_breakdown',
            return_value=metric_breakdown,
        ):
            result = CardinalityStatisticSimilarity.compute(
                real_data=Mock(),
                synthetic_data=Mock(),
                metadata=Mock(),
                statistic='mean',
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
                'tableA': {'fields': {'col1': {}}},
                'tableB': {
                    'fields': {
                        'col1': {'ref': {'table': 'tableA', 'field': 'col1'}},
                        'col2': {},
                    },
                },
                'tableC': {'fields': {'col2': {'ref': {'table': 'tableB', 'field': 'col2'}}}},
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
        expected_metric_breakdown = {
            ('tableA', 'tableB'): {'score': 1.0, 'real': 1.2, 'synthetic': 1.2},
            ('tableB', 'tableC'): {'score': 1.0, 'real': 0.5, 'synthetic': 0.5},
        }

        # Run
        result = CardinalityStatisticSimilarity.compute_breakdown(
            real_data, synthetic_data, metadata, 'mean')

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
                'tableA': {'fields': {'col1': {}}},
                'tableB': {'fields': {'col1': {}, 'col2': {}}},
                'tableC': {'fields': {'col2': {}}},
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
        result = CardinalityStatisticSimilarity.compute_breakdown(
            real_data, synthetic_data, metadata, 'mean')

        # Assert
        assert result == expected_metric_breakdown

    @patch('sdmetrics.multi_table.statistical.cardinality_statistic_similarity.MultiTableMetric.'
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
        result = CardinalityStatisticSimilarity.normalize(raw_score)

        # Assert
        normalize_mock.assert_called_once_with(raw_score)
        assert result == normalize_mock.return_value
