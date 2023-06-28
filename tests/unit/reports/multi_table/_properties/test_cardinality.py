"""Test multi-table cardinality properties."""

from unittest.mock import Mock, patch

from sdmetrics.reports.multi_table._properties.cardinality import Cardinality


class TestCardinality:

    @patch('sdmetrics.reports.multi_table._properties.cardinality.CardinalityShapeSimilarity')
    def test_get_score(self, mock_cardinalityshapesimilarity):
        """Test the ``get_score`` function.

        Test that when given a ``progress_bar`` and relationships, this calls
        ``CardinalityShapeSimilarity`` compute breakdown for each relationship.
        """
        # Setup
        mock_cardinalityshapesimilarity.compute_breakdown.return_value = {
            ('users', 'sessions'): {'score': 1.0},
            ('sessions', 'transactions'): {'score': 0.25},
        }
        instance = Cardinality()
        progress_bar = Mock()
        metadata = {
            'relationships': [
                'users',
                'sessions'
            ]
        }

        # Run
        score = instance.get_score('real_data', 'synthetic_data', metadata, progress_bar)

        # Assert
        assert score == 0.625
        progress_bar.update.assert_called()
        assert progress_bar.update.call_count == 2
        progress_bar.close.assert_called_once_with()

    @patch('sdmetrics.reports.multi_table._properties.cardinality.CardinalityShapeSimilarity')
    def test_get_score_raises_errors(self, mock_cardinalityshapesimilarity):
        """Test the ``get_score`` function when CardinalityShapeSimilarity can't compute score."""
        # Setup
        mock_cardinalityshapesimilarity.compute_breakdown.side_effect = [
            ValueError('Users error'),
            ValueError('Sessions error')
        ]
        instance = Cardinality()
        progress_bar = Mock()
        metadata = {
            'relationships': [
                'users',
                'sessions'
            ]
        }

        # Run
        score = instance.get_score('real_data', 'synthetic_data', metadata, progress_bar)

        # Assert
        assert score == 0
        assert instance._details == {
            'Errors': {
                'users': 'Users error',
                'sessions': 'Sessions error'
            }
        }
        progress_bar.update.assert_called()
        assert progress_bar.update.call_count == 2
        progress_bar.close.assert_called_once_with()

    @patch('sdmetrics.reports.multi_table._properties.cardinality.get_table_relationships_plot')
    def test_get_visualization(self, mock_get_table_relationships_plot):
        """Test that ``get_visualization`` calls ``get_table_relationships_plot``.

        Test that ``get_visualization`` filters over the ``instance._details`` to get
        the tables that contain the table name and calls ``get_table_relationships_plot`` with
        the ``score_breakdowns``.
        """
        # Setup
        instance = Mock()
        instance._details = {
            ('users', 'sessions'): 1.,
            ('users', 'transactions'): 0.5
        }

        # Run
        result = Cardinality.get_visualization(instance, 'sessions')

        # Assert
        assert result == mock_get_table_relationships_plot.return_value
        expected_score_breakdwons = {
            'CardinalityShapeSimilarity': {
                ('users', 'sessions'): 1.
            }
        }
        mock_get_table_relationships_plot.assert_called_with(expected_score_breakdwons)
