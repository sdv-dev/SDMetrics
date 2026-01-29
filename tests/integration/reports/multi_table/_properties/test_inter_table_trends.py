from unittest.mock import Mock

import numpy as np
import pandas as pd
from tqdm import tqdm

from sdmetrics.demos import load_demo
from sdmetrics.reports.multi_table._properties import InterTableTrends


class TestInterTableTrends:
    def test_end_to_end(self):
        """Test ``ColumnPairTrends`` multi-table property end to end."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        inter_table_trends = InterTableTrends()

        # Run
        result = inter_table_trends.get_score(real_data, synthetic_data, metadata)

        # Assert
        assert result == 0.4416666666666666

    def test_with_progress_bar(self):
        """Test that the progress bar is correctly updated."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        inter_table_trends = InterTableTrends()
        num_iter = sum(
            len(metadata['tables'][relationship['parent_table_name']]['columns'])
            * len(metadata['tables'][relationship['child_table_name']]['columns'])
            for relationship in metadata['relationships']
        )

        progress_bar = tqdm(total=num_iter)
        mock_update = Mock()
        progress_bar.update = mock_update

        # Run
        result = inter_table_trends.get_score(real_data, synthetic_data, metadata, progress_bar)

        # Assert
        assert result == 0.4416666666666666
        assert mock_update.call_count == num_iter

    def test_real_correlation_threshold_filters_pairs(self):
        """Test that low-correlation pairs are excluded from the score."""
        # Setup
        real_data = {
            'parents': pd.DataFrame({
                'parent_id': [1, 2, 3, 4],
                'numerical': [1, 2, 3, 4],
            }),
            'children': pd.DataFrame({
                'child_id': [10, 11, 12, 13],
                'parent_id': [1, 2, 3, 4],
                'numerical': [1, -1, 1, -1],
            }),
        }
        metadata = {
            'tables': {
                'parents': {
                    'primary_key': 'parent_id',
                    'columns': {
                        'parent_id': {'sdtype': 'id'},
                        'numerical': {'sdtype': 'numerical'},
                    },
                },
                'children': {
                    'primary_key': 'child_id',
                    'columns': {
                        'child_id': {'sdtype': 'id'},
                        'parent_id': {'sdtype': 'id'},
                        'numerical': {'sdtype': 'numerical'},
                    },
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'parents',
                    'parent_primary_key': 'parent_id',
                    'child_table_name': 'children',
                    'child_foreign_key': 'parent_id',
                }
            ],
        }
        inter_table_trends = InterTableTrends()
        inter_table_trends.real_correlation_threshold = 0.5

        # Run
        score = inter_table_trends.get_score(real_data, real_data, metadata)

        # Assert
        assert np.isnan(score)
        assert inter_table_trends.details['Meets Threshold?'].sum() == 0
