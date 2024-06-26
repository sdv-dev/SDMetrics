from unittest.mock import Mock

import numpy as np
from tqdm import tqdm

from sdmetrics.demos import load_demo
from sdmetrics.reports.multi_table._properties import ColumnPairTrends


class TestColumnPairTrends:
    def test_end_to_end(self):
        """Test ``ColumnPairTrends`` multi-table property end to end."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        column_pair_trends = ColumnPairTrends()

        # Run
        result = column_pair_trends.get_score(real_data, synthetic_data, metadata)

        # Assert
        assert np.isclose(result, 0.45654629583521095, atol=1e-8)

    def test_with_progress_bar(self):
        """Test that the progress bar is correctly updated."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        column_pair_trends = ColumnPairTrends()
        num_iter = sum(
            int(0.5 * len(table['columns']) * (len(table['columns']) - 1))
            for table in metadata['tables'].values()
        )

        progress_bar = tqdm(total=num_iter)
        mock_update = Mock()
        progress_bar.update = mock_update

        # Run
        result = column_pair_trends.get_score(real_data, synthetic_data, metadata, progress_bar)

        # Assert
        assert np.isclose(result, 0.45654629583521095, atol=1e-8)
        assert mock_update.call_count == num_iter
