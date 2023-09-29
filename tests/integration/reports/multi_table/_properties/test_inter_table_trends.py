from unittest.mock import Mock

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
        assert result == 0.48240740740740734

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
        assert result == 0.48240740740740734
        assert mock_update.call_count == num_iter
