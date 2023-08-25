from unittest.mock import Mock

from tqdm import tqdm

from sdmetrics.demos import load_demo
from sdmetrics.reports.multi_table._properties import Synthesis


class TestSynthesis:

    def test_end_to_end(self):
        """Test Synthesis multi-table."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        synthesis = Synthesis()

        # Run
        result = synthesis.get_score(real_data, synthetic_data, metadata)

        # Assert
        assert result == 0.6333333333333333

    def test_with_progress_bar(self):
        """Test that the progress bar is correctly updated."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        synthesis = Synthesis()
        num_tables = len(metadata['tables'])

        progress_bar = tqdm(total=num_tables)
        mock_update = Mock()
        progress_bar.update = mock_update

        # Run
        result = synthesis.get_score(real_data, synthetic_data, metadata, progress_bar)

        # Assert
        assert result == 0.6333333333333333
        assert mock_update.call_count == num_tables
