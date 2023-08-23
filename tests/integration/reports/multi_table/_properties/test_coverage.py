from unittest.mock import Mock

from tqdm import tqdm

from sdmetrics.demos import load_demo
from sdmetrics.reports.multi_table._properties import Coverage


class TestCoverage:

    def test_end_to_end(self):
        """Test the ``Coverage`` multi-table property end to end."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        coverage = Coverage()

        # Run
        result = coverage.get_score(real_data, synthetic_data, metadata)

        # Assert
        assert result == 0.9573447196980541

    def test_with_progress_bar(self):
        """Test that the progress bar is correctly updated."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        coverage = Coverage()
        num_columns = sum(len(table['columns']) for table in metadata['tables'].values())

        progress_bar = tqdm(total=num_columns)
        mock_update = Mock()
        progress_bar.update = mock_update

        # Run
        result = coverage.get_score(real_data, synthetic_data, metadata, progress_bar)

        # Assert
        assert result == 0.9573447196980541
        assert mock_update.call_count == num_columns
