from unittest.mock import Mock

from tqdm import tqdm

from sdmetrics.reports.multi_table._properties import Coverage


class TestCoverage:
    def test_end_to_end(self, converted_datetime_multi_table_demo):
        """Test the ``Coverage`` multi-table property end to end."""
        # Setup
        real_data, synthetic_data, metadata = converted_datetime_multi_table_demo
        coverage = Coverage()

        # Run
        result = coverage.get_score(real_data, synthetic_data, metadata)

        # Assert
        assert result == 0.8244218804937835

    def test_with_progress_bar(self, converted_datetime_multi_table_demo):
        """Test that the progress bar is correctly updated."""
        # Setup
        real_data, synthetic_data, metadata = converted_datetime_multi_table_demo
        coverage = Coverage()
        num_columns = sum(len(table['columns']) for table in metadata['tables'].values())

        progress_bar = tqdm(total=num_columns)
        mock_update = Mock()
        progress_bar.update = mock_update

        # Run
        result = coverage.get_score(real_data, synthetic_data, metadata, progress_bar)

        # Assert
        assert result == 0.8244218804937835
        assert mock_update.call_count == num_columns
