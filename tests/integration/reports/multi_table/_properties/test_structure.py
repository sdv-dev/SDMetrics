from unittest.mock import Mock

import pandas as pd
from tqdm import tqdm

from sdmetrics.demos import load_demo
from sdmetrics.reports.multi_table._properties import Structure


class TestStructure:

    def test_end_to_end(self):
        """Test Structure multi-table."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        structure = Structure()

        # Run
        result = structure.get_score(real_data, synthetic_data, metadata)

        # Assert
        assert result == 1.0

        expected_details = pd.DataFrame({
            'Table': ['users', 'sessions', 'transactions'],
            'Metric': ['TableStructure', 'TableStructure', 'TableStructure'],
            'Score': [1.0, 1.0, 1.0],
        })
        pd.testing.assert_frame_equal(structure.details, expected_details)

    def test_with_progress_bar(self):
        """Test that the progress bar is correctly updated."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        structure = Structure()
        num_tables = len(metadata['tables'])

        progress_bar = tqdm(total=num_tables)
        mock_update = Mock()
        progress_bar.update = mock_update

        # Run
        result = structure.get_score(real_data, synthetic_data, metadata, progress_bar)

        # Assert
        assert result == 1.0
        assert mock_update.call_count == num_tables
