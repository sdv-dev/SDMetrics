from unittest.mock import Mock

from tqdm import tqdm

from sdmetrics.demos import load_demo
from sdmetrics.reports.multi_table._properties import DataValidity


def _cast_datetime_to_string(data, metadata):
    data = data.copy()
    for table_name in metadata['tables']:
        for column, column_meta in metadata['tables'][table_name]['columns'].items():
            sdtype = column_meta.get('sdtype')
            if sdtype == 'datetime' and column_meta.get('datetime_format') is not None:
                data[table_name][column] = (
                    data[table_name][column].astype(str).replace('NaT', None)
                )
            if sdtype == 'id' and column_meta.get('regex_format') is not None:
                data[table_name][column] = (
                    data[table_name][column].astype(str)
                )  

    return data

class TestDataValidity:
    def test_end_to_end(self):
        """Test the ``DataValidity`` multi-table property end to end."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        real_data = _cast_datetime_to_string(real_data, metadata)
        synthetic_data = _cast_datetime_to_string(synthetic_data, metadata)
        
        column_shapes = DataValidity()

        # Run
        result = column_shapes.get_score(real_data, synthetic_data, metadata)

        # Assert
        assert result == 1.0

    def test_with_progress_bar(self):
        """Test that the progress bar is correctly updated."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        real_data = _cast_datetime_to_string(real_data, metadata)
        synthetic_data = _cast_datetime_to_string(synthetic_data, metadata)
        
        column_shapes = DataValidity()
        num_columns = sum(len(table['columns']) for table in metadata['tables'].values()) + 1

        progress_bar = tqdm(total=num_columns)
        mock_update = Mock()
        progress_bar.update = mock_update

        # Run
        result = column_shapes.get_score(real_data, synthetic_data, metadata, progress_bar)

        # Assert
        assert result == 1.0
        assert mock_update.call_count == num_columns
