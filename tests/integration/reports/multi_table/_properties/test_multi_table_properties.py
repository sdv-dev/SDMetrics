from unittest.mock import Mock

from tqdm import tqdm

from sdmetrics.demos import load_demo
from sdmetrics.reports.multi_table._properties import ColumnPairTrends, ColumnShapes, Coverage


def test_column_shapes_property():
    """Test ColumnShapes multi-table class."""
    # Setup
    real_data, synthetic_data, metadata = load_demo(modality='multi_table')
    column_shapes = ColumnShapes()

    # Run
    result = column_shapes.get_score(real_data, synthetic_data, metadata)

    # Assert
    assert result == 0.796825396825397


def test_column_shapes_property_with_progress_bar():
    """Test that the progress bar is correctly updated."""
    # Setup
    real_data, synthetic_data, metadata = load_demo(modality='multi_table')
    column_shapes = ColumnShapes()
    num_columns = sum(len(table['columns']) for table in metadata['tables'].values())

    progress_bar = tqdm(total=num_columns)
    mock_update = Mock()
    progress_bar.update = mock_update

    # Run
    result = column_shapes.get_score(real_data, synthetic_data, metadata, progress_bar)

    # Assert
    assert result == 0.796825396825397
    assert mock_update.call_count == num_columns


def test_column_pair_trends_property():
    """Test ColumnPairTrends multi-table class."""
    # Setup
    real_data, synthetic_data, metadata = load_demo(modality='multi_table')
    column_pair_trends = ColumnPairTrends()

    # Run
    result = column_pair_trends.get_score(real_data, synthetic_data, metadata)

    # Assert
    assert result == 0.49


def test_coverage_property():
    """Test Coverage multi-table class."""
    # Setup
    real_data, synthetic_data, metadata = load_demo(modality='multi_table')
    coverage = Coverage()

    # Run
    result = coverage.get_score(real_data, synthetic_data, metadata)

    # Assert
    assert result == 0.96


def test_coverage_property_with_progress_bar():
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
    assert result == 0.96
    assert mock_update.call_count == num_columns
