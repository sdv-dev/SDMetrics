from unittest.mock import Mock

from tqdm import tqdm

from sdmetrics.demos import load_demo
from sdmetrics.reports.multi_table._properties import (
    Boundary, ColumnPairTrends, ColumnShapes, Coverage, Synthesis)


def test_column_shapes_property():
    """Test the ``ColumnShapes`` multi-table property end to end."""
    # Setup
    real_data, synthetic_data, metadata = load_demo(modality='multi_table')
    column_shapes = ColumnShapes()

    # Run
    result = column_shapes.get_score(real_data, synthetic_data, metadata)

    # Assert
    assert result == 0.796825396825397


def test_column_shapes_property_with_progress_bar():
    """Test that the progress bar is correctly updated for the ``Column Shapes`` property."""
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
    """Test ``ColumnPairTrends`` multi-table property end to end."""
    # Setup
    real_data, synthetic_data, metadata = load_demo(modality='multi_table')
    column_pair_trends = ColumnPairTrends()

    # Run
    result = column_pair_trends.get_score(real_data, synthetic_data, metadata)

    # Assert
    assert result == 0.49


def test_coverage_property():
    """Test the ``Coverage`` multi-table property end to end."""
    # Setup
    real_data, synthetic_data, metadata = load_demo(modality='multi_table')
    coverage = Coverage()

    # Run
    result = coverage.get_score(real_data, synthetic_data, metadata)

    # Assert
    assert result == 0.96


def test_coverage_property_with_progress_bar():
    """Test that the progress bar is correctly updated for the ``Coverage`` property."""
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


def test_boundary_property():
    """Test the ``Boundary`` multi-table property end to end."""
    # Setup
    real_data, synthetic_data, metadata = load_demo(modality='multi_table')
    boundary = Boundary()

    # Run
    result = boundary.get_score(real_data, synthetic_data, metadata)

    # Assert
    assert result == 0.96


def test_boundary_property_with_progress_bar():
    """Test that the progress bar is correctly updated for the Boundary property."""
    # Setup
    real_data, synthetic_data, metadata = load_demo(modality='multi_table')
    boundary = Boundary()
    num_columns = sum(len(table['columns']) for table in metadata['tables'].values())

    progress_bar = tqdm(total=num_columns)
    mock_update = Mock()
    progress_bar.update = mock_update

    # Run
    result = boundary.get_score(real_data, synthetic_data, metadata, progress_bar)

    # Assert
    assert result == 0.96
    assert mock_update.call_count == num_columns


def test_synthesis_property():
    """Test Synthesis multi-table."""
    # Setup
    real_data, synthetic_data, metadata = load_demo(modality='multi_table')
    synthesis = Synthesis()

    # Run
    result = synthesis.get_score(real_data, synthetic_data, metadata)

    # Assert
    assert result == 0.96


def test_synthesis_property_with_progress_bar():
    """Test that the progress bar is correctly updated for the Synthesis property."""
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
    assert result == 0.96
    assert mock_update.call_count == num_tables
