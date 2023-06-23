from sdmetrics.reports.multi_table._properties import ColumnShapes, ColumnPairTrends
from sdmetrics.demos import load_demo
import tqdm


def test_column_shapes_property():
    """Test ColumnShapes multi-table class."""
    # Setup
    real_data, synthetic_data, metadata = load_demo(modality='multi_table')
    column_shapes = ColumnShapes()

    # Run
    result = column_shapes.get_score(real_data, synthetic_data, metadata)

    # Assert
    assert result == 0.797


def test_column_shapes_property_with_progress_bar():
    # Setup
    real_data, synthetic_data, metadata = load_demo(modality='multi_table')
    column_shapes = ColumnShapes()
    num_columns = sum(len(table['columns']) for table in metadata['tables'].values())
    progress_bar = tqdm.tqdm(num_columns)

    # Run
    result = column_shapes.get_score(real_data, synthetic_data, metadata)

    # Assert
    assert result == 0.797

def test_column_pair_trends_property():
    """Test ColumnPairTrends multi-table class."""
    # Setup
    real_data, synthetic_data, metadata = load_demo(modality='multi_table')
    column_pair_trends = ColumnPairTrends()

    # Run
    result = column_pair_trends.get_score(real_data, synthetic_data, metadata)

    # Assert
    assert result == 0.493

