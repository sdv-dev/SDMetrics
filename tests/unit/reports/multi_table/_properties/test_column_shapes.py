"""Test ColumnShapes multi-table class."""
from sdmetrics.reports.multi_table._properties import ColumnShapes
from sdmetrics.reports.single_table._properties import ColumnShapes as SingleTableColumnShapes


def test__init__():
    """Test the ``__init__`` method."""
    # Setup
    column_shapes = ColumnShapes()

    # Assert
    assert column_shapes._properties == {}
    assert column_shapes._single_table_property == SingleTableColumnShapes


def test__get_num_iterations():
    """Test the ``_get_num_iterations`` method."""
    # Setup
    metadata = {
        'tables': {
            'Table_1': {
                'columns': {
                    'col1': {},
                    'col2': {},
                },
            },
            'Table_2': {
                'columns': {
                    'col3': {},
                    'col4': {},
                    'col5': {},
                },
            },
        }
    }
    column_shapes = ColumnShapes()

    # Run
    num_iterations = column_shapes._get_num_iterations(metadata)

    # Assert
    assert num_iterations == 5
