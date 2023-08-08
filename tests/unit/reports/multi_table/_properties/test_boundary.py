"""Test Boundary multi-table class."""
from sdmetrics.reports.multi_table._properties import Boundary
from sdmetrics.reports.single_table._properties import Boundary as SingleTableBoundary


def test__init__():
    """Test the ``__init__`` method."""
    # Setup
    boundary = Boundary()

    # Assert
    assert boundary._properties == {}
    assert boundary._single_table_property == SingleTableBoundary


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
    boundary = Boundary()

    # Run
    num_iterations = boundary._get_num_iterations(metadata)

    # Assert
    assert num_iterations == 5
