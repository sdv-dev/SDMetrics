"""Test Coverage multi-table class."""
from sdmetrics.reports.multi_table._properties import Coverage
from sdmetrics.reports.single_table._properties import Coverage as SingleTableCoverage


def test__init__():
    """Test the ``__init__`` method."""
    # Setup
    coverage = Coverage()

    # Assert
    assert coverage._properties == {}
    assert coverage._single_table_property == SingleTableCoverage


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
    coverage = Coverage()

    # Run
    num_iterations = coverage._get_num_iterations(metadata)

    # Assert
    assert num_iterations == 5
