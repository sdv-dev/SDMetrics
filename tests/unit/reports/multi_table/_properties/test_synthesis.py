"""Test Synthesis multi-table class."""
from sdmetrics.reports.multi_table._properties import Synthesis
from sdmetrics.reports.single_table._properties import Synthesis as SingleTableSynthesis


def test__init__():
    """Test the ``__init__`` method."""
    # Setup
    synthesis = Synthesis()

    # Assert
    assert synthesis._properties == {}
    assert synthesis._single_table_property == SingleTableSynthesis


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
    synthesis = Synthesis()

    # Run
    num_iterations = synthesis._get_num_iterations(metadata)

    # Assert
    assert num_iterations == 2
