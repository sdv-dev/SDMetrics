"""Test Structure multi-table class."""
from sdmetrics.reports.multi_table._properties import Structure
from sdmetrics.reports.single_table._properties import Structure as SingleTableStructure


def test__init__():
    """Test the ``__init__`` method."""
    # Setup
    synthesis = Structure()

    # Assert
    assert synthesis._properties == {}
    assert synthesis._single_table_property == SingleTableStructure
    assert synthesis._num_iteration_case == 'table'
