"""Test Data Validity multi-table class."""
from sdmetrics.reports.multi_table._properties import DataValidity
from sdmetrics.reports.single_table._properties import DataValidity as SingleTableDataValidity


def test__init__():
    """Test the ``__init__`` method."""
    # Setup
    column_shapes = DataValidity()

    # Assert
    assert column_shapes._properties == {}
    assert column_shapes._single_table_property == SingleTableDataValidity
    assert column_shapes._num_iteration_case == 'column'
