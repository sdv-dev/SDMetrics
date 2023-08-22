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
    assert boundary._num_iteration_case == 'column'
