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
    assert coverage._num_iteration_case == 'column'
