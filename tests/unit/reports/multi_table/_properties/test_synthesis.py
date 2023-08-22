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
    assert synthesis._num_iteration_case == 'table'
