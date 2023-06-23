"""Test ColumnPairTrends multi-table class."""

from unittest.mock import Mock

import pytest

from sdmetrics.reports.multi_table._properties import ColumnPairTrends
from sdmetrics.reports.single_table._properties import ColumnPairTrends as SingleTableColumnPairTrends

def test__init__():
    """Test the ``__init__`` method."""
    # Setup
    column_pair_trends = ColumnPairTrends()

    # Assert
    assert column_pair_trends._properties == {}
    assert column_pair_trends._single_table_property == SingleTableColumnPairTrends