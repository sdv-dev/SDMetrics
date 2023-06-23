"""Test ColumnShapes multi-table class."""

from unittest.mock import Mock

import pytest

from sdmetrics.reports.multi_table._properties import ColumnShapes
from sdmetrics.reports.single_table._properties import ColumnShapes as SingleTableColumnShapes

def test__init__():
    """Test the ``__init__`` method."""
    # Setup
    column_shapes = ColumnShapes()

    # Assert
    assert column_shapes._properties == {}
    assert column_shapes._single_table_property == SingleTableColumnShapes
