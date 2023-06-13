"""Test BaseMultiTableProperty class."""

from unittest.mock import Mock

import pytest

from sdmetrics.reports.multi_table._properties import BaseMultiTableProperty


def test_get_score_raises_error():
    """Test that the method raises a ``NotImplementedError``."""
    # Setup
    base_property = BaseMultiTableProperty()

    # Run and Assert
    with pytest.raises(NotImplementedError):
        base_property.get_score(None, None, None, None)


def test_get_visualization():
    """Test that the method returns the propertie's ``get_visualization``."""
    # Setup
    base_property = BaseMultiTableProperty()
    property_mock = Mock()
    base_property._properties = {'table': property_mock}

    # Run
    result = base_property.get_visualization('table')

    # Assert
    assert result == property_mock.get_visualization.return_value
