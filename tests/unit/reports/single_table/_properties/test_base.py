"""Test BaseSingleTableProperty class."""
import pytest

from sdmetrics.reports.single_table._properties import BaseSingleTableProperty


def test_get_score_raises_error():
    """Test that the method raises a ``NotImplementedError``."""
    # Setup
    base_property = BaseSingleTableProperty()

    # Run and Assert
    with pytest.raises(NotImplementedError):
        base_property.get_score(None, None, None, None)


def test_get_visualization_raises_error():
    """Test that the method raises a ``NotImplementedError``."""
    # Setup
    base_property = BaseSingleTableProperty()

    # Run and Assert
    with pytest.raises(NotImplementedError):
        base_property.get_visualization()
