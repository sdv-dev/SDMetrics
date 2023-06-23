"""Test BaseMultiTableProperty class."""

from unittest.mock import Mock, call

import pytest

from sdmetrics.reports.multi_table._properties import BaseMultiTableProperty

def test__init__():
    """Test the ``__init__`` method."""
    # Setup
    base_property = BaseMultiTableProperty()

    # Assert
    assert base_property._properties == {}
    assert base_property._single_table_property is None


def test_get_score_raises_error():
    """Test that the method raises a ``NotImplementedError``."""
    # Setup
    base_property = BaseMultiTableProperty()

    # Run and Assert
    with pytest.raises(NotImplementedError):
        base_property.get_score(None, None, None, None)


def test_get_score_with_single_table_property():
    """Test that the method returns the property's ``get_score``."""
    # Setup
    base_property = BaseMultiTableProperty()
    mock_property = Mock()
    mock_property.get_score.return_value = 1.0
    base_property._single_table_property = Mock(return_value=mock_property)

    metadata = {
        'tables': {
            'table_1': {},
            'table_2': {},
            'table_3': {}
        }
    }

    real_data = {
        'table_1': Mock(),
        'table_2': Mock(),
        'table_3': Mock()
    }

    synthetic_data = {
        'table_1': Mock(),
        'table_2': Mock(),
        'table_3': Mock()
    }

    progress_bar = Mock()

    # Run
    result = base_property.get_score(real_data, synthetic_data, metadata, progress_bar)

    # Assert
    expected_average_score = 1.0
    expected_calls = [
        call(real_data['table_1'], synthetic_data['table_1'], metadata['tables']['table_1'], progress_bar),
        call(real_data['table_2'], synthetic_data['table_2'], metadata['tables']['table_2'], progress_bar),
        call(real_data['table_3'], synthetic_data['table_3'], metadata['tables']['table_3'], progress_bar)
    ]

    assert result == expected_average_score
    assert base_property._single_table_property.call_count == 3
    assert mock_property.get_score.call_args_list == expected_calls


def test_get_visualization():
    """Test that the method returns the property's ``get_visualization``."""
    # Setup
    base_property = BaseMultiTableProperty()
    property_mock = Mock()
    base_property._properties = {'table': property_mock}

    # Run
    result = base_property.get_visualization('table')

    # Assert
    assert result == property_mock.get_visualization.return_value
