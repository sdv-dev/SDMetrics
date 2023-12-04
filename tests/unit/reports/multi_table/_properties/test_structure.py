"""Test Structure multi-table class."""
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from sdmetrics.errors import VisualizationUnavailableError
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


@patch('sdmetrics.reports.multi_table._properties.structure.px')
def test_get_visualization(mock_px):
    """Test the ``get_visualization`` method."""
    # Setup
    structure_property = Structure()

    mock_df = pd.DataFrame({
        'Table': ['Table1', 'Table2'],
        'Score': [0.7, 0.3],
        'Metric': ['TableStructure', 'TableStructure']
    })
    structure_property.details = mock_df

    mock__compute_average = Mock(return_value=0.5)
    structure_property._compute_average = mock__compute_average

    mock_bar = Mock()
    mock_px.bar.return_value = mock_bar

    # Run
    structure_property.get_visualization()

    # Assert
    mock__compute_average.assert_called_once()

    # Expected call
    expected_kwargs = {
        'data_frame': mock_df,
        'x': 'Table',
        'y': 'Score',
        'title': (
            'Data Diagnostic: Structure (Average '
            f'Score={mock__compute_average.return_value})'
        ),
        'category_orders': {'group': mock_df['Table'].tolist()},
        'color': 'Metric',
        'color_discrete_map': {
            'TableStructure': '#000036',
        },
        'pattern_shape': 'Metric',
        'pattern_shape_sequence': [''],
        'hover_name': 'Table',
        'hover_data': {
            'Table': False,
            'Metric': True,
            'Score': True,
        },
    }

    # Check call_args of mock_px.bar
    _, kwargs = mock_px.bar.call_args

    # Check DataFrame separately
    assert kwargs.pop('data_frame').equals(expected_kwargs.pop('data_frame'))

    # Check other arguments
    assert kwargs == expected_kwargs

    mock_bar.update_yaxes.assert_called_once_with(range=[0, 1])
    mock_bar.update_layout.assert_called_once_with(
        xaxis_categoryorder='total ascending',
        plot_bgcolor='#F5F5F8',
        margin={'t': 150},
        font={'size': 18}
    )


def test_get_visualization_with_table_name():
    """Test the ``get_visualization`` when a table name is given."""
    # Setup
    synthesis = Structure()

    # Run and Assert
    expected_message = (
        'The Structure property does not have a supported visualization for individual tables.'
    )
    with pytest.raises(VisualizationUnavailableError, match=expected_message):
        synthesis.get_visualization('table_name')
