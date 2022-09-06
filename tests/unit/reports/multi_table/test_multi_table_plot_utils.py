"""Test utility methods for multi table report plots."""
from unittest.mock import Mock, patch

import pandas as pd

from sdmetrics.reports.multi_table.plot_utils import (
    _get_table_relationships_data, get_table_relationships_plot)


def test__get_table_relationships_data():
    """Test the ``_get_table_relationships_data`` function.

    Expect that the score breakdowns are converted into the expected table relationships
    data format.

    Input:
    - score breakdowns

    Output:
    - table relationships data input
    """
    # Setup
    score_breakdowns = {
        'METRIC': {
            ('table2', 'table1'): {'score': 0.2},
            ('table3', 'table1'): {'score': 0.1},
        },
    }

    # Run
    out = _get_table_relationships_data(score_breakdowns)

    # Assert
    pd.testing.assert_frame_equal(out, pd.DataFrame({
        'Child → Parent Relationship': ['table1 → table2', 'table1 → table3'],
        'Metric': ['METRIC', 'METRIC'],
        'Quality Score': [0.2, 0.1],
    }))


@patch('sdmetrics.reports.single_table.plot_utils.px.bar')
def test_get_table_relationships_plot(bar_mock):
    """Test the ``get_table_relationships_plot`` function.

    Setup:
    - Mock the plotly bar method.

    Input:
    - score breakdowns

    Output:
    - plotly figure

    Side Effects:
    - The bar method is called the expected number of times.
    """
    # Setup
    score_breakdowns = {
        'METRIC': {
            ('table2', 'table1'): {'score': 0.2},
            ('table3', 'table1'): {'score': 0.1},
        },
    }
    mock_fig = Mock()
    bar_mock.return_value = mock_fig

    # Run
    out = get_table_relationships_plot(score_breakdowns)

    # Assert
    bar_mock.assert_called_once()
    mock_fig.update_yaxes.assert_called_once_with(range=[0, 1])
    mock_fig.update_layout.assert_called_once()
    assert out == mock_fig
