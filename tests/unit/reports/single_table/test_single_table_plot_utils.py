"""Test utility methods for single table report plots."""
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd

from sdmetrics.reports.single_table.plot_utils import (
    _get_column_shapes_data, _get_numerical_correlation_matrices,
    _get_similarity_correlation_matrix, get_column_pairs_plot, get_column_shapes_plot)


def test__get_column_shapes_data():
    """Test the ``_get_column_shapes_data`` function.

    Expect that the score breakdowns are converted into the expected column shapes
    data format.

    Input:
    - score breakdowns

    Output:
    - column shapes data input
    """
    # Setup
    score_breakdowns = {
        'METRIC1': {'col1': {'score': np.nan}, 'col2': {'score': 0.1}},
        'METRIC2': {'col1': {'score': 0.2}, 'col2': {'score': np.nan}},
    }

    # Run
    out = _get_column_shapes_data(score_breakdowns)

    # Assert
    pd.testing.assert_frame_equal(out, pd.DataFrame({
        'Column Name': ['col2', 'col1'],
        'Metric': ['METRIC1', 'METRIC2'],
        'Quality Score': [0.1, 0.2],
    }))


@patch('sdmetrics.reports.single_table.plot_utils.px.bar')
def test_get_column_shapes_plot(bar_mock):
    """Test the ``get_column_shapes_plot`` function.

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
        'METRIC1': {'col1': {'score': np.nan}, 'col2': {'score': 0.1}},
        'METRIC2': {'col1': {'score': 0.2}, 'col2': {'score': np.nan}},
    }
    mock_fig = Mock()
    bar_mock.return_value = mock_fig

    # Run
    out = get_column_shapes_plot(score_breakdowns)

    # Assert
    bar_mock.assert_called_once()
    mock_fig.update_yaxes.assert_called_once_with(range=[0, 1])
    mock_fig.update_layout.assert_called_once()
    assert out == mock_fig


def test__get_similarity_correlation_matrix():
    """Test the ``_get_similarity_correlation_matrix`` function.

    Expect that the score breakdowns are converted into a correlation matrix.

    Input:
    - score breakdowns

    Output:
    - correlation matrix
    """
    # Setup
    score_breakdowns = {
        'METRIC1': {('col1', 'col2'): {'score': 0.1}, ('col2', 'col3'): {'score': 0.3}},
        'METRIC2': {('col1', 'col3'): {'score': 0.2}},
    }
    columns = ['col1', 'col2', 'col3']

    # Run
    out = _get_similarity_correlation_matrix(score_breakdowns, columns)

    # Assert
    expected = pd.DataFrame(
        [
            [1, 0.1, 0.2],
            [0.1, 1, 0.3],
            [0.2, 0.3, 1],
        ],
        columns=['col1', 'col2', 'col3'],
        index=['col1', 'col2', 'col3'],
    )

    pd.testing.assert_frame_equal(out, expected)


def test__get_numerical_correlation_matrices():
    """Test the ``_get_numerical_correlation_matrices`` function.

    Expect that the score breakdowns are converted into a real and synthetic numerical matrices.
    Expect that nan scores are ignored.

    Input:
    - score breakdowns

    Output:
    - correlation matrix
    """
    # Setup
    score_breakdowns = {
        'CorrelationSimilarity': {
            ('col1', 'col2'): {'score': 0.1, 'real': 0.1, 'synthetic': 0.4},
            ('col1', 'col3'): {'score': 0.2, 'real': 0.2, 'synthetic': 0.5},
            ('col2', 'col3'): {'score': 0.3, 'real': 0.3, 'synthetic': 0.6},
            ('col1', 'col4'): {'score': np.nan},
        },
        'METRIC2': {('col1', 'col3'): {'score': 0.2}},
    }

    # Run
    (real_correlation, synthetic_correlation) = _get_numerical_correlation_matrices(
        score_breakdowns)

    # Assert
    expected_real = pd.DataFrame(
        [
            [1, 0.1, 0.2],
            [0.1, 1, 0.3],
            [0.2, 0.3, 1],
        ],
        columns=['col1', 'col2', 'col3'],
        index=['col1', 'col2', 'col3'],
    )
    expected_synthetic = pd.DataFrame(
        [
            [1, 0.4, 0.5],
            [0.4, 1, 0.6],
            [0.5, 0.6, 1],
        ],
        columns=['col1', 'col2', 'col3'],
        index=['col1', 'col2', 'col3'],
    )

    pd.testing.assert_frame_equal(real_correlation, expected_real)
    pd.testing.assert_frame_equal(synthetic_correlation, expected_synthetic)


@patch('sdmetrics.reports.single_table.plot_utils.make_subplots')
@patch('sdmetrics.reports.single_table.plot_utils.go.Heatmap')
def test_get_column_pairs_plot(heatmap_mock, make_subplots_mock):
    """Test the ``get_column_pairs_plot`` function.

    Setup:
    - Mock the plotly ``make_subplots`` method.

    Input:
    - score breakdowns

    Side Effects:
    - The expected plotly method will be called.
    """
    # Setup
    score_breakdowns = {
        'CorrelationSimilarity': {
            ('col1', 'col2'): {'score': 0.1, 'real': 0.1, 'synthetic': 0.1},
            ('col2', 'col3'): {'score': 0.3, 'real': 0.3, 'synthetic': 0.3},
        },
        'ContingencySimilarity': {('col1', 'col3'): {'score': 0.2}},
    }
    mock_fig = Mock()
    make_subplots_mock.return_value = mock_fig
    mock_heatmap_1 = Mock()
    mock_heatmap_2 = Mock()
    mock_heatmap_3 = Mock()
    heatmap_mock.side_effect = [mock_heatmap_1, mock_heatmap_2, mock_heatmap_3]

    # Run
    out = get_column_pairs_plot(score_breakdowns)

    # Assert
    make_subplots_mock.assert_called_once()
    mock_fig.add_trace.assert_has_calls(
        [call(mock_heatmap_1, 1, 1), call(mock_heatmap_2, 2, 1), call(mock_heatmap_3, 2, 2)],
    )
    mock_fig.update_layout.assert_called_once()
    mock_fig.update_yaxes.assert_called_once_with(autorange='reversed')

    assert out == mock_fig
