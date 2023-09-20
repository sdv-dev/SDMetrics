import re
from unittest.mock import Mock, call, patch

import pandas as pd
import pytest

from sdmetrics.visualization import (
    _generate_cardinality_plot, _get_cardinality, get_cardinality_plot)
from tests.utils import DataFrameMatcher, SeriesMatcher


def test_get_cardinality():
    """Test the ``_get_cardinality`` method."""
    # Setup
    parent_table = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve']
    })
    child_table = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'parent_id': [1, 1, 2, 2, 2, 3, 3, 3, 3, 4]
    })
    parent_primary_key = 'id'
    child_foreign_key = 'parent_id'

    # Run
    result = _get_cardinality(parent_table, child_table, parent_primary_key, child_foreign_key)

    # Assert
    expected_result = pd.Series(
        [0, 1, 2, 3, 4], index=pd.Index([5, 4, 1, 2, 3], name='id'), name='# children'
    )

    pd.testing.assert_series_equal(result, expected_result)


@patch('sdmetrics.visualization.px')
def test_generate_cardinality_bar_plot(mock_px):
    """Test the ``_generate_cardinality_plot`` method."""
    # Setup
    mock_real_data = pd.Series([1, 1, 2, 2, 2])
    mock_synthetic_data = pd.Series([3, 3, 4])
    mock_data = pd.DataFrame({
        'values': [1, 1, 2, 2, 2, 3, 3, 4],
        'Data': [*['Real'] * 5, *['Synthetic'] * 3]
    })

    parent_primary_key = 'parent_key'
    child_foreign_key = 'child_key'

    mock_fig = Mock()
    mock_px.histogram.return_value = mock_fig
    mock_fig.data = [Mock(), Mock()]

    # Run
    _generate_cardinality_plot(
        mock_real_data,
        mock_synthetic_data,
        parent_primary_key,
        child_foreign_key)

    # Expected call
    expected_kwargs = {
        'x': 'values',
        'color': 'Data',
        'barmode': 'group',
        'color_discrete_sequence': ['#000036', '#01E0C9'],
        'pattern_shape': 'Data',
        'pattern_shape_sequence': ['', '/'],
        'nbins': 4,
        'histnorm': 'probability density'
    }

    # Assert
    mock_px.histogram.assert_called_once_with(DataFrameMatcher(mock_data), **expected_kwargs)

    title = (
        f"Relationship (child foreign key='{child_foreign_key}' and parent "
        f"primary key='{parent_primary_key}')"
    )

    # Check update_layout and update_traces
    mock_fig.update_layout.assert_called_once_with(
        title=title,
        xaxis_title='# of Children (per Parent)',
        yaxis_title='Frequency',
        plot_bgcolor='#F5F5F8',
        annotations=[],
        font={'size': 18}
    )

    for i, name in enumerate(['Real', 'Synthetic']):
        mock_fig.update_traces.assert_any_call(
            x=mock_fig.data[i].x,
            hovertemplate=f'<b>{name}</b><br>Frequency: %{{y}}<extra></extra>',
            selector={'name': name}
        )


@patch('sdmetrics.visualization.ff')
def test_generate_cardinality_distplot(mock_ff):
    """Test the ``_generate_cardinality_plot`` method with ``plot_type``=='distplot'."""
    # Setup
    mock_real_data = pd.Series([1, 1, 2, 2, 2], name='values')
    mock_synthetic_data = pd.Series([3, 3, 4], name='values')

    parent_primary_key = 'parent_key'
    child_foreign_key = 'child_key'

    mock_fig = Mock()
    mock_ff.create_distplot.return_value = mock_fig
    mock_fig.data = [Mock(), Mock()]

    # Run
    _generate_cardinality_plot(mock_real_data, mock_synthetic_data, parent_primary_key,
                               child_foreign_key, plot_type='distplot')

    # Expected call
    expected_kwargs = {
        'show_hist': False,
        'show_rug': False,
        'colors': ['#000036', '#01E0C9']
    }

    # Assert
    mock_ff.create_distplot.assert_called_once_with(
        [SeriesMatcher(mock_real_data), SeriesMatcher(mock_synthetic_data)],
        ['Real', 'Synthetic'],
        **expected_kwargs)

    title = (
        f"Relationship (child foreign key='{child_foreign_key}' and parent "
        f"primary key='{parent_primary_key}')"
    )

    # Check update_layout and update_traces
    mock_fig.update_layout.assert_called_once_with(
        title=title,
        xaxis_title='# of Children (per Parent)',
        yaxis_title='Frequency',
        plot_bgcolor='#F5F5F8',
        annotations=[],
        font={'size': 18}
    )

    for i, name in enumerate(['Real', 'Synthetic']):
        mock_fig.update_traces.assert_any_call(
            x=mock_fig.data[i].x,
            fill='tozeroy',
            hovertemplate=f'<b>{name}</b><br>Frequency: %{{y}}<extra></extra>',
            selector={'name': name}
        )


@patch('sdmetrics.visualization._get_cardinality')
@patch('sdmetrics.visualization._generate_cardinality_plot')
def test_get_cardinality_plot(mock_generate_cardinality_plot, mock_get_cardinality):
    """Test the ``get_cardinality_plot`` method."""
    # Setup
    real_data = {'table1': None, 'table2': None}
    synthetic_data = {'table1': None, 'table2': None}
    child_foreign_key = 'child_key'
    parent_primary_key = 'parent_key'
    parent_table_name = 'table1'
    child_table_name = 'table2'

    real_cardinality = pd.Series([1, 2, 2, 3, 5])
    synthetic_cardinality = pd.Series([2, 2, 3, 4, 5])
    mock_get_cardinality.side_effect = [real_cardinality, synthetic_cardinality]

    mock_generate_cardinality_plot.return_value = 'test_fig'

    # Run
    fig = get_cardinality_plot(
        real_data, synthetic_data, child_table_name, parent_table_name,
        child_foreign_key, parent_primary_key
    )

    # Assert
    assert fig == 'test_fig'

    # Check the calls
    calls = [
        call(real_data['table1'], real_data['table2'], 'parent_key', 'child_key'),
        call(synthetic_data['table1'], synthetic_data['table2'], 'parent_key', 'child_key')
    ]
    mock_get_cardinality.assert_has_calls(calls)

    real_cardinality['data'] = 'Real'
    synthetic_cardinality['data'] = 'Synthetic'

    pd.testing.assert_series_equal(
        real_cardinality, mock_generate_cardinality_plot.call_args[0][0]
    )
    pd.testing.assert_series_equal(
        synthetic_cardinality, mock_generate_cardinality_plot.call_args[0][1]
    )

    other_args = mock_generate_cardinality_plot.call_args[0][2:]
    assert other_args == ('parent_key', 'child_key')
    assert mock_generate_cardinality_plot.call_args.kwargs == {'plot_type': 'bar'}


def test_get_cardinality_plot_bad_plot_type():
    """Test the ``get_cardinality_plot`` method."""
    # Setup
    real_data = {'table1': None, 'table2': None}
    synthetic_data = {'table1': None, 'table2': None}
    child_foreign_key = 'child_key'
    parent_primary_key = 'parent_key'
    parent_table_name = 'table1'
    child_table_name = 'table2'

    pd.Series([1, 2, 2, 3, 5])
    pd.Series([2, 2, 3, 4, 5])

    # Run and assert
    match = re.escape("Invalid plot_type 'bad_type'. Please use one of ['bar', 'distplot'].")
    with pytest.raises(ValueError, match=match):
        get_cardinality_plot(
            real_data, synthetic_data, child_table_name, parent_table_name, child_foreign_key,
            parent_primary_key, plot_type='bad_type'
        )
