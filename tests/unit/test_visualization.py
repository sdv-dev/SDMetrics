import re
from unittest.mock import Mock, call, patch

import pandas as pd
import pytest

from sdmetrics.visualization import (
    _generate_box_plot, _generate_cardinality_plot, _generate_heatmap_plot, _generate_scatter_plot,
    _get_cardinality, get_cardinality_plot, get_column_pair_plot, get_column_plot)
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


def test_get_column_plot_column_not_found():
    """Test the ``get_column_plot`` method when column is not present."""
    # Setup
    real_data = pd.DataFrame({'values': [1, 2, 2, 3, 5]})
    synthetic_data = pd.DataFrame({'values': [2, 2, 3, 4, 5]})

    # Run and assert
    match = re.escape("Column 'start_date' not found in real table data.")
    with pytest.raises(ValueError, match=match):
        get_column_plot(real_data, synthetic_data, 'start_date')

    match = re.escape("Column 'start_date' not found in synthetic table data.")
    with pytest.raises(ValueError, match=match):
        get_column_plot(pd.DataFrame({'start_date': []}), synthetic_data, 'start_date')


def test_get_column_plot_bad_plot_type():
    """Test the ``get_column_plot`` method."""
    # Setup
    real_data = pd.DataFrame({'values': [1, 2, 2, 3, 5]})
    synthetic_data = pd.DataFrame({'values': [2, 2, 3, 4, 5]})

    # Run and assert
    match = re.escape("Invalid plot_type 'bad_type'. Please use one of ['bar', 'distplot', None].")
    with pytest.raises(ValueError, match=match):
        get_column_plot(real_data, synthetic_data, 'valeus', plot_type='bad_type')


@patch('sdmetrics.visualization._generate_column_plot')
def test_get_column_plot_plot_type_none_data_int(mock__generate_column_plot):
    """Test ``get_column_plot`` when ``plot_type`` is ``None`` and data is ``int``."""
    # Setup
    real_data = pd.DataFrame({'values': [1, 2, 2, 3, 5]})
    synthetic_data = pd.DataFrame({'values': [2, 2, 3, 4, 5]})

    # Run
    figure = get_column_plot(real_data, synthetic_data, 'values')

    # Assert
    mock__generate_column_plot.assert_called_once_with(
        real_data['values'],
        synthetic_data['values'],
        'distplot'
    )
    assert figure == mock__generate_column_plot.return_value


@patch('sdmetrics.visualization._generate_column_plot')
def test_get_column_plot_plot_type_none_data_float(mock__generate_column_plot):
    """Test ``get_column_plot`` when ``plot_type`` is ``None`` and data is ``float``."""
    # Setup
    real_data = pd.DataFrame({'values': [1., 2., 2., 3., 5.]})
    synthetic_data = pd.DataFrame({'values': [2., 2., 3., 4., 5.]})

    # Run
    figure = get_column_plot(real_data, synthetic_data, 'values')

    # Assert
    mock__generate_column_plot.assert_called_once_with(
        real_data['values'],
        synthetic_data['values'],
        'distplot'
    )
    assert figure == mock__generate_column_plot.return_value


@patch('sdmetrics.visualization._generate_column_plot')
def test_get_column_plot_plot_type_none_data_datetime(mock__generate_column_plot):
    """Test ``get_column_plot`` when ``plot_type`` is ``None`` and data is ``datetime``."""
    # Setup
    real_data = pd.DataFrame({'values': pd.to_datetime(['2021-01-20', '2022-01-21'])})
    synthetic_data = pd.DataFrame({'values': pd.to_datetime(['2021-01-20', '2022-01-21'])})

    # Run
    figure = get_column_plot(real_data, synthetic_data, 'values')

    # Assert
    mock__generate_column_plot.assert_called_once_with(
        real_data['values'],
        synthetic_data['values'],
        'distplot'
    )
    assert figure == mock__generate_column_plot.return_value


@patch('sdmetrics.visualization._generate_column_plot')
def test_get_column_plot_plot_type_none_data_category(mock__generate_column_plot):
    """Test ``get_column_plot`` when ``plot_type`` is ``None`` and data is ``category``."""
    # Setup
    real_data = pd.DataFrame({'values': ['John', 'Doe']})
    synthetic_data = pd.DataFrame({'values': ['Johanna', 'Doe']})

    # Run
    figure = get_column_plot(real_data, synthetic_data, 'values')

    # Assert
    mock__generate_column_plot.assert_called_once_with(
        real_data['values'],
        synthetic_data['values'],
        'bar'
    )
    assert figure == mock__generate_column_plot.return_value


@patch('sdmetrics.visualization._generate_column_plot')
def test_get_column_plot_plot_type_bar(mock__generate_column_plot):
    """Test ``get_column_plot`` when ``plot_type`` is ``bar``."""
    # Setup
    real_data = pd.DataFrame({'values': [1., 2., 2., 3., 5.]})
    synthetic_data = pd.DataFrame({'values': [2., 2., 3., 4., 5.]})

    # Run
    figure = get_column_plot(real_data, synthetic_data, 'values', plot_type='bar')

    # Assert
    mock__generate_column_plot.assert_called_once_with(
        real_data['values'],
        synthetic_data['values'],
        'bar'
    )
    assert figure == mock__generate_column_plot.return_value


@patch('sdmetrics.visualization._generate_column_plot')
def test_get_column_plot_plot_type_distplot(mock__generate_column_plot):
    """Test ``get_column_plot`` when ``plot_type`` is ``distplot``."""
    # Setup
    real_data = pd.DataFrame({'values': ['John', 'Doe']})
    synthetic_data = pd.DataFrame({'values': ['Johanna', 'Doe']})

    # Run
    figure = get_column_plot(real_data, synthetic_data, 'values', plot_type='distplot')

    # Assert
    mock__generate_column_plot.assert_called_once_with(
        real_data['values'],
        synthetic_data['values'],
        'distplot'
    )
    assert figure == mock__generate_column_plot.return_value


@patch('sdmetrics.visualization.px')
def test__generate_scatter_plot(px_mock):
    """Test the ``_generate_scatter_plot`` method."""
    # Setup
    real_column = pd.DataFrame({
        'col1': [1, 2, 3, 4],
        'col2': [1.1, 1.2, 1.3, 1.4],
        'Data': ['Real'] * 4
    })
    synthetic_column = pd.DataFrame({
        'col1': [1, 2, 4, 5],
        'col2': [1.1, 1.2, 1.3, 1.4],
        'Data': ['Synthetic'] * 4
    })

    all_data = pd.concat([real_column, synthetic_column], axis=0, ignore_index=True)
    columns = ['col1', 'col2']
    mock_figure = Mock()
    px_mock.scatter.return_value = mock_figure

    # Run
    fig = _generate_scatter_plot(all_data, columns)

    # Assert
    px_mock.scatter.assert_called_once_with(
        DataFrameMatcher(pd.DataFrame({
            'col1': [1, 2, 3, 4, 1, 2, 4, 5],
            'col2': [1.1, 1.2, 1.3, 1.4, 1.1, 1.2, 1.3, 1.4],
            'Data': [
                'Real',
                'Real',
                'Real',
                'Real',
                'Synthetic',
                'Synthetic',
                'Synthetic',
                'Synthetic',
            ],
        })),
        x='col1',
        y='col2',
        color='Data',
        color_discrete_map={'Real': '#000036', 'Synthetic': '#01E0C9'},
        symbol='Data',
    )
    assert mock_figure.update_layout.called_once()
    assert fig == mock_figure


@patch('sdmetrics.visualization.px')
def test__generate_heatmap_plot(px_mock):
    """Test the ``_generate_heatmap_plot`` method."""
    # Setup
    real_column = pd.DataFrame({
        'col1': [1, 2, 3, 4],
        'col2': ['a', 'b', 'c', 'd'],
        'Data': ['Real'] * 4
    })
    synthetic_column = pd.DataFrame({
        'col1': [1, 2, 4, 5],
        'col2': ['a', 'b', 'c', 'd'],
        'Data': ['Synthetic'] * 4
    })
    columns = ['col1', 'col2']
    all_data = pd.concat([real_column, synthetic_column], axis=0, ignore_index=True)

    mock_figure = Mock()
    px_mock.density_heatmap.return_value = mock_figure

    # Run
    fig = _generate_heatmap_plot(all_data, columns)

    # Assert
    px_mock.density_heatmap.assert_called_once_with(
        DataFrameMatcher(pd.DataFrame({
            'col1': [1, 2, 3, 4, 1, 2, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd'],
            'Data': [
                'Real',
                'Real',
                'Real',
                'Real',
                'Synthetic',
                'Synthetic',
                'Synthetic',
                'Synthetic',
            ],
        })),
        x='col1',
        y='col2',
        facet_col='Data',
        histnorm='probability',
    )
    assert mock_figure.update_layout.called_once()
    assert mock_figure.for_each_annotation.called_once()
    assert fig == mock_figure


@patch('sdmetrics.visualization.px')
def test__generate_box_plot(px_mock):
    """Test the ``_generate_box_plot`` method."""
    # Setup
    real_column = pd.DataFrame({
        'col1': [1, 2, 3, 4],
        'col2': ['a', 'b', 'c', 'd'],
        'Data': ['Real'] * 4
    })
    synthetic_column = pd.DataFrame({
        'col1': [1, 2, 4, 5],
        'col2': ['a', 'b', 'c', 'd'],
        'Data': ['Synthetic'] * 4
    })
    columns = ['col1', 'col2']
    all_data = pd.concat([real_column, synthetic_column], axis=0, ignore_index=True)

    mock_figure = Mock()
    px_mock.box.return_value = mock_figure

    # Run
    fig = _generate_box_plot(all_data, columns)

    # Assert
    px_mock.box.assert_called_once_with(
        DataFrameMatcher(pd.DataFrame({
            'col1': [1, 2, 3, 4, 1, 2, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd'],
            'Data': [
                'Real',
                'Real',
                'Real',
                'Real',
                'Synthetic',
                'Synthetic',
                'Synthetic',
                'Synthetic',
            ],
        })),
        x='col1',
        y='col2',
        color='Data',
        color_discrete_map={'Real': '#000036', 'Synthetic': '#01E0C9'},
    )
    assert mock_figure.update_layout.called_once()
    assert fig == mock_figure


def test_get_column_pair_plot_invalid_column_names():
    """Test ``get_column_pair_plot`` method with invalid ``column_names``."""
    # Setup
    columns = ['values']
    real_data = synthetic_data = pd.DataFrame({'values': []})

    # Run and Assert
    match = 'Must provide exactly two column names.'
    with pytest.raises(ValueError, match=match):
        get_column_pair_plot(real_data, synthetic_data, columns)


def test_get_column_pair_plot_columns_not_in_real_data():
    """Test ``get_column_pair_plot`` method with ``column_names`` not in the real data."""
    # Setup
    columns = ['start_date', 'end_date']
    real_data = synthetic_data = pd.DataFrame({'start_date': []})

    # Run and Assert
    match = re.escape("Missing column(s) {'end_date'} in real data.")
    with pytest.raises(ValueError, match=match):
        get_column_pair_plot(real_data, synthetic_data, columns)


def test_get_column_pair_plot_columns_not_in_syntehtic_data():
    """Test ``get_column_pair_plot`` method with ``column_names`` not in the synthetic data."""
    # Setup
    columns = ['start_date', 'end_date']
    real_data = pd.DataFrame({'start_date': [], 'end_date': []})
    synthetic_data = pd.DataFrame({'start_date': []})

    # Run and Assert
    match = re.escape("Missing column(s) {'end_date'} in synthetic data.")
    with pytest.raises(ValueError, match=match):
        get_column_pair_plot(real_data, synthetic_data, columns)


def test_get_column_pair_plot_invalid_plot_type():
    """Test when invalid plot type is passed as argument to ``get_column_pair_plot``."""
    # Setup
    columns = ['start_date', 'end_date']
    real_data = synthetic_data = pd.DataFrame({'start_date': [], 'end_date': []})

    # Run and Assert
    match = re.escape(
        "Invalid plot_type 'distplot'. Please use one of ['box', 'heatmap', 'scatter', None]."
    )
    with pytest.raises(ValueError, match=match):
        get_column_pair_plot(real_data, synthetic_data, columns, plot_type='distplot')


@patch('sdmetrics.visualization._generate_scatter_plot')
def test_get_column_pair_plot_plot_type_none_continuous_data(mock__generate_scatter_plot):
    """Test ``get_column_pair_plot`` with continuous data and ``plot_type`` ``None``."""
    # Setup
    columns = ['amount', 'price']
    real_data = pd.DataFrame({'amount': [1, 2, 3], 'price': [4, 5, 6]})
    synthetic_data = pd.DataFrame({'amount': [1., 2., 3.], 'price': [4., 5., 6.]})

    # Run
    fig = get_column_pair_plot(real_data, synthetic_data, columns)

    # Assert
    real_data['Data'] = 'Real'
    synthetic_data['Data'] = 'Synthetic'
    expected_call_data = pd.concat([real_data, synthetic_data], axis=0, ignore_index=True)

    mock__generate_scatter_plot.assert_called_once_with(
        DataFrameMatcher(expected_call_data),
        ['amount', 'price']
    )
    assert fig == mock__generate_scatter_plot.return_value


@patch('sdmetrics.visualization._generate_scatter_plot')
def test_get_column_pair_plot_plot_type_none_continuous_data_and_date(mock__generate_scatter_plot):
    """Test ``get_column_pair_plot`` with continuous data and ``plot_type`` ``None``."""
    # Setup
    columns = ['amount', 'date']
    real_data = pd.DataFrame({
        'amount': [1, 2, 3],
        'date': pd.to_datetime(['2021-01-01', '2022-01-01', '2023-01-01']),
    })
    synthetic_data = pd.DataFrame({
        'amount': [1., 2., 3.],
        'date': pd.to_datetime(['2021-01-01', '2022-01-01', '2023-01-01']),
    })

    # Run
    fig = get_column_pair_plot(real_data, synthetic_data, columns)

    # Assert
    real_data['Data'] = 'Real'
    synthetic_data['Data'] = 'Synthetic'
    expected_call_data = pd.concat([real_data, synthetic_data], axis=0, ignore_index=True)

    mock__generate_scatter_plot.assert_called_once_with(
        DataFrameMatcher(expected_call_data),
        ['amount', 'date'],
    )
    assert fig == mock__generate_scatter_plot.return_value


@patch('sdmetrics.visualization._generate_heatmap_plot')
def test_get_column_pair_plot_plot_type_none_discrete_data(mock__generate_heatmap_plot):
    """Test ``get_column_pair_plot`` with discrete data and ``plot_type`` ``None``."""
    # Setup
    columns = ['name', 'surname']
    real_data = pd.DataFrame({
        'name': ['John', 'Emily'],
        'surname': ['Morales', 'Terry']
    })
    synthetic_data = pd.DataFrame({
        'name': ['John', 'Johanna'],
        'surname': ['Dominic', 'Rogers']
    })

    # Run
    fig = get_column_pair_plot(real_data, synthetic_data, columns)

    # Assert
    real_data['Data'] = 'Real'
    synthetic_data['Data'] = 'Synthetic'
    expected_call_data = pd.concat([real_data, synthetic_data], axis=0, ignore_index=True)

    mock__generate_heatmap_plot.assert_called_once_with(
        DataFrameMatcher(expected_call_data),
        ['name', 'surname'],
    )
    assert fig == mock__generate_heatmap_plot.return_value


@patch('sdmetrics.visualization._generate_box_plot')
def test_get_column_pair_plot_plot_type_none_discrete_and_continuous(mock__generate_box_plot):
    """Test ``get_column_pair_plot`` with discrete and continuous data."""
    # Setup
    columns = ['name', 'counts']
    real_data = pd.DataFrame({
        'name': ['John', 'Emily'],
        'counts': [1, 2]
    })
    synthetic_data = pd.DataFrame({
        'name': ['John', 'Johanna'],
        'counts': [3, 1]
    })

    # Run
    fig = get_column_pair_plot(real_data, synthetic_data, columns)

    # Assert
    real_data['Data'] = 'Real'
    synthetic_data['Data'] = 'Synthetic'
    expected_call_data = pd.concat([real_data, synthetic_data], axis=0, ignore_index=True)

    mock__generate_box_plot.assert_called_once_with(
        DataFrameMatcher(expected_call_data),
        ['name', 'counts']
    )
    assert fig == mock__generate_box_plot.return_value


@patch('sdmetrics.visualization._generate_heatmap_plot')
def test_get_column_pair_plot_plot_type_is_box(mock__generate_heatmap_plot):
    """Test ``get_column_pair_plot`` when forcing it ot be a heatmap plot on continuous data."""
    # Setup
    columns = ['amount', 'date']
    real_data = pd.DataFrame({
        'amount': [1, 2, 3],
        'date': pd.to_datetime(['2021-01-01', '2022-01-01', '2023-01-01'])
    })
    synthetic_data = pd.DataFrame({
        'amount': [1., 2., 3.],
        'date': pd.to_datetime(['2021-01-01', '2022-01-01', '2023-01-01'])
    })

    # Run
    fig = get_column_pair_plot(real_data, synthetic_data, columns, plot_type='heatmap')

    # Assert
    real_data['Data'] = 'Real'
    synthetic_data['Data'] = 'Synthetic'
    expected_call_data = pd.concat([real_data, synthetic_data], axis=0, ignore_index=True)

    mock__generate_heatmap_plot.assert_called_once_with(
        DataFrameMatcher(expected_call_data),
        ['amount', 'date']
    )
    assert fig == mock__generate_heatmap_plot.return_value
