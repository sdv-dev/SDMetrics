from datetime import date, datetime
from unittest.mock import Mock, call, patch

import pandas as pd

from sdmetrics.reports.utils import (
    aggregate_metric_results, discretize_and_apply_metric, discretize_table_data, get_column_plot,
    make_continuous_column_plot, make_discrete_column_plot)
from tests.utils import DataFrameMatcher, SeriesMatcher


@patch('sdmetrics.reports.utils.ff')
def test_make_continuous_column_plot(ff_mock):
    """Test the ``make_continuous_column_plot`` method.

    Expect that it creates a distplot.

    Inputs:
    - real column data
    - synthetic column data
    - column data type

    Side Effects:
    - A distplot is created.
    """
    # Setup
    real_column = pd.Series([1, 2, 3, 4])
    synthetic_column = pd.Series([1, 2, 4, 5])
    sdtype = 'int'

    mock_figure = Mock()
    mock_real_data = Mock()
    mock_real_data.x = real_column
    mock_synthetic_data = Mock()
    mock_synthetic_data.x = synthetic_column
    mock_figure.data = (mock_real_data, mock_synthetic_data)
    ff_mock.create_distplot.return_value = mock_figure

    # Run
    fig = make_continuous_column_plot(real_column, synthetic_column, sdtype)

    # Assert
    ff_mock.create_distplot.assert_called_once_with(
        [SeriesMatcher(real_column), SeriesMatcher(synthetic_column)],
        ['Real', 'Synthetic'],
        show_hist=False,
        show_rug=False,
        colors=['#000036', '#01E0C9'],
    )
    assert mock_figure.update_traces.call_count == 2
    assert mock_figure.update_layout.called_once()
    assert fig == mock_figure


@patch('sdmetrics.reports.utils.ff')
def test_make_continuous_column_plot_datetime(ff_mock):
    """Test the ``make_contuous_column_plot`` method with datetime inputs.

    Expect that it creates a distplot.

    Inputs:
    - real column data
    - synthetic column data
    - column data type

    Side Effects:
    - A distplot is created.
    """
    # Setup
    real_column = pd.Series([
        datetime(2020, 5, 17),
        datetime(2021, 9, 1),
        datetime(2021, 8, 1),
    ])
    synthetic_column = pd.Series([
        datetime(2020, 9, 10),
        datetime(2021, 2, 1),
        datetime(2021, 10, 10),
    ])
    sdtype = 'datetime'

    mock_figure = Mock()
    mock_real_data = Mock()
    mock_real_data.x = real_column
    mock_synthetic_data = Mock()
    mock_synthetic_data.x = synthetic_column
    mock_figure.data = (mock_real_data, mock_synthetic_data)
    ff_mock.create_distplot.return_value = mock_figure

    # Run
    fig = make_continuous_column_plot(real_column, synthetic_column, sdtype)

    # Assert
    ff_mock.create_distplot.assert_called_once_with(
        [
            SeriesMatcher(real_column.astype('int64')),
            SeriesMatcher(synthetic_column.astype('int64')),
        ],
        ['Real', 'Synthetic'],
        show_hist=False,
        show_rug=False,
        colors=['#000036', '#01E0C9'],
    )
    assert mock_figure.update_traces.call_count == 2
    assert mock_figure.update_layout.called_once()
    assert fig == mock_figure


@patch('sdmetrics.reports.utils.px')
def test_make_discrete_column_plot(px_mock):
    """Test the ``make_discrete_column_plot`` method.

    Expect that it creates a histogram.

    Inputs:
    - real column data
    - synthetic column data
    - column data type

    Side Effects:
    - A distplot is created.
    """
    # Setup
    real_column = pd.Series([1, 2, 3, 4])
    synthetic_column = pd.Series([1, 2, 4, 5])
    sdtype = 'categorical'

    mock_figure = Mock()
    px_mock.histogram.return_value = mock_figure

    # Run
    fig = make_discrete_column_plot(real_column, synthetic_column, sdtype)

    # Assert
    px_mock.histogram.assert_called_once_with(
        DataFrameMatcher(pd.DataFrame({
            'values': [1, 2, 3, 4, 1, 2, 4, 5],
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
        x='values',
        color='Data',
        barmode='group',
        color_discrete_sequence=['#000036', '#01E0C9'],
        pattern_shape='Data',
        pattern_shape_sequence=['', '/'],
        histnorm='probability density',
    )
    assert mock_figure.update_traces.call_count == 2
    assert mock_figure.update_layout.called_once()
    assert fig == mock_figure


@patch('sdmetrics.reports.utils.make_continuous_column_plot')
def test_get_column_plot_continuous_col(make_plot_mock):
    """Test the ``get_column_plot`` method with a continuous column.

    Inputs:
    - real column data
    - synthetic column data
    - column data type

    Output:
    - column plot

    Side Effects:
    - The make continuous column plot method is called.
    """
    # Setup
    real_column = pd.Series([1, 2, 3, 4])
    synthetic_column = pd.Series([1, 2, 4, 5])
    sdtype = 'numerical'

    # Run
    out = get_column_plot(real_column, synthetic_column, sdtype)

    # Assert
    make_plot_mock.assert_called_once_with(real_column, synthetic_column, sdtype)
    assert out == make_plot_mock.return_value


@patch('sdmetrics.reports.utils.make_discrete_column_plot')
def test_get_column_plot_discrete_col(make_plot_mock):
    """Test the ``get_column_plot`` method with a discrete column.

    Inputs:
    - real column data
    - synthetic column data
    - column data type

    Output:
    - column plot

    Side Effects:
    - The make discrete column plot method is called.
    """
    # Setup
    real_column = pd.Series([1, 2, 3, 4])
    synthetic_column = pd.Series([1, 2, 4, 5])
    sdtype = 'categorical'

    # Run
    out = get_column_plot(real_column, synthetic_column, sdtype)

    # Assert
    make_plot_mock.assert_called_once_with(real_column, synthetic_column, sdtype)
    assert out == make_plot_mock.return_value


def test_discretize_table_data():
    """Test the ``discretize_table_data`` method.

    Expect that numerical and datetime fields are discretized.

    Input:
    - real data
    - synthetic data
    - metadata

    Output:
    - discretized real data
    - discretized synthetic data
    - updated metadata
    """
    # Setup
    real_data = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c'],
        'col3': [datetime(2020, 1, 2), datetime(2019, 10, 1), datetime(2021, 3, 2)],
        'col4': [True, False, True],
        'col5': [date(2020, 1, 2), date(2010, 10, 12), date(2021, 1, 2)],
    })
    synthetic_data = pd.DataFrame({
        'col1': [3, 1, 4],
        'col2': ['c', 'a', 'c'],
        'col3': [datetime(2021, 3, 2), datetime(2018, 11, 2), datetime(2020, 5, 7)],
        'col4': [False, False, True],
        'col5': [date(2020, 5, 3), date(2015, 11, 15), date(2022, 3, 2)],
    })
    metadata = {
        'fields': {
            'col1': {'type': 'numerical'},
            'col2': {'type': 'categorical'},
            'col3': {'type': 'datetime'},
            'col4': {'type': 'boolean'},
            'col5': {'type': 'datetime', 'format': '%Y-%m-%d'},
        },
    }

    # Run
    discretized_real, discretized_synth, updated_metadata = discretize_table_data(
        real_data, synthetic_data, metadata)

    # Assert
    expected_real = pd.DataFrame({
        'col1': [1, 6, 11],
        'col2': ['a', 'b', 'c'],
        'col3': [2, 1, 11],
        'col4': [True, False, True],
        'col5': [10, 1, 11],
    })
    expected_synth = pd.DataFrame({
        'col1': [11, 1, 11],
        'col2': ['c', 'a', 'c'],
        'col3': [11, 0, 5],
        'col4': [False, False, True],
        'col5': [10, 5, 11],
    })

    pd.testing.assert_frame_equal(discretized_real, expected_real)
    pd.testing.assert_frame_equal(discretized_synth, expected_synth)
    assert updated_metadata == {
        'fields': {
            'col1': {'type': 'categorical'},
            'col2': {'type': 'categorical'},
            'col3': {'type': 'categorical'},
            'col4': {'type': 'boolean'},
            'col5': {'type': 'categorical'},
        },
    }


@patch('sdmetrics.reports.utils.discretize_table_data')
def test_discretize_and_apply_metric(discretize_table_data_mock):
    """Test the ``discretize_and_apply_metric`` method.

    Expect that the correct calls to ``compute_breakdown`` are made.

    Input:
    - real data
    - synthetic data
    - metadata
    - metric

    Output:
    - metric results
    """
    # Setup
    binned_real = pd.DataFrame({
        'col1': [1, 6, 11],
        'col2': ['a', 'b', 'c'],
        'col3': [2, 1, 11],
        'col4': [True, False, True],
    })
    binned_synthetic = pd.DataFrame({
        'col1': [11, 1, 11],
        'col2': ['c', 'a', 'c'],
        'col3': [11, 0, 5],
        'col4': [False, False, True],
    })
    metadata = {
        'fields': {
            'col1': {'type': 'numerical'},
            'col2': {'type': 'categorical'},
            'col3': {'type': 'datetime'},
            'col4': {'type': 'boolean'},
        },
    }
    discretize_table_data_mock.return_value = (binned_real, binned_synthetic, metadata)
    mock_metric = Mock()
    mock_metric.column_pairs_metric.compute_breakdown.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    # Run
    metric_results = discretize_and_apply_metric(Mock(), Mock(), metadata, mock_metric)

    # Assert
    mock_metric.column_pairs_metric.compute_breakdown.assert_has_calls([
        call(
            DataFrameMatcher(binned_real[['col1', 'col2']]),
            DataFrameMatcher(binned_synthetic[['col1', 'col2']]),
        ),
        call(
            DataFrameMatcher(binned_real[['col1', 'col3']]),
            DataFrameMatcher(binned_synthetic[['col1', 'col3']]),
        ),
        call(
            DataFrameMatcher(binned_real[['col1', 'col4']]),
            DataFrameMatcher(binned_synthetic[['col1', 'col4']]),
        ),
        call(
            DataFrameMatcher(binned_real[['col2', 'col3']]),
            DataFrameMatcher(binned_synthetic[['col2', 'col3']]),
        ),
        call(
            DataFrameMatcher(binned_real[['col2', 'col4']]),
            DataFrameMatcher(binned_synthetic[['col2', 'col4']]),
        ),
        call(
            DataFrameMatcher(binned_real[['col3', 'col4']]),
            DataFrameMatcher(binned_synthetic[['col3', 'col4']]),
        ),
    ])
    assert metric_results == {
        ('col1', 'col2'): 0.1,
        ('col1', 'col3'): 0.2,
        ('col1', 'col4'): 0.3,
        ('col2', 'col3'): 0.4,
        ('col2', 'col4'): 0.5,
        ('col3', 'col4'): 0.6,
    }


def test_aggregate_metric_results():
    """Test the ``aggregate_metric_results`` method.

    Expect that the aggregated results are returned.

    Input:
    - metric results

    Output:
    - average score
    - number of errors
    """
    # Setup
    metric_results = {
        'col1': {'score': 0.1},
        'col2': {'score': 0.8},
        'col3': {'error': 'test error'},
    }

    # Run
    avg_score, num_errors = aggregate_metric_results(metric_results)

    # Assert
    assert avg_score == 0.45
    assert num_errors == 1
