import re
import warnings
from datetime import date, datetime
from unittest.mock import Mock, call, patch

import pandas as pd
import pytest

from sdmetrics.reports.utils import (
    _validate_categorical_values, aggregate_metric_results, convert_to_datetime,
    discretize_and_apply_metric, discretize_table_data, get_column_pair_plot, get_column_plot,
    make_continuous_column_pair_plot, make_continuous_column_plot, make_discrete_column_pair_plot,
    make_discrete_column_plot, make_mixed_column_pair_plot, validate_multi_table_inputs,
    validate_single_table_inputs)
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
    - real data
    - synthetic data
    - column name
    - metadata

    Output:
    - column plot

    Side Effects:
    - The make continuous column plot method is called.
    """
    # Setup
    sdtype = 'numerical'
    real_data = pd.DataFrame({'col1': [1, 2, 3, 4]})
    synthetic_data = pd.DataFrame({'col1': [1, 2, 4, 5]})
    metadata = {'columns': {'col1': {'sdtype': sdtype}}}

    # Run
    out = get_column_plot(real_data, synthetic_data, 'col1', metadata)

    # Assert
    make_plot_mock.assert_called_once_with(real_data['col1'], synthetic_data['col1'], sdtype)
    assert out == make_plot_mock.return_value


@patch('sdmetrics.reports.utils.make_discrete_column_plot')
def test_get_column_plot_discrete_col(make_plot_mock):
    """Test the ``get_column_plot`` method with a discrete column.

    Inputs:
    - real data
    - synthetic data
    - column name
    - metadata

    Output:
    - column plot

    Side Effects:
    - The make discrete column plot method is called.
    """
    # Setup
    sdtype = 'categorical'
    real_data = pd.DataFrame({'col1': [1, 2, 3, 4]})
    synthetic_data = pd.DataFrame({'col1': [1, 2, 4, 5]})
    metadata = {'columns': {'col1': {'sdtype': sdtype}}}

    # Run
    out = get_column_plot(real_data, synthetic_data, 'col1', metadata)

    # Assert
    make_plot_mock.assert_called_once_with(real_data['col1'], synthetic_data['col1'], sdtype)
    assert out == make_plot_mock.return_value


@patch('sdmetrics.reports.utils.make_continuous_column_plot')
def test_get_column_plot_datetime_col(make_plot_mock):
    """Test the ``get_column_plot`` method with a string datetime column."""
    # Setup
    sdtype = 'datetime'
    datetime_format = '%Y-%m-%d'
    real_datetimes = [
        datetime(2020, 10, 1),
        datetime(2020, 11, 1),
        datetime(2020, 12, 1),
    ]
    real_data = pd.DataFrame({
        'col1': [dt.strftime(datetime_format) for dt in real_datetimes]
    })
    real_expected = pd.DataFrame({'col1': real_datetimes})
    synthetic_datetimes = [
        datetime(2021, 10, 1),
        datetime(2021, 11, 1),
        datetime(2021, 12, 3),
    ]
    synthetic_data = pd.DataFrame({
        'col1': [dt.strftime(datetime_format) for dt in synthetic_datetimes]
    })
    synthetic_expected = pd.DataFrame({'col1': synthetic_datetimes})
    metadata = {'columns': {'col1': {'sdtype': sdtype, 'format': datetime_format}}}

    # Run
    out = get_column_plot(real_data, synthetic_data, 'col1', metadata)

    # Assert
    make_plot_mock.assert_called_once_with(SeriesMatcher(real_expected['col1']),
                                           SeriesMatcher(synthetic_expected['col1']),
                                           sdtype)
    assert out == make_plot_mock.return_value


def test_get_column_plot_invalid_sdtype():
    """Test the ``get_column_plot`` method with an invalid sdtype.

    Inputs:
    - real data
    - synthetic data
    - column name
    - metadata

    Side Effects:
    - A ValueError is raised.
    """
    # Setup
    real_data = pd.DataFrame({'col1': [1, 2, 3, 4]})
    synthetic_data = pd.DataFrame({'col1': [1, 2, 4, 5]})
    metadata = {'columns': {'col1': {'sdtype': 'invalid'}}}

    # Run and assert
    with pytest.raises(ValueError, match="sdtype of type 'invalid' not recognized"):
        get_column_plot(real_data, synthetic_data, 'col1', metadata)


def test_get_column_plot_missing_column_name_metadata():
    """Test the ``get_column_plot`` method with an incomplete metadata.

    Inputs:
    - real data
    - synthetic data
    - column name
    - metadata

    Side Effects:
    - A ValueError is raised.
    """
    # Setup
    real_data = pd.DataFrame({'col1': [1, 2, 3, 4]})
    synthetic_data = pd.DataFrame({'col1': [1, 2, 4, 5]})
    metadata = {'columns': {}}

    # Run and assert
    with pytest.raises(ValueError, match="Column 'col1' not found in metadata."):
        get_column_plot(real_data, synthetic_data, 'col1', metadata)


def test_get_column_plot_missing_column_type_metadata():
    """Test the ``get_column_plot`` method with an incomplete metadata.

    Inputs:
    - real data
    - synthetic data
    - column name
    - metadata

    Side Effects:
    - A ValueError is raised.
    """
    # Setup
    real_data = pd.DataFrame({'col1': [1, 2, 3, 4]})
    synthetic_data = pd.DataFrame({'col1': [1, 2, 4, 5]})
    metadata = {'columns': {'col1': {}}}

    # Run and assert
    with pytest.raises(ValueError, match="Metadata for column 'col1' missing 'type' information."):
        get_column_plot(real_data, synthetic_data, 'col1', metadata)


def test_get_column_plot_missing_column_name_data():
    """Test the ``get_column_plot`` method with an incomplete real data.

    Inputs:
    - real data
    - synthetic data
    - column name
    - metadata

    Side Effects:
    - A ValueError is raised.
    """
    # Setup
    real_data = pd.DataFrame({'col2': [1, 2, 3, 4]})
    synthetic_data = pd.DataFrame({'col1': [1, 2, 4, 5]})
    metadata = {'columns': {'col1': {'sdtype': 'invalid'}}}

    # Run and assert
    with pytest.raises(ValueError, match="Column 'col1' not found in real table data."):
        get_column_plot(real_data, synthetic_data, 'col1', metadata)


def test_get_column_plot_missing_column_name_synthetic_data():
    """Test the ``get_column_plot`` method with an incomplete synthetic data.

    Inputs:
    - real data
    - synthetic data
    - column name
    - metadata

    Side Effects:
    - A ValueError is raised.
    """
    # Setup
    real_data = pd.DataFrame({'col1': [1, 2, 3, 4]})
    synthetic_data = pd.DataFrame({'col2': [1, 2, 4, 5]})
    metadata = {'columns': {'col1': {'sdtype': 'invalid'}}}

    # Run and assert
    with pytest.raises(ValueError, match="Column 'col1' not found in synthetic table data."):
        get_column_plot(real_data, synthetic_data, 'col1', metadata)


def test_convert_to_datetime():
    """Test the ``convert_to_datetime`` method with a datetime column.

    Expect no conversion to happen since the input is already a pandas datetime type.

    Inputs:
    - datetime column

    Output:
    - datetime column
    """
    # Setup
    column_data = pd.Series([datetime(2020, 1, 2), datetime(2021, 1, 2)])

    # Run
    out = convert_to_datetime(column_data)

    # Assert
    pd.testing.assert_series_equal(out, column_data)


def test_convert_to_datetime_date_column():
    """Test the ``convert_to_datetime`` method with a date column.

    Expect the date column to be converted to a datetime column.

    Inputs:
    - date column

    Output:
    - datetime column
    """
    # Setup
    column_data = pd.Series([date(2020, 1, 2), date(2021, 1, 2)])

    # Run
    out = convert_to_datetime(column_data)

    # Assert
    expected = pd.Series([datetime(2020, 1, 2), datetime(2021, 1, 2)])
    pd.testing.assert_series_equal(out, expected)


def test_convert_to_datetime_str_format():
    """Test the ``convert_to_datetime`` method with a string column.

    Expect the string date column to be converted to a datetime column
    using the provided format.
    """
    # Setup
    column_data = pd.Series(['2020-01-02', '2021-01-02'])

    # Run
    out = convert_to_datetime(column_data)

    # Assert
    expected = pd.Series([datetime(2020, 1, 2), datetime(2021, 1, 2)])
    pd.testing.assert_series_equal(out, expected)


@patch('sdmetrics.reports.utils.px')
def test_make_continuous_column_pair_plot(px_mock):
    """Test the ``make_continuous_column_pair_plot`` method.

    Expect that it creates a scatter plot.

    Inputs:
    - real column data
    - synthetic column data
    - column data type

    Side Effects:
    - A distplot is created.
    """
    # Setup
    real_column = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [1.1, 1.2, 1.3, 1.4]})
    synthetic_column = pd.DataFrame({'col1': [1, 2, 4, 5], 'col2': [1.1, 1.2, 1.3, 1.4]})

    mock_figure = Mock()
    px_mock.scatter.return_value = mock_figure

    # Run
    fig = make_continuous_column_pair_plot(real_column, synthetic_column)

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


@patch('sdmetrics.reports.utils.px')
def test_make_discrete_column_pair_plot(px_mock):
    """Test the ``make_discrete_column_pair_plot`` method.

    Expect that it creates a heatmap.

    Inputs:
    - real column data
    - synthetic column data
    - column data type

    Side Effects:
    - A distplot is created.
    """
    # Setup
    real_column = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': ['a', 'b', 'c', 'd']})
    synthetic_column = pd.DataFrame({'col1': [1, 2, 4, 5], 'col2': ['a', 'b', 'c', 'd']})

    mock_figure = Mock()
    px_mock.density_heatmap.return_value = mock_figure

    # Run
    fig = make_discrete_column_pair_plot(real_column, synthetic_column)

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


@patch('sdmetrics.reports.utils.px')
def test_make_mixed_column_pair_plot(px_mock):
    """Test the ``make_mixed_column_pair_plot`` method.

    Expect that it creates a box plot.

    Inputs:
    - real column data
    - synthetic column data
    - column data type

    Side Effects:
    - A distplot is created.
    """
    # Setup
    real_column = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': ['a', 'b', 'c', 'd']})
    synthetic_column = pd.DataFrame({'col1': [1, 2, 4, 5], 'col2': ['a', 'b', 'c', 'd']})

    mock_figure = Mock()
    px_mock.box.return_value = mock_figure

    # Run
    fig = make_mixed_column_pair_plot(real_column, synthetic_column)

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


@patch('sdmetrics.reports.utils.make_continuous_column_pair_plot')
def test_get_column_pair_plot_continuous_columns(make_plot_mock):
    """Test the ``get_column_plot_pair`` method with continuous sdtypes.

    Inputs:
    - real data
    - synthetic data
    - column names
    - metadata

    Outputs:
    - The column pair plot.
    """
    # Setup
    real_data = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [datetime(2020, 10, 1), datetime(2020, 11, 1), datetime(2021, 1, 2)],
    })
    synthetic_data = pd.DataFrame({
        'col1': [2, 2, 3],
        'col2': [datetime(2021, 10, 1), datetime(2021, 11, 1), datetime(2021, 12, 3)],
    })
    columns = ['col1', 'col2']
    metadata = {'columns': {'col1': {'sdtype': 'numerical'}, 'col2': {'sdtype': 'datetime'}}}

    # Run
    out = get_column_pair_plot(real_data, synthetic_data, columns, metadata)

    # Assert
    make_plot_mock.assert_called_once_with(
        DataFrameMatcher(real_data[columns]),
        DataFrameMatcher(synthetic_data[columns]),
    )
    assert out == make_plot_mock.return_value


@patch('sdmetrics.reports.utils.make_mixed_column_pair_plot')
def test_get_column_pair_plot_mixed_columns(make_plot_mock):
    """Test the ``get_column_plot_pair`` method with mixed sdtypes.

    Inputs:
    - real data
    - synthetic data
    - column names
    - metadata

    Outputs:
    - The column pair plot.
    """
    # Setup
    real_data = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [
            datetime(2020, 10, 1),
            datetime(2020, 11, 1),
            datetime(2020, 12, 1),
        ],
    })
    synthetic_data = pd.DataFrame({
        'col1': [2, 2, 3],
        'col2': [
            datetime(2021, 10, 1),
            datetime(2021, 11, 1),
            datetime(2021, 12, 3),
        ],
    })
    columns = ['col1', 'col2']
    metadata = {'columns': {'col1': {'sdtype': 'categorical'}, 'col2': {'sdtype': 'datetime'}}}

    # Run
    out = get_column_pair_plot(real_data, synthetic_data, columns, metadata)

    # Assert
    make_plot_mock.assert_called_once_with(
        DataFrameMatcher(real_data[columns]),
        DataFrameMatcher(synthetic_data[columns]),
    )
    assert out == make_plot_mock.return_value


@patch('sdmetrics.reports.utils.make_discrete_column_pair_plot')
def test_get_column_pair_plot_discrete_columns(make_plot_mock):
    """Test the ``get_column_plot_pair`` method with discrete sdtypes.

    Inputs:
    - real data
    - synthetic data
    - column names
    - metadata

    Outputs:
    - The column pair plot.
    """
    # Setup
    real_data = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [True, False, True],
    })
    synthetic_data = pd.DataFrame({
        'col1': [2, 2, 3],
        'col2': [False, False, False],
    })
    columns = ['col1', 'col2']
    metadata = {'columns': {'col1': {'sdtype': 'categorical'}, 'col2': {'sdtype': 'boolean'}}}

    # Run
    out = get_column_pair_plot(real_data, synthetic_data, columns, metadata)

    # Assert
    make_plot_mock.assert_called_once_with(
        DataFrameMatcher(real_data[columns]),
        DataFrameMatcher(synthetic_data[columns]),
    )
    assert out == make_plot_mock.return_value


@patch('sdmetrics.reports.utils.make_mixed_column_pair_plot')
def test_get_column_pair_plot_str_datetimes(make_plot_mock):
    """Test the ``get_column_pair_plot`` method with string datetime columns.

    Expect that the string datetime columns are converted to datetimes.
    """
    # Setup
    dt_format = '%Y-%m-%d'
    real_datetimes = [
        datetime(2020, 10, 1),
        datetime(2020, 11, 1),
        datetime(2020, 12, 1),
    ]
    real_data = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [dt.strftime(dt_format) for dt in real_datetimes],
    })
    real_expected = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': real_datetimes,
    })

    synthetic_datetimes = [
        datetime(2021, 10, 1),
        datetime(2021, 11, 1),
        datetime(2021, 12, 3),
    ]
    synthetic_data = pd.DataFrame({
        'col1': [2, 2, 3],
        'col2': [dt.strftime(dt_format) for dt in synthetic_datetimes],
    })
    synthetic_expected = pd.DataFrame({
        'col1': [2, 2, 3],
        'col2': synthetic_datetimes,
    })
    columns = ['col1', 'col2']
    metadata = {
        'columns': {
            'col1': {'sdtype': 'categorical'},
            'col2': {'sdtype': 'datetime', 'format': dt_format}
        }
    }

    # Run
    out = get_column_pair_plot(real_data, synthetic_data, columns, metadata)

    # Assert
    make_plot_mock.assert_called_once_with(
        DataFrameMatcher(real_expected[columns]),
        DataFrameMatcher(synthetic_expected[columns]),
    )
    assert out == make_plot_mock.return_value


def test_get_column_pair_plot_invalid_sdtype():
    """Test the ``get_column_plot_pair`` method with an invalid sdtype.

    Inputs:
    - real data
    - synthetic data
    - column names
    - metadata

    Side Effects:
    - A ValueError is raised.
    """
    # Setup
    real_data = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [True, False, True],
    })
    synthetic_data = pd.DataFrame({
        'col1': [2, 2, 3],
        'col2': [False, False, False],
    })
    columns = ['col1', 'col2']
    metadata = {'columns': {'col1': {'sdtype': 'numerical'}, 'col2': {'sdtype': 'invalid'}}}

    # Run and assert
    with pytest.raises(ValueError, match=re.escape('sdtype(s) of type `invalid` not recognized.')):
        get_column_pair_plot(real_data, synthetic_data, columns, metadata)


def test_get_column_pair_plot_missing_column_metadata():
    """Test the ``get_column_plot_pair`` method with a missing column in the metadata.

    Inputs:
    - real data
    - synthetic data
    - column names
    - metadata

    Side Effects:
    - A ValueError is raised.
    """
    # Setup
    real_data = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [True, False, True],
    })
    synthetic_data = pd.DataFrame({
        'col1': [2, 2, 3],
        'col2': [False, False, False],
    })
    columns = ['col1', 'col2']
    metadata = {'columns': {'col1': {'sdtype': 'numerical'}}}

    # Run and assert
    with pytest.raises(ValueError, match=re.escape('Column(s) `col2` not found in metadata.')):
        get_column_pair_plot(real_data, synthetic_data, columns, metadata)


def test_get_column_pair_plot_missing_column_type_metadata():
    """Test the ``get_column_plot_pair`` method with a missing column type in the metadata.

    Inputs:
    - real data
    - synthetic data
    - column names
    - metadata

    Side Effects:
    - A ValueError is raised.
    """
    # Setup
    real_data = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [True, False, True],
    })
    synthetic_data = pd.DataFrame({
        'col1': [2, 2, 3],
        'col2': [False, False, False],
    })
    columns = ['col1', 'col2']
    metadata = {'columns': {'col1': {'sdtype': 'numerical'}, 'col2': {}}}

    # Run and assert
    with pytest.raises(
        ValueError,
        match=re.escape("Metadata for column(s) `col2` missing 'type' information."),
    ):
        get_column_pair_plot(real_data, synthetic_data, columns, metadata)


def test_get_column_pair_plot_missing_columns_metadata():
    """Test the ``get_column_plot_pair`` method with two missing columns in the metadata.

    Inputs:
    - real data
    - synthetic data
    - column names
    - metadata

    Side Effects:
    - A ValueError is raised.
    """
    # Setup
    real_data = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [True, False, True],
    })
    synthetic_data = pd.DataFrame({
        'col1': [2, 2, 3],
        'col2': [False, False, False],
    })
    columns = ['col1', 'col2']
    metadata = {'columns': {}}

    # Run and assert
    with pytest.raises(
        ValueError,
        match=re.escape('Column(s) `col1`, `col2` not found in metadata.'),
    ):
        get_column_pair_plot(real_data, synthetic_data, columns, metadata)


def test_get_column_pair_plot_missing_column_real_data():
    """Test the ``get_column_plot_pair`` method with a missing column in the real data.

    Inputs:
    - real data
    - synthetic data
    - column names
    - metadata

    Side Effects:
    - A ValueError is raised.
    """
    # Setup
    real_data = pd.DataFrame({
        'col2': [True, False, True],
    })
    synthetic_data = pd.DataFrame({
        'col1': [2, 2, 3],
        'col2': [False, False, False],
    })
    columns = ['col1', 'col2']
    metadata = {'columns': {'col1': {'sdtype': 'categorical'}, 'col2': {'sdtype': 'boolean'}}}

    # Run and assert
    with pytest.raises(
        ValueError,
        match=re.escape('Column(s) `col1` not found in the real table data.'),
    ):
        get_column_pair_plot(real_data, synthetic_data, columns, metadata)


def test_get_column_pair_plot_missing_column_synthetic_data():
    """Test the ``get_column_plot_pair`` method with a missing column in the synthetic data.

    Inputs:
    - real data
    - synthetic data
    - column names
    - metadata

    Side Effects:
    - A ValueError is raised.
    """
    # Setup
    real_data = pd.DataFrame({
        'col1': [2, 2, 3],
        'col2': [True, False, True],
    })
    synthetic_data = pd.DataFrame({
        'col1': [2, 2, 3],
    })
    columns = ['col1', 'col2']
    metadata = {'columns': {'col1': {'sdtype': 'categorical'}, 'col2': {'sdtype': 'boolean'}}}

    # Run and assert
    with pytest.raises(
        ValueError,
        match=re.escape('Column(s) `col2` not found in the synthetic table data.'),
    ):
        get_column_pair_plot(real_data, synthetic_data, columns, metadata)


def test_discretize_table_data():
    """Test the ``discretize_table_data`` method.

    Expect that numerical and datetime columns are discretized.

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
        'columns': {
            'col1': {'sdtype': 'numerical'},
            'col2': {'sdtype': 'categorical'},
            'col3': {'sdtype': 'datetime'},
            'col4': {'sdtype': 'boolean'},
            'col5': {'sdtype': 'datetime', 'format': '%Y-%m-%d'},
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
        'columns': {
            'col1': {'sdtype': 'categorical'},
            'col2': {'sdtype': 'categorical'},
            'col3': {'sdtype': 'categorical'},
            'col4': {'sdtype': 'boolean'},
            'col5': {'sdtype': 'categorical'},
        },
    }


def test_discretize_table_data_new_metadata():
    """Test the ``discretize_table_data`` method with new metadata.

    Expect that numerical and datetime columns are discretized.

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
        'columns': {
            'col1': {'sdtype': 'numerical'},
            'col2': {'sdtype': 'categorical'},
            'col3': {'sdtype': 'datetime'},
            'col4': {'sdtype': 'boolean'},
            'col5': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
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
        'columns': {
            'col1': {'sdtype': 'categorical'},
            'col2': {'sdtype': 'categorical'},
            'col3': {'sdtype': 'categorical'},
            'col4': {'sdtype': 'boolean'},
            'col5': {'sdtype': 'categorical'},
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
        'col5': ['', '', ''],
        'col6': ['', '', ''],
    })
    binned_synthetic = pd.DataFrame({
        'col1': [11, 1, 11],
        'col2': ['c', 'a', 'c'],
        'col3': [11, 0, 5],
        'col4': [False, False, True],
        'col5': ['', '', ''],
        'col6': ['', '', ''],
    })
    metadata = {
        'columns': {
            'col1': {'sdtype': 'numerical'},
            'col2': {'sdtype': 'categorical'},
            'col3': {'sdtype': 'datetime'},
            'col4': {'sdtype': 'boolean'},
            'col5': {'sdtype': 'address'},
            'col6': {'sdtype': 'text'},
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


def test__validate_categorical_values():
    """Test no extra categoricals does not crash."""
    # Setup
    sdtype = 'categorical'
    real_data = pd.DataFrame({'col1': [1, 2, 3, 4]})
    synthetic_data = pd.DataFrame({'col1': [1, 2, 4, 4]})
    metadata = {'columns': {'col1': {'sdtype': sdtype}}}
    warnings.filterwarnings('error', category=UserWarning)

    # Run
    _validate_categorical_values(real_data, synthetic_data, metadata)

    warnings.resetwarnings()


def test__validate_categorical_values_single_table():
    """Test validating categoricals for single table."""
    # Setup
    sdtype = 'categorical'
    real_data = pd.DataFrame({'col1': [1, 2, 3, 3]})
    synthetic_data = pd.DataFrame({'col1': [1, 2, 4, 5]})
    metadata = {'columns': {'col1': {'sdtype': sdtype}}}
    warning_msg = re.escape('Unexpected values ("4", "5") in column "col1"')

    warnings.filterwarnings('error', category=UserWarning)

    # Run
    with pytest.raises(UserWarning, match=warning_msg):
        _validate_categorical_values(real_data, synthetic_data, metadata)

    warnings.resetwarnings()


def test__validate_categorical_values_multi_table():
    """Test validating categoricals with table name."""
    # Setup
    sdtype = 'categorical'
    real_data = pd.DataFrame({'col1': [1, 2, 3, 3]})
    synthetic_data = pd.DataFrame({'col1': [1, 2, 4, 5]})
    metadata = {'columns': {'col1': {'sdtype': sdtype}}}
    warning_msg = re.escape('Unexpected values ("4", "5") in column "col1" and table "table1"')

    warnings.filterwarnings('error', category=UserWarning)

    # Run
    with pytest.raises(UserWarning, match=warning_msg):
        _validate_categorical_values(real_data, synthetic_data, metadata, table='table1')

    warnings.resetwarnings()


def test__validate_categorical_many_extra_values():
    """Test validating categoricals with table name."""
    # Setup
    sdtype = 'categorical'
    real_data = pd.DataFrame({'col1': [1, 2, 3, 3, 4, 4]})
    synthetic_data = pd.DataFrame({'col1': ['a', 'b', 'c', 'd', 'e', 'f']})
    metadata = {'columns': {'col1': {'sdtype': sdtype}}}
    warning_msg = re.escape('Unexpected values ("a", "b", "c", "d", "e" + more) in column "col1"')

    warnings.filterwarnings('error', category=UserWarning)

    # Run
    with pytest.raises(UserWarning, match=warning_msg):
        _validate_categorical_values(real_data, synthetic_data, metadata)

    warnings.resetwarnings()


def test_validate_single_table():
    """Test validating single table."""
    # Setup
    sdtype = 'categorical'
    real_data = pd.DataFrame({'col1': [1, 2, 3, 4]})
    synthetic_data = pd.DataFrame({'col1': [1, 1, 1, 3]})
    extra_value_synthtetic_data = pd.DataFrame({'col1': [1, 1, 1, 5]})
    metadata = {'columns': {'col1': {'sdtype': sdtype}}}

    warning_msg = re.escape('Unexpected values ("5") in column "col1"')
    warnings.filterwarnings('error', category=UserWarning)

    # Run
    validate_single_table_inputs(real_data, synthetic_data, metadata)

    with pytest.raises(UserWarning, match=warning_msg):
        validate_single_table_inputs(real_data, extra_value_synthtetic_data, metadata)

    warnings.resetwarnings()


def test_validate_multi_table():
    """Test validating categoricals for single table."""
    # Setup
    sdtype = 'categorical'
    real_data = {
        'table1': pd.DataFrame({'col1': [1, 2, 3, 3]}),
        'table2': pd.DataFrame({'col2': ['a', 'b', 'c', 'a']})
    }
    synthetic_data = {
        'table1': pd.DataFrame({'col1': [1, 2, 3, 1]}),
        'table2': pd.DataFrame({'col2': ['a', 'b', 'c', 'c']})
    }
    extra_value_synthetic_data = {
        'table1': pd.DataFrame({'col1': [1, 2, 3, 1]}),
        'table2': pd.DataFrame({'col2': ['a', 'b', 'c', 'd']})
    }
    metadata = {
        'tables': {
            'table1': {'columns': {'col1': {'sdtype': sdtype}}},
            'table2': {'columns': {'col2': {'sdtype': sdtype}}}
        }
    }
    warning_msg = re.escape('Unexpected values ("d") in column "col2" and table "table2"')

    warnings.filterwarnings('error', category=UserWarning)

    # Run
    validate_multi_table_inputs(real_data, synthetic_data, metadata)
    with pytest.raises(UserWarning, match=warning_msg):
        validate_multi_table_inputs(real_data, extra_value_synthetic_data, metadata)

    warnings.resetwarnings()


def test_validate_multi_table_parent_child_dtype_mismatch():
    """Test validating categoricals for single table."""
    # Setup
    sdtype = 'numerical'
    real_data = {
        'table1': pd.DataFrame({'primary_key': [1, 2, 3, 4]}),
        'table2': pd.DataFrame({'foreign_key': ['1', '2', '2', '3']})
    }
    synthetic_data = {
        'table1': pd.DataFrame({'primary_key': [1, 2, 3, 1]}),
        'table2': pd.DataFrame({'foreign_key': ['2', '2', '1', '3']})
    }
    metadata = {
        'tables': {
            'table1': {'columns': {'primary_key': {'sdtype': sdtype}}},
            'table2': {'columns': {'foreign_key': {'sdtype': sdtype}}}
        },
        'relationships': [
            {
                'parent_table_name': 'table1',
                'child_table_name': 'table2',
                'parent_primary_key': 'primary_key',
                'child_foreign_key': 'foreign_key'
            }
        ]
    }
    error_msg = re.escape(
        "The 'table1' table and 'table2' table cannot be merged. Please make sure the primary key "
        "in 'table1' ('primary_key') and the foreign key in 'table2' ('foreign_key') have the same"
        ' data type.'
    )

    # Run and Assert
    with pytest.raises(ValueError, match=error_msg):
        validate_multi_table_inputs(real_data, synthetic_data, metadata)
