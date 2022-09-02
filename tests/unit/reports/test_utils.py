from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd

from sdmetrics.reports.utils import plot_column
from tests.utils import SeriesMatcher


@patch('sdmetrics.reports.utils.ff')
def test_plot_column(ff_mock):
    """Test the ``plot_column`` method.

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
    plot_column(real_column, synthetic_column, sdtype)

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
    mock_figure.show.assert_called_once_with()


@patch('sdmetrics.reports.utils.ff')
def test_plot_column_datetime(ff_mock):
    """Test the ``plot_column`` method with datetime inputs.

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
    plot_column(real_column, synthetic_column, sdtype)

    # Assert
    ff_mock.create_distplot.assert_called_once_with(
        [SeriesMatcher(real_column.view(int)), SeriesMatcher(synthetic_column.view(int))],
        ['Real', 'Synthetic'],
        show_hist=False,
        show_rug=False,
        colors=['#000036', '#01E0C9'],
    )
    assert mock_figure.update_traces.call_count == 2
    assert mock_figure.update_layout.called_once()
    mock_figure.show.assert_called_once_with()
