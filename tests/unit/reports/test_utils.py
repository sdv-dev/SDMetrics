from datetime import date, datetime
from unittest.mock import Mock, call, patch

import pandas as pd

from sdmetrics.reports.utils import (
    aggregate_metric_results, convert_to_datetime, discretize_and_apply_metric,
    discretize_table_data)
from tests.utils import DataFrameMatcher


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
