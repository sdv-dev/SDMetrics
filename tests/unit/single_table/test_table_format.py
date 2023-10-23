from unittest.mock import patch

import pandas as pd
import pytest

from sdmetrics.single_table import TableFormat


@pytest.fixture()
def real_data():
    return pd.DataFrame({
        'col_1': [1, 2, 3, 4, 5],
        'col_2': ['A', 'B', 'C', 'B', 'A'],
        'col_3': [True, False, True, False, True],
        'col_4': pd.to_datetime([
            '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'
        ]),
        'col_5': [1.0, 2.0, 3.0, 4.0, 5.0]
    })


class TestTableFormat:

    def test_compute_breakdown(self, real_data):
        """Test the ``compute_breakdown`` method."""
        # Setup
        synthetic_data = pd.DataFrame({
            'col_1': [3, 2, 1, 4, 5],
            'col_2': ['A', 'B', 'C', 'D', 'E'],
            'col_3': [True, False, True, False, True],
            'col_4': pd.to_datetime([
                '2020-01-11', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'
            ]),
            'col_5': [4.0, 2.0, 3.0, 4.0, 5.0]
        })

        metric = TableFormat()

        # Run
        result = metric.compute_breakdown(real_data, synthetic_data)

        # Assert
        expected_result = {
            'score': 1.0,
        }
        assert result == expected_result

    def test_compute_breakdown_with_missing_columns(self, real_data):
        """Test the ``compute_breakdown`` method with missing columns."""
        # Setup
        synthetic_data = pd.DataFrame({
            'col_1': [3, 2, 1, 4, 5],
            'col_2': ['A', 'B', 'C', 'D', 'E'],
            'col_3': [True, False, True, False, True],
            'col_4': pd.to_datetime([
                '2020-01-11', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'
            ]),
        })

        metric = TableFormat()

        # Run
        result = metric.compute_breakdown(real_data, synthetic_data)

        # Assert
        expected_result = {
            'score': 0.8,
            'missing columns in synthetic data': ['col_5']
        }
        assert result == expected_result

    def test_compute_breakdown_with_invalid_names(self, real_data):
        """Test the ``compute_breakdown`` method with invalid names."""
        # Setup
        synthetic_data = pd.DataFrame({
            'col_1': [3, 2, 1, 4, 5],
            'col_2': ['A', 'B', 'C', 'D', 'E'],
            'col_3': [True, False, True, False, True],
            'col_4': pd.to_datetime([
                '2020-01-11', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'
            ]),
            'col_5': [4.0, 2.0, 3.0, 4.0, 5.0],
            'col_6': [4.0, 2.0, 3.0, 4.0, 5.0],
        })

        metric = TableFormat()

        # Run
        result = metric.compute_breakdown(real_data, synthetic_data)

        # Assert
        expected_result = {
            'score': 0.8333333333333334,
            'invalid column names': ['col_6']
        }
        assert result == expected_result

    def test_compute_breakdown_with_invalid_dtypes(self, real_data):
        """Test the ``compute_breakdown`` method with invalid dtypes."""
        # Setup
        synthetic_data = pd.DataFrame({
            'col_1': [3.0, 2.0, 1.0, 4.0, 5.0],
            'col_2': ['A', 'B', 'C', 'D', 'E'],
            'col_3': [True, False, True, False, True],
            'col_4': [
                '2020-01-11', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'
            ],
            'col_5': [4.0, 2.0, 3.0, 4.0, 5.0],
        })

        metric = TableFormat()

        # Run
        result = metric.compute_breakdown(real_data, synthetic_data)

        # Assert
        expected_result = {
            'score': 0.6,
            'invalid column data types': ['col_1', 'col_4']
        }
        assert result == expected_result

    def test_compute_breakdown_ignore_dtype_columns(self, real_data):
        """Test the ``compute_breakdown`` method when ignore_dtype_columns is set."""
        # Setup
        synthetic_data = pd.DataFrame({
            'col_1': [3.0, 2.0, 1.0, 4.0, 5.0],
            'col_2': ['A', 'B', 'C', 'D', 'E'],
            'col_3': [True, False, True, False, True],
            'col_4': [
                '2020-01-11', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'
            ],
            'col_5': [4.0, 2.0, 3.0, 4.0, 5.0],
        })

        metric = TableFormat()

        # Run
        result = metric.compute_breakdown(
            real_data, synthetic_data, ignore_dtype_columns=['col_4']
        )

        # Assert
        expected_result = {
            'score': 0.8,
            'invalid column data types': ['col_1']
        }
        assert result == expected_result

    def test_compute_breakdown_multiple_error(self, real_data):
        """Test the ``compute_breakdown`` method with the different failure modes."""
        synthetic_data = pd.DataFrame({
            'col_1': [1, 2, 1, 4, 5],
            'col_3': [True, False, True, False, True],
            'col_4': [
                '2020-01-11', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'
            ],
            'col_5': [4.0, 2.0, 3.0, 4.0, 5.0],
            'col_6': [4.0, 2.0, 3.0, 4.0, 5.0],
        })

        metric = TableFormat()

        # Run
        result = metric.compute_breakdown(real_data, synthetic_data)

        # Assert
        expected_result = {
            'score': 0.5120000000000001,
            'missing columns in synthetic data': ['col_2'],
            'invalid column names': ['col_6'],
            'invalid column data types': ['col_4']
        }
        assert result == expected_result

    @patch('sdmetrics.single_table.table_format.TableFormat.compute_breakdown')
    def test_compute(self, compute_breakdown_mock, real_data):
        """Test the ``compute`` method."""
        # Setup
        synthetic_data = pd.DataFrame({
            'col_1': [3, 2, 1, 4, 5],
            'col_2': ['A', 'B', 'C', 'D', 'E'],
            'col_3': [True, False, True, False, True],
            'col_4': pd.to_datetime([
                '2020-01-11', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'
            ]),
            'col_5': [4.0, 2.0, 3.0, 4.0, 5.0]
        })
        metric = TableFormat()
        compute_breakdown_mock.return_value = {'score': 0.6}

        # Run
        result = metric.compute(real_data, synthetic_data)

        # Assert
        compute_breakdown_mock.assert_called_once_with(real_data, synthetic_data, None)
        assert result == 0.6

    @patch('sdmetrics.single_table.table_format.SingleTableMetric.normalize')
    def test_normalize(self, normalize_mock):
        """Test the ``normalize`` method."""
        # Setup
        metric = TableFormat()
        raw_score = 0.9

        # Run
        result = metric.normalize(raw_score)

        # Assert
        normalize_mock.assert_called_once_with(raw_score)
        assert result == normalize_mock.return_value
