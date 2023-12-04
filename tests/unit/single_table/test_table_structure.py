from unittest.mock import patch

import pandas as pd
import pytest

from sdmetrics.single_table import TableStructure


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


class TestTableStructure:

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

        metric = TableStructure()

        # Run
        result = metric.compute_breakdown(real_data, synthetic_data)

        # Assert
        expected_result = {'score': 1.0}
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

        metric = TableStructure()

        # Run
        result = metric.compute_breakdown(real_data, synthetic_data)

        # Assert
        expected_result = {'score': 0.8}
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

        metric = TableStructure()

        # Run
        result = metric.compute_breakdown(real_data, synthetic_data)

        # Assert
        expected_result = {'score': 0.8333333333333334}
        assert result == expected_result

    def test_compute_breakdown_multiple_error(self, real_data):
        """Test the ``compute_breakdown`` method with the different failure modes."""
        synthetic_data = pd.DataFrame({
            'col_1': [1, 2, 1, 4, 5],
            'col_3': [True, False, True, False, True],
            'col_5': [4.0, 2.0, 3.0, 4.0, 5.0],
            'col_6': [4.0, 2.0, 3.0, 4.0, 5.0],
        })

        metric = TableStructure()

        # Run
        result = metric.compute_breakdown(real_data, synthetic_data)

        # Assert
        expected_result = {'score': 0.5}
        assert result == expected_result

    @patch('sdmetrics.single_table.table_structure.TableStructure.compute_breakdown')
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
        metric = TableStructure()
        compute_breakdown_mock.return_value = {'score': 0.6}

        # Run
        result = metric.compute(real_data, synthetic_data)

        # Assert
        compute_breakdown_mock.assert_called_once_with(real_data, synthetic_data)
        assert result == 0.6
