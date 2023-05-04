from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table import MultiSingleColumnMetric


class TestMultiSingleColumnMetric:

    def test__compute(self):
        """Test the ``_compute`` method.

        Expect that a mapping of column to metric value is returned.

        Setup:
        - Mock ``_validate_inputs`` to return desired metadata.
        - Mock ``_select fields`` to return the desired fields to compute metrics for.

        Input:
        - real data
        - synthetic data

        Output:
        - A mapping of column name to computed metric value, with null values for the columns
          that are not valid for this metric.
        """
        # Setup
        metadata = {'columns': {'a': {}, 'b': {}, 'c': {}, 'd': {}}}

        data = pd.DataFrame({
            'a': [0, 1, 2, 3],
            'b': [100, 200, 300, 400],
            'c': ['one', 'two', 'three', 'four'],
            'd': ['a', 'b', 'c', 'd'],
        })

        metric_mock = Mock()
        metric_mock._validate_inputs.return_value = (data, data, metadata)
        metric_mock._select_fields.return_value = ['a', 'b']
        metric_mock.single_column_metric.compute_breakdown.side_effect = [
            {'score': 1.0}, {'score': 2.0},
        ]
        metric_mock.single_column_metric_kwargs = None

        # Run
        result = MultiSingleColumnMetric._compute(metric_mock, data, data)

        # Assert
        assert result == {
            'a': {'score': 1.0},
            'b': {'score': 2.0},
            'c': {'score': np.nan},
            'd': {'score': np.nan},
        }

    def test__compute_store_errors_false(self):
        """Test the ``_compute`` method with an error and ``store_errors`` set to ``False``.

        Expect that if ``store_errors`` is ``False`` and one column errors out when computing
        the single column metric, the ``__compute`` method will also throw that error.

        Setup:
        - Mock ``_validate_inputs`` to return desired metadata.
        - Mock ``_select fields`` to return the desired fields to compute metrics for.
        - Mock ``single_column_metric.compute`` to return an error for one column.

        Input:
        - real data
        - synthetic data
        - store_errors=False

        Side Effects:
        - An error is thrown.
        """
        # Setup
        data = pd.DataFrame({
            'a': [0, 1, 2, 3],
            'b': [100, 200, 300, 400],
            'c': ['one', 'two', 'three', 'four'],
            'd': ['a', 'b', 'c', 'd'],
        })

        metadata = {'columns': {'a': {}, 'b': {}, 'c': {}, 'd': {}}}
        test_error = ValueError('test error')

        metric_mock = Mock()
        metric_mock._validate_inputs.return_value = (data, data, metadata)
        metric_mock._select_fields.return_value = ['a', 'b', 'c']
        metric_mock.single_column_metric.compute_breakdown.side_effect = [1.0, 2.0, test_error]
        metric_mock.single_column_metric_kwargs = None

        # Run and assert
        with pytest.raises(ValueError, match='test error'):
            MultiSingleColumnMetric._compute(metric_mock, data, data, store_errors=False)

    def test__compute_store_errors_true(self):
        """Test the ``_compute`` method with an error and ``store_errors`` set to ``True``.

        Expect that a mapping of column to metric value is returned. If the metric errors
        out for one of the columns and ``store_errors`` is set to ``True``, expect that
        the error is stored in the result.

        Setup:
        - Mock ``_validate_inputs`` to return desired metadata.
        - Mock ``_select fields`` to return the desired fields to compute metrics for.
        - Mock ``single_column_metric.compute`` to return an error for one column.

        Input:
        - real data
        - synthetic data
        - store_errors=True

        Output:
        - A mapping of column name to computed metric value, with null values for the columns
          that are not valid for this metric, and errors for the columns that errored out.
        """
        # Setup
        metadata = {'columns': {'a': {}, 'b': {}, 'c': {}, 'd': {}}}
        test_error = ValueError('test error')

        data = pd.DataFrame({
            'a': [0, 1, 2, 3],
            'b': [100, 200, 300, 400],
            'c': ['one', 'two', 'three', 'four'],
            'd': ['a', 'b', 'c', 'd'],
        })

        metric_mock = Mock()
        metric_mock._validate_inputs.return_value = (data, data, metadata)
        metric_mock._select_fields.return_value = ['a', 'b', 'c']
        metric_mock.single_column_metric.compute_breakdown.side_effect = [
            {'score': 1.0}, {'score': 2.0}, {'error': test_error},
        ]
        metric_mock.single_column_metric_kwargs = None

        # Run
        result = MultiSingleColumnMetric._compute(metric_mock, data, data, store_errors=True)

        # Assert
        assert result == {
            'a': {'score': 1.0},
            'b': {'score': 2.0},
            'c': {'error': test_error},
            'd': {'score': np.nan},
        }

    def test_compute(self):
        """Test the ``compute`` method.

        Expect that an average of the column metric scores is returned.

        Setup:
        - ``_compute`` helper method should return a mapping of column name to metric scores.

        Input:
        - real data
        - synthetic data

        Output:
        - The average metric score.
        """
        # Setup
        metric_breakdown = {
            'a': {'score': 1.0},
            'b': {'score': 2.0},
            'c': {'score': 1.0},
            'd': {'score': np.nan},
        }

        # Run
        with patch.object(MultiSingleColumnMetric, '_compute', return_value=metric_breakdown):
            result = MultiSingleColumnMetric.compute(real_data=Mock(), synthetic_data=Mock())

        # Assert
        assert result == 4.0 / 3.0

    def test_compute_breakdown(self):
        """Test the ``compute_breakdown`` method.

        Expect that a breakdown of the column metric scores by column is returned.

        Setup:
        - ``_compute`` helper method should return a mapping of column name to metric scores.

        Input:
        - real data
        - synthetic data

        Output:
        - A mapping of the metric scores broken down by column.
        """
        # Setup
        metric_breakdown = {
            'a': {'score': 1.0},
            'b': {'score': 2.0},
            'c': {'score': 1.0},
            'd': {'score': np.nan},
        }

        # Run
        with patch.object(MultiSingleColumnMetric, '_compute', return_value=metric_breakdown):
            result = MultiSingleColumnMetric.compute_breakdown(Mock(), Mock())

        # Assert
        assert result == metric_breakdown
