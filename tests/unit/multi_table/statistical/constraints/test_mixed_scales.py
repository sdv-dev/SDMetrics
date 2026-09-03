import re
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from sdmetrics.multi_table.statistical.constraints import MixedScales
from sdmetrics.multi_table.statistical.constraints._utils import CustomNan
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


@pytest.fixture
def data():
    return {
        'table': pd.DataFrame({
            'col_A': [1.00, 2.00, 10.0, np.nan, 1000.00],
            'col_B': ['a', 'a', 'a', 'a', 'a'],
            'col_C': ['a', 'a', 'b', 'b', 'c'],
        }),
    }


@pytest.fixture
def metadata():
    return {
        'tables': {
            'table': {
                'columns': {
                    'col_A': {'sdtype': 'numerical'},
                    'col_B': {'sdtype': 'categorical'},
                    'col_C': {'sdtype': 'categorical'},
                }
            }
        }
    }


@pytest.fixture
def constraint():
    return MixedScales(
        mixed_scale_column_name='col_A',
        segment_column_names=['col_B', 'col_C'],
        table_name='table',
    )


class TestMixedScales:
    def test___init__(self):
        """Test the ``__init__`` method sets the parameters."""
        # Setup
        segment_column_names = ['col_B', 'col_C']

        # Run
        instance = MixedScales(
            mixed_scale_column_name='col_A',
            segment_column_names=segment_column_names,
            table_name='table',
        )

        # Assert
        assert instance.mixed_scale_column_name == 'col_A'
        assert instance.segment_column_names == ['col_B', 'col_C']
        assert instance.segment_column_names is not segment_column_names
        assert instance.table_name == 'table'
        assert instance._segment_info == {}
        assert instance._segment_tuple == ()

    def test___init__without_table_name(self):
        """Test the ``__init__`` method accepts a missing table name."""
        # Run
        instance = MixedScales(
            mixed_scale_column_name='col_A', segment_column_names=['col_B', 'col_C']
        )

        # Assert
        assert instance.table_name is None

    def test___init__invalid_parameters(self):
        """Test the ``__init__`` method errors with invalid arguments."""
        # Run and Assert
        err_msg = "The 'table_name' parameter must be a string."
        with pytest.raises(ValueError, match=err_msg):
            MixedScales(
                mixed_scale_column_name='col_A',
                segment_column_names=['col_B', 'col_C'],
                table_name=1,
            )

        err_msg = re.escape('`mixed_scale_column_name` must be a string.')
        with pytest.raises(ValueError, match=err_msg):
            MixedScales(mixed_scale_column_name=1, segment_column_names=['col_B'])

        err_msg = re.escape('`segment_column_names` must be a list of strings.')
        with pytest.raises(ValueError, match=err_msg):
            MixedScales(mixed_scale_column_name='col_A', segment_column_names='col_B')

        with pytest.raises(ValueError, match=err_msg):
            MixedScales(mixed_scale_column_name='col_A', segment_column_names=['col_B', 1])

    def test__validate_data(self, data, metadata, constraint):
        """Test ``_validate_data`` passes with the expected sdtypes."""
        # Run and Assert
        constraint._validate_data(data, metadata)

    def test__validate_data_boolean_segment_column(self, data, metadata, constraint):
        """Test ``_validate_data`` also accepts a boolean segment column."""
        # Setup
        metadata['tables']['table']['columns']['col_B']['sdtype'] = 'boolean'

        # Run and Assert
        constraint._validate_data(data, metadata)

    def test__validate_data_non_numerical_column(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if the mixed scale column is not numerical."""
        # Setup
        metadata['tables']['table']['columns']['col_A']['sdtype'] = 'categorical'
        expected_error = re.escape(
            'A MixedScales constraint is being applied to columns with mismatched sdtypes '
            'col_A. The mixed_scale_column must be numerical.'
        )

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__validate_data_non_categorical_segment_column(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if a segment column is not categorical."""
        # Setup
        metadata['tables']['table']['columns']['col_B']['sdtype'] = 'numerical'
        expected_error = re.escape(
            'A MixedScales constraint is being applied to segment columns with mismatched '
            'sdtypes col_B. All segment columns must be categorical.'
        )

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    @patch('sdmetrics.multi_table.statistical.constraints.mixed_scales.NumericalFormatter')
    def test__fit_with_mock(self, mock_formatter, data, metadata):
        """Test the ``_fit`` method.

        Here we test :
            - The expected ``_segment_info`` keys are used.
            - A ``NumericalFormatter`` is created for each entry in ``_segment_info``.
            - 'min', 'max', and 'null_proportion' values are added to ``_segment_info``.
        """
        # Setup
        constraint = MixedScales('col_A', ['col_B', 'col_C'])
        constraint.metadata = metadata
        min_value = 'very low'
        max_value = 'very high'
        expected_vals = {
            ('a', 'a'): {'max': max_value, 'min': min_value, 'null_proportion': 0.0},
            ('a', 'b'): {'max': max_value, 'min': min_value, 'null_proportion': 0.5},
            ('a', 'c'): {'max': max_value, 'min': min_value, 'null_proportion': 0.0},
        }
        mock_formatter()._min_value = min_value
        mock_formatter()._max_value = max_value

        # Run
        constraint._fit(data, metadata)

        # Assert
        for key, values in expected_vals.items():
            assert isinstance(constraint._segment_info[key]['formatter'], Mock)
            for name, value in values.items():
                assert constraint._segment_info[key][name] == value

        assert constraint._segment_tuple == (('a', 'a'), ('a', 'b'), ('a', 'c'))

    def test__fit(self, data, metadata, constraint):
        """Test ``_fit`` learns the bounds of every segment."""
        # Run
        constraint._fit(data, metadata)

        # Assert
        assert set(constraint._segment_info) == {('a', 'a'), ('a', 'b'), ('a', 'c')}
        assert constraint._segment_info[('a', 'a')]['min'] == 1.0
        assert constraint._segment_info[('a', 'a')]['max'] == 2.0
        assert constraint._segment_info[('a', 'b')]['min'] == 10.0
        assert constraint._segment_info[('a', 'b')]['max'] == 10.0
        assert constraint._segment_info[('a', 'c')]['min'] == 1000.0
        assert constraint._segment_info[('a', 'c')]['max'] == 1000.0
        assert constraint._segment_tuple == (('a', 'a'), ('a', 'b'), ('a', 'c'))

    def test__fit_learns_the_null_proportion(self, data, metadata, constraint):
        """Test ``_fit`` learns the proportion of missing values of every segment."""
        # Run
        constraint._fit(data, metadata)

        # Assert
        assert constraint._segment_info[('a', 'a')]['null_proportion'] == 0.0
        assert constraint._segment_info[('a', 'b')]['null_proportion'] == 0.5
        assert constraint._segment_info[('a', 'c')]['null_proportion'] == 0.0

    def test__fit_single_segment_column(self, data, metadata):
        """Test ``_fit`` keys the segments by value when there is a single column."""
        # Setup
        instance = MixedScales(
            mixed_scale_column_name='col_A',
            segment_column_names=['col_C'],
            table_name='table',
        )

        # Run
        instance._fit(data, metadata)

        # Assert
        assert set(instance._segment_info) == {'a', 'b', 'c'}
        assert instance._segment_info['a']['min'] == 1.0
        assert instance._segment_info['a']['max'] == 2.0
        assert instance._segment_info['c']['min'] == 1000.0
        # ``_segment_tuple`` keeps the value that ``groupby`` yields, which is a scalar
        # on pandas 1.x and a length 1 tuple on pandas 2.x, so only its size is checked.
        assert len(instance._segment_tuple) == 3

    def test__fit_with_null_segment(self, data, metadata, constraint):
        """Test ``_fit`` keeps a segment of missing values under a ``CustomNan`` key."""
        # Setup
        data['table']['col_C'] = ['a', 'a', 'b', 'b', None]

        # Run
        constraint._fit(data, metadata)

        # Assert
        assert set(constraint._segment_info) == {('a', 'a'), ('a', 'b'), ('a', CustomNan())}
        assert constraint._segment_info[('a', CustomNan())]['min'] == 1000.0
        assert constraint._segment_info[('a', CustomNan())]['max'] == 1000.0

    def test__is_valid(self, data, metadata, constraint):
        """Test the ``is_valid`` method"""
        # Setup
        constraint.metadata = metadata
        constraint._fitted = True
        constraint._segment_info = {
            ('a', 'a'): {'max': 2.0, 'min': 1.0, 'null_proportion': 0.0},
            ('a', 'b'): {'max': 10.0, 'min': 10.0, 'null_proportion': 0.5},
            ('a', 'c'): {'max': 1000.0, 'min': 1000.0, 'null_proportion': 0.0},
        }

        # Run
        result = constraint._is_valid(data)

        # Assert
        assert set(result.keys()) == {'table'}
        expected_table_result = pd.Series([True] * 5, name=constraint.mixed_scale_column_name)
        pd.testing.assert_series_equal(result['table'], expected_table_result)

    def test__is_valid_with_invalid_values(self, data, metadata, constraint):
        """Test ``_is_valid`` flags the values that are out of their segment bounds."""
        # Setup
        synthetic_data = {
            'table': pd.DataFrame({
                'col_A': [1.5, 5.0, 10.0, 9.0, 1000.0, 1.0, np.nan],
                'col_B': ['a', 'a', 'a', 'a', 'a', 'a', 'a'],
                'col_C': ['a', 'a', 'b', 'b', 'c', 'd', 'a'],
            })
        }
        constraint.fit(data, metadata)

        # Run
        is_valid = constraint._is_valid(synthetic_data, metadata)

        # Assert
        expected = pd.Series([True, False, True, False, True, False, True], name='col_A')
        pd.testing.assert_series_equal(is_valid['table'], expected)

    def test__is_valid_unknown_segment(self, data, metadata, constraint):
        """Test ``_is_valid`` flags every row of a segment that is not in the real data."""
        # Setup
        synthetic_data = {
            'table': pd.DataFrame({
                'col_A': [1.0, 2.0],
                'col_B': ['z', 'z'],
                'col_C': ['z', 'z'],
            })
        }
        constraint.fit(data, metadata)

        # Run
        is_valid = constraint._is_valid(synthetic_data, metadata)

        # Assert
        expected = pd.Series([False, False], name='col_A')
        pd.testing.assert_series_equal(is_valid['table'], expected)

    def test__is_valid_with_null_segment(self, data, metadata, constraint):
        """Test ``_is_valid`` checks the bounds of a segment of missing values."""
        # Setup
        data['table']['col_C'] = ['a', 'a', 'b', 'b', None]
        synthetic_data = {
            'table': pd.DataFrame({
                'col_A': [1.5, 5.0, 1000.0, 1.0],
                'col_B': ['a', 'a', 'a', 'a'],
                'col_C': ['a', 'a', None, None],
            })
        }
        constraint.fit(data, metadata)

        # Run
        is_valid = constraint._is_valid(synthetic_data, metadata)

        # Assert
        expected = pd.Series([True, False, True, False], name='col_A')
        pd.testing.assert_series_equal(is_valid['table'], expected)

    def test__is_valid_unfit(self, data, metadata):
        """Test the ``is_valid`` method before constraint has been fitted."""
        # Setup
        constraint = MixedScales('col_A', ['col_B', 'col_C'])

        # Run
        result = constraint._is_valid(data, metadata)

        # Assert
        assert set(result.keys()) == {'table'}
        expected_table_result = pd.Series([True] * 5)
        pd.testing.assert_series_equal(result['table'], expected_table_result)

    def test_get_score(self, data, metadata, constraint):
        """Test ``get_score`` returns the proportion of valid rows."""
        # Setup
        constraint.fit(data, metadata)

        # Run and Assert
        assert constraint.get_score(data, metadata) == 1.0

    def test_get_score_invalid_data(self, data, metadata, constraint):
        """Test ``get_score`` scores the synthetic data against the learned bounds."""
        # Setup
        synthetic_data = {
            'table': pd.DataFrame({
                'col_A': [1.5, 5.0, 10.0, 9.0, 1000.0, 1.0, np.nan],
                'col_B': ['a', 'a', 'a', 'a', 'a', 'a', 'a'],
                'col_C': ['a', 'a', 'b', 'b', 'c', 'd', 'a'],
            })
        }
        constraint.fit(data, metadata)

        # Run and Assert
        assert constraint.get_score(synthetic_data, metadata) == pytest.approx(4 / 7)

    def test_get_score_empty_table(self, data, metadata, constraint):
        """Test ``get_score`` returns NaN when there are no rows to check."""
        # Setup
        data['table'] = data['table'].iloc[:0]
        constraint.fit(data, metadata)

        # Run and Assert
        assert pd.isna(constraint.get_score(data, metadata))
