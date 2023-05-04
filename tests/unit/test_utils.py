from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from sdmetrics.utils import (
    HyperTransformer, get_alternate_keys, get_cardinality_distribution, get_columns_from_metadata,
    get_missing_percentage, get_type_from_column_meta)


def test_get_cardinality_distribution():
    """Test the ``get_cardinality_distribution`` utility function.

    Input:
    - parent column
    - child column

    Output:
    - the expected cardinality distribution.
    """
    # Setup
    parent_column = pd.Series([1, 2, 3, 4, 5])
    child_column = pd.Series([3, 4, 1, 4, 4, 5, 1])

    # Run
    cardinality_distribution = get_cardinality_distribution(parent_column, child_column)

    # Assert
    assert cardinality_distribution.to_list() == [2.0, 0.0, 1.0, 3.0, 1.0]


def test_get_missing_percentage():
    """Test the ``get_missing_percentage`` utility function.

    Input:
    - test column

    Output:
    - the expected percentage of NaN inside the column.
    """
    # Setup
    column = pd.Series([1, 2, 3, np.nan, 5, 6, np.nan])

    # Run
    percentage_nan = get_missing_percentage(column)

    # Assert
    assert percentage_nan == 28.57


def test_get_columns_from_metadata():
    """Test the ``get_columns_from_metadata`` method with current metadata format.

    Expect that the columns are returned.
    """
    # Setup
    metadata = {'columns': {'col1': {'sdtype': 'numerical'}}}

    # Run
    out = get_columns_from_metadata(metadata)

    # Assert
    assert out == {'col1': {'sdtype': 'numerical'}}


def test_get_columns_from_metadata_new_metadata():
    """Test the ``get_columns_from_metadata`` method with the new metadata format.

    Expect that the column type is returned.
    """
    # Setup
    metadata = {'columns': {'col1': {'sdtype': 'numerical'}}}

    # Run
    out = get_columns_from_metadata(metadata)

    # Assert
    assert out == {'col1': {'sdtype': 'numerical'}}


def test_get_type_from_column_meta():
    """Test the ``get_columns_from_column_meta`` method with the current metadata format.

    Expect that the column type is returned.
    """
    # Setup
    column_meta = {'sdtype': 'numerical'}

    # Run
    out = get_type_from_column_meta(column_meta)

    # Assert
    assert out == 'numerical'


def test_get_type_from_column_meta_new_metadata():
    """Test the ``get_columns_from_column_meta`` method with the new metadata format.

    Expect that the column type is returned.
    """
    # Setup
    field_meta = {'sdtype': 'numerical'}

    # Run
    out = get_type_from_column_meta(field_meta)

    # Assert
    assert out == 'numerical'


def test_get_alternate_keys():
    """Test that the list of alternate keys is returned from the metadata."""
    # Setup
    metadata = {'alternate_keys': ['A', ['B', 'C']]}

    # Run
    out = get_alternate_keys(metadata)

    # Assert
    assert out == ['A', 'B', 'C']


class TestHyperTransformer:

    @patch('sdmetrics.utils.OneHotEncoder')
    def test_fit(self, one_hot_encoder_mock):
        """Test the ``fit`` method.

        Validate that the ``column_transforms`` and ``column_kind`` properties are filled
        accordingly.

        Input:
            - test data.
        Side effects:
            - ``column_transforms`` and ``column_kind`` should be filled according to the input.
        """
        # Setup
        data = pd.DataFrame({
            'numerical': [1.0, 2.0, 3.0],
            'categorical': ['a', 'b', 'c'],
            'datetime': [datetime(2020, 1, 1), datetime(2021, 2, 1), datetime(2022, 3, 1)],
            'boolean': [True, False, False],
        })
        ht = HyperTransformer()

        # Run
        ht.fit(data)

        # Assert
        assert ht.column_transforms == {
            'numerical': {'mean': 2.0},
            'categorical': {'one_hot_encoder': one_hot_encoder_mock.return_value},
            'datetime': {'mean': 1.6120224e+18},
            'boolean': {'mode': 0.0},
        }
        assert ht.column_kind == {
            'numerical': 'f',
            'categorical': 'O',
            'datetime': 'M',
            'boolean': 'b',
        }

    @patch('sdmetrics.utils.OneHotEncoder')
    def test_transform(self, one_hot_encoder_mock):
        """Test the ``transform`` method.

        Expect that the data is transformed according to the ``column_transforms``.

        Input:
            - test data.
        Output:
            - transformed data.
        """
        # Setup
        data = pd.DataFrame({
            'numerical': [1.0, 2.0, 3.0],
            'categorical': ['a', 'b', 'c'],
            'datetime': [datetime(2020, 1, 1), datetime(2021, 2, 1), datetime(2022, 3, 1)],
            'boolean': [True, False, False],
        })

        enc = Mock()
        enc.transform.return_value.toarray.return_value = np.array([[1, 0], [0, 1], [0, 0]])

        ht = HyperTransformer()
        ht.column_transforms = {
            'numerical': {'mean': 2.0},
            'categorical': {'one_hot_encoder': enc},
            'datetime': {'mean': 1.6120224e+18},
            'boolean': {'mode': 0.0},
        }
        ht.column_kind = {
            'numerical': 'f',
            'categorical': 'O',
            'datetime': 'M',
            'boolean': 'b',
        }

        # Run
        transformed = ht.transform(data)

        # Assert
        expected = pd.DataFrame({
            'numerical': [1.0, 2.0, 3.0],
            'datetime': [1.577837e+18, 1.612138e+18, 1.646093e+18],
            'boolean': [1.0, 0.0, 0.0],
            'value0': [1, 0, 0],
            'value1': [0, 1, 0],
        })
        pd.testing.assert_frame_equal(transformed, expected, check_dtype=False)

    def test_fit_transform(self):
        """Test the ``fit_transform`` method.

        Validate that this method calls ``fit`` and ``transform`` once each.

        Input:
            - test data.
        Output:
            - the dataframe resulting from fitting and transforming the passed data.
        Side effects:
            - ``fit`` and ``transform`` should each be called once.
        """
        # Setup
        ht = Mock(spec_set=HyperTransformer)
        data = Mock()

        # Run
        out = HyperTransformer.fit_transform(ht, data)

        # Assert
        ht.fit.assert_called_once_with(data)
        ht.transform.assert_called_once_with(data)
        assert out == ht.transform.return_value
