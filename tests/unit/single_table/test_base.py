"""Tests for the base single table metric class."""

from datetime import date, datetime

import pandas as pd
import pytest

from sdmetrics.single_table import SingleTableMetric


class TestSingleTableMetric:

    def test__validate_inputs(self):
        """Test the ``_validate_inputs`` method.

        Expect that the input is returned as-is when valid.

        Input:
        - real_data
        - synthetic data
        - metadata

        Output:
        - validated real data
        - validated synthetic data
        - validated metadata
        """
        # Setup
        real_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        synthetic_data = pd.DataFrame({'col1': [2, 2, 3], 'col2': ['c', 'b', 'c']})
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'categorical'},
            },
        }

        # Run
        validated_real, validated_synthetic, validated_meta = SingleTableMetric._validate_inputs(
            real_data, synthetic_data, metadata)

        # Assert
        pd.testing.assert_frame_equal(real_data, validated_real)
        pd.testing.assert_frame_equal(synthetic_data, validated_synthetic)
        assert validated_meta == metadata

    def test__validate_inputs_no_metadata(self):
        """Test the ``_validate_inputs`` method with no metadata.

        Expect that if no metadata is provided, metadata is generated for the provided data.

        Input:
        - real_data
        - synthetic data
        - metadata

        Output:
        - validated real data
        - validated synthetic data
        - validated metadata
        """
        # Setup
        real_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        synthetic_data = pd.DataFrame({'col1': [2, 2, 3], 'col2': ['c', 'b', 'c']})

        # Run
        validated_real, validated_synthetic, metadata = SingleTableMetric._validate_inputs(
            real_data, synthetic_data)

        # Assert
        expected_metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'categorical'},
            },
        }
        pd.testing.assert_frame_equal(real_data, validated_real)
        pd.testing.assert_frame_equal(synthetic_data, validated_synthetic)
        assert metadata == expected_metadata

    def test__validate_inputs_invalid_metadata(self):
        """Test the ``_validate_inputs`` method with invalid_metadata.

        Expect that a ValueError is thrown.

        Input:
        - real_data
        - synthetic data
        - metadata

        Output:
        - validated real data
        - validated synthetic data
        - validated metadata
        """
        # Setup
        real_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c'], 'col3': [1, 2, 3]})
        synthetic_data = pd.DataFrame({
            'col1': [2, 2, 3],
            'col2': ['c', 'b', 'c'],
            'col3': [1, 2, 3],
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'categorical'},
            },
        }

        # Run and assert
        expected_error_msg = 'Column col3 not found in metadata'
        with pytest.raises(ValueError, match=expected_error_msg):
            SingleTableMetric._validate_inputs(real_data, synthetic_data, metadata)

    def test__validate_inputs_unexpected_data_column(self):
        """Test the ``_validate_inputs`` method with an unexpected data column.

        Expect that a ValueError is thrown.

        Input:
        - real_data
        - synthetic data
        - metadata

        Output:
        - validated real data
        - validated synthetic data
        - validated metadata
        """
        # Setup
        real_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        synthetic_data = pd.DataFrame({'col1': [2, 2, 3], 'col2': ['c', 'b', 'c']})
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'categorical'},
                'col3': {'sdtype': 'boolean'},
            },
        }

        # Run and assert
        expected_error_msg = 'Field col3 not found in data'
        with pytest.raises(ValueError, match=expected_error_msg):
            SingleTableMetric._validate_inputs(real_data, synthetic_data, metadata)

    def test__validate_inputs_different_columns(self):
        """Test the ``_validate_inputs`` method with mismatched real and synthetic columns.

        Expect that a ValueError is thrown.

        Input:
        - real_data
        - synthetic data
        - metadata

        Output:
        - validated real data
        - validated synthetic data
        - validated metadata
        """
        # Setup
        real_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c'], 'col3': [1, 2, 3]})
        synthetic_data = pd.DataFrame({'col1': [2, 2, 3], 'col2': ['c', 'b', 'c']})

        # Run and assert
        expected_error_msg = '`real_data` and `synthetic_data` must have the same columns'
        with pytest.raises(ValueError, match=expected_error_msg):
            SingleTableMetric._validate_inputs(real_data, synthetic_data)

    def test__validate_inputs_date(self):
        """Test the ``_validate_inputs`` method with a date column.

        Expect that the date column is properly converted to datetime within the real and
        synthetic data.

        Input:
        - real_data
        - synthetic data
        - metadata

        Output:
        - validated real data
        - validated synthetic data
        - validated metadata
        """
        # Setup
        real_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [date(2020, 10, 1), date(2021, 1, 2), date(2022, 3, 1)],
        })
        synthetic_data = pd.DataFrame({
            'col1': [2, 2, 3],
            'col2': ['c', 'b', 'c'],
            'col3': [date(2020, 11, 12), date(2022, 11, 2), date(2022, 3, 13)],
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'categorical'},
                'col3': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            },
        }

        # Run
        validated_real, validated_synthetic, metadata = SingleTableMetric._validate_inputs(
            real_data, synthetic_data, metadata)

        # Assert
        expected_real_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [datetime(2020, 10, 1), datetime(2021, 1, 2), datetime(2022, 3, 1)],
        })
        expected_synthetic_data = pd.DataFrame({
            'col1': [2, 2, 3],
            'col2': ['c', 'b', 'c'],
            'col3': [datetime(2020, 11, 12), datetime(2022, 11, 2), datetime(2022, 3, 13)],
        })
        pd.testing.assert_frame_equal(validated_real, expected_real_data)
        pd.testing.assert_frame_equal(validated_synthetic, expected_synthetic_data)
        assert metadata == metadata

    def test__select_fields(self):
        """Test the ``_select_fields`` method with pii. Expect that pii column is skipped."""
        # Setup
        metadata = {
            'alternate_keys': ['colD', ['colE', 'colF']],
            'columns': {
                'colA': {'sdtype': 'datetime'},
                'colB': {'sdtype': 'categorical'},
                'colC': {'sdtype': 'categorical', 'pii': True},
                'colD': {'sdtype': 'categorical'},
                'colE': {'sdtype': 'categorical'},
                'colF': {'sdtype': 'categorical'},
            },
        }

        # Run
        out = SingleTableMetric._select_fields(metadata, 'categorical')

        # Assert
        assert out == ['colB']
