import re

import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_column.statistical import RegexFormatAdherence


class TestRegexFormatAdherence:
    @pytest.mark.parametrize(
        'regex_format',
        [(r'[0-9A-Z]{5}'), (r'([A-Z]{2})_[0-9A-Z]{5}'), (r'(?P<country>[A-Z]{2})_[0-9A-Z]{5}')],
    )
    def test__validate_regex_format(self, regex_format):
        """Test the ``_validate_regex_format`` method.

        Expect that the regex_format is valid.

        Input:
        - Regex format.
        """
        # Setup
        metric = RegexFormatAdherence()

        # Run & Assert
        metric._validate_regex_format(regex_format)

    def test__validate_regex_format_must_be_string(self):
        """Test the ``_validate_regex_format`` method.

        Expect that the regex_format raises an error when format is not a string.
        """
        # Setup
        regex_format = 1.0
        message = '`regex_format` must be a string.'

        metric = RegexFormatAdherence()

        # Run & Assert
        with pytest.raises(ValueError, match=message):
            metric._validate_regex_format(regex_format)

    @pytest.mark.parametrize(
        'bad_format',
        [(r'['), (r'(?P<name>'), (r'*abc'), (r'(?P<name>\w+)(?P<name>[A-z]])'), (r'{2,1}')],
    )
    def test__validate_regex_format_invalid(self, bad_format):
        """Test the ``_validate_regex_format`` method.

        Expect that the regex_format raises an error when format is invalid.
        """
        # Setup
        message = f"Invalid regex format string '{bad_format}'."

        metric = RegexFormatAdherence()

        # Run & Assert
        with pytest.raises(ValueError, match=re.escape(message)):
            metric._validate_regex_format(bad_format)

    def test__validate_regex_column(self):
        """Test the `_validate_regex_column` method.

        Expect that valid rows (that match the format) are returned.

        Input:
        - Column data.
        - Regex format.

        Output:
        - pd.Series of valid rows.
        - pd.DataFrame of groups or Npne.
        """
        # Setup
        data = pd.Series(['first', 'second', 'third'])
        regex_format = r'[a-z]{5, 6}'
        expected = pd.Series([True, True, True])

        metric = RegexFormatAdherence()

        # Run
        result, groups = metric._validate_regex_column(data, regex_format)

        # Assert
        result.equals(expected)
        assert groups is None

    def test__validate_regex_column_nan(self):
        """Test the `_validate_regex_column` method.

        Expect nulls are counted as True.
        """
        # Setup
        data = pd.Series([np.nan, 'second', '1234'])
        regex_format = r'[a-z]{5, 6}'
        expected = pd.Series([True, True, False])

        metric = RegexFormatAdherence()

        # Run
        result, groups = metric._validate_regex_column(data, regex_format)

        # Assert
        result.equals(expected)
        assert groups is None

    def test__validate_regex_column_with_groups(self):
        """Test the `_validate_regex_column` method.

        Expect groups are returned.
        """
        # Setup
        data = pd.Series(['+1(123)456-7891', '+44(123)456-7891', None, '+33(123)456-7891', 'abc'])
        regex_format = (
            r'\+(?P<country_code>\d{1,2})'
            r'\((?P<area_code>\d{3})\)'
            r'(?P<phone_number>\d{3}-\d{4})'
        )
        expected = pd.DataFrame(
            {
                'country_code': ['1', '44', '33'],
                'area_code': ['123'] * 3,
                'phone_number': ['456-7891'] * 3,
            },
            index=[0, 1, 3],
        )

        metric = RegexFormatAdherence()

        # Run
        result, groups = metric._validate_regex_column(data, regex_format)

        # Assert
        assert len(result) == 4
        pd.testing.assert_frame_equal(groups, expected)

    def test__validate_regex_column_with_groups_compare(self):
        """Test the `_validate_regex_column` method.

        Expect groups are returned.
        """
        # Setup
        data = pd.Series(['+1(123)456-7891', None, 'abc'])
        regex_format = r'\+(?P<country_code>\d{1,2})\(\d{3}\)\d{3}-\d{4}'
        compare = pd.DataFrame(['1'])

        metric = RegexFormatAdherence()

        # Run
        result, groups = metric._validate_regex_column(data, regex_format, compare)

        # Assert
        assert len(result) == 2  # including nan

    def test_compute(self, caplog):
        """Test the `compute` method.

        Expect that the percentage of rows that match the regex format is returned.

        Input:
        - Real data.
        - Synthetic data.
        - Regex format.

        Output:
        - The evaluated metric.
        """
        # Setup
        real_data = pd.Series(['first', 'second', 'third', '1234'])
        synthetic_data = pd.Series([np.nan, 'second', 'third', '1234'])
        regex_format = r'[a-z]{5, 6}'
        message = 'The real data does not match the given regex format.'

        metric = RegexFormatAdherence()

        # Run
        result = metric.compute(real_data, synthetic_data, regex_format)

        # Assert
        result == 0.75
        message in caplog.text

    def test__compute_with_groups(self, caplog):
        """Test the `compute` method.

        Expect that groups that are not present in the real data are not
        counted as a match.
        """
        # Setup
        real_data = pd.Series(['+1(123)456-7891', '+44(123)456-7891'])
        synthetic_data = pd.Series([
            '+1(123)456-7891',
            'abc',
            '+44(123)456-7891',
            '+33(123)456-7891',
            None,
            '+44(987)654-3211',
        ])
        regex_format = (
            r'\+(?P<country_code>\d{1,2})'
            r'\((?P<area_code>\d{3})\)'
            r'(?P<phone_number>\d{3}-\d{4})'
        )

        metric = RegexFormatAdherence()

        # Run
        result = metric.compute(real_data, synthetic_data, regex_format)

        # Assert
        assert result == 0.5
        assert caplog.text == ''
