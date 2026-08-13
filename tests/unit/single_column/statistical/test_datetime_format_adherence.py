from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_column.statistical import DatetimeFormatAdherence


class TestDatetimeFormatAdherence:
    @pytest.mark.parametrize(
        'datetime_format',
        [('%Y/%m/%d'), ('%m-%d-%Y'), ('%Y-%m-%d %H-%M-%S'), ('%Y-%m-%d %H:%M:%S%z')],
    )
    def test__validate_datetime_format(self, datetime_format):
        """Test the ``_validate_datetime_format`` method.

        Expect that the datetime_format is valid.

        Input:
        - Datetime format.
        """
        # Setup
        metric = DatetimeFormatAdherence()

        # Run & Assert
        metric._validate_datetime_format(datetime_format)

    def test__validate_datetime_format_must_be_string(self):
        """Test the ``_validate_datetime_format`` method.

        Expect that the datetime_format raises an error when format is not a string.
        """
        # Setup
        datetime_format = 1.0
        message = '`datetime_format` must be a string.'

        metric = DatetimeFormatAdherence()

        # Run & Assert
        with pytest.raises(ValueError, match=message):
            metric._validate_datetime_format(datetime_format)

    @pytest.mark.parametrize(
        'bad_format',
        [('%Y/%m/%-%'), ('%%m-%d-%Y'), ('Date: %')],
    )
    def test__validate_datetime_format_invalid(self, bad_format):
        """Test the ``_validate_datetime_format`` method.

        Expect that the datetime_format raises an error when format is invalid.
        """
        # Setup
        message = f"Invalid datetime format string '{bad_format}'"

        metric = DatetimeFormatAdherence()

        # Run & Assert
        with pytest.raises(ValueError, match=message):
            metric._validate_datetime_format(bad_format)

    def test__filter_valid_datetime_rows(self):
        """Test the `_filter_valid_datetime_rows` method.

        Expect that valid rows (that match the format) are returned.

        Input:
        - Column data.
        - Datetime format.

        Output:
        - pd.Series of valid rows.
        """
        # Setup
        data = pd.Series(['9-10-2020', '29-7-2020', '15-12-2020'])
        datetime_format = '%d-%m-%Y'
        expected = pd.Series(['9-10-2020', '29-7-2020', '15-12-2020'])

        metric = DatetimeFormatAdherence()

        # Run
        result = metric._filter_valid_datetime_rows(data, datetime_format)

        # Assert
        assert result.equals(expected)

    def test__filter_valid_datetime_rows_nan(self):
        """Test the `_filter_valid_datetime_rows` method.

        Expect nulls are counted as True.
        """
        # Setup
        data = pd.Series(['9-10-2020', None, '15-12-2020'])
        datetime_format = '%d-%m-%Y'
        expected = pd.Series(['9-10-2020', None, '15-12-2020'])

        metric = DatetimeFormatAdherence()

        # Run
        result = metric._filter_valid_datetime_rows(data, datetime_format)

        # Assert
        assert result.equals(expected)

    def test_compute(self):
        """Test the ``compute`` method.

        Expect that the percentage of rows that match the datetime format is returned.

        Input:
        - Real data.
        - Synthetic data.
        - Datetime format.

        Output:
        - The evaluated metric.
        """
        # Setup
        real_data = pd.Series(['10-10-2020', '10-11-2021', '10-12-2022'])
        synthetic_data = pd.Series(['9-10-2020', '29-7-2020', '15-12-2020'])
        datetime_format = '%d-%m-%Y'

        metric = DatetimeFormatAdherence()

        # Run
        result = metric.compute(real_data, synthetic_data, datetime_format)

        # Assert
        assert result == 1.0

    def test_compute_warning(self):
        """Test the ``compute`` method gives warning.

        If real data doesn't match the format, a warning is given.
        """
        # Setup
        real_data = pd.Series(['10/10/2020', '10/11/2021', '10/12/2022'])
        synthetic_data = pd.Series(['9-10-2020', '29-7-2020', '15-12-2020'])
        datetime_format = '%d-%m-%Y'
        message = 'The real data does not match the given datetime format.'

        metric = DatetimeFormatAdherence()

        # Run
        with pytest.warns(UserWarning) as record:
            result = metric.compute(real_data, synthetic_data, datetime_format)

        # Assert
        assert result == 1.0
        assert len(record) == 1
        assert str(record[0].message) == message

    def test_compute_nans(self):
        """Test the ``compute`` method with nan values.

        Nan values are not counted as a mismatch.
        """
        # Setup
        real_data = pd.Series(['10-10-2020', '10-11-2021', '10-12-2022'])
        synthetic_data = pd.Series(['9-10-2020', np.nan, '15-12-2020'])
        datetime_format = '%d-%m-%Y'

        metric = DatetimeFormatAdherence()

        # Run
        result = metric.compute(real_data, synthetic_data, datetime_format)

        # Assert
        assert result == 1.0

    def test_compute_nans_with_incomplete_score(self):
        """Test the ``compute`` method with nan and incomplete score."""
        # Setup
        real_data = pd.Series(['10-10-2020', '10-11-2021', '10-12-2022'])
        synthetic_data = pd.Series([
            '2026-01-01',
            '2026-01-02',
            np.nan,
            '2026-01-02 12:20:59',
        ])
        datetime_format = '%Y-%m-%d'

        metric = DatetimeFormatAdherence()

        # Run
        result = metric.compute(real_data, synthetic_data, datetime_format)

        # Assert
        assert result == 0.75

    def test_compute_datetime_error(self):
        """Test the ``compute`` method with datetime64 type.

        Expect that an error is raised if the synthetic data is present as datetime64,
        then the format cannot be checked and the score is nan.
        """
        # Setup
        real_data = pd.Series(
            [
                datetime(2020, 10, 1),
                datetime(2021, 1, 2),
                datetime(2021, 9, 12),
                datetime(2022, 10, 1),
            ],
            dtype='datetime64[ns]',
        )
        synthetic_data = pd.Series(
            [
                datetime(2020, 11, 1),
                datetime(2021, 1, 2),
                datetime(2021, 2, 9),
                pd.NaT,
            ],
            dtype='datetime64[ns]',
        )
        datetime_format = '%d-%m-%Y'

        metric = DatetimeFormatAdherence()

        # Run
        result = metric.compute(real_data, synthetic_data, datetime_format)

        # Assert
        assert pd.isna(result)
