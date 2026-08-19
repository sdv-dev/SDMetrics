"""Utils for testing."""

import pandas as pd


class DataFrameMatcher:
    """Match a given Pandas DataFrame in a mock function call."""

    def __init__(self, df):
        """Initialize the DataFrame."""
        self.df = df

    def __eq__(self, other):
        """Assert equality using pandas testing module."""
        pd.testing.assert_frame_equal(self.df, other)
        return True


class SeriesMatcher:
    """Match a given Pandas Series in a mock function call."""

    def __init__(self, data):
        """Initialize the Series."""
        self.data = data

    def __eq__(self, other):
        """Assert equality using pandas testing module."""
        pd.testing.assert_series_equal(self.data, other)
        return True


class IteratorMatcher:
    """Match a given iterator in a mock function call."""

    def __init__(self, iterator):
        """Initialize the iterator."""
        self.iterator = iterator

    def __eq__(self, other):
        """Assert equality by expanding the iterator."""
        assert all(x == y for x, y in zip(self.iterator, other))
        return True


def get_error_type(error):
    if error is not None:
        colon_index = error.find(':')
        return error[:colon_index]
    return None


def check_if_value_in_threshold(value, expected_value, threshold):
    assert abs(value - expected_value) < threshold


def assert_report_scores_are_not_nan(report):
    """Assert that every report property and detail has a score."""
    properties = report.get_properties()
    missing_property_scores = properties.loc[properties['Score'].isna()]
    assert missing_property_scores.empty

    for property_name in properties['Property']:
        details = report.get_details(property_name)
        missing_detail_scores = details.loc[details['Score'].isna()]
        assert missing_detail_scores.empty


def _cast_datetime_and_id_to_string(data, metadata):
    """Cast datetime and id columns to string representation."""
    data = data.copy()
    for table_name in metadata['tables']:
        for column, column_meta in metadata['tables'][table_name]['columns'].items():
            sdtype = column_meta.get('sdtype')
            datetime_format = column_meta.get('datetime_format')
            if sdtype == 'datetime' and datetime_format is not None:
                data[table_name][column] = data[table_name][column].dt.strftime(datetime_format)
            if sdtype == 'id' and column_meta.get('regex_format') is not None:
                data[table_name][column] = data[table_name][column].astype(str)

    return data
