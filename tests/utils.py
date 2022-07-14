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
