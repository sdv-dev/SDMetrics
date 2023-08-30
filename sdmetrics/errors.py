"""Custom errors for SDMetrics."""


class IncomputableMetricError(Exception):
    """Raised when a metric cannot be computed."""


class ConstantInputError(Exception):
    """Thrown when the input data has all the same values."""


class InvalidDataError(Exception):
    """Error to raise when data is not valid."""
