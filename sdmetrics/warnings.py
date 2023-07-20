"""Warnings for sdmetrics."""


class SDMetricsWarning(RuntimeWarning):
    """Class to represent SDMetrics warnings."""


class ConstantInputWarning(SDMetricsWarning):
    """Thrown when the input data has all the same values."""

    def __init__(self, message):
        self.message = message
